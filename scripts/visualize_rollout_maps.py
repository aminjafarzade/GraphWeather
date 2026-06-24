from __future__ import annotations

import argparse
import json
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.config import YParams, setup_logging
from src.data import ClimateNetCDFDataset, DataConfig
from src.graph_bundle import load_graph_bundle
from src.models import GraphWeatherModel


VARIABLE_METADATA: dict[str, dict[str, Any]] = {
    "z500": {"channel": 6, "unit": "m^2 s^-2", "label": "z500 geopotential"},
    "t850": {"channel": 33, "unit": "K", "label": "t850 temperature"},
    "u500": {"channel": 42, "unit": "m s^-1", "label": "u500 wind"},
    "t2m": {"channel": 60, "unit": "K", "label": "t2m temperature"},
    "msl": {"channel": 64, "unit": "Pa", "label": "msl pressure"},
}

VARIABLE_ALIASES: dict[str, list[str]] = {
    "z500": ["z500", "z_500", "z@500", "geopotential_500", "z500_geopotential"],
    "t850": ["t850", "t_850", "t@850", "temperature_850"],
    "u500": ["u500", "u_500", "u@500", "u_wind_500", "zonal_wind_500"],
    "t2m": ["t2m", "2m_temperature", "temperature_2m", "t_2m"],
    "msl": [
        "msl",
        "mslp",
        "mean_sea_level_pressure",
        "mean_sea_level_pressure_surface",
        "mean_sea_level_pressure_msl",
    ],
}

GRAVITY = 9.80665


def _get(params: Any, name: str, default: Any = None) -> Any:
    return getattr(params, name, default)


def _resolve_path(path: str | os.PathLike[str], base_dir: Path = project_root) -> str:
    raw = Path(path).expanduser()
    if raw.is_absolute() or raw.exists():
        return str(raw)
    candidate = base_dir / raw
    return str(candidate if candidate.exists() else raw)


def _decode_channel_name(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _stats_1d(path: str) -> np.ndarray:
    arr = np.load(path).squeeze()
    if arr.ndim == 2:
        arr = arr[0]
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D normalization stats after squeeze, got {arr.shape}")
    return arr.astype(np.float32)


def select_output_stats_vector(
    stats: np.ndarray,
    out_channels: list[int],
    output_channel_count: int,
    stats_path: str,
    stats_kind: str,
) -> tuple[np.ndarray, list[int], str]:
    """Select the normalization vector aligned with model output channels."""
    stats = np.asarray(stats).squeeze()
    if stats.ndim == 2:
        stats = stats[0]
    if stats.ndim != 1:
        raise ValueError(f"Expected 1-D {stats_kind} stats after squeeze, got {stats.shape} from {stats_path}")

    output_channel_count = int(output_channel_count)
    out_channels = [int(x) for x in out_channels]
    if len(stats) == output_channel_count:
        indices = list(range(output_channel_count))
        source = f"{stats_kind}: output-channel stats from {stats_path}"
    elif len(stats) == 2 * output_channel_count:
        indices = list(range(output_channel_count, 2 * output_channel_count))
        source = (
            f"{stats_kind}: selected current/output half from 2x concatenated stats "
            f"({stats_path})"
        )
        logging.warning(
            "WARNING: %s contains %d entries for %d output channels; using indices %d..%d as current/output stats.",
            stats_path,
            len(stats),
            output_channel_count,
            indices[0],
            indices[-1],
        )
    elif out_channels and max(out_channels) < len(stats):
        indices = out_channels
        source = f"{stats_kind}: original-channel indexed stats from {stats_path}"
    else:
        raise ValueError(
            f"{stats_kind} stats length {len(stats)} from {stats_path} cannot be aligned with "
            f"{output_channel_count} output channels and out_channels={out_channels[:5]}..."
        )

    selected = stats[indices].astype(np.float32)
    if selected.shape[0] != output_channel_count:
        raise AssertionError(
            f"Selected {stats_kind} stats length {selected.shape[0]} does not match output channel count {output_channel_count}"
        )
    return selected, indices, source


def load_output_normalization_stats(params: Any) -> dict[str, Any]:
    out_channels = [int(x) for x in _get(params, "out_channels", [])]
    if not out_channels:
        raise ValueError("out_channels is required to align normalization stats.")
    means_path = str(_get(params, "global_means_path", ""))
    stds_path = str(_get(params, "global_stds_path", ""))
    if not means_path or not stds_path:
        raise ValueError("global_means_path and global_stds_path are required for denormalization.")
    means_all = _stats_1d(means_path)
    stds_all = _stats_1d(stds_path)
    output_count = len(out_channels)
    means, mean_indices, mean_source = select_output_stats_vector(
        means_all,
        out_channels,
        output_count,
        means_path,
        "mean",
    )
    stds, std_indices, std_source = select_output_stats_vector(
        stds_all,
        out_channels,
        output_count,
        stds_path,
        "std",
    )
    if len(mean_indices) != len(std_indices) or len(means) != len(stds):
        raise AssertionError("Mean/std output stats are not aligned.")
    return {
        "means": means,
        "stds": stds,
        "mean_indices": mean_indices,
        "std_indices": std_indices,
        "mean_source": mean_source,
        "std_source": std_source,
        "mean_path": means_path,
        "std_path": stds_path,
    }


def _load_netcdf4():
    try:
        import netCDF4 as nc
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "netCDF4 is required for rollout map visualization. Install requirements.txt in your environment."
        ) from exc
    return nc


def _load_cartopy():
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Cartopy is required for geographic visualization. "
            "Please install cartopy to use visualize_rollout_maps.py."
        ) from exc
    return ccrs, cfeature


def _parse_items(values: list[str] | None) -> list[str]:
    if not values:
        return []
    items: list[str] = []
    for value in values:
        items.extend(chunk.strip() for chunk in value.replace(",", " ").split() if chunk.strip())
    return items


def _norm_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _canonical_for_name(name: str) -> str | None:
    norm = _norm_name(name)
    for canonical, aliases in VARIABLE_ALIASES.items():
        if norm == _norm_name(canonical) or norm in {_norm_name(alias) for alias in aliases}:
            return canonical
    return None


def _metadata_for_canonical(canonical: str | None, actual_name: str) -> dict[str, Any]:
    if canonical and canonical in VARIABLE_METADATA:
        return VARIABLE_METADATA[canonical]
    return {"unit": "", "label": actual_name}


def _resolve_config_args(args: argparse.Namespace) -> tuple[str, str]:
    default_yaml = project_root / "configs" / "gnn_5p625.yaml"
    yaml_path = args.yaml_config
    config_name = args.config_name

    if args.config:
        looks_like_yaml = args.config.endswith((".yaml", ".yml")) or Path(args.config).exists()
        if looks_like_yaml:
            yaml_path = args.config
            if config_name is None and Path(args.config).name == "weather_dual_resolution.yaml":
                config_name = "raw"
        elif config_name is None:
            config_name = args.config

    yaml_path = yaml_path or str(default_yaml)
    config_name = config_name or "raw_5p625"
    return _resolve_path(yaml_path), config_name


def _channel_names_from_config(params: Any) -> list[str] | None:
    for key in ("variable_names", "channel_names", "variables"):
        value = _get(params, key, None)
        if isinstance(value, (list, tuple)) and value:
            return [str(x) for x in value]
    return None


def _names_from_payload(payload: Any) -> list[str] | None:
    if isinstance(payload, list) and payload:
        if all(isinstance(item, str) for item in payload):
            return [str(x) for x in payload]
        if all(isinstance(item, dict) for item in payload):
            names: dict[int, str] = {}
            for item in payload:
                idx = item.get("channel", item.get("index", item.get("original_channel_idx")))
                name = item.get("name", item.get("actual_name", item.get("variable_name")))
                if idx is not None and name is not None:
                    names[int(idx)] = str(name)
            if names:
                return [names.get(idx, f"Var{idx}") for idx in range(max(names) + 1)]
    if not isinstance(payload, dict):
        return None
    for key in ("channel_names", "variable_names", "variables"):
        value = payload.get(key)
        if isinstance(value, list) and value:
            nested = _names_from_payload(value)
            if nested:
                return nested
        if isinstance(value, dict) and value:
            names: dict[int, str] = {}
            for raw_idx, item in value.items():
                if isinstance(item, dict):
                    idx = item.get("channel", item.get("index", raw_idx))
                    name = item.get("actual_name", item.get("name", item.get("variable_name")))
                else:
                    idx = raw_idx
                    name = item
                try:
                    names[int(idx)] = str(name)
                except (TypeError, ValueError):
                    continue
            if names:
                return [names.get(idx, f"Var{idx}") for idx in range(max(names) + 1)]
    return None


def _channel_names_from_json_paths(paths: list[Path]) -> tuple[list[str] | None, str | None]:
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        names = _names_from_payload(payload)
        if names:
            return names, str(path)
    return None, None


def _channel_names_from_checkpoint(checkpoint_path: str | None) -> tuple[list[str] | None, str | None]:
    if not checkpoint_path:
        return None, None
    checkpoint_dir = Path(checkpoint_path).resolve().parent
    paths = [
        checkpoint_dir / name
        for name in ("variable_channel_mapping.json", "channel_names.json", "variable_metadata.json", "metadata.json")
    ]
    names, source = _channel_names_from_json_paths(paths)
    if names:
        return names, source
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        return None, None
    if isinstance(checkpoint, dict):
        for key in ("metadata", "config", "params"):
            names = _names_from_payload(checkpoint.get(key, None))
            if names:
                return names, f"checkpoint {key}: {checkpoint_path}"
    return None, None


def _channel_names_from_stats_metadata(params: Any) -> tuple[list[str] | None, str | None]:
    paths: list[Path] = []
    for stat_path in (_get(params, "global_means_path", None), _get(params, "global_stds_path", None)):
        if not stat_path:
            continue
        parent = Path(str(stat_path)).expanduser().parent
        paths.extend(
            [
                parent / "variable_channel_mapping.json",
                parent / "channel_names.json",
                parent / "variable_metadata.json",
                parent / "metadata.json",
            ]
        )
    return _channel_names_from_json_paths(paths)


def get_channel_names_from_dataset_or_config(
    dataset: ClimateNetCDFDataset,
    params: Any,
    checkpoint_path: str | None = None,
) -> tuple[list[str], str]:
    for key in ("variable_names", "channel_names", "variables"):
        names = _names_from_payload(getattr(dataset, key, None))
        if names:
            return names, f"dataset.{key}"

    nc = _load_netcdf4()
    with nc.Dataset(dataset.files_paths[0], "r") as ds:
        if "channel" in ds.variables:
            names = [_decode_channel_name(x) for x in ds.variables["channel"][:]]
            return names, f"NetCDF channel variable: {dataset.files_paths[0]}"

    config_names = _channel_names_from_config(params)
    if config_names:
        return config_names, "config variable_names/channel_names"

    stats_names, stats_source = _channel_names_from_stats_metadata(params)
    if stats_names:
        return stats_names, f"normalization stats metadata: {stats_source}"

    checkpoint_names, checkpoint_source = _channel_names_from_checkpoint(checkpoint_path)
    if checkpoint_names:
        return checkpoint_names, f"experiment metadata: {checkpoint_source}"

    max_channel = max(max(dataset.config.in_channels), max(dataset.config.out_channels))
    return [f"Var{i}" for i in range(max_channel + 1)], "synthetic Var{i}; no variable metadata found"


def _split_data_path(params: Any, split: str) -> str:
    if split == "train":
        return str(params.train_data_path)
    if split == "valid":
        return str(params.valid_data_path)
    if split == "test":
        return str(_get(params, "test_dataset_path", _get(params, "valid_data_path", "")))
    raise ValueError(f"Unsupported split: {split}")


def _build_dataset(params: Any, split: str, rollout_steps: int) -> ClimateNetCDFDataset:
    data_cfg = DataConfig(
        dt=int(params.dt),
        n_history=int(params.n_history),
        in_channels=list(params.in_channels),
        out_channels=list(params.out_channels),
        crop_size_x=_get(params, "crop_size_x", None),
        crop_size_y=_get(params, "crop_size_y", None),
        roll=False,
        orography=bool(_get(params, "orography", False)),
        orography_path=_get(params, "orography_path", None),
        add_noise=False,
        noise_std=0.0,
        normalize=(str(_get(params, "normalization", "zscore")).lower() == "zscore"),
        normalization=str(_get(params, "normalization", "zscore")),
        global_means_path=params.global_means_path,
        global_stds_path=params.global_stds_path,
        add_grid=bool(_get(params, "add_grid", False)),
        gridtype=str(_get(params, "gridtype", "linear")),
        N_grid_channels=int(_get(params, "N_grid_channels", 0)),
        rollout_steps=int(rollout_steps),
        batch_size=1,
        num_workers=0,
        resolution_mode=str(_get(params, "resolution_mode", "5p625")),
        expected_grid_shape=_get(params, "expected_grid_shape", _get(params, "grid_shape", None)),
    )
    return ClimateNetCDFDataset(data_cfg, _split_data_path(params, split), train=False)


def _as_coordinate_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64).squeeze()
    except Exception:
        return None
    if arr.ndim != 1 or arr.size == 0:
        return None
    return arr


def _coordinate_pair_from_config(params: Any) -> tuple[np.ndarray, np.ndarray, str] | None:
    lat = None
    lon = None
    for key in ("lat", "lats", "latitude", "latitudes"):
        lat = _as_coordinate_array(_get(params, key, None))
        if lat is not None:
            break
    for key in ("lon", "lons", "longitude", "longitudes"):
        lon = _as_coordinate_array(_get(params, key, None))
        if lon is not None:
            break
    if lat is not None and lon is not None:
        return lat, lon, "config lat/lon arrays"
    return None


def _coordinate_pair_from_metadata_file(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, str] | None:
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        try:
            if path.suffix == ".npz":
                payload = np.load(path)
                lat_value = payload["lat"] if "lat" in payload else payload["latitude"] if "latitude" in payload else None
                lon_value = payload["lon"] if "lon" in payload else payload["longitude"] if "longitude" in payload else None
            else:
                payload = json.loads(path.read_text(encoding="utf-8"))
                lat_value = None
                lon_value = None
                if isinstance(payload, dict):
                    for key in ("lat", "lats", "latitude", "latitudes"):
                        if key in payload:
                            lat_value = payload[key]
                            break
                    for key in ("lon", "lons", "longitude", "longitudes"):
                        if key in payload:
                            lon_value = payload[key]
                            break
            lat = _as_coordinate_array(lat_value)
            lon = _as_coordinate_array(lon_value)
        except Exception:
            continue
        if lat is not None and lon is not None:
            return lat, lon, f"metadata file: {path}"
    return None


def _metadata_coordinate_paths(
    dataset: ClimateNetCDFDataset,
    params: Any,
    checkpoint_path: str | None,
) -> list[Path]:
    paths: list[Path] = []
    for root in [Path(dataset.data_dir)] + [Path(p).parent for p in dataset.files_paths[:1]]:
        paths.extend([root / "lat_lon.json", root / "coordinates.json", root / "metadata.json", root / "lat_lon.npz"])
    for stat_path in (_get(params, "global_means_path", None), _get(params, "global_stds_path", None)):
        if stat_path:
            root = Path(str(stat_path)).expanduser().parent
            paths.extend([root / "lat_lon.json", root / "coordinates.json", root / "metadata.json", root / "lat_lon.npz"])
    if checkpoint_path:
        root = Path(checkpoint_path).resolve().parent
        paths.extend([root / "lat_lon.json", root / "coordinates.json", root / "metadata.json", root / "lat_lon.npz"])

    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _fallback_lat_lon(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    dlat = 180.0 / float(height)
    dlon = 360.0 / float(width)
    lats = np.linspace(-90.0 + dlat / 2.0, 90.0 - dlat / 2.0, height, dtype=np.float64)
    lons = np.linspace(0.0, 360.0 - dlon, width, dtype=np.float64)
    return lats, lons


def lat_lon_debug_payload(
    lats: np.ndarray,
    lons: np.ndarray,
    source: str,
    expected_height: int | None = None,
    expected_width: int | None = None,
) -> dict[str, Any]:
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    warnings: list[str] = []
    if expected_height is not None and lats.size != int(expected_height):
        warnings.append(f"latitude length {lats.size} does not match grid height {expected_height}")
    if expected_width is not None and lons.size != int(expected_width):
        warnings.append(f"longitude length {lons.size} does not match grid width {expected_width}")
    if np.any(np.isclose(np.abs(lats), 90.0, atol=1.0e-8)):
        warnings.append(
            "WARNING: latitude grid includes exact poles. Check whether these are true data centers or incorrectly generated endpoints."
        )

    lat_diff = np.diff(lats) if lats.size > 1 else np.asarray([], dtype=np.float64)
    lon_diff = np.diff(lons) if lons.size > 1 else np.asarray([], dtype=np.float64)
    if lat_diff.size == 0:
        lat_order = "single"
    elif np.all(lat_diff > 0):
        lat_order = "ascending"
    elif np.all(lat_diff < 0):
        lat_order = "descending"
    else:
        lat_order = "non-monotonic"
        warnings.append("WARNING: latitude grid is non-monotonic.")
    if lon_diff.size == 0:
        lon_order = "single"
    elif np.all(lon_diff > 0):
        lon_order = "ascending"
    elif np.all(lon_diff < 0):
        lon_order = "descending"
    else:
        lon_order = "non-monotonic"
        warnings.append("WARNING: longitude grid is non-monotonic.")

    lon_min = float(np.nanmin(lons))
    lon_max = float(np.nanmax(lons))
    if lon_min >= 0.0 and lon_max > 180.0:
        lon_convention = "0-360"
    elif lon_min < 0.0 and lon_max <= 180.0:
        lon_convention = "-180-180"
    else:
        lon_convention = "mixed/unknown"
    return {
        "source": source,
        "lat_shape": list(lats.shape),
        "lon_shape": list(lons.shape),
        "lat_min": float(np.nanmin(lats)),
        "lat_max": float(np.nanmax(lats)),
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_first_5": [float(x) for x in lats[:5]],
        "lat_last_5": [float(x) for x in lats[-5:]],
        "lon_first_5": [float(x) for x in lons[:5]],
        "lon_last_5": [float(x) for x in lons[-5:]],
        "lat_order": lat_order,
        "lon_order": lon_order,
        "lon_convention": lon_convention,
        "warnings": warnings,
    }


def print_lat_lon_debug(debug: dict[str, Any]) -> None:
    print(f"lat/lon source: {debug['source']}")
    print(f"lat shape: {debug['lat_shape']}")
    print(f"lon shape: {debug['lon_shape']}")
    print(f"lat min/max: {debug['lat_min']} / {debug['lat_max']}")
    print(f"lon min/max: {debug['lon_min']} / {debug['lon_max']}")
    print(f"lat first 5: {debug['lat_first_5']}")
    print(f"lat last 5: {debug['lat_last_5']}")
    print(f"lon first 5: {debug['lon_first_5']}")
    print(f"lon last 5: {debug['lon_last_5']}")
    print(f"lat ascending or descending: {debug['lat_order']}")
    print(f"lon convention: {debug['lon_convention']}")
    for warning in debug.get("warnings", []):
        print(warning)


def get_lat_lon_from_dataset_or_config(
    dataset: ClimateNetCDFDataset,
    params: Any,
    checkpoint_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    height = int(dataset.img_shape_x)
    width = int(dataset.img_shape_y)

    for lat_key in ("lat", "lats", "latitude", "latitudes"):
        lat = _as_coordinate_array(getattr(dataset, lat_key, None))
        if lat is None:
            continue
        for lon_key in ("lon", "lons", "longitude", "longitudes"):
            lon = _as_coordinate_array(getattr(dataset, lon_key, None))
            if lon is not None:
                source = f"dataset.{lat_key}/dataset.{lon_key}"
                return lat, lon, lat_lon_debug_payload(lat, lon, source, height, width)

    nc = _load_netcdf4()
    with nc.Dataset(dataset.files_paths[0], "r") as ds:
        lat = None
        lon = None
        for key in ("latitude", "lat"):
            if key in ds.variables:
                lat = np.asarray(ds[key][:], dtype=np.float64)
                break
        for key in ("longitude", "lon"):
            if key in ds.variables:
                lon = np.asarray(ds[key][:], dtype=np.float64)
                break
        if lat is not None and lon is not None:
            source = f"NetCDF coordinates: {dataset.files_paths[0]}"
            return lat, lon, lat_lon_debug_payload(lat, lon, source, height, width)

    config_pair = _coordinate_pair_from_config(params)
    if config_pair is not None:
        lat, lon, source = config_pair
        return lat, lon, lat_lon_debug_payload(lat, lon, source, height, width)

    metadata_pair = _coordinate_pair_from_metadata_file(_metadata_coordinate_paths(dataset, params, checkpoint_path))
    if metadata_pair is not None:
        lat, lon, source = metadata_pair
        return lat, lon, lat_lon_debug_payload(lat, lon, source, height, width)

    lat, lon = _fallback_lat_lon(height, width)
    source = "fallback generated cell-center grid"
    debug = lat_lon_debug_payload(lat, lon, source, height, width)
    debug["warnings"].append(
        "WARNING: using generated latitude/longitude cell centers because no dataset/config/metadata coordinates were found."
    )
    logging.warning(debug["warnings"][-1])
    return lat, lon, debug


def _load_lat_lon_and_channels(
    dataset: ClimateNetCDFDataset,
    params: Any,
    checkpoint_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    latitudes, longitudes, _ = get_lat_lon_from_dataset_or_config(dataset, params, checkpoint_path)
    channel_names, channel_source = get_channel_names_from_dataset_or_config(dataset, params, checkpoint_path)
    return latitudes, longitudes, channel_names, channel_source


def _coordinate_orders(lats: np.ndarray, lons: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    lat_order = np.arange(lats.size)
    if lats.size > 1 and lats[0] > lats[-1]:
        lat_order = lat_order[::-1]
    sorted_lats = lats[lat_order]

    if np.nanmax(lons) > 180.0 or np.nanmin(lons) >= 0.0:
        display_lons = ((lons + 180.0) % 360.0) - 180.0
    else:
        display_lons = lons.copy()
    lon_order = np.argsort(display_lons)
    sorted_lons = display_lons[lon_order]
    return sorted_lats, sorted_lons, lat_order, lon_order


def _apply_coordinate_orders(seq: np.ndarray, lat_order: np.ndarray, lon_order: np.ndarray) -> np.ndarray:
    return seq[..., lat_order, :][..., lon_order]


def _build_model(params: Any, dataset: ClimateNetCDFDataset, inp: torch.Tensor, target_seq: torch.Tensor, device: torch.device) -> GraphWeatherModel:
    graph = load_graph_bundle(params.graph_path, map_location="cpu").to(device)
    graph_mode = graph.metadata.get("resolution_mode", None)
    active_mode = str(_get(params, "resolution_mode", "5p625"))
    if graph_mode is None and active_mode != "5p625":
        raise ValueError(f"Graph cache has no resolution metadata and cannot be used for {active_mode}.")
    if graph_mode is not None and str(graph_mode) != active_mode:
        raise ValueError(f"Graph cache mode {graph_mode} does not match active mode {active_mode}.")
    model = GraphWeatherModel(
        graph=graph,
        grid_shape=(int(dataset.img_shape_x), int(dataset.img_shape_y)),
        input_channels=int(inp.shape[0]),
        output_channels=int(target_seq.shape[1]),
        n_history=int(params.n_history),
        hidden_dim=int(_get(params, "hidden_dim", 96)),
        edge_dim=int(_get(params, "edge_dim", 6)),
        heads=int(_get(params, "num_heads", 4)),
        encoder_blocks=int(_get(params, "encoder_blocks", 1)),
        decoder_blocks=int(_get(params, "decoder_blocks", 1)),
        l0_blocks=int(_get(params, "l0_blocks", 2)),
        l1_blocks=int(_get(params, "l1_blocks", 2)),
        l2_blocks=int(_get(params, "l2_blocks", 1)),
        l1_refine_blocks=int(_get(params, "l1_refine_blocks", 1)),
        l0_refine_blocks=int(_get(params, "l0_refine_blocks", 1)),
    ).to(device)
    return model


def _load_checkpoint(model: GraphWeatherModel, checkpoint_path: str, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
    cleaned = OrderedDict()
    for key, value in state.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value
    metadata = dict(checkpoint.get("metadata", {}))
    use_delta_normalization = bool(metadata.get("use_delta_normalization", False))
    if use_delta_normalization and ("delta_mean" not in cleaned or "delta_std" not in cleaned):
        raise RuntimeError("Delta normalization is enabled but checkpoint lacks delta_mean/delta_std buffers.")
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    allowed_missing = set() if use_delta_normalization else {"delta_mean", "delta_std"}
    bad_missing = [key for key in missing if key not in allowed_missing]
    if bad_missing or unexpected:
        raise RuntimeError(f"Checkpoint model_state mismatch. Missing={bad_missing}; unexpected={list(unexpected)}")
    model.use_delta_normalization = use_delta_normalization
    model.delta_norm_center = bool(metadata.get("delta_norm_center", False))
    model.eval()
    return checkpoint


def _validate_checkpoint_resolution(checkpoint: dict[str, Any], params: Any) -> None:
    metadata = dict(checkpoint.get("metadata", {}))
    active_mode = str(_get(params, "resolution_mode", "5p625"))
    checkpoint_mode = metadata.get("resolution_mode", None)
    if checkpoint_mode is None:
        if active_mode == "5p625":
            logging.warning("Checkpoint has no resolution metadata; treating it as legacy 5p625.")
            return
        raise RuntimeError(f"Cannot visualize a legacy checkpoint without resolution metadata in {active_mode} mode.")
    if str(checkpoint_mode) != active_mode:
        raise RuntimeError(f"Cannot visualize a {checkpoint_mode} checkpoint in {active_mode} mode.")


@torch.no_grad()
def rollout_predictions(model: GraphWeatherModel, inp: torch.Tensor, rollout_steps: int, device: torch.device) -> torch.Tensor:
    model.eval()
    previous, current = model.adapter.extract_two_steps(inp[None, ...].to(device, dtype=torch.float32))
    preds: list[torch.Tensor] = []
    for _ in range(int(rollout_steps)):
        pred = model.forward_steps(previous, current)
        preds.append(pred[0].detach().cpu())
        next_current = current.clone()
        next_current[:, : model.output_channels] = pred
        previous, current = current, next_current
    return torch.stack(preds, dim=0)


def _target_sequence(target: torch.Tensor) -> torch.Tensor:
    if target.dim() == 3:
        return target.unsqueeze(0)
    if target.dim() == 4:
        return target
    raise ValueError(f"Expected target [C,H,W] or [S,C,H,W], got {tuple(target.shape)}")


def _denormalize_if_needed(
    seq: np.ndarray,
    params: Any,
    denormalize: bool,
) -> tuple[np.ndarray, bool]:
    if not denormalize:
        return seq, False
    if str(_get(params, "normalization", "zscore")).lower() != "zscore":
        return seq, False
    if not params.global_means_path or not params.global_stds_path:
        return seq, False
    stats = load_output_normalization_stats(params)
    means = stats["means"]
    stds = stats["stds"]
    if int(seq.shape[1]) != len(means):
        raise AssertionError(
            f"Sequence output channel count {seq.shape[1]} does not match normalization stats length {len(means)}"
        )
    return seq * stds.reshape(1, -1, 1, 1) + means.reshape(1, -1, 1, 1), True


def resolve_variable_channels(
    requested: list[str],
    plot_all: bool,
    out_channels: list[int],
    all_channel_names: list[str],
    allow_hardcoded_fallback: bool,
) -> list[dict[str, Any]]:
    output_name_by_local = {
        local_idx: all_channel_names[channel] if channel < len(all_channel_names) else f"Var{channel}"
        for local_idx, channel in enumerate(out_channels)
    }
    name_to_original = {_norm_name(name): idx for idx, name in enumerate(all_channel_names)}
    if plot_all:
        requested = [output_name_by_local[i] for i in range(len(out_channels))]
    if not requested:
        requested = ["z500", "t2m", "msl", "t850"]

    resolved: list[dict[str, Any]] = []
    seen: set[int] = set()
    for token in requested:
        lower = token.lower()
        norm_token = _norm_name(token)
        local_idx: int | None = None
        original_channel: int | None = None
        name = token
        canonical = _canonical_for_name(token)
        resolution_source = "unresolved"

        if lower.lstrip("-").isdigit():
            value = int(lower)
            if 0 <= value < len(out_channels):
                local_idx = value
                original_channel = int(out_channels[value])
                resolution_source = "numeric local output index"
            elif value in out_channels:
                original_channel = value
                local_idx = out_channels.index(value)
                resolution_source = "numeric original channel index"
        else:
            candidate_norms = [norm_token]
            if canonical:
                candidate_norms.extend(_norm_name(alias) for alias in VARIABLE_ALIASES.get(canonical, []))
            for candidate in candidate_norms:
                if candidate not in name_to_original:
                    continue
                original_channel = int(name_to_original[candidate])
                if original_channel in out_channels:
                    local_idx = out_channels.index(original_channel)
                    resolution_source = "metadata channel name/alias"
                    break

        if local_idx is None and canonical and allow_hardcoded_fallback:
            original_channel = int(VARIABLE_METADATA[canonical]["channel"])
            if original_channel in out_channels:
                local_idx = out_channels.index(original_channel)
                resolution_source = "hardcoded fallback"
                logging.warning(
                    "WARNING: using hardcoded variable index for %s -> channel %d. Please verify channel order.",
                    token,
                    original_channel,
                )

        if local_idx is None or original_channel is None:
            raise ValueError(
                f"Could not resolve variable '{token}'. Use a variable name present in metadata, "
                "a local output index, or an original channel index included in out_channels. "
                "Pass --allow_hardcoded_variable_fallback to use built-in fallback indices for known variables."
            )
        if local_idx in seen:
            continue
        seen.add(local_idx)

        if original_channel < len(all_channel_names):
            name = all_channel_names[original_channel]
        canonical = canonical or _canonical_for_name(name)
        if resolution_source == "hardcoded fallback" and canonical and _canonical_for_name(name) != canonical:
            logging.warning(
                "WARNING: requested %s used hardcoded channel %d, but metadata name is '%s'.",
                token,
                original_channel,
                name,
            )
        meta = _metadata_for_canonical(canonical, name)
        unit = str(meta.get("unit", ""))
        label = str(meta.get("label", name))
        logging.info(
            "Resolved variable %s -> channel %d -> actual name %s -> unit %s",
            token,
            original_channel,
            name,
            unit or "unknown",
        )
        resolved.append(
            {
                "name": name,
                "requested": token,
                "canonical_name": canonical or name,
                "label": label,
                "unit": unit,
                "local_idx": local_idx,
                "channel": original_channel,
                "resolution_source": resolution_source,
            }
        )
    return resolved


def print_available_variable_mapping(
    all_channel_names: list[str],
    out_channels: list[int],
    channel_source: str,
) -> None:
    print(f"Variable/channel source: {channel_source}")
    print("Available output variables:")
    for local_idx, original_channel in enumerate(out_channels):
        actual = all_channel_names[original_channel] if original_channel < len(all_channel_names) else f"Var{original_channel}"
        canonical = _canonical_for_name(actual) or ""
        suffix = f" canonical={canonical}" if canonical else ""
        print(f"index {local_idx}: channel {original_channel}: {actual}{suffix}")


def attach_normalization_info(
    variables: list[dict[str, Any]],
    stats_info: dict[str, Any],
) -> list[dict[str, Any]]:
    for var in variables:
        local_idx = int(var["local_idx"])
        mean_indices = stats_info.get("mean_indices", [])
        std_indices = stats_info.get("std_indices", [])
        var["normalization_mean_source"] = str(stats_info.get("mean_source", ""))
        var["normalization_std_source"] = str(stats_info.get("std_source", ""))
        var["normalization_mean_path"] = str(stats_info.get("mean_path", ""))
        var["normalization_std_path"] = str(stats_info.get("std_path", ""))
        var["normalization_mean_index"] = int(mean_indices[local_idx]) if local_idx < len(mean_indices) else None
        var["normalization_std_index"] = int(std_indices[local_idx]) if local_idx < len(std_indices) else None
        means = stats_info.get("means")
        stds = stats_info.get("stds")
        if means is not None and local_idx < len(means):
            var["normalization_mean"] = float(means[local_idx])
        if stds is not None and local_idx < len(stds):
            var["normalization_std"] = float(stds[local_idx])
    return variables


def _safe_stem(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    return safe.strip("_") or "variable"


def _robust_limits(
    arrays: list[np.ndarray],
    percentile: float,
    center_zero: bool = False,
) -> tuple[float, float]:
    values = np.concatenate([np.asarray(arr, dtype=np.float64).ravel() for arr in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -1.0, 1.0
    percentile = float(np.clip(percentile, 50.0, 100.0))
    if center_zero:
        vmax = float(np.nanpercentile(np.abs(values), percentile))
        if vmax == 0.0:
            vmax = 1.0
        return -vmax, vmax
    lower = 100.0 - percentile
    vmin = float(np.nanpercentile(values, lower))
    vmax = float(np.nanpercentile(values, percentile))
    if vmin == vmax:
        delta = abs(vmin) * 0.01 + 1.0
        return vmin - delta, vmax + delta
    return vmin, vmax


def _maybe_add_cyclic(lons: np.ndarray, field: np.ndarray, add_cyclic: bool) -> tuple[np.ndarray, np.ndarray]:
    if not add_cyclic or lons.size < 2:
        return lons, field
    span = float(np.nanmax(lons) - np.nanmin(lons))
    step = float(np.nanmedian(np.diff(np.sort(lons)))) if lons.size > 1 else 0.0
    if span + step < 350.0:
        return lons, field
    try:
        from cartopy.util import add_cyclic_point

        cyclic_field, cyclic_lons = add_cyclic_point(field, coord=lons, axis=-1)
        return np.asarray(cyclic_lons), np.asarray(cyclic_field)
    except Exception:
        cyclic_lons = np.concatenate([lons, [lons[-1] + step]])
        cyclic_field = np.concatenate([field, field[..., :1]], axis=-1)
        return cyclic_lons, cyclic_field


def _plot_panel(
    ax: Any,
    lons: np.ndarray,
    lats: np.ndarray,
    field: np.ndarray,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    use_cartopy: bool,
    show_title: bool,
    add_cyclic: bool,
) -> Any:
    if show_title:
        ax.set_title(title, fontsize=10)
    plot_lons, plot_field = _maybe_add_cyclic(lons, field, add_cyclic)
    lon_grid, lat_grid = np.meshgrid(plot_lons, lats)
    if use_cartopy:
        ccrs, cfeature = _load_cartopy()
        mesh = ax.pcolormesh(
            lon_grid,
            lat_grid,
            plot_field,
            transform=ccrs.PlateCarree(),
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_global()
        ax.coastlines(linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.35)
        ax.gridlines(draw_labels=False, linewidth=0.25, alpha=0.35)
        return mesh

    mesh = ax.pcolormesh(lon_grid, lat_grid, plot_field, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    return mesh


def _add_map_axis(fig: Any, spec: Any, use_cartopy: bool) -> Any:
    if use_cartopy:
        ccrs, _ = _load_cartopy()
        return fig.add_subplot(spec, projection=ccrs.PlateCarree())
    return fig.add_subplot(spec)


def _variable_display_unit(variable: dict[str, Any], denormalized: bool, convert_z_to_height: bool) -> str:
    if not denormalized:
        return "normalized units"
    canonical = str(variable.get("canonical_name", "")).lower()
    if convert_z_to_height and canonical.startswith("z"):
        return "m"
    return str(variable.get("unit", "")) or "physical units"


def _extract_variable_maps(
    gt_seq: np.ndarray,
    pred_seq: np.ndarray,
    variable: dict[str, Any],
    convert_z_to_height: bool,
) -> tuple[np.ndarray, np.ndarray]:
    local_idx = int(variable["local_idx"])
    gt_maps = np.asarray(gt_seq[:, local_idx], dtype=np.float64)
    pred_maps = np.asarray(pred_seq[:, local_idx], dtype=np.float64)
    canonical = str(variable.get("canonical_name", "")).lower()
    if convert_z_to_height and canonical.startswith("z"):
        gt_maps = gt_maps / GRAVITY
        pred_maps = pred_maps / GRAVITY
    return gt_maps, pred_maps


def plot_variable_rollout_maps(
    gt_maps: np.ndarray,
    pred_maps: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    variable: dict[str, Any],
    lead_times: list[int],
    output_dir: Path,
    output_prefix: str,
    aggregate_label: str,
    checkpoint_path: str,
    rollout_steps: int,
    unit_label: str,
    cmap_main: str,
    cmap_bias: str,
    dpi: int,
    use_cartopy: bool,
    show_titles: bool,
    save_arrays: bool,
    robust_percentile: float,
    same_scale_across_leads: bool,
    add_cyclic: bool,
    avg_sample_rmse: np.ndarray | None = None,
    resolution_mode: str = "5p625",
    grid_shape: tuple[int, int] | None = None,
) -> dict[str, Any]:
    rows = len(lead_times)
    fig = plt.figure(
        figsize=(18.0, max(4.0, 4.0 * rows)),
        constrained_layout=False,
    )
    gs = fig.add_gridspec(
        rows,
        5,
        width_ratios=[1.0, 1.0, 0.035, 1.0, 0.035],
        wspace=0.25,
        hspace=0.35,
    )
    variable_name = str(variable["name"])
    variable_label = str(variable["label"])
    metrics: dict[str, Any] = {}
    selected_indices = [int(lead) - 1 for lead in lead_times]
    shared_main_limits: tuple[float, float] | None = None
    if same_scale_across_leads:
        shared_main_limits = _robust_limits(
            [gt_maps[selected_indices], pred_maps[selected_indices]],
            robust_percentile,
            center_zero=False,
        )

    for row, lead_time in enumerate(lead_times):
        step_idx = int(lead_time) - 1
        gt = gt_maps[step_idx]
        pred = pred_maps[step_idx]
        bias = pred - gt
        main_vmin, main_vmax = shared_main_limits or _robust_limits(
            [gt, pred],
            robust_percentile,
            center_zero=False,
        )
        bias_vmin, bias_vmax = _robust_limits([bias], robust_percentile, center_zero=True)

        rmse = float(np.sqrt(np.nanmean(np.square(bias))))
        bias_mean = float(np.nanmean(bias))
        metrics[str(lead_time)] = {
            "rmse": rmse,
            "avg_sample_rmse": None if avg_sample_rmse is None else float(avg_sample_rmse[step_idx]),
            "bias_mean": bias_mean,
            "bias_min": float(np.nanmin(bias)),
            "bias_max": float(np.nanmax(bias)),
            "gt_min": float(np.nanmin(gt)),
            "gt_max": float(np.nanmax(gt)),
            "pred_min": float(np.nanmin(pred)),
            "pred_max": float(np.nanmax(pred)),
        }

        ax_gt = _add_map_axis(fig, gs[row, 0], use_cartopy)
        ax_pred = _add_map_axis(fig, gs[row, 1], use_cartopy)
        cax_main = fig.add_subplot(gs[row, 2])
        ax_bias = _add_map_axis(fig, gs[row, 3], use_cartopy)
        cax_bias = fig.add_subplot(gs[row, 4])

        gt_mesh = _plot_panel(
            ax_gt,
            lons,
            lats,
            gt,
            f"GT {aggregate_label} day {lead_time}",
            cmap_main,
            main_vmin,
            main_vmax,
            use_cartopy,
            show_titles,
            add_cyclic,
        )
        pred_mesh = _plot_panel(
            ax_pred,
            lons,
            lats,
            pred,
            f"Prediction {aggregate_label} day {lead_time}",
            cmap_main,
            main_vmin,
            main_vmax,
            use_cartopy,
            show_titles,
            add_cyclic,
        )
        bias_mesh = _plot_panel(
            ax_bias,
            lons,
            lats,
            bias,
            f"Bias {aggregate_label} day {lead_time}\nPred - GT | RMSE={rmse:.3g}",
            cmap_bias,
            bias_vmin,
            bias_vmax,
            use_cartopy,
            show_titles,
            add_cyclic,
        )

        main_cb = fig.colorbar(pred_mesh, cax=cax_main)
        main_cb.ax.tick_params(labelsize=8)
        main_cb.set_label(unit_label, fontsize=9)
        bias_cb = fig.colorbar(bias_mesh, cax=cax_bias)
        bias_cb.ax.tick_params(labelsize=8)
        bias_cb.set_label(unit_label, fontsize=9)

        if save_arrays:
            prefix = output_dir / f"{output_prefix}_{_safe_stem(variable_name)}_day{lead_time:02d}"
            np.save(f"{prefix}_gt.npy", gt)
            np.save(f"{prefix}_pred.npy", pred)
            np.save(f"{prefix}_bias.npy", bias)

    checkpoint_stem = Path(checkpoint_path).stem
    if show_titles:
        grid_text = ""
        if grid_shape is not None:
            grid_text = f" | grid: {int(grid_shape[0])}x{int(grid_shape[1])}"
        fig.suptitle(
            f"Variable: {variable_label} | {aggregate_label} | checkpoint: {checkpoint_stem} | "
            f"mode: {resolution_mode}{grid_text} | fixed rollout {rollout_steps} days",
            fontsize=13,
        )
    fig.subplots_adjust(
        left=0.035,
        right=0.975,
        bottom=0.040,
        top=0.92 if show_titles else 0.985,
    )
    output_path = output_dir / f"{output_prefix}_{_safe_stem(variable_name)}_rollout_maps.png"
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved rollout map figure: %s", output_path)
    print(f"  {variable_name:<8} -> {output_path}")
    return metrics


def _denormalized_sequences(
    pred_seq: np.ndarray,
    target_seq: torch.Tensor,
    params: Any,
    denormalize: bool,
) -> tuple[np.ndarray, np.ndarray, bool]:
    gt_seq = target_seq.numpy()
    pred_seq, pred_denorm = _denormalize_if_needed(pred_seq, params, denormalize)
    gt_seq, gt_denorm = _denormalize_if_needed(gt_seq, params, denormalize)
    return pred_seq, gt_seq, pred_denorm and gt_denorm


def _load_sample_sequences(
    dataset: ClimateNetCDFDataset,
    model: GraphWeatherModel,
    params: Any,
    sample_index: int,
    rollout_steps: int,
    device: torch.device,
    denormalize: bool,
    lat_order: np.ndarray,
    lon_order: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    inp, target = dataset[int(sample_index)]
    target_seq = _target_sequence(target)
    pred_seq = rollout_predictions(model, inp, rollout_steps, device).numpy()
    pred_seq, gt_seq, denormalized = _denormalized_sequences(pred_seq, target_seq, params, denormalize)
    pred_seq = _apply_coordinate_orders(pred_seq, lat_order, lon_order)
    gt_seq = _apply_coordinate_orders(gt_seq, lat_order, lon_order)
    return gt_seq, pred_seq, denormalized


def _aggregate_year_mean(
    dataset: ClimateNetCDFDataset,
    model: GraphWeatherModel,
    params: Any,
    variables: list[dict[str, Any]],
    rollout_steps: int,
    device: torch.device,
    denormalize: bool,
    convert_z_to_height: bool,
    lat_order: np.ndarray,
    lon_order: np.ndarray,
    max_batches: int | None,
) -> tuple[dict[str, dict[str, Any]], bool, int]:
    n_samples = len(dataset)
    if max_batches is not None and max_batches > 0:
        n_samples = min(n_samples, int(max_batches))
    if n_samples <= 0:
        raise RuntimeError("No samples available for year_mean aggregation.")

    height = int(dataset.img_shape_x)
    width = int(dataset.img_shape_y)
    accum: dict[str, dict[str, Any]] = {}
    for variable in variables:
        key = str(variable["name"])
        accum[key] = {
            "gt_sum": np.zeros((rollout_steps, height, width), dtype=np.float64),
            "pred_sum": np.zeros((rollout_steps, height, width), dtype=np.float64),
            "sample_rmse_sum": np.zeros((rollout_steps,), dtype=np.float64),
        }

    denorm_all = True
    for sample_idx in range(n_samples):
        inp, target = dataset[sample_idx]
        target_seq = _target_sequence(target)
        pred_seq = rollout_predictions(model, inp, rollout_steps, device).numpy()
        pred_seq, gt_seq, denormalized = _denormalized_sequences(pred_seq, target_seq, params, denormalize)
        denorm_all = denorm_all and denormalized
        pred_seq = _apply_coordinate_orders(pred_seq, lat_order, lon_order)
        gt_seq = _apply_coordinate_orders(gt_seq, lat_order, lon_order)

        for variable in variables:
            key = str(variable["name"])
            gt_maps, pred_maps = _extract_variable_maps(
                gt_seq,
                pred_seq,
                variable,
                convert_z_to_height and denormalized,
            )
            bias = pred_maps - gt_maps
            accum[key]["gt_sum"] += gt_maps
            accum[key]["pred_sum"] += pred_maps
            accum[key]["sample_rmse_sum"] += np.sqrt(np.nanmean(np.square(bias), axis=(-2, -1)))

        if (sample_idx + 1) % 25 == 0 or (sample_idx + 1) == n_samples:
            logging.info("Aggregated %d/%d samples for year_mean visualization", sample_idx + 1, n_samples)

    result: dict[str, dict[str, Any]] = {}
    for variable in variables:
        key = str(variable["name"])
        result[key] = {
            "gt": accum[key]["gt_sum"] / float(n_samples),
            "pred": accum[key]["pred_sum"] / float(n_samples),
            "avg_sample_rmse": accum[key]["sample_rmse_sum"] / float(n_samples),
        }
    return result, denorm_all, n_samples


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logging.info("Saved JSON: %s", path)


def _mapping_payload(variables: list[dict[str, Any]], channel_source: str) -> dict[str, Any]:
    return {
        "channel_source": channel_source,
        "variables": {
            str(var["requested"]): {
                "requested_variable": str(var["requested"]),
                "channel": int(var["channel"]),
                "local_idx": int(var["local_idx"]),
                "actual_name": str(var["name"]),
                "canonical_name": str(var["canonical_name"]),
                "unit": str(var["unit"]),
                "resolution_source": str(var["resolution_source"]),
                "normalization_mean_source": str(var.get("normalization_mean_source", "")),
                "normalization_std_source": str(var.get("normalization_std_source", "")),
                "normalization_mean_path": str(var.get("normalization_mean_path", "")),
                "normalization_std_path": str(var.get("normalization_std_path", "")),
                "normalization_mean_channel_index": var.get("normalization_mean_index"),
                "normalization_std_channel_index": var.get("normalization_std_index"),
                "normalization_mean": var.get("normalization_mean"),
                "normalization_std": var.get("normalization_std"),
            }
            for var in variables
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize GT, prediction, and bias rollout maps with Cartopy.")
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--config", default=None, type=str, help="Config YAML path or config section name.")
    parser.add_argument("--yaml_config", default=None, type=str, help="Explicit YAML config path.")
    parser.add_argument("--config_name", default=None, type=str, help="YAML section name, default raw_5p625.")
    parser.add_argument("--resolution_mode", default=None, type=str, help="5p625 or 2p5; overrides YAML resolution_mode.")
    parser.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--aggregate_mode", default="sample", choices=["sample", "year_mean"])
    parser.add_argument("--sample_index", default=0, type=int)
    parser.add_argument("--rollout_steps", default=10, type=int)
    parser.add_argument("--lead_times", nargs="+", default=["1", "3", "5", "10"])
    parser.add_argument("--variables", nargs="*", default=["z500", "t2m", "msl", "t850"])
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--max_batches", default=None, type=int, help="For year_mean mode, maximum number of batch-size-1 samples to aggregate.")
    parser.add_argument("--dpi", default=160, type=int)
    parser.add_argument("--cmap_main", default="viridis", type=str)
    parser.add_argument("--cmap_bias", default="RdBu_r", type=str)
    parser.add_argument("--robust_percentile", default=99.0, type=float)
    parser.add_argument("--same_scale_across_leads", action="store_true")
    parser.add_argument("--plot_all_variables", action="store_true")
    parser.add_argument("--list_channels", action="store_true")
    parser.add_argument("--print_variable_mapping", action="store_true")
    parser.add_argument("--allow_hardcoded_variable_fallback", action="store_true")
    parser.add_argument("--debug_values", action="store_true", help="Reserved for scripts/debug_variable_visualization.py diagnostics.")
    parser.add_argument("--debug_latlon", action="store_true")
    parser.add_argument("--debug_edge_rows", action="store_true", help="Reserved for scripts/debug_variable_visualization.py diagnostics.")
    parser.add_argument("--exclude_edge_lat_rows", action="store_true", help="Debug-only: crop first/last latitude rows from plotted maps.")
    parser.add_argument("--denormalize", dest="denormalize", action="store_true", default=True)
    parser.add_argument("--no_denormalize", dest="denormalize", action="store_false")
    parser.add_argument("--convert_z_to_height", action="store_true")
    parser.add_argument("--use_cartopy", dest="use_cartopy", action="store_true", default=True)
    parser.add_argument("--no_use_cartopy", dest="use_cartopy", action="store_false")
    parser.add_argument("--add_cyclic", dest="add_cyclic", action="store_true", default=True)
    parser.add_argument("--no_add_cyclic", dest="add_cyclic", action="store_false")
    parser.add_argument("--show_titles", dest="show_titles", action="store_true", default=True)
    parser.add_argument("--no_show_titles", dest="show_titles", action="store_false")
    parser.add_argument("--save_arrays", action="store_true")
    parser.add_argument("--device", default=None, type=str)
    args = parser.parse_args()

    yaml_path, config_name = _resolve_config_args(args)
    params = YParams(yaml_path, config_name, resolution_mode=args.resolution_mode)
    rollout_steps = int(args.rollout_steps)
    lead_times = [int(x) for x in _parse_items([str(v) for v in args.lead_times])]
    if not lead_times:
        raise ValueError("At least one lead time is required.")
    invalid = [lead for lead in lead_times if lead < 1 or lead > rollout_steps]
    if invalid:
        raise ValueError(f"Lead times must be in [1, rollout_steps={rollout_steps}], got {invalid}")

    dataset = _build_dataset(params, args.split, rollout_steps)
    lats_raw, lons_raw, lat_lon_debug = get_lat_lon_from_dataset_or_config(dataset, params, args.checkpoint)
    channel_names, channel_source = get_channel_names_from_dataset_or_config(dataset, params, args.checkpoint)
    out_channels = [int(x) for x in params.out_channels]
    stats_info: dict[str, Any] | None = None
    if str(_get(params, "normalization", "zscore")).lower() == "zscore":
        stats_info = load_output_normalization_stats(params)
    if args.list_channels or args.print_variable_mapping:
        print_available_variable_mapping(channel_names, out_channels, channel_source)
    if args.debug_latlon:
        print()
        print_lat_lon_debug(lat_lon_debug)
    if args.print_variable_mapping:
        requested = _parse_items(args.variables)
        if requested:
            print("\nRequested variable resolution:")
            variables = resolve_variable_channels(
                requested=requested,
                plot_all=bool(args.plot_all_variables),
                out_channels=out_channels,
                all_channel_names=channel_names,
                allow_hardcoded_fallback=bool(args.allow_hardcoded_variable_fallback),
            )
            if stats_info:
                attach_normalization_info(variables, stats_info)
            for var in variables:
                print(
                    f"  {var['requested']} -> channel {var['channel']}, "
                    f"actual_name={var['name']}, unit={var['unit'] or 'unknown'}, "
                    f"mean/std source={var.get('normalization_mean_path', 'unknown')}/"
                    f"{var.get('normalization_std_path', 'unknown')}, "
                    f"mean/std channel index={var.get('normalization_mean_index')}/"
                    f"{var.get('normalization_std_index')}, source={var['resolution_source']}"
                )
        return
    if args.list_channels:
        return

    if not args.checkpoint:
        raise ValueError("--checkpoint is required unless --print_variable_mapping is used.")
    checkpoint_path = _resolve_path(args.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    output_dir = Path(args.output_dir) if args.output_dir else Path(checkpoint_path).resolve().parent / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(rank=0, log_file=str(output_dir / "visualize_rollout_maps.log"))
    params.log()

    if args.use_cartopy:
        _load_cartopy()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logging.info("Using device: %s", device)
    if args.aggregate_mode == "sample" and (args.sample_index < 0 or args.sample_index >= len(dataset)):
        raise IndexError(f"sample_index={args.sample_index} is outside split length {len(dataset)}")

    lats, lons, lat_order, lon_order = _coordinate_orders(lats_raw, lons_raw)
    variables = resolve_variable_channels(
        requested=_parse_items(args.variables),
        plot_all=bool(args.plot_all_variables),
        out_channels=out_channels,
        all_channel_names=channel_names,
        allow_hardcoded_fallback=bool(args.allow_hardcoded_variable_fallback),
    )
    if stats_info:
        attach_normalization_info(variables, stats_info)
    _write_json(output_dir / "variable_channel_mapping.json", _mapping_payload(variables, channel_source))
    if args.debug_latlon:
        print_lat_lon_debug(lat_lon_debug)
        _write_json(output_dir / "lat_lon_debug.json", lat_lon_debug)

    model_sample_idx = int(args.sample_index) if args.aggregate_mode == "sample" else 0
    inp, target = dataset[model_sample_idx]
    target_seq = _target_sequence(target)
    model = _build_model(params, dataset, inp, target_seq, device)
    checkpoint = _load_checkpoint(model, checkpoint_path, device)
    _validate_checkpoint_resolution(checkpoint, params)
    metadata = dict(checkpoint.get("metadata", {}))
    logging.info(
        "Loaded checkpoint: %s | epoch=%s | train_rollout_steps=%s",
        checkpoint_path,
        metadata.get("epoch", checkpoint.get("epoch", "unknown")),
        metadata.get("train_rollout_steps", "unknown"),
    )

    metadata_payload: dict[str, Any] = {
        "checkpoint": checkpoint_path,
        "checkpoint_epoch": metadata.get("epoch", checkpoint.get("epoch", None)),
        "checkpoint_train_rollout_steps": metadata.get("train_rollout_steps"),
        "split": args.split,
        "aggregate_mode": args.aggregate_mode,
        "sample_index": int(args.sample_index) if args.aggregate_mode == "sample" else None,
        "max_batches": args.max_batches,
        "rollout_steps": rollout_steps,
        "lead_times": lead_times,
        "resolution_mode": str(_get(params, "resolution_mode", "5p625")),
        "grid_shape": [int(dataset.img_shape_x), int(dataset.img_shape_y)],
        "denormalize_requested": bool(args.denormalize),
        "denormalized": None,
        "convert_z_to_height": bool(args.convert_z_to_height),
        "channel_source": channel_source,
        "lat_lon": lat_lon_debug,
        "bias_definition": "prediction_minus_ground_truth",
        "exclude_edge_lat_rows": bool(args.exclude_edge_lat_rows),
        "color_scale": {
            "cmap_main": str(args.cmap_main),
            "cmap_bias": str(args.cmap_bias),
            "robust_percentile": float(args.robust_percentile),
            "same_scale_across_leads": bool(args.same_scale_across_leads),
        },
        "cartopy": {
            "use_cartopy": bool(args.use_cartopy),
            "add_cyclic": bool(args.add_cyclic),
        },
        "variables": {},
    }

    if args.aggregate_mode == "sample":
        gt_seq, pred_seq, denormalized = _load_sample_sequences(
            dataset=dataset,
            model=model,
            params=params,
            sample_index=int(args.sample_index),
            rollout_steps=rollout_steps,
            device=device,
            denormalize=bool(args.denormalize),
            lat_order=lat_order,
            lon_order=lon_order,
        )
        if args.denormalize and not denormalized:
            logging.warning("Denormalization was requested but could not be applied; plotting normalized values.")
        aggregate_data = None
        output_prefix = f"sample{int(args.sample_index):03d}"
        aggregate_label = f"sample {int(args.sample_index)}"
        print("Saved sample rollout maps:")
    else:
        aggregate_data, denormalized, sample_count = _aggregate_year_mean(
            dataset=dataset,
            model=model,
            params=params,
            variables=variables,
            rollout_steps=rollout_steps,
            device=device,
            denormalize=bool(args.denormalize),
            convert_z_to_height=bool(args.convert_z_to_height),
            lat_order=lat_order,
            lon_order=lon_order,
            max_batches=args.max_batches,
        )
        if args.denormalize and not denormalized:
            logging.warning("Denormalization was requested but could not be applied for every sample; plotting normalized values.")
        metadata_payload["sample_count"] = sample_count
        output_prefix = "yearmean"
        aggregate_label = "year mean"
        print("Saved year-mean rollout maps:")

    metadata_payload["denormalized"] = bool(denormalized)
    for variable in variables:
        if args.aggregate_mode == "sample":
            gt_maps, pred_maps = _extract_variable_maps(
                gt_seq,
                pred_seq,
                variable,
                bool(args.convert_z_to_height and denormalized),
            )
            avg_sample_rmse = None
        else:
            var_data = aggregate_data[str(variable["name"])]
            gt_maps = var_data["gt"]
            pred_maps = var_data["pred"]
            avg_sample_rmse = var_data["avg_sample_rmse"]
        unit_label = _variable_display_unit(variable, bool(denormalized), bool(args.convert_z_to_height))
        plot_gt_maps = gt_maps
        plot_pred_maps = pred_maps
        plot_lats = lats
        plot_prefix = output_prefix
        plot_label = aggregate_label
        if args.exclude_edge_lat_rows:
            if gt_maps.shape[-2] <= 2:
                raise ValueError("--exclude_edge_lat_rows requires at least 3 latitude rows.")
            plot_gt_maps = gt_maps[..., 1:-1, :]
            plot_pred_maps = pred_maps[..., 1:-1, :]
            plot_lats = lats[1:-1]
            plot_prefix = f"{output_prefix}_without_edge_rows"
            plot_label = f"{aggregate_label} without first/last latitude rows"
        metrics = plot_variable_rollout_maps(
            gt_maps=plot_gt_maps,
            pred_maps=plot_pred_maps,
            lats=plot_lats,
            lons=lons,
            variable=variable,
            lead_times=lead_times,
            output_dir=output_dir,
            output_prefix=plot_prefix,
            aggregate_label=plot_label,
            checkpoint_path=checkpoint_path,
            rollout_steps=rollout_steps,
            unit_label=unit_label,
            cmap_main=str(args.cmap_main),
            cmap_bias=str(args.cmap_bias),
            dpi=int(args.dpi),
            use_cartopy=bool(args.use_cartopy),
            show_titles=bool(args.show_titles),
            save_arrays=bool(args.save_arrays),
            robust_percentile=float(args.robust_percentile),
            same_scale_across_leads=bool(args.same_scale_across_leads),
            add_cyclic=bool(args.add_cyclic),
            avg_sample_rmse=avg_sample_rmse,
            resolution_mode=str(_get(params, "resolution_mode", "5p625")),
            grid_shape=(int(dataset.img_shape_x), int(dataset.img_shape_y)),
        )
        metadata_payload["variables"][str(variable["name"])] = {
            "label": variable["label"],
            "channel": int(variable["channel"]),
            "local_idx": int(variable["local_idx"]),
            "canonical_name": str(variable["canonical_name"]),
            "unit": unit_label,
            "lead_metrics": metrics,
        }
        for lead, lead_metrics in metrics.items():
            if args.aggregate_mode == "year_mean":
                print(f"{variable['name']} day {lead} mean-bias RMSE: {lead_metrics['rmse']:.6g}")
            else:
                print(f"{variable['name']} day {lead} bias RMSE: {lead_metrics['rmse']:.6g}")

    _write_json(output_dir / "visualization_metadata.json", metadata_payload)
    print("\nVariable mapping:")
    for variable in variables:
        print(
            f"  {variable['requested']:<8} -> channel {variable['channel']}, "
            f"actual_name={variable['name']}, unit={metadata_payload['variables'][str(variable['name'])]['unit']}"
        )


if __name__ == "__main__":
    main()
