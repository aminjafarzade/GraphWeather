from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

Path(os.environ.setdefault("XDG_CACHE_HOME", "/tmp")).mkdir(parents=True, exist_ok=True)
Path(os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")).mkdir(parents=True, exist_ok=True)

from src.batch_adapter import GridNodeAdapter
from src.config import YParams, setup_logging
from src.data import ClimateNetCDFDataset, DataConfig
from src.device import _resolve_device
from src.features import (
    ExtraFeatureSettings,
    RolloutFeatureBuilder,
    VariableResolver,
    canonical_name,
    dayofyear_sincos,
    feature_metadata_matches,
)
from src.graph_bundle import load_graph_bundle
from src.losses import LatitudeWeightedMSE
from src.models import GraphWeatherModel
from src.architecture import validate_checkpoint_architecture


VARIABLES_TO_REPORT = ["t2m", "msl", "t850", "z500", "tisr", "orog", "lsm"]
CHECK_TOL = 1.0e-6
LOOSE_TOL = 1.0e-4


def _get(params: Any, name: str, default: Any = None) -> Any:
    if isinstance(params, dict):
        return params.get(name, default)
    return getattr(params, name, default)


def _nested(params: Any, *keys: str, default: Any = None) -> Any:
    current = params
    for key in keys:
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            current = getattr(current, key, default)
    return current


def _looks_like_yaml(value: str) -> bool:
    return value.endswith((".yaml", ".yml")) or Path(value).expanduser().exists()


def _resolve_config_args(args: argparse.Namespace) -> tuple[str, str]:
    yaml_path = args.yaml_config
    config_name = args.config_name
    if args.config:
        if _looks_like_yaml(args.config):
            yaml_path = args.config
            if config_name is None:
                if Path(args.config).name == "weather_dual_resolution.yaml":
                    config_name = "raw_static_forcing"
                elif Path(args.config).name == "weather_dual_resolution_l3.yaml":
                    config_name = "raw_l3"
                elif Path(args.config).name == "weather_dual_resolution_l3_stage_warmup_cosine.yaml":
                    config_name = "raw_l3_stage_warmup_cosine"
                elif Path(args.config).name == "weather_dual_resolution_l3_blocks3.yaml":
                    config_name = "raw_l3_blocks3"
                elif Path(args.config).name == "weather_dual_resolution_l3_heavy_unet.yaml":
                    config_name = "raw_l3_heavy_unet"
                elif Path(args.config).name == "weather_dual_resolution_l3_full_rollout.yaml":
                    config_name = "raw_l3_full_rollout"
                elif Path(args.config).name == "weather_dual_resolution_l3_hidden128.yaml":
                    config_name = "raw_l3_hidden128"
                elif Path(args.config).name == "weather_dual_resolution_l3_hidden160.yaml":
                    config_name = "raw_l3_hidden160"
                elif Path(args.config).name == "weather_dual_resolution_l4_ratio15_hidden128_fixed_orog.yaml":
                    config_name = "raw_l4_ratio15_hidden128_fixed_orog"
                elif Path(args.config).name == "weather_dual_resolution_l4_hidden128_72_36_24_18_9_fixed_orog.yaml":
                    config_name = "raw_l4_hidden128_72_36_24_18_9_fixed_orog"
        elif config_name is None:
            config_name = args.config
    yaml_path = yaml_path or str(project_root / "configs" / "gnn_5p625.yaml")
    config_name = config_name or "raw_5p625"
    return str(Path(yaml_path).expanduser().resolve()), config_name


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
        roll=bool(_get(params, "roll", False)),
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
        return_metadata=bool((_get(params, "extra_features", {}) or {}).get("enabled", False))
        if isinstance(_get(params, "extra_features", {}) or {}, dict)
        else False,
    )
    return ClimateNetCDFDataset(data_cfg, _split_data_path(params, split), train=(split == "train"))


def _unpack_sample(sample: Any) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if isinstance(sample, dict):
        return sample["input"], sample["target"], {k: v for k, v in sample.items() if k not in {"input", "target"}}
    inp, target = sample
    return inp, target, {}


def _target_sequence(target: torch.Tensor) -> torch.Tensor:
    if target.dim() == 3:
        return target.unsqueeze(0)
    if target.dim() == 4:
        return target
    raise ValueError(f"Expected target [C,H,W] or [S,C,H,W], got {tuple(target.shape)}")


def _metadata_batch(metadata: dict[str, Any], key: str, device: torch.device | str = "cpu") -> torch.Tensor | None:
    value = metadata.get(key)
    if not torch.is_tensor(value):
        return None
    if value.dim() == 1:
        value = value.unsqueeze(0)
    return value.to(device=device)


def _feature_column(builder: RolloutFeatureBuilder, name: str) -> int | None:
    try:
        return builder.feature_names.index(name)
    except ValueError:
        return None


def _grid_from_nodes(values: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return values.reshape(height, width)


def _max_mean_diff(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    diff = (a.detach().to(torch.float32).cpu() - b.detach().to(torch.float32).cpu()).abs()
    return float(diff.max().item()), float(diff.mean().item())


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return None
    x = x[mask]
    y = y[mask]
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _status(passed: bool | None, message: str = "") -> dict[str, Any]:
    if passed is None:
        state = "SKIP"
    elif passed:
        state = "PASS"
    else:
        state = "FAIL"
    return {"status": state, "pass": passed, "message": message}


def _tensor_stats(values: torch.Tensor) -> dict[str, float | int]:
    arr = values.detach().to(torch.float32).cpu().numpy().reshape(-1)
    finite = np.isfinite(arr)
    finite_values = arr[finite]
    return {
        "min": None if finite_values.size == 0 else float(np.min(finite_values)),
        "max": None if finite_values.size == 0 else float(np.max(finite_values)),
        "mean": None if finite_values.size == 0 else float(np.mean(finite_values)),
        "std": None if finite_values.size == 0 else float(np.std(finite_values)),
        "nan_count": int(np.isnan(arr).sum()),
        "inf_count": int(np.isinf(arr).sum()),
    }


@dataclass
class FeatureStatsAccumulator:
    names: list[str]

    def __post_init__(self) -> None:
        n = len(self.names)
        self.count = np.zeros(n, dtype=np.int64)
        self.sum = np.zeros(n, dtype=np.float64)
        self.sumsq = np.zeros(n, dtype=np.float64)
        self.min = np.full(n, np.inf, dtype=np.float64)
        self.max = np.full(n, -np.inf, dtype=np.float64)
        self.nan_count = np.zeros(n, dtype=np.int64)
        self.inf_count = np.zeros(n, dtype=np.int64)

    def update(self, aux: torch.Tensor) -> None:
        arr = aux.detach().to(torch.float32).cpu().numpy().reshape(-1, len(self.names))
        for idx in range(len(self.names)):
            values = arr[:, idx]
            self.nan_count[idx] += int(np.isnan(values).sum())
            self.inf_count[idx] += int(np.isinf(values).sum())
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            self.count[idx] += int(finite.size)
            self.sum[idx] += float(np.sum(finite))
            self.sumsq[idx] += float(np.sum(np.square(finite, dtype=np.float64)))
            self.min[idx] = min(self.min[idx], float(np.min(finite)))
            self.max[idx] = max(self.max[idx], float(np.max(finite)))

    def rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for idx, name in enumerate(self.names):
            count = int(self.count[idx])
            mean = None if count == 0 else float(self.sum[idx] / count)
            variance = None if count == 0 else max(0.0, float(self.sumsq[idx] / count - (mean or 0.0) ** 2))
            rows.append(
                {
                    "name": name,
                    "min": None if count == 0 else float(self.min[idx]),
                    "max": None if count == 0 else float(self.max[idx]),
                    "mean": mean,
                    "std": None if variance is None else float(math.sqrt(variance)),
                    "nan_count": int(self.nan_count[idx]),
                    "inf_count": int(self.inf_count[idx]),
                    "finite_count": count,
                }
            )
        return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _resolve_variables(
    params: Any,
    dataset: ClimateNetCDFDataset,
    settings: ExtraFeatureSettings,
    logger: logging.Logger | Any = logging,
) -> dict[str, Any]:
    resolver = VariableResolver(params, getattr(dataset, "channel_names", None), list(params.out_channels), logger=logger)
    required: set[str] = set()
    if settings.known_forcings_enabled:
        required.update(canonical_name(v) or v for v in settings.known_forcing_variables)
    resolved: dict[str, Any] = {}
    for variable in VARIABLES_TO_REPORT:
        item = resolver.resolve(variable, required=variable in required)
        resolved[variable] = {
            "canonical": item.canonical,
            "channel": item.channel,
            "local_index": item.local_index,
            "actual_name": item.actual_name,
            "source": item.source,
        }
    if settings.known_forcings_enabled:
        for variable in settings.known_forcing_variables:
            canonical = canonical_name(variable) or variable
            item = resolver.resolve(canonical, required=True)
            if item.local_index is None:
                raise ValueError(f"{canonical!r} is configured as a known forcing but is not present in out_channels.")
    if "orog" in {canonical_name(v) or v for v in settings.copy_variables} and resolved["orog"]["local_index"] is None:
        message = "orog is configured for copy/static handling but could not be resolved."
        if settings.require_static_features:
            raise ValueError(message)
        logging.warning("WARNING: %s require_static_features=false, so this is reported but not fatal.", message)
    return {"resolved_variables": resolved}


def _dataset_lat_lon(dataset: ClimateNetCDFDataset, graph: Any) -> tuple[np.ndarray, np.ndarray, str]:
    path = dataset.files_paths[0]
    with dataset.nc.Dataset(path, "r") as ds:
        lat = None
        lon = None
        for key in ("latitude", "lat"):
            if key in ds.variables:
                lat = np.asarray(ds.variables[key][:], dtype=np.float64).reshape(-1)
                break
        for key in ("longitude", "lon"):
            if key in ds.variables:
                lon = np.asarray(ds.variables[key][:], dtype=np.float64).reshape(-1)
                break
        if lat is not None and lon is not None:
            return lat, lon, f"NetCDF coordinates: {path}"
    lat_grid = graph.L0.lat_lon[:, 0].detach().cpu().numpy().reshape(int(graph.L0.height), int(graph.L0.width))
    lon_grid = graph.L0.lat_lon[:, 1].detach().cpu().numpy().reshape(int(graph.L0.height), int(graph.L0.width))
    lat = lat_grid[:, 0]
    lon = lon_grid[0]
    if np.nanmax(np.abs(lat)) <= math.pi / 2.0 + 1.0e-4:
        lat = np.rad2deg(lat)
    if np.nanmax(np.abs(lon)) <= 2.0 * math.pi + 1.0e-4:
        lon = np.rad2deg(lon)
    return lat, lon, "graph L0 lat_lon"


def _check_lat_lon_ordering(
    builder: RolloutFeatureBuilder,
    aux: torch.Tensor | None,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    height: int,
    width: int,
) -> dict[str, Any]:
    if aux is None:
        return {**_status(None, "extra features disabled"), "rows": []}
    columns = {name: _feature_column(builder, name) for name in ("sin_lat", "cos_lat", "sin_lon", "cos_lon")}
    if any(value is None for value in columns.values()):
        return {**_status(None, "lat/lon sin-cos features are not enabled"), "rows": []}
    rows_to_check = [
        (0, 0),
        (0, width - 1),
        (height // 2, width // 2),
        (height - 1, 0),
        (height - 1, width - 1),
    ]
    details = []
    max_diff = 0.0
    for row, col in rows_to_check:
        node_id = row * width + col
        lat = float(latitudes[row])
        lon = float(longitudes[col])
        expected = {
            "sin_lat": math.sin(math.radians(lat)),
            "cos_lat": math.cos(math.radians(lat)),
            "sin_lon": math.sin(math.radians(lon)),
            "cos_lon": math.cos(math.radians(lon)),
        }
        actual = {name: float(aux[0, node_id, int(col_idx)].item()) for name, col_idx in columns.items() if col_idx is not None}
        row_diff = max(abs(actual[name] - expected[name]) for name in expected)
        max_diff = max(max_diff, row_diff)
        details.append(
            {
                "node_id": node_id,
                "row": row,
                "col": col,
                "lat": lat,
                "lon": lon,
                **{f"expected_{k}": v for k, v in expected.items()},
                **{f"actual_{k}": v for k, v in actual.items()},
                "max_abs_diff": row_diff,
                "pass": row_diff < CHECK_TOL,
            }
        )
    sin_lat_grid = _grid_from_nodes(aux[0, :, int(columns["sin_lat"])], height, width)
    sin_lon_grid = _grid_from_nodes(aux[0, :, int(columns["sin_lon"])], height, width)
    row_variation = float((sin_lat_grid - sin_lat_grid[:, :1]).abs().max().item())
    col_variation = float((sin_lon_grid - sin_lon_grid[:1, :]).abs().max().item())
    passed = max_diff < CHECK_TOL and row_variation < CHECK_TOL and col_variation < CHECK_TOL
    message = "" if passed else "Lat/lon feature ordering failed: node ordering or lat/lon grid alignment may be wrong."
    return {
        **_status(passed, message),
        "max_abs_diff": max_diff,
        "sin_lat_column_variation": row_variation,
        "sin_lon_row_variation": col_variation,
        "rows": details,
    }


def _check_tisr_alignment(
    builder: RolloutFeatureBuilder,
    target_seq: torch.Tensor,
    aux_by_lead: list[torch.Tensor | None],
    tisr_local_idx: int | None,
) -> dict[str, Any]:
    col = _feature_column(builder, "known_tisr")
    if tisr_local_idx is None or col is None:
        return {**_status(None, "known_tisr feature or TISR channel is unavailable"), "leads": []}
    rows = []
    for step, aux in enumerate(aux_by_lead):
        if aux is None:
            return {**_status(False, "TISR alignment failed: auxiliary features were not built."), "leads": rows}
        target_grid = target_seq[step, int(tisr_local_idx)]
        known_grid = _grid_from_nodes(aux[0, :, int(col)], target_grid.shape[-2], target_grid.shape[-1])
        max_diff, mean_diff = _max_mean_diff(known_grid, target_grid)
        rows.append(
            {
                "lead": step + 1,
                "max_abs_diff": max_diff,
                "mean_abs_diff": mean_diff,
                "pass": max_diff < LOOSE_TOL,
            }
        )
    passed = all(row["pass"] for row in rows)
    message = "" if passed else "TISR alignment failed: known forcing at lead k does not match target TISR at lead k. Possible off-by-one error or using input time instead of target time."
    return {**_status(passed, message), "leads": rows}


def _check_dayofyear_alignment(
    builder: RolloutFeatureBuilder,
    metadata: dict[str, Any],
    aux_by_lead: list[torch.Tensor | None],
) -> dict[str, Any]:
    sin_col = _feature_column(builder, "sin_dayofyear")
    cos_col = _feature_column(builder, "cos_dayofyear")
    target_doy = metadata.get("target_dayofyear")
    target_days = metadata.get("target_days_in_year")
    input_doy = metadata.get("input_dayofyear")
    if sin_col is None or cos_col is None:
        return {**_status(None, "day-of-year features are not enabled"), "leads": []}
    if not torch.is_tensor(target_doy) or not torch.is_tensor(target_days):
        return {**_status(False, "target timestamp metadata is missing"), "leads": []}
    rows = []
    for step, aux in enumerate(aux_by_lead):
        if aux is None:
            return {**_status(False, "Day-of-year feature alignment failed: auxiliary features were not built."), "leads": rows}
        doy = target_doy[int(step)].reshape(1)
        days = target_days[int(step)].reshape(1)
        expected = dayofyear_sincos(doy, days, dtype=torch.float32, device=torch.device("cpu"))[0]
        actual = torch.stack([aux[0, 0, int(sin_col)].cpu(), aux[0, 0, int(cos_col)].cpu()]).to(torch.float32)
        max_diff = float((actual - expected).abs().max().item())
        rows.append(
            {
                "lead": step + 1,
                "input_day_of_year": None if not torch.is_tensor(input_doy) else int(input_doy[-1].item()),
                "target_day_of_year": int(doy.item()),
                "days_in_year": int(days.item()),
                "expected_sin": float(expected[0].item()),
                "expected_cos": float(expected[1].item()),
                "actual_sin": float(actual[0].item()),
                "actual_cos": float(actual[1].item()),
                "max_abs_diff": max_diff,
                "pass": max_diff < CHECK_TOL,
            }
        )
    passed = all(row["pass"] for row in rows)
    message = "" if passed else "Day-of-year feature alignment failed: feature may be using input time instead of target time."
    return {**_status(passed, message), "leads": rows, "year_boundary_check": _year_boundary_check(rows)}


def _year_boundary_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for prev, cur in zip(rows, rows[1:]):
        if int(prev["target_day_of_year"]) >= 365 and int(cur["target_day_of_year"]) == 1:
            prev_vec = np.asarray([prev["actual_sin"], prev["actual_cos"]], dtype=np.float64)
            cur_vec = np.asarray([cur["actual_sin"], cur["actual_cos"]], dtype=np.float64)
            return {
                "status": "CHECKED",
                "lead_pair": [prev["lead"], cur["lead"]],
                "vector_distance": float(np.linalg.norm(prev_vec - cur_vec)),
            }
    return {"status": "SKIP", "message": "sample does not cross a year boundary"}


def _check_orography_alignment(
    builder: RolloutFeatureBuilder,
    current: torch.Tensor,
    aux: torch.Tensor | None,
    orog_local_idx: int | None,
) -> dict[str, Any]:
    col = _feature_column(builder, "orography")
    if orog_local_idx is None:
        return {**_status(None, "orography channel is unavailable"), "orography_alignment": None}
    if col is None or aux is None:
        return {**_status(None, "orography auxiliary feature is not enabled"), "orography_alignment": None}
    reference = current[0, int(orog_local_idx)].detach().cpu().numpy()
    aux_grid = _grid_from_nodes(aux[0, :, int(col)], reference.shape[0], reference.shape[1]).detach().cpu().numpy()
    diff = np.abs(aux_grid - reference)
    corr_same = _corr(aux_grid, reference)
    corr_lat = _corr(aux_grid, np.flip(reference, axis=0))
    corr_lon = _corr(aux_grid, np.flip(reference, axis=1))
    corr_both = _corr(aux_grid, np.flip(np.flip(reference, axis=0), axis=1))
    correlations = {
        "corr_same": corr_same,
        "corr_lat_flipped": corr_lat,
        "corr_lon_flipped": corr_lon,
        "corr_both_flipped": corr_both,
        "max_abs_diff": float(np.nanmax(diff)),
        "mean_abs_diff": float(np.nanmean(diff)),
    }
    comparable = {k: v for k, v in correlations.items() if k.startswith("corr_") and v is not None}
    highest = max(comparable, key=lambda key: comparable[key]) if comparable else None
    passed = bool(corr_same is not None and corr_same > 0.999 and highest == "corr_same")
    message = ""
    if not passed:
        message = "Orography orientation check failed or is ambiguous."
        if highest == "corr_lat_flipped":
            message = "Orography may be flipped in latitude."
        elif highest == "corr_lon_flipped":
            message = "Orography may be flipped in longitude."
        elif highest == "corr_both_flipped":
            message = "Orography may be flipped in latitude and longitude."
    return {**_status(passed, message), "orography_alignment": correlations, "highest_correlation": highest}


def _check_lsm(
    builder: RolloutFeatureBuilder,
    current: torch.Tensor,
    aux: torch.Tensor | None,
    lsm_local_idx: int | None,
    requested: bool,
    require_static_features: bool,
) -> dict[str, Any]:
    col = _feature_column(builder, "land_sea_mask")
    if not requested:
        return {**_status(None, "land-sea mask is not requested"), "stats": None}
    if col is None or aux is None:
        message = "Land-sea mask requested but missing. This Priority 2 run is missing one of the most important T2M features."
        if require_static_features:
            return {**_status(False, message), "fatal_if_required": True, "stats": None}
        return {**_status(False, message), "fatal_if_required": False, "stats": None}
    lsm_grid = _grid_from_nodes(aux[0, :, int(col)], int(current.shape[-2]), int(current.shape[-1]))
    result = _tensor_stats(lsm_grid)
    result["shape"] = [int(current.shape[-2]), int(current.shape[-1])]
    if lsm_local_idx is not None:
        ref = current[0, int(lsm_local_idx)].detach().cpu().numpy()
        arr = lsm_grid.detach().cpu().numpy()
        result.update(
            {
                "corr_same": _corr(arr, ref),
                "corr_lat_flipped": _corr(arr, np.flip(ref, axis=0)),
                "corr_lon_flipped": _corr(arr, np.flip(ref, axis=1)),
                "corr_both_flipped": _corr(arr, np.flip(np.flip(ref, axis=0), axis=1)),
            }
        )
    values_ok = result["min"] is not None and result["max"] is not None and float(result["min"]) >= -1.0e-6 and float(result["max"]) <= 1.0 + 1.0e-6
    return {**_status(bool(values_ok), "" if values_ok else "Land-sea mask values are outside expected [0, 1] range."), "stats": result}


def _check_loss_mask(
    builder: RolloutFeatureBuilder,
    height: int,
    width: int,
    output_channels: int,
    t2m_local_idx: int | None,
) -> dict[str, Any]:
    mask = builder.loss_channel_mask(output_channels)
    if mask is None:
        return {**_status(None, "no loss channel mask configured"), "mask": None}
    loss_fn = LatitudeWeightedMSE(torch.zeros(height, dtype=torch.float32))
    target = torch.zeros(1, output_channels, height, width)
    pred = target.clone()
    excluded = sorted((name, int(idx)) for name, idx in builder.exclude_loss_channels.items())
    for _, idx in excluded:
        pred[:, idx] += 100000.0
    excluded_loss = float(loss_fn(pred, target, channel_mask=mask).item())
    included_loss = None
    if t2m_local_idx is not None:
        pred2 = target.clone()
        pred2[:, int(t2m_local_idx)] += 1.0
        included_loss = float(loss_fn(pred2, target, channel_mask=mask).item())
    passed = excluded_loss < CHECK_TOL and included_loss is not None and included_loss > CHECK_TOL
    return {
        **_status(passed, "" if passed else "Loss mask exclusion failed."),
        "excluded_variables": [{"name": name, "local_index": idx} for name, idx in excluded],
        "mask": [float(x) for x in mask.tolist()],
        "masked_loss_with_only_excluded_channel_error": excluded_loss,
        "masked_loss_with_included_t2m_error": included_loss,
    }


def _build_model(
    params: Any,
    graph: Any,
    inp: torch.Tensor,
    output_channels: int,
    aux_feature_dim: int,
    device: torch.device,
) -> GraphWeatherModel:
    lead_cfg = dict(_get(params, "lead_conditioning", {}) or {})
    lead_added = int(lead_cfg.get("added_input_channels", 0))
    model_input_channels = int(inp.shape[0]) + lead_added
    return GraphWeatherModel(
        graph=graph.to(device),
        grid_shape=(int(inp.shape[-2]), int(inp.shape[-1])),
        input_channels=model_input_channels,
        output_channels=int(output_channels),
        n_history=int(params.n_history),
        hidden_dim=int(_get(params, "hidden_dim", 96)),
        edge_dim=int(_get(params, "edge_dim", 6)),
        heads=int(_get(params, "num_heads", 4)),
        k_neighbors=int(_get(params, "k_neighbors", 8)),
        level_k_neighbors=_get(params, "level_k_neighbors", None),
        encoder_blocks=int(_get(params, "encoder_blocks", 1)),
        decoder_blocks=int(_get(params, "decoder_blocks", 1)),
        l0_blocks=int(_get(params, "l0_blocks", 2)),
        l1_blocks=int(_get(params, "l1_blocks", 2)),
        l2_blocks=int(_get(params, "l2_blocks", 1)),
        l1_refine_blocks=int(_get(params, "l1_refine_blocks", 1)),
        l0_refine_blocks=int(_get(params, "l0_refine_blocks", 1)),
        num_graph_levels=int(_get(params, "num_graph_levels", 3)),
        use_l3=bool(_get(params, "use_l3", False)),
        l3_blocks=int(_get(params, "l3_blocks", 1)),
        l4_blocks=int(_get(params, "l4_blocks", 1)),
        l3_refine_after_l4_blocks=int(_get(params, "l3_refine_after_l4_blocks", 1)),
        l2_refine_after_l3_blocks=int(_get(params, "l2_refine_after_l3_blocks", 1)),
        skip_fusion=dict(_get(params, "skip_fusion", {}) or {}),
        pooling=dict(_get(params, "pooling", {}) or {}),
        l0_refine=dict(_get(params, "l0_refine", {}) or {}),
        lead_conditioning=dict(_get(params, "lead_conditioning", {}) or {}),
        aux_feature_dim=int(aux_feature_dim),
    ).to(device)


def _load_checkpoint_if_requested(
    model: GraphWeatherModel,
    checkpoint_path: str | None,
    feature_builder: RolloutFeatureBuilder,
    device: torch.device,
) -> dict[str, Any]:
    if not checkpoint_path:
        return {"loaded": False, "checkpoint_path": None}
    path = str(Path(checkpoint_path).expanduser().resolve())
    checkpoint = torch.load(path, map_location=device)
    metadata = dict(checkpoint.get("metadata", {}))
    validate_checkpoint_architecture(metadata, model)
    active = feature_builder.checkpoint_metadata(
        base_input_channels=int(model.adapter.input_channels),
        total_input_channels=int(model.adapter.input_channels + feature_builder.aux_feature_dim),
    )
    ok, reason = feature_metadata_matches(active, metadata)
    if not ok:
        raise RuntimeError(f"Checkpoint extra-feature configuration mismatch: {reason}")
    state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
    cleaned = {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}
    use_delta = bool(metadata.get("use_delta_normalization", False))
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    allowed_missing = set() if use_delta else {"delta_mean", "delta_std"}
    bad_missing = [key for key in missing if key not in allowed_missing]
    if bad_missing or unexpected:
        raise RuntimeError(f"Checkpoint model_state mismatch. Missing={bad_missing}; unexpected={list(unexpected)}")
    model.use_delta_normalization = use_delta
    model.delta_norm_center = bool(metadata.get("delta_norm_center", False))
    model.eval()
    return {"loaded": True, "checkpoint_path": path, "metadata": metadata}


@torch.no_grad()
def _check_rollout_overrides(
    params: Any,
    graph: Any,
    feature_builder: RolloutFeatureBuilder,
    inp: torch.Tensor,
    target_seq: torch.Tensor,
    metadata: dict[str, Any],
    checkpoint_path: str | None,
    tisr_local_idx: int | None,
    orog_local_idx: int | None,
    device: torch.device,
) -> dict[str, Any]:
    if not feature_builder.enabled:
        return {**_status(None, "extra features disabled"), "leads": []}
    model = _build_model(
        params,
        graph,
        inp,
        output_channels=int(target_seq.shape[1]),
        aux_feature_dim=int(feature_builder.aux_feature_dim),
        device=device,
    )
    checkpoint_info = _load_checkpoint_if_requested(model, checkpoint_path, feature_builder, device)
    inp_device = inp.unsqueeze(0).to(device=device, dtype=torch.float32)
    target_device = target_seq.unsqueeze(0).to(device=device, dtype=torch.float32)
    previous, current = model.adapter.extract_two_steps(inp_device)
    initial_orog = None if orog_local_idx is None else current[:, int(orog_local_idx)].detach().clone()
    rows = []
    for step in range(int(target_seq.shape[0])):
        target_norm = target_device[:, step]
        aux = feature_builder.build_step_features(
            current=current,
            target_norm=target_norm,
            target_dayofyear=_metadata_batch(metadata, "target_dayofyear", device=device),
            target_days_in_year=_metadata_batch(metadata, "target_days_in_year", device=device),
            step_idx=step,
        )
        pred = model.forward_steps(previous, current, aux_features=aux, lead=step + 1)
        pred = feature_builder.apply_overrides(pred, current=current, target_norm=target_norm)
        lead_result: dict[str, Any] = {"lead": step + 1}
        if tisr_local_idx is not None:
            max_diff, mean_diff = _max_mean_diff(pred[:, int(tisr_local_idx)], target_norm[:, int(tisr_local_idx)])
            lead_result["tisr_override_max_abs_diff"] = max_diff
            lead_result["tisr_override_mean_abs_diff"] = mean_diff
            lead_result["tisr_override_pass"] = max_diff < LOOSE_TOL
        if orog_local_idx is not None and initial_orog is not None:
            max_diff, mean_diff = _max_mean_diff(pred[:, int(orog_local_idx)], initial_orog)
            lead_result["orog_copy_max_abs_diff"] = max_diff
            lead_result["orog_copy_mean_abs_diff"] = mean_diff
            lead_result["orog_copy_pass"] = max_diff < LOOSE_TOL
        rows.append(lead_result)
        next_current = current.clone()
        next_current[:, : model.output_channels] = pred
        previous, current = current, next_current
    tisr_ok = all(row.get("tisr_override_pass", True) for row in rows)
    orog_ok = all(row.get("orog_copy_pass", True) for row in rows)
    return {
        "status": "PASS" if tisr_ok and orog_ok else "FAIL",
        "pass": bool(tisr_ok and orog_ok),
        "message": "" if tisr_ok and orog_ok else "TISR override or orography copy failed during rollout.",
        "checkpoint": checkpoint_info,
        "leads": rows,
    }


def _save_arrays(
    output_dir: Path,
    sample_idx: int,
    aux_by_lead: list[torch.Tensor | None],
    builder: RolloutFeatureBuilder,
    target_seq: torch.Tensor,
    tisr_local_idx: int | None,
    current: torch.Tensor,
    orog_local_idx: int | None,
) -> None:
    array_dir = output_dir / "arrays"
    array_dir.mkdir(parents=True, exist_ok=True)
    for step, aux in enumerate(aux_by_lead):
        if aux is not None:
            np.save(array_dir / f"sample{sample_idx:03d}_lead{step + 1:02d}_aux.npy", aux.detach().cpu().numpy())
            col = _feature_column(builder, "known_tisr")
            if col is not None:
                np.save(
                    array_dir / f"sample{sample_idx:03d}_lead{step + 1:02d}_known_tisr_feature.npy",
                    aux[0, :, int(col)].detach().cpu().numpy(),
                )
        if tisr_local_idx is not None:
            np.save(
                array_dir / f"sample{sample_idx:03d}_lead{step + 1:02d}_target_tisr.npy",
                target_seq[step, int(tisr_local_idx)].detach().cpu().numpy(),
            )
    if orog_local_idx is not None:
        np.save(array_dir / f"sample{sample_idx:03d}_current_orog.npy", current[0, int(orog_local_idx)].detach().cpu().numpy())


def _save_debug_plots(
    output_dir: Path,
    builder: RolloutFeatureBuilder,
    aux_by_lead: list[torch.Tensor | None],
    height: int,
    width: int,
) -> None:
    if not aux_by_lead or aux_by_lead[0] is None:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        logging.warning("matplotlib unavailable; skipping debug plots.")
        return
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_specs = [
        ("sin_lat", "aux_sin_lat.png"),
        ("sin_lon", "aux_sin_lon.png"),
        ("sin_dayofyear", "aux_dayofyear_lead1.png"),
        ("orography", "aux_orog.png"),
        ("land_sea_mask", "aux_lsm.png"),
        ("known_tisr", "known_tisr_lead1.png"),
    ]
    last_aux = aux_by_lead[-1]
    if last_aux is not None and _feature_column(builder, "known_tisr") is not None:
        plot_specs.append(("known_tisr_last", f"known_tisr_lead{len(aux_by_lead)}.png"))
    for name, filename in plot_specs:
        source_aux = last_aux if name == "known_tisr_last" else aux_by_lead[0]
        column_name = "known_tisr" if name == "known_tisr_last" else name
        col = _feature_column(builder, column_name)
        if source_aux is None or col is None:
            continue
        grid = _grid_from_nodes(source_aux[0, :, int(col)], height, width).detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(8, 4.2))
        mesh = ax.imshow(grid, origin="upper", aspect="auto")
        ax.set_title(column_name)
        fig.colorbar(mesh, ax=ax)
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=150)
        plt.close(fig)


def _write_text_report(path: Path, report: dict[str, Any]) -> None:
    checks = report["checks"]
    lines = ["Extra feature debug summary:"]
    ordered = [
        ("tisr_lead_alignment", "known_tisr lead k matches target tisr lead k"),
        ("tisr_rollout_override", "TISR rollout override"),
        ("orography_copy_rollout", "Orography copy through rollout"),
        ("loss_mask", "Loss mask excludes orog/tisr"),
        ("dayofyear_target_time", "Day-of-year uses target time"),
        ("latlon_ordering", "Lat/lon feature ordering"),
        ("orography_orientation", "Orography orientation"),
        ("land_sea_mask", "Land-sea mask availability/orientation"),
        ("feature_finiteness", "NaN/Inf feature check"),
    ]
    for key, label in ordered:
        item = checks.get(key, {"status": "SKIP", "message": "not checked"})
        suffix = f" - {item.get('message')}" if item.get("message") else ""
        lines.append(f"[{item.get('status', 'SKIP')}] {label}{suffix}")
    lines.append("")
    lines.append("Resolved variables:")
    for name, item in report["variable_mapping"]["resolved_variables"].items():
        lines.append(
            f"{name:<5} -> channel {item.get('channel')} local_index {item.get('local_index')} "
            f"actual_name={item.get('actual_name')} source={item.get('source')}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_aux_for_sample(
    builder: RolloutFeatureBuilder,
    current: torch.Tensor,
    target_seq: torch.Tensor,
    metadata: dict[str, Any],
) -> list[torch.Tensor | None]:
    aux_by_lead: list[torch.Tensor | None] = []
    for step in range(int(target_seq.shape[0])):
        aux = builder.build_step_features(
            current=current,
            target_norm=target_seq[step : step + 1],
            target_dayofyear=_metadata_batch(metadata, "target_dayofyear"),
            target_days_in_year=_metadata_batch(metadata, "target_days_in_year"),
            step_idx=step,
        )
        aux_by_lead.append(aux)
    return aux_by_lead


def _sample_indices(dataset: ClimateNetCDFDataset, sample_index: int, max_samples: int | None, check_all: bool) -> list[int]:
    if check_all:
        limit = len(dataset) if max_samples is None else min(len(dataset), int(max_samples))
        return list(range(limit))
    count = 1 if max_samples is None else max(1, int(max_samples))
    end = min(len(dataset), int(sample_index) + count)
    return list(range(int(sample_index), end))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Debug Priority 2 extra features: lat/lon, day-of-year, orography, land-sea mask, "
            "known future TISR forcing, rollout overrides, and loss masks."
        )
    )
    parser.add_argument("--config", default=str(project_root / "configs" / "gnn_5p625.yaml"), type=str)
    parser.add_argument("--yaml_config", default=None, type=str)
    parser.add_argument("--config_name", default=None, type=str)
    parser.add_argument("--resolution_mode", default=None, type=str)
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--sample_index", default=0, type=int)
    parser.add_argument("--rollout_steps", default=10, type=int)
    parser.add_argument("--max_samples", default=None, type=int)
    parser.add_argument("--check_all", action="store_true")
    parser.add_argument("--save_arrays", action="store_true")
    parser.add_argument("--save_plots", action="store_true")
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--output_dir", default=str(project_root / "experiments" / "debug_extra_features"), type=str)
    parser.add_argument("--device", default=None, type=str)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(rank=0, log_file=str(output_dir / "debug_extra_features.log"))

    yaml_path, config_name = _resolve_config_args(args)
    params = YParams(yaml_path, config_name, resolution_mode=args.resolution_mode)
    dataset = _build_dataset(params, args.split, int(args.rollout_steps))
    graph = load_graph_bundle(params.graph_path, map_location="cpu")
    settings = ExtraFeatureSettings.from_params(params)
    variable_mapping = _resolve_variables(params, dataset, settings)
    _save_json(output_dir / "variable_mapping.json", variable_mapping)

    if str(_get(params, "normalization", "zscore")).lower() == "zscore":
        output_means, output_stds = dataset.output_normalization_vectors()
    else:
        output_means = None
        output_stds = None
    builder = RolloutFeatureBuilder.from_params(
        params,
        graph=graph,
        channel_names=getattr(dataset, "channel_names", None),
        out_channels=list(params.out_channels),
        output_means=output_means,
        output_stds=output_stds,
        logger=logging,
    )
    builder.log_startup(
        base_input_channels=(int(params.n_history) + 1) * len(list(params.in_channels)),
        total_input_channels=(int(params.n_history) + 1) * len(list(params.in_channels)) + int(builder.aux_feature_dim),
        output_channels=len(list(params.out_channels)),
    )

    indices = _sample_indices(dataset, int(args.sample_index), args.max_samples, bool(args.check_all))
    if not indices:
        raise ValueError("No samples selected for debug.")
    stats_acc = FeatureStatsAccumulator(builder.feature_names)
    adapter: GridNodeAdapter | None = None
    sample_reports = []
    first_aux_by_lead: list[torch.Tensor | None] = []
    first_current: torch.Tensor | None = None
    first_target_seq: torch.Tensor | None = None
    first_metadata: dict[str, Any] | None = None
    first_inp: torch.Tensor | None = None

    for sample_idx in indices:
        inp, target, metadata = _unpack_sample(dataset[sample_idx])
        target_seq = _target_sequence(target)
        adapter = adapter or GridNodeAdapter(
            grid_shape=(int(target_seq.shape[-2]), int(target_seq.shape[-1])),
            input_channels=int(inp.shape[0]),
            output_channels=int(target_seq.shape[1]),
            n_history=int(params.n_history),
        )
        _, current = adapter.extract_two_steps(inp.unsqueeze(0))
        aux_by_lead = _build_aux_for_sample(builder, current, target_seq, metadata)
        for aux in aux_by_lead:
            if aux is not None:
                stats_acc.update(aux)
        sample_reports.append(
            {
                "sample_index": int(sample_idx),
                "center_index": None if not torch.is_tensor(metadata.get("center_index")) else int(metadata["center_index"].item()),
                "target_indices": None
                if not torch.is_tensor(metadata.get("target_indices"))
                else [int(x) for x in metadata["target_indices"].tolist()],
            }
        )
        if sample_idx == indices[0]:
            first_aux_by_lead = aux_by_lead
            first_current = current
            first_target_seq = target_seq
            first_metadata = metadata
            first_inp = inp

    if first_current is None or first_target_seq is None or first_metadata is None or first_inp is None:
        raise RuntimeError("Internal error: no selected sample was loaded.")

    resolved = variable_mapping["resolved_variables"]
    tisr_local = resolved["tisr"]["local_index"]
    orog_local = resolved["orog"]["local_index"]
    lsm_local = resolved["lsm"]["local_index"]
    t2m_local = resolved["t2m"]["local_index"]
    height, width = int(first_target_seq.shape[-2]), int(first_target_seq.shape[-1])
    latitudes, longitudes, latlon_source = _dataset_lat_lon(dataset, graph)

    checks: dict[str, Any] = {}
    checks["tisr_lead_alignment"] = _check_tisr_alignment(builder, first_target_seq, first_aux_by_lead, tisr_local)
    day_check = _check_dayofyear_alignment(builder, first_metadata, first_aux_by_lead)
    checks["dayofyear_target_time"] = day_check
    checks["latlon_ordering"] = _check_lat_lon_ordering(builder, first_aux_by_lead[0] if first_aux_by_lead else None, latitudes, longitudes, height, width)
    checks["orography_orientation"] = _check_orography_alignment(builder, first_current, first_aux_by_lead[0] if first_aux_by_lead else None, orog_local)
    checks["land_sea_mask"] = _check_lsm(
        builder,
        first_current,
        first_aux_by_lead[0] if first_aux_by_lead else None,
        lsm_local,
        requested=bool(settings.land_sea_mask),
        require_static_features=bool(settings.require_static_features),
    )
    checks["loss_mask"] = _check_loss_mask(builder, height, width, int(first_target_seq.shape[1]), t2m_local)
    device = _resolve_device(args.device or "cuda", local_rank=0)
    rollout_check = _check_rollout_overrides(
        params,
        graph,
        builder,
        first_inp,
        first_target_seq,
        first_metadata,
        args.checkpoint,
        tisr_local,
        orog_local,
        device,
    )
    checks["tisr_rollout_override"] = {
        **_status(
            None
            if rollout_check.get("pass") is None
            else all(row.get("tisr_override_pass", True) for row in rollout_check.get("leads", [])),
            rollout_check.get("message", ""),
        ),
        "leads": rollout_check.get("leads", []),
        "checkpoint": rollout_check.get("checkpoint"),
    }
    checks["orography_copy_rollout"] = {
        **_status(
            None
            if rollout_check.get("pass") is None
            else all(row.get("orog_copy_pass", True) for row in rollout_check.get("leads", [])),
            rollout_check.get("message", ""),
        ),
        "leads": rollout_check.get("leads", []),
        "checkpoint": rollout_check.get("checkpoint"),
    }

    feature_rows = stats_acc.rows()
    _write_csv(output_dir / "feature_stats.csv", feature_rows)
    finite_ok = all(int(row["nan_count"]) == 0 and int(row["inf_count"]) == 0 for row in feature_rows)
    checks["feature_finiteness"] = _status(finite_ok, "" if finite_ok else "NaN or Inf found in auxiliary features.")

    if args.save_arrays:
        _save_arrays(output_dir, indices[0], first_aux_by_lead, builder, first_target_seq, tisr_local, first_current, orog_local)
    if args.save_plots:
        _save_debug_plots(output_dir, builder, first_aux_by_lead, height, width)

    report = {
        "config": {"yaml_config": yaml_path, "config_name": config_name, "resolution_mode": str(_get(params, "resolution_mode", ""))},
        "split": args.split,
        "rollout_steps": int(args.rollout_steps),
        "sample_indices": indices,
        "samples": sample_reports,
        "data_path": _split_data_path(params, args.split),
        "grid_shape": [height, width],
        "lat_lon_source": latlon_source,
        "feature_builder": builder.metadata,
        "feature_names": builder.feature_names,
        "variable_mapping": variable_mapping,
        "checks": checks,
        "feature_stats_csv": str(output_dir / "feature_stats.csv"),
    }
    _save_json(output_dir / "debug_extra_features_report.json", report)
    _write_text_report(output_dir / "debug_extra_features_report.txt", report)

    print((output_dir / "debug_extra_features_report.txt").read_text(encoding="utf-8"))
    print(f"Saved JSON report: {output_dir / 'debug_extra_features_report.json'}")
    print(f"Saved variable mapping: {output_dir / 'variable_mapping.json'}")
    print(f"Saved feature stats: {output_dir / 'feature_stats.csv'}")
    if not finite_ok:
        raise RuntimeError("NaN or Inf found in auxiliary features.")


if __name__ == "__main__":
    main()
