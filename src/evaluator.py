from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import time
import calendar
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

DEFAULT_WANDB_ENTITY = "amin1jafarzade-kaist"

from .architecture import validate_checkpoint_architecture
from .affine_calibration import (
    AffineCalibration,
    VALID_CALIBRATION_APPLY_MODES,
    apply_affine_calibration_to_tensor,
    empty_affine_accumulator,
    update_affine_accumulator,
    validate_apply_mode,
)
from .climatology import (
    DayOfYearClimatology,
    build_dayofyear_climatology,
    date_from_time,
    dayofyear,
    decode_time_values,
    find_nc_files,
    format_time,
    load_dayofyear_climatology,
    parse_date,
)
from .data import _add_grid_channels
from .delta_stats import load_delta_stats
from .device import _resolve_device
from .features import RolloutFeatureBuilder, VariableResolver, feature_metadata_matches
from .graph_bundle import load_graph_bundle
from .lead_conditioning import lead_conditioning_debug_values
from .models import GraphWeatherModel
from .resolution import cell_center_lat_lon
from .target_handling import TargetHandling, TargetHandlingSettings, target_handling_metadata_matches
from .weatherbench2_metrics import mean_per_ic_rmse, rmse_from_per_ic_mse


def _get(params: Any, name: str, default: Any = None) -> Any:
    return getattr(params, name, default)


def _load_netcdf4():
    try:
        import netCDF4 as nc
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "netCDF4 is required for evaluation. Install requirements.txt in your environment."
        ) from exc
    return nc


def _stats_1d(path: str) -> np.ndarray:
    arr = np.load(path).squeeze()
    if arr.ndim == 2:
        if arr.shape[0] == 2:
            arr = arr[-1]
        else:
            arr = arr[0]
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D normalization stats after squeeze, got {arr.shape}")
    return arr.astype(np.float32)


def _select_state_stats(stats: np.ndarray, channels: list[int], output_channel_count: int, label: str) -> np.ndarray:
    stats = np.asarray(stats, dtype=np.float32).reshape(-1)
    channels_arr = np.asarray(channels, dtype=np.int64)
    if stats.shape[0] == int(output_channel_count):
        selected = stats[channels_arr] if int(channels_arr.max(initial=-1)) < int(output_channel_count) else stats
    elif stats.shape[0] == 2 * int(output_channel_count) and int(channels_arr.max(initial=-1)) < int(output_channel_count):
        selected = stats[int(output_channel_count) + channels_arr]
        logging.warning(
            "%s stats contain %d entries for %d state channels; using current/output half.",
            label,
            int(stats.shape[0]),
            int(output_channel_count),
        )
    elif int(channels_arr.max(initial=-1)) < stats.shape[0]:
        selected = stats[channels_arr]
    else:
        raise ValueError(
            f"{label} stats length {stats.shape[0]} cannot be aligned with channels={channels[:5]}..."
        )
    if selected.shape[0] != len(channels):
        raise AssertionError(f"{label} selected stats length {selected.shape[0]} does not match {len(channels)} channels.")
    return selected.astype(np.float32)


def _latitude_weights(latitudes_deg: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(latitudes_deg.astype(np.float64))).clip(min=0.0)
    if float(weights.sum()) <= 0.0:
        weights = np.ones_like(weights, dtype=np.float64)
    return weights


VARIABLE_ALIASES: dict[str, list[str]] = {
    "msl": ["msl", "mslp", "mean_sea_level_pressure", "mean_sea_level_pressure_surface"],
    "t2m": ["t2m", "2m_temperature", "temperature_2m"],
    "t850": ["t850", "t_850", "temperature_850", "t@850"],
    "z500": ["z500", "z_500", "z@500", "geopotential_500"],
    "tisr": ["tisr", "top_incoming_solar_radiation"],
    "orog": ["orog", "orography", "geopotential_at_surface", "surface_geopotential"],
    "lsm": ["lsm", "land_sea_mask", "land_mask"],
    "q700": ["q700", "q_700", "specific_humidity_700"],
    "u850": ["u850", "u_850", "u_component_wind_850"],
}


SENSITIVITY_MODES: tuple[str, ...] = (
    "normal",
    "override_orog_tisr",
    "tisr_zero",
    "tisr_random",
    "tisr_shuffle",
    "orog_zero",
    "orog_random",
    "orog_shuffle",
    "orog_tisr_zero",
    "orog_tisr_random",
    "orog_tisr_shuffle",
)


SENSITIVITY_MODE_VARIABLES: dict[str, tuple[str, ...]] = {
    "normal": (),
    "override_orog_tisr": ("orog", "tisr"),
    "tisr_zero": ("tisr",),
    "tisr_random": ("tisr",),
    "tisr_shuffle": ("tisr",),
    "orog_zero": ("orog",),
    "orog_random": ("orog",),
    "orog_shuffle": ("orog",),
    "orog_tisr_zero": ("orog", "tisr"),
    "orog_tisr_random": ("orog", "tisr"),
    "orog_tisr_shuffle": ("orog", "tisr"),
}


def _norm_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _canonical_for_name(name: str) -> str | None:
    norm = _norm_name(name)
    for canonical, aliases in VARIABLE_ALIASES.items():
        if norm == _norm_name(canonical) or norm in {_norm_name(alias) for alias in aliases}:
            return canonical
    return None


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        values = value
    else:
        values = [value]
    parsed: list[str] = []
    for item in values:
        parsed.extend(chunk.strip() for chunk in str(item).replace(",", " ").split() if chunk.strip())
    return parsed


def _as_raw_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _as_optional_float_list(value: Any) -> list[float | None]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        values = value
    else:
        values = [value]
    parsed: list[float | None] = []
    for item in values:
        text = str(item).strip()
        if not text:
            parsed.append(None)
        else:
            parsed.append(float(text))
    return parsed


def _json_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _json_series(values: np.ndarray) -> list[float | None]:
    return [_json_number(value) for value in np.asarray(values, dtype=np.float64).reshape(-1)]


@dataclass
class ExternalBaseline:
    label: str
    params_m: float | None
    source_csv: str
    curves: dict[str, dict[str, np.ndarray]]
    raw_variables: dict[str, list[str]]


def load_external_baselines(
    csv_paths: list[str],
    labels: list[str] | None,
    params_m: list[float | None] | None,
    fixed_steps: int,
    logger: logging.Logger | Any = logging,
) -> list[ExternalBaseline]:
    if not csv_paths:
        return []
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pandas is required for --external_baseline_csv. Install pandas or remove the external baseline option."
        ) from exc

    baselines: list[ExternalBaseline] = []
    labels = labels or []
    params_m = params_m or []
    required = {"variable", "rmse", "acc"}
    optional_ci = {
        "rmse_lower": "rmse_ci_lower",
        "rmse_upper": "rmse_ci_upper",
        "acc_lower": "acc_ci_lower",
        "acc_upper": "acc_ci_upper",
    }
    for idx, csv_path in enumerate(csv_paths):
        path = str(Path(csv_path).expanduser())
        if not os.path.exists(path):
            raise FileNotFoundError(f"external_baseline_csv does not exist: {path}")
        label = labels[idx] if idx < len(labels) else Path(path).stem
        params = params_m[idx] if idx < len(params_m) else None
        df = pd.read_csv(path)
        columns = set(df.columns)
        missing = sorted(required - columns)
        if missing:
            raise ValueError(f"External baseline CSV {path} is missing required columns: {missing}")
        if "timestep" not in columns:
            if "lead_time" not in columns:
                raise ValueError(
                    f"External baseline CSV {path} is missing required lead column: expected 'timestep' or 'lead_time'."
                )
            df = df.rename(columns={"lead_time": "timestep"})
        df = df.copy()
        df["timestep"] = pd.to_numeric(df["timestep"], errors="coerce")
        df["rmse"] = pd.to_numeric(df["rmse"], errors="coerce")
        df["acc"] = pd.to_numeric(df["acc"], errors="coerce")
        for column in optional_ci:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")

        if (df["timestep"] == 0).any():
            logger.warning("External baseline %s contains lead day 0; ignoring those rows.", label)
        if (df["timestep"] > int(fixed_steps)).any():
            logger.warning("External baseline %s contains lead days beyond %d; ignoring extra leads.", label, fixed_steps)
        df = df[(df["timestep"] >= 1) & (df["timestep"] <= int(fixed_steps))]
        df = df[np.isfinite(df["timestep"]) & np.isfinite(df["rmse"]) & np.isfinite(df["acc"])]

        curves: dict[str, dict[str, np.ndarray]] = {}
        raw_variables: dict[str, list[str]] = {}
        for raw_variable, group in df.groupby("variable", dropna=True):
            raw_name = str(raw_variable).strip()
            canonical = _canonical_for_name(raw_name) or raw_name
            raw_variables.setdefault(canonical, [])
            if raw_name not in raw_variables[canonical]:
                raw_variables[canonical].append(raw_name)
            if canonical not in curves:
                curves[canonical] = {
                    "rmse": np.full((int(fixed_steps),), np.nan, dtype=np.float64),
                    "acc": np.full((int(fixed_steps),), np.nan, dtype=np.float64),
                }
                for src_col, dst_key in optional_ci.items():
                    if src_col in df.columns:
                        curves[canonical][dst_key] = np.full((int(fixed_steps),), np.nan, dtype=np.float64)
            for _, row in group.iterrows():
                lead = int(row["timestep"])
                if lead < 1 or lead > int(fixed_steps):
                    continue
                pos = lead - 1
                curves[canonical]["rmse"][pos] = float(row["rmse"])
                curves[canonical]["acc"][pos] = float(row["acc"])
                for src_col, dst_key in optional_ci.items():
                    if src_col in group.columns and dst_key in curves[canonical]:
                        value = row[src_col]
                        curves[canonical][dst_key][pos] = float(value) if np.isfinite(value) else np.nan

        for variable, curve in curves.items():
            missing_leads = [
                lead
                for lead in range(1, int(fixed_steps) + 1)
                if not np.isfinite(curve["rmse"][lead - 1]) or not np.isfinite(curve["acc"][lead - 1])
            ]
            if missing_leads:
                logger.warning(
                    "External baseline %s variable %s is missing lead days %s; missing points will be skipped in plots.",
                    label,
                    variable,
                    missing_leads,
                )
        baselines.append(
            ExternalBaseline(
                label=str(label),
                params_m=params,
                source_csv=path,
                curves=curves,
                raw_variables=raw_variables,
            )
        )
        logger.info("Loaded external baseline %s from %s with variables: %s", label, path, sorted(curves))
    return baselines


EXTERNAL_BASELINE_FAIRNESS_NOTE = (
    "External baseline comparison assumes same dataset, variables, units, lead times, RMSE/ACC definitions, "
    "climatology, and test initializations. If these differ, the comparison is only approximate."
)


def _accumulate_daily_climatology_chunk(
    clim_sum: np.ndarray,
    clim_count: np.ndarray,
    chunk: np.ndarray,
    start_time_idx: int,
    n_days: int,
) -> None:
    """Accumulate a contiguous time chunk into daily climatology sums."""
    if chunk.shape[0] == 0:
        return
    days = np.arange(start_time_idx, start_time_idx + chunk.shape[0], dtype=np.int64) % int(n_days)
    if np.unique(days).size == days.size:
        clim_sum[days] += chunk
        clim_count[days] += 1
    else:
        np.add.at(clim_sum, days, chunk)
        np.add.at(clim_count, days, 1)


def weighted_acc_per_channel(
    pred_anom: torch.Tensor,
    target_anom: torch.Tensor,
    lat_weights: torch.Tensor,
) -> torch.Tensor:
    """Latitude-weighted ACC for [B,C,H,W] anomaly tensors, returned as [B,C]."""
    weight = lat_weights.to(device=pred_anom.device, dtype=pred_anom.dtype).view(1, 1, -1, 1)
    xy = (pred_anom * target_anom * weight).reshape(pred_anom.shape[0], pred_anom.shape[1], -1).sum(dim=-1)
    xx = (pred_anom * pred_anom * weight).reshape(pred_anom.shape[0], pred_anom.shape[1], -1).sum(dim=-1)
    yy = (target_anom * target_anom * weight).reshape(target_anom.shape[0], target_anom.shape[1], -1).sum(dim=-1)
    return xy / torch.sqrt(xx * yy + 1.0e-8)


@dataclass
class EvalConfig:
    resolution_mode: str
    grid_shape: tuple[int, int]
    test_dataset_path: str
    output_dir: str
    checkpoint_path: str
    experiment_dir: str
    graph_path: str
    graph_connectivity_strategy: str
    graph_format_version: int | str
    hierarchy_type: str
    use_l4_ratio15: bool
    k_neighbors: int
    input_channels: int
    output_channels: int
    level_k_neighbors: list[int]
    level_shapes: list[list[int]]
    node_counts: list[int]
    edge_counts: list[int]
    train_data_path: str
    global_means_path: str
    global_stds_path: str
    in_channels: list[int]
    out_channels: list[int]
    n_history: int
    dt: int
    forecast_steps: list[int]
    split: str
    selection: str
    eval_start_timestep: int
    n_initial_conditions: int
    max_initial_conditions: Optional[int]
    ic_stride: int
    start_offset: int
    start_date: Optional[str]
    end_date: Optional[str]
    climatology_path: Optional[str]
    compute_climatology: bool
    build_climatology_if_missing: bool
    climatology_chunk_size: int
    device: str
    add_grid: bool
    gridtype: str
    n_grid_channels: int
    orography: bool
    orography_path: Optional[str]
    hidden_dim: int
    edge_dim: int
    num_heads: int
    encoder_blocks: int
    decoder_blocks: int
    l0_blocks: int
    l1_blocks: int
    l2_blocks: int
    num_graph_levels: int
    use_l3: bool
    l3_blocks: int
    l4_blocks: int
    l3_refine_after_l4_blocks: int
    l2_refine_after_l3_blocks: int
    l1_refine_blocks: int
    l0_refine_blocks: int
    skip_fusion: dict[str, Any]
    pooling: dict[str, Any]
    l0_refine: dict[str, Any]
    lead_conditioning: dict[str, Any]
    plot_variables: list[str]
    plot_format: str
    eval_fixed_rollout_steps: int
    eval_compare_stage_checkpoints: bool
    eval_stage_checkpoints: dict[int, str]
    eval_global_best_checkpoint: Optional[str]
    plot_persistence: bool
    bootstrap_samples: int
    confidence_level: float
    bootstrap_seed: int
    plot_confidence_intervals: bool
    external_baseline_csv: list[str]
    external_baseline_label: list[str]
    external_baseline_params_m: list[float | None]
    rmse_backend: str
    save_lead0: bool
    include_lead0_in_summary: bool
    first_test_file_only: bool
    extra_features: dict[str, Any]
    target_handling: dict[str, Any]
    eval_target_override: bool
    eval_copy_variables: list[str]
    eval_known_future_variables: list[str]
    eval_exclude_loss_variables: list[str]
    debug_eval_target_override: bool
    eval_sensitivity_mode: str
    eval_sensitivity_noise_seed: int
    debug_eval_sensitivity: bool
    eval_sensitivity_debug_samples: int
    affine_calibration_path: Optional[str]
    calibration_apply_mode: str
    static_fields: dict[str, Any]
    variable_metadata: dict[str, Any]
    require_static_features: bool
    use_delta_normalization: bool
    delta_stats_path: Optional[str]
    delta_norm_center: bool
    delta_norm_eps: float
    wandb: dict[str, Any]

    @classmethod
    def from_params(cls, params: Any) -> "EvalConfig":
        forecast_steps = _get(params, "eval_forecast_steps", None)
        if forecast_steps is None:
            forecast_steps = _get(params, "eval_rollout_steps", [_get(params, "max_rollout_steps", 10)])
        if isinstance(forecast_steps, int):
            forecast_steps = [forecast_steps]
        forecast_steps = sorted({int(x) for x in forecast_steps if int(x) > 0})
        fixed_rollout_steps = _get(params, "eval_fixed_rollout_steps", None)
        if fixed_rollout_steps is None:
            fixed_rollout_steps = max(forecast_steps) if forecast_steps else _get(params, "max_rollout_steps", 10)

        experiment_dir = str(_get(params, "experiment_dir", _get(params, "exp_dir", ".")))
        output_dir = _get(params, "eval_output_dir", None) or os.path.join(
            experiment_dir,
            "evaluation",
        )
        test_path = _get(params, "test_dataset_path", None) or _get(params, "valid_data_path", "")
        plot_variables = _get(params, "eval_plot_variables", [])
        if plot_variables is None:
            plot_variables = []
        elif isinstance(plot_variables, str):
            plot_variables = [x for x in plot_variables.replace(",", " ").split() if x]
        else:
            plot_variables = [str(x) for x in plot_variables]
        stage_checkpoints = _get(params, "eval_stage_checkpoints", {})
        if stage_checkpoints is None:
            stage_checkpoints = {}
        stage_checkpoints = {int(k): str(v) for k, v in dict(stage_checkpoints).items()}
        compare_stage_checkpoints = bool(_get(params, "eval_compare_stage_checkpoints", False))
        global_best_checkpoint = _get(params, "eval_global_best_checkpoint", None)
        checkpoint = _get(params, "eval_checkpoint_path", None) or _get(params, "checkpoint_path", "")
        if not checkpoint and not compare_stage_checkpoints and global_best_checkpoint:
            checkpoint = str(global_best_checkpoint)
        eval_target_override = bool(_get(params, "eval_target_override", False))
        raw_eval_copy_variables = _get(params, "eval_copy_variables", None)
        raw_eval_known_future_variables = _get(params, "eval_known_future_variables", None)
        raw_eval_exclude_loss_variables = _get(params, "eval_exclude_loss_variables", None)
        eval_copy_variables = _as_string_list(raw_eval_copy_variables)
        eval_known_future_variables = _as_string_list(raw_eval_known_future_variables)
        eval_exclude_loss_variables = _as_string_list(raw_eval_exclude_loss_variables)
        if eval_target_override:
            if raw_eval_copy_variables is None and not eval_copy_variables:
                eval_copy_variables = ["orog"]
            if raw_eval_known_future_variables is None and not eval_known_future_variables:
                eval_known_future_variables = ["tisr"]
            if raw_eval_exclude_loss_variables is None and not eval_exclude_loss_variables:
                eval_exclude_loss_variables = [*eval_copy_variables, *eval_known_future_variables]
        eval_sensitivity_mode = str(_get(params, "eval_sensitivity_mode", "normal")).strip().lower()
        if eval_sensitivity_mode not in SENSITIVITY_MODES:
            raise ValueError(
                f"Unsupported eval_sensitivity_mode={eval_sensitivity_mode!r}. "
                f"Expected one of: {', '.join(SENSITIVITY_MODES)}"
            )

        return cls(
            resolution_mode=str(_get(params, "resolution_mode", "5p625")),
            grid_shape=tuple(int(x) for x in _get(params, "grid_shape", [32, 64])),
            test_dataset_path=str(test_path),
            output_dir=str(output_dir),
            checkpoint_path=str(checkpoint),
            experiment_dir=experiment_dir,
            graph_path=str(_get(params, "graph_path", "")),
            graph_connectivity_strategy=str(_get(params, "graph_connectivity_strategy", "hybrid_row_aware_knn")),
            graph_format_version=_get(params, "graph_format_version", 0),
            hierarchy_type=str(_get(params, "hierarchy_type", "standard")),
            use_l4_ratio15=bool(_get(params, "use_l4_ratio15", False)),
            k_neighbors=int(_get(params, "k_neighbors", 8)),
            input_channels=int(_get(params, "input_channels", 2 * len(_get(params, "in_channels", [])))),
            output_channels=int(_get(params, "output_channels", len(_get(params, "out_channels", [])))),
            level_k_neighbors=[int(x) for x in _get(params, "level_k_neighbors", [])],
            level_shapes=[[int(v) for v in shape] for shape in _get(params, "level_shapes", [])],
            node_counts=[int(x) for x in _get(params, "node_counts", [])],
            edge_counts=[int(x) for x in _get(params, "edge_counts", [])],
            train_data_path=str(_get(params, "train_data_path", "")),
            global_means_path=str(_get(params, "global_means_path", "")),
            global_stds_path=str(_get(params, "global_stds_path", "")),
            in_channels=[int(x) for x in _get(params, "in_channels", [])],
            out_channels=[int(x) for x in _get(params, "out_channels", [])],
            n_history=int(_get(params, "n_history", 1)),
            dt=int(_get(params, "dt", 1)),
            forecast_steps=forecast_steps,
            split=str(_get(params, "eval_split", _get(params, "split", "test"))),
            selection=str(_get(params, "eval_selection", _get(params, "selection", "first_n"))),
            eval_start_timestep=int(_get(params, "eval_start_timestep", 1)),
            n_initial_conditions=int(_get(params, "n_initial_conditions", _get(params, "eval_n_initial_conditions", 1))),
            max_initial_conditions=(
                None
                if _get(params, "max_initial_conditions", None) is None
                else int(_get(params, "max_initial_conditions"))
            ),
            ic_stride=int(_get(params, "eval_ic_stride", 1)),
            start_offset=int(_get(params, "eval_start_offset", _get(params, "start_offset", 0))),
            start_date=_get(params, "eval_start_date", _get(params, "start_date", None)),
            end_date=_get(params, "eval_end_date", _get(params, "end_date", None)),
            climatology_path=_get(params, "climatology_path", None),
            compute_climatology=bool(_get(params, "compute_climatology", True)),
            build_climatology_if_missing=bool(_get(params, "build_climatology_if_missing", False)),
            climatology_chunk_size=int(_get(params, "eval_climatology_chunk_size", _get(params, "climatology_chunk_size", 128))),
            device=str(_get(params, "eval_device", "cuda" if torch.cuda.is_available() else "cpu")),
            add_grid=bool(_get(params, "add_grid", False)),
            gridtype=str(_get(params, "gridtype", "linear")),
            n_grid_channels=int(_get(params, "N_grid_channels", 0)),
            orography=bool(_get(params, "orography", False)),
            orography_path=_get(params, "orography_path", None),
            skip_fusion=dict(_get(params, "skip_fusion", {}) or {}),
            pooling=dict(_get(params, "pooling", {}) or {}),
            l0_refine=dict(_get(params, "l0_refine", {}) or {}),
            lead_conditioning=dict(_get(params, "lead_conditioning", {}) or {}),
            hidden_dim=int(_get(params, "hidden_dim", 96)),
            edge_dim=int(_get(params, "edge_dim", 6)),
            num_heads=int(_get(params, "num_heads", 4)),
            encoder_blocks=int(_get(params, "encoder_blocks", 1)),
            decoder_blocks=int(_get(params, "decoder_blocks", 1)),
            l0_blocks=int(_get(params, "l0_blocks", 2)),
            l1_blocks=int(_get(params, "l1_blocks", 2)),
            l2_blocks=int(_get(params, "l2_blocks", 1)),
            num_graph_levels=int(_get(params, "num_graph_levels", 3)),
            use_l3=bool(_get(params, "use_l3", False)),
            l3_blocks=int(_get(params, "l3_blocks", 1)),
            l4_blocks=int(_get(params, "l4_blocks", 1)),
            l3_refine_after_l4_blocks=int(_get(params, "l3_refine_after_l4_blocks", 1)),
            l2_refine_after_l3_blocks=int(_get(params, "l2_refine_after_l3_blocks", 1)),
            l1_refine_blocks=int(_get(params, "l1_refine_blocks", 1)),
            l0_refine_blocks=int(_get(params, "l0_refine_blocks", 1)),
            plot_variables=plot_variables,
            plot_format=str(_get(params, "eval_plot_format", "png")),
            eval_fixed_rollout_steps=int(fixed_rollout_steps),
            eval_compare_stage_checkpoints=compare_stage_checkpoints,
            eval_stage_checkpoints=stage_checkpoints,
            eval_global_best_checkpoint=global_best_checkpoint,
            plot_persistence=bool(_get(params, "plot_persistence", True)),
            bootstrap_samples=int(_get(params, "bootstrap_samples", 0)),
            confidence_level=float(_get(params, "confidence_level", 0.95)),
            bootstrap_seed=int(_get(params, "bootstrap_seed", 42)),
            plot_confidence_intervals=bool(_get(params, "plot_confidence_intervals", False)),
            external_baseline_csv=_as_raw_string_list(_get(params, "external_baseline_csv", [])),
            external_baseline_label=_as_raw_string_list(_get(params, "external_baseline_label", [])),
            external_baseline_params_m=_as_optional_float_list(_get(params, "external_baseline_params_m", [])),
            rmse_backend=str(_get(params, "rmse_backend", "current")).strip().lower(),
            save_lead0=bool(_get(params, "save_lead0", False)),
            include_lead0_in_summary=bool(_get(params, "include_lead0_in_summary", False)),
            first_test_file_only=bool(_get(params, "first_test_file_only", True)),
            extra_features=dict(_get(params, "extra_features", {}) or {}),
            target_handling=dict(_get(params, "target_handling", {}) or {}),
            eval_target_override=eval_target_override,
            eval_copy_variables=eval_copy_variables,
            eval_known_future_variables=eval_known_future_variables,
            eval_exclude_loss_variables=eval_exclude_loss_variables,
            debug_eval_target_override=bool(_get(params, "debug_eval_target_override", False)),
            eval_sensitivity_mode=eval_sensitivity_mode,
            eval_sensitivity_noise_seed=int(_get(params, "eval_sensitivity_noise_seed", 123)),
            debug_eval_sensitivity=bool(_get(params, "debug_eval_sensitivity", False)),
            eval_sensitivity_debug_samples=int(_get(params, "eval_sensitivity_debug_samples", 3)),
            affine_calibration_path=_get(params, "affine_calibration_path", None),
            calibration_apply_mode=validate_apply_mode(_get(params, "calibration_apply_mode", "output_only")),
            static_fields=dict(_get(params, "static_fields", {}) or {}),
            variable_metadata=dict(_get(params, "variable_metadata", {}) or {}),
            require_static_features=bool(_get(params, "require_static_features", False)),
            use_delta_normalization=bool(_get(params, "use_delta_normalization", False)),
            delta_stats_path=_get(params, "delta_stats_path", None),
            delta_norm_center=bool(_get(params, "delta_norm_center", False)),
            delta_norm_eps=float(_get(params, "delta_norm_eps", 1.0e-6)),
            wandb=dict(_get(params, "wandb", {}) or {}),
        )

    def validate(self) -> None:
        if not self.test_dataset_path:
            raise ValueError("test_dataset_path is required for evaluation.")
        if not self.checkpoint_path and not self.eval_compare_stage_checkpoints:
            raise ValueError("eval_checkpoint_path or checkpoint_path is required for evaluation.")
        for path_name in ["test_dataset_path", "graph_path", "global_means_path", "global_stds_path"]:
            value = getattr(self, path_name)
            if not os.path.exists(value):
                raise FileNotFoundError(f"{path_name} does not exist: {value}")
        if self.checkpoint_path and not (
            os.path.exists(self.checkpoint_path)
            or os.path.exists(os.path.join(self.experiment_dir, self.checkpoint_path))
        ):
            raise FileNotFoundError(f"checkpoint_path does not exist: {self.checkpoint_path}")
        selection = str(self.selection).lower()
        if selection not in {"first_n", "stride", "all"}:
            raise ValueError("selection must be one of: first_n, stride, all.")
        if self.ic_stride <= 0:
            raise ValueError("stride/eval_ic_stride must be positive.")
        if self.start_offset < 0:
            raise ValueError("start_offset must be non-negative.")
        if self.max_initial_conditions is not None and self.max_initial_conditions <= 0:
            raise ValueError("max_initial_conditions must be positive when provided.")
        if self.bootstrap_samples < 0:
            raise ValueError("bootstrap_samples must be >= 0.")
        if not (0.0 < self.confidence_level < 1.0):
            raise ValueError("confidence_level must be in (0, 1).")
        if self.external_baseline_label and len(self.external_baseline_label) != len(self.external_baseline_csv):
            raise ValueError("external_baseline_label must have the same count as external_baseline_csv.")
        if self.external_baseline_params_m and len(self.external_baseline_params_m) != len(self.external_baseline_csv):
            raise ValueError("external_baseline_params_m must have the same count as external_baseline_csv.")
        if self.rmse_backend not in {"current", "weatherbench2", "both"}:
            raise ValueError("rmse_backend must be one of: current, weatherbench2, both.")
        for path in self.external_baseline_csv:
            if not os.path.exists(str(Path(path).expanduser())):
                raise FileNotFoundError(f"external_baseline_csv does not exist: {path}")
        if (self.compute_climatology or self.build_climatology_if_missing) and not self.train_data_path:
            raise ValueError("train_data_path is required when climatology may be built.")
        if self.climatology_path and not os.path.exists(str(self.climatology_path)):
            if not (self.compute_climatology or self.build_climatology_if_missing):
                raise FileNotFoundError(
                    f"climatology_path does not exist: {self.climatology_path}. "
                    "Pass --build_climatology_if_missing to build it, or provide a valid calendar-aware climatology."
                )
        if not self.climatology_path and not (self.compute_climatology or self.build_climatology_if_missing):
            raise ValueError(
                "A calendar-aware climatology is required for ACC. Provide --climatology_path "
                "or pass --build_climatology_if_missing."
            )
        if self.climatology_chunk_size <= 0:
            raise ValueError("climatology_chunk_size must be positive.")
        if self.n_history != 1:
            raise NotImplementedError("Evaluation currently supports n_history=1, matching the current GraphWeather setup.")
        if self.affine_calibration_path and not os.path.exists(str(Path(self.affine_calibration_path).expanduser())):
            raise FileNotFoundError(f"affine_calibration_path does not exist: {self.affine_calibration_path}")
        if self.calibration_apply_mode not in VALID_CALIBRATION_APPLY_MODES:
            raise ValueError(
                f"calibration_apply_mode must be one of: {', '.join(VALID_CALIBRATION_APPLY_MODES)}"
            )


class GraphWeatherEvaluator:
    def __init__(self, cfg: EvalConfig, logger: logging.Logger | Any = logging):
        cfg.validate()
        self.cfg = cfg
        self.logger = logger
        self.wandb_run = self._init_wandb_run()
        self.nc = _load_netcdf4()
        self.device = _resolve_device(cfg.device, local_rank=0)
        self.means_all = _stats_1d(cfg.global_means_path)
        self.stds_all = _stats_1d(cfg.global_stds_path)
        required_channel = max(cfg.in_channels + cfg.out_channels) if (cfg.in_channels or cfg.out_channels) else -1
        if required_channel >= len(self.means_all) or required_channel >= len(self.stds_all):
            raise ValueError(
                f"Normalization stats do not contain channel {required_channel}. "
                f"Check {cfg.resolution_mode} statistics paths."
            )
        output_count = len(cfg.out_channels)
        if output_count != 67:
            raise AssertionError(f"Expected 67 output channels, got {output_count}")
        self.in_means = _select_state_stats(self.means_all, cfg.in_channels, output_count, "input mean")
        self.in_stds = _select_state_stats(self.stds_all, cfg.in_channels, output_count, "input std")
        self.out_means = _select_state_stats(self.means_all, cfg.out_channels, output_count, "output mean")
        self.out_stds = _select_state_stats(self.stds_all, cfg.out_channels, output_count, "output std")
        if self.out_means.shape != (67,) or self.out_stds.shape != (67,):
            raise AssertionError(f"Output normalization stats must have shape [67], got {self.out_means.shape}/{self.out_stds.shape}")
        self.model: Optional[GraphWeatherModel] = None
        self.variable_names: list[str] = []
        self.channel_names: list[str] = []
        self.orography_field: Optional[np.ndarray] = None
        self._time_values: list[Any] = []
        self._selected_ics: list[int] | None = None
        self._selection_metadata: dict[str, Any] = {}
        self._climatology_metadata: dict[str, Any] = {}
        self._climatology_available_days: set[int] = set()
        self._climatology_norm_by_day: dict[int, torch.Tensor] = {}
        self._climatology_warned_days: set[int] = set()
        self._climatology_time_fallback_warned = False
        self._graphweather_params_m: float | None = None
        self.feature_builder: RolloutFeatureBuilder | None = None
        self.target_handler: TargetHandling | None = None
        self.eval_target_override_handler: TargetHandling | None = None
        self.eval_loss_channel_mask: torch.Tensor | None = None
        self.affine_calibration: AffineCalibration | None = None
        self._eval_target_override_debug: dict[str, Any] | None = None
        self.sensitivity_channels: dict[str, int] = {}
        self._sensitivity_debug: dict[str, Any] | None = None
        self._sensitivity_logged_fallbacks: set[str] = set()
        self.external_baselines = load_external_baselines(
            cfg.external_baseline_csv,
            cfg.external_baseline_label,
            cfg.external_baseline_params_m,
            fixed_steps=int(cfg.eval_fixed_rollout_steps),
            logger=self.logger,
        )
        if self.external_baselines:
            self.logger.warning(EXTERNAL_BASELINE_FAIRNESS_NOTE)
            print(EXTERNAL_BASELINE_FAIRNESS_NOTE)
        if cfg.orography:
            if not cfg.orography_path:
                raise ValueError("orography_path is required when orography=true.")
            with self.nc.Dataset(cfg.orography_path, "r") as ds:
                key = "orog" if "orog" in ds.variables else next(iter(ds.variables))
                self.orography_field = np.asarray(ds[key][:], dtype=np.float32).squeeze()

    def _init_wandb_run(self) -> Any:
        cfg = dict(getattr(self.cfg, "wandb", {}) or {})
        if not bool(cfg.get("enabled", False)):
            return None
        if wandb is None:
            self.logger.warning("W&B evaluation logging requested, but wandb is not installed. Continuing with local logs only.")
            return None
        if getattr(wandb, "run", None) is not None:
            run = wandb.run
        else:
            project = cfg.get("project") or os.environ.get("WANDB_PROJECT")
            entity = cfg.get("entity") or os.environ.get("WANDB_ENTITY") or DEFAULT_WANDB_ENTITY
            if not project:
                self.logger.warning("W&B evaluation logging requested, but no wandb.project or WANDB_PROJECT is set. Continuing with local logs only.")
                return None
            try:
                run = wandb.init(
                    project=project,
                    entity=entity,
                    name=cfg.get("run_name"),
                    tags=list(cfg.get("tags", []) or []),
                    config={
                        "eval.fixed_rollout_steps": int(self.cfg.eval_fixed_rollout_steps),
                        "eval.rmse_backend": str(self.cfg.rmse_backend),
                        "eval.selection": str(self.cfg.selection),
                        "eval.split": str(self.cfg.split),
                        "model.hidden_dim": int(self.cfg.hidden_dim),
                        "graph.level_k_neighbors": list(self.cfg.level_k_neighbors),
                    },
                )
            except Exception as exc:  # pragma: no cover - depends on external W&B state
                self.logger.warning("W&B evaluation initialization failed; continuing with local logs only: %s", exc)
                return None
        return run

    def _wandb_log(self, payload: dict[str, Any], *, step: int | None = None) -> None:
        run = getattr(self, "wandb_run", None)
        if run is None or not payload:
            return
        try:
            run.log(payload, step=step)
        except Exception as exc:  # pragma: no cover - depends on external W&B state
            self.logger.warning("W&B evaluation logging failed: %s", exc)

    @staticmethod
    def _wandb_table(rows: list[dict[str, Any]], columns: list[str]) -> Any:
        if wandb is None or not rows:
            return None
        return wandb.Table(columns=columns, data=[[row.get(column) for column in columns] for row in rows])

    def _find_nc_files(self, path: str) -> list[str]:
        return find_nc_files(path)

    def _load_variable_names(self, ds: Any) -> list[str]:
        if "channel" in ds.variables:
            raw = ds.variables["channel"][:]
            names = [str(x) for x in raw]
        else:
            names = [f"Var{i}" for i in range(max(self.cfg.out_channels) + 1)]
        self.channel_names = list(names)
        self.variable_names = []
        for idx in self.cfg.out_channels:
            self.variable_names.append(names[idx] if idx < len(names) else f"Var{idx}")
        return self.variable_names

    def _resolve_output_variables(
        self,
        requested: list[str],
        *,
        calibrate_all_dynamic_variables: bool = False,
    ) -> list[dict[str, Any]]:
        if calibrate_all_dynamic_variables:
            variables = []
            for local_idx, name in enumerate(self.variable_names):
                key = _canonical_for_name(name) or str(name)
                if key == "orog":
                    continue
                variables.append(
                    {
                        "name": str(name),
                        "canonical_name": key,
                        "variable_idx": int(local_idx),
                        "channel": int(self.cfg.out_channels[local_idx]),
                    }
                )
            return variables

        resolver = VariableResolver(
            self.cfg,
            self.channel_names or None,
            self.cfg.out_channels,
            logger=self.logger,
        )
        variables: list[dict[str, Any]] = []
        seen: set[int] = set()
        for variable in requested:
            item = resolver.resolve(variable, required=True)
            if item.local_index is None:
                raise ValueError(f"Variable {variable!r} resolved to channel {item.channel}, which is not in out_channels.")
            if item.canonical == "orog":
                self.logger.info("Skipping orog for affine calibration; fixed/static fields are never calibrated.")
                continue
            local_idx = int(item.local_index)
            if local_idx in seen:
                continue
            seen.add(local_idx)
            variables.append(
                {
                    "name": str(item.actual_name or self.variable_names[local_idx]),
                    "canonical_name": str(item.canonical),
                    "variable_idx": local_idx,
                    "channel": int(self.cfg.out_channels[local_idx]),
                }
            )
        return variables

    def _load_affine_calibration_if_needed(self) -> None:
        self.affine_calibration = None
        if not self.cfg.affine_calibration_path:
            return
        calibration = AffineCalibration.load(
            self.cfg.affine_calibration_path,
            variable_names=self.variable_names,
            out_channels=self.cfg.out_channels,
            logger=self.logger,
        )
        self.affine_calibration = calibration
        variables = calibration.variables
        leads = calibration.leads
        lead_text = "none" if not leads else f"{min(leads)}..{max(leads)}"
        lines = [
            "Affine calibration: enabled",
            f"Calibration path: {calibration.path}",
            f"Apply mode: {self.cfg.calibration_apply_mode}",
            f"Variables calibrated: {', '.join(variables) if variables else '(none)'}",
            f"Leads calibrated: {lead_text}",
        ]
        for line in lines:
            self.logger.info(line)
            print(line)

    def _apply_affine_calibration(self, pred_norm: torch.Tensor, lead: int, *, in_place: bool = False) -> torch.Tensor:
        if self.affine_calibration is None:
            return pred_norm
        return self.affine_calibration.apply(pred_norm, int(lead), in_place=in_place)

    def _resolve_checkpoint_path(self, path: str | os.PathLike[str]) -> str:
        path = str(path)
        if not path:
            return ""
        if os.path.isabs(path):
            return path
        candidate = os.path.join(self.cfg.experiment_dir, path)
        if os.path.exists(candidate):
            return candidate
        return path

    def _checkpoint_metadata(self, checkpoint_path: str) -> dict[str, Any]:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        metadata = dict(checkpoint.get("metadata", {}))
        if "epoch" not in metadata:
            metadata["epoch"] = checkpoint.get("epoch", None)
        return metadata

    def _build_eval_target_override_handler(self) -> TargetHandling | None:
        self.eval_target_override_handler = None
        self.eval_loss_channel_mask = None
        if not self.cfg.eval_target_override:
            return None

        settings = TargetHandlingSettings(
            enabled=True,
            copy_variables=tuple(self.cfg.eval_copy_variables),
            known_future_variables=tuple(self.cfg.eval_known_future_variables),
            exclude_loss_variables=tuple(self.cfg.eval_exclude_loss_variables),
        )
        handler = TargetHandling.from_settings(
            settings,
            self.cfg,
            channel_names=self.channel_names or None,
            out_channels=self.cfg.out_channels,
            logger=self.logger,
            context_name="evaluation target override",
        )
        self.eval_target_override_handler = handler
        self.eval_loss_channel_mask = handler.loss_channel_mask(len(self.cfg.out_channels), device=self.device)
        self._log_eval_target_override(handler)
        return handler

    def _log_eval_target_override(self, handler: TargetHandling) -> None:
        lines = ["Evaluation target override: enabled", "copy_variables:"]
        for name, idx in handler.copy_channels.items():
            lines.append(f"  {name} -> channel {int(idx)}")
        if not handler.copy_channels:
            lines.append("  (none)")
        lines.append("known_future_variables:")
        for name, idx in handler.known_future_channels.items():
            lines.append(f"  {name} -> channel {int(idx)}")
        if not handler.known_future_channels:
            lines.append("  (none)")
        lines.append("exclude_loss_variables:")
        for name, idx in handler.exclude_loss_channels.items():
            lines.append(f"  {name} -> channel {int(idx)}")
        if not handler.exclude_loss_channels:
            lines.append("  (none)")

        for line in lines:
            self.logger.info(line)
            print(line)

    def _eval_target_override_payload(self) -> dict[str, Any]:
        handler = self.eval_target_override_handler
        payload: dict[str, Any] = {
            "enabled": bool(self.cfg.eval_target_override),
            "debug_enabled": bool(self.cfg.debug_eval_target_override),
        }
        if handler is None:
            payload.update(
                {
                    "copy_variables": {},
                    "known_future_variables": {},
                    "exclude_loss_variables": {},
                }
            )
            return payload
        payload.update(
            {
                "copy_variables": {name: int(idx) for name, idx in handler.copy_channels.items()},
                "known_future_variables": {name: int(idx) for name, idx in handler.known_future_channels.items()},
                "exclude_loss_variables": {name: int(idx) for name, idx in handler.exclude_loss_channels.items()},
                "note": (
                    "Per-variable metrics are computed after evaluation-only target override. "
                    "Overridden variables should not be interpreted as model skill."
                ),
            }
        )
        return payload

    def _loss_payload_for_json(self, metrics: dict[str, np.ndarray]) -> dict[str, Any] | None:
        if "aggregate_loss_by_lead" not in metrics:
            return None
        return {
            "name": metrics.get("aggregate_loss_name", "aggregate_loss"),
            "description": metrics.get("aggregate_loss_description"),
            "excluded_variables": list(metrics.get("aggregate_loss_excluded_variables", [])),
            "included_channel_count": int(metrics.get("aggregate_loss_included_channel_count", 0)),
            "mean": _json_number(metrics.get("aggregate_loss_mean")),
            "by_lead": _json_series(metrics["aggregate_loss_by_lead"]),
        }

    def _reset_eval_target_override_debug(self, label: str, forecast_steps: int) -> None:
        self._eval_target_override_debug = None
        handler = self.eval_target_override_handler
        if not (self.cfg.eval_target_override and self.cfg.debug_eval_target_override and handler is not None):
            return
        self._eval_target_override_debug = {
            "label": str(label),
            "enabled": True,
            "tolerance": 1.0e-6,
            "max_samples": 3,
            "samples_checked": 0,
            "forecast_steps": int(forecast_steps),
            "copy_variables": {name: int(idx) for name, idx in handler.copy_channels.items()},
            "known_future_variables": {name: int(idx) for name, idx in handler.known_future_channels.items()},
            "leads": {
                str(lead): {
                    "copy_variables": {
                        name: {"channel": int(idx), "max_abs_diff": 0.0}
                        for name, idx in handler.copy_channels.items()
                    },
                    "known_future_variables": {
                        name: {"channel": int(idx), "max_abs_diff": 0.0}
                        for name, idx in handler.known_future_channels.items()
                    },
                }
                for lead in range(1, int(forecast_steps) + 1)
            },
        }

    def _should_debug_eval_target_override_ic(self) -> bool:
        stats = self._eval_target_override_debug
        if stats is None:
            return False
        return int(stats.get("samples_checked", 0)) < int(stats.get("max_samples", 0))

    def _finish_debug_eval_target_override_ic(self, did_check: bool) -> None:
        if did_check and self._eval_target_override_debug is not None:
            self._eval_target_override_debug["samples_checked"] = int(
                self._eval_target_override_debug.get("samples_checked", 0)
            ) + 1

    def _record_eval_target_override_debug(
        self,
        *,
        pred_norm: torch.Tensor,
        initial_state: torch.Tensor,
        target_norm: torch.Tensor,
        lead: int,
    ) -> None:
        stats = self._eval_target_override_debug
        handler = self.eval_target_override_handler
        if stats is None or handler is None:
            return
        lead_key = str(int(lead))
        lead_stats = stats["leads"][lead_key]
        for name, idx in handler.copy_channels.items():
            idx = int(idx)
            diff = (pred_norm[:, idx] - initial_state[:, idx]).detach().abs().max().item()
            entry = lead_stats["copy_variables"][name]
            entry["max_abs_diff"] = max(float(entry.get("max_abs_diff", 0.0)), float(diff))
        for name, idx in handler.known_future_channels.items():
            idx = int(idx)
            diff = (pred_norm[:, idx] - target_norm[:, idx]).detach().abs().max().item()
            entry = lead_stats["known_future_variables"][name]
            entry["max_abs_diff"] = max(float(entry.get("max_abs_diff", 0.0)), float(diff))

    def _write_eval_target_override_debug_report(self, output_root: Path, label: str) -> None:
        stats = self._eval_target_override_debug
        if stats is None:
            return
        tolerance = float(stats.get("tolerance", 1.0e-6))
        failures: list[str] = []
        for lead_key, lead_stats in stats["leads"].items():
            for group_name in ("copy_variables", "known_future_variables"):
                for variable, entry in lead_stats[group_name].items():
                    if float(entry["max_abs_diff"]) > tolerance:
                        failures.append(
                            f"{variable} lead {lead_key} max_abs_diff {entry['max_abs_diff']:.6g} exceeds {tolerance:.6g}"
                        )
        stats["pass"] = not failures
        stats["failures"] = failures

        output_root.mkdir(parents=True, exist_ok=True)
        safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(label)).strip("_")
        suffix = "" if str(label).lower().replace(" ", "_") in {"global_best", "global"} else f"_{safe_label or 'rollout'}"
        json_path = output_root / f"target_override_debug_report{suffix}.json"
        txt_path = output_root / f"target_override_debug_report{suffix}.txt"
        json_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

        lines = [
            "Evaluation target override debug report:",
            f"- label: {label}",
            f"- samples checked: {stats.get('samples_checked')}",
            f"- tolerance: {tolerance:.6g}",
            "",
        ]
        for group_name, heading in [
            ("copy_variables", "copy variable max_abs_diff by lead"),
            ("known_future_variables", "known future variable max_abs_diff by lead"),
        ]:
            lines.append(f"{heading}:")
            variables = stats.get(group_name, {})
            if not variables:
                lines.append("  (none)")
            for variable in variables:
                values = [
                    f"lead {lead}: {stats['leads'][str(lead)][group_name][variable]['max_abs_diff']:.6g}"
                    for lead in range(1, int(stats["forecast_steps"]) + 1)
                ]
                lines.append(f"  {variable}: " + ", ".join(values))
            lines.append("")
        if failures:
            lines.append("FAIL:")
            lines.extend(f"- {item}" for item in failures)
        else:
            lines.append("PASS")
        txt_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        self.logger.info("Saved target override debug JSON: %s", json_path)
        self.logger.info("Saved target override debug text: %s", txt_path)
        if failures:
            raise AssertionError("Evaluation target override debug checks failed: " + "; ".join(failures))

    def _build_sensitivity_channels(self) -> dict[str, int]:
        self.sensitivity_channels = {}
        mode = str(self.cfg.eval_sensitivity_mode)
        required_variables = SENSITIVITY_MODE_VARIABLES.get(mode, ())
        if not required_variables:
            if bool(self.cfg.debug_eval_sensitivity):
                self._log_sensitivity_startup()
            return {}

        resolver = VariableResolver(
            self.cfg,
            self.channel_names or None,
            self.cfg.out_channels,
            logger=self.logger,
        )
        resolved: dict[str, int] = {}
        for variable in required_variables:
            item = resolver.resolve(variable, required=False)
            if item.local_index is None:
                raise ValueError(
                    f"Orog/TISR sensitivity mode {mode!r} requires variable {variable!r}, "
                    "but it could not be resolved in model output channels."
                )
            resolved[item.canonical] = int(item.local_index)
        self.sensitivity_channels = resolved
        self._log_sensitivity_startup()
        return resolved

    def _log_sensitivity_startup(self) -> None:
        mode = str(self.cfg.eval_sensitivity_mode)
        lines = [
            "Orog/TISR sensitivity test:",
            f"  mode: {mode}",
            f"  noise_seed: {int(self.cfg.eval_sensitivity_noise_seed)}",
        ]
        for variable in ("orog", "tisr"):
            if variable in self.sensitivity_channels:
                lines.append(f"  {variable} -> channel {int(self.sensitivity_channels[variable])}")
            elif variable in SENSITIVITY_MODE_VARIABLES.get(mode, ()):
                lines.append(f"  {variable} -> unresolved")
        for line in lines:
            self.logger.info(line)
            print(line)

    def _sensitivity_payload(self) -> dict[str, Any]:
        return {
            "mode": str(self.cfg.eval_sensitivity_mode),
            "enabled": str(self.cfg.eval_sensitivity_mode) != "normal",
            "noise_seed": int(self.cfg.eval_sensitivity_noise_seed),
            "channels": {name: int(idx) for name, idx in self.sensitivity_channels.items()},
            "state_space": "normalized",
            "notes": [
                "Zero perturbations set normalized state channels to 0.",
                "Random perturbations use standard normal noise in normalized state space.",
                "Shuffle perturbations permute across batch when batch_size>1; this evaluator falls back to longitude roll when batch_size=1.",
            ],
        }

    def _reset_sensitivity_debug(self, label: str, forecast_steps: int) -> None:
        self._sensitivity_debug = None
        if not bool(self.cfg.debug_eval_sensitivity):
            return
        self._sensitivity_debug = {
            "label": str(label),
            "mode": str(self.cfg.eval_sensitivity_mode),
            "enabled": str(self.cfg.eval_sensitivity_mode) != "normal",
            "noise_seed": int(self.cfg.eval_sensitivity_noise_seed),
            "state_space": "normalized",
            "tolerance": 1.0e-6,
            "max_samples": max(0, int(self.cfg.eval_sensitivity_debug_samples)),
            "samples_checked": 0,
            "forecast_steps": int(forecast_steps),
            "channels": {name: int(idx) for name, idx in self.sensitivity_channels.items()},
            "fallbacks": [],
            "records": [],
            "summary": {},
            "pass": True,
            "failures": [],
            "warnings": [],
        }

    def _should_debug_sensitivity_ic(self) -> bool:
        stats = self._sensitivity_debug
        if stats is None:
            return False
        return int(stats.get("samples_checked", 0)) < int(stats.get("max_samples", 0))

    def _finish_debug_sensitivity_ic(self, did_check: bool) -> None:
        if did_check and self._sensitivity_debug is not None:
            self._sensitivity_debug["samples_checked"] = int(self._sensitivity_debug.get("samples_checked", 0)) + 1

    def _sensitivity_seed(self, *, mode: str, variable: str, ic: int, lead: int) -> int:
        key = f"{mode}|{variable}|{int(ic)}|{int(lead)}|{int(self.cfg.eval_sensitivity_noise_seed)}"
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        offset = int.from_bytes(digest[:8], byteorder="little", signed=False)
        return int((int(self.cfg.eval_sensitivity_noise_seed) + offset) % (2**63 - 1))

    def _randn_like_deterministic(self, reference: torch.Tensor, *, seed: int) -> torch.Tensor:
        try:
            generator = torch.Generator(device=reference.device)
            generator.manual_seed(int(seed))
            return torch.randn(reference.shape, device=reference.device, dtype=reference.dtype, generator=generator)
        except (TypeError, RuntimeError):
            generator = torch.Generator()
            generator.manual_seed(int(seed))
            return torch.randn(reference.shape, dtype=reference.dtype, generator=generator).to(reference.device)

    def _log_sensitivity_fallback(self, key: str, message: str) -> None:
        if key in self._sensitivity_logged_fallbacks:
            return
        self._sensitivity_logged_fallbacks.add(key)
        self.logger.warning(message)
        print(message)
        if self._sensitivity_debug is not None:
            self._sensitivity_debug.setdefault("fallbacks", []).append(message)

    def _shuffle_channel(
        self,
        values: torch.Tensor,
        *,
        variable: str,
        mode: str,
        ic: int,
        lead: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        bsz = int(values.shape[0])
        info: dict[str, Any] = {"method": "batch_permutation", "permutation_identity": False}
        if bsz > 1:
            seed = self._sensitivity_seed(mode=mode, variable=variable, ic=ic, lead=lead)
            generator = torch.Generator(device=values.device)
            generator.manual_seed(int(seed))
            perm = torch.randperm(bsz, device=values.device, generator=generator)
            info["permutation"] = [int(x) for x in perm.detach().cpu().tolist()]
            info["permutation_identity"] = bool(torch.equal(perm, torch.arange(bsz, device=values.device)))
            return values[perm], info

        shift = max(1, int(values.shape[-1]) // 3)
        info.update({"method": "longitude_roll_fallback", "shift": int(shift), "permutation_identity": False})
        self._log_sensitivity_fallback(
            f"{mode}:{variable}:batch1",
            f"Orog/TISR sensitivity {mode}: batch size is 1; using longitude roll fallback for {variable}.",
        )
        return torch.roll(values, shifts=shift, dims=-1), info

    def _sensitivity_actions_for_mode(self, mode: str) -> list[tuple[str, str]]:
        if mode == "override_orog_tisr":
            return [("orog", "override"), ("tisr", "override")]
        if mode.startswith("tisr_"):
            return [("tisr", mode.split("_", 1)[1])]
        if mode.startswith("orog_tisr_"):
            action = mode.rsplit("_", 1)[1]
            return [("orog", action), ("tisr", action)]
        if mode.startswith("orog_"):
            return [("orog", mode.split("_", 1)[1])]
        return []

    def _apply_sensitivity_mode(
        self,
        *,
        pred_norm: torch.Tensor,
        current_state: torch.Tensor,
        initial_state: torch.Tensor,
        target_norm: torch.Tensor,
        lead: int,
        ic: int,
        debug: bool,
    ) -> torch.Tensor:
        mode = str(self.cfg.eval_sensitivity_mode)
        if mode == "normal":
            return pred_norm
        actions = self._sensitivity_actions_for_mode(mode)
        if not actions:
            return pred_norm

        out = pred_norm.clone()
        for variable, action in actions:
            idx = self.sensitivity_channels.get(variable)
            if idx is None:
                raise RuntimeError(f"Sensitivity mode {mode!r} has no resolved channel for {variable!r}.")
            idx = int(idx)
            before = out[:, idx].clone()
            target = target_norm[:, idx]
            expected = None
            shuffle_info: dict[str, Any] = {}

            if action == "override":
                if variable == "orog":
                    replacement = initial_state[:, idx]
                    expected = replacement
                elif variable == "tisr":
                    replacement = target
                    expected = replacement
                else:
                    raise RuntimeError(f"Unsupported override variable: {variable}")
            elif action == "zero":
                replacement = torch.zeros_like(before)
                expected = replacement
            elif action == "random":
                seed = self._sensitivity_seed(mode=mode, variable=variable, ic=ic, lead=lead)
                replacement = self._randn_like_deterministic(before, seed=seed)
            elif action == "shuffle":
                source = target if variable == "tisr" else initial_state[:, idx]
                replacement, shuffle_info = self._shuffle_channel(
                    source,
                    variable=variable,
                    mode=mode,
                    ic=ic,
                    lead=lead,
                )
            else:
                raise RuntimeError(f"Unsupported sensitivity action: {action}")

            out[:, idx] = replacement
            if debug:
                self._record_sensitivity_debug(
                    mode=mode,
                    variable=variable,
                    action=action,
                    channel=idx,
                    lead=lead,
                    before=before,
                    after=out[:, idx],
                    target=target,
                    initial=initial_state[:, idx],
                    expected=expected,
                    shuffle_info=shuffle_info,
                )
        return out

    def _record_sensitivity_debug(
        self,
        *,
        mode: str,
        variable: str,
        action: str,
        channel: int,
        lead: int,
        before: torch.Tensor,
        after: torch.Tensor,
        target: torch.Tensor,
        initial: torch.Tensor,
        expected: torch.Tensor | None,
        shuffle_info: dict[str, Any],
    ) -> None:
        stats = self._sensitivity_debug
        if stats is None:
            return
        tol = float(stats.get("tolerance", 1.0e-6))
        before_f = before.detach().float()
        after_f = after.detach().float()
        target_f = target.detach().float()
        initial_f = initial.detach().float()
        diff_before_after = (after_f - before_f).abs()
        record = {
            "lead": int(lead),
            "variable": str(variable),
            "channel": int(channel),
            "action": str(action),
            "before_pred_channel_mean": float(before_f.mean().item()),
            "after_pred_channel_mean": float(after_f.mean().item()),
            "target_channel_mean": float(target_f.mean().item()),
            "initial_channel_mean": float(initial_f.mean().item()),
            "after_pred_channel_std": float(after_f.std(unbiased=False).item()),
            "max_abs_diff_before_after": float(diff_before_after.max().item()),
            "shuffle_info": shuffle_info,
        }
        if expected is not None:
            expected_f = expected.detach().float()
            record["max_abs_diff_after_expected"] = float((after_f - expected_f).abs().max().item())
        if action == "zero":
            record["max_abs_after_zero"] = float(after_f.abs().max().item())

        stats["records"].append(record)
        summary_key = f"{variable}:lead{int(lead)}"
        summary = stats["summary"].setdefault(
            summary_key,
            {
                "variable": variable,
                "lead": int(lead),
                "max_abs_diff_before_after": 0.0,
                "max_abs_diff_after_expected": 0.0,
                "max_abs_after_zero": 0.0,
                "max_after_std": 0.0,
            },
        )
        summary["max_abs_diff_before_after"] = max(
            float(summary["max_abs_diff_before_after"]),
            float(record["max_abs_diff_before_after"]),
        )
        summary["max_after_std"] = max(float(summary["max_after_std"]), float(record["after_pred_channel_std"]))
        if "max_abs_diff_after_expected" in record:
            summary["max_abs_diff_after_expected"] = max(
                float(summary["max_abs_diff_after_expected"]),
                float(record["max_abs_diff_after_expected"]),
            )
        if "max_abs_after_zero" in record:
            summary["max_abs_after_zero"] = max(float(summary["max_abs_after_zero"]), float(record["max_abs_after_zero"]))

        failures = stats.setdefault("failures", [])
        warnings = stats.setdefault("warnings", [])
        if action == "override" and float(record.get("max_abs_diff_after_expected", 0.0)) > tol:
            failures.append(f"{mode} {variable} lead {lead}: override did not match expected source.")
        elif action == "zero" and float(record.get("max_abs_after_zero", 0.0)) > tol:
            failures.append(f"{mode} {variable} lead {lead}: zero perturbation left nonzero values.")
        elif action == "random":
            if float(record["after_pred_channel_std"]) <= tol:
                failures.append(f"{mode} {variable} lead {lead}: random perturbation has near-zero std.")
            if float(record["max_abs_diff_before_after"]) <= tol:
                failures.append(f"{mode} {variable} lead {lead}: random perturbation did not change prediction.")
        elif action == "shuffle":
            if not bool(shuffle_info.get("permutation_identity", False)) and float(record["max_abs_diff_before_after"]) <= tol:
                warnings.append(
                    f"{mode} {variable} lead {lead}: shuffle/roll did not change values; source field may be spatially constant."
                )

    def _write_sensitivity_debug_report(self, output_root: Path) -> None:
        stats = self._sensitivity_debug
        if stats is None:
            return
        failures = list(dict.fromkeys(str(item) for item in stats.get("failures", [])))
        warnings = list(dict.fromkeys(str(item) for item in stats.get("warnings", [])))
        stats["failures"] = failures
        stats["warnings"] = warnings
        stats["pass"] = not failures
        output_root.mkdir(parents=True, exist_ok=True)
        json_path = output_root / "sensitivity_debug_report.json"
        txt_path = output_root / "sensitivity_debug_report.txt"
        json_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

        lines = [
            "Orog/TISR sensitivity debug report:",
            f"- mode: {stats.get('mode')}",
            f"- state space: {stats.get('state_space')}",
            f"- noise seed: {stats.get('noise_seed')}",
            f"- samples checked: {stats.get('samples_checked')}",
            f"- pass: {stats.get('pass')}",
            "",
            "Channel summary:",
        ]
        if not stats.get("summary"):
            lines.append("  (no perturbation records)")
        for key, summary in sorted(stats.get("summary", {}).items()):
            line = (
                f"  {key}: max before-after {float(summary['max_abs_diff_before_after']):.6g}, "
                f"max expected diff {float(summary['max_abs_diff_after_expected']):.6g}, "
                f"max zero abs {float(summary['max_abs_after_zero']):.6g}, "
                f"max after std {float(summary['max_after_std']):.6g}"
            )
            lines.append(line)
        if stats.get("fallbacks"):
            lines.extend(["", "Fallbacks:"])
            lines.extend(f"- {item}" for item in stats["fallbacks"])
        if warnings:
            lines.extend(["", "Warnings:"])
            lines.extend(f"- {item}" for item in warnings)
        if failures:
            lines.extend(["", "Failures:"])
            lines.extend(f"- {item}" for item in failures)
        txt_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        self.logger.info("Saved sensitivity debug JSON: %s", json_path)
        self.logger.info("Saved sensitivity debug text: %s", txt_path)

    def _load_model(self, height: int, width: int, checkpoint_path: str) -> GraphWeatherModel:
        checkpoint_path = self._resolve_checkpoint_path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        metadata = dict(checkpoint.get("metadata", {}))
        self._validate_checkpoint_resolution(metadata)
        validate_checkpoint_architecture(metadata, self.cfg)
        graph = load_graph_bundle(self.cfg.graph_path, map_location="cpu").to(self.device)
        self._validate_graph_resolution(graph.metadata)
        graph_levels = int(graph.metadata.get("num_graph_levels", getattr(graph, "num_graph_levels", 3)))
        if graph_levels != int(getattr(self.cfg, "num_graph_levels", 3)):
            raise ValueError(
                f"Graph cache num_graph_levels={graph_levels} does not match active model "
                f"num_graph_levels={getattr(self.cfg, 'num_graph_levels', 3)}."
            )
        per_step_channels = len(self.cfg.in_channels)
        if self.cfg.add_grid:
            per_step_channels += self.cfg.n_grid_channels
        if self.cfg.orography:
            per_step_channels += 1
        self.feature_builder = RolloutFeatureBuilder.from_params(
            self.cfg,
            graph=graph,
            channel_names=self.channel_names or None,
            out_channels=self.cfg.out_channels,
            output_means=self.out_means,
            output_stds=self.out_stds,
            logger=self.logger,
        )
        self.target_handler = TargetHandling.from_params(
            self.cfg,
            channel_names=self.channel_names or None,
            out_channels=self.cfg.out_channels,
            logger=self.logger,
        )
        lead_conditioning = dict(self.cfg.lead_conditioning or {})
        lead_added_input_channels = int(lead_conditioning.get("added_input_channels", 0))
        if bool(lead_conditioning.get("enabled", False)) and lead_added_input_channels <= 0:
            lead_added_input_channels = 2
            lead_conditioning["added_input_channels"] = 2
        base_input_channels = int(2 * per_step_channels + lead_added_input_channels)
        total_input_channels = int(base_input_channels + self.feature_builder.aux_feature_dim)
        active_feature_metadata = self.feature_builder.checkpoint_metadata(
            base_input_channels=base_input_channels,
            total_input_channels=total_input_channels,
        )
        ok, reason = feature_metadata_matches(active_feature_metadata, metadata)
        if not ok:
            raise RuntimeError(f"Checkpoint extra-feature configuration mismatch: {reason}")
        ok, reason = target_handling_metadata_matches(self.target_handler.metadata, metadata)
        if not ok:
            checkpoint_has_target_handling = "target_handling" in metadata
            if checkpoint_has_target_handling:
                raise RuntimeError(f"Checkpoint target-handling configuration mismatch: {reason}")
            if self.target_handler.enabled:
                message = (
                    "Evaluating legacy checkpoint without target-handling metadata using active evaluation target handling. "
                    "Model weights are unchanged."
                )
                self.logger.warning(message)
                print(message)
        if self.cfg.eval_target_override:
            checkpoint_target_handling = metadata.get("target_handling")
            if not (isinstance(checkpoint_target_handling, dict) and bool(checkpoint_target_handling.get("enabled", False))):
                message = (
                    "Using evaluation-only target override on checkpoint trained without target handling. "
                    "Model weights are unchanged."
                )
                self.logger.info(message)
                print(message)
        self._build_eval_target_override_handler()
        self._build_sensitivity_channels()
        state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
        cleaned = OrderedDict()
        for key, value in state.items():
            cleaned[key[7:] if key.startswith("module.") else key] = value

        use_delta_normalization = bool(metadata.get("use_delta_normalization", self.cfg.use_delta_normalization))
        delta_norm_center = bool(metadata.get("delta_norm_center", self.cfg.delta_norm_center))
        delta_mean = None
        delta_std = None
        if use_delta_normalization:
            if "delta_mean" in cleaned and "delta_std" in cleaned:
                delta_mean = cleaned["delta_mean"].detach().cpu()
                delta_std = cleaned["delta_std"].detach().cpu()
            elif self.cfg.delta_stats_path and os.path.exists(str(self.cfg.delta_stats_path)):
                delta_mean, delta_std, _ = load_delta_stats(
                    str(self.cfg.delta_stats_path),
                    output_channels=len(self.cfg.out_channels),
                    eps=self.cfg.delta_norm_eps,
                )
            else:
                raise RuntimeError("Delta normalization is enabled but checkpoint lacks delta_mean/delta_std buffers.")

        model = GraphWeatherModel(
            graph=graph,
            grid_shape=(height, width),
            input_channels=base_input_channels,
            output_channels=len(self.cfg.out_channels),
            n_history=self.cfg.n_history,
            hidden_dim=self.cfg.hidden_dim,
            edge_dim=self.cfg.edge_dim,
            heads=self.cfg.num_heads,
            k_neighbors=self.cfg.k_neighbors,
            level_k_neighbors=self.cfg.level_k_neighbors,
            encoder_blocks=self.cfg.encoder_blocks,
            decoder_blocks=self.cfg.decoder_blocks,
            l0_blocks=self.cfg.l0_blocks,
            l1_blocks=self.cfg.l1_blocks,
            l2_blocks=self.cfg.l2_blocks,
            l1_refine_blocks=self.cfg.l1_refine_blocks,
            l0_refine_blocks=self.cfg.l0_refine_blocks,
            num_graph_levels=self.cfg.num_graph_levels,
            use_l3=self.cfg.use_l3,
            l3_blocks=self.cfg.l3_blocks,
            l4_blocks=self.cfg.l4_blocks,
            l3_refine_after_l4_blocks=self.cfg.l3_refine_after_l4_blocks,
            l2_refine_after_l3_blocks=self.cfg.l2_refine_after_l3_blocks,
            skip_fusion=self.cfg.skip_fusion,
            pooling=self.cfg.pooling,
            l0_refine=self.cfg.l0_refine,
            lead_conditioning=lead_conditioning,
            use_delta_normalization=use_delta_normalization,
            delta_mean=delta_mean,
            delta_std=delta_std,
            delta_norm_center=delta_norm_center,
            delta_norm_eps=self.cfg.delta_norm_eps,
            aux_feature_dim=int(self.feature_builder.aux_feature_dim),
        ).to(self.device)

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        allowed_missing = set() if use_delta_normalization else {"delta_mean", "delta_std"}
        bad_missing = [key for key in missing if key not in allowed_missing]
        if bad_missing or unexpected:
            raise RuntimeError(f"Checkpoint model_state mismatch. Missing={bad_missing}; unexpected={list(unexpected)}")
        model.eval()
        param_count = sum(p.numel() for p in model.parameters())
        self._graphweather_params_m = float(param_count) / 1.0e6
        self.logger.info(
            "Loaded checkpoint: %s | epoch=%s | train_rollout_steps=%s",
            checkpoint_path,
            metadata.get("epoch", checkpoint.get("epoch", "unknown")),
            metadata.get("train_rollout_steps", "unknown"),
        )
        self.logger.info("Evaluation model parameters: %d", param_count)
        self.logger.info("Lead conditioning:")
        self.logger.info("  enabled: %s", str(bool(lead_conditioning.get("enabled", False))).lower())
        self.logger.info("  type: %s", lead_conditioning.get("type", "none"))
        self.logger.info("  max_lead: %s", lead_conditioning.get("max_lead", 0))
        self.logger.info("  added_input_channels: %d", int(lead_conditioning.get("added_input_channels", 0)))
        if bool(lead_conditioning.get("enabled", False)):
            for lead, values in lead_conditioning_debug_values(int(lead_conditioning.get("max_lead", 10))).items():
                self.logger.info("  lead %d: sin=%.6f, cos=%.6f", lead, values["sin"], values["cos"])
        self.feature_builder.log_startup(
            base_input_channels=base_input_channels,
            total_input_channels=total_input_channels,
            output_channels=len(self.cfg.out_channels),
        )
        self.target_handler.log_startup(output_channels=len(self.cfg.out_channels))
        return model

    def _forward_model_step(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        aux_features: torch.Tensor | None,
        lead: int,
    ) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model must be loaded before rollout.")
        if getattr(self.model, "lead_conditioning_enabled", False):
            if aux_features is None:
                return self.model.forward_steps(previous, current, lead=int(lead))
            return self.model.forward_steps(previous, current, aux_features=aux_features, lead=int(lead))
        if aux_features is None:
            return self.model.forward_steps(previous, current)
        return self.model.forward_steps(previous, current, aux_features=aux_features)

    def _validate_graph_resolution(self, metadata: dict[str, Any]) -> None:
        mode = metadata.get("resolution_mode", None)
        if mode is None:
            if self.cfg.resolution_mode == "5p625":
                self.logger.warning("Graph cache has no resolution metadata; treating it as legacy 5p625.")
            else:
                raise ValueError(f"Graph cache has no resolution metadata and cannot be used for {self.cfg.resolution_mode}.")
        elif str(mode) != self.cfg.resolution_mode:
            raise ValueError(f"Graph cache mode {mode} does not match active mode {self.cfg.resolution_mode}.")
        grid_shape = metadata.get("grid_shape", None)
        if grid_shape is not None and tuple(int(x) for x in grid_shape) != tuple(self.cfg.grid_shape):
            raise ValueError(f"Graph grid shape {grid_shape} does not match active grid {self.cfg.grid_shape}.")
        num_levels = int(metadata.get("num_graph_levels", 3))
        if num_levels != int(self.cfg.num_graph_levels):
            raise ValueError(
                f"Graph cache num_graph_levels={num_levels} does not match active model "
                f"num_graph_levels={self.cfg.num_graph_levels}."
            )
        graph_format_version = metadata.get("graph_format_version", 0)
        if self.cfg.graph_format_version and graph_format_version != self.cfg.graph_format_version:
            raise ValueError(
                f"Graph cache mismatch: loaded graph has graph_format_version={graph_format_version}, "
                f"config requires {self.cfg.graph_format_version}."
            )
        hierarchy = metadata.get("hierarchy_type", "standard")
        if hierarchy is not None and str(hierarchy) != str(self.cfg.hierarchy_type):
            raise ValueError(
                f"Graph cache mismatch: loaded graph has hierarchy_type={hierarchy}, "
                f"config requires {self.cfg.hierarchy_type}."
            )
        use_l4_ratio15 = bool(metadata.get("use_l4_ratio15", False))
        if use_l4_ratio15 != bool(self.cfg.use_l4_ratio15):
            raise ValueError(
                f"Graph cache mismatch: loaded graph has use_l4_ratio15={use_l4_ratio15}, "
                f"config requires {self.cfg.use_l4_ratio15}."
            )
        strategy = metadata.get("graph_connectivity_strategy", metadata.get("connectivity_strategy", None))
        if strategy is not None and str(strategy) != str(self.cfg.graph_connectivity_strategy):
            raise ValueError(
                f"Graph cache mismatch: loaded graph has connectivity_strategy={strategy}, "
                f"config requires {self.cfg.graph_connectivity_strategy}."
            )
        actual_level_k = metadata.get("level_k_neighbors", None)
        if actual_level_k is None:
            actual_level_k = [int(metadata.get("graph_k", metadata.get("k", self.cfg.k_neighbors)))] * num_levels
        actual_level_k = [int(x) for x in actual_level_k]
        expected_level_k = list(self.cfg.level_k_neighbors or [self.cfg.k_neighbors] * int(self.cfg.num_graph_levels))
        if actual_level_k != expected_level_k:
            raise ValueError(
                f"Graph cache mismatch: loaded graph has level_k_neighbors={actual_level_k}, "
                f"config requires {expected_level_k}."
            )
        checks = [
            ("level_shapes", metadata.get("level_shapes"), self.cfg.level_shapes),
            ("node_counts", metadata.get("node_counts"), self.cfg.node_counts),
            ("edge_counts", metadata.get("edge_counts"), self.cfg.edge_counts),
        ]
        for name, actual, expected in checks:
            if expected and actual is not None and actual != expected:
                raise ValueError(
                    f"Graph cache mismatch: loaded graph has {name}={actual}, config requires {expected}."
                )

    def _validate_checkpoint_resolution(self, metadata: dict[str, Any]) -> None:
        mode = metadata.get("resolution_mode", None)
        if mode is None:
            if self.cfg.resolution_mode == "5p625":
                self.logger.warning("Checkpoint has no resolution metadata; treating it as legacy 5p625.")
                return
            raise RuntimeError(f"Cannot evaluate a legacy checkpoint without resolution metadata in {self.cfg.resolution_mode} mode.")
        if str(mode) != self.cfg.resolution_mode:
            raise RuntimeError(f"Cannot evaluate a {mode} checkpoint in {self.cfg.resolution_mode} mode.")

    def _normalize_input_step(self, raw_step_all_channels: np.ndarray) -> torch.Tensor:
        step = np.asarray(raw_step_all_channels[self.cfg.in_channels], dtype=np.float32)
        step = (step - self.in_means.reshape(-1, 1, 1)) / (self.in_stds.reshape(-1, 1, 1) + 1.0e-8)
        if self.cfg.add_grid:
            step = _add_grid_channels(step, self.cfg.gridtype, self.cfg.n_grid_channels)
        if self.cfg.orography:
            if self.orography_field is None:
                raise ValueError("orography=true but no orography field was loaded.")
            step = np.concatenate([step, self.orography_field[None, :, :].astype(np.float32)], axis=0)
        return torch.as_tensor(step[None, ...], device=self.device, dtype=torch.float32)

    def _normalize_target_step(self, raw_step_all_channels: np.ndarray) -> torch.Tensor:
        step = np.asarray(raw_step_all_channels[self.cfg.out_channels], dtype=np.float32)
        step = (step - self.out_means.reshape(-1, 1, 1)) / (self.out_stds.reshape(-1, 1, 1) + 1.0e-8)
        return torch.as_tensor(step[None, ...], device=self.device, dtype=torch.float32)

    def _target_time_tensors(self, target_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._time_values:
            self._warn_climatology_time_fallback()
            doy = int(target_idx) % 365 + 1
            days = 365
        else:
            value = self._time_values[int(target_idx)]
            doy = int(dayofyear(value))
            year = int(date_from_time(value).year)
            days = 366 if calendar.isleap(year) else 365
        return (
            torch.as_tensor([[doy]], device=self.device, dtype=torch.long),
            torch.as_tensor([[days]], device=self.device, dtype=torch.long),
        )

    def _default_climatology_path(self) -> Path:
        return Path(self.cfg.output_dir) / "daily_climatology_dayofyear.nc"

    def _load_or_build_climatology(self, height: int, width: int) -> DayOfYearClimatology:
        requested_path = Path(str(self.cfg.climatology_path)).expanduser() if self.cfg.climatology_path else None
        if requested_path and requested_path.exists():
            self.logger.info("Loading calendar-aware climatology: %s", requested_path)
            climatology = load_dayofyear_climatology(requested_path, self.cfg.out_channels, height, width)
            self._climatology_metadata = dict(climatology.metadata)
            return climatology

        default_path = self._default_climatology_path()
        if not requested_path and default_path.exists():
            self.logger.info("Loading default calendar-aware climatology: %s", default_path)
            climatology = load_dayofyear_climatology(default_path, self.cfg.out_channels, height, width)
            self._climatology_metadata = dict(climatology.metadata)
            return climatology

        if not (self.cfg.compute_climatology or self.cfg.build_climatology_if_missing):
            missing = requested_path if requested_path else default_path
            raise FileNotFoundError(
                f"Calendar-aware climatology not found: {missing}. "
                "Build it with scripts/build_climatology.py or pass --build_climatology_if_missing."
            )

        save_path = requested_path or default_path
        total_start = time.perf_counter()
        metadata = build_dayofyear_climatology(
            data_path=self.cfg.train_data_path,
            output_path=save_path,
            out_channels=self.cfg.out_channels,
            split="train",
            chunk_size=int(self.cfg.climatology_chunk_size),
            logger=self.logger,
        )
        self.logger.info("Built climatology in %.1fs", time.perf_counter() - total_start)
        climatology = load_dayofyear_climatology(save_path, self.cfg.out_channels, height, width)
        self._climatology_metadata = {**dict(climatology.metadata), **metadata}
        return climatology

    def _prepare_climatology_lookup(self, climatology: DayOfYearClimatology) -> torch.Tensor:
        if self._time_values and (
            not bool(climatology.metadata.get("calendar_aware", False))
            or not bool(climatology.metadata.get("dayofyear_from_real_time_coordinate", False))
        ):
            raise ValueError(
                "ACC requires a calendar-aware day-of-year climatology built from real NetCDF time coordinates. "
                "Run scripts/build_climatology.py and pass --climatology_path to the evaluator."
            )
        if not self._time_values:
            self._warn_climatology_time_fallback()
        clim_norm = (climatology.values - self.out_means.reshape(1, -1, 1, 1)) / (
            self.out_stds.reshape(1, -1, 1, 1) + 1.0e-8
        )
        climatology_norm = torch.as_tensor(clim_norm, device=self.device, dtype=torch.float32)
        self._climatology_available_days = climatology.available_dayofyears
        self._climatology_norm_by_day = {}
        for idx, doy in enumerate(climatology.dayofyears):
            if int(doy) in self._climatology_available_days and np.all(np.isfinite(clim_norm[idx])):
                self._climatology_norm_by_day[int(doy)] = climatology_norm[idx : idx + 1]
        if not self._climatology_norm_by_day:
            raise RuntimeError("Loaded climatology has no finite available day-of-year entries.")
        self._climatology_metadata.update(
            {
                "climatology_path": self._climatology_metadata.get("climatology_path")
                or self._climatology_metadata.get("path"),
                "climatology_source_split": self._climatology_metadata.get("climatology_source_split")
                or self._climatology_metadata.get("source_split"),
                "calendar_aware": bool(self._climatology_metadata.get("calendar_aware", True)),
                "dayofyear_from_real_time_coordinate": bool(
                    self._climatology_metadata.get("dayofyear_from_real_time_coordinate", True)
                ),
                "available_dayofyears": sorted(int(day) for day in self._climatology_available_days),
                "time_lookup_fallback": not bool(self._time_values),
            }
        )
        return climatology_norm

    def _warn_climatology_time_fallback(self) -> None:
        if self._climatology_time_fallback_warned:
            return
        message = "WARNING: no time coordinate found; using index % 365 climatology lookup."
        self.logger.warning(message)
        print(message)
        self._climatology_time_fallback_warned = True

    def _target_dayofyear(self, target_idx: int) -> int:
        if self._time_values:
            return int(dayofyear(self._time_values[int(target_idx)]))
        self._warn_climatology_time_fallback()
        return int(target_idx) % 365 + 1

    def _climatology_norm_for_day(self, doy: int) -> torch.Tensor:
        doy = int(doy)
        if doy in self._climatology_norm_by_day:
            return self._climatology_norm_by_day[doy]
        if doy == 366 and 365 in self._climatology_norm_by_day and 1 in self._climatology_norm_by_day:
            if doy not in self._climatology_warned_days:
                self.logger.warning(
                    "Day-of-year 366 is missing from climatology; using average of day 365 and day 1 for Feb 29 targets."
                )
                self._climatology_warned_days.add(doy)
            value = 0.5 * (self._climatology_norm_by_day[365] + self._climatology_norm_by_day[1])
            self._climatology_norm_by_day[doy] = value
            return value

        available = np.asarray(sorted(self._climatology_norm_by_day), dtype=np.int64)
        nearest = int(available[np.argmin(np.abs(available - doy))])
        if doy not in self._climatology_warned_days:
            self.logger.warning(
                "Day-of-year %d is missing from climatology; using nearest available day %d.",
                doy,
                nearest,
            )
            self._climatology_warned_days.add(doy)
        return self._climatology_norm_by_day[nearest]

    def _build_ic_indices(self, total_t: int, forecast_steps: int) -> list[int]:
        start = max(int(self.cfg.eval_start_timestep), self.cfg.dt * self.cfg.n_history)
        max_ic = total_t - forecast_steps * self.cfg.dt - 1
        if max_ic < start:
            raise ValueError(
                f"Not enough timesteps ({total_t}) for forecast_steps={forecast_steps}, "
                f"dt={self.cfg.dt}, start={start}."
            )

        candidates = list(range(start, max_ic + 1))
        start_date = parse_date(self.cfg.start_date)
        end_date = parse_date(self.cfg.end_date)
        if start_date or end_date:
            if not self._time_values:
                raise ValueError("start_date/end_date filtering requires decoded dataset time coordinates.")
            filtered: list[int] = []
            for ic in candidates:
                current_date = date_from_time(self._time_values[ic])
                if start_date and current_date < start_date:
                    continue
                if end_date and current_date > end_date:
                    continue
                filtered.append(ic)
            candidates = filtered

        if self.cfg.start_offset:
            candidates = candidates[int(self.cfg.start_offset) :]

        selection = str(self.cfg.selection).lower()
        if selection == "first_n":
            selected = candidates[: int(self.cfg.n_initial_conditions)]
        elif selection == "stride":
            selected = candidates[:: int(self.cfg.ic_stride)]
        elif selection == "all":
            selected = candidates
        else:
            raise ValueError(f"Unsupported selection mode: {self.cfg.selection}")

        if self.cfg.max_initial_conditions is not None:
            selected = selected[: int(self.cfg.max_initial_conditions)]
        if not selected:
            raise ValueError("Evaluation start selection produced zero valid initial conditions.")

        first_time = self._time_values[selected[0]] if self._time_values else None
        last_time = self._time_values[selected[-1]] if self._time_values else None
        years = sorted({int(date_from_time(self._time_values[idx]).year) for idx in selected}) if self._time_values else []
        self._selection_metadata = {
            "split": self.cfg.split,
            "rollout_steps": int(forecast_steps),
            "selection": selection,
            "stride": int(self.cfg.ic_stride),
            "n_initial_conditions_requested": int(self.cfg.n_initial_conditions),
            "max_initial_conditions": self.cfg.max_initial_conditions,
            "start_offset": int(self.cfg.start_offset),
            "start_date": self.cfg.start_date,
            "end_date": self.cfg.end_date,
            "selected_initial_conditions": int(len(selected)),
            "first_start_index": int(selected[0]),
            "last_start_index": int(selected[-1]),
            "first_start_time": format_time(first_time),
            "last_start_time": format_time(last_time),
            "years": years,
        }
        self.logger.info("Evaluation start selection:")
        for key in [
            "split",
            "rollout_steps",
            "selection",
            "stride",
            "selected_initial_conditions",
            "first_start_time",
            "last_start_time",
        ]:
            self.logger.info("  %s: %s", key, self._selection_metadata.get(key))
        return selected

    def _bootstrap_confidence_intervals(
        self,
        per_ic_mse: np.ndarray,
        per_ic_acc: np.ndarray,
    ) -> dict[str, np.ndarray]:
        if int(self.cfg.bootstrap_samples) <= 0:
            return {}
        n_ics = int(per_ic_mse.shape[0])
        if n_ics <= 0:
            return {}
        alpha = 1.0 - float(self.cfg.confidence_level)
        lower_q = 100.0 * alpha / 2.0
        upper_q = 100.0 * (1.0 - alpha / 2.0)
        rng = np.random.default_rng(int(self.cfg.bootstrap_seed))
        rmse_samples = np.empty((int(self.cfg.bootstrap_samples), per_ic_mse.shape[1], per_ic_mse.shape[2]), dtype=np.float64)
        acc_samples = np.empty_like(rmse_samples)
        for sample_idx in range(int(self.cfg.bootstrap_samples)):
            picks = rng.integers(0, n_ics, size=n_ics)
            rmse_samples[sample_idx] = np.sqrt(np.nanmean(per_ic_mse[picks], axis=0))
            acc_samples[sample_idx] = np.nanmean(per_ic_acc[picks], axis=0)
        return {
            "rmse_ci_lower": np.nanpercentile(rmse_samples, lower_q, axis=0),
            "rmse_ci_upper": np.nanpercentile(rmse_samples, upper_q, axis=0),
            "acc_ci_lower": np.nanpercentile(acc_samples, lower_q, axis=0),
            "acc_ci_upper": np.nanpercentile(acc_samples, upper_q, axis=0),
        }

    def _aggregate_metrics(
        self,
        per_ic_mse_values: list[np.ndarray],
        per_ic_acc_values: list[np.ndarray],
    ) -> dict[str, np.ndarray]:
        per_ic_mse = np.asarray(per_ic_mse_values, dtype=np.float64)
        per_ic_acc = np.asarray(per_ic_acc_values, dtype=np.float64)
        weighted_rmse = np.sqrt(np.nanmean(per_ic_mse, axis=0))
        avg_acc = np.nanmean(per_ic_acc, axis=0)
        metrics = {
            "rmse": weighted_rmse,
            "acc": avg_acc,
            "per_ic_mse": per_ic_mse,
            "per_ic_rmse": np.sqrt(per_ic_mse),
            "per_ic_acc": per_ic_acc,
        }
        metrics.update(self._bootstrap_confidence_intervals(per_ic_mse, per_ic_acc))
        return metrics

    def _copy_metrics_with_backend(self, metrics: dict[str, Any], backend: str) -> dict[str, Any]:
        copied: dict[str, Any] = dict(metrics)
        copied["per_ic_mse"] = np.asarray(metrics["per_ic_mse"], dtype=np.float64)
        copied["per_ic_rmse"] = np.sqrt(copied["per_ic_mse"])
        if "per_ic_acc" in metrics:
            copied["per_ic_acc"] = np.asarray(metrics["per_ic_acc"], dtype=np.float64)
        copied["rmse_backend"] = backend
        return copied

    def _derive_current_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        current = self._copy_metrics_with_backend(metrics, "current")
        current["rmse_definition"] = "repository current evaluator aggregation"
        return current

    def _derive_weatherbench2_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        wb2 = self._copy_metrics_with_backend(metrics, "weatherbench2")
        per_ic_mse = np.asarray(wb2["per_ic_mse"], dtype=np.float64)
        wb2["rmse"] = rmse_from_per_ic_mse(per_ic_mse)
        wb2["acc"] = np.nanmean(np.asarray(metrics["per_ic_acc"], dtype=np.float64), axis=0)
        wb2["mean_per_ic_rmse_diagnostic"] = mean_per_ic_rmse(per_ic_mse)
        wb2["rmse_definition"] = (
            "WeatherBench2-compatible: latitude-weighted spatial MSE per IC, "
            "mean over ICs, then sqrt."
        )
        wb2.update(self._bootstrap_confidence_intervals(per_ic_mse, np.asarray(metrics["per_ic_acc"], dtype=np.float64)))
        return wb2

    def _summary_lead_indices(self, metrics: dict[str, Any]) -> list[int]:
        lead_times = list(metrics.get("lead_times", range(1, int(np.asarray(metrics["rmse"]).shape[0]) + 1)))
        include_lead0 = bool(getattr(self.cfg, "include_lead0_in_summary", False))
        indices: list[int] = []
        for idx, lead in enumerate(lead_times):
            if int(lead) == 0 and not include_lead0:
                continue
            indices.append(idx)
        return indices

    def _metric_lead_times(self, metrics: dict[str, Any]) -> list[int]:
        return [int(x) for x in metrics.get("lead_times", range(1, int(np.asarray(metrics["rmse"]).shape[0]) + 1))]

    def _difference_label(self, percent_delta: float | None) -> str:
        if percent_delta is None or not np.isfinite(percent_delta):
            return "undefined"
        magnitude = abs(float(percent_delta))
        if magnitude < 0.1:
            return "equivalent"
        if magnitude <= 1.0:
            return "small difference"
        return "important difference"

    def _evaluate_loaded_model_rollout(
        self,
        fields: Any,
        total_t: int,
        height: int,
        width: int,
        fixed_rollout_steps: int,
        climatology_norm: torch.Tensor,
        lat_weights: torch.Tensor,
        lat_weights_np: np.ndarray,
        label: str,
    ) -> dict[str, np.ndarray]:
        ics = self._selected_ics or self._build_ic_indices(total_t, fixed_rollout_steps)
        self.logger.info(
            "Evaluating %s with fixed rollout_steps=%d over %d initial conditions: first_ic=%s",
            label,
            fixed_rollout_steps,
            len(ics),
            ics[0] if ics else None,
        )
        self._reset_eval_target_override_debug(label, fixed_rollout_steps)
        self._reset_sensitivity_debug(label, fixed_rollout_steps)
        per_ic_mse_values = []
        per_ic_acc_values = []
        per_ic_loss_values = []
        for count, ic in enumerate(ics):
            debug_this_ic = self._should_debug_eval_target_override_ic()
            debug_sensitivity_this_ic = self._should_debug_sensitivity_ic()
            mse, acc, loss_by_lead = self._rollout_one_ic(
                fields,
                ic,
                fixed_rollout_steps,
                climatology_norm,
                lat_weights,
                lat_weights_np,
                debug_eval_target_override=debug_this_ic,
                debug_eval_sensitivity=debug_sensitivity_this_ic,
            )
            self._finish_debug_eval_target_override_ic(debug_this_ic)
            self._finish_debug_sensitivity_ic(debug_sensitivity_this_ic)
            per_ic_mse_values.append(mse)
            per_ic_acc_values.append(acc)
            if loss_by_lead is not None:
                per_ic_loss_values.append(loss_by_lead)
            if (count + 1) % 10 == 0 or (count + 1) == len(ics):
                self.logger.info("%s completed IC %d/%d (ic=%d)", label, count + 1, len(ics), ic)
        metrics = self._aggregate_metrics(per_ic_mse_values, per_ic_acc_values)
        if per_ic_loss_values:
            per_ic_loss = np.asarray(per_ic_loss_values, dtype=np.float64)
            loss_by_lead = np.nanmean(per_ic_loss, axis=0)
            handler = self.eval_target_override_handler
            excluded = list(handler.exclude_loss_channels.keys()) if handler is not None else []
            included_count = (
                int(self.eval_loss_channel_mask.detach().cpu().sum().item())
                if self.eval_loss_channel_mask is not None
                else int(len(self.cfg.out_channels))
            )
            metrics["aggregate_loss_by_lead"] = loss_by_lead
            metrics["aggregate_loss_mean"] = np.asarray(float(np.nanmean(loss_by_lead)), dtype=np.float64)
            metrics["aggregate_loss_name"] = "dynamic_only_normalized_mse"
            metrics["aggregate_loss_description"] = (
                "Latitude-weighted normalized MSE aggregated over non-excluded channels. "
                "When evaluation target override is enabled, overridden channels are excluded from this loss."
            )
            metrics["aggregate_loss_excluded_variables"] = excluded
            metrics["aggregate_loss_included_channel_count"] = included_count
        metrics["initial_condition_indices"] = np.asarray(ics, dtype=np.int64)
        self._write_eval_target_override_debug_report(Path(self.cfg.output_dir), label)
        self._write_sensitivity_debug_report(Path(self.cfg.output_dir))
        return metrics

    def _evaluate_persistence_rollout(
        self,
        fields: Any,
        total_t: int,
        height: int,
        width: int,
        fixed_rollout_steps: int,
        climatology_norm: torch.Tensor,
        lat_weights: torch.Tensor,
        lat_weights_np: np.ndarray,
    ) -> dict[str, np.ndarray]:
        ics = self._selected_ics or self._build_ic_indices(total_t, fixed_rollout_steps)
        self.logger.info(
            "Evaluating persistence with fixed rollout_steps=%d over %d initial conditions",
            fixed_rollout_steps,
            len(ics),
        )
        per_ic_mse_values = []
        per_ic_acc_values = []
        for ic in ics:
            mse, acc = self._rollout_one_ic_persistence(
                fields,
                ic,
                fixed_rollout_steps,
                climatology_norm,
                lat_weights,
                lat_weights_np,
                height,
                width,
            )
            per_ic_mse_values.append(mse)
            per_ic_acc_values.append(acc)
        metrics = self._aggregate_metrics(per_ic_mse_values, per_ic_acc_values)
        metrics["initial_condition_indices"] = np.asarray(ics, dtype=np.int64)
        return metrics

    @torch.no_grad()
    def accumulate_affine_calibration_statistics(
        self,
        fields: Any,
        total_t: int,
        height: int,
        width: int,
        fixed_rollout_steps: int,
        lat_weights: torch.Tensor,
        variables: list[dict[str, Any]],
        *,
        label: str = "affine calibration fit",
    ) -> dict[str, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Model must be loaded before accumulating affine calibration statistics.")
        variable_indices = [int(variable["variable_idx"]) for variable in variables]
        accumulator = empty_affine_accumulator(int(fixed_rollout_steps), len(variable_indices))
        ics = self._selected_ics or self._build_ic_indices(total_t, fixed_rollout_steps)
        self.logger.info(
            "Fitting affine calibration from %s with fixed rollout_steps=%d over %d initial conditions",
            label,
            fixed_rollout_steps,
            len(ics),
        )
        n_channels = len(self.cfg.out_channels)
        for count, ic in enumerate(ics):
            raw_prev = np.asarray(fields[ic - self.cfg.dt, :, :height, :width], dtype=np.float32)
            raw_cur = np.asarray(fields[ic, :, :height, :width], dtype=np.float32)
            previous = self._normalize_input_step(raw_prev)
            current = self._normalize_input_step(raw_cur)
            initial_state = current
            for step_idx in range(int(fixed_rollout_steps)):
                lead = step_idx + 1
                target_idx = ic + lead * self.cfg.dt
                raw_target = np.asarray(fields[target_idx, :, :height, :width], dtype=np.float32)
                target_norm = self._normalize_target_step(raw_target)
                aux = None
                if self.feature_builder is not None:
                    target_doy, target_days = self._target_time_tensors(target_idx)
                    aux = self.feature_builder.build_step_features(
                        current=current,
                        target_norm=target_norm,
                        target_dayofyear=target_doy,
                        target_days_in_year=target_days,
                        step_idx=0,
                    )
                pred_norm = self._forward_model_step(previous, current, aux, lead=lead)
                if self.feature_builder is not None:
                    pred_norm = self.feature_builder.apply_overrides(pred_norm, current=current, target_norm=target_norm)
                if self.target_handler is not None:
                    pred_norm = self.target_handler.apply(
                        pred_next=pred_norm,
                        current_state=current,
                        initial_state=initial_state,
                        target_norm=target_norm,
                        lead=lead,
                    )
                if self.eval_target_override_handler is not None:
                    pred_norm = self.eval_target_override_handler.apply(
                        pred_next=pred_norm,
                        current_state=current,
                        initial_state=initial_state,
                        target_norm=target_norm,
                        lead=lead,
                    )

                update_affine_accumulator(
                    accumulator,
                    lead_idx=step_idx,
                    pred_norm=pred_norm,
                    target_norm=target_norm,
                    lat_weights=lat_weights,
                    variable_indices=variable_indices,
                )

                next_current = current.clone()
                next_current[:, :n_channels] = pred_norm
                previous, current = current, next_current

            if (count + 1) % 10 == 0 or (count + 1) == len(ics):
                self.logger.info("%s completed IC %d/%d (ic=%d)", label, count + 1, len(ics), ic)
        accumulator["initial_condition_indices"] = np.asarray(ics, dtype=np.int64)
        return accumulator

    @torch.no_grad()
    def _rollout_one_ic(
        self,
        fields: Any,
        ic: int,
        forecast_steps: int,
        climatology_norm: torch.Tensor,
        lat_weights: torch.Tensor,
        lat_weights_np: np.ndarray,
        debug_eval_target_override: bool = False,
        debug_eval_sensitivity: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        if self.model is None:
            raise RuntimeError("Model must be loaded before rollout.")

        n_channels = len(self.cfg.out_channels)
        height = self.model.graph.L0.height
        width = self.model.graph.L0.width
        mse = np.zeros((forecast_steps, n_channels), dtype=np.float64)
        acc = np.zeros((forecast_steps, n_channels), dtype=np.float64)
        loss_by_lead = (
            np.zeros((forecast_steps,), dtype=np.float64) if self.eval_loss_channel_mask is not None else None
        )
        std_view = torch.as_tensor(self.out_stds, device=self.device, dtype=torch.float32).view(1, -1, 1, 1)
        weight = lat_weights_np.reshape(1, -1, 1)
        denom = float(lat_weights_np.sum() * width)
        loss_weight = lat_weights.to(device=self.device, dtype=torch.float32).view(1, 1, -1, 1)

        raw_prev = np.asarray(fields[ic - self.cfg.dt, :, :height, :width], dtype=np.float32)
        raw_cur = np.asarray(fields[ic, :, :height, :width], dtype=np.float32)
        previous = self._normalize_input_step(raw_prev)
        current = self._normalize_input_step(raw_cur)
        initial_state = current
        for step_idx in range(forecast_steps):
            lead = step_idx + 1
            target_idx = ic + lead * self.cfg.dt
            raw_target = np.asarray(fields[target_idx, :, :height, :width], dtype=np.float32)
            target_norm = self._normalize_target_step(raw_target)
            aux = None
            if self.feature_builder is not None:
                target_doy, target_days = self._target_time_tensors(target_idx)
                aux = self.feature_builder.build_step_features(
                    current=current,
                    target_norm=target_norm,
                    target_dayofyear=target_doy,
                    target_days_in_year=target_days,
                    step_idx=0,
                )
            pred_norm = self._forward_model_step(previous, current, aux, lead=lead)
            if self.feature_builder is not None:
                pred_norm = self.feature_builder.apply_overrides(pred_norm, current=current, target_norm=target_norm)
            if self.target_handler is not None:
                pred_norm = self.target_handler.apply(
                    pred_next=pred_norm,
                    current_state=current,
                    initial_state=initial_state,
                    target_norm=target_norm,
                    lead=lead,
                )
            if self.eval_target_override_handler is not None:
                pred_norm = self.eval_target_override_handler.apply(
                    pred_next=pred_norm,
                    current_state=current,
                    initial_state=initial_state,
                    target_norm=target_norm,
                    lead=lead,
                )
            if debug_eval_target_override:
                self._record_eval_target_override_debug(
                    pred_norm=pred_norm,
                    initial_state=initial_state,
                    target_norm=target_norm,
                    lead=lead,
                )
            pred_norm = self._apply_sensitivity_mode(
                pred_norm=pred_norm,
                current_state=current,
                initial_state=initial_state,
                target_norm=target_norm,
                lead=lead,
                ic=ic,
                debug=debug_eval_sensitivity,
            )

            pred_state_norm = pred_norm
            pred_metric_norm = pred_norm
            if self.affine_calibration is not None:
                if self.cfg.calibration_apply_mode == "autoregressive_state":
                    pred_state_norm = self._apply_affine_calibration(pred_norm, lead, in_place=False)
                    pred_metric_norm = pred_state_norm
                else:
                    pred_metric_norm = self._apply_affine_calibration(pred_norm, lead, in_place=False)

            err_phys = ((pred_metric_norm - target_norm) * std_view)[0].detach().cpu().double().numpy()
            mse[step_idx] = ((err_phys ** 2) * weight).sum(axis=(-2, -1)) / denom
            if loss_by_lead is not None and self.eval_loss_channel_mask is not None:
                mask = self.eval_loss_channel_mask.to(device=pred_metric_norm.device, dtype=pred_metric_norm.dtype).view(1, -1, 1, 1)
                err2_norm = (pred_metric_norm - target_norm).pow(2)
                loss_denom = (
                    pred_metric_norm.shape[0]
                    * pred_metric_norm.shape[-1]
                    * loss_weight.to(dtype=pred_metric_norm.dtype).sum().clamp_min(1.0e-8)
                    * mask.sum().clamp_min(1.0e-8)
                )
                loss_by_lead[step_idx] = float(
                    ((err2_norm * loss_weight.to(dtype=pred_metric_norm.dtype) * mask).sum() / loss_denom).detach().cpu().item()
                )

            clim = self._climatology_norm_for_day(self._target_dayofyear(target_idx))
            acc[step_idx] = weighted_acc_per_channel(pred_metric_norm - clim, target_norm - clim, lat_weights)[0].detach().cpu().numpy()

            next_current = current.clone()
            next_current[:, :n_channels] = pred_state_norm
            previous, current = current, next_current

        return mse, acc, loss_by_lead

    @torch.no_grad()
    def _rollout_one_ic_persistence(
        self,
        fields: Any,
        ic: int,
        forecast_steps: int,
        climatology_norm: torch.Tensor,
        lat_weights: torch.Tensor,
        lat_weights_np: np.ndarray,
        height: int,
        width: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_channels = len(self.cfg.out_channels)
        mse = np.zeros((forecast_steps, n_channels), dtype=np.float64)
        acc = np.zeros((forecast_steps, n_channels), dtype=np.float64)
        std_view = torch.as_tensor(self.out_stds, device=self.device, dtype=torch.float32).view(1, -1, 1, 1)
        weight = lat_weights_np.reshape(1, -1, 1)
        denom = float(lat_weights_np.sum() * width)

        raw_cur = np.asarray(fields[ic, :, :height, :width], dtype=np.float32)
        pred_norm = self._normalize_target_step(raw_cur)

        for step_idx in range(forecast_steps):
            lead = step_idx + 1
            target_idx = ic + lead * self.cfg.dt
            raw_target = np.asarray(fields[target_idx, :, :height, :width], dtype=np.float32)
            target_norm = self._normalize_target_step(raw_target)

            err_phys = ((pred_norm - target_norm) * std_view)[0].detach().cpu().double().numpy()
            mse[step_idx] = ((err_phys ** 2) * weight).sum(axis=(-2, -1)) / denom

            clim = self._climatology_norm_for_day(self._target_dayofyear(target_idx))
            acc[step_idx] = weighted_acc_per_channel(pred_norm - clim, target_norm - clim, lat_weights)[0].detach().cpu().numpy()

        return mse, acc

    def _save_metrics(
        self,
        metrics: dict[str, np.ndarray],
        forecast_steps: int,
        output_dir: Path,
    ) -> list[dict[str, Any]]:
        output_dir.mkdir(parents=True, exist_ok=True)
        rmse = metrics["rmse"]
        acc = metrics["acc"]
        rmse_ci_lower = metrics.get("rmse_ci_lower")
        rmse_ci_upper = metrics.get("rmse_ci_upper")
        acc_ci_lower = metrics.get("acc_ci_lower")
        acc_ci_upper = metrics.get("acc_ci_upper")
        rows: list[dict[str, Any]] = []
        for lead_idx in range(rmse.shape[0]):
            lead_time = lead_idx + 1
            for var_idx in range(rmse.shape[1]):
                row = {
                    "rollout_steps": forecast_steps,
                    "lead_time": lead_time,
                    "variable_idx": var_idx,
                    "original_channel_idx": int(self.cfg.out_channels[var_idx]),
                    "variable_name": self.variable_names[var_idx],
                    "rmse": float(rmse[lead_idx, var_idx]),
                    "acc": float(acc[lead_idx, var_idx]),
                }
                if rmse_ci_lower is not None:
                    row.update(
                        {
                            "rmse_ci_lower": float(rmse_ci_lower[lead_idx, var_idx]),
                            "rmse_ci_upper": float(rmse_ci_upper[lead_idx, var_idx]),
                            "acc_ci_lower": float(acc_ci_lower[lead_idx, var_idx]),
                            "acc_ci_upper": float(acc_ci_upper[lead_idx, var_idx]),
                        }
                    )
                rows.append(row)

        with open(output_dir / "evaluation_metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        rmse_header = ["lead_time"] + self.variable_names
        with open(output_dir / "rollout_rmse.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(rmse_header)
            for lead_idx in range(rmse.shape[0]):
                writer.writerow([lead_idx + 1] + [float(x) for x in rmse[lead_idx]])

        acc_header = ["lead_time"] + self.variable_names
        with open(output_dir / "rollout_acc.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(acc_header)
            for lead_idx in range(acc.shape[0]):
                writer.writerow([lead_idx + 1] + [float(x) for x in acc[lead_idx]])

        self.logger.info("Saved metrics: %s", output_dir / "evaluation_metrics.csv")
        self.logger.info("Saved RMSE: %s", output_dir / "rollout_rmse.csv")
        self.logger.info("Saved ACC: %s", output_dir / "rollout_acc.csv")
        self._save_aggregate_loss(metrics, forecast_steps, output_dir)
        self._save_override_adjusted_note(output_dir)
        return rows

    def _selected_variable_rows(self) -> list[tuple[int, str]]:
        rows: list[tuple[int, str]] = []
        for var_idx in self._json_variable_indices():
            rows.append((int(var_idx), self._variable_key(int(var_idx))))
        return rows

    def _metrics_for_output(self, metrics: dict[str, Any]) -> tuple[list[int], np.ndarray, np.ndarray]:
        rmse = np.asarray(metrics["rmse"], dtype=np.float64)
        acc = np.asarray(metrics["acc"], dtype=np.float64)
        lead_times = self._metric_lead_times(metrics)
        if bool(getattr(self.cfg, "save_lead0", False)):
            rmse = np.concatenate([np.zeros((1, rmse.shape[1]), dtype=np.float64), rmse], axis=0)
            acc = np.concatenate([np.ones((1, acc.shape[1]), dtype=np.float64), acc], axis=0)
            lead_times = [0] + lead_times
        return lead_times, rmse, acc

    def _valid_time_strings(self, ics: np.ndarray, lead_times: list[int]) -> np.ndarray:
        if not self._time_values:
            return np.asarray([], dtype="<U1")
        values = np.empty((len(ics), len(lead_times)), dtype=object)
        for ic_pos, ic in enumerate(np.asarray(ics, dtype=np.int64)):
            for lead_pos, lead in enumerate(lead_times):
                target_idx = int(ic) + int(lead) * int(self.cfg.dt)
                values[ic_pos, lead_pos] = (
                    format_time(self._time_values[target_idx])
                    if 0 <= target_idx < len(self._time_values)
                    else None
                )
        return values.astype(str)

    def _save_weatherbench2_per_ic_npz(
        self,
        output_root: Path,
        metrics: dict[str, Any],
        variable_indices: list[int],
    ) -> Path:
        ics = np.asarray(metrics.get("initial_condition_indices", []), dtype=np.int64)
        lead_times = self._metric_lead_times(metrics)
        per_ic_mse = np.asarray(metrics["per_ic_mse"], dtype=np.float64)[:, :, variable_indices]
        per_ic_rmse = np.sqrt(per_ic_mse)
        per_ic_acc = np.asarray(metrics["per_ic_acc"], dtype=np.float64)[:, :, variable_indices]
        if bool(getattr(self.cfg, "save_lead0", False)):
            zeros = np.zeros((per_ic_mse.shape[0], 1, per_ic_mse.shape[2]), dtype=np.float64)
            ones = np.ones((per_ic_acc.shape[0], 1, per_ic_acc.shape[2]), dtype=np.float64)
            per_ic_mse = np.concatenate([zeros, per_ic_mse], axis=1)
            per_ic_rmse = np.concatenate([zeros, per_ic_rmse], axis=1)
            per_ic_acc = np.concatenate([ones, per_ic_acc], axis=1)
            lead_times = [0] + lead_times
        variables = np.asarray([self._variable_key(idx) for idx in variable_indices], dtype=str)
        out_channels = np.asarray([int(self.cfg.out_channels[idx]) for idx in variable_indices], dtype=np.int64)
        path = output_root / "per_ic_metrics.npz"
        np.savez_compressed(
            path,
            per_ic_mse=per_ic_mse,
            per_ic_rmse=per_ic_rmse,
            per_ic_acc=per_ic_acc,
            ics=ics,
            valid_times=self._valid_time_strings(ics, lead_times),
            variables=variables,
            out_channels=out_channels,
            lead_times=np.asarray(lead_times, dtype=np.int64),
        )
        self.logger.info("Saved WeatherBench2 per-IC metrics: %s", path)
        return path

    def _save_weatherbench2_outputs(
        self,
        output_root: Path,
        fixed_rollout_steps: int,
        metrics: dict[str, Any],
        persistence: dict[str, Any] | None,
        checkpoint_metadata: dict[str, Any],
        graph_label: str,
        checkpoint_path: str | None = None,
    ) -> None:
        output_root.mkdir(parents=True, exist_ok=True)
        variable_rows = self._selected_variable_rows()
        variable_indices = [idx for idx, _ in variable_rows]
        lead_times, rmse_out, acc_out = self._metrics_for_output(metrics)

        long_rows: list[dict[str, Any]] = []
        for lead_pos, lead_time in enumerate(lead_times):
            for var_idx, variable in variable_rows:
                long_rows.append(
                    {
                        "lead_time": int(lead_time),
                        "variable": variable,
                        "rmse": float(rmse_out[lead_pos, var_idx]),
                        "acc": float(acc_out[lead_pos, var_idx]),
                    }
                )
        metrics_csv = output_root / "weatherbench2_evaluation_metrics.csv"
        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["lead_time", "variable", "rmse", "acc"])
            writer.writeheader()
            writer.writerows(long_rows)

        rmse_csv = output_root / "weatherbench2_rollout_rmse.csv"
        acc_csv = output_root / "weatherbench2_rollout_acc.csv"
        header = ["lead_time"] + [variable for _, variable in variable_rows]
        with open(rmse_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for lead_pos, lead_time in enumerate(lead_times):
                writer.writerow([int(lead_time)] + [float(rmse_out[lead_pos, idx]) for idx, _ in variable_rows])
        with open(acc_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for lead_pos, lead_time in enumerate(lead_times):
                writer.writerow([int(lead_time)] + [float(acc_out[lead_pos, idx]) for idx, _ in variable_rows])

        per_ic_path = self._save_weatherbench2_per_ic_npz(output_root, metrics, variable_indices)
        include_lead0 = bool(getattr(self.cfg, "include_lead0_in_summary", False))
        summary_indices = [
            idx
            for idx, lead_time in enumerate(lead_times)
            if int(lead_time) != 0 or include_lead0
        ]
        summary_leads = [int(lead_times[idx]) for idx in summary_indices]
        summary_payload = {
            "rmse_backend": "weatherbench2",
            "rmse_definition": metrics.get("rmse_definition"),
            "lead_times": self._metric_lead_times(metrics),
            "summary_lead_times": summary_leads,
            "lead0_note": "Lead 0 = initial condition, not forecast" if bool(getattr(self.cfg, "save_lead0", False)) else None,
                "include_lead0_in_summary": include_lead0,
            "selection": self._selection_metadata,
            "climatology": self._climatology_metadata,
            "bootstrap": {
                "samples": int(self.cfg.bootstrap_samples),
                "confidence_level": float(self.cfg.confidence_level),
                "seed": int(self.cfg.bootstrap_seed),
            },
            "checkpoint": {
                "label": graph_label,
                "epoch": checkpoint_metadata.get("epoch"),
                "train_rollout_steps": checkpoint_metadata.get("train_rollout_steps"),
                "params_m": self._graphweather_params_from_metadata(checkpoint_metadata),
            },
            "per_ic_metrics": str(per_ic_path),
            "variables": {},
        }
        for var_idx, variable in variable_rows:
            rmse = np.asarray(metrics["rmse"][:, var_idx], dtype=np.float64)
            acc = np.asarray(metrics["acc"][:, var_idx], dtype=np.float64)
            entry = {
                "variable_name": self.variable_names[var_idx],
                "channel": int(self.cfg.out_channels[var_idx]),
                "rmse_by_lead": _json_series(rmse),
                "acc_by_lead": _json_series(acc),
                "summary_rmse_mean": _json_number(np.nanmean(rmse_out[summary_indices, var_idx])) if summary_indices else None,
                "summary_acc_mean": _json_number(np.nanmean(acc_out[summary_indices, var_idx])) if summary_indices else None,
                f"day{fixed_rollout_steps}_rmse": _json_number(rmse[fixed_rollout_steps - 1]),
                f"day{fixed_rollout_steps}_acc": _json_number(acc[fixed_rollout_steps - 1]),
            }
            if "rmse_ci_lower" in metrics:
                entry["rmse_ci_lower"] = _json_series(metrics["rmse_ci_lower"][:, var_idx])
                entry["rmse_ci_upper"] = _json_series(metrics["rmse_ci_upper"][:, var_idx])
                entry["acc_ci_lower"] = _json_series(metrics["acc_ci_lower"][:, var_idx])
                entry["acc_ci_upper"] = _json_series(metrics["acc_ci_upper"][:, var_idx])
            if persistence is not None:
                entry["persistence_rmse_by_lead"] = _json_series(persistence["rmse"][:, var_idx])
                entry["persistence_acc_by_lead"] = _json_series(persistence["acc"][:, var_idx])
            summary_payload["variables"][variable] = entry

        summary_json = output_root / "weatherbench2_summary.json"
        summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        if bool(getattr(self.cfg, "save_lead0", False)):
            (output_root / "weatherbench2_lead0_note.txt").write_text(
                "Lead 0 = initial condition, not forecast. It is excluded from summaries unless "
                "--include_lead0_in_summary is set.\n",
                encoding="utf-8",
            )

        self.logger.info("Saved WeatherBench2 metrics CSV: %s", metrics_csv)
        self.logger.info("Saved WeatherBench2 RMSE CSV: %s", rmse_csv)
        self.logger.info("Saved WeatherBench2 ACC CSV: %s", acc_csv)
        self.logger.info("Saved WeatherBench2 summary JSON: %s", summary_json)

        table_rows = self._weatherbench2_table_rows(
            variable_rows=variable_rows,
            lead_times=lead_times,
            rmse_out=rmse_out,
            acc_out=acc_out,
            checkpoint_path=checkpoint_path,
        )
        stepwise_rows = self._weatherbench2_stepwise_rows(table_rows)
        self._append_weatherbench2_local_log(table_rows)
        self._append_weatherbench2_stepwise_local_log(stepwise_rows)
        self._log_weatherbench2_wandb(
            table_rows=table_rows,
            stepwise_rows=stepwise_rows,
            variable_rows=variable_rows,
            lead_times=lead_times,
            rmse_out=rmse_out,
            acc_out=acc_out,
            summary_indices=summary_indices,
        )

        self._write_weatherbench2_vs_kai(output_root, fixed_rollout_steps, metrics, checkpoint_metadata)
        self._plot_weatherbench2_outputs(output_root, fixed_rollout_steps, metrics, persistence, graph_label)

    def _weatherbench2_table_rows(
        self,
        *,
        variable_rows: list[tuple[int, str]],
        lead_times: list[int],
        rmse_out: np.ndarray,
        acc_out: np.ndarray,
        checkpoint_path: str | None,
    ) -> list[dict[str, Any]]:
        run_name = str((self.cfg.wandb or {}).get("run_name") or Path(self.cfg.output_dir).name)
        checkpoint = checkpoint_path or self.cfg.checkpoint_path
        rows: list[dict[str, Any]] = []
        for lead_pos, lead_time in enumerate(lead_times):
            for var_idx, variable in variable_rows:
                rows.append(
                    {
                        "run_name": run_name,
                        "checkpoint": checkpoint,
                        "variable": variable,
                        "lead": int(lead_time),
                        "rmse": float(rmse_out[lead_pos, var_idx]),
                        "acc": float(acc_out[lead_pos, var_idx]),
                        "rmse_backend": "weatherbench2",
                    }
                )
        return rows

    def _append_weatherbench2_local_log(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        logs_dir = Path(self.cfg.output_dir) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        path = logs_dir / "weatherbench_rmse_acc.csv"
        exists = path.exists()
        fields = ["run_name", "checkpoint", "variable", "lead", "rmse", "acc", "rmse_backend"]
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerows({field: row.get(field) for field in fields} for row in rows)

    @staticmethod
    def _baseline_improvement_pct(delta: float | None, baseline: float | None) -> float:
        if delta is None or baseline is None:
            return float("nan")
        if not np.isfinite(float(delta)) or not np.isfinite(float(baseline)) or abs(float(baseline)) <= 1.0e-12:
            return float("nan")
        return float(float(delta) / float(baseline))

    def _weatherbench2_stepwise_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        stepwise_rows: list[dict[str, Any]] = []
        for row in rows:
            variable = str(row.get("variable"))
            lead = int(row.get("lead", 0))
            rmse = float(row.get("rmse", float("nan")))
            acc = float(row.get("acc", float("nan")))
            baseline_rmse = float("nan")
            baseline_acc = float("nan")
            for baseline in getattr(self, "external_baselines", []) or []:
                curve = baseline.curves.get(variable)
                if curve is None or lead < 1 or lead > len(curve.get("rmse", [])):
                    continue
                candidate_rmse = float(curve["rmse"][lead - 1])
                candidate_acc = float(curve["acc"][lead - 1])
                if np.isfinite(candidate_rmse) or np.isfinite(candidate_acc):
                    baseline_rmse = candidate_rmse
                    baseline_acc = candidate_acc
                    break
            rmse_delta = float(baseline_rmse - rmse) if np.isfinite(baseline_rmse) and np.isfinite(rmse) else float("nan")
            acc_delta = float(acc - baseline_acc) if np.isfinite(baseline_acc) and np.isfinite(acc) else float("nan")
            stepwise_rows.append(
                {
                    "run_name": row.get("run_name"),
                    "epoch_or_checkpoint": row.get("checkpoint"),
                    "variable": variable,
                    "lead": lead,
                    "rmse": rmse,
                    "acc": acc,
                    "baseline_rmse": baseline_rmse,
                    "baseline_acc": baseline_acc,
                    "rmse_improvement_abs": rmse_delta,
                    "rmse_improvement_pct": self._baseline_improvement_pct(rmse_delta, baseline_rmse),
                    "acc_improvement_abs": acc_delta,
                    "acc_improvement_pct": self._baseline_improvement_pct(acc_delta, baseline_acc),
                }
            )
        return stepwise_rows

    def _append_weatherbench2_stepwise_local_log(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        logs_dir = Path(self.cfg.output_dir) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        path = logs_dir / "weatherbench_stepwise.csv"
        exists = path.exists()
        fields = [
            "run_name",
            "epoch_or_checkpoint",
            "variable",
            "lead",
            "rmse",
            "acc",
            "baseline_rmse",
            "baseline_acc",
            "rmse_improvement_abs",
            "rmse_improvement_pct",
            "acc_improvement_abs",
            "acc_improvement_pct",
        ]
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerows({field: row.get(field) for field in fields} for row in rows)

    def _log_weatherbench2_wandb(
        self,
        *,
        table_rows: list[dict[str, Any]],
        stepwise_rows: list[dict[str, Any]],
        variable_rows: list[tuple[int, str]],
        lead_times: list[int],
        rmse_out: np.ndarray,
        acc_out: np.ndarray,
        summary_indices: list[int],
    ) -> None:
        if getattr(self, "wandb_run", None) is None:
            return
        payload: dict[str, Any] = {}
        for lead_pos, lead_time in enumerate(lead_times):
            for var_idx, variable in variable_rows:
                payload[f"weatherbench/rmse/{variable}/lead_{int(lead_time)}"] = float(rmse_out[lead_pos, var_idx])
                payload[f"weatherbench/acc/{variable}/lead_{int(lead_time)}"] = float(acc_out[lead_pos, var_idx])
                payload[f"weatherbench_stepwise/rmse/{variable}/lead_{int(lead_time):02d}"] = float(rmse_out[lead_pos, var_idx])
                payload[f"weatherbench_stepwise/acc/{variable}/lead_{int(lead_time):02d}"] = float(acc_out[lead_pos, var_idx])
        for var_idx, variable in variable_rows:
            if summary_indices:
                payload[f"weatherbench/summary/{variable}/rmse_mean"] = float(np.nanmean(rmse_out[summary_indices, var_idx]))
                payload[f"weatherbench/summary/{variable}/acc_mean"] = float(np.nanmean(acc_out[summary_indices, var_idx]))
        for row in stepwise_rows:
            variable = str(row.get("variable"))
            lead = int(row.get("lead", 0))
            if np.isfinite(float(row.get("rmse_improvement_pct", float("nan")))):
                payload[f"weatherbench_stepwise/rmse/{variable}/lead_{lead:02d}/improvement_pct_baseline"] = float(
                    row["rmse_improvement_pct"]
                )
                payload[f"weatherbench_stepwise/rmse/{variable}/lead_{lead:02d}/improvement_abs_baseline"] = float(
                    row["rmse_improvement_abs"]
                )
            if np.isfinite(float(row.get("acc_improvement_pct", float("nan")))):
                payload[f"weatherbench_stepwise/acc/{variable}/lead_{lead:02d}/improvement_pct_baseline"] = float(
                    row["acc_improvement_pct"]
                )
                payload[f"weatherbench_stepwise/acc/{variable}/lead_{lead:02d}/improvement_abs_baseline"] = float(
                    row["acc_improvement_abs"]
                )
        columns = ["run_name", "checkpoint", "variable", "lead", "rmse", "acc", "rmse_backend"]
        table = self._wandb_table(table_rows, columns)
        if table is not None:
            payload["tables/weatherbench/rmse_by_lead"] = table
            payload["tables/weatherbench/acc_by_lead"] = table
        stepwise_columns = [
            "run_name",
            "epoch_or_checkpoint",
            "variable",
            "lead",
            "rmse",
            "acc",
            "baseline_rmse",
            "baseline_acc",
            "rmse_improvement_abs",
            "rmse_improvement_pct",
            "acc_improvement_abs",
            "acc_improvement_pct",
        ]
        stepwise_table = self._wandb_table(stepwise_rows, stepwise_columns)
        if stepwise_table is not None:
            payload["tables/weatherbench_stepwise"] = stepwise_table
        self._wandb_log(payload, step=int(self.cfg.eval_fixed_rollout_steps))

    def _write_backend_comparison(
        self,
        output_root: Path,
        current_metrics: dict[str, Any],
        weatherbench2_metrics: dict[str, Any],
    ) -> None:
        variable_rows = self._selected_variable_rows()
        lead_times = self._metric_lead_times(weatherbench2_metrics)
        rows: list[dict[str, Any]] = []
        for lead_pos, lead_time in enumerate(lead_times):
            for var_idx, variable in variable_rows:
                current_rmse = float(current_metrics["rmse"][lead_pos, var_idx])
                wb2_rmse = float(weatherbench2_metrics["rmse"][lead_pos, var_idx])
                rmse_delta = wb2_rmse - current_rmse
                rmse_percent_delta = None if current_rmse == 0.0 else 100.0 * rmse_delta / current_rmse
                current_acc = float(current_metrics["acc"][lead_pos, var_idx])
                wb2_acc = float(weatherbench2_metrics["acc"][lead_pos, var_idx])
                rows.append(
                    {
                        "variable": variable,
                        "lead_time": int(lead_time),
                        "current_rmse": current_rmse,
                        "weatherbench2_rmse": wb2_rmse,
                        "rmse_delta": rmse_delta,
                        "rmse_percent_delta": rmse_percent_delta,
                        "difference_label": self._difference_label(rmse_percent_delta),
                        "current_acc": current_acc,
                        "weatherbench2_acc": wb2_acc,
                    }
                )

        csv_path = output_root / "rmse_backend_comparison.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        payload = {
            "thresholds": {
                "equivalent": "difference < 0.1%",
                "small difference": "0.1%-1%",
                "important difference": ">1%",
            },
            "rows": rows,
        }
        json_path = output_root / "rmse_backend_comparison.json"
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        counts: dict[str, int] = {}
        for row in rows:
            counts[row["difference_label"]] = counts.get(row["difference_label"], 0) + 1
        lines = [
            "RMSE backend comparison:",
            "- current_rmse: repository current evaluator output",
            "- weatherbench2_rmse: sqrt(mean per-IC latitude-weighted MSE)",
            "",
            "Difference counts:",
        ]
        lines.extend(f"- {label}: {count}" for label, count in sorted(counts.items()))
        txt_path = output_root / "rmse_backend_comparison.txt"
        txt_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        self.logger.info("Saved RMSE backend comparison CSV: %s", csv_path)
        self.logger.info("Saved RMSE backend comparison JSON: %s", json_path)
        self.logger.info("Saved RMSE backend comparison text: %s", txt_path)

    def _write_weatherbench2_vs_kai(
        self,
        output_root: Path,
        fixed_rollout_steps: int,
        metrics: dict[str, Any],
        checkpoint_metadata: dict[str, Any],
    ) -> None:
        if not self.external_baselines:
            return
        rows: list[dict[str, Any]] = []
        day_rows: list[dict[str, Any]] = []
        lines = [
            "WeatherBench2 vs external baseline comparison",
            EXTERNAL_BASELINE_FAIRNESS_NOTE,
            "",
        ]
        lead_times = self._metric_lead_times(metrics)
        for baseline in self.external_baselines:
            lines.append(f"Compared with {baseline.label}:")
            for var_idx, variable in self._selected_variable_rows():
                curve = baseline.curves.get(variable)
                if curve is None:
                    continue
                for lead_pos, lead_time in enumerate(lead_times):
                    if lead_time < 1 or lead_time > len(curve["rmse"]):
                        continue
                    kai_pos = int(lead_time) - 1
                    kai_rmse = float(curve["rmse"][kai_pos])
                    kai_acc = float(curve["acc"][kai_pos])
                    if not np.isfinite(kai_rmse) or not np.isfinite(kai_acc):
                        continue
                    model_rmse = float(metrics["rmse"][lead_pos, var_idx])
                    model_acc = float(metrics["acc"][lead_pos, var_idx])
                    rmse_delta = model_rmse - kai_rmse
                    row = {
                        "baseline_label": baseline.label,
                        "variable": variable,
                        "lead_time": int(lead_time),
                        "model_rmse": model_rmse,
                        "kai_rmse": kai_rmse,
                        "rmse_delta": rmse_delta,
                        "rmse_percent_delta": None if kai_rmse == 0.0 else 100.0 * rmse_delta / kai_rmse,
                        "model_acc": model_acc,
                        "kai_acc": kai_acc,
                        "acc_delta": model_acc - kai_acc,
                    }
                    rows.append(row)
                    if int(lead_time) == int(fixed_rollout_steps):
                        day_rows.append(row)
                if variable in baseline.curves and len(baseline.curves[variable]["rmse"]) >= fixed_rollout_steps:
                    model_day_rmse = float(metrics["rmse"][fixed_rollout_steps - 1, var_idx])
                    kai_day_rmse = float(baseline.curves[variable]["rmse"][fixed_rollout_steps - 1])
                    model_day_acc = float(metrics["acc"][fixed_rollout_steps - 1, var_idx])
                    kai_day_acc = float(baseline.curves[variable]["acc"][fixed_rollout_steps - 1])
                    if np.isfinite(kai_day_rmse) and np.isfinite(kai_day_acc):
                        rmse_winner = baseline.label if kai_day_rmse < model_day_rmse else "model"
                        acc_winner = "model" if model_day_acc > kai_day_acc else baseline.label
                        lines.append(
                            f"- {variable}: day-{fixed_rollout_steps} lower RMSE = {rmse_winner}; "
                            f"higher ACC = {acc_winner}."
                        )
            lines.append("")
        if not rows:
            return

        csv_path = output_root / "weatherbench2_vs_kai_summary.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        json_path = output_root / "weatherbench2_vs_kai_summary.json"
        json_path.write_text(
            json.dumps(
                {
                    "rmse_backend": "weatherbench2",
                    "checkpoint": {
                        "epoch": checkpoint_metadata.get("epoch"),
                        "train_rollout_steps": checkpoint_metadata.get("train_rollout_steps"),
                    },
                    "rows": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        txt_path = output_root / "weatherbench2_vs_kai_report.txt"
        txt_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        day_path = output_root / "weatherbench2_vs_kai_day10.csv"
        with open(day_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(day_rows)
        self.logger.info("Saved WeatherBench2 vs Kai CSV: %s", csv_path)
        self.logger.info("Saved WeatherBench2 vs Kai JSON: %s", json_path)
        self.logger.info("Saved WeatherBench2 vs Kai report: %s", txt_path)
        self.logger.info("Saved WeatherBench2 vs Kai day-10 CSV: %s", day_path)

    def _plot_weatherbench2_outputs(
        self,
        output_root: Path,
        fixed_rollout_steps: int,
        metrics: dict[str, Any],
        persistence: dict[str, Any] | None,
        graph_label: str,
    ) -> None:
        variable_rows = self._selected_variable_rows()
        if not variable_rows:
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "matplotlib is required for WeatherBench2 plots. Install requirements.txt or disable plotting."
            ) from exc

        plot_dir = output_root / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        fmt = self.cfg.plot_format.lstrip(".").lower()
        lead_times = np.asarray(self._metric_lead_times(metrics), dtype=np.int64)
        labeled_metrics: dict[str, dict[str, Any]] = {graph_label: metrics}
        if persistence is not None:
            labeled_metrics["persistence"] = persistence
        labeled_metrics.update(self._external_metric_arrays([idx for idx, _ in variable_rows]))

        def plot_one(variable_idx: int, variable: str, metric_name: str) -> None:
            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            for label, item in labeled_metrics.items():
                style = item.get("_style")
                if style == "external":
                    values = item[metric_name][: len(lead_times), variable_idx]
                    linestyle, marker = "-.", "s"
                elif str(label).lower() == "persistence":
                    values = item[metric_name][:, variable_idx]
                    linestyle, marker = "--", "x"
                else:
                    values = item[metric_name][:, variable_idx]
                    linestyle, marker = "-", "o"
                ax.plot(lead_times, values, linestyle=linestyle, marker=marker, linewidth=1.8, label=label)
            ax.set_title(f"{variable} WeatherBench2 {metric_name.upper()} by lead")
            ax.set_xlabel("Lead time")
            ax.set_ylabel(metric_name.upper())
            if metric_name == "acc":
                ax.set_ylim(-1.05, 1.05)
            ax.set_xticks(lead_times)
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            path = plot_dir / f"wb2_{metric_name}_vs_lead_{self._safe_plot_stem(variable, variable_idx)}.{fmt}"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            self.logger.info("Saved WeatherBench2 plot: %s", path)

        for var_idx, variable in variable_rows:
            plot_one(var_idx, variable, "rmse")
            plot_one(var_idx, variable, "acc")

        labels = list(labeled_metrics)
        x = np.arange(len(variable_rows), dtype=np.float64)
        width = min(0.8 / max(len(labels), 1), 0.24)
        for metric_name in ("rmse", "acc"):
            fig, ax = plt.subplots(figsize=(10.5, 5.4))
            for label_idx, label in enumerate(labels):
                item = labeled_metrics[label]
                values = []
                for var_idx, _ in variable_rows:
                    arr = np.asarray(item[metric_name], dtype=np.float64)
                    values.append(float(arr[fixed_rollout_steps - 1, var_idx]))
                offset = (label_idx - (len(labels) - 1) / 2.0) * width
                ax.bar(x + offset, values, width=width, label=label)
            ax.set_xticks(x)
            ax.set_xticklabels([variable for _, variable in variable_rows])
            ax.set_ylabel(f"Day-{fixed_rollout_steps} {metric_name.upper()}")
            ax.set_title(f"WeatherBench2 day-{fixed_rollout_steps} {metric_name.upper()}")
            ax.grid(True, axis="y", alpha=0.25)
            ax.legend()
            fig.tight_layout()
            path = plot_dir / f"wb2_day10_{metric_name}_bar.{fmt}"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            self.logger.info("Saved WeatherBench2 bar plot: %s", path)

        fig, axes = plt.subplots(len(variable_rows), 2, figsize=(14.0, max(5.0, 2.7 * len(variable_rows))), squeeze=False)
        for row, (var_idx, variable) in enumerate(variable_rows):
            for label, item in labeled_metrics.items():
                style = item.get("_style")
                linestyle, marker = ("-.", "s") if style == "external" else ("--", "x") if str(label).lower() == "persistence" else ("-", "o")
                axes[row, 0].plot(lead_times, item["rmse"][: len(lead_times), var_idx], linestyle=linestyle, marker=marker, linewidth=1.7, label=label)
                axes[row, 1].plot(lead_times, item["acc"][: len(lead_times), var_idx], linestyle=linestyle, marker=marker, linewidth=1.7, label=label)
            axes[row, 0].set_title(f"{variable} RMSE")
            axes[row, 1].set_title(f"{variable} ACC")
            axes[row, 0].set_ylabel("RMSE")
            axes[row, 1].set_ylabel("ACC")
            axes[row, 1].set_ylim(-1.05, 1.05)
            axes[row, 0].grid(True, alpha=0.3)
            axes[row, 1].grid(True, alpha=0.3)
            if row == len(variable_rows) - 1:
                axes[row, 0].set_xlabel("Lead time")
                axes[row, 1].set_xlabel("Lead time")
        handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="upper center", ncol=min(3, len(legend_labels)))
        fig.suptitle("WeatherBench2 RMSE/ACC comparison", fontsize=14)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
        path = plot_dir / f"wb2_vs_kai_rmse_acc_all_variables.{fmt}"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        self.logger.info("Saved WeatherBench2 all-variable plot: %s", path)

    def _save_aggregate_loss(self, metrics: dict[str, Any], forecast_steps: int, output_dir: Path) -> None:
        if "aggregate_loss_by_lead" not in metrics:
            return
        loss_by_lead = np.asarray(metrics["aggregate_loss_by_lead"], dtype=np.float64)
        rows = [
            {
                "rollout_steps": int(forecast_steps),
                "lead_time": int(lead_idx + 1),
                "loss_name": str(metrics.get("aggregate_loss_name", "aggregate_loss")),
                "normalized_mse_loss": float(loss_by_lead[lead_idx]),
                "excluded_variables": " ".join(str(x) for x in metrics.get("aggregate_loss_excluded_variables", [])),
                "included_channel_count": int(metrics.get("aggregate_loss_included_channel_count", 0)),
            }
            for lead_idx in range(loss_by_lead.shape[0])
        ]
        csv_path = output_dir / "dynamic_only_loss.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        payload = self._loss_payload_for_json(metrics) or {}
        json_path = output_dir / "dynamic_only_loss.json"
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.logger.info("Saved dynamic-only aggregate loss CSV: %s", csv_path)
        self.logger.info("Saved dynamic-only aggregate loss JSON: %s", json_path)

    def _save_override_adjusted_note(self, output_dir: Path) -> None:
        if not bool(getattr(self.cfg, "eval_target_override", False)):
            return
        note = (
            "Evaluation target override is enabled.\n"
            "All-channel rollout_rmse.csv, rollout_acc.csv, and evaluation_metrics.csv are override-adjusted.\n"
            "Metrics for overridden variables are diagnostic only and should not be interpreted as model skill.\n"
            "Use dynamic_only_loss.csv for aggregate loss excluding configured override variables.\n"
        )
        path = output_dir / "target_override_metrics_note.txt"
        path.write_text(note, encoding="utf-8")
        self.logger.info("Saved target override metrics note: %s", path)

    def _resolve_plot_variables(self) -> list[int]:
        requested = self.cfg.plot_variables
        if not requested:
            return []
        if any(str(item).lower() == "all" for item in requested):
            return list(range(len(self.cfg.out_channels)))

        lower_names = {_norm_name(name): idx for idx, name in enumerate(self.variable_names)}
        resolved: list[int] = []
        for item in requested:
            token = str(item).strip()
            if not token:
                continue

            var_idx: Optional[int] = None
            if token.lstrip("-").isdigit():
                value = int(token)
                if 0 <= value < len(self.cfg.out_channels):
                    var_idx = value
                elif value in self.cfg.out_channels:
                    var_idx = self.cfg.out_channels.index(value)
            else:
                candidates = [_norm_name(token)]
                canonical = _canonical_for_name(token)
                if canonical:
                    candidates.extend(_norm_name(alias) for alias in VARIABLE_ALIASES.get(canonical, []))
                for candidate in candidates:
                    if candidate in lower_names:
                        var_idx = lower_names[candidate]
                        break

            if var_idx is None:
                self.logger.warning(
                    "Could not resolve plot variable '%s'. Use variable name, local variable_idx, original channel index, or 'all'.",
                    token,
                )
                continue
            if var_idx not in resolved:
                resolved.append(var_idx)
        return resolved

    def _variable_key(self, var_idx: int) -> str:
        name = self.variable_names[var_idx]
        return _canonical_for_name(name) or name

    def _external_metric_arrays(self, variable_indices: list[int]) -> dict[str, dict[str, Any]]:
        if not self.external_baselines:
            return {}
        fixed_steps = int(self.cfg.eval_fixed_rollout_steps)
        n_vars = len(self.cfg.out_channels)
        labeled: dict[str, dict[str, Any]] = {}
        for baseline in self.external_baselines:
            metrics: dict[str, Any] = {
                "rmse": np.full((fixed_steps, n_vars), np.nan, dtype=np.float64),
                "acc": np.full((fixed_steps, n_vars), np.nan, dtype=np.float64),
                "_style": "external",
                "_params_m": baseline.params_m,
                "_source_csv": baseline.source_csv,
            }
            has_ci = any(
                key in curve
                for curve in baseline.curves.values()
                for key in ("rmse_ci_lower", "rmse_ci_upper", "acc_ci_lower", "acc_ci_upper")
            )
            if has_ci:
                metrics["rmse_ci_lower"] = np.full((fixed_steps, n_vars), np.nan, dtype=np.float64)
                metrics["rmse_ci_upper"] = np.full((fixed_steps, n_vars), np.nan, dtype=np.float64)
                metrics["acc_ci_lower"] = np.full((fixed_steps, n_vars), np.nan, dtype=np.float64)
                metrics["acc_ci_upper"] = np.full((fixed_steps, n_vars), np.nan, dtype=np.float64)
            for var_idx in variable_indices:
                key = self._variable_key(var_idx)
                curve = baseline.curves.get(key)
                if curve is None:
                    self.logger.warning(
                        "WARNING: external baseline %s does not contain variable %s. Skipping %s external curve.",
                        baseline.label,
                        key,
                        key,
                    )
                    print(f"WARNING: external baseline {baseline.label} does not contain variable {key}. Skipping {key} external curve.")
                    continue
                metrics["rmse"][:, var_idx] = curve["rmse"]
                metrics["acc"][:, var_idx] = curve["acc"]
                for ci_key in ("rmse_ci_lower", "rmse_ci_upper", "acc_ci_lower", "acc_ci_upper"):
                    if ci_key in curve and ci_key in metrics:
                        metrics[ci_key][:, var_idx] = curve[ci_key]
            labeled[baseline.label] = metrics
        return labeled

    def _external_baselines_json(self, variable_indices: list[int]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for baseline in self.external_baselines:
            baseline_payload: dict[str, Any] = {
                "params_m": baseline.params_m,
                "source_csv": baseline.source_csv,
                "rmse": {},
                "acc": {},
            }
            for var_idx in variable_indices:
                key = self._variable_key(var_idx)
                curve = baseline.curves.get(key)
                if curve is None:
                    continue
                baseline_payload["rmse"][key] = _json_series(curve["rmse"])
                baseline_payload["acc"][key] = _json_series(curve["acc"])
                if "rmse_ci_lower" in curve:
                    baseline_payload.setdefault("rmse_lower", {})[key] = _json_series(curve["rmse_ci_lower"])
                    baseline_payload.setdefault("rmse_upper", {})[key] = _json_series(curve["rmse_ci_upper"])
                if "acc_ci_lower" in curve:
                    baseline_payload.setdefault("acc_lower", {})[key] = _json_series(curve["acc_ci_lower"])
                    baseline_payload.setdefault("acc_upper", {})[key] = _json_series(curve["acc_ci_upper"])
            payload[baseline.label] = baseline_payload
        return payload

    def _safe_plot_stem(self, variable_name: str, variable_idx: int) -> str:
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in variable_name)
        return safe or f"var{variable_idx:02d}"

    def _plot_comparison_variables(
        self,
        labeled_metrics: dict[str, dict[str, np.ndarray]],
        output_root: Path,
        filename_prefix: str,
        title_prefix: str,
    ) -> None:
        variable_indices = self._resolve_plot_variables()
        if not variable_indices:
            return

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "matplotlib is required for --plot_variables. Install requirements.txt or remove the plotting option."
            ) from exc

        plot_dir = output_root / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        fmt = self.cfg.plot_format.lstrip(".").lower()

        for var_idx in variable_indices:
            variable_name = self.variable_names[var_idx]
            original_channel = self.cfg.out_channels[var_idx]
            fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
            model_rmse_curves = []

            for label, metrics in labeled_metrics.items():
                rmse = metrics["rmse"][:, var_idx]
                acc = metrics["acc"][:, var_idx]
                leads = np.arange(1, rmse.shape[0] + 1)
                style = metrics.get("_style")
                if style == "external":
                    linestyle = "-."
                    marker = "s"
                elif str(label).lower() == "persistence":
                    linestyle = "--"
                    marker = "x"
                else:
                    linestyle = "-"
                    marker = "o"
                axes[0].plot(leads, rmse, marker=marker, linestyle=linestyle, linewidth=1.8, label=label)
                axes[1].plot(leads, acc, marker=marker, linestyle=linestyle, linewidth=1.8, label=label)
                should_shade = "rmse_ci_lower" in metrics and (
                    self.cfg.plot_confidence_intervals or style == "external"
                )
                if should_shade:
                    axes[0].fill_between(
                        leads,
                        metrics["rmse_ci_lower"][:, var_idx],
                        metrics["rmse_ci_upper"][:, var_idx],
                        alpha=0.16,
                    )
                    axes[1].fill_between(
                        leads,
                        metrics["acc_ci_lower"][:, var_idx],
                        metrics["acc_ci_upper"][:, var_idx],
                        alpha=0.16,
                    )
                if label != "persistence" and style != "external":
                    model_rmse_curves.append((label, rmse.copy()))

            if len(model_rmse_curves) > 1:
                first = model_rmse_curves[0][1]
                if all(np.array_equal(first, curve) for _, curve in model_rmse_curves[1:]):
                    self.logger.warning(
                        "WARNING: all stage evaluation curves are identical for %s. "
                        "Check whether the same checkpoint or cached rollout result is being reused.",
                        variable_name,
                    )

            title = f"{variable_name} (variable_idx={var_idx}, channel={original_channel})"
            comparison_label = " vs ".join(str(label) for label in labeled_metrics)
            axes[0].set_title(f"{title_prefix} RMSE - {title}\n{comparison_label}")
            axes[0].set_ylabel("RMSE")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()

            axes[1].set_title(f"{title_prefix} ACC - {title}")
            axes[1].set_xlabel("Lead time (days)")
            axes[1].set_ylabel("ACC")
            axes[1].set_ylim(-1.05, 1.05)
            axes[1].set_xticks(np.arange(1, labeled_metrics[next(iter(labeled_metrics))]["rmse"].shape[0] + 1))
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()

            fig.tight_layout()
            output_path = plot_dir / f"{filename_prefix}_{self._safe_plot_stem(variable_name, var_idx)}_rmse_acc.{fmt}"
            fig.savefig(output_path, dpi=160)
            plt.close(fig)
            self.logger.info("Saved rollout plot for %s: %s", variable_name, output_path)

    def _prepare_context(self) -> tuple[Any, Any, int, int, int, torch.Tensor, torch.Tensor, np.ndarray]:
        files = self._find_nc_files(self.cfg.test_dataset_path)
        self.logger.info("Found %d test files.", len(files))
        if len(files) > 1:
            if bool(self.cfg.first_test_file_only):
                self.logger.info("Using first file only because --first_test_file_only was set.")
            else:
                self.logger.info("Evaluating all files.")
        test_file = files[0]
        self.logger.info("Running evaluation on: %s", test_file)

        ds = self.nc.Dataset(test_file, "r")
        fields = ds["fields"]
        total_t, _, height, width = fields.shape
        expected_shape = tuple(self.cfg.grid_shape)
        if (int(height), int(width)) != expected_shape:
            raise ValueError(
                f"Expected {self.cfg.resolution_mode} grid shape {expected_shape}, "
                f"received {(int(height), int(width))}. Check resolution_mode and dataset path."
        )
        self._load_variable_names(ds)
        self._load_affine_calibration_if_needed()
        try:
            self._time_values = decode_time_values(ds)
            if len(self._time_values) != int(total_t):
                raise ValueError(f"Decoded time length {len(self._time_values)} does not match fields length {total_t}.")
        except (KeyError, TypeError, ValueError) as exc:
            self._time_values = []
            self._warn_climatology_time_fallback()
            self.logger.warning("Time coordinate decode failed: %s", exc)
        if "latitude" in ds.variables:
            latitudes = np.asarray(ds["latitude"][:height], dtype=np.float64)
        elif "lat" in ds.variables:
            latitudes = np.asarray(ds["lat"][:height], dtype=np.float64)
        else:
            latitudes = cell_center_lat_lon(height, width)[0].numpy().astype(np.float64)

        climatology = self._load_or_build_climatology(height, width)
        climatology_norm = self._prepare_climatology_lookup(climatology)
        self._selected_ics = self._build_ic_indices(total_t, int(self.cfg.eval_fixed_rollout_steps))
        years = sorted({int(date_from_time(t).year) for t in self._time_values}) if self._time_values else []
        self._selection_metadata.update(
            {
                "file": test_file,
                "file_years": years,
                "calendar_aware": bool(self._time_values),
            }
        )
        self.logger.info("  file/year: %s / %s", Path(test_file).name, ",".join(str(year) for year in years))
        lat_weights_np = _latitude_weights(latitudes)
        lat_weights = torch.as_tensor(lat_weights_np, device=self.device, dtype=torch.float32)
        return ds, fields, total_t, height, width, climatology_norm, lat_weights, lat_weights_np

    def _metric_series_for_json(self, metrics: dict[str, np.ndarray], metric_name: str, var_idx: int) -> dict[str, Any]:
        series = {
            "mean": [float(x) for x in metrics[metric_name][:, var_idx]],
        }
        lower_key = f"{metric_name}_ci_lower"
        upper_key = f"{metric_name}_ci_upper"
        if lower_key in metrics and upper_key in metrics:
            series["ci_lower"] = [float(x) for x in metrics[lower_key][:, var_idx]]
            series["ci_upper"] = [float(x) for x in metrics[upper_key][:, var_idx]]
        return series

    def _metrics_for_json(self, metrics: dict[str, np.ndarray], variable_indices: list[int]) -> dict[str, dict[str, Any]]:
        payload: dict[str, dict[str, Any]] = {}
        for var_idx in variable_indices:
            key = self._variable_key(var_idx)
            payload[key] = {
                "name": self.variable_names[var_idx],
                "variable_idx": int(var_idx),
                "channel": int(self.cfg.out_channels[var_idx]),
                "rmse": self._metric_series_for_json(metrics, "rmse", var_idx),
                "acc": self._metric_series_for_json(metrics, "acc", var_idx),
            }
        return payload

    def _json_variable_indices(self) -> list[int]:
        requested = self._resolve_plot_variables()
        return requested if requested else list(range(len(self.cfg.out_channels)))

    def _write_fixed_json(
        self,
        output_path: Path,
        fixed_rollout_steps: int,
        checkpoint_payloads: dict[str, dict[str, Any]],
        external_baselines: dict[str, Any] | None = None,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "eval_fixed_rollout_steps": fixed_rollout_steps,
            "lead_times": list(range(1, fixed_rollout_steps + 1)),
            "selection": self._selection_metadata,
            "climatology": self._climatology_metadata,
            "bootstrap": {
                "samples": int(self.cfg.bootstrap_samples),
                "confidence_level": float(self.cfg.confidence_level),
                "seed": int(self.cfg.bootstrap_seed),
            },
            "features": self.feature_builder.metadata if self.feature_builder is not None else {"extra_features_enabled": False},
            "evaluation_target_override": self._eval_target_override_payload(),
            "orog_tisr_sensitivity": self._sensitivity_payload(),
            "affine_calibration": {
                "enabled": self.affine_calibration is not None,
                "path": None if self.affine_calibration is None else self.affine_calibration.path,
                "apply_mode": None if self.affine_calibration is None else self.cfg.calibration_apply_mode,
                "variables": [] if self.affine_calibration is None else self.affine_calibration.variables,
                "leads": [] if self.affine_calibration is None else self.affine_calibration.leads,
            },
            "checkpoints": checkpoint_payloads,
            "external_baselines": external_baselines or {},
            "external_baseline_fairness_note": EXTERNAL_BASELINE_FAIRNESS_NOTE if external_baselines else None,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self.logger.info("Saved fixed-rollout comparison JSON: %s", output_path)

    def _write_global_summary_csv(
        self,
        output_root: Path,
        fixed_rollout_steps: int,
        metrics: dict[str, np.ndarray],
        persistence: dict[str, np.ndarray] | None,
        checkpoint_metadata: dict[str, Any],
    ) -> Path:
        rows: list[dict[str, Any]] = []
        variable_indices = self._json_variable_indices()
        for var_idx in variable_indices:
            model_rmse_day = float(metrics["rmse"][fixed_rollout_steps - 1, var_idx])
            model_acc_day = float(metrics["acc"][fixed_rollout_steps - 1, var_idx])
            row = {
                "checkpoint_epoch": checkpoint_metadata.get("epoch"),
                "train_rollout_steps": checkpoint_metadata.get("train_rollout_steps"),
                "selected_initial_conditions": self._selection_metadata.get("selected_initial_conditions"),
                "first_start_time": self._selection_metadata.get("first_start_time"),
                "last_start_time": self._selection_metadata.get("last_start_time"),
                "variable": self._variable_key(var_idx),
                "variable_name": self.variable_names[var_idx],
                "channel": int(self.cfg.out_channels[var_idx]),
                "rmse_avg_1_10": float(np.nanmean(metrics["rmse"][:fixed_rollout_steps, var_idx])),
                "rmse_day10": model_rmse_day,
                "acc_avg_1_10": float(np.nanmean(metrics["acc"][:fixed_rollout_steps, var_idx])),
                "acc_day10": model_acc_day,
            }
            if "rmse_ci_lower" in metrics:
                row.update(
                    {
                        "rmse_day10_ci_lower": float(metrics["rmse_ci_lower"][fixed_rollout_steps - 1, var_idx]),
                        "rmse_day10_ci_upper": float(metrics["rmse_ci_upper"][fixed_rollout_steps - 1, var_idx]),
                        "acc_day10_ci_lower": float(metrics["acc_ci_lower"][fixed_rollout_steps - 1, var_idx]),
                        "acc_day10_ci_upper": float(metrics["acc_ci_upper"][fixed_rollout_steps - 1, var_idx]),
                    }
                )
            if persistence is not None:
                pers_rmse = float(persistence["rmse"][fixed_rollout_steps - 1, var_idx])
                pers_acc = float(persistence["acc"][fixed_rollout_steps - 1, var_idx])
                row.update(
                    {
                        "persistence_rmse_day10": pers_rmse,
                        "persistence_acc_day10": pers_acc,
                        "rmse_skill_vs_persistence_day10": float(1.0 - model_rmse_day / pers_rmse) if pers_rmse > 0 else None,
                        "acc_improvement_vs_persistence_day10": float(model_acc_day - pers_acc),
                    }
                )
            rows.append(row)

        path = output_root / f"fixed{fixed_rollout_steps}_global_best_summary.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        self.logger.info("Saved global-best summary CSV: %s", path)
        return path

    def _write_global_summary_text(
        self,
        output_root: Path,
        fixed_rollout_steps: int,
        metrics: dict[str, np.ndarray],
        persistence: dict[str, np.ndarray] | None,
        checkpoint_metadata: dict[str, Any],
    ) -> Path:
        lines = [
            "Global best checkpoint evaluation:",
            f"- checkpoint epoch: {checkpoint_metadata.get('epoch')}",
            f"- training rollout stage: {checkpoint_metadata.get('train_rollout_steps')}",
            f"- number of initial conditions: {self._selection_metadata.get('selected_initial_conditions')}",
            f"- date range: {self._selection_metadata.get('first_start_time')} to {self._selection_metadata.get('last_start_time')}",
            f"- climatology: {self._climatology_metadata.get('climatology_path') or self._climatology_metadata.get('path')}",
            "",
            f"Day-{fixed_rollout_steps} metrics:",
        ]
        for var_idx in self._json_variable_indices():
            key = self._variable_key(var_idx)
            rmse_day = float(metrics["rmse"][fixed_rollout_steps - 1, var_idx])
            acc_day = float(metrics["acc"][fixed_rollout_steps - 1, var_idx])
            line = f"- {key}: RMSE {rmse_day:.6g}, ACC {acc_day:.6g}"
            if persistence is not None:
                pers_rmse = float(persistence["rmse"][fixed_rollout_steps - 1, var_idx])
                pers_acc = float(persistence["acc"][fixed_rollout_steps - 1, var_idx])
                skill = float(1.0 - rmse_day / pers_rmse) if pers_rmse > 0 else float("nan")
                line += f" | persistence RMSE {pers_rmse:.6g}, ACC {pers_acc:.6g}, RMSE skill {skill:.3%}"
            lines.append(line)
        text = "\n".join(lines)
        path = output_root / f"fixed{fixed_rollout_steps}_global_best_summary.txt"
        path.write_text(text + "\n", encoding="utf-8")
        self.logger.info("Saved global-best summary text: %s", path)
        print(text)
        return path

    def _graphweather_params_from_metadata(self, checkpoint_metadata: dict[str, Any]) -> float | None:
        for key in ("params_m", "model_params_m", "parameter_count_m", "num_params_m"):
            if checkpoint_metadata.get(key) is not None:
                return float(checkpoint_metadata[key])
        for key in ("params", "model_params", "parameter_count", "num_params"):
            if checkpoint_metadata.get(key) is not None:
                return float(checkpoint_metadata[key]) / 1.0e6
        return self._graphweather_params_m

    def _graphweather_label(self, checkpoint_metadata: dict[str, Any]) -> str:
        params = self._graphweather_params_from_metadata(checkpoint_metadata)
        if params is None:
            return "GraphWeather"
        return f"GraphWeather {params:.2f}M"

    def _write_external_comparison_summary(
        self,
        output_root: Path,
        fixed_rollout_steps: int,
        metrics: dict[str, np.ndarray],
        checkpoint_metadata: dict[str, Any],
    ) -> None:
        if not self.external_baselines:
            return
        graph_params = self._graphweather_params_from_metadata(checkpoint_metadata)
        graph_name = self._graphweather_label(checkpoint_metadata)
        rows: list[dict[str, Any]] = []
        lines = [EXTERNAL_BASELINE_FAIRNESS_NOTE, ""]
        for baseline in self.external_baselines:
            lines.append(f"Compared with {baseline.label}:")
            for var_idx in self._json_variable_indices():
                key = self._variable_key(var_idx)
                curve = baseline.curves.get(key)
                if curve is None:
                    continue
                model_rmse = np.asarray(metrics["rmse"][:fixed_rollout_steps, var_idx], dtype=np.float64)
                model_acc = np.asarray(metrics["acc"][:fixed_rollout_steps, var_idx], dtype=np.float64)
                ext_rmse = np.asarray(curve["rmse"][:fixed_rollout_steps], dtype=np.float64)
                ext_acc = np.asarray(curve["acc"][:fixed_rollout_steps], dtype=np.float64)
                if not np.isfinite(ext_rmse[-1]) or not np.isfinite(ext_acc[-1]):
                    self.logger.warning(
                        "External baseline %s variable %s lacks finite day-%d metrics; skipping comparison row.",
                        baseline.label,
                        key,
                        fixed_rollout_steps,
                    )
                    continue
                day10_rmse_diff = float(model_rmse[-1] - ext_rmse[-1])
                day10_rmse_percent = float(100.0 * day10_rmse_diff / ext_rmse[-1]) if ext_rmse[-1] != 0 else None
                day10_acc_delta = float(model_acc[-1] - ext_acc[-1])
                rows.append(
                    {
                        "variable": key,
                        "model_name": graph_name,
                        "params_m": graph_params,
                        "avg_rmse_1_10": float(np.nanmean(model_rmse)),
                        "day10_rmse": float(model_rmse[-1]),
                        "avg_acc_1_10": float(np.nanmean(model_acc)),
                        "day10_acc": float(model_acc[-1]),
                        "rmse_day10_vs_external": day10_rmse_diff,
                        "rmse_day10_percent_vs_external": day10_rmse_percent,
                        "acc_day10_vs_external": day10_acc_delta,
                        "acc_day10_delta_vs_external": day10_acc_delta,
                    }
                )
                rows.append(
                    {
                        "variable": key,
                        "model_name": baseline.label,
                        "params_m": baseline.params_m,
                        "avg_rmse_1_10": float(np.nanmean(ext_rmse)),
                        "day10_rmse": float(ext_rmse[-1]),
                        "avg_acc_1_10": float(np.nanmean(ext_acc)),
                        "day10_acc": float(ext_acc[-1]),
                        "rmse_day10_vs_external": 0.0,
                        "rmse_day10_percent_vs_external": 0.0,
                        "acc_day10_vs_external": 0.0,
                        "acc_day10_delta_vs_external": 0.0,
                    }
                )
                rmse_winner = baseline.label if ext_rmse[-1] < model_rmse[-1] else "GraphWeather"
                acc_winner = "GraphWeather" if model_acc[-1] > ext_acc[-1] else baseline.label
                lines.append(
                    f"- {key}: {rmse_winner} has lower day-{fixed_rollout_steps} RMSE, "
                    f"{acc_winner} has higher day-{fixed_rollout_steps} ACC."
                )
            lines.append("")
        if not rows:
            return
        csv_path = output_root / "external_baseline_comparison_summary.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        txt_path = output_root / "external_baseline_comparison_summary.txt"
        text = "\n".join(lines).rstrip() + "\n"
        txt_path.write_text(text, encoding="utf-8")
        self.logger.info("Saved external baseline comparison CSV: %s", csv_path)
        self.logger.info("Saved external baseline comparison text: %s", txt_path)
        print(text)

    def _stage_checkpoint_items(self) -> list[tuple[str, str]]:
        items: list[tuple[str, str]] = []
        for stage, path in sorted(self.cfg.eval_stage_checkpoints.items()):
            resolved = self._resolve_checkpoint_path(path)
            if not os.path.exists(resolved):
                self.logger.warning("Skipping missing stage checkpoint S=%d: %s", stage, resolved)
                continue
            items.append((f"trained S={stage}", resolved))

        global_path = self.cfg.eval_global_best_checkpoint
        if global_path:
            resolved = self._resolve_checkpoint_path(global_path)
            if os.path.exists(resolved):
                items.append(("global best", resolved))
            else:
                self.logger.warning("Skipping missing global best checkpoint: %s", resolved)
        return items

    def evaluate_single_checkpoint(self) -> dict[str, np.ndarray]:
        fixed_steps = int(self.cfg.eval_fixed_rollout_steps)
        output_root = Path(self.cfg.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        ds, fields, total_t, height, width, climatology_norm, lat_weights, lat_weights_np = self._prepare_context()
        try:
            checkpoint_path = self._resolve_checkpoint_path(self.cfg.checkpoint_path)
            metadata = self._checkpoint_metadata(checkpoint_path)
            self.model = self._load_model(height, width, checkpoint_path)
            graph_label = self._graphweather_label(metadata)
            variable_indices = self._json_variable_indices()
            metrics = self._evaluate_loaded_model_rollout(
                fields,
                total_t,
                height,
                width,
                fixed_steps,
                climatology_norm,
                lat_weights,
                lat_weights_np,
                "global best",
            )
            current_metrics = self._derive_current_metrics(metrics)
            weatherbench2_metrics = self._derive_weatherbench2_metrics(metrics)
            primary_metrics = weatherbench2_metrics if self.cfg.rmse_backend == "weatherbench2" else current_metrics

            self._log_summary(fixed_steps, primary_metrics)
            if self.cfg.rmse_backend in {"current", "both"}:
                self._save_metrics(current_metrics, fixed_steps, output_root / f"S{fixed_steps}")

            labeled_metrics = {graph_label: current_metrics}
            persistence = None
            persistence_wb2 = None
            checkpoint_payloads = {
                "global_best": {
                    "label": graph_label,
                    "checkpoint_path": checkpoint_path,
                    "epoch": metadata.get("epoch"),
                    "train_rollout_steps": metadata.get("train_rollout_steps"),
                    "params_m": self._graphweather_params_from_metadata(metadata),
                    "metrics": self._metrics_for_json(current_metrics, variable_indices),
                    "aggregate_loss": self._loss_payload_for_json(current_metrics),
                }
            }
            if self.cfg.plot_persistence:
                persistence_raw = self._evaluate_persistence_rollout(
                    fields,
                    total_t,
                    height,
                    width,
                    fixed_steps,
                    climatology_norm,
                    lat_weights,
                    lat_weights_np,
                )
                persistence = self._derive_current_metrics(persistence_raw)
                persistence_wb2 = self._derive_weatherbench2_metrics(persistence_raw)
                labeled_metrics["persistence"] = persistence
                checkpoint_payloads["persistence"] = {
                    "label": "persistence",
                    "checkpoint_path": None,
                    "epoch": None,
                    "train_rollout_steps": None,
                    "metrics": self._metrics_for_json(persistence, variable_indices),
                    "aggregate_loss": None,
                }
            if self.cfg.rmse_backend in {"current", "both"}:
                external_metrics = self._external_metric_arrays(variable_indices)
                labeled_metrics.update(external_metrics)
                external_json = self._external_baselines_json(variable_indices)
                self._write_fixed_json(
                    output_root / f"fixed{fixed_steps}_global_best_metrics.json",
                    fixed_steps,
                    checkpoint_payloads,
                    external_baselines=external_json,
                )
                self._write_global_summary_csv(output_root, fixed_steps, current_metrics, persistence, metadata)
                self._write_global_summary_text(output_root, fixed_steps, current_metrics, persistence, metadata)
                self._write_external_comparison_summary(output_root, fixed_steps, current_metrics, metadata)
                self._plot_comparison_variables(
                    labeled_metrics,
                    output_root,
                    f"fixed{fixed_steps}_global_best",
                    f"Fixed {fixed_steps}-day rollout evaluation",
                )
            if self.cfg.rmse_backend in {"weatherbench2", "both"}:
                self._save_weatherbench2_outputs(
                    output_root,
                    fixed_steps,
                    weatherbench2_metrics,
                    persistence_wb2,
                    metadata,
                    graph_label,
                    checkpoint_path=checkpoint_path,
                )
            if self.cfg.rmse_backend == "both":
                self._write_backend_comparison(output_root, current_metrics, weatherbench2_metrics)
            return primary_metrics
        finally:
            ds.close()

    def evaluate_stage_checkpoints(self) -> dict[str, dict[str, np.ndarray]]:
        fixed_steps = int(self.cfg.eval_fixed_rollout_steps)
        output_root = Path(self.cfg.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        ds, fields, total_t, height, width, climatology_norm, lat_weights, lat_weights_np = self._prepare_context()
        try:
            labeled_metrics: dict[str, dict[str, np.ndarray]] = {}
            checkpoint_payloads: dict[str, dict[str, Any]] = {}
            variable_indices = self._json_variable_indices()
            checkpoint_items = self._stage_checkpoint_items()
            if not checkpoint_items:
                raise FileNotFoundError(
                    "No stage/global checkpoints were found for stage comparison. "
                    "Check experiment_dir and eval_stage_checkpoints."
                )
            for label, checkpoint_path in checkpoint_items:
                metadata = self._checkpoint_metadata(checkpoint_path)
                self.logger.info(
                    "Evaluating %s checkpoint %s from epoch %s with fixed rollout_steps=%d",
                    label,
                    checkpoint_path,
                    metadata.get("epoch", "unknown"),
                    fixed_steps,
                )
                self.model = self._load_model(height, width, checkpoint_path)
                raw_metrics = self._evaluate_loaded_model_rollout(
                    fields,
                    total_t,
                    height,
                    width,
                    fixed_steps,
                    climatology_norm,
                    lat_weights,
                    lat_weights_np,
                    label,
                )
                current_metrics = self._derive_current_metrics(raw_metrics)
                weatherbench2_metrics = self._derive_weatherbench2_metrics(raw_metrics)
                metrics = weatherbench2_metrics if self.cfg.rmse_backend == "weatherbench2" else current_metrics
                labeled_metrics[label] = metrics
                safe_label = label.replace(" ", "_").replace("=", "")
                if self.cfg.rmse_backend in {"current", "both"}:
                    self._save_metrics(current_metrics, fixed_steps, output_root / safe_label)
                if self.cfg.rmse_backend in {"weatherbench2", "both"}:
                    self._save_weatherbench2_outputs(
                        output_root / f"{safe_label}_weatherbench2",
                        fixed_steps,
                        weatherbench2_metrics,
                        None,
                        metadata,
                        label,
                        checkpoint_path=checkpoint_path,
                    )
                if self.cfg.rmse_backend == "both":
                    self._write_backend_comparison(output_root / f"{safe_label}_weatherbench2", current_metrics, weatherbench2_metrics)
                checkpoint_payloads[safe_label] = {
                    "label": label,
                    "checkpoint_path": checkpoint_path,
                    "epoch": metadata.get("epoch"),
                    "train_rollout_steps": metadata.get("train_rollout_steps"),
                    "metrics": self._metrics_for_json(metrics, variable_indices),
                    "aggregate_loss": self._loss_payload_for_json(metrics),
                }

            if self.cfg.plot_persistence:
                persistence_raw = self._evaluate_persistence_rollout(
                    fields,
                    total_t,
                    height,
                    width,
                    fixed_steps,
                    climatology_norm,
                    lat_weights,
                    lat_weights_np,
                )
                persistence = (
                    self._derive_weatherbench2_metrics(persistence_raw)
                    if self.cfg.rmse_backend == "weatherbench2"
                    else self._derive_current_metrics(persistence_raw)
                )
                labeled_metrics["persistence"] = persistence
                checkpoint_payloads["persistence"] = {
                    "label": "persistence",
                    "checkpoint_path": None,
                    "epoch": None,
                    "train_rollout_steps": None,
                    "metrics": self._metrics_for_json(persistence, variable_indices),
                    "aggregate_loss": None,
                }
            external_metrics = self._external_metric_arrays(variable_indices)
            labeled_metrics.update(external_metrics)
            external_json = self._external_baselines_json(variable_indices)

            self._write_fixed_json(
                output_root / f"fixed{fixed_steps}_stage_comparison_metrics.json",
                fixed_steps,
                checkpoint_payloads,
                external_baselines=external_json,
            )
            self._plot_comparison_variables(
                labeled_metrics,
                output_root,
                f"fixed{fixed_steps}_stage_comparison",
                "Fixed 10-day rollout evaluation by training stage" if fixed_steps == 10 else f"Fixed {fixed_steps}-day rollout evaluation by training stage",
            )
            return labeled_metrics
        finally:
            ds.close()

    def evaluate(self) -> dict[str, Any]:
        if self.cfg.eval_compare_stage_checkpoints:
            return self.evaluate_stage_checkpoints()
        return {"global best": self.evaluate_single_checkpoint()}

    def _log_summary(self, forecast_steps: int, metrics: dict[str, np.ndarray]) -> None:
        self.logger.info("=" * 80)
        title = "FIXED ROLLOUT EVALUATION SUMMARY"
        if self.cfg.eval_target_override:
            title += " (override-adjusted all-channel metrics)"
        self.logger.info("%s steps=%d", title, forecast_steps)
        self.logger.info("=" * 80)
        rmse = metrics["rmse"]
        acc = metrics["acc"]
        for lead in range(rmse.shape[0]):
            self.logger.info(
                "rollout_steps=%d lead %02d | mean RMSE %.6f | mean ACC %.6f",
                forecast_steps,
                lead + 1,
                float(np.mean(rmse[lead])),
                float(np.mean(acc[lead])),
            )
        if "aggregate_loss_by_lead" in metrics:
            self.logger.info(
                "Dynamic-only aggregate loss mean: %.6f | excluded variables: %s",
                float(metrics.get("aggregate_loss_mean", float("nan"))),
                ", ".join(str(x) for x in metrics.get("aggregate_loss_excluded_variables", [])),
            )


def run_evaluation_from_params(params: Any, logger: logging.Logger | Any = logging) -> dict[int, dict[str, np.ndarray]]:
    cfg = EvalConfig.from_params(params)
    evaluator = GraphWeatherEvaluator(cfg, logger=logger)
    return evaluator.evaluate()
