from __future__ import annotations

import csv
import json
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

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
from .graph_bundle import load_graph_bundle
from .models import GraphWeatherModel
from .resolution import cell_center_lat_lon


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
    "q700": ["q700", "q_700", "specific_humidity_700"],
    "u850": ["u850", "u_850", "u_component_wind_850"],
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
    required = {"variable", "timestep", "rmse", "acc"}
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
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"External baseline CSV {path} is missing required columns: {missing}")
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
    l1_refine_blocks: int
    l0_refine_blocks: int
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
    use_delta_normalization: bool
    delta_stats_path: Optional[str]
    delta_norm_center: bool
    delta_norm_eps: float

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

        return cls(
            resolution_mode=str(_get(params, "resolution_mode", "5p625")),
            grid_shape=tuple(int(x) for x in _get(params, "grid_shape", [32, 64])),
            test_dataset_path=str(test_path),
            output_dir=str(output_dir),
            checkpoint_path=str(checkpoint),
            experiment_dir=experiment_dir,
            graph_path=str(_get(params, "graph_path", "")),
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
            hidden_dim=int(_get(params, "hidden_dim", 96)),
            edge_dim=int(_get(params, "edge_dim", 6)),
            num_heads=int(_get(params, "num_heads", 4)),
            encoder_blocks=int(_get(params, "encoder_blocks", 1)),
            decoder_blocks=int(_get(params, "decoder_blocks", 1)),
            l0_blocks=int(_get(params, "l0_blocks", 2)),
            l1_blocks=int(_get(params, "l1_blocks", 2)),
            l2_blocks=int(_get(params, "l2_blocks", 1)),
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
            use_delta_normalization=bool(_get(params, "use_delta_normalization", False)),
            delta_stats_path=_get(params, "delta_stats_path", None),
            delta_norm_center=bool(_get(params, "delta_norm_center", False)),
            delta_norm_eps=float(_get(params, "delta_norm_eps", 1.0e-6)),
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


class GraphWeatherEvaluator:
    def __init__(self, cfg: EvalConfig, logger: logging.Logger | Any = logging):
        cfg.validate()
        self.cfg = cfg
        self.logger = logger
        self.nc = _load_netcdf4()
        self.device = torch.device(cfg.device if cfg.device != "cuda" or torch.cuda.is_available() else "cpu")
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
        self.orography_field: Optional[np.ndarray] = None
        self._time_values: list[Any] = []
        self._selected_ics: list[int] | None = None
        self._selection_metadata: dict[str, Any] = {}
        self._climatology_metadata: dict[str, Any] = {}
        self._climatology_available_days: set[int] = set()
        self._climatology_norm_by_day: dict[int, torch.Tensor] = {}
        self._climatology_warned_days: set[int] = set()
        self._graphweather_params_m: float | None = None
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

    def _find_nc_files(self, path: str) -> list[str]:
        return find_nc_files(path)

    def _load_variable_names(self, ds: Any) -> list[str]:
        if "channel" in ds.variables:
            raw = ds.variables["channel"][:]
            names = [str(x) for x in raw]
        else:
            names = [f"Var{i}" for i in range(max(self.cfg.out_channels) + 1)]
        self.variable_names = []
        for idx in self.cfg.out_channels:
            self.variable_names.append(names[idx] if idx < len(names) else f"Var{idx}")
        return self.variable_names

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

    def _load_model(self, height: int, width: int, checkpoint_path: str) -> GraphWeatherModel:
        graph = load_graph_bundle(self.cfg.graph_path, map_location="cpu").to(self.device)
        self._validate_graph_resolution(graph.metadata)
        per_step_channels = len(self.cfg.in_channels)
        if self.cfg.add_grid:
            per_step_channels += self.cfg.n_grid_channels
        if self.cfg.orography:
            per_step_channels += 1

        checkpoint_path = self._resolve_checkpoint_path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        metadata = dict(checkpoint.get("metadata", {}))
        self._validate_checkpoint_resolution(metadata)
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
            input_channels=2 * per_step_channels,
            output_channels=len(self.cfg.out_channels),
            n_history=self.cfg.n_history,
            hidden_dim=self.cfg.hidden_dim,
            edge_dim=self.cfg.edge_dim,
            heads=self.cfg.num_heads,
            encoder_blocks=self.cfg.encoder_blocks,
            decoder_blocks=self.cfg.decoder_blocks,
            l0_blocks=self.cfg.l0_blocks,
            l1_blocks=self.cfg.l1_blocks,
            l2_blocks=self.cfg.l2_blocks,
            l1_refine_blocks=self.cfg.l1_refine_blocks,
            l0_refine_blocks=self.cfg.l0_refine_blocks,
            use_delta_normalization=use_delta_normalization,
            delta_mean=delta_mean,
            delta_std=delta_std,
            delta_norm_center=delta_norm_center,
            delta_norm_eps=self.cfg.delta_norm_eps,
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
        return model

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
        if not bool(climatology.metadata.get("calendar_aware", False)) or not bool(
            climatology.metadata.get("dayofyear_from_real_time_coordinate", False)
        ):
            raise ValueError(
                "ACC requires a calendar-aware day-of-year climatology built from real NetCDF time coordinates. "
                "Run scripts/build_climatology.py and pass --climatology_path to the evaluator."
            )
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
            }
        )
        return climatology_norm

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
            "per_ic_acc": per_ic_acc,
        }
        metrics.update(self._bootstrap_confidence_intervals(per_ic_mse, per_ic_acc))
        return metrics

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
        per_ic_mse_values = []
        per_ic_acc_values = []
        for count, ic in enumerate(ics):
            mse, acc = self._rollout_one_ic(fields, ic, fixed_rollout_steps, climatology_norm, lat_weights, lat_weights_np)
            per_ic_mse_values.append(mse)
            per_ic_acc_values.append(acc)
            if (count + 1) % 10 == 0 or (count + 1) == len(ics):
                self.logger.info("%s completed IC %d/%d (ic=%d)", label, count + 1, len(ics), ic)
        metrics = self._aggregate_metrics(per_ic_mse_values, per_ic_acc_values)
        metrics["initial_condition_indices"] = np.asarray(ics, dtype=np.int64)
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
    def _rollout_one_ic(
        self,
        fields: Any,
        ic: int,
        forecast_steps: int,
        climatology_norm: torch.Tensor,
        lat_weights: torch.Tensor,
        lat_weights_np: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Model must be loaded before rollout.")

        n_channels = len(self.cfg.out_channels)
        height = self.model.graph.L0.height
        width = self.model.graph.L0.width
        mse = np.zeros((forecast_steps, n_channels), dtype=np.float64)
        acc = np.zeros((forecast_steps, n_channels), dtype=np.float64)
        std_view = torch.as_tensor(self.out_stds, device=self.device, dtype=torch.float32).view(1, -1, 1, 1)
        weight = lat_weights_np.reshape(1, -1, 1)
        denom = float(lat_weights_np.sum() * width)

        raw_prev = np.asarray(fields[ic - self.cfg.dt, :, :height, :width], dtype=np.float32)
        raw_cur = np.asarray(fields[ic, :, :height, :width], dtype=np.float32)
        previous = self._normalize_input_step(raw_prev)
        current = self._normalize_input_step(raw_cur)
        for step_idx in range(forecast_steps):
            pred_norm = self.model.forward_steps(previous, current)
            lead = step_idx + 1
            target_idx = ic + lead * self.cfg.dt
            raw_target = np.asarray(fields[target_idx, :, :height, :width], dtype=np.float32)
            target_norm = self._normalize_target_step(raw_target)

            err_phys = ((pred_norm - target_norm) * std_view)[0].detach().cpu().double().numpy()
            mse[step_idx] = ((err_phys ** 2) * weight).sum(axis=(-2, -1)) / denom

            if not self._time_values:
                raise RuntimeError("Decoded time coordinates are required for calendar-aware ACC.")
            clim = self._climatology_norm_for_day(dayofyear(self._time_values[target_idx]))
            acc[step_idx] = weighted_acc_per_channel(pred_norm - clim, target_norm - clim, lat_weights)[0].detach().cpu().numpy()

            next_current = current.clone()
            next_current[:, :n_channels] = pred_norm
            previous, current = current, next_current

        return mse, acc

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

            if not self._time_values:
                raise RuntimeError("Decoded time coordinates are required for calendar-aware ACC.")
            clim = self._climatology_norm_for_day(dayofyear(self._time_values[target_idx]))
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
        return rows

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
        self._time_values = decode_time_values(ds)
        if len(self._time_values) != int(total_t):
            raise ValueError(f"Decoded time length {len(self._time_values)} does not match fields length {total_t}.")
        if "latitude" in ds.variables:
            latitudes = np.asarray(ds["latitude"][:height], dtype=np.float64)
        elif "lat" in ds.variables:
            latitudes = np.asarray(ds["lat"][:height], dtype=np.float64)
        else:
            latitudes = cell_center_lat_lon(height, width)[0].numpy().astype(np.float64)

        climatology = self._load_or_build_climatology(height, width)
        climatology_norm = self._prepare_climatology_lookup(climatology)
        self._selected_ics = self._build_ic_indices(total_t, int(self.cfg.eval_fixed_rollout_steps))
        years = sorted({int(date_from_time(t).year) for t in self._time_values})
        self._selection_metadata.update(
            {
                "file": test_file,
                "file_years": years,
                "calendar_aware": True,
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
            self._log_summary(fixed_steps, metrics)
            self._save_metrics(metrics, fixed_steps, output_root / f"S{fixed_steps}")

            labeled_metrics = {graph_label: metrics}
            persistence = None
            checkpoint_payloads = {
                "global_best": {
                    "label": graph_label,
                    "checkpoint_path": checkpoint_path,
                    "epoch": metadata.get("epoch"),
                    "train_rollout_steps": metadata.get("train_rollout_steps"),
                    "params_m": self._graphweather_params_from_metadata(metadata),
                    "metrics": self._metrics_for_json(metrics, variable_indices),
                }
            }
            if self.cfg.plot_persistence:
                persistence = self._evaluate_persistence_rollout(
                    fields,
                    total_t,
                    height,
                    width,
                    fixed_steps,
                    climatology_norm,
                    lat_weights,
                    lat_weights_np,
                )
                labeled_metrics["persistence"] = persistence
                checkpoint_payloads["persistence"] = {
                    "label": "persistence",
                    "checkpoint_path": None,
                    "epoch": None,
                    "train_rollout_steps": None,
                    "metrics": self._metrics_for_json(persistence, variable_indices),
                }
            external_metrics = self._external_metric_arrays(variable_indices)
            labeled_metrics.update(external_metrics)
            external_json = self._external_baselines_json(variable_indices)
            self._write_fixed_json(
                output_root / f"fixed{fixed_steps}_global_best_metrics.json",
                fixed_steps,
                checkpoint_payloads,
                external_baselines=external_json,
            )
            self._write_global_summary_csv(output_root, fixed_steps, metrics, persistence, metadata)
            self._write_global_summary_text(output_root, fixed_steps, metrics, persistence, metadata)
            self._write_external_comparison_summary(output_root, fixed_steps, metrics, metadata)
            self._plot_comparison_variables(
                labeled_metrics,
                output_root,
                f"fixed{fixed_steps}_global_best",
                f"Fixed {fixed_steps}-day rollout evaluation",
            )
            return metrics
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
                metrics = self._evaluate_loaded_model_rollout(
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
                labeled_metrics[label] = metrics
                safe_label = label.replace(" ", "_").replace("=", "")
                self._save_metrics(metrics, fixed_steps, output_root / safe_label)
                checkpoint_payloads[safe_label] = {
                    "label": label,
                    "checkpoint_path": checkpoint_path,
                    "epoch": metadata.get("epoch"),
                    "train_rollout_steps": metadata.get("train_rollout_steps"),
                    "metrics": self._metrics_for_json(metrics, variable_indices),
                }

            if self.cfg.plot_persistence:
                persistence = self._evaluate_persistence_rollout(
                    fields,
                    total_t,
                    height,
                    width,
                    fixed_steps,
                    climatology_norm,
                    lat_weights,
                    lat_weights_np,
                )
                labeled_metrics["persistence"] = persistence
                checkpoint_payloads["persistence"] = {
                    "label": "persistence",
                    "checkpoint_path": None,
                    "epoch": None,
                    "train_rollout_steps": None,
                    "metrics": self._metrics_for_json(persistence, variable_indices),
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
        self.logger.info("FIXED ROLLOUT EVALUATION SUMMARY steps=%d", forecast_steps)
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


def run_evaluation_from_params(params: Any, logger: logging.Logger | Any = logging) -> dict[int, dict[str, np.ndarray]]:
    cfg = EvalConfig.from_params(params)
    evaluator = GraphWeatherEvaluator(cfg, logger=logger)
    return evaluator.evaluate()
