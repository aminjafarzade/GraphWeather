from __future__ import annotations

import csv
import glob
import json
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from .data import _add_grid_channels
from .graph_bundle import load_graph_bundle
from .models import GraphWeatherModel


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
        arr = arr[0]
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D normalization stats after squeeze, got {arr.shape}")
    return arr.astype(np.float32)


def _latitude_weights(latitudes_deg: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(latitudes_deg.astype(np.float64))).clip(min=0.0)
    if float(weights.sum()) <= 0.0:
        weights = np.ones_like(weights, dtype=np.float64)
    return weights


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
    eval_start_timestep: int
    n_initial_conditions: int
    ic_stride: int
    climatology_path: Optional[str]
    compute_climatology: bool
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
            eval_start_timestep=int(_get(params, "eval_start_timestep", 1)),
            n_initial_conditions=int(_get(params, "n_initial_conditions", _get(params, "eval_n_initial_conditions", 1))),
            ic_stride=int(_get(params, "eval_ic_stride", 1)),
            climatology_path=_get(params, "climatology_path", None),
            compute_climatology=bool(_get(params, "compute_climatology", True)),
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
        if self.compute_climatology and not self.train_data_path:
            raise ValueError("train_data_path is required when compute_climatology=true.")
        if not self.compute_climatology and not self.climatology_path:
            raise ValueError("climatology_path is required when compute_climatology=false.")
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
        self.in_means = self.means_all[cfg.in_channels]
        self.in_stds = self.stds_all[cfg.in_channels]
        self.out_means = self.means_all[cfg.out_channels]
        self.out_stds = self.stds_all[cfg.out_channels]
        self.model: Optional[GraphWeatherModel] = None
        self.variable_names: list[str] = []
        self.orography_field: Optional[np.ndarray] = None
        if cfg.orography:
            if not cfg.orography_path:
                raise ValueError("orography_path is required when orography=true.")
            with self.nc.Dataset(cfg.orography_path, "r") as ds:
                key = "orog" if "orog" in ds.variables else next(iter(ds.variables))
                self.orography_field = np.asarray(ds[key][:], dtype=np.float32).squeeze()

    def _find_nc_files(self, path: str) -> list[str]:
        if os.path.isfile(path):
            return [path]
        files = sorted(glob.glob(os.path.join(path, "*.nc")))
        if not files:
            raise FileNotFoundError(f"No .nc files found under {path}")
        return files

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
        per_step_channels = len(self.cfg.in_channels)
        if self.cfg.add_grid:
            per_step_channels += self.cfg.n_grid_channels
        if self.cfg.orography:
            per_step_channels += 1

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
        ).to(self.device)

        checkpoint_path = self._resolve_checkpoint_path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
        cleaned = OrderedDict()
        for key, value in state.items():
            cleaned[key[7:] if key.startswith("module.") else key] = value
        model.load_state_dict(cleaned, strict=True)
        model.eval()
        metadata = dict(checkpoint.get("metadata", {}))
        self.logger.info(
            "Loaded checkpoint: %s | epoch=%s | train_rollout_steps=%s",
            checkpoint_path,
            metadata.get("epoch", checkpoint.get("epoch", "unknown")),
            metadata.get("train_rollout_steps", "unknown"),
        )
        self.logger.info("Evaluation model parameters: %d", sum(p.numel() for p in model.parameters()))
        return model

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

    def _load_or_build_climatology(self, height: int, width: int) -> np.ndarray:
        if self.cfg.climatology_path and os.path.exists(self.cfg.climatology_path):
            self.logger.info("Loading climatology: %s", self.cfg.climatology_path)
            if self.cfg.climatology_path.endswith(".npz"):
                return np.load(self.cfg.climatology_path)["climatology"].astype(np.float32)
            if self.cfg.climatology_path.endswith(".npy"):
                return np.load(self.cfg.climatology_path).astype(np.float32)
            with self.nc.Dataset(self.cfg.climatology_path, "r") as ds:
                key = "climatology" if "climatology" in ds.variables else next(iter(ds.variables))
                clim = np.asarray(ds[key][:], dtype=np.float32)
                return clim[:, self.cfg.out_channels, :height, :width] if clim.shape[1] > len(self.cfg.out_channels) else clim

        output_dir = Path(self.cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / "daily_climatology.npz"
        if save_path.exists():
            self.logger.info("Loading cached climatology: %s", save_path)
            return np.load(save_path)["climatology"].astype(np.float32)

        self.logger.info("Computing daily climatology from: %s", self.cfg.train_data_path)
        files = self._find_nc_files(self.cfg.train_data_path)
        n_days = 366
        n_channels = len(self.cfg.out_channels)
        clim_sum = np.zeros((n_days, n_channels, height, width), dtype=np.float64)
        clim_count = np.zeros((n_days,), dtype=np.int64)
        for file_path in files:
            with self.nc.Dataset(file_path, "r") as ds:
                fields = ds["fields"]
                total_t = fields.shape[0]
                for start in range(0, total_t, 32):
                    stop = min(start + 32, total_t)
                    chunk = np.asarray(fields[start:stop, self.cfg.out_channels, :height, :width], dtype=np.float32)
                    for local_idx, time_idx in enumerate(range(start, stop)):
                        day = time_idx % n_days
                        clim_sum[day] += chunk[local_idx]
                        clim_count[day] += 1
        valid = clim_count > 0
        if not np.any(valid):
            raise RuntimeError("Failed to compute climatology: no valid samples.")
        for day in range(n_days):
            if clim_count[day] > 0:
                clim_sum[day] /= float(clim_count[day])
            else:
                nearest = int(np.flatnonzero(valid)[np.argmin(np.abs(np.flatnonzero(valid) - day))])
                clim_sum[day] = clim_sum[nearest]
        climatology = clim_sum.astype(np.float32)
        np.savez_compressed(save_path, climatology=climatology)
        self.logger.info("Saved climatology: %s", save_path)
        return climatology

    def _build_ic_indices(self, total_t: int, forecast_steps: int) -> list[int]:
        start = max(int(self.cfg.eval_start_timestep), self.cfg.dt * self.cfg.n_history)
        max_ic = total_t - forecast_steps * self.cfg.dt - 1
        if max_ic < start:
            raise ValueError(
                f"Not enough timesteps ({total_t}) for forecast_steps={forecast_steps}, "
                f"dt={self.cfg.dt}, start={start}."
            )
        ics = list(range(start, max_ic + 1, self.cfg.ic_stride))
        return ics[: self.cfg.n_initial_conditions]

    def _aggregate_metrics(
        self,
        sum_sq_err: np.ndarray,
        acc_values: list[np.ndarray],
        lat_weights_np: np.ndarray,
        width: int,
        n_ics: int,
    ) -> dict[str, np.ndarray]:
        pixel_rmse = np.sqrt(sum_sq_err / max(1, n_ics))
        weighted_rmse = (
            pixel_rmse * lat_weights_np.reshape(1, 1, -1, 1)
        ).sum(axis=(-2, -1)) / (lat_weights_np.sum() * width)
        avg_acc = np.mean(np.asarray(acc_values), axis=0)
        return {"rmse": weighted_rmse, "acc": avg_acc}

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
        ics = self._build_ic_indices(total_t, fixed_rollout_steps)
        self.logger.info(
            "Evaluating %s with fixed rollout_steps=%d over %d initial conditions: first_ic=%s",
            label,
            fixed_rollout_steps,
            len(ics),
            ics[0] if ics else None,
        )
        sum_sq_err = np.zeros((fixed_rollout_steps + 1, len(self.cfg.out_channels), height, width), dtype=np.float64)
        acc_values = []
        for count, ic in enumerate(ics):
            sq_err, acc = self._rollout_one_ic(fields, ic, fixed_rollout_steps, climatology_norm, lat_weights)
            sum_sq_err += sq_err
            acc_values.append(acc)
            if (count + 1) % 10 == 0 or (count + 1) == len(ics):
                self.logger.info("%s completed IC %d/%d (ic=%d)", label, count + 1, len(ics), ic)
        return self._aggregate_metrics(sum_sq_err, acc_values, lat_weights_np, width, len(ics))

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
        ics = self._build_ic_indices(total_t, fixed_rollout_steps)
        self.logger.info(
            "Evaluating persistence with fixed rollout_steps=%d over %d initial conditions",
            fixed_rollout_steps,
            len(ics),
        )
        sum_sq_err = np.zeros((fixed_rollout_steps + 1, len(self.cfg.out_channels), height, width), dtype=np.float64)
        acc_values = []
        for ic in ics:
            sq_err, acc = self._rollout_one_ic_persistence(
                fields,
                ic,
                fixed_rollout_steps,
                climatology_norm,
                lat_weights,
                height,
                width,
            )
            sum_sq_err += sq_err
            acc_values.append(acc)
        return self._aggregate_metrics(sum_sq_err, acc_values, lat_weights_np, width, len(ics))

    @torch.no_grad()
    def _rollout_one_ic(
        self,
        fields: Any,
        ic: int,
        forecast_steps: int,
        climatology_norm: torch.Tensor,
        lat_weights: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Model must be loaded before rollout.")

        n_channels = len(self.cfg.out_channels)
        height = self.model.graph.L0.height
        width = self.model.graph.L0.width
        sq_err = np.zeros((forecast_steps + 1, n_channels, height, width), dtype=np.float64)
        acc = np.zeros((forecast_steps + 1, n_channels), dtype=np.float64)
        std_view = torch.as_tensor(self.out_stds, device=self.device, dtype=torch.float32).view(1, -1, 1, 1)

        raw_prev = np.asarray(fields[ic - self.cfg.dt, :, :height, :width], dtype=np.float32)
        raw_cur = np.asarray(fields[ic, :, :height, :width], dtype=np.float32)
        previous = self._normalize_input_step(raw_prev)
        current = self._normalize_input_step(raw_cur)
        pred_norm = self._normalize_target_step(raw_cur)

        for lead in range(forecast_steps + 1):
            target_idx = ic + lead * self.cfg.dt
            raw_target = np.asarray(fields[target_idx, :, :height, :width], dtype=np.float32)
            target_norm = self._normalize_target_step(raw_target)

            err_phys = ((pred_norm - target_norm) * std_view)[0].detach().cpu().double().numpy()
            sq_err[lead] = err_phys ** 2

            clim_day = target_idx % climatology_norm.shape[0]
            clim = climatology_norm[clim_day : clim_day + 1]
            acc[lead] = weighted_acc_per_channel(pred_norm - clim, target_norm - clim, lat_weights)[0].detach().cpu().numpy()

            if lead < forecast_steps:
                next_pred = self.model.forward_steps(previous, current)
                next_current = current.clone()
                next_current[:, :n_channels] = next_pred
                previous, current = current, next_current
                pred_norm = next_pred

        return sq_err, acc

    @torch.no_grad()
    def _rollout_one_ic_persistence(
        self,
        fields: Any,
        ic: int,
        forecast_steps: int,
        climatology_norm: torch.Tensor,
        lat_weights: torch.Tensor,
        height: int,
        width: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_channels = len(self.cfg.out_channels)
        sq_err = np.zeros((forecast_steps + 1, n_channels, height, width), dtype=np.float64)
        acc = np.zeros((forecast_steps + 1, n_channels), dtype=np.float64)
        std_view = torch.as_tensor(self.out_stds, device=self.device, dtype=torch.float32).view(1, -1, 1, 1)

        raw_cur = np.asarray(fields[ic, :, :height, :width], dtype=np.float32)
        pred_norm = self._normalize_target_step(raw_cur)

        for lead in range(forecast_steps + 1):
            target_idx = ic + lead * self.cfg.dt
            raw_target = np.asarray(fields[target_idx, :, :height, :width], dtype=np.float32)
            target_norm = self._normalize_target_step(raw_target)

            err_phys = ((pred_norm - target_norm) * std_view)[0].detach().cpu().double().numpy()
            sq_err[lead] = err_phys ** 2

            clim_day = target_idx % climatology_norm.shape[0]
            clim = climatology_norm[clim_day : clim_day + 1]
            acc[lead] = weighted_acc_per_channel(pred_norm - clim, target_norm - clim, lat_weights)[0].detach().cpu().numpy()

        return sq_err, acc

    def _save_metrics(
        self,
        metrics: dict[str, np.ndarray],
        forecast_steps: int,
        output_dir: Path,
    ) -> list[dict[str, Any]]:
        output_dir.mkdir(parents=True, exist_ok=True)
        rmse = metrics["rmse"]
        acc = metrics["acc"]
        rows: list[dict[str, Any]] = []
        for lead in range(rmse.shape[0]):
            for var_idx in range(rmse.shape[1]):
                rows.append(
                    {
                        "rollout_steps": forecast_steps,
                        "lead_time": lead,
                        "variable_idx": var_idx,
                        "original_channel_idx": int(self.cfg.out_channels[var_idx]),
                        "variable_name": self.variable_names[var_idx],
                        "rmse": float(rmse[lead, var_idx]),
                        "acc": float(acc[lead, var_idx]),
                    }
                )

        with open(output_dir / "evaluation_metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        rmse_header = ["lead_time"] + self.variable_names
        with open(output_dir / "rollout_rmse.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(rmse_header)
            for lead in range(rmse.shape[0]):
                writer.writerow([lead] + [float(x) for x in rmse[lead]])

        acc_header = ["lead_time"] + self.variable_names
        with open(output_dir / "rollout_acc.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(acc_header)
            for lead in range(acc.shape[0]):
                writer.writerow([lead] + [float(x) for x in acc[lead]])

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

        lower_names = {name.lower(): idx for idx, name in enumerate(self.variable_names)}
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
                var_idx = lower_names.get(token.lower())

            if var_idx is None:
                self.logger.warning(
                    "Could not resolve plot variable '%s'. Use variable name, local variable_idx, original channel index, or 'all'.",
                    token,
                )
                continue
            if var_idx not in resolved:
                resolved.append(var_idx)
        return resolved

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
                if label == "persistence":
                    self.logger.info(
                        "Skipping persistence in comparison plot for %s; persistence metrics remain in saved JSON.",
                        variable_name,
                    )
                    continue
                rmse = metrics["rmse"][:, var_idx]
                acc = metrics["acc"][:, var_idx]
                leads = np.arange(rmse.shape[0])
                axes[0].plot(leads, rmse, marker="o", linestyle="-", linewidth=1.8, label=label)
                axes[1].plot(leads, acc, marker="o", linestyle="-", linewidth=1.8, label=label)
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
            axes[0].set_title(f"{title_prefix} RMSE - {title}")
            axes[0].set_ylabel("RMSE")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()

            axes[1].set_title(f"{title_prefix} ACC - {title}")
            axes[1].set_xlabel("Lead time (days)")
            axes[1].set_ylabel("ACC")
            axes[1].set_ylim(-1.05, 1.05)
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
        self._load_variable_names(ds)
        if "latitude" in ds.variables:
            latitudes = np.asarray(ds["latitude"][:height], dtype=np.float64)
        elif "lat" in ds.variables:
            latitudes = np.asarray(ds["lat"][:height], dtype=np.float64)
        else:
            latitudes = np.linspace(-90.0 + 90.0 / height, 90.0 - 90.0 / height, height)

        climatology = self._load_or_build_climatology(height, width)
        clim_norm = (climatology - self.out_means.reshape(1, -1, 1, 1)) / (
            self.out_stds.reshape(1, -1, 1, 1) + 1.0e-8
        )
        climatology_norm = torch.as_tensor(clim_norm, device=self.device, dtype=torch.float32)
        lat_weights_np = _latitude_weights(latitudes)
        lat_weights = torch.as_tensor(lat_weights_np, device=self.device, dtype=torch.float32)
        return ds, fields, total_t, height, width, climatology_norm, lat_weights, lat_weights_np

    def _metrics_for_json(self, metrics: dict[str, np.ndarray], variable_indices: list[int]) -> dict[str, dict[str, list[float]]]:
        rmse = {}
        acc = {}
        for var_idx in variable_indices:
            name = self.variable_names[var_idx]
            rmse[name] = [float(x) for x in metrics["rmse"][1:, var_idx]]
            acc[name] = [float(x) for x in metrics["acc"][1:, var_idx]]
        return {"rmse": rmse, "acc": acc}

    def _json_variable_indices(self) -> list[int]:
        requested = self._resolve_plot_variables()
        return requested if requested else list(range(len(self.cfg.out_channels)))

    def _write_fixed_json(
        self,
        output_path: Path,
        fixed_rollout_steps: int,
        checkpoint_payloads: dict[str, dict[str, Any]],
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "eval_fixed_rollout_steps": fixed_rollout_steps,
            "lead_times": list(range(1, fixed_rollout_steps + 1)),
            "checkpoints": checkpoint_payloads,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self.logger.info("Saved fixed-rollout comparison JSON: %s", output_path)

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

            labeled_metrics = {"global best": metrics}
            checkpoint_payloads = {
                "global_best": {
                    "label": "global best",
                    "checkpoint_path": checkpoint_path,
                    "epoch": metadata.get("epoch"),
                    "train_rollout_steps": metadata.get("train_rollout_steps"),
                    **self._metrics_for_json(metrics, self._json_variable_indices()),
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
                    **self._metrics_for_json(persistence, self._json_variable_indices()),
                }
            self._write_fixed_json(
                output_root / f"fixed{fixed_steps}_global_best_metrics.json",
                fixed_steps,
                checkpoint_payloads,
            )
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
                    **self._metrics_for_json(metrics, self._json_variable_indices()),
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
                    **self._metrics_for_json(persistence, self._json_variable_indices()),
                }

            self._write_fixed_json(
                output_root / f"fixed{fixed_steps}_stage_comparison_metrics.json",
                fixed_steps,
                checkpoint_payloads,
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
                lead,
                float(np.mean(rmse[lead])),
                float(np.mean(acc[lead])),
            )


def run_evaluation_from_params(params: Any, logger: logging.Logger | Any = logging) -> dict[int, dict[str, np.ndarray]]:
    cfg = EvalConfig.from_params(params)
    evaluator = GraphWeatherEvaluator(cfg, logger=logger)
    return evaluator.evaluate()
