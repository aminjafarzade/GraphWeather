from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import attention_entropy_stats, dirichlet_energy_stats, embedding_stats, flatten_embeddings, gradient_metrics, sample_rows
from .spectral import SpectralCurve, spectral_band_summary, spectral_rmse_2d
from .graph_structure import compute_graph_structure_metrics
from .power_spectrum import power_spectrum_radial, variance_ratio
from .wandb_plots import (
    log_attention_tables_and_plots,
    log_histograms,
    log_layer_tables_and_plots,
    log_optimization_plots,
    log_rollout_heatmap,
    log_rollout_tables_and_plots,
    log_spatial_maps,
    log_spectral_tables_and_plots,
    log_summary_table,
)
from .wandb_logger import log_metrics as wandb_log_metrics
from .wandb_logger import maybe_get_wandb_run


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _safe_name(name: str) -> str:
    return str(name).replace(" ", "_")


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class DiagnosticsManager:
    """Optional diagnostics runner for rollout stability, representations, attention, spectra, and system metrics.

    Assumptions:
    - Trainer predictions and targets are normalized weather states shaped [B, C, H, W].
    - Graph activations are [B, N, C]; grid activations are [B, C, H, W].
    - Current LocalGraphAttention normalizes over fixed incoming neighbors, so attention tensors are [B, N, K, H].
    """

    def __init__(
        self,
        config: Any,
        model: torch.nn.Module,
        loss_fn: Any,
        device: torch.device,
        logger: Any = logging,
        wandb_run: Any = None,
        run_name: str | None = None,
        rank: int = 0,
    ):
        self.config = self._resolve_config(config)
        self.enabled = bool(self.config.get("enabled", False))
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.logger = logger
        self.rank = int(rank)
        self.is_rank0 = self.rank == 0
        self.run_name = str(
            run_name
            or self.config.get("run_name")
            or (self.config.get("wandb", {}) or {}).get("run_name")
            or "diagnostics_run"
        )
        self.output_root = Path(str(self.config.get("output_dir", "diagnostics"))).expanduser()
        self.output_dir = self.output_root / self.run_name
        if self.enabled and self.is_rank0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.wandb_run = wandb_run if wandb_run is not None else maybe_get_wandb_run(self.config, logger=self.logger)
        self.layer_rows: list[dict[str, Any]] = []
        self.attention_rows: list[dict[str, Any]] = []
        self.attention_head_rows: list[dict[str, Any]] = []
        self.rollout_rows: list[dict[str, Any]] = []
        self.spectral_curve_rows: list[dict[str, Any]] = []
        self.spectral_band_rows: list[dict[str, Any]] = []
        self.spatial_map_items: list[dict[str, Any]] = []
        self.attention_histograms: dict[str, np.ndarray] = {}
        self.cosine_histograms: dict[str, np.ndarray] = {}
        self._active_epoch = 0
        self._active_phase = "valid"
        self._active_visuals_enabled = False
        self._active_histograms_enabled = False
        self._active_spatial_maps_enabled = False
        self._warned_attention_missing = False
        self._warned_spectral_skip = False
        self._warned_spatial_skip = False
        self._latest_gradient_metrics: dict[str, float] = {}
        self._previous_layer_summary_metrics: dict[tuple[str, str], float] = {}
        self._previous_attention_summary_metrics: dict[tuple[str, str], float] = {}
        # --- new-diagnostics buffers/caches ---
        self.power_spectrum_rows: list[dict[str, Any]] = []
        self.variance_ratio_rows: list[dict[str, Any]] = []
        self.graph_structure_rows: list[dict[str, Any]] = []
        self._num_nodes_to_edge_index: dict[int, torch.Tensor] | None = None
        self._graph_structure_done = False
        self._warned_dirichlet_missing = False
        self._warned_power_spectrum_skip = False

    @staticmethod
    def _resolve_config(config: Any) -> dict[str, Any]:
        root = getattr(config, "params", config)
        raw = _cfg_get(root, "diagnostics", {}) or {}
        if not isinstance(raw, dict):
            raise ValueError("diagnostics must be a mapping when provided.")
        defaults = {
            "enabled": False,
            "log_every_epochs": 1,
            "heavy_every_epochs": 5,
            "run_after_training": True,
            "rollout_horizons": [1, 2, 4, 6, 8, 10],
            "max_train_diag_batches": 2,
            "max_valid_diag_batches": 4,
            "max_full_diag_batches": 32,
            "embedding_sample_nodes": 2048,
            "pairwise_sample_nodes": 1024,
            "collect_embeddings": True,
            "collect_attention": True,
            "spectral_rmse": True,
            "spectral_variables": ["t2m", "u10", "v10", "msl", "z500"],
            "grid_shape": None,
            "baseline_compare": {
                "enabled": False,
                "baseline_name": None,
                "baseline_rollout_curve_csv": None,
                "baseline_scalars_json": None,
            },
            "wandb": {
                "enabled": True,
                "project": None,
                "entity": "amin1jafarzade-kaist",
                "run_name": None,
                "tags": ["diagnostics"],
                "log_scalars": True,
                "log_tables": True,
                "log_plots": False,
                "log_images": False,
                "log_histograms": False,
                "log_artifacts": False,
            },
            "plots": {
                "enabled": False,
                "save_local": True,
                "output_dir": "diagnostics",
                "plot_every_epochs": 5,
                "plot_after_training": True,
                "map_horizons": [1, 4, 10],
                "spectral_horizons": [1, 4, 10],
                "variables": ["t2m", "u10", "v10", "msl", "z500"],
                "max_map_batches": 2,
                "max_map_samples": 4,
                "max_hist_values": 200000,
                "rollout_plots": True,
                "layer_plots": True,
                "attention_plots": True,
                "spectral_plots": True,
                "spatial_maps": True,
                "optimization_plots": True,
            },
            "dirichlet_energy": True,
            "graph_structure": True,
            "graph_structure_max_dense_nodes": 20000,
            "power_spectrum": {
                "enabled": True,
                "leads": [1, 5, 10],
                "variables": [],
                "prefer_sht": True,
            },
            "output_dir": "diagnostics",
        }
        merged = dict(defaults)
        merged.update(raw)
        baseline_compare = dict(defaults["baseline_compare"])
        baseline_compare.update(dict(raw.get("baseline_compare", {}) or {}))
        merged["baseline_compare"] = baseline_compare
        wandb_cfg = dict(defaults["wandb"])
        wandb_cfg.update(dict(raw.get("wandb", {}) or {}))
        merged["wandb"] = wandb_cfg
        plots_cfg = dict(defaults["plots"])
        plots_cfg.update(dict(raw.get("plots", {}) or {}))
        merged["plots"] = plots_cfg
        power_cfg = dict(defaults["power_spectrum"])
        power_cfg.update(dict(raw.get("power_spectrum", {}) or {}))
        merged["power_spectrum"] = power_cfg
        if not bool(merged.get("enabled", False)):
            return {"enabled": False}
        merged["rollout_horizons"] = [int(x) for x in list(merged.get("rollout_horizons", [])) if int(x) > 0]
        for list_key in ("map_horizons", "spectral_horizons"):
            plots_cfg[list_key] = [int(x) for x in list(plots_cfg.get(list_key, [])) if int(x) > 0]
        plots_cfg["variables"] = [str(x) for x in list(plots_cfg.get("variables", []) or [])]
        for key in (
            "log_every_epochs",
            "heavy_every_epochs",
            "max_train_diag_batches",
            "max_valid_diag_batches",
            "max_full_diag_batches",
            "embedding_sample_nodes",
            "pairwise_sample_nodes",
        ):
            merged[key] = int(merged[key])
        for key in ("plot_every_epochs", "max_map_batches", "max_map_samples", "max_hist_values"):
            plots_cfg[key] = int(plots_cfg[key])
        for key in ("collect_embeddings", "collect_attention", "spectral_rmse", "run_after_training"):
            merged[key] = bool(merged.get(key, False))
        for key in ("log_scalars", "log_tables", "log_plots", "log_images", "log_histograms", "log_artifacts"):
            wandb_cfg[key] = bool(wandb_cfg.get(key, False))
        for key in (
            "enabled",
            "save_local",
            "plot_after_training",
            "rollout_plots",
            "layer_plots",
            "attention_plots",
            "spectral_plots",
            "spatial_maps",
            "optimization_plots",
        ):
            plots_cfg[key] = bool(plots_cfg.get(key, False))
        return merged

    def should_run_light(self, epoch: int) -> bool:
        if not self.enabled:
            return False
        every = max(1, int(self.config.get("log_every_epochs", 1)))
        return int(epoch) % every == 0

    def should_run_heavy(self, epoch: int) -> bool:
        if not self.enabled:
            return False
        every = max(1, int(self.config.get("heavy_every_epochs", 5)))
        return int(epoch) % every == 0

    def _plots_config(self) -> dict[str, Any]:
        return dict(self.config.get("plots", {}) or {})

    def _should_log_visuals(self, epoch: int, *, is_post_training: bool = False) -> bool:
        plots = self._plots_config()
        if not bool(plots.get("enabled", False)):
            return False
        if is_post_training and bool(plots.get("plot_after_training", True)):
            return True
        every = max(1, int(plots.get("plot_every_epochs", 5)))
        return int(epoch) % every == 0

    def _max_hist_values(self) -> int:
        plots = self._plots_config()
        return max(1, int(plots.get("max_hist_values", 200000)))

    @staticmethod
    def _sample_numpy(values: np.ndarray, max_values: int) -> np.ndarray:
        flat = np.asarray(values, dtype=np.float32).reshape(-1)
        flat = flat[np.isfinite(flat)]
        if flat.size <= int(max_values):
            return flat
        idx = np.linspace(0, flat.size - 1, int(max_values), dtype=np.int64)
        return flat[idx]

    def _sample_tensor_values(self, tensor: torch.Tensor, max_values: int) -> np.ndarray:
        flat = tensor.detach().float().reshape(-1)
        if flat.numel() > int(max_values):
            idx = torch.linspace(0, flat.numel() - 1, int(max_values), device=flat.device).long()
            flat = flat.index_select(0, idx)
        return flat.cpu().numpy().astype(np.float32, copy=False)

    def _sample_cosine_histogram(self, tensor: torch.Tensor) -> np.ndarray:
        max_values = self._max_hist_values()
        pairwise_nodes = int(self.config.get("pairwise_sample_nodes", 1024))
        max_rows_for_hist = max(2, min(pairwise_nodes, int(np.sqrt(max_values * 2.0)) + 1))
        x = sample_rows(flatten_embeddings(tensor), max_rows_for_hist).detach().float()
        if x.shape[0] < 2:
            return np.asarray([], dtype=np.float32)
        z = F.normalize(x, p=2, dim=-1, eps=1.0e-12)
        cosine = z @ z.transpose(0, 1)
        mask = ~torch.eye(cosine.shape[0], dtype=torch.bool, device=cosine.device)
        return self._sample_tensor_values(cosine[mask], max_values=max_values)

    def _attention_weight_histogram(self, attn: torch.Tensor) -> np.ndarray:
        max_values = self._max_hist_values()
        p = attn.detach().float().clamp_min(0.0)
        if p.dim() >= 3:
            p = p / (p.sum(dim=-2, keepdim=True) + 1.0e-12)
        return self._sample_tensor_values(p, max_values=max_values)

    def _attention_head_metrics(self, name: str, attn: torch.Tensor) -> list[dict[str, Any]]:
        p = attn.detach().float()
        if p.dim() == 3:
            p = p.unsqueeze(0)
        if p.dim() != 4:
            return []
        p = p.clamp_min(0.0)
        degree = int(p.shape[2])
        if degree <= 0:
            return []
        p = p / (p.sum(dim=2, keepdim=True) + 1.0e-12)
        entropy = -(p * torch.log(p + 1.0e-12)).sum(dim=2)
        entropy_norm = entropy / max(float(np.log(max(degree, 1))), 1.0e-12)
        max_weight = p.max(dim=2).values
        rows = []
        for head in range(int(p.shape[-1])):
            rows.append(
                {
                    "epoch": int(self._active_epoch),
                    "phase": self._active_phase,
                    "layer_name": str(name),
                    "head": int(head),
                    "entropy_norm": float(entropy_norm[..., head].mean().item()),
                    "max_weight_mean": float(max_weight[..., head].mean().item()),
                }
            )
        return rows

    def reset_buffers(self, epoch: int, phase: str) -> None:
        self._active_epoch = int(epoch)
        self._active_phase = str(phase)
        self.layer_rows = []
        self.attention_rows = []
        self.attention_head_rows = []
        self.rollout_rows = []
        self.spectral_curve_rows = []
        self.spectral_band_rows = []
        self.spatial_map_items = []
        self.power_spectrum_rows = []
        self.variance_ratio_rows = []
        self.graph_structure_rows = []
        self.attention_histograms = {}
        self.cosine_histograms = {}
        self._active_visuals_enabled = False
        self._active_histograms_enabled = False
        self._active_spatial_maps_enabled = False

    def _level_edge_index(self, num_nodes: int) -> torch.Tensor | None:
        if self._num_nodes_to_edge_index is None:
            self._num_nodes_to_edge_index = {}
            model = self.model
            graph = getattr(model, "graph", None) or getattr(getattr(model, "module", None), "graph", None)
            if graph is not None:
                for name in ("L0", "L1", "L2", "L3", "L4"):
                    level = getattr(graph, name, None)
                    if level is None:
                        continue
                    self._num_nodes_to_edge_index[int(level.num_nodes)] = level.edge_index
        return self._num_nodes_to_edge_index.get(int(num_nodes))

    def _graph_structure_diagnostics(self) -> dict[str, float]:
        if self._graph_structure_done or not bool(self.config.get("graph_structure", True)):
            return {}
        self._graph_structure_done = True
        model = self.model
        graph = getattr(model, "graph", None) or getattr(getattr(model, "module", None), "graph", None)
        if graph is None:
            return {}
        try:
            max_dense = int(self.config.get("graph_structure_max_dense_nodes", 20000))
            rows = compute_graph_structure_metrics(graph, max_dense_nodes=max_dense)
        except Exception as exc:
            self.logger.warning("Graph-structure diagnostics skipped: %s", exc)
            return {}
        self.graph_structure_rows = rows
        metrics: dict[str, float] = {}
        for row in rows:
            level = str(row["level"])
            for key in ("spectral_gap_lambda2", "mean_effective_resistance", "kirchhoff_index", "avg_degree"):
                value = row.get(key)
                if value is not None:
                    metrics[f"diagnostics_graph/{level}/{key}"] = float(value)
                    metrics[f"model/graph/{level}/{key}"] = float(value)
        return metrics

    def _power_spectrum_vs_era5(self, trainer: Any, loader: Any, *, epoch: int, max_batches: int) -> dict[str, float]:
        cfg = dict(self.config.get("power_spectrum", {}) or {})
        if not bool(cfg.get("enabled", True)):
            return {}
        requested = list(cfg.get("variables", []) or self.config.get("spectral_variables", []) or [])
        variables = self._resolve_named_variables(trainer, requested, diagnostic_name="Power spectrum")
        if not variables:
            return {}
        leads = sorted({int(x) for x in cfg.get("leads", [1, 5, 10]) if int(x) > 0})
        if not leads:
            return {}
        prefer_sht = bool(cfg.get("prefer_sht", True))
        grid_shape = self._grid_shape(trainer)
        acc: dict[tuple[int, str], dict[str, Any]] = {}
        for batch_idx, data in enumerate(loader):
            if batch_idx >= int(max_batches):
                break
            inp, target, metadata = trainer._to_device_batch(data)
            target_seq = trainer._target_sequence(target)
            avail = sorted(h for h in leads if h <= int(target_seq.shape[1]))
            if not avail:
                continue
            previous, current = trainer.model.adapter.extract_two_steps(inp)
            initial_state = current
            with trainer._autocast_context():
                for step in range(max(avail)):
                    lead = step + 1
                    gt = target_seq[:, step]
                    aux = trainer._build_aux_for_step(current, target_seq, metadata, step)
                    pred = trainer._forward_model_step(previous, current, aux, lead=lead)
                    feature_builder = getattr(trainer, "feature_builder", None)
                    if feature_builder is not None:
                        pred = feature_builder.apply_overrides(pred, current=current, target_norm=gt)
                    target_handler = getattr(trainer, "target_handler", None)
                    if target_handler is not None:
                        pred = target_handler.apply(
                            pred_next=pred, current_state=current,
                            initial_state=initial_state, target_sequence=target_seq, lead=lead,
                        )
                    if lead in avail:
                        try:
                            pred_grid = self._as_grid(pred.detach(), grid_shape)
                            truth_grid = self._as_grid(target_seq[:, lead - 1], grid_shape)
                        except Exception as exc:
                            if not self._warned_power_spectrum_skip:
                                self.logger.warning("Power spectrum skipped: %s", exc)
                                self._warned_power_spectrum_skip = True
                            break
                        for name, idx in variables:
                            if int(idx) < 0 or int(idx) >= int(pred_grid.shape[1]):
                                continue
                            pf = pred_grid[:, int(idx)]
                            tf = truth_grid[:, int(idx)]
                            pb, pp, backend = power_spectrum_radial(pf, prefer_sht=prefer_sht)
                            _, tp, _ = power_spectrum_radial(tf, prefer_sht=prefer_sht)
                            entry = acc.setdefault((lead, name), {
                                "bins": pb, "pred": np.zeros_like(pp), "truth": np.zeros_like(tp),
                                "vr": 0.0, "count": 0, "backend": backend,
                            })
                            m = min(entry["pred"].shape[0], pp.shape[0], tp.shape[0])
                            entry["pred"][:m] += pp[:m]
                            entry["truth"][:m] += tp[:m]
                            entry["vr"] += variance_ratio(pf, tf)
                            entry["count"] += 1
                    next_step = current.clone()
                    next_step[:, : trainer.model.output_channels] = pred
                    previous, current = current, next_step
        metrics: dict[str, float] = {}
        for (lead, name), entry in sorted(acc.items()):
            c = max(int(entry["count"]), 1)
            pred_ps = entry["pred"] / c
            truth_ps = entry["truth"] / c
            vr = entry["vr"] / c
            ratio = np.divide(pred_ps, np.maximum(truth_ps, 1e-20))
            metrics[f"diagnostics_power/{name}/S{lead}/variance_ratio"] = float(vr)
            self.variance_ratio_rows.append({
                "epoch": int(epoch), "phase": self._active_phase, "lead": int(lead),
                "variable": str(name), "variance_ratio": float(vr), "backend": entry["backend"],
            })
            for k, pp_k, tp_k, r_k in zip(entry["bins"].tolist(), pred_ps.tolist(), truth_ps.tolist(), ratio.tolist()):
                self.power_spectrum_rows.append({
                    "epoch": int(epoch), "phase": self._active_phase, "lead": int(lead),
                    "variable": str(name), "wavenumber_bin": int(k),
                    "power_pred": float(pp_k), "power_truth": float(tp_k),
                    "power_ratio": float(r_k), "backend": entry["backend"],
                })
        return metrics

    def add_embedding(self, name: str, tensor: torch.Tensor) -> None:
        if not self.enabled or not bool(self.config.get("collect_embeddings", True)):
            return
        try:
            metrics = embedding_stats(
                tensor,
                embedding_sample_nodes=int(self.config.get("embedding_sample_nodes", 2048)),
                pairwise_sample_nodes=int(self.config.get("pairwise_sample_nodes", 1024)),
            )
        except Exception as exc:  # diagnostics must not interrupt training
            self.logger.warning("Embedding diagnostics skipped for %s: %s", name, exc)
            return
        row = {
            "epoch": int(self._active_epoch),
            "phase": self._active_phase,
            "layer_name": str(name),
            **metrics,
        }
        if bool(self.config.get("dirichlet_energy", True)):
            try:
                x = tensor.detach()
                if x.dim() == 3:
                    num_nodes = int(x.shape[1])
                elif x.dim() == 2:
                    num_nodes = int(x.shape[0])
                else:
                    num_nodes = -1
                edge_index = self._level_edge_index(num_nodes) if num_nodes > 0 else None
                if edge_index is not None:
                    row.update(dirichlet_energy_stats(x, edge_index))
                elif not self._warned_dirichlet_missing:
                    self.logger.warning(
                        "Dirichlet energy skipped for %s: node count %s matches no graph level.",
                        name, num_nodes,
                    )
                    self._warned_dirichlet_missing = True
            except Exception as exc:
                self.logger.warning("Dirichlet diagnostics skipped for %s: %s", name, exc)
        self.layer_rows.append(row)
        if self._active_histograms_enabled:
            try:
                values = self._sample_cosine_histogram(tensor)
                if values.size:
                    self.cosine_histograms[_safe_name(str(name))] = values
            except Exception as exc:
                self.logger.warning("Cosine histogram diagnostics skipped for %s: %s", name, exc)

    def add_attention(
        self,
        name: str,
        attn: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        num_nodes: int | None = None,
    ) -> None:
        if not self.enabled or not bool(self.config.get("collect_attention", True)):
            return
        try:
            metrics = attention_entropy_stats(attn, edge_index=edge_index, num_nodes=num_nodes)
        except Exception as exc:
            self.logger.warning("Attention diagnostics skipped for %s: %s", name, exc)
            return
        row = {
            "epoch": int(self._active_epoch),
            "phase": self._active_phase,
            "layer_name": str(name),
            **metrics,
        }
        self.attention_rows.append(row)
        if self._active_histograms_enabled:
            try:
                values = self._attention_weight_histogram(attn)
                if values.size:
                    self.attention_histograms[_safe_name(str(name))] = values
            except Exception as exc:
                self.logger.warning("Attention histogram diagnostics skipped for %s: %s", name, exc)
        if self._active_visuals_enabled:
            try:
                self.attention_head_rows.extend(self._attention_head_metrics(name, attn))
            except Exception as exc:
                self.logger.warning("Attention per-head diagnostics skipped for %s: %s", name, exc)

    def collect_layer_metrics_from_forward(self, trainer: Any, loader: Any, phase: str, epoch: int) -> None:
        if not self.enabled:
            return
        if not bool(self.config.get("collect_embeddings", True)) and not bool(self.config.get("collect_attention", True)):
            return
        was_training = bool(trainer.model.training)
        trainer.model.eval()
        self._active_epoch = int(epoch)
        self._active_phase = str(phase)
        try:
            data = next(iter(loader))
        except StopIteration:
            return
        try:
            with torch.no_grad():
                inp, target, metadata = trainer._to_device_batch(data)
                target_seq = trainer._target_sequence(target)
                if target_seq.shape[1] < 1:
                    return
                previous, current = trainer.model.adapter.extract_two_steps(inp)
                aux = trainer._build_aux_for_step(current, target_seq, metadata, 0)
                with trainer._autocast_context():
                    _ = trainer._forward_model_step(previous, current, aux, lead=1, diagnostics_collector=self)
        finally:
            if was_training:
                trainer.model.train()
        if bool(self.config.get("collect_attention", True)) and not self.attention_rows and not self._warned_attention_missing:
            self.logger.warning("Attention diagnostics skipped: attention weights are not exposed by this model path.")
            self._warned_attention_missing = True

    def observe_gradients(
        self,
        model: torch.nn.Module,
        *,
        step: int,
        epoch: int,
        stage: str = "pre_clip",
    ) -> dict[str, float]:
        if not self.enabled:
            return {}
        raw = gradient_metrics(model)
        metrics: dict[str, float] = {}
        if stage == "pre_clip":
            metrics["train/grad_norm/global"] = raw.get("global", float("nan"))
            metrics["train/grad_norm/pre_clip"] = raw.get("global", float("nan"))
        elif stage == "post_clip":
            metrics["train/grad_norm/post_clip"] = raw.get("global", float("nan"))
        else:
            metrics[f"train/grad_norm/{stage}/global"] = raw.get("global", float("nan"))
        for name, value in raw.items():
            if name == "global":
                continue
            metrics[f"train/grad_norm/{name}"] = value
            metrics[f"train/grad_norm/{stage}/{name}"] = value
        metrics["train/global_step"] = float(step)
        metrics["train/epoch"] = float(epoch)
        self._latest_gradient_metrics = metrics
        if self.is_rank0:
            self._append_gradient_csv(epoch=epoch, step=step, stage=stage, raw=raw)
            if bool((self.config.get("wandb", {}) or {}).get("log_scalars", True)):
                wandb_log_metrics(self.wandb_run, metrics, step=int(step))
        return metrics

    def _append_gradient_csv(self, *, epoch: int, step: int, stage: str, raw: dict[str, float]) -> None:
        if not self.output_dir:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "gradient_norms.csv"
        exists = path.exists()
        fields = ["epoch", "step", "stage", "block_name", "grad_norm"]
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                writer.writeheader()
            for name, value in sorted(raw.items()):
                writer.writerow(
                    {
                        "epoch": int(epoch),
                        "step": int(step),
                        "stage": str(stage),
                        "block_name": str(name),
                        "grad_norm": float(value),
                    }
                )

    def _rollout_diagnostics(
        self,
        trainer: Any,
        loader: Any,
        *,
        phase: str,
        epoch: int,
        max_batches: int,
    ) -> dict[str, float]:
        horizons = sorted({int(x) for x in self.config.get("rollout_horizons", [1, 2, 4, 6, 8, 10]) if int(x) > 0})
        if not horizons:
            return {}
        max_horizon = max(horizons)
        step_sums = np.zeros((max_horizon,), dtype=np.float64)
        step_counts = np.zeros((max_horizon,), dtype=np.float64)
        batches = 0
        samples = 0
        spatial_accumulator = self._init_spatial_map_accumulator(trainer)
        for batch_idx, data in enumerate(loader):
            if batch_idx >= int(max_batches):
                break
            inp, target, metadata = trainer._to_device_batch(data)
            target_seq = trainer._target_sequence(target)
            available_steps = min(max_horizon, int(target_seq.shape[1]))
            if available_steps <= 0:
                continue
            bsz = int(inp.shape[0])
            previous, current = trainer.model.adapter.extract_two_steps(inp)
            initial_state = current
            with trainer._autocast_context():
                for step in range(available_steps):
                    lead = step + 1
                    gt = target_seq[:, step]
                    aux = trainer._build_aux_for_step(current, target_seq, metadata, step)
                    pred = trainer._forward_model_step(previous, current, aux, lead=lead)
                    feature_builder = getattr(trainer, "feature_builder", None)
                    if feature_builder is not None:
                        pred = feature_builder.apply_overrides(pred, current=current, target_norm=gt)
                    target_handler = getattr(trainer, "target_handler", None)
                    if target_handler is not None:
                        pred = target_handler.apply(
                            pred_next=pred,
                            current_state=current,
                            initial_state=initial_state,
                            target_sequence=target_seq,
                            lead=lead,
                        )
                    self._accumulate_spatial_maps(
                        spatial_accumulator,
                        pred=pred,
                        target=gt,
                        lead=lead,
                        batch_idx=batch_idx,
                    )
                    loss = trainer._step_loss(pred, gt)
                    step_sums[step] += float(loss.detach().item()) * bsz
                    step_counts[step] += bsz
                    next_step = current.clone()
                    next_step[:, : trainer.model.output_channels] = pred
                    previous, current = current, next_step
            batches += 1
            samples += bsz
        metrics: dict[str, float] = {}
        prefix = "train_diag" if phase.startswith("train") else phase
        step_losses = np.divide(
            step_sums,
            np.maximum(step_counts, 1.0),
            out=np.full_like(step_sums, np.nan),
            where=step_counts > 0,
        )
        for horizon in horizons:
            valid = step_counts[:horizon] > 0
            if horizon <= len(step_losses) and bool(valid.all()):
                loss_mean = float(np.mean(step_losses[:horizon]))
                loss_final = float(step_losses[horizon - 1])
            else:
                loss_mean = float("nan")
                loss_final = float("nan")
            ratio = loss_final / (loss_mean + 1.0e-12) if np.isfinite(loss_mean) and np.isfinite(loss_final) else float("nan")
            metrics[f"{prefix}/rollout/S{horizon}/loss_mean"] = loss_mean
            metrics[f"{prefix}/rollout/S{horizon}/loss_final"] = loss_final
            metrics[f"{prefix}/rollout/S{horizon}/final_to_mean_ratio"] = float(ratio)
            for step in range(1, horizon + 1):
                value = float(step_losses[step - 1]) if step - 1 < len(step_losses) and step_counts[step - 1] > 0 else float("nan")
                self.rollout_rows.append(
                    {
                        "epoch": int(epoch),
                        "phase": phase,
                        "horizon": int(horizon),
                        "step": int(step),
                        "loss": value,
                    }
                )
        metrics[f"{prefix}/rollout/batches"] = float(batches)
        metrics[f"{prefix}/rollout/samples"] = float(samples)
        self.spatial_map_items = self._finalize_spatial_maps(spatial_accumulator)
        return metrics

    def _resolve_spectral_variables(self, trainer: Any) -> list[tuple[str, int]]:
        requested = list(self.config.get("spectral_variables", []) or [])
        return self._resolve_named_variables(trainer, requested, diagnostic_name="Spectral")

    def _resolve_named_variables(
        self,
        trainer: Any,
        requested: list[Any],
        *,
        diagnostic_name: str,
    ) -> list[tuple[str, int]]:
        if not requested:
            return []
        try:
            from ..features import VariableResolver

            resolver = VariableResolver(
                trainer.params,
                getattr(trainer.train_dataset, "channel_names", None),
                list(getattr(trainer.params, "out_channels", [])),
                logger=self.logger,
            )
            resolved = []
            for variable in requested:
                item = resolver.resolve(str(variable), required=False)
                if item.local_index is None:
                    self.logger.warning("%s diagnostics skipped variable %s: not present in output channels.", diagnostic_name, variable)
                    continue
                resolved.append((str(item.canonical), int(item.local_index)))
            return resolved
        except Exception as exc:
            self.logger.warning("%s variable resolution failed: %s", diagnostic_name, exc)
            result = []
            for variable in requested:
                try:
                    result.append((str(variable), int(variable)))
                except Exception:
                    continue
            return result

    def _grid_shape(self, trainer: Any) -> tuple[int, int] | None:
        configured = self.config.get("grid_shape", None)
        if configured:
            return (int(configured[0]), int(configured[1]))
        graph = getattr(getattr(trainer, "graph", None), "L0", None)
        if graph is not None and getattr(graph, "height", None) and getattr(graph, "width", None):
            return (int(graph.height), int(graph.width))
        height = getattr(trainer.params, "crop_size_x", None)
        width = getattr(trainer.params, "crop_size_y", None)
        if height is not None and width is not None:
            return (int(height), int(width))
        return None

    def _as_grid(self, tensor: torch.Tensor, grid_shape: tuple[int, int] | None) -> torch.Tensor:
        x = tensor.detach().float()
        if x.dim() == 4:
            return x
        if x.dim() == 3 and grid_shape is not None:
            height, width = int(grid_shape[0]), int(grid_shape[1])
            if int(x.shape[1]) != height * width:
                raise ValueError(f"Tensor node count {x.shape[1]} does not match grid_shape={grid_shape}.")
            return x.reshape(x.shape[0], height, width, x.shape[2]).permute(0, 3, 1, 2)
        raise ValueError(f"Cannot convert tensor with shape {tuple(x.shape)} to grid.")

    def _init_spatial_map_accumulator(self, trainer: Any) -> dict[str, Any] | None:
        if not self._active_spatial_maps_enabled:
            return None
        plots = self._plots_config()
        requested = list(plots.get("variables", []) or [])
        variables = self._resolve_named_variables(trainer, requested, diagnostic_name="Spatial map")
        if not variables:
            return None
        horizons = sorted({int(x) for x in plots.get("map_horizons", [1, 4, 10]) if int(x) > 0})
        if not horizons:
            return None
        return {
            "variables": variables,
            "horizons": horizons,
            "max_batches": max(1, int(plots.get("max_map_batches", 2))),
            "max_samples": max(1, int(plots.get("max_map_samples", 4))),
            "grid_shape": self._grid_shape(trainer),
            "data": {},
        }

    def _accumulate_spatial_maps(
        self,
        accumulator: dict[str, Any] | None,
        *,
        pred: torch.Tensor,
        target: torch.Tensor,
        lead: int,
        batch_idx: int,
    ) -> None:
        if accumulator is None:
            return
        if int(lead) not in set(accumulator["horizons"]):
            return
        if int(batch_idx) >= int(accumulator["max_batches"]):
            return
        try:
            pred_grid = self._as_grid(pred, accumulator.get("grid_shape"))
            target_grid = self._as_grid(target, accumulator.get("grid_shape"))
        except Exception as exc:
            if not self._warned_spatial_skip:
                self.logger.warning("Skipped spatial maps: cannot infer grid shape or reshape predictions: %s", exc)
                self._warned_spatial_skip = True
            return
        if tuple(pred_grid.shape) != tuple(target_grid.shape):
            if not self._warned_spatial_skip:
                self.logger.warning("Skipped spatial maps: pred/target shape mismatch %s != %s", tuple(pred_grid.shape), tuple(target_grid.shape))
                self._warned_spatial_skip = True
            return
        for variable, idx in accumulator["variables"]:
            if int(idx) < 0 or int(idx) >= int(pred_grid.shape[1]):
                continue
            key = (int(lead), str(variable))
            entry = accumulator["data"].setdefault(
                key,
                {
                    "sum_error": None,
                    "sum_sq_error": None,
                    "count": 0,
                    "example_error": None,
                },
            )
            remaining = int(accumulator["max_samples"]) - int(entry["count"])
            if remaining <= 0:
                continue
            take = min(int(pred_grid.shape[0]), remaining)
            err = pred_grid[:take, int(idx)] - target_grid[:take, int(idx)]
            sum_error = err.sum(dim=0).detach().cpu().numpy().astype(np.float64)
            sum_sq_error = err.square().sum(dim=0).detach().cpu().numpy().astype(np.float64)
            if entry["sum_error"] is None:
                entry["sum_error"] = sum_error
                entry["sum_sq_error"] = sum_sq_error
            else:
                entry["sum_error"] = entry["sum_error"] + sum_error
                entry["sum_sq_error"] = entry["sum_sq_error"] + sum_sq_error
            entry["count"] = int(entry["count"]) + int(take)
            if entry["example_error"] is None and take > 0:
                entry["example_error"] = err[0].detach().cpu().numpy().astype(np.float64)

    def _finalize_spatial_maps(self, accumulator: dict[str, Any] | None) -> list[dict[str, Any]]:
        if accumulator is None:
            return []
        items: list[dict[str, Any]] = []
        for (horizon, variable), entry in sorted(accumulator["data"].items()):
            count = int(entry.get("count", 0))
            if count <= 0:
                continue
            bias = np.asarray(entry["sum_error"], dtype=np.float64) / float(count)
            rmse = np.sqrt(np.asarray(entry["sum_sq_error"], dtype=np.float64) / float(count))
            items.append({"variable": variable, "horizon": int(horizon), "kind": "bias", "array": bias})
            items.append({"variable": variable, "horizon": int(horizon), "kind": "rmse", "array": rmse})
            if entry.get("example_error") is not None:
                items.append(
                    {
                        "variable": variable,
                        "horizon": int(horizon),
                        "kind": "example_error",
                        "array": np.asarray(entry["example_error"], dtype=np.float64),
                    }
                )
        return items

    def _spectral_diagnostics(
        self,
        trainer: Any,
        loader: Any,
        *,
        phase: str,
        epoch: int,
        max_batches: int,
    ) -> tuple[dict[str, float], dict[tuple[int, str], SpectralCurve]]:
        del phase
        if not bool(self.config.get("spectral_rmse", True)):
            return {}, {}
        variables = self._resolve_spectral_variables(trainer)
        if not variables:
            return {}, {}
        rollout_horizons = sorted({int(x) for x in self.config.get("rollout_horizons", [10]) if int(x) > 0})
        if not rollout_horizons:
            return {}, {}
        final_horizon = max(rollout_horizons)
        plots = self._plots_config()
        configured_horizons = list(plots.get("spectral_horizons", []) or [])
        selected_horizons = {int(x) for x in configured_horizons if int(x) > 0} if self._active_visuals_enabled else set()
        selected_horizons.add(final_horizon)
        selected_horizons = set(h for h in selected_horizons if h > 0)
        max_horizon = max(selected_horizons)
        grid_shape = self._grid_shape(trainer)
        curves_by_key: dict[tuple[int, str], list[SpectralCurve]] = {
            (horizon, name): [] for horizon in sorted(selected_horizons) for name, _ in variables
        }
        for batch_idx, data in enumerate(loader):
            if batch_idx >= int(max_batches):
                break
            inp, target, metadata = trainer._to_device_batch(data)
            target_seq = trainer._target_sequence(target)
            available_horizons = sorted(h for h in selected_horizons if h <= int(target_seq.shape[1]))
            if not available_horizons:
                continue
            previous, current = trainer.model.adapter.extract_two_steps(inp)
            initial_state = current
            preds_by_horizon: dict[int, torch.Tensor] = {}
            with trainer._autocast_context():
                for step in range(max(available_horizons)):
                    lead = step + 1
                    gt = target_seq[:, step]
                    aux = trainer._build_aux_for_step(current, target_seq, metadata, step)
                    pred = trainer._forward_model_step(previous, current, aux, lead=lead)
                    feature_builder = getattr(trainer, "feature_builder", None)
                    if feature_builder is not None:
                        pred = feature_builder.apply_overrides(pred, current=current, target_norm=gt)
                    target_handler = getattr(trainer, "target_handler", None)
                    if target_handler is not None:
                        pred = target_handler.apply(
                            pred_next=pred,
                            current_state=current,
                            initial_state=initial_state,
                            target_sequence=target_seq,
                            lead=lead,
                        )
                    if lead in available_horizons:
                        preds_by_horizon[lead] = pred.detach()
                    next_step = current.clone()
                    next_step[:, : trainer.model.output_channels] = pred
                    previous, current = current, next_step
            if not preds_by_horizon:
                continue
            for horizon, pred_at_horizon in preds_by_horizon.items():
                try:
                    curves = spectral_rmse_2d(
                        pred_at_horizon,
                        target_seq[:, horizon - 1],
                        [idx for _, idx in variables],
                        grid_shape=grid_shape,
                    )
                except Exception as exc:
                    if not self._warned_spectral_skip:
                        self.logger.warning(
                            "Spectral diagnostics skipped: cannot infer grid shape or FFT fields. "
                            "pred_shape=%s target_shape=%s grid_shape=%s error=%s",
                            tuple(pred_at_horizon.shape),
                            tuple(target_seq[:, horizon - 1].shape),
                            grid_shape,
                            exc,
                        )
                        self._warned_spectral_skip = True
                    return {}, {}
                for name, idx in variables:
                    if idx in curves:
                        curves_by_key[(int(horizon), name)].append(curves[idx])
        averaged: dict[tuple[int, str], SpectralCurve] = {}
        metrics: dict[str, float] = {}
        for (horizon, name), curves in curves_by_key.items():
            if not curves:
                continue
            max_len = max(curve.rmse.size for curve in curves)
            sums = np.zeros((max_len,), dtype=np.float64)
            counts = np.zeros((max_len,), dtype=np.float64)
            bins = np.arange(max_len, dtype=np.int64)
            for curve in curves:
                length = curve.rmse.size
                sums[:length] += curve.rmse
                counts[:length] += 1.0
                bins[:length] = curve.bins[:length]
            rmse = np.divide(sums, np.maximum(counts, 1.0), out=np.full_like(sums, np.nan), where=counts > 0)
            averaged[(int(horizon), name)] = SpectralCurve(bins=bins, rmse=rmse)
            bands = spectral_band_summary(averaged[(int(horizon), name)])
            for key, value in bands.items():
                metrics[f"spectral/S{int(horizon)}/{name}/{key}"] = float(value)
                if int(horizon) == int(final_horizon):
                    metrics[f"spectral/final/{name}/{key}"] = float(value)
                    metrics[f"diagnostics_spectral/{name}/{key}"] = float(value)
                self.spectral_band_rows.append(
                    {
                        "epoch": int(epoch),
                        "phase": self._active_phase,
                        "horizon": int(horizon),
                        "variable": str(name),
                        "band": str(key).replace("_rmse", ""),
                        "spectral_rmse": float(value),
                    }
                )
            if int(horizon) == int(final_horizon):
                low = float(bands.get("low_k_rmse", float("nan")))
                mid = float(bands.get("mid_k_rmse", float("nan")))
                high = float(bands.get("high_k_rmse", float("nan")))
                metrics[f"diagnostics_spectral/{name}/low_to_high_ratio"] = (
                    float(low / high) if np.isfinite(low) and np.isfinite(high) and abs(high) > 1.0e-12 else float("nan")
                )
                metrics[f"diagnostics_spectral/{name}/low_to_mid_ratio"] = (
                    float(low / mid) if np.isfinite(low) and np.isfinite(mid) and abs(mid) > 1.0e-12 else float("nan")
                )
            for bin_idx, value in zip(bins.tolist(), rmse.tolist()):
                self.spectral_curve_rows.append(
                    {
                        "epoch": int(epoch),
                        "phase": self._active_phase,
                        "horizon": int(horizon),
                        "variable": str(name),
                        "wavenumber_bin": int(bin_idx),
                        "spectral_rmse": float(value),
                    }
                )
        return metrics, averaged

    @torch.no_grad()
    def run_train_diagnostics(self, *, epoch: int, trainer: Any, heavy: bool = False) -> dict[str, float]:
        return self._run_diagnostics(epoch=epoch, trainer=trainer, loader=trainer.train_data_loader, phase="train_diag", heavy=heavy)

    @torch.no_grad()
    def run_valid_diagnostics(self, *, epoch: int, trainer: Any, heavy: bool = False) -> dict[str, float]:
        return self._run_diagnostics(epoch=epoch, trainer=trainer, loader=trainer.valid_data_loader, phase="valid", heavy=heavy)

    @torch.no_grad()
    def run_full_post_training_diagnostics(
        self,
        *,
        trainer: Any,
        split: str = "valid",
        output_dir: str | None = None,
    ) -> dict[str, float]:
        if output_dir is not None:
            self.output_dir = Path(output_dir).expanduser()
            if self.is_rank0:
                self.output_dir.mkdir(parents=True, exist_ok=True)
        max_batches = int(self.config.get("max_full_diag_batches", 32))
        loader = trainer.valid_data_loader
        if str(split).lower() == "test":
            test_path = getattr(trainer.params, "test_dataset_path", None) or getattr(trainer.params, "test_data_path", None)
            if test_path:
                try:
                    from dataclasses import replace

                    from ..data import build_data_loader

                    max_rollout = max(int(x) for x in self.config.get("rollout_horizons", [trainer.max_rollout_steps]))
                    test_cfg = replace(trainer.data_cfg, rollout_steps=int(max_rollout))
                    loader, _ = build_data_loader(test_cfg, test_path, train=False)
                except Exception as exc:
                    self.logger.warning("Test diagnostics loader could not be built; falling back to valid split: %s", exc)
                    loader = trainer.valid_data_loader
            else:
                self.logger.warning("Test diagnostics requested but no test_dataset_path/test_data_path is configured; using valid split.")
        metrics = self._run_diagnostics(
            epoch=int(getattr(trainer, "epoch", 0)),
            trainer=trainer,
            loader=loader,
            phase=split,
            heavy=True,
            max_batches=max_batches,
            is_post_training=True,
        )
        if self.is_rank0:
            path = self.output_dir / "final_full_diagnostics.json"
            with path.open("w", encoding="utf-8") as f:
                json.dump(_jsonable(metrics), f, indent=2, sort_keys=True)
        return metrics

    def _run_diagnostics(
        self,
        *,
        epoch: int,
        trainer: Any,
        loader: Any,
        phase: str,
        heavy: bool,
        max_batches: int | None = None,
        is_post_training: bool = False,
    ) -> dict[str, float]:
        if not self.enabled:
            return {}
        if not self.is_rank0:
            return {}
        start = time.perf_counter()
        self.reset_buffers(epoch, phase)
        plots = self._plots_config()
        self._active_visuals_enabled = self._should_log_visuals(epoch, is_post_training=is_post_training)
        self._active_histograms_enabled = bool(
            self._active_visuals_enabled and (plots.get("attention_plots", True) or plots.get("layer_plots", True))
        )
        self._active_spatial_maps_enabled = bool(self._active_visuals_enabled and plots.get("spatial_maps", True))
        was_training = bool(trainer.model.training)
        trainer.model.eval()
        if max_batches is None:
            max_batches = int(
                self.config.get("max_train_diag_batches" if phase.startswith("train") else "max_valid_diag_batches", 4)
            )
        metrics = self._rollout_diagnostics(
            trainer,
            loader,
            phase=phase,
            epoch=epoch,
            max_batches=max_batches,
        )
        self.collect_layer_metrics_from_forward(trainer, loader, phase=phase, epoch=epoch)
        spectral_curves: dict[tuple[int, str], SpectralCurve] = {}
        if heavy:
            spectral_metrics, spectral_curves = self._spectral_diagnostics(
                trainer,
                loader,
                phase=phase,
                epoch=epoch,
                max_batches=max_batches,
            )
            metrics.update(spectral_metrics)
            try:
                metrics.update(self._power_spectrum_vs_era5(trainer, loader, epoch=epoch, max_batches=max_batches))
            except Exception as exc:
                self.logger.warning("Power-spectrum diagnostics skipped: %s", exc)
        try:
            metrics.update(self._graph_structure_diagnostics())
        except Exception as exc:
            self.logger.warning("Graph-structure diagnostics skipped: %s", exc)
        metrics.update(self._layer_scalar_metrics())
        metrics.update(self._attention_scalar_metrics())
        metrics.update(self._latest_gradient_metrics)
        metrics["system/diagnostics_time_sec"] = float(time.perf_counter() - start)
        if self.device.type == "cuda":
            metrics["system/gpu_peak_allocated_mb"] = float(torch.cuda.max_memory_allocated(self.device) / (1024.0 ** 2))
            metrics["system/gpu_peak_reserved_mb"] = float(torch.cuda.max_memory_reserved(self.device) / (1024.0 ** 2))
        if was_training:
            trainer.model.train()
        self.log_metrics(metrics, epoch=epoch)
        self._save_epoch_outputs(epoch=epoch, metrics=metrics, spectral_curves=spectral_curves)
        self._log_visual_outputs(epoch=epoch, metrics=metrics, trainer=trainer, is_post_training=is_post_training)
        self._print_summary(epoch=epoch, metrics=metrics)
        return metrics

    @staticmethod
    def _layer_summary_aliases(layer_name: str) -> list[str]:
        layer = _safe_name(str(layer_name)).lower()
        aliases: list[str] = []
        if "unpool10" in layer or "unpool_10" in layer or ("unpool" in layer and "10" in layer):
            aliases.append("unpool10")
        if "l0_refine" in layer or ("l0" in layer and "refine" in layer):
            aliases.append("l0_refine")
        if "processor_output" in layer or ("processor" in layer and "output" in layer):
            aliases.append("processor_output")
        return aliases

    @staticmethod
    def _attention_summary_aliases(layer_name: str) -> list[str]:
        layer = _safe_name(str(layer_name)).lower()
        aliases: list[str] = []
        if "l0_refine" in layer or ("l0" in layer and "refine" in layer):
            aliases.append("l0_refine")
        if "l3" in layer and ("block" in layer or "processor" in layer or "attn" in layer or "attention" in layer):
            aliases.append("l3_block")
        if "decoder" in layer or layer.startswith("dec") or "decode" in layer:
            aliases.append("decoder")
        return aliases

    def _layer_scalar_metrics(self) -> dict[str, float]:
        metrics = {}
        summary_values: dict[tuple[str, str], float] = {}
        layer_metric_keys = (
            "embedding_variance",
            "cosine_mean",
            "cosine_std",
            "mad_cosine",
            "effective_rank",
            "effective_rank_norm",
            "stable_rank",
        )
        for row in self.layer_rows:
            layer = _safe_name(str(row["layer_name"]))
            for key in layer_metric_keys:
                value = row.get(key)
                if value is not None:
                    metrics[f"model/layer/{layer}/{key}"] = float(value)
                    metrics[f"diagnostics_layer/{layer}/{key}"] = float(value)
                    for alias in self._layer_summary_aliases(layer):
                        summary_values[(alias, key)] = float(value)
        if self.layer_rows:
            deepest = self.layer_rows[-1]
            for key in layer_metric_keys:
                value = deepest.get(key)
                if value is not None:
                    summary_values[("deepest_processor", key)] = float(value)
        for (alias, key), value in summary_values.items():
            metrics[f"diagnostics_layer_summary/{alias}/{key}"] = float(value)
            previous = self._previous_layer_summary_metrics.get((alias, key))
            if previous is not None and np.isfinite(previous) and np.isfinite(value):
                metrics[f"diagnostics_layer_delta/{alias}/{key}_delta_prev"] = float(value - previous)
        self._previous_layer_summary_metrics.update(summary_values)
        return metrics

    def _attention_scalar_metrics(self) -> dict[str, float]:
        metrics = {}
        summary_values: dict[tuple[str, str], float] = {}
        for row in self.attention_rows:
            layer = _safe_name(str(row["layer_name"]))
            for key in (
                "entropy",
                "entropy_norm",
                "max_weight_mean",
                "max_weight_std",
                "degree_mean",
                "uniform_baseline_max_weight",
            ):
                value = row.get(key)
                if value is not None:
                    metrics[f"model/attention/{layer}/{key}"] = float(value)
                    metrics[f"diagnostics_attention/{layer}/{key}"] = float(value)
            max_weight = row.get("max_weight_mean")
            uniform = row.get("uniform_baseline_max_weight")
            if max_weight is not None and uniform is not None:
                try:
                    selectivity = float(max_weight) / max(float(uniform), 1.0e-12)
                except (TypeError, ValueError):
                    selectivity = float("nan")
                metrics[f"diagnostics_attention/{layer}/selectivity_ratio"] = selectivity
            for alias in self._attention_summary_aliases(layer):
                for key in ("entropy_norm", "max_weight_mean"):
                    value = row.get(key)
                    if value is not None:
                        summary_values[(alias, key)] = float(value)
        for (alias, key), value in summary_values.items():
            metrics[f"diagnostics_attention_summary/{alias}/{key}"] = float(value)
            previous = self._previous_attention_summary_metrics.get((alias, key))
            if previous is not None and np.isfinite(previous) and np.isfinite(value):
                metrics[f"diagnostics_attention_delta/{alias}/{key}_delta_prev"] = float(value - previous)
        self._previous_attention_summary_metrics.update(summary_values)
        return metrics

    def log_metrics(self, metrics: dict[str, float], epoch: int) -> None:
        if not self.enabled or not self.is_rank0:
            return
        payload = dict(metrics)
        payload["diagnostics/epoch"] = float(epoch)
        if bool((self.config.get("wandb", {}) or {}).get("log_scalars", True)):
            wandb_log_metrics(self.wandb_run, payload, step=int(epoch))

    def save_system_metrics(self, epoch: int, metrics: dict[str, float]) -> None:
        if not self.enabled or not self.is_rank0:
            return
        metrics = dict(metrics)
        for key, value in list(metrics.items()):
            if key.startswith("system/"):
                metrics[f"diagnostics_system/{key.split('/', 1)[1]}"] = value
        self.log_metrics(metrics, epoch=epoch)
        path = self.output_dir / f"epoch_{int(epoch):04d}_scalars.json"
        payload = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        payload.update({key: _jsonable(value) for key, value in metrics.items()})
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def _save_epoch_outputs(
        self,
        *,
        epoch: int,
        metrics: dict[str, float],
        spectral_curves: dict[tuple[int, str], SpectralCurve],
    ) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tables_dir = self.output_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        stem = f"epoch_{int(epoch):04d}"
        if self._active_phase != "valid":
            stem = f"{stem}_{_safe_name(self._active_phase)}"
        with (self.output_dir / f"{stem}_scalars.json").open("w", encoding="utf-8") as f:
            json.dump(_jsonable(metrics), f, indent=2, sort_keys=True)
        self._write_csv(
            self.output_dir / f"{stem}_layer_metrics.csv",
            self.layer_rows,
            [
                "epoch",
                "phase",
                "layer_name",
                "embedding_variance",
                "cosine_mean",
                "cosine_std",
                "mad_cosine",
                "effective_rank",
                "effective_rank_norm",
                "stable_rank",
                "dirichlet_energy",
                "dirichlet_energy_per_edge",
                "dirichlet_energy_norm",
            ],
        )
        self._write_csv(
            tables_dir / f"{stem}_layer_metrics.csv",
            self.layer_rows,
            [
                "epoch",
                "phase",
                "layer_name",
                "embedding_variance",
                "cosine_mean",
                "cosine_std",
                "mad_cosine",
                "effective_rank",
                "effective_rank_norm",
                "stable_rank",
                "dirichlet_energy",
                "dirichlet_energy_per_edge",
                "dirichlet_energy_norm",
            ],
        )
        self._write_csv(
            self.output_dir / f"{stem}_attention_metrics.csv",
            self.attention_rows,
            [
                "epoch",
                "phase",
                "layer_name",
                "entropy",
                "entropy_norm",
                "max_weight_mean",
                "max_weight_std",
                "degree_mean",
                "uniform_baseline_max_weight",
            ],
        )
        self._write_csv(
            tables_dir / f"{stem}_attention_metrics.csv",
            self.attention_rows,
            [
                "epoch",
                "phase",
                "layer_name",
                "entropy",
                "entropy_norm",
                "max_weight_mean",
                "max_weight_std",
                "degree_mean",
                "uniform_baseline_max_weight",
            ],
        )
        self._write_csv(
            tables_dir / f"{stem}_attention_head_metrics.csv",
            self.attention_head_rows,
            ["epoch", "phase", "layer_name", "head", "entropy_norm", "max_weight_mean"],
        )
        self._write_csv(
            self.output_dir / f"{stem}_rollout_curve.csv",
            self.rollout_rows,
            ["epoch", "phase", "horizon", "step", "loss"],
        )
        self._write_csv(
            tables_dir / f"{stem}_rollout_curve.csv",
            self.rollout_rows,
            ["epoch", "phase", "horizon", "step", "loss"],
        )
        self._append_rollout_history(self.rollout_rows)
        self._append_diagnostics_summary_csvs(epoch=epoch, metrics=metrics)
        self._write_csv(
            tables_dir / f"{stem}_spectral_curve.csv",
            self.spectral_curve_rows,
            ["epoch", "phase", "horizon", "variable", "wavenumber_bin", "spectral_rmse"],
        )
        self._write_csv(
            tables_dir / f"{stem}_spectral_bands.csv",
            self.spectral_band_rows,
            ["epoch", "phase", "horizon", "variable", "band", "spectral_rmse"],
        )
        if spectral_curves:
            payload = {}
            final_horizon = max((horizon for horizon, _ in spectral_curves.keys()), default=None)
            for (horizon, name), curve in spectral_curves.items():
                payload[f"S{int(horizon)}_{name}_bins"] = curve.bins
                payload[f"S{int(horizon)}_{name}_rmse"] = curve.rmse
                if final_horizon is not None and int(horizon) == int(final_horizon):
                    payload[f"{name}_bins"] = curve.bins
                    payload[f"{name}_rmse"] = curve.rmse
            np.savez(self.output_dir / f"{stem}_spectral.npz", **payload)
        power_fields = ["epoch", "phase", "lead", "variable", "wavenumber_bin",
                        "power_pred", "power_truth", "power_ratio", "backend"]
        self._write_csv(self.output_dir / f"{stem}_power_spectrum.csv", self.power_spectrum_rows, power_fields)
        self._write_csv(tables_dir / f"{stem}_power_spectrum.csv", self.power_spectrum_rows, power_fields)
        self._write_csv(
            self.output_dir / f"{stem}_variance_ratio.csv",
            self.variance_ratio_rows,
            ["epoch", "phase", "lead", "variable", "variance_ratio", "backend"],
        )
        if self.graph_structure_rows:
            self._write_csv(
                self.output_dir / "graph_structure_metrics.csv",
                self.graph_structure_rows,
                ["level", "num_nodes", "num_edges", "avg_degree",
                 "spectral_gap_lambda2", "mean_effective_resistance", "kirchhoff_index"],
            )
        if self.power_spectrum_rows:
            ps_payload: dict[str, list] = {}
            for row in self.power_spectrum_rows:
                key = f"S{int(row['lead'])}_{_safe_name(str(row['variable']))}"
                ps_payload.setdefault(f"{key}_bins", []).append(int(row["wavenumber_bin"]))
                ps_payload.setdefault(f"{key}_power_pred", []).append(float(row["power_pred"]))
                ps_payload.setdefault(f"{key}_power_truth", []).append(float(row["power_truth"]))
            np.savez(self.output_dir / f"{stem}_power_spectrum.npz",
                     **{k: np.asarray(v) for k, v in ps_payload.items()})

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fields})

    def _append_rollout_history(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        path = self.output_dir / "tables" / "rollout_history.csv"
        exists = path.exists()
        fields = ["epoch", "phase", "horizon", "step", "loss"]
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fields})

    def _append_metric_prefix_csv(self, path: Path, *, epoch: int, prefix: str, metrics: dict[str, float]) -> None:
        rows = [
            {
                "epoch": int(epoch),
                "phase": self._active_phase,
                "metric": key,
                "value": float(value),
            }
            for key, value in sorted(metrics.items())
            if key.startswith(prefix) and isinstance(value, (int, float))
        ]
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists()
        fields = ["epoch", "phase", "metric", "value"]
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _append_diagnostics_summary_csvs(self, *, epoch: int, metrics: dict[str, float]) -> None:
        logs_dir = self.output_dir / "logs"
        self._append_metric_prefix_csv(
            logs_dir / "diagnostics_layer_summary.csv",
            epoch=epoch,
            prefix="diagnostics_layer_summary/",
            metrics=metrics,
        )
        self._append_metric_prefix_csv(
            logs_dir / "diagnostics_attention_summary.csv",
            epoch=epoch,
            prefix="diagnostics_attention_summary/",
            metrics=metrics,
        )
        self._append_metric_prefix_csv(
            logs_dir / "diagnostics_spectral_summary.csv",
            epoch=epoch,
            prefix="diagnostics_spectral/",
            metrics=metrics,
        )

    @staticmethod
    def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        with path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            for key, value in list(row.items()):
                if value is None:
                    continue
                try:
                    if str(value).strip() == "":
                        continue
                    as_float = float(value)
                    row[key] = int(as_float) if as_float.is_integer() and key in {"epoch", "step", "horizon"} else as_float
                except (TypeError, ValueError):
                    pass
        return rows

    def _metric_value(self, metrics: dict[str, float], key: str) -> float | None:
        value = metrics.get(key)
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            return float(value)
        return None

    def _build_summary_row(self, *, epoch: int, metrics: dict[str, float], trainer: Any) -> dict[str, Any]:
        prefix = "train_diag" if self._active_phase.startswith("train") else self._active_phase
        graph_metadata = dict(getattr(getattr(trainer, "graph", None), "metadata", {}) or {})
        params = getattr(trainer, "params", {})
        deepest_layer = self.layer_rows[-1] if self.layer_rows else {}
        deepest_attention = self.attention_rows[-1] if self.attention_rows else {}
        layer_name = _safe_name(str(deepest_layer.get("layer_name", ""))) if deepest_layer else ""
        attention_name = _safe_name(str(deepest_attention.get("layer_name", ""))) if deepest_attention else ""
        row = {
            "run_name": self.run_name,
            "epoch": int(epoch),
            "model_name": str(_cfg_get(params, "nettype", "graph_weather")),
            "hidden_dim": _cfg_get(params, "hidden_dim", None),
            "num_params": getattr(trainer, "num_parameters", None),
            "graph_levels": _cfg_get(params, "num_graph_levels", graph_metadata.get("num_graph_levels")),
            "level_k_neighbors": json.dumps(_jsonable(_cfg_get(params, "level_k_neighbors", graph_metadata.get("level_k_neighbors")))),
            "edge_counts": json.dumps(_jsonable(graph_metadata.get("edge_counts", None))),
            "S1_final": self._metric_value(metrics, f"{prefix}/rollout/S1/loss_final"),
            "S4_final": self._metric_value(metrics, f"{prefix}/rollout/S4/loss_final"),
            "S10_final": self._metric_value(metrics, f"{prefix}/rollout/S10/loss_final"),
            "S10_final_to_mean": self._metric_value(metrics, f"{prefix}/rollout/S10/final_to_mean_ratio"),
            "z500_low_k_rmse": self._metric_value(metrics, "spectral/final/z500/low_k_rmse"),
            "msl_low_k_rmse": self._metric_value(metrics, "spectral/final/msl/low_k_rmse"),
            "t2m_low_k_rmse": self._metric_value(metrics, "spectral/final/t2m/low_k_rmse"),
            "deepest_cosine_mean": self._metric_value(metrics, f"model/layer/{layer_name}/cosine_mean") if layer_name else None,
            "deepest_effective_rank_norm": self._metric_value(metrics, f"model/layer/{layer_name}/effective_rank_norm") if layer_name else None,
            "deepest_attention_entropy_norm": self._metric_value(metrics, f"model/attention/{attention_name}/entropy_norm") if attention_name else None,
            "deepest_attention_max_weight": self._metric_value(metrics, f"model/attention/{attention_name}/max_weight_mean") if attention_name else None,
            "gpu_peak_allocated_mb": self._metric_value(metrics, "system/gpu_peak_allocated_mb"),
            "diagnostics_time_sec": self._metric_value(metrics, "system/diagnostics_time_sec"),
        }
        return row

    def _merge_visual_result(self, target: dict[str, Any], result: dict[str, Any]) -> None:
        for key in ("tables", "plots", "images", "histograms", "warnings"):
            values = result.get(key, []) if result else []
            if isinstance(values, list):
                target.setdefault(key, []).extend(values)

    def _log_visual_outputs(
        self,
        *,
        epoch: int,
        metrics: dict[str, float],
        trainer: Any,
        is_post_training: bool,
    ) -> None:
        if not self._active_visuals_enabled:
            return
        del is_post_training
        plots = self._plots_config()
        step = int(epoch)
        summary: dict[str, Any] = {"tables": [], "plots": [], "images": [], "histograms": [], "warnings": []}
        try:
            if bool(plots.get("rollout_plots", True)):
                self._merge_visual_result(
                    summary,
                    log_rollout_tables_and_plots(
                        self.rollout_rows,
                        config=self.config,
                        run=self.wandb_run,
                        output_dir=self.output_dir,
                        step=step,
                        phase=self._active_phase,
                        epoch=epoch,
                        logger=self.logger,
                    ),
                )
                history_rows = self._read_csv_rows(self.output_dir / "tables" / "rollout_history.csv")
                self._merge_visual_result(
                    summary,
                    log_rollout_heatmap(
                        history_rows,
                        config=self.config,
                        run=self.wandb_run,
                        output_dir=self.output_dir,
                        step=step,
                        phase=self._active_phase,
                        epoch=epoch,
                        logger=self.logger,
                    ),
                )
            if bool(plots.get("layer_plots", True)):
                self._merge_visual_result(
                    summary,
                    log_layer_tables_and_plots(
                        self.layer_rows,
                        config=self.config,
                        run=self.wandb_run,
                        output_dir=self.output_dir,
                        step=step,
                        phase=self._active_phase,
                        epoch=epoch,
                        logger=self.logger,
                    ),
                )
            if bool(plots.get("attention_plots", True)):
                self._merge_visual_result(
                    summary,
                    log_attention_tables_and_plots(
                        self.attention_rows,
                        head_df=self.attention_head_rows,
                        config=self.config,
                        run=self.wandb_run,
                        output_dir=self.output_dir,
                        step=step,
                        phase=self._active_phase,
                        epoch=epoch,
                        logger=self.logger,
                    ),
                )
            if bool(plots.get("attention_plots", True) or plots.get("layer_plots", True)):
                self._merge_visual_result(
                    summary,
                    log_histograms(
                        {
                            "attention": self.attention_histograms,
                            "cosine": self.cosine_histograms,
                        },
                        config=self.config,
                        run=self.wandb_run,
                        output_dir=self.output_dir,
                        step=step,
                        phase=self._active_phase,
                        epoch=epoch,
                        logger=self.logger,
                    ),
                )
            if bool(plots.get("spectral_plots", True)):
                self._merge_visual_result(
                    summary,
                    log_spectral_tables_and_plots(
                        self.spectral_curve_rows,
                        self.spectral_band_rows,
                        config=self.config,
                        run=self.wandb_run,
                        output_dir=self.output_dir,
                        step=step,
                        phase=self._active_phase,
                        epoch=epoch,
                        logger=self.logger,
                    ),
                )
            if bool(plots.get("spatial_maps", True)):
                self._merge_visual_result(
                    summary,
                    log_spatial_maps(
                        self.spatial_map_items,
                        config=self.config,
                        run=self.wandb_run,
                        output_dir=self.output_dir,
                        step=step,
                        phase=self._active_phase,
                        epoch=epoch,
                        logger=self.logger,
                    ),
                )
            if bool(plots.get("optimization_plots", True)):
                gradient_rows = self._read_csv_rows(self.output_dir / "gradient_norms.csv")
                self._merge_visual_result(
                    summary,
                    log_optimization_plots(
                        gradient_rows,
                        config=self.config,
                        run=self.wandb_run,
                        output_dir=self.output_dir,
                        step=step,
                        epoch=epoch,
                        logger=self.logger,
                    ),
                )
            self._merge_visual_result(
                summary,
                log_summary_table(
                    self._build_summary_row(epoch=epoch, metrics=metrics, trainer=trainer),
                    config=self.config,
                    run=self.wandb_run,
                    output_dir=self.output_dir,
                    step=step,
                    epoch=epoch,
                    logger=self.logger,
                ),
            )
        except Exception as exc:
            self.logger.warning("Diagnostics visualization logging skipped: %s", exc)
            summary.setdefault("warnings", []).append(f"Visualization logging failed: {exc}")
        self._print_visual_summary(summary)

    def _print_visual_summary(self, summary: dict[str, Any]) -> None:
        wandb_cfg = dict(self.config.get("wandb", {}) or {})
        wandb_active = self.wandb_run is not None and bool(wandb_cfg.get("enabled", False))

        def yes_no(values: list[Any]) -> str:
            return "yes" if values else "no"

        tables = sorted({str(item).replace("wandb_", "") for item in summary.get("tables", []) if not str(item).startswith("wandb_")})
        plots = sorted({str(item) for item in summary.get("plots", [])})
        self.logger.info("W&B diagnostics logged:")
        self.logger.info("  Scalars: %s", "yes" if wandb_active and bool(wandb_cfg.get("log_scalars", True)) else "no")
        self.logger.info("  Tables: %s", "/".join(tables) if tables else "none")
        self.logger.info("  Plots: %s", "/".join(plots) if plots else "none")
        self.logger.info("  Images: spatial maps %s", yes_no(summary.get("images", [])))
        self.logger.info("  Histograms: attention/cosine %s", yes_no(summary.get("histograms", [])))
        self.logger.info("  Local output: %s", self.output_dir)
        for warning in summary.get("warnings", []):
            self.logger.info("  %s", warning)

    def _print_summary(self, *, epoch: int, metrics: dict[str, float]) -> None:
        def fmt(key: str) -> str:
            value = metrics.get(key, float("nan"))
            return f"{float(value):.6g}" if isinstance(value, (int, float)) and np.isfinite(value) else "nan"

        horizons = [1, 4, 10]
        rollout_parts = []
        for horizon in horizons:
            for prefix in ("valid", "train_diag"):
                key = f"{prefix}/rollout/S{horizon}/loss_final"
                if key in metrics:
                    rollout_parts.append(f"S{horizon} final={fmt(key)}")
                    break
        deepest_layer = self.layer_rows[-1]["layer_name"] if self.layer_rows else None
        deepest_attention = self.attention_rows[-1]["layer_name"] if self.attention_rows else None
        spectral_keys = [key for key in metrics if key.startswith("spectral/final/") and key.endswith("low_k_rmse")]
        spectral_line = "none"
        if spectral_keys:
            base = spectral_keys[0].rsplit("/", 1)[0]
            spectral_line = (
                f"{base.split('/')[-1]} low/mid/high="
                f"{fmt(base + '/low_k_rmse')}/{fmt(base + '/mid_k_rmse')}/{fmt(base + '/high_k_rmse')}"
            )
        self.logger.info("Diagnostics epoch %d:", int(epoch))
        self.logger.info("  Rollout: %s", ", ".join(rollout_parts) if rollout_parts else "none")
        if deepest_layer is not None:
            layer = _safe_name(str(deepest_layer))
            self.logger.info(
                "  Oversmoothing: deepest=%s cosine_mean=%s effective_rank_norm=%s",
                deepest_layer,
                fmt(f"model/layer/{layer}/cosine_mean"),
                fmt(f"model/layer/{layer}/effective_rank_norm"),
            )
        else:
            self.logger.info("  Oversmoothing: no layer metrics")
        if deepest_attention is not None:
            layer = _safe_name(str(deepest_attention))
            self.logger.info(
                "  Attention: deepest=%s entropy_norm=%s max_weight_mean=%s",
                deepest_attention,
                fmt(f"model/attention/{layer}/entropy_norm"),
                fmt(f"model/attention/{layer}/max_weight_mean"),
            )
        else:
            self.logger.info("  Attention: no attention metrics")
        self.logger.info("  Spectral: %s", spectral_line)
        self.logger.info(
            "  System: diag_time=%ss peak_gpu=%sMB",
            fmt("system/diagnostics_time_sec"),
            fmt("system/gpu_peak_allocated_mb"),
        )
