from __future__ import annotations

import logging
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import replace
from typing import Any

import numpy as np
import torch
import torch.amp as amp
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from .data import DataConfig, build_data_loader
from .delta_stats import compute_delta_stats, load_delta_stats
from .graph_builder import (
    build_graph_bundle,
    expected_graph_metadata,
    graph_topology_metadata,
    lat_lon_from_netcdf,
    load_raw_graph_bundle,
    save_graph,
    validate_graph_cache_metadata,
)
from .graph_bundle import load_graph_bundle
from .losses import LatitudeWeightedMSE, graph_gradient_loss
from .models import GraphWeatherModel
from .resolution import get_resolution_spec, resolution_metadata


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _get(params: Any, name: str, default: Any) -> Any:
    return getattr(params, name, default)


def _visible_cuda_device_tokens() -> list[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return [token.strip() for token in raw.split(",") if token.strip()]


def _map_physical_cuda_index_to_visible(index: int) -> int | None:
    tokens = _visible_cuda_device_tokens()
    if not tokens:
        return None
    index_text = str(int(index))
    if index_text in tokens:
        return tokens.index(index_text)
    return None


def _cuda_device_description(device: torch.device) -> str:
    try:
        index = device.index if device.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(index)
        major, minor = torch.cuda.get_device_capability(index)
        return f"{device} ({name}, compute capability sm_{major}{minor})"
    except Exception:
        return str(device)


def _assert_cuda_device_usable(device: torch.device) -> None:
    if device.type != "cuda":
        return
    try:
        with torch.cuda.device(device):
            probe = torch.empty((1,), device=device)
            probe = probe + 1.0
            torch.cuda.synchronize(device)
    except RuntimeError as exc:
        message = str(exc)
        if "no kernel image is available" in message or "invalid device function" in message:
            arch_list = []
            if hasattr(torch.cuda, "get_arch_list"):
                try:
                    arch_list = list(torch.cuda.get_arch_list())
                except Exception:
                    arch_list = []
            raise RuntimeError(
                "Selected CUDA device cannot run CUDA kernels with this PyTorch build: "
                f"{_cuda_device_description(device)}. "
                f"torch.version.cuda={torch.version.cuda}; "
                f"torch CUDA arch list={arch_list or 'unknown'}. "
                "Choose a different GPU, or install/rebuild PyTorch with support for this GPU's compute capability. "
                "When masking a physical GPU with CUDA_VISIBLE_DEVICES, use --device cuda or --device cuda:0."
            ) from exc
        raise


def _resolve_device(device_name: Any, local_rank: int) -> torch.device:
    requested = str(device_name or "auto").strip().lower()
    if requested in {"", "auto"}:
        requested = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    elif requested.isdigit():
        requested = f"cuda:{requested}"
    elif requested == "cuda":
        requested = f"cuda:{local_rank}"

    device = torch.device(requested)
    if device.type != "cuda":
        return device
    if not torch.cuda.is_available():
        raise ValueError(f"Requested training device '{requested}', but CUDA is not available.")
    device_count = torch.cuda.device_count()
    if device.index is not None and device.index >= device_count:
        mapped_index = _map_physical_cuda_index_to_visible(int(device.index))
        if mapped_index is not None and mapped_index < device_count:
            logging.info(
                "Mapped requested CUDA device %s to visible device cuda:%d using CUDA_VISIBLE_DEVICES=%s",
                requested,
                mapped_index,
                os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            )
            return torch.device(f"cuda:{mapped_index}")
        raise ValueError(
            f"Requested training device '{requested}', but only {device_count} CUDA device(s) are visible. "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}."
        )
    return device


def resolve_amp_dtype(enable_amp: bool, amp_dtype: str):
    if not enable_amp or not torch.cuda.is_available():
        return False, None, False

    requested = str(amp_dtype or "auto_bf16_fp16").strip().lower()
    if requested in {"auto", "auto_bf16_fp16"}:
        if torch.cuda.is_bf16_supported():
            return True, torch.bfloat16, False
        return True, torch.float16, True

    if requested in {"bf16", "bfloat16"}:
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 requested but CUDA device does not support bf16")
        return True, torch.bfloat16, False

    if requested in {"fp16", "float16"}:
        return True, torch.float16, True

    raise ValueError(f"Unknown amp_dtype: {amp_dtype}")


def _amp_dtype_name(dtype: torch.dtype | None) -> str:
    if dtype is torch.bfloat16:
        return "bfloat16"
    if dtype is torch.float16:
        return "float16"
    if dtype is None:
        return "disabled"
    return str(dtype).replace("torch.", "")


def warmup_cosine_lr(
    epoch_index_zero_based: int,
    base_lr: float,
    min_lr: float,
    warmup_epochs: int,
    warmup_start_factor: float,
    max_epochs: int,
) -> float:
    epoch = int(epoch_index_zero_based) + 1
    base_lr = float(base_lr)
    min_lr = float(min_lr)
    warmup_epochs = max(0, int(warmup_epochs))
    max_epochs = max(1, int(max_epochs))
    warmup_start_factor = float(warmup_start_factor)

    if warmup_epochs > 0 and epoch <= warmup_epochs:
        alpha = float(epoch) / float(max(1, warmup_epochs))
        factor = warmup_start_factor + alpha * (1.0 - warmup_start_factor)
        return base_lr * factor

    progress = float(epoch - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
    progress = min(max(progress, 0.0), 1.0)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine_factor


class Trainer:
    def __init__(self, params: Any, world_rank: int = 0, local_rank: int = 0):
        self.params = params
        self.world_rank = int(world_rank)
        self.local_rank = int(local_rank)
        self.device = _resolve_device(_get(params, "device", "auto"), self.local_rank)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            _assert_cuda_device_usable(self.device)
            torch.backends.cudnn.benchmark = True
        logging.info("Training device: %s", self.device)
        raw_max_grad_norm = _get(params, "max_gradient_norm", 1.0)
        self.max_gradient_norm = None if raw_max_grad_norm is None else float(raw_max_grad_norm)
        self.gradient_accumulation_steps = max(1, int(_get(params, "gradient_accumulation_steps", 1)))
        self.batch_size = int(_get(params, "batch_size", 1))
        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        self.pin_memory = bool(_get(params, "pin_memory", torch.cuda.is_available()))
        self.num_data_workers = int(_get(params, "num_data_workers", _get(params, "num_workers", 0)))
        self.persistent_workers = bool(_get(params, "persistent_workers", False)) and self.num_data_workers > 0
        prefetch_factor = _get(params, "prefetch_factor", None)
        self.prefetch_factor = None if prefetch_factor is None else int(prefetch_factor)
        self.log_cuda_memory = bool(_get(params, "log_cuda_memory", False))
        self.load_only_current_rollout = bool(_get(params, "load_only_current_rollout", False))
        self.single_pass_multi_horizon_validation = bool(_get(params, "single_pass_multi_horizon_validation", False))
        eval_rollout_steps = _get(params, "eval_rollout_steps", [])
        if eval_rollout_steps is None:
            eval_rollout_steps = []
        self.eval_rollout_steps = sorted({int(x) for x in eval_rollout_steps if int(x) > 0})
        self.iters = 0
        self.epoch = 0
        self.start_epoch = 0
        self.max_train_batches = self._optional_positive_int("max_train_batches")
        self.max_valid_batches = self._optional_positive_int("max_valid_batches")

        rollout_schedule = list(_get(params, "rollout_schedule", [_get(params, "rollout_steps", 1)]))
        self.rollout_schedule = [int(x) for x in rollout_schedule]
        self.max_rollout_steps = int(_get(params, "max_rollout_steps", max(self.rollout_schedule)))
        self.rollout_schedule = [min(int(x), self.max_rollout_steps) for x in self.rollout_schedule]
        self.rollout_stage_epochs = list(_get(params, "rollout_stage_epochs", []))
        self.lr_schedule_type = str(_get(params, "lr_schedule_type", "none")).strip().lower()
        self.base_lr = float(_get(params, "lr", 1.0e-4))
        self.min_lr = float(_get(params, "min_lr", 0.0))
        self.warmup_epochs = int(_get(params, "warmup_epochs", 0))
        self.warmup_start_factor = float(_get(params, "warmup_start_factor", 0.1))
        self.checkpoint_metric = str(_get(params, "checkpoint_metric", "valid_loss"))
        self.checkpoint_mode = str(_get(params, "checkpoint_mode", "min")).lower()
        self.stage_checkpoint_metric_mode = str(_get(params, "stage_checkpoint_metric_mode", "stage_horizon")).lower()
        self.best_score_global = float("inf") if self.checkpoint_mode == "min" else -float("inf")
        self.best_score_by_stage = {
            int(stage): (float("inf") if self.checkpoint_mode == "min" else -float("inf"))
            for stage in sorted(set(self.rollout_schedule))
        }
        self._last_epoch_metadata: dict[str, Any] = {}
        self._current_train_target_rollout_steps: int | None = None
        self._last_stage_rollout_steps: int | None = None

        initial_train_rollout = self._rollout_steps_for_epoch() if self.load_only_current_rollout else self.max_rollout_steps
        max_eval_rollout = max(self.eval_rollout_steps) if self.eval_rollout_steps else self.max_rollout_steps

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
            add_noise=bool(_get(params, "add_noise", False)),
            noise_std=float(_get(params, "noise_std", 0.0)),
            normalize=(str(_get(params, "normalization", "zscore")).lower() == "zscore"),
            normalization=str(_get(params, "normalization", "zscore")),
            global_means_path=params.global_means_path,
            global_stds_path=params.global_stds_path,
            add_grid=bool(_get(params, "add_grid", False)),
            gridtype=str(_get(params, "gridtype", "linear")),
            N_grid_channels=int(_get(params, "N_grid_channels", 2)),
            rollout_steps=int(initial_train_rollout),
            batch_size=self.batch_size,
            num_workers=self.num_data_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            resolution_mode=str(_get(params, "resolution_mode", "5p625")),
            expected_grid_shape=_get(params, "expected_grid_shape", _get(params, "grid_shape", None)),
        )
        self.data_cfg = data_cfg
        self.train_data_loader, self.train_dataset = self._build_train_loader(int(initial_train_rollout))
        valid_cfg = replace(data_cfg, rollout_steps=int(max_eval_rollout))
        self.valid_data_loader, self.valid_dataset = build_data_loader(valid_cfg, params.valid_data_path, train=False)

        params.img_shape_x = self.train_dataset.img_shape_x
        params.img_shape_y = self.train_dataset.img_shape_y
        params.crop_size_x = data_cfg.crop_size_x or self.train_dataset.img_shape_x
        params.crop_size_y = data_cfg.crop_size_y or self.train_dataset.img_shape_y

        self.graph_expected_metadata = self._ensure_graph()
        self.graph_topology_metadata = graph_topology_metadata(self.graph_expected_metadata)
        self.graph = load_graph_bundle(
            params.graph_path,
            map_location="cpu",
            expected_metadata=self.graph_expected_metadata,
        ).to(self.device)

        sample_inp, sample_tar = next(iter(self.train_data_loader))
        out_chans = sample_tar.shape[2] if sample_tar.dim() == 5 else sample_tar.shape[1]
        params.N_in_channels = int(sample_inp.shape[1])
        params.N_out_channels = int(out_chans)
        if len(list(params.out_channels)) != int(out_chans):
            raise AssertionError(f"out_channels length {len(list(params.out_channels))} does not match target channels {out_chans}")
        if int(out_chans) != 67:
            raise AssertionError(f"Expected 67 output channels, got {out_chans}")
        if str(_get(params, "normalization", "zscore")).lower() == "zscore":
            state_mean, state_std = self.train_dataset.output_normalization_vectors()
            if state_mean.shape != (67,) or state_std.shape != (67,):
                raise AssertionError(f"State normalization stats must have shape [67], got {state_mean.shape}/{state_std.shape}")

        self.use_delta_normalization = bool(_get(params, "use_delta_normalization", False))
        self.delta_norm_center = bool(_get(params, "delta_norm_center", False))
        self.delta_norm_eps = float(_get(params, "delta_norm_eps", 1.0e-6))
        self.delta_stats_path = self._resolve_optional_path(_get(params, "delta_stats_path", None))
        delta_mean, delta_std = self._prepare_delta_normalization_stats(int(out_chans))

        self.model = GraphWeatherModel(
            graph=self.graph,
            grid_shape=(int(params.crop_size_x), int(params.crop_size_y)),
            input_channels=int(sample_inp.shape[1]),
            output_channels=int(out_chans),
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
            use_delta_normalization=self.use_delta_normalization,
            delta_mean=delta_mean,
            delta_std=delta_std,
            delta_norm_center=self.delta_norm_center,
            delta_norm_eps=self.delta_norm_eps,
        ).to(self.device)
        self._apply_trainable_scope(str(_get(params, "trainable_scope", "all")))

        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.base_lr,
            weight_decay=float(_get(params, "weight_decay", 1.0e-4)),
        )
        self.amp_enabled, self.amp_dtype, self.scaler_enabled = resolve_amp_dtype(
            bool(_get(params, "enable_amp", False)) and self.device.type == "cuda",
            str(_get(params, "amp_dtype", "auto_bf16_fp16")),
        )
        self.enable_amp = self.amp_enabled
        self.gscaler = amp.GradScaler("cuda", enabled=True) if self.scaler_enabled else None
        self.resolved_amp_dtype_name = _amp_dtype_name(self.amp_dtype)

        l0_lat = self.graph.L0.lat_lon[:, 0].reshape(self.graph.L0.height, self.graph.L0.width)[:, 0]
        self.loss_obj = LatitudeWeightedMSE(l0_lat).to(self.device)
        self.graph_gradient_weight = float(_get(params, "graph_gradient_weight", 0.0))

        scheduler_name = str(_get(params, "scheduler", "CosineAnnealingLR"))
        if self.lr_schedule_type in {"warmup_cosine", "rollout_stage", "none"}:
            self.scheduler = None
        elif scheduler_name == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=int(params.max_epochs),
                last_epoch=self.start_epoch - 1,
            )
        elif scheduler_name == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                factor=0.2,
                patience=5,
                mode="min",
            )
        else:
            self.scheduler = None

        self._warn_lr_config_conflicts()

        if bool(_get(params, "resume", True)) and os.path.isfile(params.checkpoint_path):
            self.restore_checkpoint(params.checkpoint_path)

        self.num_parameters = self.count_parameters()
        logging.info("Number of trainable model parameters: %d", self.num_parameters)
        self._log_startup_config()

    def _optional_positive_int(self, name: str) -> int | None:
        value = _get(self.params, name, None)
        if value is None:
            return None
        value = int(value)
        return value if value > 0 else None

    def _warn_lr_config_conflicts(self) -> None:
        if self.lr_schedule_type == "warmup_cosine" and _get(self.params, "lr_by_rollout", None):
            logging.warning("lr_schedule_type=warmup_cosine: ignoring lr_by_rollout.")

    def _resolve_optional_path(self, value: Any) -> str | None:
        if value is None or str(value).strip() == "":
            return None
        path = os.path.expanduser(str(value))
        return path if os.path.isabs(path) else os.path.abspath(path)

    def _build_train_loader(self, target_rollout_steps: int):
        target_rollout_steps = int(target_rollout_steps)
        train_cfg = replace(self.data_cfg, rollout_steps=target_rollout_steps)
        loader, dataset = build_data_loader(train_cfg, self.params.train_data_path, train=True)
        self._current_train_target_rollout_steps = target_rollout_steps
        return loader, dataset

    def _ensure_train_loader_rollout(self, target_rollout_steps: int) -> None:
        target_rollout_steps = int(target_rollout_steps)
        if not self.load_only_current_rollout:
            return
        if self._current_train_target_rollout_steps == target_rollout_steps:
            return
        logging.info("Rebuilding train loader for rollout target horizon S=%d", target_rollout_steps)
        self.train_data_loader, self.train_dataset = self._build_train_loader(target_rollout_steps)

    def _prepare_delta_normalization_stats(self, output_channels: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.use_delta_normalization:
            logging.info("Delta normalization: disabled")
            return None, None
        if str(_get(self.params, "normalization", "zscore")).lower() != "zscore":
            raise ValueError("Delta normalization requires zscore-normalized state tensors.")
        if not self.delta_stats_path:
            raise ValueError("use_delta_normalization=true requires delta_stats_path.")

        recompute = bool(_get(self.params, "recompute_delta_stats", False))
        if recompute or not os.path.exists(self.delta_stats_path):
            logging.info(
                "Computing one-step delta stats from training split in normalized state space: %s",
                self.delta_stats_path,
            )
            stats_cfg = replace(
                self.data_cfg,
                rollout_steps=1,
                batch_size=max(1, min(self.batch_size, 8)),
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
                prefetch_factor=None,
                roll=False,
                add_noise=False,
                crop_size_x=None,
                crop_size_y=None,
            )
            stats_loader, _ = build_data_loader(stats_cfg, self.params.train_data_path, train=False)
            compute_delta_stats(
                stats_loader,
                self.delta_stats_path,
                max_batches=self._optional_positive_int("delta_stats_max_batches"),
                resolution_mode=str(_get(self.params, "resolution_mode", "5p625")),
                output_channels=int(output_channels),
                n_history=int(self.params.n_history),
                std_floor=max(self.delta_norm_eps * 10.0, 1.0e-12),
            )

        delta_mean, delta_std, metadata = load_delta_stats(
            self.delta_stats_path,
            output_channels=int(output_channels),
            eps=self.delta_norm_eps,
        )
        if tuple(delta_mean.shape) != (67,) or tuple(delta_std.shape) != (67,):
            raise AssertionError(f"Delta stats must have shape [67], got {tuple(delta_mean.shape)}/{tuple(delta_std.shape)}")
        logging.info("Delta normalization: enabled")
        logging.info("Delta stats path: %s", self.delta_stats_path)
        logging.info(
            "Delta std min/max/mean: %.6g / %.6g / %.6g",
            float(delta_std.min().item()),
            float(delta_std.max().item()),
            float(delta_std.mean().item()),
        )
        logging.info("Delta center: %s", self.delta_norm_center)
        if metadata.get("computed_in_normalized_state_space") is not True:
            logging.warning("Delta stats metadata does not explicitly confirm normalized state space: %s", self.delta_stats_path)
        stats_mode = metadata.get("resolution_mode", None)
        if stats_mode and str(stats_mode) != str(_get(self.params, "resolution_mode", "5p625")):
            raise ValueError(
                f"Delta stats resolution_mode={stats_mode} does not match active "
                f"{_get(self.params, 'resolution_mode', '5p625')}"
            )
        return delta_mean, delta_std

    def _load_model_state(self, state: dict[str, torch.Tensor], strict_delta_stats: bool) -> None:
        model_keys = set(self.model.state_dict().keys())
        state_keys = set(state.keys())
        missing = sorted(model_keys - state_keys)
        unexpected = sorted(state_keys - model_keys)
        allowed_missing = {"delta_mean", "delta_std"} if not strict_delta_stats else set()
        bad_missing = [key for key in missing if key not in allowed_missing]
        if unexpected or bad_missing:
            raise RuntimeError(f"Checkpoint model_state mismatch. Missing={bad_missing}; unexpected={unexpected}")
        if strict_delta_stats and ("delta_mean" not in state_keys or "delta_std" not in state_keys):
            raise RuntimeError("Delta normalization is enabled but checkpoint lacks delta_mean/delta_std buffers.")
        self.model.load_state_dict(state, strict=False)

    def _autocast_context(self):
        if not self.amp_enabled:
            return nullcontext()
        return amp.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=True)

    def _to_device_batch(self, data: Any) -> tuple[torch.Tensor, torch.Tensor]:
        non_blocking = bool(self.pin_memory and self.device.type == "cuda")
        inp, target = data
        return (
            inp.to(self.device, dtype=torch.float32, non_blocking=non_blocking),
            target.to(self.device, dtype=torch.float32, non_blocking=non_blocking),
        )

    def _graph_build_options(self) -> tuple[int, float, str, dict[str, Any]]:
        return (
            int(_get(self.params, "k_neighbors", 8)),
            float(_get(self.params, "resolution", 5.625)),
            str(_get(self.params, "graph_connectivity_strategy", "hybrid_row_aware_knn")),
            dict(_get(self.params, "row_aware_knn", {}) or {}),
        )

    def _ensure_graph(self) -> dict[str, Any]:
        graph_path = self.params.graph_path
        k, resolution, strategy, row_aware_knn = self._graph_build_options()
        latitudes, longitudes = lat_lon_from_netcdf(self.params.train_data_path)
        expected = expected_graph_metadata(
            latitudes,
            longitudes,
            k=k,
            resolution=resolution,
            connectivity_strategy=strategy,
            row_aware_knn=row_aware_knn,
            resolution_mode=str(_get(self.params, "resolution_mode", "5p625")),
        )
        if os.path.isfile(graph_path):
            raw = load_raw_graph_bundle(graph_path, map_location="cpu")
            mismatches = validate_graph_cache_metadata(raw, expected)
            if not mismatches:
                return expected
            mismatch_text = "; ".join(mismatches)
            if not bool(_get(self.params, "auto_build_graph", True)):
                raise ValueError(f"Graph cache metadata mismatch and auto_build_graph is disabled: {mismatch_text}")
            logging.warning("Graph cache metadata mismatch; rebuilding %s: %s", graph_path, mismatch_text)
        if not bool(_get(self.params, "auto_build_graph", True)):
            raise FileNotFoundError(f"Graph bundle not found: {graph_path}")
        bundle = build_graph_bundle(
            latitudes,
            longitudes,
            k=k,
            resolution=resolution,
            connectivity_strategy=strategy,
            row_aware_knn=row_aware_knn,
            resolution_mode=str(_get(self.params, "resolution_mode", "5p625")),
        )
        save_graph(bundle, graph_path)
        logging.info("Built graph bundle at %s", graph_path)
        return expected

    def _apply_trainable_scope(self, scope: str) -> None:
        scope = scope.lower()
        if scope == "all":
            return
        for param in self.model.parameters():
            param.requires_grad = False
        if scope in {"head", "output_head"}:
            modules = [self.model.head]
        elif scope in {"decoder_head", "head_decoder"}:
            modules = [self.model.decoder, self.model.head]
        elif scope in {"encoder_decoder_head", "light"}:
            modules = [self.model.encoder, self.model.decoder, self.model.head]
        else:
            raise ValueError(f"Unknown trainable_scope '{scope}'")
        for module in modules:
            for param in module.parameters():
                param.requires_grad = True

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _log_startup_config(self) -> None:
        spec = get_resolution_spec(_get(self.params, "resolution_mode", "5p625"))
        logging.info("Resolution mode: %s", spec.name)
        logging.info("Resolution: %s degrees", spec.resolution_degrees)
        logging.info("Grid: %d x %d", spec.height, spec.width)
        logging.info("Graph levels:")
        for name, level in (("L0", self.graph.L0), ("L1", self.graph.L1), ("L2", self.graph.L2)):
            logging.info("  %s: %d nodes, %d edges", name, level.num_nodes, int(level.edge_index.shape[1]))
        logging.info("Input channels: %d", int(_get(self.params, "N_in_channels", 0)))
        logging.info("Output channels: %d", int(_get(self.params, "N_out_channels", 0)))
        logging.info("Hidden dimension: %d", int(_get(self.params, "hidden_dim", 96)))
        logging.info("Trainable parameters: %d", self.num_parameters)
        logging.info("Batch size: %d", int(_get(self.params, "batch_size", 1)))
        logging.info("Gradient accumulation: %d", self.gradient_accumulation_steps)
        logging.info(
            "Effective batch size: %d",
            self.effective_batch_size,
        )
        logging.info("Max epochs: %d", int(_get(self.params, "max_epochs", 0)))
        logging.info("AMP enabled: %s", self.amp_enabled)
        logging.info("AMP dtype: %s", self.resolved_amp_dtype_name)
        logging.info("GradScaler enabled: %s", self.scaler_enabled)
        logging.info("num_data_workers: %d", self.num_data_workers)
        logging.info("pin_memory: %s", self.pin_memory)
        logging.info("persistent_workers: %s", self.persistent_workers)
        logging.info("prefetch_factor: %s", self.prefetch_factor if self.num_data_workers > 0 else None)
        logging.info("Load only current rollout targets: %s", self.load_only_current_rollout)
        logging.info("Single-pass multi-horizon validation: %s", self.single_pass_multi_horizon_validation)
        logging.info("Training target rollout loaded: %s", self._current_train_target_rollout_steps)
        self._log_lr_schedule_config()

    def _log_lr_schedule_config(self) -> None:
        logging.info("LR schedule type: %s", self.lr_schedule_type)
        if self.lr_schedule_type != "warmup_cosine":
            return
        logging.info("base_lr: %.2e", self.base_lr)
        logging.info("min_lr: %.2e", self.min_lr)
        logging.info("warmup_epochs: %d", self.warmup_epochs)
        logging.info("warmup_start_factor: %.6g", self.warmup_start_factor)
        logging.info("max_epochs: %d", int(_get(self.params, "max_epochs", 0)))
        if _get(self.params, "lr_by_rollout", None):
            logging.info("lr_by_rollout: ignored")
        else:
            logging.info("lr_by_rollout: not configured")
        for epoch in (1, 2, 3, 10, 20, 30, 40, 50):
            if epoch <= int(_get(self.params, "max_epochs", 0)):
                lr = self._warmup_cosine_lr_for_epoch(epoch - 1)
                logging.info("Epoch %-2d lr=%.6e", epoch, lr)

    def _rollout_steps_for_epoch(self) -> int:
        if not self.rollout_stage_epochs:
            idx = min(self.epoch, len(self.rollout_schedule) - 1)
            return min(int(self.rollout_schedule[idx]), self.max_rollout_steps)
        elapsed = 0
        for steps, epochs in zip(self.rollout_schedule, self.rollout_stage_epochs):
            elapsed += int(epochs)
            if self.epoch < elapsed:
                return min(int(steps), self.max_rollout_steps)
        return min(int(self.rollout_schedule[-1]), self.max_rollout_steps)

    def _get_stage_lr(self, rollout_steps: int) -> float:
        lr_by_rollout = _get(self.params, "lr_by_rollout", None)
        base_lr = float(_get(self.params, "lr", 1.0e-4))
        if not lr_by_rollout:
            return base_lr

        key = str(int(rollout_steps))
        if key in lr_by_rollout:
            return float(lr_by_rollout[key])
        if int(rollout_steps) in lr_by_rollout:
            return float(lr_by_rollout[int(rollout_steps)])
        return base_lr

    def _set_optimizer_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(lr)

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _warmup_cosine_lr_for_epoch(self, epoch_index_zero_based: int) -> float:
        return warmup_cosine_lr(
            epoch_index_zero_based,
            base_lr=self.base_lr,
            min_lr=self.min_lr,
            warmup_epochs=self.warmup_epochs,
            warmup_start_factor=self.warmup_start_factor,
            max_epochs=int(_get(self.params, "max_epochs", 1)),
        )

    def _lr_for_epoch(self, rollout_steps: int) -> float:
        if self.lr_schedule_type == "warmup_cosine":
            return self._warmup_cosine_lr_for_epoch(self.epoch)
        if self.lr_schedule_type == "none":
            return self.base_lr
        if self.lr_schedule_type != "rollout_stage":
            return self._current_lr()

        stage_lr = self._get_stage_lr(rollout_steps)
        if self.warmup_epochs <= 0 or self.epoch >= self.warmup_epochs:
            return stage_lr

        start_lr = stage_lr * self.warmup_start_factor
        progress = float(self.epoch + 1) / float(max(1, self.warmup_epochs))
        return start_lr + (stage_lr - start_lr) * min(1.0, progress)

    def _target_sequence(self, target: torch.Tensor) -> torch.Tensor:
        return target if target.dim() == 5 else target.unsqueeze(1)

    def _step_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        loss = self.loss_obj(pred, gt)
        if self.graph_gradient_weight > 0.0:
            loss = loss + self.graph_gradient_weight * graph_gradient_loss(
                pred,
                gt,
                self.graph.L0.edge_index,
            )
        return loss

    def _rollout_loss(self, inp: torch.Tensor, target: torch.Tensor, rollout_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        target_seq = self._target_sequence(target)
        requested_steps = int(rollout_steps)
        if target_seq.shape[1] < requested_steps:
            raise AssertionError(f"Target length {target_seq.shape[1]} is shorter than rollout_steps={requested_steps}")
        rollout_steps = requested_steps
        previous, current = self.model.adapter.extract_two_steps(inp)
        total = torch.zeros((), device=inp.device, dtype=inp.dtype)
        last_pred = None
        for step in range(rollout_steps):
            pred = self.model.forward_steps(previous, current)
            gt = target_seq[:, step]
            loss = self._step_loss(pred, gt)
            total = total + loss
            next_step = current.clone()
            next_step[:, : self.model.output_channels] = pred
            previous, current = current, next_step
            last_pred = pred
        return total / rollout_steps, last_pred

    def _metric_value(self, logs: dict[str, float], metric_name: str) -> float:
        if metric_name in logs:
            return float(logs[metric_name])
        logging.warning("Metric '%s' missing from validation logs. Falling back to valid_loss.", metric_name)
        return float(logs["valid_loss"])

    def _is_better(self, score: float, best_score: float) -> bool:
        if self.checkpoint_mode == "max":
            return score > best_score
        return score < best_score

    def _stage_checkpoint_metric_name(self, stage: int) -> str:
        if self.stage_checkpoint_metric_mode == "final_horizon":
            return f"valid_S{self.max_rollout_steps}"
        if self.stage_checkpoint_metric_mode == "stage_horizon":
            return f"valid_S{stage}"
        return f"valid_S{stage}"

    def _stage_checkpoint_path(self, stage: int) -> str:
        experiment_dir = str(_get(self.params, "experiment_dir", os.path.dirname(self.params.best_checkpoint_path)))
        return os.path.join(experiment_dir, f"best_ckpt_S{int(stage)}.tar")

    def train(self) -> None:
        for epoch in range(self.start_epoch, int(self.params.max_epochs)):
            start = time.time()
            if self.log_cuda_memory and self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            train_rollout_steps = self._rollout_steps_for_epoch()
            stage_changed = self._last_stage_rollout_steps != train_rollout_steps
            if stage_changed:
                logging.info(
                    "Rollout stage transition: S=%d | rebuilding_train_loader=%s | batch_size=%d | "
                    "accumulation=%d | effective_batch=%d",
                    train_rollout_steps,
                    bool(self.load_only_current_rollout),
                    self.batch_size,
                    self.gradient_accumulation_steps,
                    self.effective_batch_size,
                )
                self._last_stage_rollout_steps = train_rollout_steps
            self._ensure_train_loader_rollout(train_rollout_steps)
            epoch_lr = self._lr_for_epoch(train_rollout_steps)
            if self.lr_schedule_type in {"warmup_cosine", "rollout_stage", "none"}:
                self._set_optimizer_lr(epoch_lr)
            if self.lr_schedule_type == "rollout_stage":
                logging.info("Stage learning rate for S=%d: %.6g", train_rollout_steps, epoch_lr)
            elif self.lr_schedule_type == "warmup_cosine":
                logging.info("Warmup/cosine learning rate for epoch %d: %.6g", epoch + 1, epoch_lr)
            if stage_changed:
                logging.info(
                    "Stage transition settings | S=%d | target_horizon=%d | batch_size=%d | accumulation=%d | "
                    "effective_batch=%d | lr=%.6g | scheduler=%s",
                    train_rollout_steps,
                    int(self._current_train_target_rollout_steps or train_rollout_steps),
                    self.batch_size,
                    self.gradient_accumulation_steps,
                    self.effective_batch_size,
                    epoch_lr,
                    self.lr_schedule_type,
                )

            tr_time, train_logs = self.train_one_epoch()
            train_logs["train_time_sec"] = float(tr_time)
            train_rollout_steps = int(train_logs["rollout_steps"])

            if bool(_get(self.params, "validate_with_train_rollout", True)):
                valid_rollout_steps = train_rollout_steps
            else:
                valid_rollout_steps = None
            valid_time, valid_logs = self.validate_one_epoch(rollout_steps=valid_rollout_steps)
            valid_logs["valid_time_sec"] = float(valid_time)

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(valid_logs["valid_loss"])
                else:
                    self.scheduler.step()
                epoch_lr = self._current_lr()

            valid_multi = {
                key[6:]: value
                for key, value in valid_logs.items()
                if key.startswith("valid_S")
            }
            global_metric_name = self.checkpoint_metric
            global_metric_value = self._metric_value(valid_logs, global_metric_name)
            stage_metric_name = self._stage_checkpoint_metric_name(train_rollout_steps)
            stage_metric_value = self._metric_value(valid_logs, stage_metric_name)
            cuda_peak_allocated_gb = None
            cuda_peak_reserved_gb = None
            if self.log_cuda_memory and self.device.type == "cuda":
                cuda_peak_allocated_gb = torch.cuda.max_memory_allocated(self.device) / (1024.0 ** 3)
                cuda_peak_reserved_gb = torch.cuda.max_memory_reserved(self.device) / (1024.0 ** 3)
            self._last_epoch_metadata = {
                "epoch": self.epoch,
                "train_loss": train_logs["loss"],
                "train_loss_last": train_logs.get("last_loss"),
                "valid_loss": valid_logs["valid_loss"],
                "valid_final": valid_logs.get("valid_final_loss"),
                "train_rollout_steps": train_rollout_steps,
                "target_rollout_steps_loaded": int(train_logs.get("target_rollout_steps_loaded", train_rollout_steps)),
                "valid_rollout_steps": int(valid_logs["valid_rollout_steps"]),
                "epoch_lr": epoch_lr,
                "checkpoint_metric": global_metric_name,
                "checkpoint_metric_value": global_metric_value,
                "stage_checkpoint_metric": stage_metric_name,
                "stage_checkpoint_metric_value": stage_metric_value,
                "valid_multi_horizon": valid_multi,
                "persistence_metrics": {key: value for key, value in valid_logs.items() if key.startswith("persistence_")},
                "cuda_peak_allocated_gb": cuda_peak_allocated_gb,
                "cuda_peak_reserved_gb": cuda_peak_reserved_gb,
                "batch_size": self.batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "effective_batch_size": self.effective_batch_size,
                "enable_amp": self.amp_enabled,
                "amp_dtype": self.resolved_amp_dtype_name,
                "use_delta_normalization": self.use_delta_normalization,
                "delta_stats_path": self.delta_stats_path,
                "delta_norm_center": self.delta_norm_center,
                "max_epochs": int(self.params.max_epochs),
                "single_pass_multi_horizon_validation": self.single_pass_multi_horizon_validation,
                "load_only_current_rollout": self.load_only_current_rollout,
                "lr_schedule_type": self.lr_schedule_type,
                "lr": self.base_lr,
                "min_lr": self.min_lr,
                "warmup_epochs": self.warmup_epochs,
                "warmup_start_factor": self.warmup_start_factor,
                "current_lr": epoch_lr,
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            }
            self._last_epoch_metadata.update(
                resolution_metadata(
                    _get(self.params, "resolution_mode", "5p625"),
                    k=int(_get(self.params, "k_neighbors", 8)),
                    num_parameters=self.num_parameters,
                )
            )

            if bool(_get(self.params, "save_checkpoint", True)):
                last_checkpoint_path = str(_get(self.params, "last_checkpoint_path", ""))
                if last_checkpoint_path:
                    self.save_checkpoint(last_checkpoint_path)
                self.save_checkpoint(self.params.checkpoint_path)
                if self._is_better(global_metric_value, self.best_score_global):
                    self.best_score_global = global_metric_value
                    self.save_checkpoint(self.params.best_checkpoint_path)
                    logging.info(
                        "New global best checkpoint saved: %s using %s=%.6f",
                        self.params.best_checkpoint_path,
                        global_metric_name,
                        global_metric_value,
                    )

                current_stage_best = self.best_score_by_stage.get(
                    train_rollout_steps,
                    float("inf") if self.checkpoint_mode == "min" else -float("inf"),
                )
                if self._is_better(stage_metric_value, current_stage_best):
                    self.best_score_by_stage[train_rollout_steps] = stage_metric_value
                    stage_path = self._stage_checkpoint_path(train_rollout_steps)
                    self.save_checkpoint(stage_path)
                    logging.info(
                        "New best S=%d checkpoint saved: %s using %s=%.6f",
                        train_rollout_steps,
                        os.path.basename(stage_path),
                        stage_metric_name,
                        stage_metric_value,
                    )
            extra_valid = " ".join(
                f"{key} {value:.6f}"
                for key, value in sorted(valid_logs.items())
                if key.startswith("valid_S")
            )
            if extra_valid:
                extra_valid = " | " + extra_valid
            memory_suffix = ""
            if cuda_peak_allocated_gb is not None and cuda_peak_reserved_gb is not None:
                memory_suffix = (
                    f" | peak allocated={cuda_peak_allocated_gb:.2f} GB"
                    f" | peak reserved={cuda_peak_reserved_gb:.2f} GB"
                )
            epoch_time = time.time() - start
            logging.info(
                "Runtime | train_batches=%d | optimizer_steps=%d | train_samples/sec=%.3f | "
                "valid_seconds=%.2f | valid_samples/sec=%.3f | epoch_seconds=%.2f",
                int(train_logs.get("train_batches", 0)),
                int(train_logs.get("optimizer_steps", 0)),
                float(train_logs.get("train_samples_per_second", 0.0)),
                float(valid_time),
                float(valid_logs.get("valid_samples_per_second", 0.0)),
                float(epoch_time),
            )
            logging.info(
                "Epoch %d | mode=%s | grid=%dx%d | train S=%d | finished in %.2f sec | "
                "train_avg %.6f | train_last %.6f | valid %.6f S=%d | valid_final %.6f%s | "
                "lr %.2e | scheduler %s%s",
                epoch + 1,
                _get(self.params, "resolution_mode", "5p625"),
                int(_get(self.params, "crop_size_x", 0)),
                int(_get(self.params, "crop_size_y", 0)),
                train_rollout_steps,
                epoch_time,
                train_logs["loss"],
                train_logs["last_loss"],
                valid_logs["valid_loss"],
                int(valid_logs["valid_rollout_steps"]),
                valid_logs.get("valid_final_loss", float("nan")),
                extra_valid,
                epoch_lr,
                self.lr_schedule_type,
                memory_suffix,
            )
            self._append_epoch_csv(epoch + 1, train_logs, valid_logs, epoch_lr, epoch_time)

    def train_one_epoch(self) -> tuple[float, dict[str, float]]:
        self.model.train()
        rollout_steps = self._rollout_steps_for_epoch()
        self._ensure_train_loader_rollout(rollout_steps)
        start = time.time()
        last_loss = float("nan")
        total_loss = 0.0
        processed = 0
        optimizer_steps = 0
        planned_batches = len(self.train_data_loader)
        if self.max_train_batches is not None:
            planned_batches = min(planned_batches, int(self.max_train_batches))
        self.optimizer.zero_grad(set_to_none=True)
        for batch_idx, data in enumerate(self.train_data_loader):
            if self.max_train_batches is not None and batch_idx >= self.max_train_batches:
                break
            self.iters += 1
            processed += 1
            inp, target = self._to_device_batch(data)
            target_seq = self._target_sequence(target)
            if target_seq.shape[1] != int(rollout_steps):
                raise AssertionError(
                    f"Training target length {target_seq.shape[1]} does not match current rollout S={rollout_steps}"
                )
            with self._autocast_context():
                loss, _ = self._rollout_loss(inp, target, rollout_steps)
                backward_loss = loss / float(self.gradient_accumulation_steps)
            if self.gscaler is not None:
                self.gscaler.scale(backward_loss).backward()
            else:
                backward_loss.backward()
            should_step = (processed % self.gradient_accumulation_steps == 0) or (processed == planned_batches)
            if should_step:
                self._optimizer_step()
                optimizer_steps += 1
            last_loss = float(loss.detach().item())
            total_loss += last_loss
            logging.info("Epoch %d - Batch %d - Loss %.6f", self.epoch + 1, batch_idx, last_loss)
        avg_loss = total_loss / float(max(processed, 1))
        self.epoch += 1
        elapsed = time.time() - start
        samples = processed * self.batch_size
        samples_per_second = float(samples) / elapsed if elapsed > 0.0 else 0.0
        optimizer_steps_per_epoch = int(np.ceil(float(planned_batches) / float(self.gradient_accumulation_steps))) if planned_batches else 0
        logging.info(
            "Train epoch stats | batches=%d | optimizer_steps=%d | planned_optimizer_steps=%d | "
            "samples/sec=%.3f",
            processed,
            optimizer_steps,
            optimizer_steps_per_epoch,
            samples_per_second,
        )
        return elapsed, {
            "loss": avg_loss,
            "last_loss": last_loss,
            "rollout_steps": float(rollout_steps),
            "optimizer_steps": float(optimizer_steps),
            "target_rollout_steps_loaded": float(self._current_train_target_rollout_steps or rollout_steps),
            "train_samples_per_second": samples_per_second,
            "train_batches": float(processed),
        }

    def _optimizer_step(self) -> None:
        if self.gscaler is not None:
            self.gscaler.unscale_(self.optimizer)
            if self.max_gradient_norm is not None:
                clip_grad_norm_(self.model.parameters(), self.max_gradient_norm)
            self.gscaler.step(self.optimizer)
            self.gscaler.update()
        else:
            if self.max_gradient_norm is not None:
                clip_grad_norm_(self.model.parameters(), self.max_gradient_norm)
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def _append_epoch_csv(
        self,
        epoch: int,
        train_logs: dict[str, float],
        valid_logs: dict[str, float],
        lr: float,
        epoch_time_sec: float,
    ) -> None:
        experiment_dir = str(_get(self.params, "experiment_dir", ""))
        if not experiment_dir:
            return
        path = os.path.join(experiment_dir, "epoch_logs.csv")
        exists = os.path.exists(path)
        fields = [
            "epoch",
            "resolution_mode",
            "train_rollout_steps",
            "target_rollout_steps_loaded",
            "batch_size",
            "gradient_accumulation_steps",
            "effective_batch_size",
            "optimizer_steps",
            "train_loss_avg",
            "train_loss_last",
            "valid_loss",
            "valid_final",
            "valid_S1",
            "valid_S2",
            "valid_S4",
            "valid_S6",
            "valid_S8",
            "valid_S10",
            "valid_S1_final",
            "valid_S2_final",
            "valid_S4_final",
            "valid_S6_final",
            "valid_S8_final",
            "valid_S10_final",
            "persistence_S10",
            "skill_S10",
            "lr",
            "lr_schedule_type",
            "epoch_time_sec",
            "train_time_sec",
            "valid_time_sec",
            "train_samples_per_second",
            "valid_samples_per_second",
            "cuda_peak_allocated_gb",
            "cuda_peak_reserved_gb",
            "amp_dtype",
        ]
        import csv

        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                writer.writeheader()
            row = {
                "epoch": int(epoch),
                "resolution_mode": str(_get(self.params, "resolution_mode", "5p625")),
                "train_rollout_steps": int(train_logs["rollout_steps"]),
                "target_rollout_steps_loaded": int(train_logs.get("target_rollout_steps_loaded", train_logs["rollout_steps"])),
                "batch_size": self.batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "effective_batch_size": self.effective_batch_size,
                "optimizer_steps": int(train_logs.get("optimizer_steps", 0)),
                "train_loss_avg": float(train_logs["loss"]),
                "train_loss_last": float(train_logs.get("last_loss", float("nan"))),
                "valid_loss": float(valid_logs["valid_loss"]),
                "valid_final": float(valid_logs.get("valid_final_loss", float("nan"))),
                "persistence_S10": valid_logs.get("persistence_S10"),
                "skill_S10": valid_logs.get("skill_S10"),
                "lr": float(lr),
                "lr_schedule_type": self.lr_schedule_type,
                "epoch_time_sec": float(epoch_time_sec),
                "train_time_sec": float(train_logs.get("train_time_sec", 0.0)),
                "valid_time_sec": float(valid_logs.get("valid_time_sec", 0.0)),
                "train_samples_per_second": float(train_logs.get("train_samples_per_second", 0.0)),
                "valid_samples_per_second": float(valid_logs.get("valid_samples_per_second", 0.0)),
                "cuda_peak_allocated_gb": self._last_epoch_metadata.get("cuda_peak_allocated_gb"),
                "cuda_peak_reserved_gb": self._last_epoch_metadata.get("cuda_peak_reserved_gb"),
                "amp_dtype": self.resolved_amp_dtype_name,
            }
            for horizon in (1, 2, 4, 6, 8, 10):
                row[f"valid_S{horizon}"] = valid_logs.get(f"valid_S{horizon}")
                row[f"valid_S{horizon}_final"] = valid_logs.get(f"valid_S{horizon}_final")
            writer.writerow(
                row
            )

    @torch.no_grad()
    def validate_rollout_horizon(self, rollout_steps: int) -> tuple[float, float]:
        self.model.eval()
        rollout_steps = min(int(rollout_steps), self.max_rollout_steps)
        total = 0.0
        final_total = 0.0
        steps = 0
        for batch_idx, data in enumerate(self.valid_data_loader):
            if self.max_valid_batches is not None and batch_idx >= self.max_valid_batches:
                break
            inp, target = self._to_device_batch(data)
            with self._autocast_context():
                loss, _ = self._rollout_loss(inp, target, rollout_steps)
            total += float(loss.item())
            target_seq = self._target_sequence(target)
            final_step = min(rollout_steps, target_seq.shape[1]) - 1
            previous, current = self.model.adapter.extract_two_steps(inp)
            final_pred = None
            with self._autocast_context():
                for _ in range(final_step + 1):
                    final_pred = self.model.forward_steps(previous, current)
                    next_step = current.clone()
                    next_step[:, : self.model.output_channels] = final_pred
                    previous, current = current, next_step
                final_loss = self._step_loss(final_pred, target_seq[:, final_step])
            final_total += float(final_loss.item())
            steps += 1
        denom = max(steps, 1)
        return total / denom, final_total / denom

    @torch.no_grad()
    def validate_multi_horizon_single_pass(
        self,
        eval_steps: list[int],
        valid_rollout_steps: int,
    ) -> dict[str, float]:
        self.model.eval()
        eval_steps = sorted({min(int(step), self.max_rollout_steps) for step in eval_steps if int(step) > 0})
        valid_rollout_steps = min(int(valid_rollout_steps), self.max_rollout_steps)
        if valid_rollout_steps not in eval_steps:
            eval_steps.append(valid_rollout_steps)
            eval_steps = sorted(set(eval_steps))
        max_eval_steps = max(eval_steps) if eval_steps else valid_rollout_steps

        step_sums = np.zeros((max_eval_steps,), dtype=np.float64)
        persistence_sums = np.zeros((max_eval_steps,), dtype=np.float64)
        sample_count = 0
        batches = 0
        start = time.time()

        for batch_idx, data in enumerate(self.valid_data_loader):
            if self.max_valid_batches is not None and batch_idx >= self.max_valid_batches:
                break
            inp, target = self._to_device_batch(data)
            target_seq = self._target_sequence(target)
            if target_seq.shape[1] < max_eval_steps:
                raise AssertionError(
                    f"Validation target length {target_seq.shape[1]} is shorter than max eval S={max_eval_steps}"
                )
            bsz = int(inp.shape[0])
            previous, current = self.model.adapter.extract_two_steps(inp)
            persistence_pred = current[:, : self.model.output_channels]
            with self._autocast_context():
                for step in range(max_eval_steps):
                    pred = self.model.forward_steps(previous, current)
                    gt = target_seq[:, step]
                    step_loss = self._step_loss(pred, gt)
                    persistence_loss = self._step_loss(persistence_pred, gt)
                    step_sums[step] += float(step_loss.detach().item()) * bsz
                    persistence_sums[step] += float(persistence_loss.detach().item()) * bsz
                    next_step = current.clone()
                    next_step[:, : self.model.output_channels] = pred
                    previous, current = current, next_step
            sample_count += bsz
            batches += 1

        denom = float(max(sample_count, 1))
        step_losses = step_sums / denom
        persistence_losses = persistence_sums / denom
        logs: dict[str, float] = {
            "valid_rollout_steps": float(valid_rollout_steps),
            "valid_batches": float(batches),
            "valid_samples_per_second": float(sample_count) / max(time.time() - start, 1.0e-12),
        }
        for horizon in eval_steps:
            prefix = step_losses[:horizon]
            persistence_prefix = persistence_losses[:horizon]
            logs[f"valid_S{horizon}"] = float(np.mean(prefix))
            logs[f"valid_S{horizon}_final"] = float(step_losses[horizon - 1])
            logs[f"persistence_S{horizon}"] = float(np.mean(persistence_prefix))
            logs[f"persistence_S{horizon}_final"] = float(persistence_losses[horizon - 1])
            denom_persistence = logs[f"persistence_S{horizon}"]
            logs[f"skill_S{horizon}"] = (
                float(1.0 - logs[f"valid_S{horizon}"] / denom_persistence)
                if np.isfinite(denom_persistence) and abs(denom_persistence) > 1.0e-12
                else float("nan")
            )

        logs["valid_loss"] = float(logs[f"valid_S{valid_rollout_steps}"])
        logs["valid_final_loss"] = float(logs[f"valid_S{valid_rollout_steps}_final"])
        logs["valid_final"] = logs["valid_final_loss"]
        return logs

    @torch.no_grad()
    def validate_one_epoch(self, rollout_steps: int | None = None) -> tuple[float, dict[str, float]]:
        self.model.eval()
        if rollout_steps is None:
            rollout_steps = min(
                int(_get(self.params, "valid_rollout_steps", self.max_rollout_steps)),
                self.max_rollout_steps,
            )
        else:
            rollout_steps = min(int(rollout_steps), self.max_rollout_steps)

        start = time.time()
        eval_rollout_steps = list(self.eval_rollout_steps)
        if bool(self.single_pass_multi_horizon_validation):
            if not eval_rollout_steps:
                eval_rollout_steps = [rollout_steps]
            logs = self.validate_multi_horizon_single_pass(eval_rollout_steps, valid_rollout_steps=rollout_steps)
            return time.time() - start, logs

        valid_loss, valid_final_loss = self.validate_rollout_horizon(rollout_steps)
        logs: dict[str, float] = {
            "valid_loss": valid_loss,
            "valid_final_loss": valid_final_loss,
            "valid_final": valid_final_loss,
            "valid_rollout_steps": float(rollout_steps),
        }

        for horizon in eval_rollout_steps:
            horizon = min(int(horizon), self.max_rollout_steps)
            key = f"valid_S{horizon}"
            if key in logs:
                continue
            if horizon == rollout_steps:
                logs[key] = valid_loss
                logs[f"{key}_final"] = valid_final_loss
            else:
                horizon_loss, horizon_final_loss = self.validate_rollout_horizon(horizon)
                logs[key] = horizon_loss
                logs[f"{key}_final"] = horizon_final_loss

        return time.time() - start, logs

    def save_checkpoint(self, checkpoint_path: str, metadata: dict[str, Any] | None = None) -> None:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        metadata = dict(self._last_epoch_metadata if metadata is None else metadata)
        metadata.update(self.graph_topology_metadata)
        torch.save(
            {
                "iters": self.iters,
                "epoch": self.epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
                "params": dict(getattr(self.params, "params", {})),
                "metadata": metadata,
                "best_score_global": self.best_score_global,
                "best_score_by_stage": dict(self.best_score_by_stage),
            },
            checkpoint_path,
        )

    def restore_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        checkpoint_metadata = dict(checkpoint.get("metadata", {}))
        self._validate_checkpoint_resolution(checkpoint_metadata)
        if checkpoint_metadata.get("resolution_mode") is None and str(_get(self.params, "resolution_mode", "5p625")) == "5p625":
            checkpoint_metadata["resolution_mode"] = "5p625"
        checkpoint_graph = graph_topology_metadata(checkpoint_metadata)
        if checkpoint_graph != self.graph_topology_metadata:
            message = (
                "Graph topology changed from "
                f"{checkpoint_graph.get('graph_connectivity_strategy') or 'unknown'} to "
                f"{self.graph_topology_metadata.get('graph_connectivity_strategy')}. "
                "Old optimizer/checkpoint state will not be resumed. "
                "Train the corrected graph model from scratch."
            )
            if bool(_get(self.params, "allow_graph_mismatch_init", False)):
                self._load_model_state(checkpoint["model_state"], strict_delta_stats=self.use_delta_normalization)
                self.iters = 0
                self.start_epoch = 0
                self.epoch = 0
                logging.warning("%s Loaded model weights only because allow_graph_mismatch_init=True.", message)
                return
            raise RuntimeError(message)
        if self.use_delta_normalization and not bool(checkpoint_metadata.get("use_delta_normalization", False)):
            raise RuntimeError(
                "use_delta_normalization=true but checkpoint metadata does not contain delta-normalization settings. "
                "Disable use_delta_normalization to resume a legacy checkpoint."
            )
        self._load_model_state(checkpoint["model_state"], strict_delta_stats=self.use_delta_normalization)
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler_state = checkpoint.get("scheduler_state_dict", checkpoint_metadata.get("scheduler_state_dict", None))
        if self.scheduler is not None and scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)
        self.iters = int(checkpoint.get("iters", 0))
        self.start_epoch = int(checkpoint.get("epoch", 0))
        self.epoch = self.start_epoch
        if self.lr_schedule_type == "warmup_cosine":
            self._set_optimizer_lr(self._warmup_cosine_lr_for_epoch(self.epoch))
        elif self.lr_schedule_type == "none":
            self._set_optimizer_lr(self.base_lr)
        self.best_score_global = float(checkpoint.get("best_score_global", self.best_score_global))
        restored_by_stage = checkpoint.get("best_score_by_stage", None)
        if isinstance(restored_by_stage, dict):
            for key, value in restored_by_stage.items():
                self.best_score_by_stage[int(key)] = float(value)
        logging.info("Restored checkpoint %s", checkpoint_path)

    def _validate_checkpoint_resolution(self, metadata: dict[str, Any]) -> None:
        active_mode = str(_get(self.params, "resolution_mode", "5p625"))
        checkpoint_mode = metadata.get("resolution_mode", None)
        if checkpoint_mode is None:
            if active_mode == "5p625":
                logging.warning("Checkpoint has no resolution metadata; treating it as legacy 5p625.")
                return
            raise RuntimeError(
                f"Cannot resume a legacy checkpoint without resolution metadata in {active_mode} mode. "
                "Use a matching checkpoint or load weights explicitly as initialization."
            )
        if str(checkpoint_mode) != active_mode:
            raise RuntimeError(
                f"Cannot resume a {checkpoint_mode} checkpoint in {active_mode} mode. "
                "Use the matching mode or load weights explicitly as initialization."
            )
