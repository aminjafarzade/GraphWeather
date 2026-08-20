from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
import re
import time
from contextlib import nullcontext
from dataclasses import replace
from typing import Any

import numpy as np
import torch
import torch.amp as amp
import torch.distributed as dist
import yaml
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

DEFAULT_WANDB_ENTITY = "amin1jafarzade-kaist"

# How much of the hot path torch.compile wraps. See Trainer._maybe_compile_model.
COMPILE_SCOPES = ("none", "processor", "full")

from .architecture import (
    architecture_metadata,
    resolve_lead_conditioning,
    validate_checkpoint_architecture,
    validate_checkpoint_graph_mode,
)
from .data import DataConfig, build_data_loader
from .delta_stats import compute_delta_stats, load_delta_stats
from .diagnostics import DiagnosticsManager
from .device import _assert_cuda_device_usable, _resolve_device
from .features import RolloutFeatureBuilder, VariableResolver, canonical_name, feature_metadata_matches
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
from .layers import ATTENTION_IMPL_DEFAULT, resolve_attention_impl
from .lead_conditioning import format_lead_sequence, lead_conditioning_debug_values
from .losses import (
    LatitudeWeightedMSE,
    band_spectral_power_loss,
    graph_gradient_loss,
    low_frequency_spectral_loss,
)
from .lr_schedulers import RolloutStageWarmupCosineScheduler
from .models import GraphWeatherModel
from .resolution import get_resolution_spec, resolution_metadata
from .resolution import grid_kind as _grid_kind
from .target_handling import TargetHandling, combine_loss_channel_masks, target_handling_metadata_matches


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _get(params: Any, name: str, default: Any) -> Any:
    return getattr(params, name, default)


def _set(params: Any, name: str, value: Any) -> None:
    try:
        params[name] = value
    except Exception:
        setattr(params, name, value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _finite_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


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
        if (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
                and torch.cuda.is_available()):
            # In a distributed run the device is NOT negotiable: one rank per
            # GPU, bound at init_process_group. _resolve_device's kernel-probe
            # fallback silently returned cuda:0 on non-zero ranks here, piling
            # every rank's model onto GPU 0 and desyncing all collectives.
            forced = torch.device("cuda", self.local_rank)
            if self.device != forced:
                logging.warning(
                    "Distributed run: overriding resolved device %s with %s (one rank per GPU).",
                    self.device, forced,
                )
                print(f"[ddp rank {self.world_rank}] device override: {self.device} -> {forced} "
                      f"(device_count={torch.cuda.device_count()})", flush=True)
            self.device = forced
            # Do NOT kernel-probe here: the probe is what mis-resolved the
            # device in rank processes in the first place. On a uniform
            # multi-GPU node the bound device is correct by construction, and
            # a genuinely unusable GPU fails loudly on the first forward.
            torch.cuda.set_device(self.device)
        elif self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            _assert_cuda_device_usable(self.device)
            torch.backends.cudnn.benchmark = True
        logging.info("Training device: %s", self.device)
        raw_max_grad_norm = _get(params, "max_gradient_norm", 1.0)
        self.max_gradient_norm = None if raw_max_grad_norm is None else float(raw_max_grad_norm)
        self.gradient_accumulation_steps = max(1, int(_get(params, "gradient_accumulation_steps", 1)))
        self.batch_size = int(_get(params, "batch_size", 1))
        # Multi-GPU data parallelism (manual gradient all-reduce; see
        # _all_reduce_gradients). config batch_size is the GLOBAL batch: it is
        # split across ranks so a 3-GPU run keeps the exact single-GPU recipe.
        self.world_size = int(dist.get_world_size()) if (dist.is_available() and dist.is_initialized()) else 1
        if self.world_size > 1:
            if self.batch_size % self.world_size != 0:
                raise ValueError(
                    f"batch_size={self.batch_size} must be divisible by world_size={self.world_size} "
                    "(config batch_size is the global batch, split evenly across ranks)."
                )
            self.batch_size //= self.world_size
            logging.info(
                "Distributed data parallel: world_size=%d | global_batch=%d -> per_rank_batch=%d",
                self.world_size, self.batch_size * self.world_size, self.batch_size,
            )
        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        self.log_every_batches = max(1, int(_get(params, "log_every_batches", 1)))
        _set(params, "log_every_batches", self.log_every_batches)
        self.pin_memory = bool(_get(params, "pin_memory", torch.cuda.is_available()))
        self.num_data_workers = int(_get(params, "num_data_workers", _get(params, "num_workers", 0)))
        self.persistent_workers = bool(_get(params, "persistent_workers", False)) and self.num_data_workers > 0
        prefetch_factor = _get(params, "prefetch_factor", None)
        self.prefetch_factor = None if prefetch_factor is None else int(prefetch_factor)
        self.log_cuda_memory = bool(_get(params, "log_cuda_memory", False))
        self.log_timing_breakdown = bool(_get(params, "log_timing_breakdown", False))
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

        self.rollout_mode = str(_get(params, "rollout_mode", "curriculum")).strip().lower()
        if self.rollout_mode not in {"curriculum", "fixed", "fixed_full", "random", "scheduled"}:
            raise ValueError(
                f"Unsupported training.rollout_mode={self.rollout_mode!r}. "
                "Supported values are 'curriculum', 'fixed', 'fixed_full', 'random', and 'scheduled'."
            )
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
        self.fixed_train_rollout_steps = int(_get(params, "fixed_train_rollout_steps", self.max_rollout_steps))
        if self.fixed_train_rollout_steps <= 0:
            raise ValueError("training.fixed_train_rollout_steps must be positive.")
        if self.fixed_train_rollout_steps > self.max_rollout_steps:
            raise ValueError(
                f"training.fixed_train_rollout_steps={self.fixed_train_rollout_steps} exceeds "
                f"max_rollout_steps={self.max_rollout_steps}."
            )
        self._configure_random_rollout(params)
        self._configure_scheduled_rollout(params)
        self.rollout_loss_weights_config = _get(params, "rollout_loss_weights", "uniform")
        self.activation_checkpointing = bool(_get(params, "activation_checkpointing", False))
        self.checkpoint_rollout_steps = bool(_get(params, "checkpoint_rollout_steps", False))
        # Numerically equivalent, so this is purely a memory/throughput switch.
        # Defaults off so every pre-existing config keeps its measured baseline;
        # the _perfctl configs turn it on.
        self.edge_projection_cache = bool(_get(params, "edge_projection_cache", False))
        self.attention_impl = resolve_attention_impl(
            _get(params, "attention_impl", ATTENTION_IMPL_DEFAULT)
        )
        if self.rollout_mode == "fixed_full":
            stage_based_schedulers = {"rollout_stage_warmup_cosine", "rollout_stage", "manual_by_rollout"}
            if self.lr_schedule_type in stage_based_schedulers:
                raise ValueError(
                    f"{self.lr_schedule_type} is incompatible with training.rollout_mode=fixed_full. "
                    "Use warmup_cosine or plateau."
                )
            if bool(_get(params, "load_only_current_rollout", False)):
                logging.warning(
                    "training.rollout_mode=fixed_full requires full target sequences; overriding "
                    "load_only_current_rollout=true to false."
                )
                self.load_only_current_rollout = False
                _set(params, "load_only_current_rollout", False)
            self.rollout_schedule = [self.fixed_train_rollout_steps]
            self.rollout_stage_epochs = [int(_get(params, "max_epochs", 1))]
            _set(params, "rollout_schedule", list(self.rollout_schedule))
            _set(params, "rollout_stage_epochs", list(self.rollout_stage_epochs))
        if self.rollout_mode == "random":
            stage_based_schedulers = {"rollout_stage_warmup_cosine", "rollout_stage", "manual_by_rollout"}
            if self.lr_schedule_type in stage_based_schedulers:
                raise ValueError(
                    f"{self.lr_schedule_type} is incompatible with training.rollout_mode=random. "
                    "Use warmup_cosine or plateau."
                )
            if bool(_get(params, "load_only_current_rollout", False)):
                logging.warning(
                    "training.rollout_mode=random requires targets up to max random horizon; overriding "
                    "load_only_current_rollout=true to false."
                )
            self.load_only_current_rollout = False
            _set(params, "load_only_current_rollout", False)
            self.rollout_schedule = [self.random_rollout_max_horizon]
            self.rollout_stage_epochs = [int(_get(params, "max_epochs", 1))]
            _set(params, "rollout_schedule", list(self.rollout_schedule))
            _set(params, "rollout_stage_epochs", list(self.rollout_stage_epochs))
        if self.rollout_mode == "scheduled":
            stage_based_schedulers = {"rollout_stage_warmup_cosine", "rollout_stage", "manual_by_rollout"}
            if self.lr_schedule_type in stage_based_schedulers:
                raise ValueError(
                    f"{self.lr_schedule_type} is incompatible with training.rollout_mode=scheduled. "
                    "Use warmup_cosine or plateau."
                )
            if bool(_get(params, "load_only_current_rollout", False)):
                logging.warning(
                    "training.rollout_mode=scheduled requires targets up to max scheduled horizon; overriding "
                    "load_only_current_rollout=true to false."
                )
            self.load_only_current_rollout = False
            _set(params, "load_only_current_rollout", False)
            self.rollout_schedule = [self.scheduled_rollout_max_horizon]
            self.rollout_stage_epochs = [int(_get(params, "max_epochs", 1))]
            _set(params, "rollout_schedule", list(self.rollout_schedule))
            _set(params, "rollout_stage_epochs", list(self.rollout_stage_epochs))
        self._validate_rollout_loss_weights()
        self.checkpoint_metric = str(_get(params, "checkpoint_metric", "valid_loss"))
        self.checkpoint_mode = str(_get(params, "checkpoint_mode", "min")).lower()
        self.stage_checkpoint_metric_mode = str(_get(params, "stage_checkpoint_metric_mode", "stage_horizon")).lower()
        self.best_score_global = float("inf") if self.checkpoint_mode == "min" else -float("inf")
        self.best_score_by_stage = {
            int(stage): (float("inf") if self.checkpoint_mode == "min" else -float("inf"))
            for stage in sorted(set(self.rollout_schedule))
        }
        ema_config = _get(params, "ema", {}) or {}
        if not isinstance(ema_config, dict):
            ema_config = {}
        self.ema_enabled = bool(ema_config.get("enabled", False))
        self.ema_decay = float(ema_config.get("decay", 0.999))
        self._ema_state: dict[str, torch.Tensor] | None = None
        self._ema_backup: dict[str, torch.Tensor] | None = None
        if self.ema_enabled:
            logging.info(
                "EMA enabled: decay=%.6g. Validation, best-checkpoint selection and saved model_state use EMA weights; "
                "live training weights are kept in checkpoints under ema_live_model_state for resume.",
                self.ema_decay,
            )
        self._last_epoch_metadata: dict[str, Any] = {}
        self._current_train_target_rollout_steps: int | None = None
        self._last_stage_rollout_steps: int | None = None
        self._last_scheduled_phase_name: str | None = None
        self._epoch_lr_values: list[float] = []
        self._lr_schedule_plot_warning_emitted = False

        if self.rollout_mode == "fixed_full":
            initial_train_rollout = self.fixed_train_rollout_steps
        elif self.rollout_mode == "random":
            initial_train_rollout = self.random_rollout_max_horizon
        elif self.rollout_mode == "scheduled":
            initial_train_rollout = self.scheduled_rollout_max_horizon
        else:
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
            # Derived, never hand-set: hpx32 -> "healpix" makes the loader read
            # [T, C, npix] files. Every lat-lon mode resolves to "latlon".
            grid_kind=_grid_kind(_get(params, "resolution_mode", "5p625")),
            expected_grid_shape=_get(params, "expected_grid_shape", _get(params, "grid_shape", None)),
            return_metadata=bool((_get(params, "extra_features", {}) or {}).get("enabled", False))
            if isinstance(_get(params, "extra_features", {}) or {}, dict)
            else False,
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

        sample_inp, sample_tar, _ = self._unpack_batch(next(iter(self.train_data_loader)))
        out_chans = sample_tar.shape[2] if sample_tar.dim() == 5 else sample_tar.shape[1]
        self.lead_conditioning_config = resolve_lead_conditioning(params).asdict()
        self.lead_conditioning_debug_logged = False
        self.validation_lead_conditioning_debug_logged = False
        self.data_input_channels = int(sample_inp.shape[1])
        self.lead_conditioning_added_input_channels = int(self.lead_conditioning_config.get("added_input_channels", 0))
        self.model_input_channels = int(self.data_input_channels + self.lead_conditioning_added_input_channels)
        expected_config_input = _get(params, "input_channels", None)
        if expected_config_input is not None and int(expected_config_input) != self.model_input_channels:
            raise ValueError(
                f"Configured input_channels={expected_config_input} but data channels={self.data_input_channels} "
                f"and lead conditioning adds {self.lead_conditioning_added_input_channels}; "
                f"expected {self.model_input_channels}."
            )
        params.N_in_channels = int(self.model_input_channels)
        params.input_channels = int(self.model_input_channels)
        if isinstance(_get(params, "model", None), dict):
            params.model["input_channels"] = int(self.model_input_channels)
        params.N_out_channels = int(out_chans)
        if len(list(params.out_channels)) != int(out_chans):
            raise AssertionError(f"out_channels length {len(list(params.out_channels))} does not match target channels {out_chans}")
        if str(_get(params, "normalization", "zscore")).lower() == "zscore":
            state_mean, state_std = self.train_dataset.output_normalization_vectors()
            if state_mean.shape != (int(out_chans),) or state_std.shape != (int(out_chans),):
                raise AssertionError(
                    f"State normalization stats must have shape [{int(out_chans)}], "
                    f"got {state_mean.shape}/{state_std.shape}"
                )
        else:
            state_mean = None
            state_std = None

        self.feature_builder = RolloutFeatureBuilder.from_params(
            params,
            graph=self.graph,
            channel_names=getattr(self.train_dataset, "channel_names", None),
            out_channels=list(params.out_channels),
            output_means=state_mean,
            output_stds=state_std,
            logger=logging,
        )
        params.aux_feature_dim = int(self.feature_builder.aux_feature_dim)
        params.data_input_channels = int(self.data_input_channels)
        params.base_input_channels = int(self.model_input_channels)
        params.total_input_channels = int(self.model_input_channels + self.feature_builder.aux_feature_dim)
        self.target_handler = TargetHandling.from_params(
            params,
            channel_names=getattr(self.train_dataset, "channel_names", None),
            out_channels=list(params.out_channels),
            logger=logging,
        )
        self.loss_channel_mask = combine_loss_channel_masks(
            self.feature_builder.loss_channel_mask(int(out_chans), device=self.device),
            self.target_handler.loss_channel_mask(int(out_chans), device=self.device),
            output_channels=int(out_chans),
            device=self.device,
        )

        self.use_delta_normalization = bool(_get(params, "use_delta_normalization", False))
        self.delta_norm_center = bool(_get(params, "delta_norm_center", False))
        self.delta_norm_eps = float(_get(params, "delta_norm_eps", 1.0e-6))
        self.delta_stats_path = self._resolve_optional_path(_get(params, "delta_stats_path", None))
        delta_mean, delta_std = self._prepare_delta_normalization_stats(int(out_chans))
        # Kept for loss_channel_weighting.inverse_tendency_variance (GraphCast s_j).
        # Already in normalized-state units, so sigma_state == 1 by construction.
        self.loss_delta_std_norm = delta_std

        self.model = GraphWeatherModel(
            graph=self.graph,
            grid_shape=(int(params.crop_size_x), int(params.crop_size_y)),
            input_channels=int(self.model_input_channels),
            output_channels=int(out_chans),
            n_history=int(params.n_history),
            hidden_dim=int(_get(params, "hidden_dim", 96)),
            edge_dim=int(_get(params, "edge_dim", 6)),
            heads=int(_get(params, "num_heads", 4)),
            k_neighbors=int(_get(params, "k_neighbors", 8)),
            level_k_neighbors=_get(params, "level_k_neighbors", None),
            level_dims=_get(params, "level_dims", None),
            level_heads=_get(params, "level_heads", None),
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
            use_delta_normalization=self.use_delta_normalization,
            delta_mean=delta_mean,
            delta_std=delta_std,
            delta_norm_center=self.delta_norm_center,
            delta_norm_eps=self.delta_norm_eps,
            aux_feature_dim=int(self.feature_builder.aux_feature_dim),
            mesh_encoder=dict(_get(params, "mesh_encoder", {}) or {}),
            edge_encoding=dict(_get(params, "edge_encoding", {}) or {}),
            attention_impl=self.attention_impl,
            boundary_mlp=bool(_get(params, "boundary_mlp", False)),
            head_init_std=float(_get(params, "head_init_std", 0.0)),
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
        if self.world_size > 1 and self.gscaler is not None:
            raise ValueError(
                "Multi-GPU training requires bf16 autocast: the fp16 GradScaler's inf-skip "
                "decisions are per-rank and can diverge, silently desynchronizing the replicas. "
                "Use amp_dtype auto_bf16_fp16 on bf16-capable GPUs (the default) or disable AMP."
            )
        self.resolved_amp_dtype_name = _amp_dtype_name(self.amp_dtype)

        if str(getattr(self.graph, "graph_mode", "grid")) == "mesh":
            l0_lat = self.graph.grid_lat_lon[:, 0].reshape(
                int(self.graph.grid_height),
                int(self.graph.grid_width),
            )[:, 0]
        elif _grid_kind(_get(params, "resolution_mode", "5p625")) == "healpix":
            # HEALPix pixels are equal-area, so the area weighting is already in the
            # pixelization. Feeding real latitudes here would apply cos(lat) on top
            # and double-weight the tropics. cos(0) = 1 => uniform weights, and
            # LatitudeWeightedMSE renormalizes by the mean anyway.
            l0_lat = torch.zeros(int(self.graph.L0.num_nodes), dtype=torch.float32)
        else:
            l0_lat = self.graph.L0.lat_lon[:, 0].reshape(self.graph.L0.height, self.graph.L0.width)[:, 0]
        self.loss_channel_weight_cfg = dict(_get(params, "loss_channel_weighting", {}) or {})
        loss_channel_weights = self._build_loss_channel_weights(
            channel_names=getattr(self.train_dataset, "channel_names", None),
            num_channels=int(out_chans),
        )
        self.loss_obj = LatitudeWeightedMSE(l0_lat, channel_weights=loss_channel_weights).to(self.device)
        self.graph_gradient_weight = float(_get(params, "graph_gradient_weight", 0.0))
        self.spectral_latitudes_rad = l0_lat.to(self.device, dtype=torch.float32)
        self._configure_spectral_loss(
            params,
            channel_names=getattr(self.train_dataset, "channel_names", None),
            out_channels=list(params.out_channels),
        )
        self.diagnostics_manager = DiagnosticsManager(
            params,
            self.model,
            self.loss_obj,
            self.device,
            logger=logging,
            run_name=str(_get(params, "name", _get(params, "experiment_name", "diagnostics_run"))),
            rank=self.world_rank,
        )
        self._valid_step_previous_losses: dict[tuple[int, int], float] = {}
        self._valid_step_best_losses: dict[tuple[int, int], float] = {}
        self._train_step_previous_losses: dict[int, float] = {}
        self._valid_step_baseline_losses = self._load_valid_stepwise_baseline()

        scheduler_name = str(_get(params, "scheduler", "CosineAnnealingLR"))
        if self.lr_schedule_type == "rollout_stage_warmup_cosine":
            self.scheduler = self._build_rollout_stage_warmup_cosine_scheduler()
        elif self.lr_schedule_type in {"warmup_cosine", "rollout_stage", "manual_by_rollout", "none"}:
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

        self.num_parameters = self.count_parameters()
        self.architecture_metadata = architecture_metadata(
            params,
            graph_metadata=self.graph.metadata,
            num_parameters=self.num_parameters,
        )
        self.wandb_run = self._init_wandb_run()
        resume_enabled = bool(_get(params, "resume", True))
        resume_path = _get(params, "checkpoint_path", None)
        init_from_checkpoint = _get(params, "init_from_checkpoint", None)
        if resume_enabled and resume_path and os.path.isfile(resume_path):
            # Continue an interrupted run: restore weights + optimizer + scheduler + epoch counter.
            self.restore_checkpoint(resume_path)
        elif init_from_checkpoint:
            # Warm-start a NEW run from another checkpoint's weights only (fresh
            # optimizer/scheduler/epoch counter); the training regime (e.g. a rollout
            # curriculum) may differ from the checkpoint. Takes effect only when there
            # is no in-progress run checkpoint to resume, so an interrupted warm-start
            # run still continues from its own checkpoint on the next launch.
            self.initialize_from_checkpoint(str(init_from_checkpoint))

        # Compile AFTER any checkpoint load: torch.compile rewrites the submodule's
        # state_dict keys to processor._orig_mod.*, which would not match a stored
        # checkpoint. Saving goes back through _canonical_model_state() so written
        # checkpoints stay in the uncompiled key format either way.
        if self.world_size > 1:
            # Replicas must start bit-identical: broadcast rank 0's weights and
            # buffers (covers fresh init, resume, and warm-start alike). Done
            # BEFORE compile; state_dict tensors are the live storages.
            # The process group is bound to cuda:{local_rank}; every collective
            # must use tensors on exactly that device. self.device is resolved
            # independently (and can legitimately differ, e.g. auto-fallback),
            # so stage through the BOUND device, not self.device. print() not
            # logging: non-zero ranks have logging silenced.
            bound = torch.device("cuda", self.local_rank)
            if self.device != bound:
                print(f"[ddp rank {self.world_rank}] NOTE self.device={self.device} "
                      f"!= bound device {bound}", flush=True)
            with torch.no_grad():
                for key, tensor in self.model.state_dict().items():
                    if not (torch.is_tensor(tensor) and tensor.is_cuda):
                        continue
                    if tensor.device == bound and tensor.is_contiguous():
                        dist.broadcast(tensor, src=0)
                        continue
                    print(f"[ddp rank {self.world_rank}] staging {key}: on {tensor.device}, "
                          f"bound {bound}, contiguous={tensor.is_contiguous()}", flush=True)
                    staged = tensor.detach().to(bound).contiguous()
                    dist.broadcast(staged, src=0)
                    tensor.copy_(staged)
            # CRITICAL: force every enqueued collective to COMPLETE before
            # returning. dist.broadcast only enqueues on a stream; the first
            # DataLoader iteration then fork()s num_workers processes per rank,
            # and forking while NCCL collectives are in flight wedges NCCL's
            # progress threads -- observed as a 10-minute watchdog SIGABRT on
            # "BROADCAST SeqNum=11" with training frozen from the exact second
            # the workers spawned.
            torch.cuda.synchronize()
            dist.barrier()
            logging.info("Broadcast model state from rank 0 to %d ranks (synchronized).", self.world_size)
        self._maybe_compile_model()

        logging.info("Number of trainable model parameters: %d", self.num_parameters)
        self._log_architecture_config()
        self._log_startup_config()
        self._log_spectral_loss_startup()
        self._log_lead_conditioning_startup()
        self._log_initial_fusion_gates()
        self._log_initial_pooling_gates()
        self._save_startup_artifacts()

    def _resolve_compile_scope(self) -> str:
        """``none`` | ``processor`` | ``full``.

        ``compile_scope`` wins when set; otherwise the older boolean
        ``compile_processor`` selects between ``processor`` and ``none``, so
        existing configs keep their behaviour.
        """
        params = getattr(self, "params", {})
        scope = _get(params, "compile_scope", None)
        if scope is None:
            return "processor" if bool(_get(params, "compile_processor", False)) else "none"
        scope = str(scope).strip().lower()
        if scope not in COMPILE_SCOPES:
            raise ValueError(f"compile_scope must be one of {COMPILE_SCOPES}, got {scope!r}")
        return scope

    def _maybe_compile_model(self) -> None:
        """torch.compile the hot path at the configured scope.

        ``processor`` compiles the ``GraphUNetProcessor`` submodule: 9 of the 11
        attention blocks, invoked through ``__call__`` so compile intercepts it.
        It leaves the encoder and decoder blocks eager -- and those two run at L0,
        the widest level, so they are not cheap to skip.

        ``full`` instead compiles ``model.forward_steps``. Compiling the *module*
        would be a silent no-op: the training hot path calls ``forward_steps``, not
        ``forward``, and torch.compile on an nn.Module only intercepts
        ``__call__``/``forward``. Compiling the bound method covers embed, encoder,
        processor, decoder, head, the adapter reshapes and the delta denormalization
        in one graph, and -- unlike the module wrapper -- it does not rename any
        parameters, so ``_canonical_*`` have nothing to strip.

        EMA is supported at both scopes: the shadow/backup dicts are keyed through
        _canonical_named_parameters(), so their keys never carry the compile
        `_orig_mod.` prefix and stay interchangeable with an eager run's.
        """
        scope = self._resolve_compile_scope()
        if scope == "none":
            return
        mode = str(_get(self.params, "compile_processor_mode", "default"))
        if scope == "processor":
            processor = getattr(self.model, "processor", None)
            if processor is None:
                logging.warning("compile_scope=processor but the model has no .processor; skipping compile.")
                return
            try:
                self.model.processor = torch.compile(processor, mode=mode)
            except Exception:
                # A compile failure must never cost a training run; fall back to eager.
                logging.exception("torch.compile(model.processor) failed; continuing uncompiled.")
                return
            self._processor_compiled = True
            logging.info("Compiled model.processor with torch.compile(mode=%s).", mode)
            return
        forward_steps = getattr(self.model, "forward_steps", None)
        if forward_steps is None:
            logging.warning("compile_scope=full but the model has no .forward_steps; skipping compile.")
            return
        try:
            # Instance attribute shadows the bound method, so every existing
            # `self.model.forward_steps(...)` call site picks up the compiled one.
            self.model.forward_steps = torch.compile(forward_steps, mode=mode)
        except Exception:
            logging.exception("torch.compile(model.forward_steps) failed; continuing uncompiled.")
            return
        self._model_compiled = True
        logging.info("Compiled model.forward_steps with torch.compile(mode=%s).", mode)

    # Back-compat alias: the old name is referenced in comments and downstream notes.
    _maybe_compile_processor = _maybe_compile_model

    def _canonical_model_state(self) -> dict[str, Any]:
        """Model state_dict with any torch.compile ``_orig_mod.`` prefixes removed.

        Keeps checkpoints loadable by uncompiled runs and by scripts/evaluate.py
        regardless of whether this run compiled the processor.
        """
        state = self.model.state_dict()
        if not getattr(self, "_processor_compiled", False):
            return state
        return {key.replace("._orig_mod.", "."): value for key, value in state.items()}

    def _canonical_named_parameters(self):
        """``(name, param)`` with the torch.compile ``_orig_mod.`` prefix removed.

        The EMA shadow/backup dicts are keyed by these names so a shadow written by
        a compiled run and one written by an eager run are interchangeable. That
        matters most at resume: ``_restore_checkpoint`` runs *before*
        ``_maybe_compile_processor``, so it sees uncompiled names and would silently
        drop every processor parameter from the shadow if the saved keys carried the
        compile prefix.
        """
        compiled = bool(getattr(self, "_processor_compiled", False))
        for name, param in self.model.named_parameters():
            yield (name.replace("._orig_mod.", ".") if compiled else name), param

    def _optional_positive_int(self, name: str) -> int | None:
        value = _get(self.params, name, None)
        if value is None:
            return None
        value = int(value)
        return value if value > 0 else None

    def _configure_spectral_loss(
        self,
        params: Any,
        *,
        channel_names: list[str] | None,
        out_channels: list[int],
    ) -> None:
        raw = _get(params, "spectral_loss", {}) or {}
        if not isinstance(raw, dict):
            raise ValueError("spectral_loss must be a mapping when provided.")
        enabled = bool(raw.get("enabled", False))
        self.spectral_loss_enabled = enabled
        self.spectral_loss_weight = float(raw.get("weight", 0.0))
        self.spectral_loss_variables: list[dict[str, Any]] = []
        self.spectral_loss_channel_indices = torch.empty(0, device=self.device, dtype=torch.long)
        self.spectral_loss_lat_cutoff = int(raw.get("lat_cutoff", 8))
        self.spectral_loss_lon_cutoff = int(raw.get("lon_cutoff", 16))
        self.spectral_loss_include_dc = bool(raw.get("include_dc", True))
        self.spectral_loss_apply_latitude_weight = bool(raw.get("apply_latitude_weight", True))
        self.spectral_loss_space = str(raw.get("space", "normalized")).strip().lower()
        self.spectral_loss_allow_tisr = bool(raw.get("allow_tisr", False))
        # mode: 'error_low_k' is the original low-frequency ERROR-power penalty
        # (drift control; blind to blur). 'power_match' compares the band POWER
        # of prediction and target separately (phase-blind), which is the form
        # that penalizes blur instead of rewarding it.
        self.spectral_loss_mode = str(raw.get("mode", "error_low_k")).strip().lower()
        raw_band_weights = raw.get("band_weights", {"low_k": 0.0, "mid_k": 1.0, "high_k": 1.0})
        if isinstance(raw_band_weights, dict):
            unknown = set(raw_band_weights) - {"low_k", "mid_k", "high_k"}
            if unknown:
                raise ValueError(f"spectral_loss.band_weights has unknown keys: {sorted(unknown)}")
            band_weights = [
                float(raw_band_weights.get("low_k", 0.0)),
                float(raw_band_weights.get("mid_k", 0.0)),
                float(raw_band_weights.get("high_k", 0.0)),
            ]
        else:
            band_weights = [float(value) for value in list(raw_band_weights)]
        self.spectral_loss_band_weights = band_weights

        if not enabled:
            self.spectral_loss_metadata = {"enabled": False}
            return
        if self.spectral_loss_weight < 0.0:
            raise ValueError("spectral_loss.weight must be non-negative.")
        if self.spectral_loss_space != "normalized":
            raise ValueError("Only spectral_loss.space='normalized' is supported for this first diagnostic.")
        if self.spectral_loss_mode not in {"error_low_k", "power_match"}:
            raise ValueError(
                f"Unsupported spectral_loss.mode={self.spectral_loss_mode!r}; "
                "expected 'error_low_k' or 'power_match'."
            )
        if self.spectral_loss_mode == "power_match":
            if not band_weights or any(value < 0.0 for value in band_weights) or sum(band_weights) <= 0.0:
                raise ValueError(
                    "spectral_loss.band_weights must be non-negative with a positive sum "
                    "when mode='power_match'."
                )
        variables = raw.get("variables", [])
        if isinstance(variables, str):
            variables = [chunk.strip() for chunk in variables.replace(",", " ").split() if chunk.strip()]
        else:
            variables = [str(item) for item in list(variables)]
        if not variables:
            raise ValueError("spectral_loss.enabled=true requires at least one variable.")

        resolver = VariableResolver(params, channel_names, out_channels, logger=logging)
        resolved_indices: list[int] = []
        resolved_variables: list[dict[str, Any]] = []
        for variable in variables:
            canonical = canonical_name(variable) or str(variable)
            if canonical == "orog":
                raise ValueError("spectral_loss variables must not include orog; fixed orography is copied, not learned.")
            if canonical == "tisr" and not self.spectral_loss_allow_tisr:
                raise ValueError("spectral_loss variables must not include tisr unless spectral_loss.allow_tisr=true.")
            item = resolver.resolve(variable, required=True)
            if item.local_index is None:
                raise ValueError(
                    f"spectral_loss variable {variable!r} resolved to channel {item.channel}, "
                    "but that channel is not in out_channels."
                )
            local_index = int(item.local_index)
            if local_index in resolved_indices:
                continue
            resolved_indices.append(local_index)
            resolved_variables.append(
                {
                    "requested": str(variable),
                    "canonical": str(item.canonical),
                    "channel": int(item.channel) if item.channel is not None else None,
                    "local_index": local_index,
                    "source": str(item.source),
                }
            )

        self.spectral_loss_variables = resolved_variables
        self.spectral_loss_channel_indices = torch.as_tensor(resolved_indices, device=self.device, dtype=torch.long)
        self.spectral_loss_metadata = {
            "enabled": True,
            "mode": self.spectral_loss_mode,
            "weight": self.spectral_loss_weight,
            "variables": resolved_variables,
            "lat_cutoff": self.spectral_loss_lat_cutoff,
            "lon_cutoff": self.spectral_loss_lon_cutoff,
            "band_weights": list(self.spectral_loss_band_weights),
            "include_dc": self.spectral_loss_include_dc,
            "apply_latitude_weight": self.spectral_loss_apply_latitude_weight,
            "space": self.spectral_loss_space,
        }

    def _spectral_loss_checkpoint_metadata(self) -> dict[str, Any]:
        return {"spectral_loss": dict(getattr(self, "spectral_loss_metadata", {"enabled": False}))}

    def _log_spectral_loss_startup(self) -> None:
        meta = getattr(self, "spectral_loss_metadata", {"enabled": False})
        logging.info("Spectral loss:")
        logging.info("  enabled: %s", str(bool(meta.get("enabled", False))).lower())
        if not bool(meta.get("enabled", False)):
            return
        logging.info("  mode: %s", meta.get("mode", "error_low_k"))
        logging.info("  weight: %.6g", float(meta.get("weight", 0.0)))
        logging.info("  variables:")
        for item in meta.get("variables", []):
            logging.info("    %-5s -> channel %s", item.get("canonical", item.get("requested")), item.get("channel"))
        if meta.get("mode", "error_low_k") == "power_match":
            logging.info("  band_weights (low/mid/high): %s", meta.get("band_weights"))
        logging.info("  lat_cutoff: %d", int(meta.get("lat_cutoff", 0)))
        logging.info("  lon_cutoff: %d", int(meta.get("lon_cutoff", 0)))
        logging.info("  include_dc: %s", str(bool(meta.get("include_dc", True))).lower())
        logging.info("  apply_latitude_weight: %s", str(bool(meta.get("apply_latitude_weight", True))).lower())
        logging.info("  space: %s", meta.get("space", "normalized"))

    def _configure_random_rollout(self, params: Any) -> None:
        raw = dict(_get(params, "random_rollout", {}) or {})
        self.random_rollout_min_horizon = int(raw.get("min_horizon", 1))
        self.random_rollout_max_horizon = int(raw.get("max_horizon", self.max_rollout_steps))
        self.random_rollout_distribution = str(raw.get("distribution", "uniform_integer")).strip().lower()
        self.random_rollout_loss_type = str(raw.get("loss_type", "all_steps_mean")).strip().lower()
        self.random_rollout_final_step_weight = float(raw.get("final_step_weight", 0.0))
        self.random_rollout_detach_between_steps = bool(raw.get("detach_between_steps", False))
        if self.rollout_mode not in {"random", "scheduled"}:
            return
        if self.rollout_mode == "random" and not (1 <= self.random_rollout_min_horizon <= self.random_rollout_max_horizon <= self.max_rollout_steps):
            raise ValueError(
                "training.random_rollout horizons must satisfy "
                f"1 <= min_horizon <= max_horizon <= max_rollout_steps; got "
                f"min_horizon={self.random_rollout_min_horizon}, "
                f"max_horizon={self.random_rollout_max_horizon}, "
                f"max_rollout_steps={self.max_rollout_steps}."
            )
        if self.random_rollout_distribution != "uniform_integer":
            raise ValueError(
                f"Unsupported random rollout distribution: {self.random_rollout_distribution}"
            )
        if self.random_rollout_loss_type != "all_steps_mean":
            raise ValueError(
                f"Unsupported random rollout loss_type: {self.random_rollout_loss_type}"
            )
        if self.random_rollout_final_step_weight != 0.0:
            raise ValueError("training.random_rollout.final_step_weight must be 0.0 for the simple random rollout mode.")
        if self.random_rollout_detach_between_steps:
            raise ValueError("training.random_rollout.detach_between_steps=true is not supported for this experiment.")
        _set(params, "random_rollout", self._random_rollout_config())

    def _random_rollout_config(self) -> dict[str, Any]:
        return {
            "min_horizon": int(getattr(self, "random_rollout_min_horizon", 1)),
            "max_horizon": int(getattr(self, "random_rollout_max_horizon", self.max_rollout_steps)),
            "distribution": str(getattr(self, "random_rollout_distribution", "uniform_integer")),
            "loss_type": str(getattr(self, "random_rollout_loss_type", "all_steps_mean")),
            "final_step_weight": float(getattr(self, "random_rollout_final_step_weight", 0.0)),
            "detach_between_steps": bool(getattr(self, "random_rollout_detach_between_steps", False)),
        }

    def _sample_random_rollout_steps(self) -> int:
        lo = int(self.random_rollout_min_horizon)
        hi = int(self.random_rollout_max_horizon)
        if self.random_rollout_distribution != "uniform_integer":
            raise ValueError(
                f"Unsupported random rollout distribution: {self.random_rollout_distribution}"
            )
        return self._sample_uniform_integer(lo, hi)

    def _sample_uniform_integer(self, lo: int, hi: int) -> int:
        sampled = random.randint(lo, hi) if self.world_rank == 0 else lo
        dist = getattr(torch, "distributed", None)
        if dist is not None and dist.is_available() and dist.is_initialized():
            # Collectives must run on the PG's bound device (cuda:{local_rank}),
            # which is not necessarily self.device.
            device = torch.device("cuda", self.local_rank) if self.device.type == "cuda" else torch.device("cpu")
            tensor = torch.tensor([sampled], device=device, dtype=torch.long)
            dist.broadcast(tensor, src=0)
            sampled = int(tensor.item())
        return int(sampled)

    def _configure_scheduled_rollout(self, params: Any) -> None:
        raw = dict(_get(params, "scheduled_rollout", {}) or {})
        phases_raw = list(raw.get("phases", []) or [])
        self.scheduled_rollout_phases: list[dict[str, Any]] = []
        self.scheduled_rollout_max_horizon = int(getattr(self, "random_rollout_max_horizon", self.max_rollout_steps))
        if self.rollout_mode != "scheduled":
            return
        if not phases_raw:
            raise ValueError("training.scheduled_rollout.phases must contain at least one phase when rollout_mode=scheduled.")
        phases: list[dict[str, Any]] = []
        for idx, phase_raw in enumerate(phases_raw):
            phase = dict(phase_raw or {})
            name = str(phase.get("name", f"phase_{idx}")).strip() or f"phase_{idx}"
            mode = str(phase.get("mode", "")).strip().lower()
            start_epoch = int(phase.get("start_epoch", 1))
            end_raw = phase.get("end_epoch", None)
            end_epoch = None if end_raw is None else int(end_raw)
            if start_epoch < 1:
                raise ValueError(f"scheduled_rollout phase {name!r} has start_epoch={start_epoch}; expected >= 1.")
            if end_epoch is not None and end_epoch < start_epoch:
                raise ValueError(
                    f"scheduled_rollout phase {name!r} has end_epoch={end_epoch} before start_epoch={start_epoch}."
                )
            normalized: dict[str, Any] = {
                "id": idx,
                "name": name,
                "mode": mode,
                "start_epoch": start_epoch,
                "end_epoch": end_epoch,
            }
            if mode == "random":
                min_horizon = int(phase.get("min_horizon", 1))
                max_horizon = int(phase.get("max_horizon", self.max_rollout_steps))
                distribution = str(phase.get("distribution", "uniform_integer")).strip().lower()
                if not (1 <= min_horizon <= max_horizon <= self.max_rollout_steps):
                    raise ValueError(
                        f"scheduled_rollout phase {name!r} horizons must satisfy "
                        f"1 <= min_horizon <= max_horizon <= max_rollout_steps; got "
                        f"min_horizon={min_horizon}, max_horizon={max_horizon}, max_rollout_steps={self.max_rollout_steps}."
                    )
                if distribution != "uniform_integer":
                    raise ValueError(f"Unsupported scheduled random rollout distribution: {distribution}")
                normalized.update(
                    {
                        "min_horizon": min_horizon,
                        "max_horizon": max_horizon,
                        "distribution": distribution,
                    }
                )
            elif mode == "fixed":
                horizon = int(phase.get("horizon", self.max_rollout_steps))
                if not (1 <= horizon <= self.max_rollout_steps):
                    raise ValueError(
                        f"scheduled_rollout phase {name!r} fixed horizon must satisfy "
                        f"1 <= horizon <= max_rollout_steps; got horizon={horizon}, max_rollout_steps={self.max_rollout_steps}."
                    )
                normalized["horizon"] = horizon
            else:
                raise ValueError(f"Unsupported scheduled_rollout phase mode={mode!r} for phase {name!r}.")
            phases.append(normalized)

        phases.sort(key=lambda item: int(item["start_epoch"]))
        expected_start = 1
        seen_open_ended = False
        for phase in phases:
            start_epoch = int(phase["start_epoch"])
            end_epoch = phase["end_epoch"]
            if seen_open_ended:
                raise ValueError("scheduled_rollout phases after end_epoch=null are unreachable.")
            if start_epoch != expected_start:
                raise ValueError(
                    "scheduled_rollout phases must be sorted, non-overlapping, and gap-free from epoch 1; "
                    f"expected start_epoch={expected_start}, got {start_epoch} for phase {phase['name']!r}."
                )
            if end_epoch is None:
                seen_open_ended = True
            else:
                expected_start = int(end_epoch) + 1
        if not seen_open_ended and int(_get(params, "max_epochs", 1)) >= expected_start:
            raise ValueError(
                "scheduled_rollout phases do not cover all configured max_epochs; "
                "set the last phase end_epoch to null or extend the schedule."
            )

        self.scheduled_rollout_phases = phases
        horizons = []
        for phase in phases:
            if phase["mode"] == "random":
                horizons.append(int(phase["max_horizon"]))
            else:
                horizons.append(int(phase["horizon"]))
        self.scheduled_rollout_max_horizon = max(horizons)
        _set(params, "scheduled_rollout", self._scheduled_rollout_config())

    def _scheduled_rollout_config(self) -> dict[str, Any]:
        return {"phases": [dict(phase) for phase in getattr(self, "scheduled_rollout_phases", [])]}

    def _get_scheduled_rollout_phase(self, epoch: int) -> dict[str, Any]:
        epoch = int(epoch)
        for phase in self.scheduled_rollout_phases:
            start_epoch = int(phase["start_epoch"])
            end_epoch = phase.get("end_epoch")
            if epoch >= start_epoch and (end_epoch is None or epoch <= int(end_epoch)):
                return dict(phase)
        raise ValueError(f"No scheduled rollout phase is active for epoch {epoch}.")

    def _scheduled_rollout_horizon_for_batch(self, phase: dict[str, Any]) -> int:
        mode = str(phase.get("mode", "")).strip().lower()
        if mode == "random":
            return self._sample_uniform_integer(int(phase["min_horizon"]), int(phase["max_horizon"]))
        if mode == "fixed":
            return int(phase["horizon"])
        raise ValueError(f"Unsupported scheduled rollout phase mode: {mode}")

    def _validate_rollout_loss_weights(self) -> None:
        raw = self.rollout_loss_weights_config
        if isinstance(raw, str):
            if raw.strip().lower() != "uniform":
                raise ValueError(
                    "training.rollout_loss_weights must be 'uniform' or an explicit non-negative list."
                )
            self.rollout_loss_weights_name = "uniform"
            self.rollout_loss_weights_values = None
            return
        values = [float(x) for x in list(raw)]
        if self.rollout_mode == "fixed_full" and len(values) != self.fixed_train_rollout_steps:
            raise ValueError(
                "training.rollout_loss_weights length must equal "
                f"fixed_train_rollout_steps={self.fixed_train_rollout_steps}; got {len(values)}."
            )
        if not values:
            raise ValueError("training.rollout_loss_weights list must not be empty.")
        if any(value < 0.0 for value in values):
            raise ValueError("training.rollout_loss_weights values must be non-negative.")
        if sum(values) <= 0.0:
            raise ValueError("training.rollout_loss_weights sum must be positive.")
        self.rollout_loss_weights_name = "explicit"
        self.rollout_loss_weights_values = values

    def _rollout_loss_weights_for_steps(
        self,
        rollout_steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        rollout_steps = int(rollout_steps)
        if self.rollout_loss_weights_values is None:
            return torch.ones(rollout_steps, device=device, dtype=dtype)
        values = list(self.rollout_loss_weights_values)
        if len(values) < rollout_steps:
            raise ValueError(
                f"rollout_loss_weights has length {len(values)} but rollout_steps={rollout_steps}."
            )
        weights = torch.tensor(values[:rollout_steps], device=device, dtype=dtype)
        if torch.any(weights < 0):
            raise ValueError("rollout_loss_weights values must be non-negative.")
        if float(weights.sum().item()) <= 0.0:
            raise ValueError("rollout_loss_weights sum must be positive.")
        return weights

    def _rollout_loss_weights_metadata(self) -> Any:
        return self.rollout_loss_weights_name if self.rollout_loss_weights_values is None else list(self.rollout_loss_weights_values)

    def _training_rollout_metadata(self) -> dict[str, Any]:
        metadata = {
            "training_rollout_mode": self.rollout_mode,
            "rollout_mode": self.rollout_mode,
            "fixed_train_rollout_steps": int(self.fixed_train_rollout_steps),
            "training_rollout_steps": int(self._rollout_steps_for_epoch()),
            "backprop_through_rollout": True,
            "detach_between_rollout_steps": False,
            "rollout_loss_weights": self._rollout_loss_weights_metadata(),
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "attention_impl": str(getattr(self, "attention_impl", ATTENTION_IMPL_DEFAULT)),
            "edge_projection_cache": bool(getattr(self, "edge_projection_cache", False)),
            "compile_scope": self._resolve_compile_scope(),
        }
        if self.rollout_mode == "random":
            metadata["random_rollout"] = self._random_rollout_config()
            metadata["random_rollout_min_horizon"] = int(self.random_rollout_min_horizon)
            metadata["random_rollout_max_horizon"] = int(self.random_rollout_max_horizon)
            metadata["random_rollout_distribution"] = str(self.random_rollout_distribution)
            metadata["random_rollout_loss_type"] = str(self.random_rollout_loss_type)
            metadata["random_rollout_final_step_weight"] = float(self.random_rollout_final_step_weight)
            metadata["random_rollout_detach_between_steps"] = bool(self.random_rollout_detach_between_steps)
        if self.rollout_mode == "scheduled":
            metadata["scheduled_rollout"] = self._scheduled_rollout_config()
            metadata["scheduled_rollout_max_horizon"] = int(self.scheduled_rollout_max_horizon)
            try:
                metadata["scheduled_rollout_active_phase"] = self._get_scheduled_rollout_phase(int(self.epoch))
            except ValueError:
                pass
            metadata["random_rollout_loss_type"] = str(self.random_rollout_loss_type)
            metadata["random_rollout_final_step_weight"] = float(self.random_rollout_final_step_weight)
            metadata["random_rollout_detach_between_steps"] = bool(self.random_rollout_detach_between_steps)
        return metadata

    def _planned_optimizer_steps_per_epoch(self) -> int:
        planned_batches = len(self.train_data_loader)
        if self.max_train_batches is not None:
            planned_batches = min(planned_batches, int(self.max_train_batches))
        return max(1, int(np.ceil(float(max(planned_batches, 1)) / float(self.gradient_accumulation_steps))))

    def _build_rollout_stage_warmup_cosine_scheduler(self) -> RolloutStageWarmupCosineScheduler:
        lr_schedule = list(_get(self.params, "rollout_stage_lr_schedule", []) or [])
        if not lr_schedule:
            raise ValueError(
                "lr_schedule_type=rollout_stage_warmup_cosine requires rollout_stage_lr_schedule."
            )
        stage_epoch_sum = sum(int(x) for x in self.rollout_stage_epochs)
        max_epochs = int(_get(self.params, "max_epochs", stage_epoch_sum))
        if stage_epoch_sum != max_epochs:
            logging.warning(
                "rollout_stage_warmup_cosine stage epochs sum to %d but max_epochs=%d. "
                "Training will follow the repository's existing epoch limit behavior.",
                stage_epoch_sum,
                max_epochs,
            )
        scheduler = RolloutStageWarmupCosineScheduler(
            self.optimizer,
            rollout_schedule=list(self.rollout_schedule),
            rollout_stage_epochs=[int(x) for x in self.rollout_stage_epochs],
            rollout_stage_lr_schedule=lr_schedule,
            steps_per_epoch=self._planned_optimizer_steps_per_epoch(),
        )
        return scheduler

    def _warn_lr_config_conflicts(self) -> None:
        if self.lr_schedule_type == "warmup_cosine" and _get(self.params, "lr_by_rollout", None):
            logging.warning("lr_schedule_type=warmup_cosine: ignoring lr_by_rollout.")

    def _resolve_optional_path(self, value: Any) -> str | None:
        if value is None or str(value).strip() == "":
            return None
        path = os.path.expanduser(str(value))
        return path if os.path.isabs(path) else os.path.abspath(path)

    def _wandb_config(self) -> dict[str, Any]:
        raw = _get(self.params, "wandb", None)
        if isinstance(raw, dict):
            return dict(raw)
        diagnostics = _get(self.params, "diagnostics", {}) or {}
        if isinstance(diagnostics, dict):
            diag_wandb = diagnostics.get("wandb", {})
            if isinstance(diag_wandb, dict):
                return dict(diag_wandb)
        return {}

    def _init_wandb_run(self) -> Any:
        if self.world_rank != 0:
            return None
        cfg = self._wandb_config()
        if not bool(cfg.get("enabled", False)):
            return None
        if wandb is None:
            logging.warning("W&B logging requested, but wandb is not installed. Continuing with local logs only.")
            return None

        run_name = str(cfg.get("run_name") or _get(self.params, "name", _get(self.params, "experiment_name", "graphweather_run")))
        tags = [str(item) for item in list(cfg.get("tags", []) or [])]
        project = cfg.get("project") or os.environ.get("WANDB_PROJECT")
        entity = cfg.get("entity") or os.environ.get("WANDB_ENTITY") or DEFAULT_WANDB_ENTITY
        if getattr(wandb, "run", None) is not None:
            run = wandb.run
        else:
            if not project:
                logging.warning("W&B logging requested, but no wandb.project or WANDB_PROJECT is set. Continuing with local logs only.")
                return None
            try:
                params_payload = _jsonable(getattr(self.params, "params", {}))
                run = wandb.init(
                    project=project,
                    entity=entity,
                    name=run_name,
                    tags=tags,
                    config=params_payload,
                )
            except Exception as exc:  # pragma: no cover - depends on external W&B state
                logging.warning("W&B initialization failed; continuing with local logs only: %s", exc)
                return None

        try:
            run.config.update(self._wandb_run_config_payload(), allow_val_change=True)
        except Exception as exc:  # pragma: no cover - depends on external W&B state
            logging.warning("W&B config update failed: %s", exc)
        return run

    def _wandb_run_config_payload(self) -> dict[str, Any]:
        l0_refine = dict(_get(self.params, "l0_refine", {}) or {})
        return {
            "model.hidden_dim": int(_get(self.params, "hidden_dim", 0)),
            "model.num_params": int(getattr(self, "num_parameters", 0)),
            "model/l0_refine/type": str(l0_refine.get("type", "attention")),
            "model/l0_refine/mlp_expansion": int(l0_refine.get("mlp_expansion", 2)),
            "model/l0_refine/dropout": float(l0_refine.get("dropout", 0.0)),
            "model/l0_refine/residual_scale_init": float(l0_refine.get("residual_scale_init", 0.1)),
            "model/l0_refine/learnable_residual_scale": bool(l0_refine.get("learnable_residual_scale", True)),
            "graph.level_k_neighbors": _jsonable(_get(self.params, "level_k_neighbors", [])),
            "training.rollout_mode": str(self.rollout_mode),
            "training.random_rollout.min_horizon": int(getattr(self, "random_rollout_min_horizon", 1)),
            "training.random_rollout.max_horizon": int(getattr(self, "random_rollout_max_horizon", self.max_rollout_steps)),
            "training.random_rollout.distribution": str(getattr(self, "random_rollout_distribution", "uniform_integer")),
            "training.random_rollout.loss_type": str(getattr(self, "random_rollout_loss_type", "all_steps_mean")),
            "training.random_rollout.final_step_weight": float(getattr(self, "random_rollout_final_step_weight", 0.0)),
            "training.random_rollout.detach_between_steps": bool(getattr(self, "random_rollout_detach_between_steps", False)),
            "training.scheduled_rollout.phases": _jsonable(getattr(self, "scheduled_rollout_phases", [])),
            "training.scheduled_rollout.max_horizon": int(getattr(self, "scheduled_rollout_max_horizon", 0) or 0),
            "data.rollout_steps_loaded": int(getattr(self, "_current_train_target_rollout_steps", 0) or 0),
        }

    def _wandb_log(self, payload: dict[str, Any], *, step: int | None = None) -> None:
        run = getattr(self, "wandb_run", None)
        if run is None or not payload:
            return
        try:
            run.log(payload, step=step)
        except Exception as exc:  # pragma: no cover - depends on external W&B state
            logging.warning("W&B logging failed: %s", exc)

    def _wandb_table(self, rows: list[dict[str, Any]], columns: list[str]) -> Any:
        if wandb is None or not rows:
            return None
        return wandb.Table(columns=columns, data=[[row.get(column) for column in columns] for row in rows])

    def _experiment_logs_dir(self) -> str | None:
        experiment_dir = str(_get(self.params, "experiment_dir", ""))
        if not experiment_dir:
            return None
        path = os.path.join(experiment_dir, "logs")
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def _append_csv_row(path: str, fields: list[str], row: dict[str, Any]) -> None:
        exists = os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow({field: row.get(field) for field in fields})

    @staticmethod
    def _improvement_abs(reference: float | None, current: float | None) -> float:
        if reference is None or current is None:
            return float("nan")
        if not np.isfinite(float(reference)) or not np.isfinite(float(current)):
            return float("nan")
        return float(float(reference) - float(current))

    @staticmethod
    def _improvement_pct(reference: float | None, current: float | None) -> float:
        if reference is None or current is None:
            return float("nan")
        reference = float(reference)
        current = float(current)
        if not np.isfinite(reference) or not np.isfinite(current) or abs(reference) <= 1.0e-12:
            return float("nan")
        return float((reference - current) / reference)

    def _baseline_compare_config(self) -> dict[str, Any]:
        diagnostics = _get(self.params, "diagnostics", {}) or {}
        raw = diagnostics.get("baseline_compare", {}) if isinstance(diagnostics, dict) else {}
        cfg = {
            "enabled": False,
            "baseline_name": None,
            "baseline_rollout_curve_csv": None,
            "baseline_scalars_json": None,
        }
        cfg.update(dict(raw or {}))
        return cfg

    def _load_valid_stepwise_baseline(self) -> dict[tuple[int, int], float]:
        cfg = self._baseline_compare_config()
        if not bool(cfg.get("enabled", False)):
            return {}
        loaded: dict[tuple[int, int], tuple[int, int, float]] = {}

        csv_path = self._resolve_optional_path(cfg.get("baseline_rollout_curve_csv"))
        if csv_path:
            if not os.path.exists(csv_path):
                logging.warning("Validation step baseline CSV is missing: %s", csv_path)
            else:
                try:
                    with open(csv_path, "r", newline="", encoding="utf-8") as f:
                        for row_index, row in enumerate(csv.DictReader(f)):
                            phase = str(row.get("phase", "valid")).strip().lower()
                            if phase and phase not in {"valid", "validation", "val"}:
                                continue
                            try:
                                horizon = int(float(str(row.get("horizon", "")).replace("S", "")))
                                step = int(float(row.get("step", "")))
                                loss = float(row.get("loss", "nan"))
                            except (TypeError, ValueError):
                                continue
                            if horizon <= 0 or step <= 0 or not np.isfinite(loss):
                                continue
                            try:
                                epoch = int(float(row.get("epoch", row_index)))
                            except (TypeError, ValueError):
                                epoch = row_index
                            key = (horizon, step)
                            current = loaded.get(key)
                            if current is None or (epoch, row_index) >= (current[0], current[1]):
                                loaded[key] = (epoch, row_index, loss)
                    logging.info("Loaded %d validation step baseline losses from %s", len(loaded), csv_path)
                except Exception as exc:
                    logging.warning("Failed to load validation step baseline CSV %s: %s", csv_path, exc)

        json_path = self._resolve_optional_path(cfg.get("baseline_scalars_json"))
        if json_path:
            if not os.path.exists(json_path):
                logging.warning("Validation step baseline scalars JSON is missing: %s", json_path)
            else:
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        scalars = json.load(f)
                    for key, value in dict(scalars).items():
                        if not isinstance(key, str) or "/loss" not in key:
                            continue
                        parts = key.strip().split("/")
                        if len(parts) < 4 or parts[0] != "valid_stepwise":
                            continue
                        try:
                            horizon = int(parts[1].lstrip("S"))
                            step = int(parts[2].replace("step_", ""))
                            loss = float(value)
                        except (TypeError, ValueError):
                            continue
                        if np.isfinite(loss):
                            loaded[(horizon, step)] = (10**12, 0, loss)
                    logging.info("Loaded validation step baseline scalar JSON from %s", json_path)
                except Exception as exc:
                    logging.warning("Failed to load validation step baseline JSON %s: %s", json_path, exc)

        return {key: value[2] for key, value in loaded.items()}

    def _valid_stepwise_rows(self, epoch: int, valid_logs: dict[str, Any]) -> list[dict[str, Any]]:
        cached = valid_logs.get("_valid_stepwise_rows")
        if isinstance(cached, list):
            return [dict(row) for row in cached]
        rows: list[dict[str, Any]] = []
        for raw_row in self._valid_rollout_curve_rows(epoch, valid_logs):
            try:
                horizon = int(raw_row.get("horizon", 0))
                step = int(raw_row.get("step", 0))
                loss = float(raw_row.get("loss", float("nan")))
            except (TypeError, ValueError):
                continue
            if horizon <= 0 or step <= 0 or not np.isfinite(loss):
                continue
            key = (horizon, step)
            previous = self._valid_step_previous_losses.get(key)
            previous_best = self._valid_step_best_losses.get(key)
            baseline = self._valid_step_baseline_losses.get(key)
            best_so_far = loss if previous_best is None or not np.isfinite(previous_best) else min(float(previous_best), loss)
            row = {
                "epoch": int(epoch),
                "horizon": horizon,
                "step": step,
                "loss": loss,
                "previous_loss": previous,
                "improvement_abs_prev": self._improvement_abs(previous, loss),
                "improvement_pct_prev": self._improvement_pct(previous, loss),
                "best_so_far_loss": best_so_far,
                "improvement_abs_best": self._improvement_abs(previous_best, loss),
                "improvement_pct_best": self._improvement_pct(previous_best, loss),
                "baseline_loss": baseline,
                "improvement_abs_baseline": self._improvement_abs(baseline, loss),
                "improvement_pct_baseline": self._improvement_pct(baseline, loss),
            }
            rows.append(row)
            self._valid_step_previous_losses[key] = loss
            self._valid_step_best_losses[key] = best_so_far
        valid_logs["_valid_stepwise_rows"] = [dict(row) for row in rows]  # type: ignore[assignment]
        return rows

    def _train_stepwise_rows(self, epoch: int, train_logs: dict[str, Any]) -> list[dict[str, Any]]:
        cached = train_logs.get("_train_stepwise_rows")
        if isinstance(cached, list):
            return [dict(row) for row in cached]
        rows: list[dict[str, Any]] = []
        for step in range(1, self.max_rollout_steps + 1):
            loss = _finite_or_nan(train_logs.get(f"train_loss_lead{step}"))
            count = _finite_or_nan(train_logs.get(f"train_loss_lead{step}_count", 0.0))
            previous = self._train_step_previous_losses.get(step)
            row = {
                "epoch": int(epoch),
                "step": int(step),
                "loss_mean": loss,
                "count": count,
                "previous_loss_mean": previous,
                "improvement_abs_prev": self._improvement_abs(previous, loss),
                "improvement_pct_prev": self._improvement_pct(previous, loss),
            }
            rows.append(row)
            if np.isfinite(loss):
                self._train_step_previous_losses[step] = float(loss)
        train_logs["_train_stepwise_rows"] = [dict(row) for row in rows]  # type: ignore[assignment]
        return rows

    def _valid_rollout_curve_rows(self, epoch: int, valid_logs: dict[str, Any]) -> list[dict[str, Any]]:
        rows = valid_logs.get("_valid_rollout_curve_rows")
        if isinstance(rows, list):
            return [{**dict(row), "epoch": int(epoch)} for row in rows]
        curve_rows: list[dict[str, Any]] = []
        for horizon in self.eval_rollout_steps or [int(valid_logs.get("valid_rollout_steps", self.max_rollout_steps))]:
            mean_loss = valid_logs.get(f"valid_S{int(horizon)}")
            final_loss = valid_logs.get(f"valid_S{int(horizon)}_final")
            if mean_loss is not None:
                curve_rows.append({"epoch": int(epoch), "horizon": int(horizon), "step": 0, "loss": float(mean_loss)})
            if final_loss is not None:
                curve_rows.append({"epoch": int(epoch), "horizon": int(horizon), "step": int(horizon), "loss": float(final_loss)})
        return curve_rows

    def _append_random_rollout_local_logs(self, epoch: int, train_logs: dict[str, Any]) -> None:
        if self.rollout_mode not in {"random", "scheduled"}:
            return
        logs_dir = self._experiment_logs_dir()
        if logs_dir is None:
            return
        phase_name = str(train_logs.get("train_rollout_phase_name", "random_rollout" if self.rollout_mode == "random" else ""))
        phase_mode = str(train_logs.get("train_rollout_phase_mode", "random" if self.rollout_mode == "random" else ""))
        count_fields = ["epoch", "phase_name", "phase_mode", "horizon_mean", "horizon_min", "horizon_max", "horizon_std"] + [
            f"S{horizon}" for horizon in range(1, self.max_rollout_steps + 1)
        ]
        count_row = {
            "epoch": int(epoch),
            "phase_name": phase_name,
            "phase_mode": phase_mode,
            "horizon_mean": train_logs.get("train_random_sampled_horizon_mean"),
            "horizon_min": train_logs.get("train_random_sampled_horizon_min"),
            "horizon_max": train_logs.get("train_random_sampled_horizon_max"),
            "horizon_std": train_logs.get("train_random_sampled_horizon_std"),
        }
        for horizon in range(1, self.max_rollout_steps + 1):
            count_row[f"S{horizon}"] = train_logs.get(f"train_random_horizon_count_S{horizon}", 0.0)
        count_filename = "scheduled_rollout_horizon_counts.csv" if self.rollout_mode == "scheduled" else "random_rollout_horizon_counts.csv"
        self._append_csv_row(
            os.path.join(logs_dir, count_filename),
            count_fields,
            count_row,
        )

        by_horizon_fields = ["epoch", "phase_name", "phase_mode", "horizon", "count", "loss_mean", "loss_final"]
        for horizon in range(1, self.max_rollout_steps + 1):
            self._append_csv_row(
                os.path.join(logs_dir, "random_rollout_train_by_horizon.csv"),
                by_horizon_fields,
                {
                    "epoch": int(epoch),
                    "phase_name": phase_name,
                    "phase_mode": phase_mode,
                    "horizon": int(horizon),
                    "count": train_logs.get(f"train_random_horizon_count_S{horizon}", 0.0),
                    "loss_mean": train_logs.get(f"train_random_by_horizon_S{horizon}_loss_mean"),
                    "loss_final": train_logs.get(f"train_random_by_horizon_S{horizon}_loss_final"),
                },
            )

        by_lead_fields = ["epoch", "phase_name", "phase_mode", "lead", "count", "loss_mean"]
        for lead in range(1, self.max_rollout_steps + 1):
            self._append_csv_row(
                os.path.join(logs_dir, "random_rollout_train_by_lead.csv"),
                by_lead_fields,
                {
                    "epoch": int(epoch),
                    "phase_name": phase_name,
                    "phase_mode": phase_mode,
                    "lead": int(lead),
                    "count": train_logs.get(f"train_loss_lead{lead}_count", 0.0),
                    "loss_mean": train_logs.get(f"train_loss_lead{lead}"),
                },
            )
        train_stepwise_fields = [
            "epoch",
            "phase_name",
            "phase_mode",
            "step",
            "loss_mean",
            "count",
            "previous_loss_mean",
            "improvement_abs_prev",
            "improvement_pct_prev",
        ]
        for row in self._train_stepwise_rows(epoch, train_logs):
            row = {**row, "phase_name": phase_name, "phase_mode": phase_mode}
            self._append_csv_row(
                os.path.join(logs_dir, "train_stepwise_losses.csv"),
                train_stepwise_fields,
                row,
            )
        train_by_horizon_fields = ["epoch", "phase_name", "phase_mode", "horizon", "count", "loss_mean", "loss_final"]
        for horizon in range(1, self.max_rollout_steps + 1):
            self._append_csv_row(
                os.path.join(logs_dir, "train_by_horizon_losses.csv"),
                train_by_horizon_fields,
                {
                    "epoch": int(epoch),
                    "phase_name": phase_name,
                    "phase_mode": phase_mode,
                    "horizon": int(horizon),
                    "count": train_logs.get(f"train_random_horizon_count_S{horizon}", 0.0),
                    "loss_mean": train_logs.get(f"train_random_by_horizon_S{horizon}_loss_mean"),
                    "loss_final": train_logs.get(f"train_random_by_horizon_S{horizon}_loss_final"),
                },
            )

    def _append_valid_rollout_curve_local_log(self, epoch: int, valid_logs: dict[str, Any]) -> None:
        logs_dir = self._experiment_logs_dir()
        if logs_dir is None:
            return
        fields = ["epoch", "horizon", "step", "loss"]
        path = os.path.join(logs_dir, "valid_rollout_curve.csv")
        for row in self._valid_rollout_curve_rows(epoch, valid_logs):
            self._append_csv_row(path, fields, row)
        summary_fields = ["epoch", "horizon", "loss_mean", "loss_final", "final_to_mean_ratio"]
        for horizon in self.eval_rollout_steps or [int(valid_logs.get("valid_rollout_steps", self.max_rollout_steps))]:
            horizon = int(horizon)
            self._append_csv_row(
                os.path.join(logs_dir, "valid_rollout_summary.csv"),
                summary_fields,
                {
                    "epoch": int(epoch),
                    "horizon": horizon,
                    "loss_mean": valid_logs.get(f"valid_S{horizon}"),
                    "loss_final": valid_logs.get(f"valid_S{horizon}_final"),
                    "final_to_mean_ratio": valid_logs.get(f"valid_S{horizon}_final_to_mean_ratio"),
                },
            )

    def _append_valid_stepwise_local_logs(self, epoch: int, valid_logs: dict[str, Any]) -> None:
        logs_dir = self._experiment_logs_dir()
        if logs_dir is None:
            return
        fields = [
            "epoch",
            "horizon",
            "step",
            "loss",
            "previous_loss",
            "improvement_abs_prev",
            "improvement_pct_prev",
            "best_so_far_loss",
            "improvement_abs_best",
            "improvement_pct_best",
            "baseline_loss",
            "improvement_abs_baseline",
            "improvement_pct_baseline",
        ]
        rows = self._valid_stepwise_rows(epoch, valid_logs)
        for row in rows:
            self._append_csv_row(os.path.join(logs_dir, "valid_stepwise_losses.csv"), fields, row)
            self._append_csv_row(os.path.join(logs_dir, "valid_stepwise_improvements.csv"), fields, row)
            if int(row.get("horizon", 0)) == 10:
                self._append_csv_row(os.path.join(logs_dir, "valid_stepwise_S10.csv"), fields, row)

    def _log_epoch_wandb(
        self,
        *,
        epoch: int,
        train_logs: dict[str, Any],
        valid_logs: dict[str, Any],
        lr: float,
        epoch_time_sec: float,
        train_time_sec: float,
        valid_time_sec: float,
        cuda_peak_allocated_gb: float | None,
        cuda_peak_reserved_gb: float | None,
    ) -> None:
        if getattr(self, "wandb_run", None) is None:
            return
        payload: dict[str, Any] = {
            "epoch": int(epoch),
            "train/loss": _finite_or_nan(train_logs.get("loss")),
            "lr": float(lr),
            "system/epoch_time_sec": float(epoch_time_sec),
            "system/train_time_sec": float(train_time_sec),
            "system/valid_time_sec": float(valid_time_sec),
            "system/lr": float(lr),
            "diagnostics_system/epoch_time_sec": float(epoch_time_sec),
            "diagnostics_system/train_time_sec": float(train_time_sec),
            "diagnostics_system/valid_time_sec": float(valid_time_sec),
            "diagnostics_system/lr": float(lr),
        }
        train_s1, valid_s1, overfit_gap = self._s1_overfit_values(train_logs, valid_logs)
        payload.update(
            {
                "loss/train_S1_epoch": train_s1,
                "loss/valid_S1_epoch": valid_s1,
                "loss/overfit_gap_S1": overfit_gap,
            }
        )
        # Primary outcomes of the attention-matmul / edge-cache control: the skill
        # metrics above are the control, these two are the deliverable.
        payload["perf/median_step_time_s"] = _finite_or_nan(train_logs.get("median_step_time_sec"))
        payload["perf/attention_impl"] = str(getattr(self, "attention_impl", ATTENTION_IMPL_DEFAULT))
        payload["perf/edge_projection_cache"] = bool(self.edge_projection_cache)
        payload["perf/compile_scope"] = self._resolve_compile_scope()
        if cuda_peak_allocated_gb is not None:
            payload["perf/peak_memory_allocated_gb"] = float(cuda_peak_allocated_gb)
            payload["system/gpu_peak_allocated_mb"] = float(cuda_peak_allocated_gb * 1024.0)
            payload["diagnostics_system/gpu_peak_allocated_mb"] = float(cuda_peak_allocated_gb * 1024.0)
        if cuda_peak_reserved_gb is not None:
            payload["system/gpu_peak_reserved_mb"] = float(cuda_peak_reserved_gb * 1024.0)
            payload["diagnostics_system/gpu_peak_reserved_mb"] = float(cuda_peak_reserved_gb * 1024.0)
        if getattr(self, "_last_grad_norm_pre_clip", None) is not None:
            payload["train/grad_norm/global"] = float(self._last_grad_norm_pre_clip)
            payload["train/grad_norm/pre_clip"] = float(self._last_grad_norm_pre_clip)
            payload["train/grad_norm/post_clip"] = float(self._last_grad_norm_post_clip)
        l0_refine_scales = self._l0_refine_residual_scale_values()
        for index, value in l0_refine_scales.items():
            payload[f"model/l0_refine/{index}/residual_scale"] = float(value)
        if len(l0_refine_scales) == 1 and "0" in l0_refine_scales:
            payload["model/l0_refine/residual_scale"] = float(l0_refine_scales["0"])

        for horizon in self.eval_rollout_steps or [int(valid_logs.get("valid_rollout_steps", self.max_rollout_steps))]:
            horizon = int(horizon)
            mean_key = f"valid_S{horizon}"
            final_key = f"valid_S{horizon}_final"
            if mean_key in valid_logs:
                payload[f"valid/rollout/S{horizon}/loss_mean"] = _finite_or_nan(valid_logs.get(mean_key))
                payload[f"valid_rollout_summary/S{horizon}/loss_mean"] = _finite_or_nan(valid_logs.get(mean_key))
            if final_key in valid_logs:
                payload[f"valid/rollout/S{horizon}/loss_final"] = _finite_or_nan(valid_logs.get(final_key))
                payload[f"valid_rollout_summary/S{horizon}/loss_final"] = _finite_or_nan(valid_logs.get(final_key))
            mean_value = _finite_or_nan(valid_logs.get(mean_key))
            final_value = _finite_or_nan(valid_logs.get(final_key))
            if np.isfinite(mean_value) and abs(mean_value) > 1.0e-12 and np.isfinite(final_value):
                payload[f"valid/rollout/S{horizon}/final_to_mean_ratio"] = float(final_value / mean_value)
                payload[f"valid_rollout_summary/S{horizon}/final_to_mean_ratio"] = float(final_value / mean_value)

        for row in self._valid_stepwise_rows(epoch, valid_logs):
            horizon = int(row["horizon"])
            step = int(row["step"])
            prefix = f"valid_stepwise/S{horizon}/step_{step:02d}"
            payload[f"{prefix}/loss"] = _finite_or_nan(row.get("loss"))
            payload[f"{prefix}/previous_loss"] = _finite_or_nan(row.get("previous_loss"))
            payload[f"{prefix}/improvement_abs_prev"] = _finite_or_nan(row.get("improvement_abs_prev"))
            payload[f"{prefix}/improvement_pct_prev"] = _finite_or_nan(row.get("improvement_pct_prev"))
            payload[f"{prefix}/best_so_far_loss"] = _finite_or_nan(row.get("best_so_far_loss"))
            payload[f"{prefix}/improvement_abs_best"] = _finite_or_nan(row.get("improvement_abs_best"))
            payload[f"{prefix}/improvement_pct_best"] = _finite_or_nan(row.get("improvement_pct_best"))
            if row.get("baseline_loss") is not None:
                payload[f"{prefix}/baseline_loss"] = _finite_or_nan(row.get("baseline_loss"))
                payload[f"{prefix}/improvement_abs_baseline"] = _finite_or_nan(row.get("improvement_abs_baseline"))
                payload[f"{prefix}/improvement_pct_baseline"] = _finite_or_nan(row.get("improvement_pct_baseline"))

        for row in self._train_stepwise_rows(epoch, train_logs):
            step = int(row["step"])
            prefix = f"train_stepwise/step_{step:02d}"
            payload[f"{prefix}/loss_mean"] = _finite_or_nan(row.get("loss_mean"))
            payload[f"{prefix}/count"] = _finite_or_nan(row.get("count"))
            payload[f"{prefix}/improvement_abs_prev"] = _finite_or_nan(row.get("improvement_abs_prev"))
            payload[f"{prefix}/improvement_pct_prev"] = _finite_or_nan(row.get("improvement_pct_prev"))

        if self.rollout_mode in {"random", "scheduled"}:
            payload.update(
                {
                    "diagnostics_system/sampled_horizon_mean": _finite_or_nan(train_logs.get("train_random_sampled_horizon_mean")),
                    "diagnostics_system/random_rollout_effective_avg_horizon": _finite_or_nan(train_logs.get("train_random_sampled_horizon_mean")),
                }
            )
            if self.rollout_mode == "random":
                payload.update(
                    {
                        "train/random_rollout/horizon_mean": _finite_or_nan(train_logs.get("train_random_sampled_horizon_mean")),
                        "train/random_rollout/horizon_min": _finite_or_nan(train_logs.get("train_random_sampled_horizon_min")),
                        "train/random_rollout/horizon_max": _finite_or_nan(train_logs.get("train_random_sampled_horizon_max")),
                        "train/random_rollout/horizon_std": _finite_or_nan(train_logs.get("train_random_sampled_horizon_std")),
                        "train/random_rollout/loss_total": _finite_or_nan(train_logs.get("train_random_loss_total")),
                        "train/random_rollout/loss_mean_all_steps": _finite_or_nan(train_logs.get("train_random_loss_mean_all_steps")),
                        "train/random_rollout/loss_final_sampled_horizon": _finite_or_nan(train_logs.get("train_random_loss_final_sampled_horizon")),
                        "train/random_rollout/final_to_mean_ratio": _finite_or_nan(train_logs.get("train_random_final_to_mean_ratio")),
                    }
                )
            if self.rollout_mode == "scheduled":
                payload.update(
                    {
                        "train/rollout_phase/id": _finite_or_nan(train_logs.get("train_rollout_phase_id")),
                        "train/rollout_phase/name": str(train_logs.get("train_rollout_phase_name", "")),
                        "train/rollout_phase/mode": str(train_logs.get("train_rollout_phase_mode", "")),
                        "train/rollout_phase/start_epoch": _finite_or_nan(train_logs.get("train_rollout_phase_start_epoch")),
                        "train/rollout_phase/end_epoch": _finite_or_nan(train_logs.get("train_rollout_phase_end_epoch")),
                        "train/rollout_phase/fixed_horizon": _finite_or_nan(train_logs.get("train_rollout_phase_fixed_horizon")),
                        "train/rollout_phase/random_min_horizon": _finite_or_nan(train_logs.get("train_rollout_phase_random_min_horizon")),
                        "train/rollout_phase/random_max_horizon": _finite_or_nan(train_logs.get("train_rollout_phase_random_max_horizon")),
                        "train/scheduled_rollout/horizon_mean": _finite_or_nan(train_logs.get("train_scheduled_sampled_horizon_mean")),
                        "train/scheduled_rollout/horizon_min": _finite_or_nan(train_logs.get("train_scheduled_sampled_horizon_min")),
                        "train/scheduled_rollout/horizon_max": _finite_or_nan(train_logs.get("train_scheduled_sampled_horizon_max")),
                        "train/scheduled_rollout/horizon_std": _finite_or_nan(train_logs.get("train_scheduled_sampled_horizon_std")),
                        "train/scheduled_rollout/loss_total": _finite_or_nan(train_logs.get("train_scheduled_loss_total")),
                        "train/scheduled_rollout/loss_mean_all_steps": _finite_or_nan(train_logs.get("train_scheduled_loss_mean_all_steps")),
                        "train/scheduled_rollout/loss_final_sampled_horizon": _finite_or_nan(
                            train_logs.get("train_scheduled_loss_final_sampled_horizon")
                        ),
                        "train/scheduled_rollout/final_to_mean_ratio": _finite_or_nan(
                            train_logs.get("train_scheduled_final_to_mean_ratio")
                        ),
                    }
                )
            for horizon in range(1, self.max_rollout_steps + 1):
                payload[f"train_by_horizon/S{horizon}/loss_mean"] = _finite_or_nan(
                    train_logs.get(f"train_random_by_horizon_S{horizon}_loss_mean")
                )
                payload[f"train_by_horizon/S{horizon}/loss_final"] = _finite_or_nan(
                    train_logs.get(f"train_random_by_horizon_S{horizon}_loss_final")
                )
                payload[f"train_by_horizon/S{horizon}/count"] = _finite_or_nan(
                    train_logs.get(f"train_random_horizon_count_S{horizon}", 0.0)
                )
                if self.rollout_mode == "random":
                    payload[f"train/random_rollout/horizon_count_S{horizon}"] = _finite_or_nan(
                        train_logs.get(f"train_random_horizon_count_S{horizon}", 0.0)
                    )
                    payload[f"train/random_rollout/by_horizon/S{horizon}/loss_mean"] = _finite_or_nan(
                        train_logs.get(f"train_random_by_horizon_S{horizon}_loss_mean")
                    )
                    payload[f"train/random_rollout/by_horizon/S{horizon}/loss_final"] = _finite_or_nan(
                        train_logs.get(f"train_random_by_horizon_S{horizon}_loss_final")
                    )
                    payload[f"train/random_rollout/by_lead/lead{horizon}_loss"] = _finite_or_nan(
                        train_logs.get(f"train_loss_lead{horizon}")
                    )
                    payload[f"train/random_rollout/by_lead/lead{horizon}_count"] = _finite_or_nan(
                        train_logs.get(f"train_loss_lead{horizon}_count", 0.0)
                    )
                if self.rollout_mode == "scheduled":
                    payload[f"train/scheduled_rollout/horizon_count_S{horizon}"] = _finite_or_nan(
                        train_logs.get(f"train_scheduled_horizon_count_S{horizon}", 0.0)
                    )
                    payload[f"train/scheduled_rollout/by_horizon/S{horizon}/loss_mean"] = _finite_or_nan(
                        train_logs.get(f"train_scheduled_by_horizon_S{horizon}_loss_mean")
                    )
                    payload[f"train/scheduled_rollout/by_horizon/S{horizon}/loss_final"] = _finite_or_nan(
                        train_logs.get(f"train_scheduled_by_horizon_S{horizon}_loss_final")
                    )
                    payload[f"train/scheduled_rollout/by_lead/lead{horizon}_loss"] = _finite_or_nan(
                        train_logs.get(f"train_loss_lead{horizon}")
                    )
                    payload[f"train/scheduled_rollout/by_lead/lead{horizon}_count"] = _finite_or_nan(
                        train_logs.get(f"train_loss_lead{horizon}_count", 0.0)
                    )
            sampled = np.asarray(train_logs.get("_sampled_horizons", []), dtype=np.float64)
            if wandb is not None and sampled.size:
                if self.rollout_mode == "random":
                    payload["train/random_rollout/horizon_histogram"] = wandb.Histogram(sampled)
                if self.rollout_mode == "scheduled":
                    payload["train/scheduled_rollout/horizon_histogram"] = wandb.Histogram(sampled)

        curve_rows = self._valid_rollout_curve_rows(epoch, valid_logs)
        valid_table = self._wandb_table(curve_rows, ["epoch", "horizon", "step", "loss"])
        if valid_table is not None:
            payload["tables/valid/rollout_curve"] = valid_table
            if wandb is not None:
                try:
                    payload["plots/valid/rollout_step_loss"] = wandb.plot.line(
                        valid_table,
                        "step",
                        "loss",
                        title="Validation rollout step loss",
                    )
                except Exception as exc:  # pragma: no cover - depends on external W&B state
                    logging.warning("W&B validation rollout plot creation failed: %s", exc)
        valid_stepwise_columns = [
            "epoch",
            "horizon",
            "step",
            "loss",
            "previous_loss",
            "improvement_abs_prev",
            "improvement_pct_prev",
            "best_so_far_loss",
            "improvement_abs_best",
            "improvement_pct_best",
            "baseline_loss",
            "improvement_abs_baseline",
            "improvement_pct_baseline",
        ]
        for horizon in sorted({int(row["horizon"]) for row in self._valid_stepwise_rows(epoch, valid_logs)}):
            rows = [row for row in self._valid_stepwise_rows(epoch, valid_logs) if int(row["horizon"]) == horizon]
            table = self._wandb_table(rows, valid_stepwise_columns)
            if table is not None:
                payload[f"tables/valid_stepwise/S{horizon}"] = table
        self._wandb_log(payload, step=int(getattr(self, "iters", epoch)))

    def _unpack_batch(self, data: Any) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        if isinstance(data, dict):
            return data["input"], data["target"], data
        inp, target = data
        return inp, target, {}

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
        # Shared-file build: rank 0 computes and writes; other ranks wait at the
        # barrier, then find the file on disk and simply load it.
        if self.world_size > 1 and self.world_rank != 0:
            dist.barrier()
        try:
            return self._prepare_delta_normalization_stats_impl(output_channels)
        finally:
            if self.world_size > 1 and self.world_rank == 0:
                dist.barrier()

    def _prepare_delta_normalization_stats_impl(self, output_channels: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
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
                return_metadata=False,
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
        if tuple(delta_mean.shape) != (int(output_channels),) or tuple(delta_std.shape) != (int(output_channels),):
            raise AssertionError(
                f"Delta stats must have shape [{int(output_channels)}], "
                f"got {tuple(delta_mean.shape)}/{tuple(delta_std.shape)}"
            )
        logging.info("Delta normalization: enabled")
        logging.info("Delta stats path: %s", self.delta_stats_path)
        logging.info(
            "Delta std min/max/mean: %.6g / %.6g / %.6g",
            float(delta_std.min().item()),
            float(delta_std.max().item()),
            float(delta_std.mean().item()),
        )
        logging.info("Delta center: %s", self.delta_norm_center)
        excluded_channels: dict[str, int] = {}
        feature_builder = getattr(self, "feature_builder", None)
        if feature_builder is not None:
            excluded_channels.update(getattr(feature_builder, "exclude_loss_channels", {}) or {})
        target_handler = getattr(self, "target_handler", None)
        if target_handler is not None:
            excluded_channels.update(getattr(target_handler, "exclude_loss_channels", {}) or {})
        if excluded_channels:
            for name, channel in excluded_channels.items():
                idx = int(channel)
                if float(delta_std[idx].item()) <= max(self.delta_norm_eps * 100.0, 1.0e-5):
                    logging.warning(
                        "Excluded channel %s has tiny delta std %.6g; setting delta mean/std to 0/1 because predictions are overridden or excluded.",
                        name,
                        float(delta_std[idx].item()),
                    )
                delta_mean[idx] = 0.0
                delta_std[idx] = 1.0
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

    def _load_model_state_partial(self, state: dict[str, torch.Tensor], strict_delta_stats: bool) -> None:
        current = self.model.state_dict()
        filtered: dict[str, torch.Tensor] = {}
        skipped: list[str] = []
        for key, value in state.items():
            clean_key = key[7:] if key.startswith("module.") else key
            if clean_key not in current:
                skipped.append(clean_key)
                continue
            if tuple(value.shape) != tuple(current[clean_key].shape):
                skipped.append(clean_key)
                continue
            filtered[clean_key] = value
        if strict_delta_stats and ("delta_mean" not in filtered or "delta_std" not in filtered):
            raise RuntimeError("Delta normalization is enabled but checkpoint lacks compatible delta_mean/delta_std buffers.")
        missing, unexpected = self.model.load_state_dict(filtered, strict=False)
        allowed_missing = {"delta_mean", "delta_std"} if not strict_delta_stats else set()
        missing = [key for key in missing if key not in allowed_missing]
        logging.warning(
            "Partial checkpoint initialization loaded %d matching tensors; skipped %d tensors; left %d tensors initialized.",
            len(filtered),
            len(skipped),
            len(missing),
        )
        if skipped:
            logging.warning("Partial checkpoint skipped tensors: %s", ", ".join(sorted(skipped)))
        if missing:
            logging.warning("Partial checkpoint missing current tensors: %s", ", ".join(sorted(missing)))

    def _autocast_context(self):
        if not self.amp_enabled:
            return nullcontext()
        return amp.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=True)

    def _to_device_batch(self, data: Any) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        non_blocking = bool(self.pin_memory and self.device.type == "cuda")
        inp, target, metadata = self._unpack_batch(data)
        device_metadata: dict[str, Any] = {}
        for key, value in metadata.items():
            if key in {"input", "target"}:
                continue
            if torch.is_tensor(value):
                device_metadata[key] = value.to(self.device, non_blocking=non_blocking)
            else:
                device_metadata[key] = value
        return (
            inp.to(self.device, dtype=torch.float32, non_blocking=non_blocking),
            target.to(self.device, dtype=torch.float32, non_blocking=non_blocking),
            device_metadata,
        )

    def _graph_build_options(self) -> tuple[int, float, str, dict[str, Any]]:
        return (
            int(_get(self.params, "k_neighbors", 8)),
            float(_get(self.params, "resolution", 5.625)),
            str(_get(self.params, "graph_connectivity_strategy", "hybrid_row_aware_knn")),
            dict(_get(self.params, "row_aware_knn", {}) or {}),
        )

    def _ensure_mesh_graph(self, graph_path: str, mesh_cfg: dict) -> dict[str, Any]:
        """Build/validate a mesh bundle (mesh_encoder mode). Uses the training
        data's own lat/lon axes so grid<->mesh edges align with the grid nodes the
        GridNodeAdapter produces (row-major over lat, lon).

        ``mesh_encoder.mesh_type`` selects the mesh geometry: "icosphere" (default,
        unchanged) or "healpix", where the grid<->mesh boundary is a fixed
        conservative regrid instead of barycentric interpolation."""
        from .mesh_builder import (
            bipartite_edge_feature_set,
            bipartite_mapping_type,
            build_and_save,
            expected_mesh_metadata,
            validate_mesh_cache_metadata,
        )

        mesh_type = str(mesh_cfg.get("mesh_type", "icosphere")).strip().lower()
        if mesh_type not in {"icosphere", "healpix"}:
            raise ValueError(
                f"mesh_encoder.mesh_type must be 'icosphere' or 'healpix', got {mesh_type!r}."
            )
        if mesh_type == "healpix":
            return self._ensure_healpix_mesh_graph(graph_path, mesh_cfg)

        num_graph_levels = int(_get(self.params, "num_graph_levels", 4))
        latitudes, longitudes = lat_lon_from_netcdf(self.params.train_data_path)
        lat = latitudes.detach().cpu().numpy() if hasattr(latitudes, "detach") else np.asarray(latitudes)
        lon = longitudes.detach().cpu().numpy() if hasattr(longitudes, "detach") else np.asarray(longitudes)
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
        grid_ll = np.deg2rad(np.stack([lat_grid.reshape(-1), lon_grid.reshape(-1)], axis=1)).astype(np.float32)
        grid_shape = (int(lat.shape[0]), int(lon.shape[0]))
        resolution_mode = _get(self.params, "resolution_mode", None)
        bipartite_features = bipartite_edge_feature_set(
            mesh_cfg.get("boundary_type", "legacy")
        )
        bipartite_mapping = bipartite_mapping_type(
            mesh_cfg.get("boundary_type", "legacy")
        )
        grid_attention_enabled = (
            int(mesh_cfg.get("grid_attention_encoder_blocks", 0)) > 0
            or int(mesh_cfg.get("grid_attention_decoder_blocks", 0)) > 0
        )
        grid_attention_k = (
            int(mesh_cfg.get("grid_attention_k_neighbors", 8))
            if grid_attention_enabled
            else None
        )
        grid_attention_strategy = str(
            _get(self.params, "graph_connectivity_strategy", "hybrid_row_aware_knn")
        )
        grid_attention_row_aware = dict(
            _get(self.params, "row_aware_knn", {}) or {}
        )
        grid_attention_reference = _get(
            self.params,
            "grid_attention_reference_graph_path",
            None,
        )
        expected = expected_mesh_metadata(
            refinement=int(mesh_cfg.get("refinement", 5)),
            num_graph_levels=num_graph_levels,
            grid_shape=grid_shape,
            grid_lat_lon=grid_ll,
            g2m_radius_factor=float(mesh_cfg.get("g2m_radius_factor", 0.6)),
            resolution_mode=None if resolution_mode is None else str(resolution_mode),
            bipartite_mapping_type=bipartite_mapping,
            bipartite_edge_features=bipartite_features,
            coarse_level_connectivity=str(
                mesh_cfg.get("coarse_level_connectivity", "native_icosphere")
            ),
            grid_attention_k_neighbors=grid_attention_k,
            grid_attention_connectivity_strategy=grid_attention_strategy,
            grid_attention_row_aware_knn=grid_attention_row_aware,
            grid_attention_reference_graph_path=grid_attention_reference,
        )
        if os.path.isfile(graph_path):
            raw = load_raw_graph_bundle(graph_path, map_location="cpu")
            mismatches = validate_mesh_cache_metadata(raw, expected)
            if not mismatches:
                return dict(raw.get("metadata", {}) or {})
            logging.warning(
                "Rebuilding mismatched icosphere mesh bundle %s: %s",
                graph_path,
                "; ".join(mismatches),
            )
        if not bool(_get(self.params, "auto_build_graph", True)):
            raise FileNotFoundError(
                f"Matching mesh graph bundle not found and auto_build_graph disabled: {graph_path}"
            )
        meta = build_and_save(
            graph_path,
            refinement=int(mesh_cfg.get("refinement", 5)),
            num_graph_levels=num_graph_levels,
            grid_shape=grid_shape,
            grid_lat_lon=grid_ll,
            g2m_radius_factor=float(mesh_cfg.get("g2m_radius_factor", 0.6)),
            resolution_mode=None if resolution_mode is None else str(resolution_mode),
            bipartite_mapping_type=bipartite_mapping,
            bipartite_edge_features=bipartite_features,
            coarse_level_connectivity=str(
                mesh_cfg.get("coarse_level_connectivity", "native_icosphere")
            ),
            grid_attention_k_neighbors=grid_attention_k,
            grid_attention_connectivity_strategy=grid_attention_strategy,
            grid_attention_row_aware_knn=grid_attention_row_aware,
            grid_attention_reference_graph_path=grid_attention_reference,
        )
        logging.info("Built icosphere mesh bundle at %s: %s", graph_path, meta)
        return dict(meta)

    def _ensure_healpix_mesh_graph(self, graph_path: str, mesh_cfg: dict) -> dict[str, Any]:
        """Build/validate a HEALPix mesh bundle (in-model conservative regrid).

        The data grid is still the training file's own lat-lon axes, so
        resolution_mode, the delta statistics, the latitude-weighted loss and the
        evaluator are all untouched -- the HEALPix round trip lives entirely
        between grid2mesh and mesh2grid inside the model.
        """
        from .mesh_builder import (
            bipartite_edge_feature_set,
            build_and_save_healpix,
            expected_healpix_mesh_metadata,
            validate_mesh_cache_metadata,
        )

        boundary_type = str(mesh_cfg.get("boundary_type", "legacy")).strip().lower()
        if boundary_type != "fixed_spherical":
            raise ValueError(
                "mesh_encoder.mesh_type='healpix' requires "
                "boundary_type='fixed_spherical' (the conservative regrid is a fixed, "
                f"parameter-free remap), got {boundary_type!r}."
            )
        num_graph_levels = int(_get(self.params, "num_graph_levels", 4))
        nside = int(mesh_cfg.get("nside", 32))
        oversample = int(mesh_cfg.get("regrid_oversample", 8))
        latitudes, longitudes = lat_lon_from_netcdf(self.params.train_data_path)
        lat = latitudes.detach().cpu().numpy() if hasattr(latitudes, "detach") else np.asarray(latitudes)
        lon = longitudes.detach().cpu().numpy() if hasattr(longitudes, "detach") else np.asarray(longitudes)
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
        grid_ll = np.deg2rad(np.stack([lat_grid.reshape(-1), lon_grid.reshape(-1)], axis=1)).astype(np.float32)
        grid_shape = (int(lat.shape[0]), int(lon.shape[0]))
        resolution_mode = _get(self.params, "resolution_mode", None)
        bipartite_features = bipartite_edge_feature_set(boundary_type)
        # The resolved config already reconciled this with the mesh degree
        # (architecture.py): 8 keeps true adjacency, more densifies via kNN.
        level_k = _get(self.params, "level_k_neighbors", None)
        # Grid message passing before/after the conservative regrid needs a kNN
        # level over the lat-lon grid in the bundle. Same knobs the icosphere path
        # uses, so a HEALPix and an icosphere run share one grid connectivity rule.
        grid_attention_enabled = (
            int(mesh_cfg.get("grid_attention_encoder_blocks", 0)) > 0
            or int(mesh_cfg.get("grid_attention_decoder_blocks", 0)) > 0
        )
        grid_attention_k = (
            int(mesh_cfg.get("grid_attention_k_neighbors", 8))
            if grid_attention_enabled
            else None
        )
        build_kwargs = dict(
            nside=nside,
            num_graph_levels=num_graph_levels,
            grid_shape=grid_shape,
            grid_lat_lon=grid_ll,
            resolution_mode=None if resolution_mode is None else str(resolution_mode),
            regrid_oversample=oversample,
            bipartite_edge_features=bipartite_features,
            level_k_neighbors=None if level_k is None else [int(x) for x in level_k],
            grid_attention_k_neighbors=grid_attention_k,
            grid_attention_connectivity_strategy=str(
                _get(self.params, "graph_connectivity_strategy", "hybrid_row_aware_knn")
            ),
            grid_attention_row_aware_knn=dict(
                _get(self.params, "row_aware_knn", {}) or {}
            ),
        )
        expected = expected_healpix_mesh_metadata(**build_kwargs)
        if os.path.isfile(graph_path):
            raw = load_raw_graph_bundle(graph_path, map_location="cpu")
            mismatches = validate_mesh_cache_metadata(raw, expected)
            if not mismatches:
                return dict(raw.get("metadata", {}) or {})
            logging.warning(
                "Rebuilding mismatched HEALPix mesh bundle %s: %s",
                graph_path,
                "; ".join(mismatches),
            )
        if not bool(_get(self.params, "auto_build_graph", True)):
            raise FileNotFoundError(
                f"Matching mesh graph bundle not found and auto_build_graph disabled: {graph_path}"
            )
        # Read the raw param, NOT self.delta_stats_path: _ensure_graph() runs early
        # in __init__, well before that attribute is assigned.
        raw_delta_stats_path = _get(self.params, "delta_stats_path", None)
        regrid_cache_dir = (
            os.path.dirname(str(raw_delta_stats_path)) if raw_delta_stats_path else ""
        ) or "data/stats"
        meta = build_and_save_healpix(
            graph_path,
            regrid_cache_dir=regrid_cache_dir,
            **build_kwargs,
        )
        logging.info("Built HEALPix mesh bundle at %s: %s", graph_path, meta)
        return dict(meta)

    def _ensure_graph(self) -> dict[str, Any]:
        # Shared-file build: rank 0 auto-builds the graph bundle; other ranks
        # wait, then load the cache rank 0 wrote.
        if self.world_size > 1 and self.world_rank != 0:
            dist.barrier()
        try:
            return self._ensure_graph_impl()
        finally:
            if self.world_size > 1 and self.world_rank == 0:
                dist.barrier()

    def _ensure_graph_impl(self) -> dict[str, Any]:
        graph_path = self.params.graph_path
        mesh_cfg = dict(_get(self.params, "mesh_encoder", {}) or {})
        if bool(mesh_cfg.get("enabled", False)):
            return self._ensure_mesh_graph(graph_path, mesh_cfg)
        k, resolution, strategy, row_aware_knn = self._graph_build_options()
        num_graph_levels = int(_get(self.params, "num_graph_levels", 3))
        level_k_neighbors = [int(x) for x in _get(self.params, "level_k_neighbors", [k] * num_graph_levels)]
        latitudes, longitudes = lat_lon_from_netcdf(self.params.train_data_path)
        expected = expected_graph_metadata(
            latitudes,
            longitudes,
            k=k,
            resolution=resolution,
            connectivity_strategy=strategy,
            row_aware_knn=row_aware_knn,
            resolution_mode=str(_get(self.params, "resolution_mode", "5p625")),
            num_graph_levels=num_graph_levels,
            level_k_neighbors=level_k_neighbors,
            level_shapes=_get(self.params, "level_shapes", None),
            hierarchy_type=str(_get(self.params, "hierarchy_type", "standard")),
            use_l4_ratio15=bool(_get(self.params, "use_l4_ratio15", False)),
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
            num_graph_levels=num_graph_levels,
            level_k_neighbors=level_k_neighbors,
            level_shapes=_get(self.params, "level_shapes", None),
            hierarchy_type=str(_get(self.params, "hierarchy_type", "standard")),
            use_l4_ratio15=bool(_get(self.params, "use_l4_ratio15", False)),
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

    def _log_architecture_config(self) -> None:
        logging.info("GraphWeather model config:")
        for line in self._architecture_summary_lines():
            logging.info("  %s", line)
        logging.info("Trainable parameters: %d", self.num_parameters)

    def _architecture_summary_lines(self) -> list[str]:
        hidden_dim = int(_get(self.params, "hidden_dim", 96))
        num_heads = int(_get(self.params, "num_heads", _get(self.params, "heads", 4)))
        assert hidden_dim % num_heads == 0, f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
        head_dim = hidden_dim // num_heads
        lines = [
            f"use_l3: {str(bool(_get(self.params, 'use_l3', False))).lower()}",
            f"use_l4_ratio15: {str(bool(_get(self.params, 'use_l4_ratio15', False))).lower()}",
            f"hierarchy_type: {_get(self.params, 'hierarchy_type', 'standard')}",
            f"num_graph_levels: {int(_get(self.params, 'num_graph_levels', 3))}",
            f"hidden_dim: {hidden_dim}",
            f"num_heads: {num_heads}",
            f"head_dim: {head_dim}",
            f"k_neighbors: {int(_get(self.params, 'k_neighbors', 8))}",
            f"level_k_neighbors: {_get(self.params, 'level_k_neighbors', [])}",
            f"skip_fusion: {_get(self.params, 'skip_fusion', {'type': 'default'})}",
            f"pooling: {_get(self.params, 'pooling', {'type': 'default'})}",
            f"l0_refine: {_get(self.params, 'l0_refine', {'type': 'attention'})}",
            f"lead_conditioning: {_get(self.params, 'lead_conditioning', {'enabled': False})}",
            f"l0_blocks: {int(_get(self.params, 'l0_blocks', 2))}",
            f"l1_blocks: {int(_get(self.params, 'l1_blocks', 2))}",
            f"l2_blocks: {int(_get(self.params, 'l2_blocks', 1))}",
            f"l3_blocks: {int(_get(self.params, 'l3_blocks', 1))}",
            f"l4_blocks: {int(_get(self.params, 'l4_blocks', 1))}",
            f"l3_refine_after_l4_blocks: {int(_get(self.params, 'l3_refine_after_l4_blocks', 1))}",
            f"l2_refine_after_l3_blocks: {int(_get(self.params, 'l2_refine_after_l3_blocks', 1))}",
            f"l1_refine_blocks: {int(_get(self.params, 'l1_refine_blocks', 1))}",
            f"l0_refine_blocks: {int(_get(self.params, 'l0_refine_blocks', 1))}",
            f"trainable_parameters: {self.num_parameters}",
        ]
        mesh_encoder = _get(self.params, "mesh_encoder", None)
        if isinstance(mesh_encoder, dict):
            lines.insert(13, f"mesh_encoder: {mesh_encoder}")
        return lines

    def _skip_fusion_config(self) -> dict[str, Any]:
        raw = _get(self.params, "skip_fusion", None)
        if isinstance(raw, dict):
            config = dict(raw)
        else:
            config = {"type": "default"}
        config.setdefault("type", "default")
        config.setdefault("mean_type", "mean")
        config.setdefault("include_max", True)
        config.setdefault("init_scale", 1.0)
        config.setdefault("max_scale", 2.0)
        return config

    def _fusion_gate_values(self) -> dict[str, float]:
        model = getattr(self, "model", None)
        if model is None or not hasattr(model, "fusion_gate_values"):
            return {}
        return {str(key): float(value) for key, value in model.fusion_gate_values().items()}

    def _fusion_gate_fieldnames(self) -> list[str]:
        return [
            "l3_to_l2_skip",
            "l3_to_l2_up",
            "l2_to_l1_skip",
            "l2_to_l1_up",
            "l1_to_l0_skip",
            "l1_to_l0_up",
        ]

    def _log_initial_fusion_gates(self) -> None:
        config = self._skip_fusion_config()
        logging.info("Skip fusion:")
        logging.info("  type: %s", config.get("type"))
        logging.info("  init_scale: %s", config.get("init_scale"))
        logging.info("  max_scale: %s", config.get("max_scale"))
        values = self._fusion_gate_values()
        if not values:
            return
        logging.info("Initial gate scales:")
        for key in self._fusion_gate_fieldnames():
            logging.info("  %s_scale: %.3f", key, float(values.get(key, float("nan"))))

    def _log_and_append_fusion_gate_history(self, epoch: int) -> None:
        values = self._fusion_gate_values()
        if not values:
            return
        logging.info("Fusion gate scales | epoch=%d", int(epoch))
        logging.info(
            "  l3_to_l2 skip=%.6f up=%.6f",
            values.get("l3_to_l2_skip", float("nan")),
            values.get("l3_to_l2_up", float("nan")),
        )
        logging.info(
            "  l2_to_l1 skip=%.6f up=%.6f",
            values.get("l2_to_l1_skip", float("nan")),
            values.get("l2_to_l1_up", float("nan")),
        )
        logging.info(
            "  l1_to_l0 skip=%.6f up=%.6f",
            values.get("l1_to_l0_skip", float("nan")),
            values.get("l1_to_l0_up", float("nan")),
        )
        experiment_dir = str(_get(self.params, "experiment_dir", ""))
        if not experiment_dir or self.world_rank != 0:
            return

        path = os.path.join(experiment_dir, "fusion_gate_history.csv")
        exists = os.path.exists(path)
        fields = ["epoch", *self._fusion_gate_fieldnames()]
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                writer.writeheader()
            row = {"epoch": int(epoch)}
            row.update({key: values.get(key) for key in self._fusion_gate_fieldnames()})
            writer.writerow(row)

    def _fusion_gate_metadata(self) -> dict[str, Any]:
        values = self._fusion_gate_values()
        if not values:
            return {}
        return {
            "fusion_gate_values": {
                key: float(values.get(key, float("nan")))
                for key in self._fusion_gate_fieldnames()
            }
        }

    def _pooling_config(self) -> dict[str, Any]:
        raw = _get(self.params, "pooling", None)
        if isinstance(raw, dict):
            config = dict(raw)
        else:
            config = {"type": "default"}
        config.setdefault("type", "default")
        config.setdefault("init_scale", 1.0)
        config.setdefault("max_scale", 2.0)
        return config

    def _pooling_gate_values(self) -> dict[str, float]:
        model = getattr(self, "model", None)
        if model is None or not hasattr(model, "pooling_gate_values"):
            return {}
        return {str(key): float(value) for key, value in model.pooling_gate_values().items()}

    def _l0_refine_residual_scale_values(self) -> dict[str, float]:
        model = getattr(self, "model", None)
        if model is None or not hasattr(model, "l0_refine_residual_scale_values"):
            return {}
        return {str(key): float(value) for key, value in model.l0_refine_residual_scale_values().items()}

    def _pooling_gate_fieldnames(self) -> list[str]:
        return [
            "l0_to_l1_mean",
            "l0_to_l1_max",
            "l1_to_l2_mean",
            "l1_to_l2_max",
            "l2_to_l3_mean",
            "l2_to_l3_max",
        ]

    def _log_initial_pooling_gates(self) -> None:
        config = self._pooling_config()
        logging.info("Pooling:")
        logging.info("  type: %s", config.get("type"))
        logging.info("  mean_type: %s", config.get("mean_type"))
        logging.info("  include_max: %s", config.get("include_max"))
        logging.info("  init_scale: %s", config.get("init_scale"))
        logging.info("  max_scale: %s", config.get("max_scale"))
        values = self._pooling_gate_values()
        if not values:
            return
        logging.info("Initial pooling gate scales:")
        logging.info(
            "  l0_to_l1 mean=%.3f max=%.3f",
            values.get("l0_to_l1_mean", float("nan")),
            values.get("l0_to_l1_max", float("nan")),
        )
        logging.info(
            "  l1_to_l2 mean=%.3f max=%.3f",
            values.get("l1_to_l2_mean", float("nan")),
            values.get("l1_to_l2_max", float("nan")),
        )
        logging.info(
            "  l2_to_l3 mean=%.3f max=%.3f",
            values.get("l2_to_l3_mean", float("nan")),
            values.get("l2_to_l3_max", float("nan")),
        )

    def _log_and_append_pooling_gate_history(self, epoch: int) -> None:
        values = self._pooling_gate_values()
        if not values:
            return
        logging.info("Pooling gate scales | epoch=%d", int(epoch))
        logging.info(
            "  l0_to_l1 mean=%.6f max=%.6f",
            values.get("l0_to_l1_mean", float("nan")),
            values.get("l0_to_l1_max", float("nan")),
        )
        logging.info(
            "  l1_to_l2 mean=%.6f max=%.6f",
            values.get("l1_to_l2_mean", float("nan")),
            values.get("l1_to_l2_max", float("nan")),
        )
        logging.info(
            "  l2_to_l3 mean=%.6f max=%.6f",
            values.get("l2_to_l3_mean", float("nan")),
            values.get("l2_to_l3_max", float("nan")),
        )
        experiment_dir = str(_get(self.params, "experiment_dir", ""))
        if not experiment_dir or self.world_rank != 0:
            return

        path = os.path.join(experiment_dir, "pooling_gate_history.csv")
        exists = os.path.exists(path)
        fields = ["epoch", *self._pooling_gate_fieldnames()]
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                writer.writeheader()
            row = {"epoch": int(epoch)}
            row.update({key: values.get(key) for key in self._pooling_gate_fieldnames()})
            writer.writerow(row)

    def _pooling_gate_metadata(self) -> dict[str, Any]:
        values = self._pooling_gate_values()
        if not values:
            return {}
        return {
            "pooling_gate_values": {
                key: float(values.get(key, float("nan")))
                for key in self._pooling_gate_fieldnames()
            }
        }

    def _lead_conditioning_config(self) -> dict[str, Any]:
        config = dict(getattr(self, "lead_conditioning_config", {}) or {})
        if not config:
            config = resolve_lead_conditioning(getattr(self, "params", {})).asdict()
        return config

    def _lead_conditioning_enabled(self) -> bool:
        return bool(self._lead_conditioning_config().get("enabled", False))

    def _lead_conditioning_metadata(self) -> dict[str, Any]:
        return {
            "lead_conditioning": self._lead_conditioning_config(),
            "input_channels": int(_get(self.params, "N_in_channels", _get(self.params, "input_channels", 0))),
            "output_channels": int(_get(self.params, "N_out_channels", _get(self.params, "output_channels", 0))),
        }

    def _log_lead_conditioning_startup(self) -> None:
        config = self._lead_conditioning_config()
        logging.info("Lead conditioning:")
        logging.info("  enabled: %s", str(bool(config.get("enabled", False))).lower())
        logging.info("  type: %s", config.get("type", "none"))
        logging.info("  max_lead: %s", config.get("max_lead", 0))
        logging.info("  added_input_channels: %d", int(config.get("added_input_channels", 0)))
        if bool(config.get("enabled", False)):
            for lead, values in lead_conditioning_debug_values(int(config.get("max_lead", 10))).items():
                logging.info("  lead %d: sin=%.6f, cos=%.6f", lead, values["sin"], values["cos"])

    def _forward_model_step(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        aux_features: torch.Tensor | None,
        lead: int,
        diagnostics_collector: Any | None = None,
    ) -> torch.Tensor:
        if diagnostics_collector is None:
            if getattr(self.model, "lead_conditioning_enabled", False):
                if aux_features is None:
                    return self.model.forward_steps(previous, current, lead=int(lead))
                return self.model.forward_steps(previous, current, aux_features=aux_features, lead=int(lead))
            if aux_features is None:
                return self.model.forward_steps(previous, current)
            return self.model.forward_steps(previous, current, aux_features=aux_features)
        if getattr(self.model, "lead_conditioning_enabled", False):
            if aux_features is None:
                return self.model.forward_steps(
                    previous,
                    current,
                    lead=int(lead),
                    diagnostics_collector=diagnostics_collector,
                )
            return self.model.forward_steps(
                previous,
                current,
                aux_features=aux_features,
                lead=int(lead),
                diagnostics_collector=diagnostics_collector,
            )
        if aux_features is None:
            return self.model.forward_steps(previous, current, diagnostics_collector=diagnostics_collector)
        return self.model.forward_steps(previous, current, aux_features=aux_features, diagnostics_collector=diagnostics_collector)

    def _edge_cache_scope(self, warm: bool = True):
        """Lifetime of the shared static edge projections. See GraphWeatherModel.

        Training passes ``warm=False`` and wraps forward *and* backward, because
        checkpointed rollout steps recompute during backward and would see a
        different op sequence if the cache went away first. Warming then happens in
        ``_warm_edge_cache`` from inside the autocast region.
        """
        model = getattr(self, "model", None)
        if not getattr(self, "edge_projection_cache", False) or not hasattr(model, "edge_cache_scope"):
            return nullcontext()
        return model.edge_cache_scope(warm=warm)

    def _warm_edge_cache(self) -> None:
        """Populate the edge cache. Call from inside autocast, before the rollout loop.

        A no-op outside an ``_edge_cache_scope``, so the rollout works unchanged for
        callers that never open one.
        """
        model = getattr(self, "model", None)
        if getattr(self, "edge_projection_cache", False) and getattr(model, "edge_cache_enabled", False):
            model.warm_edge_cache()

    def _forward_model_step_train(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        aux_features: torch.Tensor | None,
        *,
        lead: int,
    ) -> torch.Tensor:
        if (
            not self.model.training
            or not torch.is_grad_enabled()
            or not (bool(getattr(self, "activation_checkpointing", False)) or bool(getattr(self, "checkpoint_rollout_steps", False)))
        ):
            return self._forward_model_step(previous, current, aux_features, lead=lead)

        if aux_features is None:
            def run_step(prev: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
                return self._forward_model_step(prev, cur, None, lead=lead)

            return torch_checkpoint(run_step, previous, current, use_reentrant=False)

        def run_step_with_aux(prev: torch.Tensor, cur: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
            return self._forward_model_step(prev, cur, aux, lead=lead)

        return torch_checkpoint(run_step_with_aux, previous, current, aux_features, use_reentrant=False)

    def _log_train_lead_conditioning_debug_once(self, rollout_steps: int) -> None:
        if not self._lead_conditioning_enabled() or bool(getattr(self, "lead_conditioning_debug_logged", False)):
            return
        leads = list(range(1, int(rollout_steps) + 1))
        logging.info("Lead conditioning debug:")
        logging.info("  train rollout S=%d", int(rollout_steps))
        logging.info("  batch lead values used: %s", format_lead_sequence(leads))
        self.lead_conditioning_debug_logged = True

    def _log_validation_lead_conditioning_debug_once(self, rollout_steps: int) -> None:
        if not self._lead_conditioning_enabled() or bool(getattr(self, "validation_lead_conditioning_debug_logged", False)):
            return
        leads = list(range(1, int(rollout_steps) + 1))
        logging.info("Validation lead conditioning debug:")
        logging.info("  rollout leads used: %s", format_lead_sequence(leads))
        self.validation_lead_conditioning_debug_logged = True

    def _build_loss_channel_weights(
        self,
        channel_names: list[str] | None,
        num_channels: int,
    ) -> list[float] | None:
        """Build a fixed per-channel loss weight vector (length ``num_channels``).

        (i) GraphCast-style per-pressure-level weighting: channels whose name
            ends in digits (e.g. ``u850``, ``z500``) are weighted proportional
            to that pressure level, normalized so the mean weight over the
            distinct levels is 1 (same scale as surface channels = 1).
            ``reference_level`` (default 0.0) overrides that normalization when
            > 0: the divisor becomes the given absolute level, so e.g. 1000 puts
            weight 1.0 at 1000 hPa and makes the weights independent of which
            levels happen to be present. ``min_level_weight`` (default 0.0) is a
            floor applied to the level term, so the final level weight is
            ``max(min_level_weight, level / denom)``. Both defaults reproduce the
            original mean-level behaviour exactly.
        (ii) modest per-variable up-weighting via ``variable_upweights``,
            applied multiplicatively by exact channel name (e.g. t2m:2.0,
            z500:2.0).

        Returns ``None`` when disabled so ``LatitudeWeightedMSE`` stays
        unweighted (current behavior). orog/tisr weights are irrelevant
        because the loss ``channel_mask`` already excludes them.
        """
        cfg = dict(getattr(self, "loss_channel_weight_cfg", {}) or {})
        if not bool(cfg.get("enabled", False)):
            return None
        pressure = bool(cfg.get("pressure_weighting", True))
        upweights = dict(cfg.get("variable_upweights", {}) or {})
        min_level_weight = float(cfg.get("min_level_weight", 0.0))
        reference_level = float(cfg.get("reference_level", 0.0))
        inverse_tendency_variance = bool(cfg.get("inverse_tendency_variance", False))
        itv_exponent = float(cfg.get("itv_exponent", 2.0))
        itv_min_std = float(cfg.get("itv_min_std", 0.01))
        num_channels = int(num_channels)
        names = [str(x) for x in channel_names] if channel_names else []
        if len(names) != num_channels:
            names = (names + [f"channel_{i}" for i in range(len(names), num_channels)])[:num_channels]
        levels: list[int | None] = []
        for name in names:
            m = re.match(r"^[A-Za-z_]+(\d+)$", name)
            levels.append(int(m.group(1)) if m else None)
        present = sorted({lv for lv in levels if lv is not None})
        mean_level = (sum(present) / len(present)) if present else 1.0
        # reference_level > 0 pins the scale to an absolute level (e.g. 1000 ->
        # weight 1.0 at 1000 hPa); otherwise keep the mean-level normalization.
        denom = reference_level if reference_level > 0 else mean_level
        weights = [1.0] * num_channels
        if pressure:
            for i, lv in enumerate(levels):
                if lv is not None:
                    weights[i] = max(min_level_weight, float(lv) / float(denom))
        for i, name in enumerate(names):
            if name in upweights:
                weights[i] *= float(upweights[name])
        itv_factors: list[float] | None = None
        itv_std: list[float] | None = None
        if inverse_tendency_variance:
            delta_std_norm = getattr(self, "loss_delta_std_norm", None)
            if delta_std_norm is None:
                raise ValueError(
                    "loss_channel_weighting.inverse_tendency_variance=true requires the "
                    "normalized-state delta stats; set use_delta_normalization=true and "
                    "delta_stats_path."
                )
            itv_std = [float(x) for x in delta_std_norm.detach().cpu().reshape(-1).tolist()]
            if len(itv_std) != num_channels:
                raise ValueError(
                    f"delta_std length {len(itv_std)} does not match channels {num_channels}."
                )
            itv_factors = [
                1.0 / (max(itv_min_std, itv_std[i]) ** itv_exponent) for i in range(num_channels)
            ]
            for i in range(num_channels):
                weights[i] *= itv_factors[i]
            # Channels the loss excludes (orog/tisr forcings) carry weight 0 and are
            # left out of the normalization, so the mean over *scored* channels is 1.
            mask = getattr(self, "loss_channel_mask", None)
            if mask is None:
                included = [True] * num_channels
            else:
                mask_vals = [float(x) for x in mask.detach().cpu().reshape(-1).tolist()]
                included = [bool(mask_vals[i] > 0.0) for i in range(num_channels)]
            for i in range(num_channels):
                if not included[i]:
                    weights[i] = 0.0
            included_count = sum(1 for flag in included if flag)
            mean_included = (
                sum(weights[i] for i in range(num_channels) if included[i]) / included_count
                if included_count > 0
                else 0.0
            )
            if mean_included > 0.0:
                weights = [w / mean_included for w in weights]
            scored = [weights[i] for i in range(num_channels) if included[i]]
            lo, hi = (min(scored), max(scored)) if scored else (0.0, 0.0)
            logging.info(
                "LOSS_ITV_SUMMARY | exponent=%.4f min_std=%.4f scored_channels=%d "
                "min_weight=%.6g max_weight=%.6g max_over_min=%.6g mean_weight=1.0",
                itv_exponent, itv_min_std, included_count, lo, hi,
                (hi / lo) if lo > 0.0 else float("inf"),
            )
            # Inverse-variance weighting can hand almost the whole loss to one
            # low-tendency channel (e.g. a forcing left in the loss). Surface that
            # before any GPU time is spent rather than after 100 epochs.
            weight_total = sum(scored)
            if weight_total > 0.0:
                for i, name in enumerate(names):
                    if not included[i]:
                        continue
                    share = weights[i] / weight_total
                    if share > 0.10:
                        logging.warning(
                            "LOSS_ITV_CONCENTRATION | channel %s carries %.1f%% of the total loss "
                            "weight (weight=%.4g of sum=%.4g over %d scored channels). "
                            "inverse_tendency_variance is amplifying a very low-tendency channel; "
                            "consider excluding it from the loss or lowering itv_exponent.",
                            name, share * 100.0, weights[i], weight_total, included_count,
                        )
            for i, name in enumerate(names):
                logging.info(
                    "LOSS_ITV | ch=%-6s delta_std_norm=%.6g itv_factor=%.6g final_weight=%.6g "
                    "scored=%s",
                    name, itv_std[i], itv_factors[i], weights[i], included[i],
                )
        # Record the resolved knobs (including defaults) so runs/<name>/
        # config_resolved.yaml documents exactly what the loss used.
        resolved_cfg = dict(cfg)
        resolved_cfg["min_level_weight"] = min_level_weight
        resolved_cfg["reference_level"] = reference_level
        resolved_cfg["inverse_tendency_variance"] = inverse_tendency_variance
        resolved_cfg["itv_exponent"] = itv_exponent
        resolved_cfg["itv_min_std"] = itv_min_std
        self.loss_channel_weight_cfg = resolved_cfg
        if getattr(self, "params", None) is not None:
            _set(self.params, "loss_channel_weighting", resolved_cfg)
        logging.info(
            "Loss channel weighting enabled: pressure_weighting=%s upweights=%s "
            "min_level_weight=%.4f reference_level=%.4f denom=%.4f "
            "(weight range %.4f..%.4f over %d channels)",
            pressure, upweights, min_level_weight, reference_level, float(denom),
            min(weights), max(weights), num_channels,
        )
        # Greppable per-channel table, grouped by variable then pressure level.
        groups: dict[str, list[tuple[int | None, int, float]]] = {}
        for i, name in enumerate(names):
            m = re.match(r"^([A-Za-z_]+)(\d+)$", name)
            var = m.group(1) if m else name
            groups.setdefault(var, []).append((levels[i], i, weights[i]))
        for var in sorted(groups):
            for lv, i, w in sorted(groups[var], key=lambda t: (t[0] is None, t[0] or 0)):
                logging.info(
                    "LOSS_CHANNEL_WEIGHT | var=%-6s level=%-7s idx=%-3d weight=%.4f",
                    var, "surface" if lv is None else str(lv), i, w,
                )
        return weights

    def _loss_channel_metadata(self) -> dict[str, Any]:
        output_channels = int(_get(self.params, "N_out_channels", 0))
        mask = getattr(self, "loss_channel_mask", None)
        if mask is None:
            included = output_channels
        else:
            included = int(mask.detach().cpu().to(torch.float32).sum().item())
        excluded_channels: dict[str, int] = {}
        feature_builder = getattr(self, "feature_builder", None)
        if feature_builder is not None:
            excluded_channels.update(getattr(feature_builder, "exclude_loss_channels", {}) or {})
        target_handler = getattr(self, "target_handler", None)
        if target_handler is not None:
            excluded_channels.update(getattr(target_handler, "exclude_loss_channels", {}) or {})
        return {
            "loss_channels": int(included),
            "output_channels": int(output_channels),
            "excluded_loss_variables": sorted(str(name) for name in excluded_channels),
            "excluded_loss_channels": {str(name): int(idx) for name, idx in sorted(excluded_channels.items())},
        }

    def _save_startup_artifacts(self) -> None:
        if self.world_rank != 0:
            return
        experiment_dir = str(_get(self.params, "experiment_dir", ""))
        if not experiment_dir:
            return
        os.makedirs(experiment_dir, exist_ok=True)
        config_path = os.path.join(experiment_dir, "config_resolved.yaml")
        summary_path = os.path.join(experiment_dir, "model_summary.txt")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(dict(getattr(self.params, "params", {})), f, sort_keys=False)
        summary_lines = [
            "Graph configuration:",
            f"  num_graph_levels: {int(_get(self.params, 'num_graph_levels', 3))}",
            f"  level_k_neighbors: {_get(self.params, 'level_k_neighbors', [])}",
            f"  edge_counts: {_get(self.params, 'edge_counts', [])}",
            f"  graph_path: {_get(self.params, 'graph_path', '')}",
            "",
            "GraphWeather model config:",
        ]
        summary_lines.extend(f"  {line}" for line in self._architecture_summary_lines())
        skip_fusion_config = self._skip_fusion_config()
        summary_lines.extend(
            [
                "",
                "Skip fusion:",
                f"  type: {skip_fusion_config.get('type')}",
                f"  init_scale: {skip_fusion_config.get('init_scale')}",
                f"  max_scale: {skip_fusion_config.get('max_scale')}",
            ]
        )
        gate_values = self._fusion_gate_values()
        if gate_values:
            summary_lines.append("Initial gate scales:")
            for key in self._fusion_gate_fieldnames():
                summary_lines.append(f"  {key}_scale: {gate_values.get(key, float('nan')):.3f}")
        pooling_config = self._pooling_config()
        summary_lines.extend(
            [
                "",
                "Pooling:",
                f"  type: {pooling_config.get('type')}",
                f"  mean_type: {pooling_config.get('mean_type')}",
                f"  include_max: {pooling_config.get('include_max')}",
                f"  init_scale: {pooling_config.get('init_scale')}",
                f"  max_scale: {pooling_config.get('max_scale')}",
            ]
        )
        pooling_values = self._pooling_gate_values()
        if pooling_values:
            summary_lines.append("Initial pooling gate scales:")
            summary_lines.append(
                "  l0_to_l1 mean={:.3f} max={:.3f}".format(
                    pooling_values.get("l0_to_l1_mean", float("nan")),
                    pooling_values.get("l0_to_l1_max", float("nan")),
                )
            )
            summary_lines.append(
                "  l1_to_l2 mean={:.3f} max={:.3f}".format(
                    pooling_values.get("l1_to_l2_mean", float("nan")),
                    pooling_values.get("l1_to_l2_max", float("nan")),
                )
            )
            summary_lines.append(
                "  l2_to_l3 mean={:.3f} max={:.3f}".format(
                    pooling_values.get("l2_to_l3_mean", float("nan")),
                    pooling_values.get("l2_to_l3_max", float("nan")),
                )
            )
        l0_refine_config = dict(_get(self.params, "l0_refine", {}) or {"type": "attention"})
        summary_lines.extend(
            [
                "",
                "L0 refine:",
                f"  type: {l0_refine_config.get('type', 'attention')}",
                f"  mlp_expansion: {l0_refine_config.get('mlp_expansion', 2)}",
                f"  dropout: {l0_refine_config.get('dropout', 0.0)}",
                f"  residual_scale_init: {l0_refine_config.get('residual_scale_init', 0.1)}",
                f"  learnable_residual_scale: {str(bool(l0_refine_config.get('learnable_residual_scale', True))).lower()}",
            ]
        )
        l0_refine_scales = self._l0_refine_residual_scale_values()
        if l0_refine_scales:
            summary_lines.append("Initial L0 refine residual scales:")
            for index, value in sorted(l0_refine_scales.items(), key=lambda item: int(item[0])):
                summary_lines.append(f"  l0_refine.{index}: {float(value):.6f}")
        lead_config = self._lead_conditioning_config()
        summary_lines.extend(
            [
                "",
                "Lead conditioning:",
                f"  enabled: {str(bool(lead_config.get('enabled', False))).lower()}",
                f"  type: {lead_config.get('type', 'none')}",
                f"  max_lead: {lead_config.get('max_lead', 0)}",
                f"  added_input_channels: {int(lead_config.get('added_input_channels', 0))}",
            ]
        )
        if bool(lead_config.get("enabled", False)):
            for lead, values in lead_conditioning_debug_values(int(lead_config.get("max_lead", 10))).items():
                summary_lines.append(f"  lead {lead}: sin={values['sin']:.6f}, cos={values['cos']:.6f}")
        spectral_meta = getattr(self, "spectral_loss_metadata", {"enabled": False})
        summary_lines.extend(
            [
                "",
                "Spectral loss:",
                f"  enabled: {str(bool(spectral_meta.get('enabled', False))).lower()}",
            ]
        )
        if bool(spectral_meta.get("enabled", False)):
            summary_lines.extend(
                [
                    f"  weight: {float(spectral_meta.get('weight', 0.0)):.6g}",
                    f"  variables: {spectral_meta.get('variables', [])}",
                    f"  lat_cutoff: {int(spectral_meta.get('lat_cutoff', 0))}",
                    f"  lon_cutoff: {int(spectral_meta.get('lon_cutoff', 0))}",
                    f"  include_dc: {str(bool(spectral_meta.get('include_dc', True))).lower()}",
                    f"  apply_latitude_weight: {str(bool(spectral_meta.get('apply_latitude_weight', True))).lower()}",
                    f"  space: {spectral_meta.get('space', 'normalized')}",
                ]
            )
        summary_lines.extend(
            [
                "Training rollout configuration:",
                f"  rollout_mode: {self.rollout_mode}",
                f"  training_rollout_steps: {self._rollout_steps_for_epoch()}",
                "  backprop_through_rollout: true",
                "  detach_between_rollout_steps: false",
                f"  rollout_loss_weights: {self._rollout_loss_weights_metadata()}",
                f"  activation_checkpointing: {self.activation_checkpointing}",
                f"  checkpoint_rollout_steps: {self.checkpoint_rollout_steps}",
                f"  attention_impl: {self.attention_impl}",
                f"  edge_projection_cache: {str(bool(self.edge_projection_cache)).lower()}",
                f"  compile_scope: {self._resolve_compile_scope()}",
                f"  batch_size: {self.batch_size}",
                f"  gradient_accumulation_steps: {self.gradient_accumulation_steps}",
                f"  effective_batch_size: {self.effective_batch_size}",
                f"  log_every_batches: {self.log_every_batches}",
                f"Resolution mode: {_get(self.params, 'resolution_mode', 'unknown')}",
                f"Grid shape: {_get(self.params, 'grid_shape', 'unknown')}",
                f"Hidden dimension: {int(_get(self.params, 'hidden_dim', 96))}",
                f"Input channels: {int(_get(self.params, 'N_in_channels', 0))}",
                f"Output channels: {int(_get(self.params, 'N_out_channels', 0))}",
                f"Graph path: {_get(self.params, 'graph_path', '')}",
            ]
        )
        if self.rollout_mode == "random":
            summary_lines.extend(
                [
                    "Random rollout:",
                    f"  min_horizon: {int(self.random_rollout_min_horizon)}",
                    f"  max_horizon: {int(self.random_rollout_max_horizon)}",
                    f"  distribution: {self.random_rollout_distribution}",
                    f"  loss_type: {self.random_rollout_loss_type}",
                    f"  final_step_weight: {float(self.random_rollout_final_step_weight):.6g}",
                    f"  detach_between_steps: {str(bool(self.random_rollout_detach_between_steps)).lower()}",
                ]
            )
        if self.rollout_mode == "scheduled":
            summary_lines.extend(
                [
                    "Scheduled rollout:",
                    f"  max_loaded_horizon: {int(self.scheduled_rollout_max_horizon)}",
                    f"  phases: {self._scheduled_rollout_config().get('phases', [])}",
                    f"  loss_type: {self.random_rollout_loss_type}",
                    f"  final_step_weight: {float(self.random_rollout_final_step_weight):.6g}",
                    f"  detach_between_steps: {str(bool(self.random_rollout_detach_between_steps)).lower()}",
                ]
            )
        loss_meta = self._loss_channel_metadata()
        target_meta = self.target_handler.metadata if hasattr(self, "target_handler") else {}
        summary_lines.extend(
            [
                "Target handling:",
                f"  enabled: {str(bool(target_meta.get('enabled', False))).lower()}",
                f"  copy_variables: {target_meta.get('copy_variables', [])}",
                f"  known_future_variables: {target_meta.get('known_future_variables', [])}",
                f"  exclude_loss_variables: {target_meta.get('exclude_loss_variables', [])}",
                f"Loss channels: {loss_meta['loss_channels']}/{loss_meta['output_channels']}",
            ]
        )
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines) + "\n")
        logging.info("Saved resolved config: %s", config_path)
        logging.info("Saved model summary: %s", summary_path)

    def _log_startup_config(self) -> None:
        spec = get_resolution_spec(_get(self.params, "resolution_mode", "5p625"))
        logging.info("Graph configuration:")
        logging.info("  num_graph_levels: %d", int(_get(self.params, "num_graph_levels", 3)))
        logging.info("  level_k_neighbors: %s", _get(self.params, "level_k_neighbors", []))
        logging.info("  edge_counts: %s", _get(self.params, "edge_counts", []))
        logging.info("  graph_path: %s", _get(self.params, "graph_path", ""))
        logging.info("Model configuration:")
        logging.info("  hidden_dim: %d", int(_get(self.params, "hidden_dim", 96)))
        logging.info("  num_heads: %d", int(_get(self.params, "num_heads", _get(self.params, "heads", 4))))
        logging.info(
            "  head_dim: %d",
            int(_get(self.params, "head_dim", int(_get(self.params, "hidden_dim", 96)) // int(_get(self.params, "num_heads", _get(self.params, "heads", 4))))),
        )
        logging.info("  use_l3: %s", str(bool(_get(self.params, "use_l3", False))).lower())
        logging.info("  use_l4_ratio15: %s", str(bool(_get(self.params, "use_l4_ratio15", False))).lower())
        logging.info("  hierarchy_type: %s", str(_get(self.params, "hierarchy_type", "standard")))
        logging.info("  l0_blocks: %d", int(_get(self.params, "l0_blocks", 2)))
        logging.info("  l1_blocks: %d", int(_get(self.params, "l1_blocks", 2)))
        logging.info("  l2_blocks: %d", int(_get(self.params, "l2_blocks", 1)))
        logging.info("  l3_blocks: %d", int(_get(self.params, "l3_blocks", 1)))
        logging.info("  l4_blocks: %d", int(_get(self.params, "l4_blocks", 1)))
        logging.info("  l3_refine_after_l4_blocks: %d", int(_get(self.params, "l3_refine_after_l4_blocks", 1)))
        logging.info("  l2_refine_after_l3_blocks: %d", int(_get(self.params, "l2_refine_after_l3_blocks", 1)))
        logging.info("  l1_refine_blocks: %d", int(_get(self.params, "l1_refine_blocks", 1)))
        logging.info("  l0_refine_blocks: %d", int(_get(self.params, "l0_refine_blocks", 1)))
        logging.info("  l0_refine: %s", _get(self.params, "l0_refine", {"type": "attention"}))
        logging.info("  trainable_parameters: %d", self.num_parameters)
        logging.info("Training rollout mode: %s", self.rollout_mode)
        logging.info("Training rollout steps: %d", self._rollout_steps_for_epoch())
        logging.info("Backprop through rollout: true")
        logging.info("Detach between rollout steps: false")
        logging.info("Rollout loss weights: %s", self._rollout_loss_weights_metadata())
        if self.rollout_mode == "fixed_full":
            logging.info("Curriculum schedule: disabled")
        elif self.rollout_mode == "random":
            logging.info(
                "Random rollout: min=%d max=%d distribution=%s loss_type=%s final_step_weight=%.6g detach_between_steps=%s",
                int(self.random_rollout_min_horizon),
                int(self.random_rollout_max_horizon),
                self.random_rollout_distribution,
                self.random_rollout_loss_type,
                float(self.random_rollout_final_step_weight),
                str(bool(self.random_rollout_detach_between_steps)).lower(),
            )
        elif self.rollout_mode == "scheduled":
            logging.info(
                "Scheduled rollout: max_loaded_horizon=%d phases=%s loss_type=%s final_step_weight=%.6g detach_between_steps=%s",
                int(self.scheduled_rollout_max_horizon),
                self._scheduled_rollout_config().get("phases", []),
                self.random_rollout_loss_type,
                float(self.random_rollout_final_step_weight),
                str(bool(self.random_rollout_detach_between_steps)).lower(),
            )
        else:
            logging.info(
                "Curriculum schedule: rollout_schedule=%s rollout_stage_epochs=%s",
                self.rollout_schedule,
                self.rollout_stage_epochs,
            )
        logging.info("Scheduler: %s", self.lr_schedule_type)
        logging.info("Resolution mode: %s", spec.name)
        logging.info("Resolution: %s degrees", spec.resolution_degrees)
        logging.info("Grid: %d x %d", spec.height, spec.width)
        logging.info("Graph levels:")
        level_names = ["L0", "L1", "L2"]
        if bool(getattr(self.graph, "use_l3", False)):
            level_names.append("L3")
        if bool(getattr(self.graph, "use_l4", False)):
            level_names.append("L4")
        for name in level_names:
            level = getattr(self.graph, name)
            logging.info("  %s: %d nodes, %d edges", name, level.num_nodes, int(level.edge_index.shape[1]))
        logging.info("Input channels: %d", int(_get(self.params, "N_in_channels", 0)))
        logging.info("Output channels: %d", int(_get(self.params, "N_out_channels", 0)))
        logging.info("Hidden dimension: %d", int(_get(self.params, "hidden_dim", 96)))
        logging.info(
            "Model width: hidden_dim=%d, num_heads=%d, head_dim=%d",
            int(_get(self.params, "hidden_dim", 96)),
            int(_get(self.params, "num_heads", _get(self.params, "heads", 4))),
            int(_get(self.params, "head_dim", int(_get(self.params, "hidden_dim", 96)) // int(_get(self.params, "num_heads", _get(self.params, "heads", 4))))),
        )
        logging.info("Trainable parameters: %d", self.num_parameters)
        if hasattr(self, "feature_builder"):
            self.feature_builder.log_startup(
                base_input_channels=int(_get(self.params, "base_input_channels", _get(self.params, "N_in_channels", 0))),
                total_input_channels=int(_get(self.params, "total_input_channels", _get(self.params, "N_in_channels", 0))),
                output_channels=int(_get(self.params, "N_out_channels", 0)),
            )
        if hasattr(self, "target_handler"):
            self.target_handler.log_startup(output_channels=int(_get(self.params, "N_out_channels", 0)))
        logging.info("Batch size: %d", int(_get(self.params, "batch_size", 1)))
        logging.info("Gradient accumulation steps: %d", self.gradient_accumulation_steps)
        logging.info(
            "Effective batch size: %d",
            self.effective_batch_size,
        )
        logging.info("Log every batches: %d", self.log_every_batches)
        logging.info("Fixed train rollout steps: %d", self.fixed_train_rollout_steps)
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
        if self.lr_schedule_type == "rollout_stage_warmup_cosine":
            logging.info("LR scheduler: rollout_stage_warmup_cosine")
            logging.info("Stage LR schedule:")
            scheduler = self.scheduler
            if isinstance(scheduler, RolloutStageWarmupCosineScheduler):
                for stage in scheduler.stages:
                    logging.info(
                        "  S=%-2d: warmup %d epoch%s, %.6g -> %.6g -> %.6g",
                        stage.rollout_steps,
                        stage.warmup_epochs,
                        "" if stage.warmup_epochs == 1 else "s",
                        stage.warmup_start_lr,
                        stage.max_lr,
                        stage.min_lr,
                    )
            return
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
        if getattr(self, "rollout_mode", "curriculum") == "fixed_full":
            return min(int(self.fixed_train_rollout_steps), self.max_rollout_steps)
        if getattr(self, "rollout_mode", "curriculum") == "random":
            return min(int(self.random_rollout_max_horizon), self.max_rollout_steps)
        if getattr(self, "rollout_mode", "curriculum") == "scheduled":
            return min(int(self.scheduled_rollout_max_horizon), self.max_rollout_steps)
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

    def _append_lr_schedule_row(self, row: dict[str, Any]) -> None:
        experiment_dir = str(_get(self.params, "experiment_dir", ""))
        if not experiment_dir:
            return

        path = os.path.join(experiment_dir, "lr_schedule.csv")
        exists = os.path.exists(path)
        fields = ["global_step", "epoch", "rollout_steps", "stage_index", "stage_step", "lr"]
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow(
                {
                    "global_step": int(row["global_step"]),
                    "epoch": int(row["epoch"]),
                    "rollout_steps": int(row["rollout_steps"]),
                    "stage_index": int(row["stage_index"]),
                    "stage_step": int(row["stage_step"]),
                    "lr": f"{float(row['lr']):.12g}",
                }
            )

    def _write_lr_schedule_plot(self) -> None:
        if self.lr_schedule_type != "rollout_stage_warmup_cosine":
            return
        experiment_dir = str(_get(self.params, "experiment_dir", ""))
        if not experiment_dir:
            return
        csv_path = os.path.join(experiment_dir, "lr_schedule.csv")
        if not os.path.exists(csv_path):
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            steps: list[int] = []
            lrs: list[float] = []
            rollout_steps: list[int] = []
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    steps.append(int(row["global_step"]))
                    lrs.append(float(row["lr"]))
                    rollout_steps.append(int(row["rollout_steps"]))
            if not steps:
                return
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(steps, lrs, linewidth=1.4)
            last_stage = None
            for step, rollout in zip(steps, rollout_steps):
                if rollout != last_stage:
                    ax.axvline(step, color="0.65", linewidth=0.8, alpha=0.7)
                    ax.text(step, max(lrs), f"S={rollout}", fontsize=8, rotation=90, va="top", ha="right")
                    last_stage = rollout
            ax.set_xlabel("Optimizer step")
            ax.set_ylabel("Learning rate")
            ax.set_title("Rollout-stage warmup + cosine LR")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(os.path.join(experiment_dir, "lr_schedule.png"), dpi=160)
            plt.close(fig)
        except Exception as exc:
            if not self._lr_schedule_plot_warning_emitted:
                logging.warning("Could not write lr_schedule.png: %s", exc)
                self._lr_schedule_plot_warning_emitted = True

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
        if self.lr_schedule_type not in {"rollout_stage", "manual_by_rollout"}:
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
        loss_mask = getattr(self, "loss_channel_mask", None)
        try:
            loss = self.loss_obj(pred, gt, channel_mask=loss_mask)
        except TypeError:
            loss = self.loss_obj(pred, gt)
        if self.graph_gradient_weight > 0.0:
            grad_pred = pred
            grad_gt = gt
            if loss_mask is not None:
                include = loss_mask.to(device=pred.device) > 0.5
                grad_pred = pred[:, include]
                grad_gt = gt[:, include]
            loss = loss + self.graph_gradient_weight * graph_gradient_loss(
                grad_pred,
                grad_gt,
                self.graph.L0.edge_index,
            )
        return loss

    def _spectral_step_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        if not bool(getattr(self, "spectral_loss_enabled", False)):
            return torch.zeros((), device=pred.device, dtype=torch.float32)
        if str(getattr(self, "spectral_loss_mode", "error_low_k")) == "power_match":
            return band_spectral_power_loss(
                pred,
                gt,
                getattr(self, "spectral_loss_channel_indices"),
                band_weights=getattr(self, "spectral_loss_band_weights", [0.0, 1.0, 1.0]),
                weight_latitude=bool(getattr(self, "spectral_loss_apply_latitude_weight", True)),
                latitudes_rad=getattr(self, "spectral_latitudes_rad", None),
            )
        return low_frequency_spectral_loss(
            pred,
            gt,
            getattr(self, "spectral_loss_channel_indices"),
            int(getattr(self, "spectral_loss_lat_cutoff", 8)),
            int(getattr(self, "spectral_loss_lon_cutoff", 16)),
            weight_latitude=bool(getattr(self, "spectral_loss_apply_latitude_weight", True)),
            include_dc=bool(getattr(self, "spectral_loss_include_dc", True)),
            latitudes_rad=getattr(self, "spectral_latitudes_rad", None),
        )

    def _step_metadata(self, metadata: dict[str, Any], key: str) -> torch.Tensor | None:
        value = metadata.get(key)
        return value if torch.is_tensor(value) else None

    def _build_aux_for_step(
        self,
        current: torch.Tensor,
        target_seq: torch.Tensor,
        metadata: dict[str, Any],
        step: int,
    ) -> torch.Tensor | None:
        feature_builder = getattr(self, "feature_builder", None)
        if feature_builder is None:
            return None
        target_norm = target_seq[:, int(step)]
        return feature_builder.build_step_features(
            current=current,
            target_norm=target_norm,
            target_dayofyear=self._step_metadata(metadata, "target_dayofyear"),
            target_days_in_year=self._step_metadata(metadata, "target_days_in_year"),
            step_idx=int(step),
        )

    def _rollout_loss(
        self,
        inp: torch.Tensor,
        target: torch.Tensor,
        rollout_steps: int,
        metadata: dict[str, Any] | None = None,
        return_lead_losses: bool = False,
        return_loss_components: bool = False,
        include_spectral_loss: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]] | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], dict[str, Any]]:
        metadata = metadata or {}
        target_seq = self._target_sequence(target)
        requested_steps = int(rollout_steps)
        if target_seq.shape[1] < requested_steps:
            raise AssertionError(f"Target length {target_seq.shape[1]} is shorter than rollout_steps={requested_steps}")
        rollout_steps = requested_steps
        previous, current = self.model.adapter.extract_two_steps(inp)
        total = torch.zeros((), device=inp.device, dtype=inp.dtype)
        weights = self._rollout_loss_weights_for_steps(rollout_steps, device=inp.device, dtype=total.dtype)
        weight_sum = weights.sum()
        lead_losses: list[torch.Tensor] = []
        grid_lead_losses: list[torch.Tensor] = []
        spectral_raw_lead_losses: list[torch.Tensor] = []
        spectral_weighted_lead_losses: list[torch.Tensor] = []
        last_pred = None
        initial_state = current
        self._log_train_lead_conditioning_debug_once(rollout_steps)
        # Inside the caller's autocast context and outside every checkpointed step.
        self._warm_edge_cache()
        for step in range(rollout_steps):
            lead = step + 1
            gt = target_seq[:, step]
            aux = self._build_aux_for_step(current, target_seq, metadata, step)
            pred = self._forward_model_step_train(previous, current, aux, lead=lead)
            feature_builder = getattr(self, "feature_builder", None)
            if feature_builder is not None:
                pred = feature_builder.apply_overrides(pred, current=current, target_norm=gt)
            target_handler = getattr(self, "target_handler", None)
            if target_handler is not None:
                pred = target_handler.apply(
                    pred_next=pred,
                    current_state=current,
                    initial_state=initial_state,
                    target_sequence=target_seq,
                    lead=lead,
                )
            grid_loss = self._step_loss(pred, gt)
            spectral_raw = (
                self._spectral_step_loss(pred, gt)
                if include_spectral_loss
                else torch.zeros((), device=pred.device, dtype=torch.float32)
            )
            spectral_weighted = float(getattr(self, "spectral_loss_weight", 0.0)) * spectral_raw
            loss = grid_loss + spectral_weighted
            lead_losses.append(loss)
            grid_lead_losses.append(grid_loss)
            spectral_raw_lead_losses.append(spectral_raw)
            spectral_weighted_lead_losses.append(spectral_weighted)
            total = total + weights[step] * loss
            next_step = current.clone()
            next_step[:, : self.model.output_channels] = pred
            previous, current = current, next_step
            last_pred = pred
        mean_loss = total / weight_sum
        if return_loss_components:
            components = {
                "grid_lead_losses": grid_lead_losses,
                "spectral_raw_lead_losses": spectral_raw_lead_losses,
                "spectral_weighted_lead_losses": spectral_weighted_lead_losses,
                "grid_loss": sum(weights[idx] * grid_lead_losses[idx] for idx in range(rollout_steps)) / weight_sum,
                "spectral_raw_loss": sum(weights[idx] * spectral_raw_lead_losses[idx] for idx in range(rollout_steps)) / weight_sum,
                "spectral_weighted_loss": sum(weights[idx] * spectral_weighted_lead_losses[idx] for idx in range(rollout_steps)) / weight_sum,
            }
            return mean_loss, last_pred, lead_losses, components
        if return_lead_losses:
            return mean_loss, last_pred, lead_losses
        return mean_loss, last_pred

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
            scheduled_phase = self._get_scheduled_rollout_phase(epoch + 1) if self.rollout_mode == "scheduled" else None
            stage_changed = self._last_stage_rollout_steps != train_rollout_steps
            phase_changed = (
                self.rollout_mode == "scheduled"
                and scheduled_phase is not None
                and self._last_scheduled_phase_name != str(scheduled_phase.get("name"))
            )
            if stage_changed or phase_changed:
                if self.rollout_mode == "fixed_full":
                    logging.info(
                        "Fixed full-rollout training: S=%d | curriculum disabled | batch_size=%d | "
                        "accumulation=%d | effective_batch=%d",
                        train_rollout_steps,
                        self.batch_size,
                        self.gradient_accumulation_steps,
                        self.effective_batch_size,
                    )
                elif self.rollout_mode == "random":
                    logging.info(
                        "Random rollout training: S~Uniform{%d..%d} | target_horizon=%d | "
                        "batch_size=%d | accumulation=%d | effective_batch=%d",
                        int(self.random_rollout_min_horizon),
                        int(self.random_rollout_max_horizon),
                        train_rollout_steps,
                        self.batch_size,
                        self.gradient_accumulation_steps,
                        self.effective_batch_size,
                    )
                elif self.rollout_mode == "scheduled" and scheduled_phase is not None:
                    if scheduled_phase["mode"] == "random":
                        logging.info(
                            "Scheduled rollout phase transition: epoch=%d | rollout_phase=%s | mode=random | "
                            "sampled_S~Uniform{%d..%d} | train_loaded_Smax=%d | batch_size=%d | accumulation=%d | effective_batch=%d",
                            epoch + 1,
                            scheduled_phase["name"],
                            int(scheduled_phase["min_horizon"]),
                            int(scheduled_phase["max_horizon"]),
                            train_rollout_steps,
                            self.batch_size,
                            self.gradient_accumulation_steps,
                            self.effective_batch_size,
                        )
                    else:
                        logging.info(
                            "Scheduled rollout phase transition: epoch=%d | rollout_phase=%s | mode=fixed | "
                            "train_actual_S=%d | train_loaded_Smax=%d | batch_size=%d | accumulation=%d | effective_batch=%d",
                            epoch + 1,
                            scheduled_phase["name"],
                            int(scheduled_phase["horizon"]),
                            train_rollout_steps,
                            self.batch_size,
                            self.gradient_accumulation_steps,
                            self.effective_batch_size,
                        )
                else:
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
                if scheduled_phase is not None:
                    self._last_scheduled_phase_name = str(scheduled_phase.get("name"))
            self._ensure_train_loader_rollout(train_rollout_steps)
            epoch_lr = self._lr_for_epoch(train_rollout_steps)
            if self.lr_schedule_type in {"warmup_cosine", "rollout_stage", "manual_by_rollout", "none"}:
                self._set_optimizer_lr(epoch_lr)
            if self.lr_schedule_type in {"rollout_stage", "manual_by_rollout"}:
                logging.info("Stage learning rate for S=%d: %.6g", train_rollout_steps, epoch_lr)
            elif self.lr_schedule_type == "warmup_cosine":
                logging.info("Warmup/cosine learning rate for epoch %d: %.6g", epoch + 1, epoch_lr)
            elif self.lr_schedule_type == "rollout_stage_warmup_cosine":
                logging.info(
                    "Rollout-stage warmup/cosine learning rate for epoch %d starts at %.6g",
                    epoch + 1,
                    self._current_lr(),
                )
            if stage_changed or phase_changed:
                label = (
                    "Fixed rollout settings"
                    if self.rollout_mode == "fixed_full"
                    else "Scheduled rollout settings"
                    if self.rollout_mode == "scheduled"
                    else "Random rollout settings"
                    if self.rollout_mode == "random"
                    else "Stage transition settings"
                )
                if self.rollout_mode == "scheduled" and scheduled_phase is not None:
                    actual = (
                        f"sampled_S~Uniform{{{int(scheduled_phase['min_horizon'])}..{int(scheduled_phase['max_horizon'])}}}"
                        if scheduled_phase["mode"] == "random"
                        else f"train_actual_S={int(scheduled_phase['horizon'])}"
                    )
                    logging.info(
                        "%s | rollout_phase=%s | phase_mode=%s | %s | train_loaded_Smax=%d | "
                        "batch_size=%d | accumulation=%d | effective_batch=%d | lr=%.6g | scheduler=%s",
                        label,
                        scheduled_phase["name"],
                        scheduled_phase["mode"],
                        actual,
                        int(self._current_train_target_rollout_steps or train_rollout_steps),
                        self.batch_size,
                        self.gradient_accumulation_steps,
                        self.effective_batch_size,
                        epoch_lr,
                        self.lr_schedule_type,
                    )
                else:
                    logging.info(
                        "%s | S=%d | target_horizon=%d | batch_size=%d | accumulation=%d | "
                        "effective_batch=%d | lr=%.6g | scheduler=%s",
                        label,
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
            self._ema_swap_in()
            valid_time, valid_logs = self.validate_one_epoch(rollout_steps=valid_rollout_steps)
            valid_logs["valid_time_sec"] = float(valid_time)

            if self.scheduler is not None:
                if isinstance(self.scheduler, RolloutStageWarmupCosineScheduler):
                    epoch_lr = float(train_logs.get("lr_end", self._current_lr()))
                elif isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(valid_logs["valid_loss"])
                    epoch_lr = self._current_lr()
                else:
                    self.scheduler.step()
                    epoch_lr = self._current_lr()

            diagnostics_time = 0.0
            diagnostics = getattr(self, "diagnostics_manager", None)
            diagnostics_enabled = bool(getattr(diagnostics, "enabled", False))
            diagnostics_ran = False
            if diagnostics_enabled:
                diagnostics_start = time.time()
                diagnostics_epoch = int(self.epoch)
                run_light = diagnostics.should_run_light(diagnostics_epoch)
                run_heavy = diagnostics.should_run_heavy(diagnostics_epoch)
                if run_light or run_heavy:
                    diagnostics_ran = True
                    diagnostics.run_train_diagnostics(epoch=diagnostics_epoch, trainer=self, heavy=False)
                    diagnostics.run_valid_diagnostics(epoch=diagnostics_epoch, trainer=self, heavy=run_heavy)
                diagnostics_time = time.time() - diagnostics_start

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
                "train_loss_mean": train_logs.get("train_loss_mean", train_logs["loss"]),
                "train_loss_final": train_logs.get("train_loss_final"),
                "train_loss_by_lead": {
                    str(lead): train_logs.get(f"train_loss_lead{lead}")
                    for lead in range(1, self.max_rollout_steps + 1)
                    if train_logs.get(f"train_loss_lead{lead}") is not None
                },
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
                "median_step_time_sec": train_logs.get("median_step_time_sec"),
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
                **self._training_rollout_metadata(),
                "lr_schedule_type": self.lr_schedule_type,
                "lr": self.base_lr,
                "min_lr": self.min_lr,
                "warmup_epochs": self.warmup_epochs,
                "warmup_start_factor": self.warmup_start_factor,
                "current_lr": epoch_lr,
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
                "rollout_stage_lr_schedule": _get(self.params, "rollout_stage_lr_schedule", None),
                "global_optimizer_step": int(getattr(self.scheduler, "global_step", 0))
                if isinstance(self.scheduler, RolloutStageWarmupCosineScheduler)
                else None,
                "diagnostics_time_sec": float(diagnostics_time),
            }
            if self.rollout_mode in {"random", "scheduled"}:
                self._last_epoch_metadata["random_rollout"] = {
                    "sampled_horizon_mean": train_logs.get("train_random_sampled_horizon_mean"),
                    "sampled_horizon_min": train_logs.get("train_random_sampled_horizon_min"),
                    "sampled_horizon_max": train_logs.get("train_random_sampled_horizon_max"),
                    "sampled_horizon_std": train_logs.get("train_random_sampled_horizon_std"),
                    "loss_mean_all_steps": train_logs.get("train_random_loss_mean_all_steps"),
                    "loss_final_sampled_horizon": train_logs.get("train_random_loss_final_sampled_horizon"),
                    "final_to_mean_ratio": train_logs.get("train_random_final_to_mean_ratio"),
                }
            if self.rollout_mode == "scheduled":
                self._last_epoch_metadata["scheduled_rollout"] = {
                    "phases": self._scheduled_rollout_config().get("phases", []),
                    "active_phase": {
                        "id": train_logs.get("train_rollout_phase_id"),
                        "name": train_logs.get("train_rollout_phase_name"),
                        "mode": train_logs.get("train_rollout_phase_mode"),
                        "start_epoch": train_logs.get("train_rollout_phase_start_epoch"),
                        "end_epoch": train_logs.get("train_rollout_phase_end_epoch"),
                        "fixed_horizon": train_logs.get("train_rollout_phase_fixed_horizon"),
                        "random_min_horizon": train_logs.get("train_rollout_phase_random_min_horizon"),
                        "random_max_horizon": train_logs.get("train_rollout_phase_random_max_horizon"),
                    },
                    "sampled_horizon_mean": train_logs.get("train_scheduled_sampled_horizon_mean"),
                    "sampled_horizon_min": train_logs.get("train_scheduled_sampled_horizon_min"),
                    "sampled_horizon_max": train_logs.get("train_scheduled_sampled_horizon_max"),
                    "sampled_horizon_std": train_logs.get("train_scheduled_sampled_horizon_std"),
                    "loss_mean_all_steps": train_logs.get("train_scheduled_loss_mean_all_steps"),
                    "loss_final_sampled_horizon": train_logs.get("train_scheduled_loss_final_sampled_horizon"),
                    "final_to_mean_ratio": train_logs.get("train_scheduled_final_to_mean_ratio"),
                }
            self._last_epoch_metadata.update(
                self.feature_builder.checkpoint_metadata(
                    base_input_channels=int(_get(self.params, "base_input_channels", _get(self.params, "N_in_channels", 0))),
                    total_input_channels=int(_get(self.params, "total_input_channels", _get(self.params, "N_in_channels", 0))),
                )
            )
            self._last_epoch_metadata.update(self.target_handler.checkpoint_metadata())
            self._last_epoch_metadata.update(self._loss_channel_metadata())
            self._last_epoch_metadata.update(
                resolution_metadata(
                    _get(self.params, "resolution_mode", "5p625"),
                    k=int(_get(self.params, "k_neighbors", 8)),
                    num_parameters=self.num_parameters,
                    num_graph_levels=int(_get(self.params, "num_graph_levels", 3)),
                    level_k_neighbors=_get(self.params, "level_k_neighbors", None),
                    level_shapes=_get(self.params, "level_shapes", None),
                    hierarchy_type=str(_get(self.params, "hierarchy_type", "standard")),
                    use_l4_ratio15=bool(_get(self.params, "use_l4_ratio15", False)),
                )
            )
            self._last_epoch_metadata.update(
                architecture_metadata(
                    self.params,
                    graph_metadata=self.graph.metadata,
                    num_parameters=self.num_parameters,
                )
            )
            self._last_epoch_metadata.update(self._fusion_gate_metadata())
            self._last_epoch_metadata.update(self._pooling_gate_metadata())
            self._last_epoch_metadata.update(self._lead_conditioning_metadata())
            self._last_epoch_metadata.update(self._spectral_loss_checkpoint_metadata())

            if bool(_get(self.params, "save_checkpoint", True)):
                checkpoint_start = time.time()
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
                checkpoint_time = time.time() - checkpoint_start
            else:
                checkpoint_time = 0.0
            self._ema_swap_out()
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
            self._last_epoch_metadata["checkpoint_time_sec"] = float(checkpoint_time)
            self._last_epoch_metadata["epoch_wall_time_sec"] = float(epoch_time)
            self._append_random_rollout_local_logs(epoch + 1, train_logs)
            self._append_valid_rollout_curve_local_log(epoch + 1, valid_logs)
            self._append_valid_stepwise_local_logs(epoch + 1, valid_logs)
            self._log_epoch_wandb(
                epoch=epoch + 1,
                train_logs=train_logs,
                valid_logs=valid_logs,
                lr=epoch_lr,
                epoch_time_sec=epoch_time,
                train_time_sec=tr_time,
                valid_time_sec=valid_time,
                cuda_peak_allocated_gb=cuda_peak_allocated_gb,
                cuda_peak_reserved_gb=cuda_peak_reserved_gb,
            )
            if diagnostics_enabled and diagnostics_ran:
                system_diag_metrics = {
                    "system/epoch_time_sec": float(epoch_time),
                    "system/train_time_sec": float(tr_time),
                    "system/valid_time_sec": float(valid_time),
                    "system/diagnostics_time_sec": float(diagnostics_time),
                    "system/train_samples_per_sec": float(train_logs.get("train_samples_per_second", 0.0)),
                    "system/valid_samples_per_sec": float(valid_logs.get("valid_samples_per_second", 0.0)),
                    "system/samples_per_sec": float(train_logs.get("train_samples_per_second", 0.0)),
                }
                if cuda_peak_allocated_gb is not None and cuda_peak_reserved_gb is not None:
                    system_diag_metrics["system/gpu_peak_allocated_mb"] = float(cuda_peak_allocated_gb * 1024.0)
                    system_diag_metrics["system/gpu_peak_reserved_mb"] = float(cuda_peak_reserved_gb * 1024.0)
                if self.rollout_mode in {"random", "scheduled"}:
                    system_diag_metrics["system/sampled_horizon_mean"] = _finite_or_nan(
                        train_logs.get("train_random_sampled_horizon_mean")
                    )
                    system_diag_metrics["system/random_rollout_effective_avg_horizon"] = _finite_or_nan(
                        train_logs.get("train_random_sampled_horizon_mean")
                    )
                if self.rollout_mode == "scheduled":
                    system_diag_metrics["system/scheduled_rollout_effective_avg_horizon"] = _finite_or_nan(
                        train_logs.get("train_scheduled_sampled_horizon_mean")
                    )
                    system_diag_metrics["system/rollout_phase_id"] = _finite_or_nan(train_logs.get("train_rollout_phase_id"))
                diagnostics.save_system_metrics(int(self.epoch), system_diag_metrics)
            logging.info(
                "Runtime | train_batches=%d | optimizer_steps=%d | train_samples/sec=%.3f | "
                "valid_seconds=%.2f | valid_samples/sec=%.3f | diagnostics_seconds=%.2f | epoch_seconds=%.2f",
                int(train_logs.get("train_batches", 0)),
                int(train_logs.get("optimizer_steps", 0)),
                float(train_logs.get("train_samples_per_second", 0.0)),
                float(valid_time),
                float(valid_logs.get("valid_samples_per_second", 0.0)),
                float(diagnostics_time),
                float(epoch_time),
            )
            logging.info(
                "Perf | attention_impl=%s | edge_projection_cache=%s | compile_scope=%s | "
                "median_step_time_s=%.4f | peak_memory_allocated_gb=%s",
                str(getattr(self, "attention_impl", ATTENTION_IMPL_DEFAULT)),
                str(bool(self.edge_projection_cache)).lower(),
                self._resolve_compile_scope(),
                float(train_logs.get("median_step_time_sec", float("nan"))),
                "n/a" if cuda_peak_allocated_gb is None else f"{float(cuda_peak_allocated_gb):.3f}",
            )
            if self.log_timing_breakdown:
                logging.info(
                    "Timing breakdown | data_seconds=%.2f | forward_loss_seconds=%.2f | "
                    "backward_seconds=%.2f | optimizer_seconds=%.2f | validation_seconds=%.2f | diagnostics_seconds=%.2f | "
                    "checkpoint_seconds=%.2f | epoch_wall_seconds=%.2f",
                    float(train_logs.get("data_time_sec", 0.0)),
                    float(train_logs.get("forward_loss_time_sec", 0.0)),
                    float(train_logs.get("backward_time_sec", 0.0)),
                    float(train_logs.get("optimizer_time_sec", 0.0)),
                    float(valid_time),
                    float(diagnostics_time),
                    float(checkpoint_time),
                    float(epoch_time),
                )
            logging.info(
                "LR epoch summary | epoch=%d | S=%d | stage_index=%s | stage_epoch=%s | "
                "lr_start=%.6g | lr_end=%.6g | lr_min=%.6g | lr_max=%.6g",
                epoch + 1,
                train_rollout_steps,
                train_logs.get("stage_index"),
                train_logs.get("stage_epoch"),
                float(train_logs.get("lr_start", epoch_lr)),
                float(train_logs.get("lr_end", epoch_lr)),
                float(train_logs.get("lr_min_this_epoch", epoch_lr)),
                float(train_logs.get("lr_max_this_epoch", epoch_lr)),
            )
            self._log_and_append_fusion_gate_history(epoch + 1)
            self._log_and_append_pooling_gate_history(epoch + 1)
            rollout_phase_suffix = ""
            if self.rollout_mode == "scheduled":
                rollout_phase_suffix = (
                    f" | rollout_phase={train_logs.get('train_rollout_phase_name', '')}"
                    f" phase_mode={train_logs.get('train_rollout_phase_mode', '')}"
                    f" sampled_S_mean={_finite_or_nan(train_logs.get('train_scheduled_sampled_horizon_mean')):.3f}"
                    f" sampled_S_min={_finite_or_nan(train_logs.get('train_scheduled_sampled_horizon_min')):.0f}"
                    f" sampled_S_max={_finite_or_nan(train_logs.get('train_scheduled_sampled_horizon_max')):.0f}"
                    f" train_loaded_Smax={int(train_logs.get('target_rollout_steps_loaded', train_rollout_steps))}"
                )
            logging.info(
                "Epoch %d | mode=%s | grid=%dx%d | train S=%d%s | finished in %.2f sec | "
                "train_avg %.6f | train_last %.6f | valid %.6f S=%d | valid_final %.6f%s | "
                "lr %.2e | scheduler %s%s",
                epoch + 1,
                _get(self.params, "resolution_mode", "5p625"),
                int(_get(self.params, "crop_size_x", 0)),
                int(_get(self.params, "crop_size_y", 0)),
                train_rollout_steps,
                rollout_phase_suffix,
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
            self._append_training_metrics_csv(epoch + 1, train_logs, valid_logs, epoch_lr, epoch_time)
            self._append_s1_overfit_curve_csv(epoch + 1, train_logs, valid_logs, epoch_lr)
            self._write_lr_schedule_plot()
        diagnostics = getattr(self, "diagnostics_manager", None)
        if bool(getattr(diagnostics, "enabled", False)) and bool(diagnostics.config.get("run_after_training", True)):
            diagnostics.run_full_post_training_diagnostics(trainer=self, split="valid")

    def train_one_epoch(self) -> tuple[float, dict[str, float]]:
        self.model.train()
        rollout_steps = self._rollout_steps_for_epoch()
        self._ensure_train_loader_rollout(rollout_steps)
        if self.world_size > 1:
            sampler = getattr(self.train_data_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(int(self.epoch))
        random_rollout = self.rollout_mode == "random"
        scheduled_rollout = self.rollout_mode == "scheduled"
        variable_horizon_training = random_rollout or scheduled_rollout
        epoch_user = int(self.epoch + 1)
        scheduled_phase = self._get_scheduled_rollout_phase(epoch_user) if scheduled_rollout else None
        required_target_steps = (
            int(self.random_rollout_max_horizon)
            if random_rollout
            else int(self.scheduled_rollout_max_horizon)
            if scheduled_rollout
            else int(rollout_steps)
        )
        self._epoch_lr_values = []
        lr_start = self._current_lr()
        stage_info = {}
        if isinstance(self.scheduler, RolloutStageWarmupCosineScheduler):
            stage_info = self.scheduler.stage_for_epoch_info(self.epoch)
        start = time.time()
        last_batch_end = start
        data_time = 0.0
        forward_loss_time = 0.0
        backward_time = 0.0
        optimizer_time = 0.0
        last_loss = float("nan")
        last_loss_tensor: torch.Tensor | None = None
        total_loss_tensor: torch.Tensor | None = None
        total_grid_tensor: torch.Tensor | None = None
        total_spectral_raw_tensor: torch.Tensor | None = None
        total_spectral_weighted_tensor: torch.Tensor | None = None
        total_final_loss_tensor: torch.Tensor | None = None
        lead_loss_totals_tensor: torch.Tensor | None = None
        lead_grid_totals_tensor: torch.Tensor | None = None
        lead_spectral_totals_tensor: torch.Tensor | None = None
        lead_loss_counts = np.zeros((self.max_rollout_steps,), dtype=np.int64)
        sampled_horizons: list[int] = []
        horizon_loss_totals = np.zeros((self.max_rollout_steps + 1,), dtype=np.float64)
        horizon_final_totals = np.zeros((self.max_rollout_steps + 1,), dtype=np.float64)
        horizon_counts = np.zeros((self.max_rollout_steps + 1,), dtype=np.int64)
        log_every_batches = max(1, int(getattr(self, "log_every_batches", 1)))
        processed = 0
        optimizer_steps = 0
        # Wall-clock between consecutive optimizer steps. One optimizer step is the
        # unit that stays comparable across (batch 4, accum 3) and (batch 12, accum 1),
        # since both consume effective_batch_size samples. No explicit CUDA sync: in
        # steady state the loop can only issue steps as fast as the GPU retires them,
        # so the median of these deltas is the true per-step period.
        optimizer_step_times: list[float] = []
        step_window_start = time.time()
        planned_batches = len(self.train_data_loader)
        if self.max_train_batches is not None:
            planned_batches = min(planned_batches, int(self.max_train_batches))
        self.optimizer.zero_grad(set_to_none=True)
        for batch_idx, data in enumerate(self.train_data_loader):
            batch_received = time.time()
            data_time += batch_received - last_batch_end
            if self.max_train_batches is not None and batch_idx >= self.max_train_batches:
                break
            self.iters += 1
            processed += 1
            transfer_start = time.time()
            inp, target, metadata = self._to_device_batch(data)
            data_time += time.time() - transfer_start
            target_seq = self._target_sequence(target)
            if random_rollout:
                batch_rollout_steps = self._sample_random_rollout_steps()
            elif scheduled_rollout:
                if scheduled_phase is None:
                    raise ValueError(f"No scheduled rollout phase is active for epoch {epoch_user}.")
                batch_rollout_steps = self._scheduled_rollout_horizon_for_batch(scheduled_phase)
            else:
                batch_rollout_steps = int(rollout_steps)
            if self.rollout_mode == "fixed_full":
                if target_seq.shape[1] < int(rollout_steps):
                    raise AssertionError(
                        "fixed_full rollout requires target sequence length >= "
                        f"fixed_train_rollout_steps={rollout_steps}; got {target_seq.shape[1]}."
                    )
            elif target_seq.shape[1] < int(required_target_steps):
                raise AssertionError(
                    f"Training target length {target_seq.shape[1]} is shorter than required rollout S={required_target_steps}"
                )
            elif self.load_only_current_rollout and target_seq.shape[1] != int(rollout_steps):
                raise AssertionError(
                    f"Training target length {target_seq.shape[1]} does not match current rollout S={rollout_steps}"
                )
            forward_start = time.time()
            # The scope spans forward AND backward: with checkpoint_rollout_steps on,
            # each step is recomputed during backward and must see the same cache
            # state it saw on the way in. It closes before _optimizer_step().
            with self._edge_cache_scope(warm=False):
                with self._autocast_context():
                    loss, _, lead_losses, loss_components = self._rollout_loss(
                        inp,
                        target,
                        batch_rollout_steps,
                        metadata=metadata,
                        return_lead_losses=True,
                        return_loss_components=True,
                    )
                    backward_loss = loss / float(self.gradient_accumulation_steps)
                forward_loss_time += time.time() - forward_start
                backward_start = time.time()
                if self.gscaler is not None:
                    self.gscaler.scale(backward_loss).backward()
                else:
                    backward_loss.backward()
                backward_time += time.time() - backward_start
            should_step = (processed % self.gradient_accumulation_steps == 0) or (processed == planned_batches)
            if should_step:
                optimizer_start = time.time()
                self._optimizer_step()
                optimizer_time += time.time() - optimizer_start
                optimizer_steps += 1
                step_window_end = time.time()
                optimizer_step_times.append(step_window_end - step_window_start)
                step_window_start = step_window_end
            loss_detached = loss.detach().float()
            grid_detached = loss_components["grid_loss"].detach().float()
            spectral_raw_detached = loss_components["spectral_raw_loss"].detach().float()
            spectral_weighted_detached = loss_components["spectral_weighted_loss"].detach().float()
            final_loss_detached = lead_losses[-1].detach().float()
            last_loss_tensor = loss_detached
            total_loss_tensor = (
                loss_detached.clone()
                if total_loss_tensor is None
                else total_loss_tensor + loss_detached
            )
            total_grid_tensor = (
                grid_detached.clone()
                if total_grid_tensor is None
                else total_grid_tensor + grid_detached
            )
            total_spectral_raw_tensor = (
                spectral_raw_detached.clone()
                if total_spectral_raw_tensor is None
                else total_spectral_raw_tensor + spectral_raw_detached
            )
            total_spectral_weighted_tensor = (
                spectral_weighted_detached.clone()
                if total_spectral_weighted_tensor is None
                else total_spectral_weighted_tensor + spectral_weighted_detached
            )
            total_final_loss_tensor = (
                final_loss_detached.clone()
                if total_final_loss_tensor is None
                else total_final_loss_tensor + final_loss_detached
            )
            sampled_horizons.append(int(batch_rollout_steps))
            if 0 < int(batch_rollout_steps) <= self.max_rollout_steps:
                horizon_loss_totals[int(batch_rollout_steps)] += float(loss_detached.item())
                horizon_final_totals[int(batch_rollout_steps)] += float(final_loss_detached.item())
                horizon_counts[int(batch_rollout_steps)] += 1
            if lead_loss_totals_tensor is None:
                lead_loss_totals_tensor = torch.zeros(
                    (self.max_rollout_steps,),
                    device=loss_detached.device,
                    dtype=torch.float32,
                )
                lead_grid_totals_tensor = torch.zeros_like(lead_loss_totals_tensor)
                lead_spectral_totals_tensor = torch.zeros_like(lead_loss_totals_tensor)
            for lead_idx, lead_loss in enumerate(lead_losses[: self.max_rollout_steps]):
                lead_loss_totals_tensor[lead_idx] = (
                    lead_loss_totals_tensor[lead_idx] + lead_loss.detach().float()
                )
                if lead_grid_totals_tensor is not None:
                    lead_grid_totals_tensor[lead_idx] = (
                        lead_grid_totals_tensor[lead_idx]
                        + loss_components["grid_lead_losses"][lead_idx].detach().float()
                    )
                if lead_spectral_totals_tensor is not None:
                    lead_spectral_totals_tensor[lead_idx] = (
                        lead_spectral_totals_tensor[lead_idx]
                        + loss_components["spectral_raw_lead_losses"][lead_idx].detach().float()
                    )
                lead_loss_counts[lead_idx] += 1
            if variable_horizon_training:
                ratio = float(final_loss_detached.item()) / max(float(loss_detached.item()), 1.0e-12)
                payload = {
                    "train/loss": float(loss_detached.item()),
                }
                if random_rollout:
                    payload.update(
                        {
                            "train/random_rollout/sampled_horizon": int(batch_rollout_steps),
                            "train/random_rollout/loss_total": float(loss_detached.item()),
                            "train/random_rollout/loss_mean_all_steps": float(loss_detached.item()),
                            "train/random_rollout/loss_final_sampled_horizon": float(final_loss_detached.item()),
                            "train/random_rollout/final_to_mean_ratio": ratio,
                        }
                    )
                if scheduled_rollout and scheduled_phase is not None:
                    payload.update(
                        {
                            "train/scheduled_rollout/sampled_horizon": int(batch_rollout_steps),
                            "train/scheduled_rollout/loss_total": float(loss_detached.item()),
                            "train/scheduled_rollout/loss_mean_all_steps": float(loss_detached.item()),
                            "train/scheduled_rollout/loss_final_sampled_horizon": float(final_loss_detached.item()),
                            "train/scheduled_rollout/final_to_mean_ratio": ratio,
                            "train/rollout_phase/id": float(scheduled_phase.get("id", 0)),
                        }
                    )
                if getattr(self, "_last_grad_norm_pre_clip", None) is not None:
                    payload["train/grad_norm/global"] = float(self._last_grad_norm_pre_clip)
                    payload["train/grad_norm/pre_clip"] = float(self._last_grad_norm_pre_clip)
                    payload["train/grad_norm/post_clip"] = float(self._last_grad_norm_post_clip)
                self._wandb_log(payload, step=int(self.iters))
            if batch_idx % log_every_batches == 0:
                last_loss = float(loss_detached.item())
                logging.info(
                    "Epoch %d - Batch %d - rollout_phase=%s - train_actual_S %d - train_loaded_Smax %d - "
                    "train_loss_total %.6f train_loss_grid %.6f "
                    "train_loss_spectral_raw %.6f train_loss_spectral_weighted %.6f",
                    self.epoch + 1,
                    batch_idx,
                    str(scheduled_phase.get("name", self.rollout_mode)) if scheduled_phase is not None else self.rollout_mode,
                    int(batch_rollout_steps),
                    int(self._current_train_target_rollout_steps or rollout_steps),
                    last_loss,
                    float(grid_detached.item()),
                    float(spectral_raw_detached.item()),
                    float(spectral_weighted_detached.item()),
                )
            last_batch_end = time.time()
        if processed > 0 and total_loss_tensor is not None:
            avg_loss = float((total_loss_tensor / float(processed)).item())
            avg_grid = float((total_grid_tensor / float(processed)).item()) if total_grid_tensor is not None else avg_loss
            avg_spectral_raw = (
                float((total_spectral_raw_tensor / float(processed)).item())
                if total_spectral_raw_tensor is not None
                else 0.0
            )
            avg_spectral_weighted = (
                float((total_spectral_weighted_tensor / float(processed)).item())
                if total_spectral_weighted_tensor is not None
                else 0.0
            )
            avg_final_sampled = (
                float((total_final_loss_tensor / float(processed)).item())
                if total_final_loss_tensor is not None
                else float("nan")
            )
            last_loss = float(last_loss_tensor.item()) if last_loss_tensor is not None else float("nan")
            lead_loss_totals = lead_loss_totals_tensor.detach().cpu().numpy() if lead_loss_totals_tensor is not None else np.zeros((self.max_rollout_steps,), dtype=np.float32)
            lead_grid_totals = lead_grid_totals_tensor.detach().cpu().numpy() if lead_grid_totals_tensor is not None else np.zeros((self.max_rollout_steps,), dtype=np.float32)
            lead_spectral_totals = lead_spectral_totals_tensor.detach().cpu().numpy() if lead_spectral_totals_tensor is not None else np.zeros((self.max_rollout_steps,), dtype=np.float32)
        else:
            avg_loss = float("nan")
            avg_grid = float("nan")
            avg_spectral_raw = float("nan")
            avg_spectral_weighted = float("nan")
            avg_final_sampled = float("nan")
            lead_loss_totals = np.zeros((self.max_rollout_steps,), dtype=np.float32)
            lead_grid_totals = np.zeros((self.max_rollout_steps,), dtype=np.float32)
            lead_spectral_totals = np.zeros((self.max_rollout_steps,), dtype=np.float32)
        lead_loss_means = {
            f"train_loss_lead{idx + 1}": (
                float(lead_loss_totals[idx] / lead_loss_counts[idx])
                if lead_loss_counts[idx] > 0
                else float("nan")
            )
            for idx in range(self.max_rollout_steps)
        }
        lead_grid_loss_means = {
            f"train_grid_loss_lead{idx + 1}": (
                float(lead_grid_totals[idx] / lead_loss_counts[idx])
                if lead_loss_counts[idx] > 0
                else float("nan")
            )
            for idx in range(self.max_rollout_steps)
        }
        lead_spectral_loss_means = {
            f"train_spectral_loss_lead{idx + 1}": (
                float(lead_spectral_totals[idx] / lead_loss_counts[idx])
                if lead_loss_counts[idx] > 0
                else float("nan")
            )
            for idx in range(self.max_rollout_steps)
        }
        final_loss_key = f"train_loss_lead{int(rollout_steps)}"
        train_loss_final = avg_final_sampled if variable_horizon_training else float(lead_loss_means.get(final_loss_key, float("nan")))
        random_stats: dict[str, Any] = {}
        if variable_horizon_training:
            horizons_np = np.asarray(sampled_horizons, dtype=np.float64)
            random_stats.update(
                {
                    "train_random_sampled_horizon_mean": float(np.mean(horizons_np)) if horizons_np.size else float("nan"),
                    "train_random_sampled_horizon_min": float(np.min(horizons_np)) if horizons_np.size else float("nan"),
                    "train_random_sampled_horizon_max": float(np.max(horizons_np)) if horizons_np.size else float("nan"),
                    "train_random_sampled_horizon_std": float(np.std(horizons_np)) if horizons_np.size else float("nan"),
                    "train_random_loss_mean_all_steps": float(avg_loss),
                    "train_random_loss_total": float(avg_loss),
                    "train_random_loss_final_sampled_horizon": float(avg_final_sampled),
                    "train_random_final_to_mean_ratio": (
                        float(avg_final_sampled / avg_loss)
                        if np.isfinite(avg_final_sampled) and np.isfinite(avg_loss) and abs(avg_loss) > 1.0e-12
                        else float("nan")
                    ),
                }
            )
            if scheduled_rollout and scheduled_phase is not None:
                random_stats.update(
                    {
                        "train_scheduled_sampled_horizon_mean": random_stats["train_random_sampled_horizon_mean"],
                        "train_scheduled_sampled_horizon_min": random_stats["train_random_sampled_horizon_min"],
                        "train_scheduled_sampled_horizon_max": random_stats["train_random_sampled_horizon_max"],
                        "train_scheduled_sampled_horizon_std": random_stats["train_random_sampled_horizon_std"],
                        "train_scheduled_loss_mean_all_steps": float(avg_loss),
                        "train_scheduled_loss_total": float(avg_loss),
                        "train_scheduled_loss_final_sampled_horizon": float(avg_final_sampled),
                        "train_scheduled_final_to_mean_ratio": random_stats["train_random_final_to_mean_ratio"],
                        "train_rollout_phase_id": float(scheduled_phase.get("id", 0)),
                        "train_rollout_phase_name": str(scheduled_phase.get("name", "")),
                        "train_rollout_phase_mode": str(scheduled_phase.get("mode", "")),
                        "train_rollout_phase_start_epoch": float(scheduled_phase.get("start_epoch", float("nan"))),
                        "train_rollout_phase_end_epoch": (
                            float(scheduled_phase["end_epoch"])
                            if scheduled_phase.get("end_epoch") is not None
                            else float("nan")
                        ),
                        "train_rollout_phase_fixed_horizon": (
                            float(scheduled_phase.get("horizon"))
                            if scheduled_phase.get("mode") == "fixed"
                            else float("nan")
                        ),
                        "train_rollout_phase_random_min_horizon": (
                            float(scheduled_phase.get("min_horizon"))
                            if scheduled_phase.get("mode") == "random"
                            else float("nan")
                        ),
                        "train_rollout_phase_random_max_horizon": (
                            float(scheduled_phase.get("max_horizon"))
                            if scheduled_phase.get("mode") == "random"
                            else float("nan")
                        ),
                    }
                )
            for horizon in range(1, self.max_rollout_steps + 1):
                count = int(horizon_counts[horizon])
                random_stats[f"train_random_horizon_count_S{horizon}"] = float(count)
                random_stats[f"train_random_by_horizon_S{horizon}_loss_mean"] = (
                    float(horizon_loss_totals[horizon] / count) if count > 0 else float("nan")
                )
                random_stats[f"train_random_by_horizon_S{horizon}_loss_final"] = (
                    float(horizon_final_totals[horizon] / count) if count > 0 else float("nan")
                )
                random_stats[f"train_random_by_lead_lead{horizon}_count"] = float(lead_loss_counts[horizon - 1])
                if scheduled_rollout:
                    random_stats[f"train_scheduled_horizon_count_S{horizon}"] = random_stats[f"train_random_horizon_count_S{horizon}"]
                    random_stats[f"train_scheduled_by_horizon_S{horizon}_loss_mean"] = random_stats[
                        f"train_random_by_horizon_S{horizon}_loss_mean"
                    ]
                    random_stats[f"train_scheduled_by_horizon_S{horizon}_loss_final"] = random_stats[
                        f"train_random_by_horizon_S{horizon}_loss_final"
                    ]
            random_stats["_sampled_horizons"] = [float(x) for x in sampled_horizons]  # type: ignore[assignment]
        lr_values = self._epoch_lr_values or [lr_start]
        lr_end = lr_values[-1]
        lr_min = min(lr_values)
        lr_max = max(lr_values)
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
        if self.log_timing_breakdown:
            logging.info(
                "Train timing breakdown | data_seconds=%.2f | forward_loss_seconds=%.2f | "
                "backward_seconds=%.2f | optimizer_seconds=%.2f",
                data_time,
                forward_loss_time,
                backward_time,
                optimizer_time,
            )
        lead_summary = " ".join(
            f"lead{lead}={lead_loss_means[f'train_loss_lead{lead}']:.6f}"
            for lead in range(1, min(int(rollout_steps), self.max_rollout_steps) + 1)
        )
        logging.info(
            "Train lead losses | train_avg_total=%.6f | train_avg_grid=%.6f | "
            "train_avg_spectral_raw=%.6f | train_avg_spectral_weighted=%.6f | "
            "train_loss_final=%.6f | %s",
            avg_loss,
            avg_grid,
            avg_spectral_raw,
            avg_spectral_weighted,
            train_loss_final,
            lead_summary,
        )
        if variable_horizon_training:
            count_summary = " ".join(
                f"S{horizon}={int(random_stats.get(f'train_random_horizon_count_S{horizon}', 0.0))}"
                for horizon in range(1, self.max_rollout_steps + 1)
            )
            if scheduled_rollout and scheduled_phase is not None:
                logging.info(
                    "Scheduled rollout epoch %d | rollout_phase=%s | phase_mode=%s | sampled_S_mean=%.3f min=%d max=%d std=%.3f | "
                    "counts: %s | train_loaded_Smax=%d | train mean loss=%.6f | train sampled-final loss=%.6f",
                    self.epoch,
                    str(scheduled_phase.get("name")),
                    str(scheduled_phase.get("mode")),
                    float(random_stats.get("train_random_sampled_horizon_mean", float("nan"))),
                    int(random_stats.get("train_random_sampled_horizon_min", 0.0)),
                    int(random_stats.get("train_random_sampled_horizon_max", 0.0)),
                    float(random_stats.get("train_random_sampled_horizon_std", float("nan"))),
                    count_summary,
                    int(self._current_train_target_rollout_steps or rollout_steps),
                    avg_loss,
                    train_loss_final,
                )
            else:
                logging.info(
                    "Random rollout epoch %d | sampled horizon mean=%.3f min=%d max=%d std=%.3f | "
                    "counts: %s | train mean loss=%.6f | train sampled-final loss=%.6f",
                    self.epoch,
                    float(random_stats.get("train_random_sampled_horizon_mean", float("nan"))),
                    int(random_stats.get("train_random_sampled_horizon_min", 0.0)),
                    int(random_stats.get("train_random_sampled_horizon_max", 0.0)),
                    float(random_stats.get("train_random_sampled_horizon_std", float("nan"))),
                    count_summary,
                    avg_loss,
                    train_loss_final,
                )
        logs = {
            "loss": avg_loss,
            "last_loss": last_loss,
            "train_loss_total": avg_loss,
            "train_loss_grid": avg_grid,
            "train_loss_spectral_raw": avg_spectral_raw,
            "train_loss_spectral_weighted": avg_spectral_weighted,
            "train_avg_total": avg_loss,
            "train_avg_grid": avg_grid,
            "train_avg_spectral_raw": avg_spectral_raw,
            "train_avg_spectral_weighted": avg_spectral_weighted,
            "train_loss_mean": avg_loss,
            "train_loss_final": train_loss_final,
            "rollout_steps": float(rollout_steps),
            "optimizer_steps": float(optimizer_steps),
            "target_rollout_steps_loaded": float(self._current_train_target_rollout_steps or rollout_steps),
            "train_samples_per_second": samples_per_second,
            "train_batches": float(processed),
            "data_time_sec": float(data_time),
            "forward_loss_time_sec": float(forward_loss_time),
            "loss_time_sec": float(forward_loss_time),
            "backward_time_sec": float(backward_time),
            "optimizer_time_sec": float(optimizer_time),
            "median_step_time_sec": (
                float(sorted(optimizer_step_times)[len(optimizer_step_times) // 2])
                if optimizer_step_times
                else float("nan")
            ),
            "stage_index": float(stage_info.get("stage_index", float("nan"))),
            "stage_epoch": float(stage_info.get("stage_epoch", float("nan"))),
            "lr_start": float(lr_start),
            "lr_end": float(lr_end),
            "lr_min_this_epoch": float(lr_min),
            "lr_max_this_epoch": float(lr_max),
            **lead_loss_means,
            **lead_grid_loss_means,
            **lead_spectral_loss_means,
        }
        for lead in range(1, self.max_rollout_steps + 1):
            logs[f"train_loss_lead{lead}_count"] = float(lead_loss_counts[lead - 1])
        logs.update(random_stats)
        return elapsed, logs

    def _all_reduce_gradients(self) -> None:
        """Average gradients across ranks (manual data parallelism).

        The hot path calls ``model.forward_steps`` -- a bound method that
        torch.compile replaces -- so DDP's forward-hook machinery would never
        fire. At this model size (~5M params, ~20MB of grads) a single flat
        all-reduce after backward costs ~1-2 ms on NVLink, so overlap buys
        nothing and this stays wrapper-free: no state_dict prefixes, no
        find_unused_parameters, no compile interception concerns.
        """
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        if not grads:
            return
        flat = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(float(self.world_size))
        for grad, synced in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
            grad.copy_(synced)

    def _optimizer_step(self) -> None:
        if self.world_size > 1:
            self._all_reduce_gradients()
        diagnostics = getattr(self, "diagnostics_manager", None)
        diagnostics_enabled = bool(getattr(diagnostics, "enabled", False))
        self._last_grad_norm_pre_clip = None
        self._last_grad_norm_post_clip = None
        if self.gscaler is not None:
            self.gscaler.unscale_(self.optimizer)
            if diagnostics_enabled:
                diagnostics.observe_gradients(self.model, step=self.iters, epoch=self.epoch + 1, stage="pre_clip")
            if self.max_gradient_norm is not None:
                grad_norm = clip_grad_norm_(self.model.parameters(), self.max_gradient_norm)
                self._last_grad_norm_pre_clip = float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm)
                self._last_grad_norm_post_clip = float(min(self._last_grad_norm_pre_clip, float(self.max_gradient_norm)))
            if diagnostics_enabled:
                diagnostics.observe_gradients(self.model, step=self.iters, epoch=self.epoch + 1, stage="post_clip")
            self.gscaler.step(self.optimizer)
            self.gscaler.update()
        else:
            if diagnostics_enabled:
                diagnostics.observe_gradients(self.model, step=self.iters, epoch=self.epoch + 1, stage="pre_clip")
            if self.max_gradient_norm is not None:
                grad_norm = clip_grad_norm_(self.model.parameters(), self.max_gradient_norm)
                self._last_grad_norm_pre_clip = float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm)
                self._last_grad_norm_post_clip = float(min(self._last_grad_norm_pre_clip, float(self.max_gradient_norm)))
            if diagnostics_enabled:
                diagnostics.observe_gradients(self.model, step=self.iters, epoch=self.epoch + 1, stage="post_clip")
            self.optimizer.step()
        if self.lr_schedule_type == "rollout_stage_warmup_cosine" and isinstance(
            self.scheduler,
            RolloutStageWarmupCosineScheduler,
        ):
            row = self.scheduler.step()
            self._epoch_lr_values.append(float(row["lr"]))
            self._append_lr_schedule_row(row)
        self._ema_update()
        self.optimizer.zero_grad(set_to_none=True)

    def _ema_update(self) -> None:
        if not bool(getattr(self, "ema_enabled", False)):
            return
        with torch.no_grad():
            if getattr(self, "_ema_state", None) is None:
                self._ema_state = {
                    name: param.detach().clone().float()
                    for name, param in self._canonical_named_parameters()
                    if param.requires_grad
                }
                return
            decay = self.ema_decay
            for name, param in self._canonical_named_parameters():
                if not param.requires_grad:
                    continue
                shadow = self._ema_state.get(name)
                if shadow is None:
                    self._ema_state[name] = param.detach().clone().float()
                    continue
                shadow.mul_(decay).add_(param.detach().float(), alpha=1.0 - decay)

    def _ema_swap_in(self) -> None:
        """Load EMA weights into the model; live weights are parked until _ema_swap_out."""
        if (
            not bool(getattr(self, "ema_enabled", False))
            or getattr(self, "_ema_state", None) is None
            or getattr(self, "_ema_backup", None) is not None
        ):
            return
        with torch.no_grad():
            backup: dict[str, torch.Tensor] = {}
            for name, param in self._canonical_named_parameters():
                shadow = self._ema_state.get(name)
                if shadow is None:
                    continue
                backup[name] = param.detach().clone()
                param.copy_(shadow.to(dtype=param.dtype))
            self._ema_backup = backup

    def _ema_swap_out(self) -> None:
        if getattr(self, "_ema_backup", None) is None:
            return
        with torch.no_grad():
            for name, param in self._canonical_named_parameters():
                live = self._ema_backup.get(name)
                if live is not None:
                    param.copy_(live)
        self._ema_backup = None

    def _append_epoch_csv(
        self,
        epoch: int,
        train_logs: dict[str, float],
        valid_logs: dict[str, float],
        lr: float,
        epoch_time_sec: float,
    ) -> None:
        if self.world_rank != 0:
            return
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
            "train_loss_mean",
            "train_loss_final",
            "train_loss_total",
            "train_loss_grid",
            "train_loss_spectral_raw",
            "train_loss_spectral_weighted",
            "train_avg_total",
            "train_avg_grid",
            "train_avg_spectral_raw",
            "train_avg_spectral_weighted",
            "train_loss_lead1",
            "train_loss_lead2",
            "train_loss_lead3",
            "train_loss_lead4",
            "train_loss_lead5",
            "train_loss_lead6",
            "train_loss_lead7",
            "train_loss_lead8",
            "train_loss_lead9",
            "train_loss_lead10",
            "train_grid_loss_lead1",
            "train_grid_loss_lead2",
            "train_grid_loss_lead3",
            "train_grid_loss_lead4",
            "train_grid_loss_lead5",
            "train_grid_loss_lead6",
            "train_grid_loss_lead7",
            "train_grid_loss_lead8",
            "train_grid_loss_lead9",
            "train_grid_loss_lead10",
            "train_spectral_loss_lead1",
            "train_spectral_loss_lead2",
            "train_spectral_loss_lead3",
            "train_spectral_loss_lead4",
            "train_spectral_loss_lead5",
            "train_spectral_loss_lead6",
            "train_spectral_loss_lead7",
            "train_spectral_loss_lead8",
            "train_spectral_loss_lead9",
            "train_spectral_loss_lead10",
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
            "lr_start",
            "lr_end",
            "lr_min_this_epoch",
            "lr_max_this_epoch",
            "stage_index",
            "stage_epoch",
            "lr_schedule_type",
            "epoch_time_sec",
            "train_time_sec",
            "valid_time_sec",
            "diagnostics_time_sec",
            "median_step_time_sec",
            "data_time_sec",
            "forward_loss_time_sec",
            "loss_time_sec",
            "backward_time_sec",
            "optimizer_time_sec",
            "checkpoint_time_sec",
            "epoch_wall_time_sec",
            "train_samples_per_second",
            "valid_samples_per_second",
            "cuda_peak_allocated_gb",
            "cuda_peak_reserved_gb",
            "amp_dtype",
        ]

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
                "train_loss_mean": float(train_logs.get("train_loss_mean", train_logs["loss"])),
                "train_loss_final": float(train_logs.get("train_loss_final", train_logs.get("last_loss", float("nan")))),
                "train_loss_total": float(train_logs.get("train_loss_total", train_logs["loss"])),
                "train_loss_grid": float(train_logs.get("train_loss_grid", train_logs["loss"])),
                "train_loss_spectral_raw": float(train_logs.get("train_loss_spectral_raw", 0.0)),
                "train_loss_spectral_weighted": float(train_logs.get("train_loss_spectral_weighted", 0.0)),
                "train_avg_total": float(train_logs.get("train_avg_total", train_logs["loss"])),
                "train_avg_grid": float(train_logs.get("train_avg_grid", train_logs["loss"])),
                "train_avg_spectral_raw": float(train_logs.get("train_avg_spectral_raw", 0.0)),
                "train_avg_spectral_weighted": float(train_logs.get("train_avg_spectral_weighted", 0.0)),
                "valid_loss": float(valid_logs["valid_loss"]),
                "valid_final": float(valid_logs.get("valid_final_loss", float("nan"))),
                "persistence_S10": valid_logs.get("persistence_S10"),
                "skill_S10": valid_logs.get("skill_S10"),
                "lr": float(lr),
                "lr_start": float(train_logs.get("lr_start", lr)),
                "lr_end": float(train_logs.get("lr_end", lr)),
                "lr_min_this_epoch": float(train_logs.get("lr_min_this_epoch", lr)),
                "lr_max_this_epoch": float(train_logs.get("lr_max_this_epoch", lr)),
                "stage_index": train_logs.get("stage_index"),
                "stage_epoch": train_logs.get("stage_epoch"),
                "lr_schedule_type": self.lr_schedule_type,
                "epoch_time_sec": float(epoch_time_sec),
                "train_time_sec": float(train_logs.get("train_time_sec", 0.0)),
                "valid_time_sec": float(valid_logs.get("valid_time_sec", 0.0)),
                "diagnostics_time_sec": float(self._last_epoch_metadata.get("diagnostics_time_sec", 0.0)),
                "median_step_time_sec": float(train_logs.get("median_step_time_sec", float("nan"))),
                "data_time_sec": float(train_logs.get("data_time_sec", 0.0)),
                "forward_loss_time_sec": float(train_logs.get("forward_loss_time_sec", 0.0)),
                "loss_time_sec": float(train_logs.get("loss_time_sec", train_logs.get("forward_loss_time_sec", 0.0))),
                "backward_time_sec": float(train_logs.get("backward_time_sec", 0.0)),
                "optimizer_time_sec": float(train_logs.get("optimizer_time_sec", 0.0)),
                "checkpoint_time_sec": float(self._last_epoch_metadata.get("checkpoint_time_sec", 0.0)),
                "epoch_wall_time_sec": float(self._last_epoch_metadata.get("epoch_wall_time_sec", epoch_time_sec)),
                "train_samples_per_second": float(train_logs.get("train_samples_per_second", 0.0)),
                "valid_samples_per_second": float(valid_logs.get("valid_samples_per_second", 0.0)),
                "cuda_peak_allocated_gb": self._last_epoch_metadata.get("cuda_peak_allocated_gb"),
                "cuda_peak_reserved_gb": self._last_epoch_metadata.get("cuda_peak_reserved_gb"),
                "amp_dtype": self.resolved_amp_dtype_name,
            }
            for horizon in (1, 2, 4, 6, 8, 10):
                row[f"valid_S{horizon}"] = valid_logs.get(f"valid_S{horizon}")
                row[f"valid_S{horizon}_final"] = valid_logs.get(f"valid_S{horizon}_final")
            for lead in range(1, 11):
                row[f"train_loss_lead{lead}"] = train_logs.get(f"train_loss_lead{lead}")
                row[f"train_grid_loss_lead{lead}"] = train_logs.get(f"train_grid_loss_lead{lead}")
                row[f"train_spectral_loss_lead{lead}"] = train_logs.get(f"train_spectral_loss_lead{lead}")
            writer.writerow(
                row
            )

    def _append_training_metrics_csv(
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
        path = os.path.join(experiment_dir, "training_metrics.csv")
        exists = os.path.exists(path)
        fields = [
            "epoch",
            "train_rollout_steps",
            "train_loss_total",
            "train_loss_grid",
            "train_loss_spectral_raw",
            "train_loss_spectral_weighted",
            "train_loss_mean",
            "train_loss_final",
            "train_loss_lead1",
            "train_loss_lead2",
            "train_loss_lead3",
            "train_loss_lead4",
            "train_loss_lead5",
            "train_loss_lead6",
            "train_loss_lead7",
            "train_loss_lead8",
            "train_loss_lead9",
            "train_loss_lead10",
            "valid_S1",
            "valid_S2",
            "valid_S4",
            "valid_S6",
            "valid_S8",
            "valid_S10",
            "valid_S10_final",
            "lr",
            "epoch_time",
            "peak_gpu_memory",
        ]

        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                writer.writeheader()
            row = {
                "epoch": int(epoch),
                "train_rollout_steps": int(train_logs["rollout_steps"]),
                "train_loss_total": float(train_logs.get("train_loss_total", train_logs["loss"])),
                "train_loss_grid": float(train_logs.get("train_loss_grid", train_logs["loss"])),
                "train_loss_spectral_raw": float(train_logs.get("train_loss_spectral_raw", 0.0)),
                "train_loss_spectral_weighted": float(train_logs.get("train_loss_spectral_weighted", 0.0)),
                "train_loss_mean": float(train_logs.get("train_loss_mean", train_logs["loss"])),
                "train_loss_final": float(train_logs.get("train_loss_final", train_logs.get("last_loss", float("nan")))),
                "valid_S1": valid_logs.get("valid_S1"),
                "valid_S2": valid_logs.get("valid_S2"),
                "valid_S4": valid_logs.get("valid_S4"),
                "valid_S6": valid_logs.get("valid_S6"),
                "valid_S8": valid_logs.get("valid_S8"),
                "valid_S10": valid_logs.get("valid_S10"),
                "valid_S10_final": valid_logs.get("valid_S10_final"),
                "lr": float(lr),
                "epoch_time": float(epoch_time_sec),
                "peak_gpu_memory": self._last_epoch_metadata.get("cuda_peak_allocated_gb"),
            }
            for lead in range(1, 11):
                row[f"train_loss_lead{lead}"] = train_logs.get(f"train_loss_lead{lead}")
            writer.writerow(row)

    @staticmethod
    def _first_finite_value(*values: Any) -> float:
        for value in values:
            result = _finite_or_nan(value)
            if np.isfinite(result):
                return float(result)
        return float("nan")

    def _s1_overfit_values(
        self,
        train_logs: dict[str, Any],
        valid_logs: dict[str, Any],
    ) -> tuple[float, float, float]:
        train_s1 = self._first_finite_value(
            train_logs.get("train_loss_lead1"),
            train_logs.get("train_loss_final"),
            train_logs.get("train_loss_mean"),
            train_logs.get("loss"),
        )
        valid_s1 = self._first_finite_value(
            valid_logs.get("valid_S1_final"),
            valid_logs.get("valid_S1"),
            valid_logs.get("valid_final_loss"),
            valid_logs.get("valid_loss"),
        )
        gap = valid_s1 - train_s1 if np.isfinite(train_s1) and np.isfinite(valid_s1) else float("nan")
        return train_s1, valid_s1, float(gap)

    def _append_s1_overfit_curve_csv(
        self,
        epoch: int,
        train_logs: dict[str, Any],
        valid_logs: dict[str, Any],
        lr: float,
    ) -> None:
        experiment_dir = str(_get(self.params, "experiment_dir", ""))
        if not experiment_dir:
            return
        path = os.path.join(experiment_dir, "s1_overfit_curve.csv")
        exists = os.path.exists(path)
        fields = ["epoch", "train_S1_loss", "valid_S1_loss", "overfit_gap_S1", "lr"]
        train_s1, valid_s1, gap = self._s1_overfit_values(train_logs, valid_logs)

        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow(
                {
                    "epoch": int(epoch),
                    "train_S1_loss": float(train_s1),
                    "valid_S1_loss": float(valid_s1),
                    "overfit_gap_S1": float(gap),
                    "lr": float(lr),
                }
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
            inp, target, metadata = self._to_device_batch(data)
            with self._autocast_context(), self._edge_cache_scope(warm=False):
                loss, _ = self._rollout_loss(
                    inp,
                    target,
                    rollout_steps,
                    metadata=metadata,
                    include_spectral_loss=False,
                )
            total += float(loss.item())
            target_seq = self._target_sequence(target)
            final_step = min(rollout_steps, target_seq.shape[1]) - 1
            previous, current = self.model.adapter.extract_two_steps(inp)
            initial_state = current
            final_pred = None
            self._log_validation_lead_conditioning_debug_once(final_step + 1)
            with self._autocast_context(), self._edge_cache_scope():
                for step in range(final_step + 1):
                    lead = step + 1
                    target_seq = self._target_sequence(target)
                    aux = self._build_aux_for_step(current, target_seq, metadata, step)
                    final_pred = self._forward_model_step(previous, current, aux, lead=lead)
                    final_pred = getattr(self, "feature_builder", None).apply_overrides(
                        final_pred,
                        current=current,
                        target_norm=target_seq[:, step],
                    ) if getattr(self, "feature_builder", None) is not None else final_pred
                    target_handler = getattr(self, "target_handler", None)
                    if target_handler is not None:
                        final_pred = target_handler.apply(
                            pred_next=final_pred,
                            current_state=current,
                            initial_state=initial_state,
                            target_sequence=target_seq,
                            lead=lead,
                        )
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
            inp, target, metadata = self._to_device_batch(data)
            target_seq = self._target_sequence(target)
            if target_seq.shape[1] < max_eval_steps:
                raise AssertionError(
                    f"Validation target length {target_seq.shape[1]} is shorter than max eval S={max_eval_steps}"
                )
            bsz = int(inp.shape[0])
            previous, current = self.model.adapter.extract_two_steps(inp)
            initial_state = current
            persistence_pred = current[:, : self.model.output_channels]
            self._log_validation_lead_conditioning_debug_once(max_eval_steps)
            with self._autocast_context(), self._edge_cache_scope():
                for step in range(max_eval_steps):
                    lead = step + 1
                    gt = target_seq[:, step]
                    aux = self._build_aux_for_step(current, target_seq, metadata, step)
                    pred = self._forward_model_step(previous, current, aux, lead=lead)
                    feature_builder = getattr(self, "feature_builder", None)
                    if feature_builder is not None:
                        pred = feature_builder.apply_overrides(pred, current=current, target_norm=gt)
                    target_handler = getattr(self, "target_handler", None)
                    if target_handler is not None:
                        pred = target_handler.apply(
                            pred_next=pred,
                            current_state=current,
                            initial_state=initial_state,
                            target_sequence=target_seq,
                            lead=lead,
                        )
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
        rollout_curve_rows: list[dict[str, Any]] = []
        for horizon in eval_steps:
            prefix = step_losses[:horizon]
            persistence_prefix = persistence_losses[:horizon]
            logs[f"valid_S{horizon}"] = float(np.mean(prefix))
            logs[f"valid_S{horizon}_final"] = float(step_losses[horizon - 1])
            logs[f"valid_S{horizon}_final_to_mean_ratio"] = (
                float(logs[f"valid_S{horizon}_final"] / logs[f"valid_S{horizon}"])
                if abs(float(logs[f"valid_S{horizon}"])) > 1.0e-12
                else float("nan")
            )
            logs[f"persistence_S{horizon}"] = float(np.mean(persistence_prefix))
            logs[f"persistence_S{horizon}_final"] = float(persistence_losses[horizon - 1])
            denom_persistence = logs[f"persistence_S{horizon}"]
            logs[f"skill_S{horizon}"] = (
                float(1.0 - logs[f"valid_S{horizon}"] / denom_persistence)
                if np.isfinite(denom_persistence) and abs(denom_persistence) > 1.0e-12
                else float("nan")
            )
            for step in range(1, horizon + 1):
                rollout_curve_rows.append(
                    {
                        "epoch": int(getattr(self, "epoch", 0) + 1),
                        "horizon": int(horizon),
                        "step": int(step),
                        "loss": float(step_losses[step - 1]),
                    }
                )

        logs["valid_loss"] = float(logs[f"valid_S{valid_rollout_steps}"])
        logs["valid_final_loss"] = float(logs[f"valid_S{valid_rollout_steps}_final"])
        logs["valid_final"] = logs["valid_final_loss"]
        logs["_valid_rollout_curve_rows"] = rollout_curve_rows  # type: ignore[assignment]
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
            logs[f"{key}_final_to_mean_ratio"] = (
                float(logs[f"{key}_final"] / logs[key])
                if abs(float(logs[key])) > 1.0e-12
                else float("nan")
            )

        return time.time() - start, logs

    def save_checkpoint(self, checkpoint_path: str, metadata: dict[str, Any] | None = None) -> None:
        if self.world_rank != 0:
            return
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        metadata = dict(self._last_epoch_metadata if metadata is None else metadata)
        metadata.update(self.graph_topology_metadata)
        metadata.update(
            architecture_metadata(
                self.params,
                graph_metadata=self.graph.metadata,
                num_parameters=self.num_parameters,
            )
        )
        metadata["graph_path"] = str(_get(self.params, "graph_path", ""))
        target_handler = getattr(self, "target_handler", None)
        if target_handler is not None:
            metadata.update(target_handler.checkpoint_metadata())
        metadata.update(self._loss_channel_metadata())
        metadata.update(self._training_rollout_metadata())
        metadata.update(self._fusion_gate_metadata())
        metadata.update(self._pooling_gate_metadata())
        metadata.update(self._lead_conditioning_metadata())
        metadata.update(self._spectral_loss_checkpoint_metadata())
        metadata.setdefault("lr_schedule_type", self.lr_schedule_type)
        metadata["rollout_stage_lr_schedule"] = _get(self.params, "rollout_stage_lr_schedule", None)
        metadata["current_epoch"] = int(self.epoch)
        metadata["global_optimizer_step"] = (
            int(self.scheduler.global_step)
            if isinstance(self.scheduler, RolloutStageWarmupCosineScheduler)
            else None
        )
        metadata["scheduler_state_dict"] = self.scheduler.state_dict() if self.scheduler is not None else None
        payload = {
            "iters": self.iters,
            "epoch": self.epoch,
            "model_state": self._canonical_model_state(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "lr_schedule_type": self.lr_schedule_type,
            "rollout_stage_lr_schedule": _get(self.params, "rollout_stage_lr_schedule", None),
            "current_epoch": self.epoch,
            "global_optimizer_step": int(getattr(self.scheduler, "global_step", 0))
            if isinstance(self.scheduler, RolloutStageWarmupCosineScheduler)
            else None,
            "params": dict(getattr(self.params, "params", {})),
            "metadata": metadata,
            "best_score_global": self.best_score_global,
            "best_score_by_stage": dict(self.best_score_by_stage),
        }
        if bool(getattr(self, "ema_enabled", False)) and getattr(self, "_ema_state", None) is not None:
            payload["ema"] = {"enabled": True, "decay": float(self.ema_decay)}
            if self._ema_backup is not None:
                # model_state currently holds the EMA weights (saved inside the swap
                # window); keep the live training weights so resume stays exact.
                payload["ema_live_model_state"] = {
                    name: tensor.detach().cpu() for name, tensor in self._ema_backup.items()
                }
            else:
                payload["ema_state"] = {
                    name: tensor.detach().cpu() for name, tensor in self._ema_state.items()
                }
        torch.save(payload, checkpoint_path)

    def _validate_checkpoint_training_rollout(self, checkpoint_metadata: dict[str, Any]) -> None:
        checkpoint_mode = str(
            checkpoint_metadata.get(
                "training_rollout_mode",
                checkpoint_metadata.get("rollout_mode", "curriculum"),
            )
        ).strip().lower()
        current_mode = getattr(self, "rollout_mode", "curriculum")
        if checkpoint_mode != current_mode:
            raise RuntimeError(
                "Checkpoint training mode mismatch: "
                f"checkpoint was trained with rollout_mode={checkpoint_mode}, "
                f"current config uses rollout_mode={current_mode}. "
                "Resume is unsafe. Train from scratch or use explicit partial initialization."
            )
        if current_mode == "fixed_full":
            checkpoint_steps = int(checkpoint_metadata.get("fixed_train_rollout_steps", -1))
            current_steps = int(
                getattr(
                    self,
                    "fixed_train_rollout_steps",
                    getattr(self, "max_rollout_steps", checkpoint_steps),
                )
            )
            if checkpoint_steps != current_steps:
                raise RuntimeError(
                    "Checkpoint training mode mismatch: "
                    f"checkpoint fixed_train_rollout_steps={checkpoint_steps}, "
                    f"current fixed_train_rollout_steps={current_steps}. "
                    "Resume is unsafe. Train from scratch or use explicit partial initialization."
                )

    def restore_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        checkpoint_metadata = dict(checkpoint.get("metadata", {}))
        self._validate_checkpoint_resolution(checkpoint_metadata)
        validate_checkpoint_graph_mode(checkpoint_metadata, self.params)
        try:
            validate_checkpoint_architecture(checkpoint_metadata, self.params)
            self._validate_checkpoint_training_rollout(checkpoint_metadata)
            self._validate_checkpoint_target_handling(checkpoint_metadata)
        except RuntimeError as exc:
            if bool(_get(self.params, "init_from_checkpoint_allow_partial", False)):
                self._validate_checkpoint_feature_metadata(checkpoint_metadata)
                self._load_model_state_partial(
                    checkpoint["model_state"],
                    strict_delta_stats=self.use_delta_normalization,
                )
                self.iters = 0
                self.start_epoch = 0
                self.epoch = 0
                logging.warning("%s Loaded matching tensors only because init_from_checkpoint_allow_partial=True.", exc)
                return
            raise
        if checkpoint_metadata.get("resolution_mode") is None and str(_get(self.params, "resolution_mode", "5p625")) == "5p625":
            checkpoint_metadata["resolution_mode"] = "5p625"
        checkpoint_graph = graph_topology_metadata(checkpoint_metadata)
        current_graph = graph_topology_metadata(self.graph_topology_metadata)
        if checkpoint_graph != current_graph:
            mismatch_details = [
                f"{key}: checkpoint={checkpoint_graph.get(key)!r}, current={current_graph.get(key)!r}"
                for key in sorted(set(checkpoint_graph) | set(current_graph))
                if checkpoint_graph.get(key) != current_graph.get(key)
            ]
            message = (
                "Checkpoint graph mismatch: "
                + "; ".join(mismatch_details)
                + ". Train from scratch or use explicit partial initialization."
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
        self._validate_checkpoint_feature_metadata(checkpoint_metadata)
        self._load_model_state(checkpoint["model_state"], strict_delta_stats=self.use_delta_normalization)
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if bool(getattr(self, "ema_enabled", False)):
            param_names = {
                name for name, param in self._canonical_named_parameters() if param.requires_grad
            }
            ema_live_state = checkpoint.get("ema_live_model_state")
            ema_state = checkpoint.get("ema_state")
            if ema_live_state is not None:
                # model_state held the EMA weights: adopt them as the shadow and
                # restore the live training weights into the model.
                self._ema_state = {
                    name: tensor.detach().clone().float().to(self.device)
                    for name, tensor in checkpoint["model_state"].items()
                    if name in param_names
                }
                self.model.load_state_dict(dict(ema_live_state), strict=False)
                logging.info("Restored EMA shadow (from model_state) and live training weights (from ema_live_model_state).")
            elif ema_state is not None:
                self._ema_state = {
                    name: tensor.detach().clone().float().to(self.device)
                    for name, tensor in ema_state.items()
                    if name in param_names
                }
                logging.info("Restored EMA shadow from checkpoint ema_state.")
            else:
                logging.info("EMA enabled but checkpoint has no EMA state; shadow will re-initialize from current weights.")
        self.iters = int(checkpoint.get("iters", 0))
        self.start_epoch = int(checkpoint.get("epoch", 0))
        self.epoch = self.start_epoch
        scheduler_state = checkpoint.get("scheduler_state_dict", checkpoint_metadata.get("scheduler_state_dict", None))
        if self.scheduler is not None and scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)
        elif isinstance(self.scheduler, RolloutStageWarmupCosineScheduler):
            reconstructed_step = checkpoint_metadata.get("global_optimizer_step", checkpoint.get("global_optimizer_step", None))
            if reconstructed_step is None:
                reconstructed_step = int(self.epoch) * self._planned_optimizer_steps_per_epoch()
            self.scheduler.load_state_dict({"global_step": int(reconstructed_step)})
            logging.warning(
                "Checkpoint lacks rollout_stage_warmup_cosine scheduler state; reconstructed global_optimizer_step=%d.",
                int(reconstructed_step),
            )
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

    def initialize_from_checkpoint(self, checkpoint_path: str) -> None:
        """Warm-start model weights from another checkpoint WITHOUT resuming its
        training state.

        Unlike :meth:`restore_checkpoint`, the optimizer, LR scheduler, epoch
        counter, and best-score trackers all start fresh, so the new run may use a
        different training regime (for example a multi-step rollout curriculum)
        than the checkpoint was trained under. This is the supported way to
        continue training a finished single-step run through a rollout curriculum.

        The model architecture, graph topology, resolution, delta-normalization,
        and feature metadata must still match the checkpoint; the training
        rollout mode/schedule is intentionally *not* validated, and a
        target-handling mismatch only warns (target handling is a runtime
        override/loss-mask policy with no weights attached, and warm-starts
        exist precisely to change the training objective). Set
        ``init_from_checkpoint_strict: false`` (or the legacy
        ``init_from_checkpoint_allow_partial: true``) to instead load only the
        tensors whose names and shapes match, leaving the rest at their freshly
        initialized values (a warning lists what was skipped).
        """
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"init_from_checkpoint path not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if "model_state" not in checkpoint:
            raise RuntimeError(f"init_from_checkpoint file {checkpoint_path} has no 'model_state'.")
        checkpoint_metadata = dict(checkpoint.get("metadata", {}))
        self._validate_checkpoint_resolution(checkpoint_metadata)
        validate_checkpoint_graph_mode(checkpoint_metadata, self.params)
        allow_partial = bool(_get(self.params, "init_from_checkpoint_allow_partial", False))
        strict = bool(_get(self.params, "init_from_checkpoint_strict", True)) and not allow_partial
        if strict:
            validate_checkpoint_architecture(checkpoint_metadata, self.params)
            try:
                self._validate_checkpoint_target_handling(checkpoint_metadata)
            except RuntimeError as exc:
                logging.warning(
                    "%s Proceeding with the warm-start anyway: target handling applies to "
                    "the NEW run only (state overrides + loss mask) and does not affect "
                    "weight compatibility.",
                    exc,
                )
            self._validate_checkpoint_feature_metadata(checkpoint_metadata)
            checkpoint_graph = graph_topology_metadata(checkpoint_metadata)
            current_graph = graph_topology_metadata(self.graph_topology_metadata)
            if checkpoint_graph != current_graph:
                mismatch_details = [
                    f"{key}: checkpoint={checkpoint_graph.get(key)!r}, current={current_graph.get(key)!r}"
                    for key in sorted(set(checkpoint_graph) | set(current_graph))
                    if checkpoint_graph.get(key) != current_graph.get(key)
                ]
                raise RuntimeError(
                    "init_from_checkpoint graph mismatch: "
                    + "; ".join(mismatch_details)
                    + ". Use a matching graph or set init_from_checkpoint_strict=false."
                )
            if self.use_delta_normalization and not bool(checkpoint_metadata.get("use_delta_normalization", False)):
                raise RuntimeError(
                    "use_delta_normalization=true but the init_from_checkpoint source lacks "
                    "delta-normalization metadata. Disable delta normalization or choose a "
                    "delta-normalized checkpoint."
                )
            self._load_model_state(checkpoint["model_state"], strict_delta_stats=self.use_delta_normalization)
        else:
            self._validate_checkpoint_feature_metadata(checkpoint_metadata)
            self._load_model_state_partial(checkpoint["model_state"], strict_delta_stats=self.use_delta_normalization)
        self.iters = 0
        self.start_epoch = 0
        self.epoch = 0
        logging.info(
            "Warm-started model weights from %s (init_from_checkpoint: fresh optimizer/scheduler/epoch; strict=%s).",
            checkpoint_path,
            strict,
        )

    def _validate_checkpoint_feature_metadata(self, metadata: dict[str, Any]) -> None:
        feature_builder = getattr(self, "feature_builder", None)
        if feature_builder is None:
            return
        active = feature_builder.checkpoint_metadata(
            base_input_channels=int(_get(self.params, "base_input_channels", _get(self.params, "N_in_channels", 0))),
            total_input_channels=int(_get(self.params, "total_input_channels", _get(self.params, "N_in_channels", 0))),
        )
        ok, reason = feature_metadata_matches(active, metadata)
        if not ok:
            raise RuntimeError(f"Checkpoint extra-feature configuration mismatch: {reason}")

    def _validate_checkpoint_target_handling(self, metadata: dict[str, Any]) -> None:
        target_handler = getattr(self, "target_handler", None)
        if target_handler is None:
            return
        ok, reason = target_handling_metadata_matches(target_handler.metadata, metadata)
        if not ok:
            raise RuntimeError(f"Checkpoint target-handling configuration mismatch: {reason}")

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
