from __future__ import annotations

import logging
import os
import random
import time
from typing import Any

import numpy as np
import torch
import torch.amp as amp
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from .data import DataConfig, build_data_loader
from .graph_builder import build_graph_bundle, lat_lon_from_netcdf, save_graph
from .graph_bundle import load_graph_bundle
from .losses import LatitudeWeightedMSE, graph_gradient_loss
from .models import GraphWeatherModel


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _get(params: Any, name: str, default: Any) -> Any:
    return getattr(params, name, default)


class Trainer:
    def __init__(self, params: Any, world_rank: int = 0, local_rank: int = 0):
        self.params = params
        self.world_rank = int(world_rank)
        self.local_rank = int(local_rank)
        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        self.max_gradient_norm = float(_get(params, "max_gradient_norm", 1.0))
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
        self.lr_schedule_type = str(_get(params, "lr_schedule_type", "none")).lower()
        self.warmup_epochs = int(_get(params, "warmup_epochs", 0))
        self._last_epoch_metadata: dict[str, Any] = {}

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
            rollout_steps=self.max_rollout_steps,
            batch_size=int(params.batch_size),
            num_workers=int(_get(params, "num_data_workers", 0)),
        )
        self.data_cfg = data_cfg
        self.train_data_loader, self.train_dataset = build_data_loader(data_cfg, params.train_data_path, train=True)
        self.valid_data_loader, self.valid_dataset = build_data_loader(data_cfg, params.valid_data_path, train=False)

        params.img_shape_x = self.train_dataset.img_shape_x
        params.img_shape_y = self.train_dataset.img_shape_y
        params.crop_size_x = data_cfg.crop_size_x or self.train_dataset.img_shape_x
        params.crop_size_y = data_cfg.crop_size_y or self.train_dataset.img_shape_y

        self._ensure_graph()
        self.graph = load_graph_bundle(params.graph_path, map_location="cpu").to(self.device)

        sample_inp, sample_tar = next(iter(self.train_data_loader))
        out_chans = sample_tar.shape[2] if sample_tar.dim() == 5 else sample_tar.shape[1]
        params.N_in_channels = int(sample_inp.shape[1])
        params.N_out_channels = int(out_chans)

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
        ).to(self.device)
        self._apply_trainable_scope(str(_get(params, "trainable_scope", "all")))

        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=float(params.lr),
            weight_decay=float(_get(params, "weight_decay", 1.0e-4)),
        )
        self.enable_amp = bool(_get(params, "enable_amp", False)) and self.device.type == "cuda"
        self.gscaler = amp.GradScaler("cuda") if self.enable_amp else None

        l0_lat = self.graph.L0.lat_lon[:, 0].reshape(self.graph.L0.height, self.graph.L0.width)[:, 0]
        self.loss_obj = LatitudeWeightedMSE(l0_lat).to(self.device)
        self.graph_gradient_weight = float(_get(params, "graph_gradient_weight", 0.0))

        scheduler_name = str(_get(params, "scheduler", "CosineAnnealingLR"))
        if self.lr_schedule_type == "rollout_stage":
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

        if bool(_get(params, "resume", True)) and os.path.isfile(params.checkpoint_path):
            self.restore_checkpoint(params.checkpoint_path)

        logging.info("Number of trainable model parameters: %d", self.count_parameters())

    def _optional_positive_int(self, name: str) -> int | None:
        value = _get(self.params, name, None)
        if value is None:
            return None
        value = int(value)
        return value if value > 0 else None

    def _ensure_graph(self) -> None:
        graph_path = self.params.graph_path
        if os.path.isfile(graph_path):
            return
        if not bool(_get(self.params, "auto_build_graph", True)):
            raise FileNotFoundError(f"Graph bundle not found: {graph_path}")
        latitudes, longitudes = lat_lon_from_netcdf(self.params.train_data_path)
        bundle = build_graph_bundle(
            latitudes,
            longitudes,
            k=int(_get(self.params, "k_neighbors", 8)),
            resolution=float(_get(self.params, "resolution", 5.625)),
        )
        save_graph(bundle, graph_path)
        logging.info("Built graph bundle at %s", graph_path)

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

    def _lr_for_epoch(self, rollout_steps: int) -> float:
        if self.lr_schedule_type != "rollout_stage":
            return self._current_lr()

        stage_lr = self._get_stage_lr(rollout_steps)
        if self.warmup_epochs <= 0 or self.epoch >= self.warmup_epochs:
            return stage_lr

        warmup_start_factor = float(_get(self.params, "warmup_start_factor", 0.1))
        start_lr = stage_lr * warmup_start_factor
        progress = float(self.epoch + 1) / float(max(1, self.warmup_epochs))
        return start_lr + (stage_lr - start_lr) * min(1.0, progress)

    def _target_sequence(self, target: torch.Tensor) -> torch.Tensor:
        return target if target.dim() == 5 else target.unsqueeze(1)

    def _rollout_loss(self, inp: torch.Tensor, target: torch.Tensor, rollout_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        target_seq = self._target_sequence(target)
        rollout_steps = min(int(rollout_steps), target_seq.shape[1])
        previous, current = self.model.adapter.extract_two_steps(inp)
        total = torch.zeros((), device=inp.device, dtype=inp.dtype)
        last_pred = None
        for step in range(rollout_steps):
            pred = self.model.forward_steps(previous, current)
            gt = target_seq[:, step]
            loss = self.loss_obj(pred, gt)
            if self.graph_gradient_weight > 0.0:
                loss = loss + self.graph_gradient_weight * graph_gradient_loss(
                    pred,
                    gt,
                    self.graph.L0.edge_index,
                )
            total = total + loss
            next_step = current.clone()
            next_step[:, : self.model.output_channels] = pred
            previous, current = current, next_step
            last_pred = pred
        return total / rollout_steps, last_pred

    def train(self) -> None:
        best_valid_loss = float("inf")
        for epoch in range(self.start_epoch, int(self.params.max_epochs)):
            start = time.time()
            train_rollout_steps = self._rollout_steps_for_epoch()
            epoch_lr = self._lr_for_epoch(train_rollout_steps)
            if self.lr_schedule_type == "rollout_stage":
                self._set_optimizer_lr(epoch_lr)

            tr_time, train_logs = self.train_one_epoch()
            train_rollout_steps = int(train_logs["rollout_steps"])

            if bool(_get(self.params, "validate_with_train_rollout", True)):
                valid_rollout_steps = train_rollout_steps
            else:
                valid_rollout_steps = None
            valid_time, valid_logs = self.validate_one_epoch(rollout_steps=valid_rollout_steps)

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
            self._last_epoch_metadata = {
                "epoch": self.epoch,
                "train_loss": train_logs["loss"],
                "valid_loss": valid_logs["valid_loss"],
                "train_rollout_steps": train_rollout_steps,
                "valid_rollout_steps": int(valid_logs["valid_rollout_steps"]),
                "lr": epoch_lr,
                "valid_multi_horizon": valid_multi,
            }

            if bool(_get(self.params, "save_checkpoint", True)):
                self.save_checkpoint(self.params.checkpoint_path)
                if valid_logs["valid_loss"] <= best_valid_loss:
                    self.save_checkpoint(self.params.best_checkpoint_path)
                    best_valid_loss = valid_logs["valid_loss"]
            extra_valid = " ".join(
                f"{key} {value:.6f}"
                for key, value in sorted(valid_logs.items())
                if key.startswith("valid_S")
            )
            if extra_valid:
                extra_valid = " | " + extra_valid
            logging.info(
                "Epoch %d finished in %.2f sec | train %.6f S=%d | valid %.6f S=%d%s | lr %.2e",
                epoch + 1,
                time.time() - start,
                train_logs["loss"],
                train_rollout_steps,
                valid_logs["valid_loss"],
                int(valid_logs["valid_rollout_steps"]),
                extra_valid,
                epoch_lr,
            )

    def train_one_epoch(self) -> tuple[float, dict[str, float]]:
        self.model.train()
        rollout_steps = self._rollout_steps_for_epoch()
        start = time.time()
        last_loss = float("nan")
        for batch_idx, data in enumerate(self.train_data_loader):
            if self.max_train_batches is not None and batch_idx >= self.max_train_batches:
                break
            self.iters += 1
            inp, target = [x.to(self.device, dtype=torch.float32) for x in data]
            self.optimizer.zero_grad(set_to_none=True)
            with amp.autocast(device_type=self.device.type, enabled=self.enable_amp):
                loss, _ = self._rollout_loss(inp, target, rollout_steps)
            if self.gscaler is not None:
                self.gscaler.scale(loss).backward()
                self.gscaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), self.max_gradient_norm)
                self.gscaler.step(self.optimizer)
                self.gscaler.update()
            else:
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.max_gradient_norm)
                self.optimizer.step()
            last_loss = float(loss.detach().item())
            logging.info("Epoch %d - Batch %d - Loss %.6f", self.epoch + 1, batch_idx, last_loss)
        self.epoch += 1
        return time.time() - start, {"loss": last_loss, "rollout_steps": float(rollout_steps)}

    @torch.no_grad()
    def validate_rollout_horizon(self, rollout_steps: int) -> float:
        self.model.eval()
        rollout_steps = min(int(rollout_steps), self.max_rollout_steps)
        total = 0.0
        steps = 0
        for batch_idx, data in enumerate(self.valid_data_loader):
            if self.max_valid_batches is not None and batch_idx >= self.max_valid_batches:
                break
            inp, target = [x.to(self.device, dtype=torch.float32) for x in data]
            loss, _ = self._rollout_loss(inp, target, rollout_steps)
            total += float(loss.item())
            steps += 1
        return total / max(steps, 1)

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
        valid_loss = self.validate_rollout_horizon(rollout_steps)
        logs: dict[str, float] = {
            "valid_loss": valid_loss,
            "valid_rollout_steps": float(rollout_steps),
        }

        eval_rollout_steps = _get(self.params, "eval_rollout_steps", [])
        if eval_rollout_steps is None:
            eval_rollout_steps = []
        for horizon in eval_rollout_steps:
            horizon = min(int(horizon), self.max_rollout_steps)
            key = f"valid_S{horizon}"
            if key in logs:
                continue
            if horizon == rollout_steps:
                logs[key] = valid_loss
            else:
                logs[key] = self.validate_rollout_horizon(horizon)

        return time.time() - start, logs

    def save_checkpoint(self, checkpoint_path: str) -> None:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(
            {
                "iters": self.iters,
                "epoch": self.epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "params": dict(getattr(self.params, "params", {})),
                "metadata": dict(self._last_epoch_metadata),
            },
            checkpoint_path,
        )

    def restore_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.iters = int(checkpoint.get("iters", 0))
        self.start_epoch = int(checkpoint.get("epoch", 0))
        self.epoch = self.start_epoch
        logging.info("Restored checkpoint %s", checkpoint_path)
