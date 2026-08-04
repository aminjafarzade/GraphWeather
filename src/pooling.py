from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


class MeanMaxPool(nn.Module):
    def __init__(self, dim: int, pooling: dict[str, Any] | None = None, name: str = ""):
        super().__init__()
        self.name = str(name)
        raw = dict(pooling or {})
        self.pooling_type = str(raw.get("type", "default")).strip().lower()
        self.init_scale = float(raw.get("init_scale", 1.0))
        self.max_scale = float(raw.get("max_scale", 2.0))
        self.mean_type = str(raw.get("mean_type", "mean")).strip().lower()
        self.include_max = bool(raw.get("include_max", True))
        if self.pooling_type not in {"default", "parent_index_meanmax", "scalar_gated_meanmax"}:
            raise ValueError(f"Unsupported pooling.type={self.pooling_type!r}.")
        if self.mean_type not in {"mean", "area_weighted"}:
            raise ValueError(f"Unsupported pooling.mean_type={self.mean_type!r}.")
        if self.init_scale <= 0.0 or self.max_scale <= 0.0 or self.init_scale >= self.max_scale:
            raise ValueError(
                f"pooling init_scale={self.init_scale:g} must be > 0 and < max_scale={self.max_scale:g}."
            )
        if self.pooling_type == "scalar_gated_meanmax":
            initial_ratio = self.init_scale / self.max_scale
            initial_logit = math.log(initial_ratio / (1.0 - initial_ratio))
            self.mean_gate_logit = nn.Parameter(torch.tensor(float(initial_logit), dtype=torch.float32))
            self.max_gate_logit = nn.Parameter(torch.tensor(float(initial_logit), dtype=torch.float32))
        else:
            self.register_parameter("mean_gate_logit", None)
            self.register_parameter("max_gate_logit", None)
        self.proj = nn.Linear((2 if self.include_max else 1) * dim, dim)

    def _scale_tensor(self, logit: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return self.max_scale * torch.sigmoid(logit.to(device=device, dtype=dtype))

    def mean_scale_tensor(self, dtype: torch.dtype | None = None, device: torch.device | None = None) -> torch.Tensor:
        if self.mean_gate_logit is None:
            return torch.tensor(1.0, dtype=dtype or torch.float32, device=device)
        return self._scale_tensor(
            self.mean_gate_logit,
            dtype or self.mean_gate_logit.dtype,
            device or self.mean_gate_logit.device,
        )

    def max_scale_tensor(self, dtype: torch.dtype | None = None, device: torch.device | None = None) -> torch.Tensor:
        if self.max_gate_logit is None:
            return torch.tensor(1.0, dtype=dtype or torch.float32, device=device)
        return self._scale_tensor(
            self.max_gate_logit,
            dtype or self.max_gate_logit.dtype,
            device or self.max_gate_logit.device,
        )

    def gate_values(self) -> dict[str, float]:
        if self.pooling_type != "scalar_gated_meanmax":
            return {}
        return {
            "mean": float(self.mean_scale_tensor(dtype=torch.float32, device=torch.device("cpu")).detach().item()),
            "max": float(self.max_scale_tensor(dtype=torch.float32, device=torch.device("cpu")).detach().item()),
        }

    def forward(
        self,
        h_fine: torch.Tensor,
        pool_map: torch.Tensor,
        num_coarse: int,
        child_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, _, dim = h_fine.shape
        pool_map = pool_map.to(device=h_fine.device)
        if self.mean_type == "area_weighted":
            if child_weights is None:
                raise ValueError("pooling.mean_type='area_weighted' requires child_weights.")
            weights = child_weights.to(device=h_fine.device, dtype=h_fine.dtype).reshape(-1)
            if int(weights.numel()) != int(h_fine.shape[1]):
                raise ValueError(
                    f"child_weights length {int(weights.numel())} does not match fine nodes {int(h_fine.shape[1])}."
                )
            weighted = h_fine * weights.view(1, -1, 1)
            mean = torch.zeros(bsz, num_coarse, dim, device=h_fine.device, dtype=h_fine.dtype)
            mean.index_add_(1, pool_map, weighted)
            weight_sum = torch.zeros(num_coarse, device=h_fine.device, dtype=h_fine.dtype)
            weight_sum.index_add_(0, pool_map, weights)
            mean = mean / weight_sum.clamp_min(torch.finfo(h_fine.dtype).eps).view(1, -1, 1)
        else:
            mean = torch.zeros(bsz, num_coarse, dim, device=h_fine.device, dtype=h_fine.dtype)
            mean.index_add_(1, pool_map, h_fine)
            counts = torch.bincount(pool_map, minlength=num_coarse).to(device=h_fine.device, dtype=h_fine.dtype)
            mean = mean / counts.clamp_min(1.0).view(1, -1, 1)

        if self.pooling_type == "scalar_gated_meanmax":
            mean = self.mean_scale_tensor(dtype=mean.dtype, device=mean.device) * mean
        if not self.include_max:
            return self.proj(mean)

        maxv = torch.full((bsz, num_coarse, dim), -torch.inf, device=h_fine.device, dtype=h_fine.dtype)
        index = pool_map.view(1, -1, 1).expand(bsz, -1, dim)
        maxv.scatter_reduce_(1, index, h_fine, reduce="amax", include_self=True)
        maxv = torch.where(torch.isfinite(maxv), maxv, torch.zeros_like(maxv))
        if self.pooling_type == "scalar_gated_meanmax":
            maxv = self.max_scale_tensor(dtype=maxv.dtype, device=maxv.device) * maxv
        return self.proj(torch.cat([mean, maxv], dim=-1))


class ParentUnpoolFuse(nn.Module):
    """Replicate coarse parents onto fine nodes, then merge the encoder skip.

    ``sum`` performs the parameter-free additive U-Net skip exactly. ``default``
    and ``scalar_gated`` retain the historical concat-MLP behavior so existing
    checkpoints keep their original architecture.
    """

    def __init__(self, dim: int, skip_fusion: dict[str, Any] | None = None, name: str = ""):
        super().__init__()
        self.name = str(name)
        raw = dict(skip_fusion or {})
        self.skip_fusion_type = str(raw.get("type", "default")).strip().lower()
        self.init_scale = float(raw.get("init_scale", 1.0))
        self.max_scale = float(raw.get("max_scale", 2.0))
        if self.skip_fusion_type not in {"default", "scalar_gated", "sum"}:
            raise ValueError(f"Unsupported skip_fusion.type={self.skip_fusion_type!r}.")
        if self.init_scale <= 0.0 or self.max_scale <= 0.0 or self.init_scale >= self.max_scale:
            raise ValueError(
                f"skip_fusion init_scale={self.init_scale:g} must be > 0 and < max_scale={self.max_scale:g}."
            )
        if self.skip_fusion_type == "scalar_gated":
            initial_ratio = self.init_scale / self.max_scale
            initial_logit = math.log(initial_ratio / (1.0 - initial_ratio))
            self.skip_gate_logit = nn.Parameter(torch.tensor(float(initial_logit), dtype=torch.float32))
            self.up_gate_logit = nn.Parameter(torch.tensor(float(initial_logit), dtype=torch.float32))
        else:
            self.register_parameter("skip_gate_logit", None)
            self.register_parameter("up_gate_logit", None)
        self.fuse = (
            None
            if self.skip_fusion_type == "sum"
            else nn.Sequential(
                nn.Linear(2 * dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )
        )

    def _scale_tensor(self, logit: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return self.max_scale * torch.sigmoid(logit.to(device=device, dtype=dtype))

    def skip_scale_tensor(self, dtype: torch.dtype | None = None, device: torch.device | None = None) -> torch.Tensor:
        if self.skip_gate_logit is None:
            return torch.tensor(1.0, dtype=dtype or torch.float32, device=device)
        return self._scale_tensor(
            self.skip_gate_logit,
            dtype or self.skip_gate_logit.dtype,
            device or self.skip_gate_logit.device,
        )

    def up_scale_tensor(self, dtype: torch.dtype | None = None, device: torch.device | None = None) -> torch.Tensor:
        if self.up_gate_logit is None:
            return torch.tensor(1.0, dtype=dtype or torch.float32, device=device)
        return self._scale_tensor(
            self.up_gate_logit,
            dtype or self.up_gate_logit.dtype,
            device or self.up_gate_logit.device,
        )

    def gate_values(self) -> dict[str, float]:
        if self.skip_fusion_type != "scalar_gated":
            return {}
        return {
            "skip": float(self.skip_scale_tensor(dtype=torch.float32, device=torch.device("cpu")).detach().item()),
            "up": float(self.up_scale_tensor(dtype=torch.float32, device=torch.device("cpu")).detach().item()),
        }

    def forward(self, h_coarse: torch.Tensor, parent_map: torch.Tensor, h_skip: torch.Tensor) -> torch.Tensor:
        parent_map = parent_map.to(device=h_coarse.device)
        h_up = h_coarse[:, parent_map, :]
        if self.skip_fusion_type == "sum":
            return h_skip + h_up
        if self.skip_fusion_type == "scalar_gated":
            h_skip = self.skip_scale_tensor(dtype=h_skip.dtype, device=h_skip.device) * h_skip
            h_up = self.up_scale_tensor(dtype=h_up.dtype, device=h_up.device) * h_up
        if self.fuse is None:
            raise RuntimeError("Non-sum skip fusion requires a fusion module.")
        return self.fuse(torch.cat([h_skip, h_up], dim=-1))
