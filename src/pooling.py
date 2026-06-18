from __future__ import annotations

import torch
from torch import nn


class MeanMaxPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(2 * dim, dim)

    def forward(self, h_fine: torch.Tensor, pool_map: torch.Tensor, num_coarse: int) -> torch.Tensor:
        bsz, _, dim = h_fine.shape
        pool_map = pool_map.to(device=h_fine.device)
        mean = torch.zeros(bsz, num_coarse, dim, device=h_fine.device, dtype=h_fine.dtype)
        mean.index_add_(1, pool_map, h_fine)
        counts = torch.bincount(pool_map, minlength=num_coarse).to(device=h_fine.device, dtype=h_fine.dtype)
        mean = mean / counts.clamp_min(1.0).view(1, -1, 1)

        maxv = torch.full((bsz, num_coarse, dim), -torch.inf, device=h_fine.device, dtype=h_fine.dtype)
        index = pool_map.view(1, -1, 1).expand(bsz, -1, dim)
        maxv.scatter_reduce_(1, index, h_fine, reduce="amax", include_self=True)
        maxv = torch.where(torch.isfinite(maxv), maxv, torch.zeros_like(maxv))
        return self.proj(torch.cat([mean, maxv], dim=-1))


class ParentUnpoolFuse(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, h_coarse: torch.Tensor, parent_map: torch.Tensor, h_skip: torch.Tensor) -> torch.Tensor:
        parent_map = parent_map.to(device=h_coarse.device)
        h_up = h_coarse[:, parent_map, :]
        return self.fuse(torch.cat([h_skip, h_up], dim=-1))

