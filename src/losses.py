from __future__ import annotations

import torch
from torch import nn


class LatitudeWeightedMSE(nn.Module):
    """Latitude-weighted mean squared state loss for [B,C,H,W] tensors."""

    def __init__(self, latitudes_rad: torch.Tensor):
        super().__init__()
        weights = torch.cos(latitudes_rad.to(torch.float32)).clamp_min(0.0)
        weights = weights / weights.mean().clamp_min(1e-8)
        self.register_buffer("weights", weights.view(1, 1, -1, 1), persistent=False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"Prediction/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
        err2 = (pred - target).pow(2)
        return (err2 * self.weights.to(device=pred.device, dtype=pred.dtype)).mean()


def graph_gradient_loss(pred: torch.Tensor, target: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """MSE of edge differences on the native graph."""
    bsz, channels, height, width = pred.shape
    pred_nodes = pred.permute(0, 2, 3, 1).reshape(bsz, height * width, channels)
    target_nodes = target.permute(0, 2, 3, 1).reshape(bsz, height * width, channels)
    src = edge_index[0].to(pred.device)
    dst = edge_index[1].to(pred.device)
    pred_diff = pred_nodes[:, dst, :] - pred_nodes[:, src, :]
    target_diff = target_nodes[:, dst, :] - target_nodes[:, src, :]
    return torch.mean((pred_diff - target_diff).pow(2))

