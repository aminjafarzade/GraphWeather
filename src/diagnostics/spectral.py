from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SpectralCurve:
    bins: np.ndarray
    rmse: np.ndarray


def _as_grid_field(tensor: torch.Tensor, grid_shape: tuple[int, int] | None = None) -> torch.Tensor:
    x = tensor.detach().to(dtype=torch.float32)
    if x.dim() == 4:
        return x
    if x.dim() == 3:
        if grid_shape is None:
            raise ValueError(f"Cannot reshape [B,N,C] tensor {tuple(x.shape)} without grid_shape.")
        height, width = int(grid_shape[0]), int(grid_shape[1])
        if x.shape[1] != height * width:
            raise ValueError(f"Tensor node count {x.shape[1]} does not match grid_shape={grid_shape}.")
        return x.reshape(x.shape[0], height, width, x.shape[2]).permute(0, 3, 1, 2)
    raise ValueError(f"Expected prediction/target [B,C,H,W] or [B,N,C], got {tuple(x.shape)}")


def radial_wavenumber_bins(height: int, width: int, device: torch.device | None = None) -> torch.Tensor:
    ky = torch.fft.fftfreq(int(height), d=1.0, device=device) * int(height)
    kx = torch.fft.rfftfreq(int(width), d=1.0, device=device) * int(width)
    radius = torch.sqrt(ky[:, None].square() + kx[None, :].square())
    return torch.floor(radius).to(torch.long)


def spectral_rmse_2d(
    pred: torch.Tensor,
    target: torch.Tensor,
    variable_indices: list[int],
    grid_shape: tuple[int, int] | None = None,
) -> dict[int, SpectralCurve]:
    pred_grid = _as_grid_field(pred, grid_shape=grid_shape)
    target_grid = _as_grid_field(target, grid_shape=grid_shape)
    if pred_grid.shape != target_grid.shape:
        raise ValueError(f"Prediction/target shape mismatch: {tuple(pred_grid.shape)} != {tuple(target_grid.shape)}")
    height, width = int(pred_grid.shape[-2]), int(pred_grid.shape[-1])
    bin_ids = radial_wavenumber_bins(height, width, device=pred_grid.device)
    max_bin = int(bin_ids.max().item())
    curves: dict[int, SpectralCurve] = {}
    for idx in variable_indices:
        if idx < 0 or idx >= pred_grid.shape[1]:
            continue
        err = pred_grid[:, idx] - target_grid[:, idx]
        fft_err = torch.fft.rfft2(err, norm="ortho")
        power = fft_err.abs().square()
        values = []
        bins = []
        for bin_idx in range(max_bin + 1):
            mask = bin_ids == bin_idx
            if not bool(mask.any()):
                continue
            values.append(float(torch.sqrt(power[:, mask].mean()).item()))
            bins.append(bin_idx)
        curves[int(idx)] = SpectralCurve(
            bins=np.asarray(bins, dtype=np.int64),
            rmse=np.asarray(values, dtype=np.float64),
        )
    return curves


def spectral_band_summary(curve: SpectralCurve) -> dict[str, float]:
    if curve.rmse.size == 0:
        return {"low_k_rmse": float("nan"), "mid_k_rmse": float("nan"), "high_k_rmse": float("nan")}
    splits = np.array_split(curve.rmse, 3)
    names = ["low_k_rmse", "mid_k_rmse", "high_k_rmse"]
    return {
        name: float(np.mean(values)) if values.size else float("nan")
        for name, values in zip(names, splits)
    }
