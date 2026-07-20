from __future__ import annotations

import numpy as np
import torch

from .spectral import radial_wavenumber_bins


def _sht_power(field: torch.Tensor):
    """SH power per total wavenumber l. Returns (l, power) or None if torch_harmonics is absent."""
    try:
        from torch_harmonics import RealSHT
    except Exception:
        return None
    height, width = int(field.shape[-2]), int(field.shape[-1])
    sht = RealSHT(height, width, grid="equiangular").to(field.device)
    coeffs = sht(field.to(torch.float32))          # [B, l, m] complex
    power_l = coeffs.abs().square().mean(dim=0).sum(dim=-1)   # [l]
    ell = np.arange(power_l.shape[0], dtype=np.int64)
    return ell, power_l.detach().cpu().numpy().astype(np.float64)


def _rfft2_power(field: torch.Tensor):
    """Radial lat-lon rFFT2 power spectrum. field [B,H,W] -> (bins, power)."""
    height, width = int(field.shape[-2]), int(field.shape[-1])
    power = torch.fft.rfft2(field.to(torch.float32), norm="ortho").abs().square()
    bin_ids = radial_wavenumber_bins(height, width, device=field.device)
    max_bin = int(bin_ids.max().item())
    bins, values = [], []
    for k in range(max_bin + 1):
        mask = bin_ids == k
        if not bool(mask.any()):
            continue
        values.append(float(power[:, mask].mean().item()))
        bins.append(k)
    return np.asarray(bins, dtype=np.int64), np.asarray(values, dtype=np.float64)


def power_spectrum_radial(field: torch.Tensor, prefer_sht: bool = True):
    """Power spectrum of a single-variable field [B,H,W]. Tries SHT, falls back to rFFT2.
    Returns (bins, power, backend) with backend in {'sht','rfft2'}."""
    if prefer_sht:
        result = _sht_power(field)
        if result is not None:
            return result[0], result[1], "sht"
    bins, power = _rfft2_power(field)
    return bins, power, "rfft2"


def variance_ratio(pred: torch.Tensor, truth: torch.Tensor, eps: float = 1e-12) -> float:
    return float(pred.to(torch.float32).std().item() / (truth.to(torch.float32).std().item() + eps))
