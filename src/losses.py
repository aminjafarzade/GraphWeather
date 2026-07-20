from __future__ import annotations

import torch
from torch import nn


class LatitudeWeightedMSE(nn.Module):
    """Latitude-weighted mean squared state loss for [B,C,H,W] tensors.

    Optionally applies a fixed per-channel weight vector (length C). When
    ``channel_weights`` is None the loss is identical to the unweighted version
    (backward compatible). Weights combine multiplicatively with the loss
    ``channel_mask`` and the result stays a normalized weighted mean, so a
    global rescale of ``channel_weights`` does not change the loss and masked
    channels contribute nothing regardless of their per-channel weight.
    """

    def __init__(self, latitudes_rad: torch.Tensor, channel_weights: torch.Tensor | None = None):
        super().__init__()
        weights = torch.cos(latitudes_rad.to(torch.float32)).clamp_min(0.0)
        weights = weights / weights.mean().clamp_min(1e-8)
        self.register_buffer("weights", weights.view(1, 1, -1, 1), persistent=False)
        if channel_weights is None:
            self.register_buffer("channel_weights", None, persistent=False)
        else:
            cw = torch.as_tensor(channel_weights, dtype=torch.float32).reshape(-1)
            if not bool(torch.all(cw >= 0)):
                raise ValueError("channel_weights must be non-negative.")
            self.register_buffer("channel_weights", cw.view(1, -1, 1, 1), persistent=False)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"Prediction/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
        err2 = (pred - target).pow(2)
        weights = self.weights.to(device=pred.device, dtype=pred.dtype)
        chan_w = self.channel_weights
        if chan_w is not None:
            chan_w = chan_w.to(device=pred.device, dtype=pred.dtype)
            if chan_w.shape[1] != pred.shape[1]:
                raise ValueError(f"channel_weights length {chan_w.shape[1]} does not match channels {pred.shape[1]}")
        if channel_mask is None:
            if chan_w is None:
                return (err2 * weights).mean()
            num = (err2 * weights * chan_w).sum()
            denom = pred.shape[0] * pred.shape[-1] * weights.sum().clamp_min(1.0e-8) * chan_w.sum().clamp_min(1.0e-8)
            return num / denom
        mask = channel_mask.to(device=pred.device, dtype=pred.dtype).reshape(1, -1, 1, 1)
        if mask.shape[1] != pred.shape[1]:
            raise ValueError(f"channel_mask length {mask.shape[1]} does not match channels {pred.shape[1]}")
        eff = mask if chan_w is None else mask * chan_w
        denom = pred.shape[0] * pred.shape[-1] * weights.sum().clamp_min(1.0e-8) * eff.sum().clamp_min(1.0e-8)
        return (err2 * weights * eff).sum() / denom


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


def low_frequency_spectral_mask(
    height: int,
    width: int,
    lat_cutoff: int,
    lon_cutoff: int,
    *,
    include_dc: bool = True,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Boolean low-frequency rFFT2 mask with integer mode cutoffs."""
    height = int(height)
    width = int(width)
    lat_cutoff = int(lat_cutoff)
    lon_cutoff = int(lon_cutoff)
    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got {height}x{width}")
    if lat_cutoff < 0 or lon_cutoff < 0:
        raise ValueError("lat_cutoff and lon_cutoff must be non-negative.")
    lat_modes = torch.arange(height, device=device)
    lat_modes = torch.minimum(lat_modes, height - lat_modes)
    lon_modes = torch.arange(width // 2 + 1, device=device)
    mask = (lat_modes[:, None] <= lat_cutoff) & (lon_modes[None, :] <= lon_cutoff)
    if not include_dc:
        mask = mask.clone()
        mask[0, 0] = False
    return mask


def _default_latitudes_rad(height: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # Cell centers from north to south. This is only used when callers request
    # latitude weighting but do not provide graph/dataset latitudes.
    step = torch.pi / float(height)
    return torch.linspace(
        torch.pi / 2.0 - step / 2.0,
        -torch.pi / 2.0 + step / 2.0,
        int(height),
        device=device,
        dtype=dtype,
    )


def low_frequency_spectral_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    channel_indices: torch.Tensor | list[int] | tuple[int, ...],
    lat_cutoff: int,
    lon_cutoff: int,
    weight_latitude: bool = True,
    include_dc: bool = True,
    latitudes_rad: torch.Tensor | None = None,
) -> torch.Tensor:
    """Low-frequency spectral MSE on selected normalized state channels.

    Uses torch.fft.rfft2 over [H,W]. Selected tensors are cast to float32 before
    FFT so the loss works under AMP/bf16 autocast.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    if pred.ndim != 4:
        raise ValueError(f"Expected pred/target [B,C,H,W], got {tuple(pred.shape)}")
    if isinstance(channel_indices, torch.Tensor):
        channels = channel_indices.to(device=pred.device, dtype=torch.long)
    else:
        channels = torch.as_tensor(list(channel_indices), device=pred.device, dtype=torch.long)
    if channels.numel() == 0:
        raise ValueError("channel_indices must contain at least one channel.")
    if int(channels.min().item()) < 0 or int(channels.max().item()) >= int(pred.shape[1]):
        raise ValueError(f"channel_indices out of range for {pred.shape[1]} channels: {channels.detach().cpu().tolist()}")

    x = pred.index_select(1, channels).float()
    y = target.index_select(1, channels).float()
    err = x - y
    height, width = int(err.shape[-2]), int(err.shape[-1])

    if weight_latitude:
        if latitudes_rad is None:
            latitudes = _default_latitudes_rad(height, err.device, torch.float32)
        else:
            latitudes = latitudes_rad.to(device=err.device, dtype=torch.float32).reshape(-1)
        if int(latitudes.numel()) != height:
            raise ValueError(f"latitudes_rad length {latitudes.numel()} does not match height {height}")
        lat_weights = torch.cos(latitudes).clamp_min(0.0)
        lat_weights = lat_weights / lat_weights.mean().clamp_min(1.0e-8)
        err = err * lat_weights.sqrt().view(1, 1, height, 1)

    err_fft = torch.fft.rfft2(err, dim=(-2, -1), norm="ortho")
    mask = low_frequency_spectral_mask(
        height,
        width,
        lat_cutoff,
        lon_cutoff,
        include_dc=include_dc,
        device=err.device,
    )
    if not bool(mask.any()):
        raise ValueError("Low-frequency spectral mask selected zero coefficients.")
    low_fft = err_fft[..., mask]
    return (low_fft.real.pow(2) + low_fft.imag.pow(2)).mean()
