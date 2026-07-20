from __future__ import annotations

import math
from typing import Any

import torch


def lead_sincos_values(
    lead: int | float | torch.Tensor,
    max_lead: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_lead = int(max_lead)
    if max_lead < 1:
        raise ValueError(f"max_lead must be >= 1, got {max_lead}.")
    lead_tensor = torch.as_tensor(lead, dtype=dtype, device=device)
    angle = (2.0 * math.pi / float(max_lead)) * lead_tensor
    return torch.sin(angle), torch.cos(angle)


def build_lead_conditioning_grid(
    lead: int | float | torch.Tensor,
    *,
    batch_size: int,
    height: int,
    width: int,
    max_lead: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    lead_sin, lead_cos = lead_sincos_values(lead, max_lead, dtype=dtype, device=device)
    if lead_sin.ndim == 0:
        values = torch.stack([lead_sin, lead_cos]).view(1, 2, 1, 1)
        values = values.expand(int(batch_size), -1, int(height), int(width))
    else:
        values = torch.stack([lead_sin.reshape(-1), lead_cos.reshape(-1)], dim=1).view(-1, 2, 1, 1)
        if values.shape[0] == 1 and int(batch_size) != 1:
            values = values.expand(int(batch_size), -1, int(height), int(width))
        elif values.shape[0] != int(batch_size):
            raise ValueError(f"Lead tensor batch size {values.shape[0]} does not match batch_size={batch_size}.")
        else:
            values = values.expand(-1, -1, int(height), int(width))
    return values.contiguous()


def lead_conditioning_debug_values(max_lead: int, leads: tuple[int, ...] = (1, 5, 10)) -> dict[int, dict[str, float]]:
    values: dict[int, dict[str, float]] = {}
    for lead in leads:
        lead_sin, lead_cos = lead_sincos_values(lead, max_lead, dtype=torch.float64, device=torch.device("cpu"))
        values[int(lead)] = {"sin": float(lead_sin.item()), "cos": float(lead_cos.item())}
    return values


def format_lead_sequence(leads: list[int] | tuple[int, ...] | range | Any) -> str:
    return "[" + ", ".join(str(int(lead)) for lead in list(leads)) + "]"
