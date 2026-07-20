from __future__ import annotations

import logging
import os
from typing import Any

import torch


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


def _cuda_kernel_probe_error(device: torch.device) -> RuntimeError | None:
    if device.type != "cuda":
        return None
    try:
        with torch.cuda.device(device):
            probe = torch.empty((1,), device=device)
            probe = probe + 1.0
            torch.cuda.synchronize(device)
        return None
    except RuntimeError as exc:
        message = str(exc)
        if "no kernel image is available" in message or "invalid device function" in message:
            return exc
        raise


def _cuda_arch_list() -> list[str]:
    if not hasattr(torch.cuda, "get_arch_list"):
        return []
    try:
        return list(torch.cuda.get_arch_list())
    except Exception:
        return []


def _cuda_unsupported_device_error(device: torch.device, exc: RuntimeError | None = None) -> RuntimeError:
    return RuntimeError(
        "Selected CUDA device cannot run CUDA kernels with this PyTorch build: "
        f"{_cuda_device_description(device)}. "
        f"torch.version.cuda={torch.version.cuda}; "
        f"torch CUDA arch list={_cuda_arch_list() or 'unknown'}. "
        "Choose a different GPU, or install/rebuild PyTorch with support for this GPU's compute capability. "
        "When masking a physical GPU with CUDA_VISIBLE_DEVICES, use --device cuda or --device cuda:0."
    )


def _assert_cuda_device_usable(device: torch.device) -> None:
    exc = _cuda_kernel_probe_error(device)
    if exc is not None:
        raise _cuda_unsupported_device_error(device, exc) from exc


def _resolve_auto_cuda_device(local_rank: int = 0) -> torch.device:
    device_count = torch.cuda.device_count()
    if device_count <= 0:
        return torch.device("cpu")
    preferred = int(local_rank) % device_count
    ordered = list(range(preferred, device_count)) + list(range(0, preferred))
    skipped: list[str] = []
    for index in ordered:
        device = torch.device(f"cuda:{index}")
        exc = _cuda_kernel_probe_error(device)
        if exc is None:
            if index != preferred:
                logging.warning(
                    "Default CUDA device cuda:%d is not usable by this PyTorch build; using %s instead.",
                    preferred,
                    _cuda_device_description(device),
                )
            return device
        skipped.append(_cuda_device_description(device))
    raise RuntimeError(
        "No visible CUDA device can run kernels with this PyTorch build. "
        f"Visible devices: {', '.join(skipped) or 'none'}. "
        f"torch.version.cuda={torch.version.cuda}; "
        f"torch CUDA arch list={_cuda_arch_list() or 'unknown'}. "
        "Choose a supported GPU with CUDA_VISIBLE_DEVICES, or install/rebuild PyTorch for this GPU."
    )


def _resolve_device(device_name: Any, local_rank: int = 0) -> torch.device:
    requested = str(device_name or "auto").strip().lower()
    if requested in {"", "auto"}:
        return _resolve_auto_cuda_device(local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if requested.isdigit():
        requested = f"cuda:{requested}"
    elif requested == "cuda":
        return _resolve_auto_cuda_device(local_rank) if torch.cuda.is_available() else torch.device("cpu")

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
            f"Requested CUDA device '{requested}', but only {device_count} CUDA device(s) are visible. "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}."
        )
    return device


def _map_requested_device_to_visible(device: str | None) -> str | None:
    if device is None:
        return None
    requested = str(device).strip()
    lowered = requested.lower()
    if lowered == "":
        return None
    if lowered.isdigit():
        tokens = _visible_cuda_device_tokens()
        if not tokens:
            os.environ["CUDA_VISIBLE_DEVICES"] = lowered
            return "cuda:0"
        if lowered in tokens:
            return f"cuda:{tokens.index(lowered)}"
        return f"cuda:{lowered}"
    return requested
