from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _target_sequence(target: torch.Tensor) -> torch.Tensor:
    if target.dim() == 5:
        return target
    if target.dim() == 4:
        return target.unsqueeze(1)
    if target.dim() == 3:
        return target.unsqueeze(0).unsqueeze(0)
    raise ValueError(f"Expected target with 3, 4, or 5 dims, got {tuple(target.shape)}")


def _current_state_from_input(inp: torch.Tensor, n_history: int, output_channels: int) -> torch.Tensor:
    if inp.dim() == 3:
        inp = inp.unsqueeze(0)
    if inp.dim() != 4:
        raise ValueError(f"Expected input [B,C,H,W], got {tuple(inp.shape)}")
    num_steps = int(n_history) + 1
    if inp.shape[1] % num_steps != 0:
        raise ValueError(f"Input channels {inp.shape[1]} are not divisible by n_history+1={num_steps}")
    per_step_channels = inp.shape[1] // num_steps
    if int(output_channels) > int(per_step_channels):
        raise ValueError(f"output_channels={output_channels} exceeds per-step input channels={per_step_channels}")
    steps = inp.reshape(inp.shape[0], num_steps, per_step_channels, inp.shape[-2], inp.shape[-1])
    return steps[:, -1, : int(output_channels)]


def _save_delta_payload(
    output_path: str | os.PathLike[str],
    mean: np.ndarray,
    std: np.ndarray,
    num_samples: int,
    count: int,
    channels: list[int],
    resolution_mode: str,
    std_floor: float = 1.0e-5,
) -> dict[str, Any]:
    std = np.asarray(std, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    if not np.all(np.isfinite(mean)):
        raise ValueError("Delta mean contains NaN/Inf.")
    if not np.all(np.isfinite(std)):
        raise ValueError("Delta std contains NaN/Inf.")
    floor = float(std_floor)
    floor_mask = std <= floor
    floor_count = int(np.count_nonzero(floor_mask))
    raw_std_min = float(np.min(std))
    if floor_count:
        logging.warning(
            "Delta std floor applied to %d/%d channels: raw_min=%.6g floor=%.6g",
            floor_count,
            int(std.size),
            raw_std_min,
            floor,
        )
        std = np.maximum(std, floor)
    payload = {
        "delta_mean": mean.astype(np.float32),
        "delta_std": std.astype(np.float32),
        "num_samples": np.asarray(num_samples, dtype=np.int64),
        "num_grid_points": np.asarray(count, dtype=np.int64),
        "channels": np.asarray(channels, dtype=np.int64),
        "resolution_mode": np.asarray(str(resolution_mode)),
        "computed_in_normalized_state_space": np.asarray(True),
        "delta_std_floor": np.asarray(floor, dtype=np.float32),
        "delta_std_raw_min": np.asarray(raw_std_min, dtype=np.float32),
        "delta_std_floor_applied_count": np.asarray(floor_count, dtype=np.int64),
    }

    output_path = os.fspath(output_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez_compressed(output_path, **payload)
    logging.info(
        "Saved delta stats to %s | samples=%d | points_per_channel=%d | std min/max/mean=%.6g/%.6g/%.6g",
        output_path,
        num_samples,
        count,
        float(np.min(std)),
        float(np.max(std)),
        float(np.mean(std)),
    )
    return payload


def _can_fast_path_dataset(dataset: Dataset) -> bool:
    config = getattr(dataset, "config", None)
    return bool(
        config is not None
        and hasattr(dataset, "files_paths")
        and hasattr(dataset, "nc")
        and hasattr(dataset, "output_normalization_vectors")
        and bool(getattr(config, "normalize", False))
        and not bool(getattr(config, "roll", False))
        and not bool(getattr(config, "add_noise", False))
        and getattr(config, "crop_size_x", None) is None
        and getattr(config, "crop_size_y", None) is None
    )


def _compute_delta_stats_fast_netcdf(
    dataset: Dataset,
    output_path: str | os.PathLike[str],
    max_batches: int | None,
    batch_size: int,
    resolution_mode: str,
    output_channels: int,
    n_history: int,
    chunk_steps: int = 64,
    std_floor: float = 1.0e-5,
) -> dict[str, Any]:
    config = getattr(dataset, "config")
    dt = int(getattr(config, "dt"))
    out_channels = np.asarray(getattr(dataset, "out_channels"), dtype=np.int64)
    if out_channels.size != int(output_channels):
        raise ValueError(f"Dataset out_channels length {out_channels.size} does not match {output_channels}.")
    _, state_std = dataset.output_normalization_vectors()
    state_std = state_std.reshape(1, -1, 1, 1).astype(np.float32)

    sum_delta = np.zeros((int(output_channels),), dtype=np.float64)
    sum_delta2 = np.zeros((int(output_channels),), dtype=np.float64)
    count = 0
    num_samples = 0
    max_samples = None if max_batches is None else max(0, int(max_batches) * max(1, int(batch_size)))
    total_samples = int(getattr(dataset, "n_samples_total", 0))
    progress_every = max(1, min(1000, total_samples // 10 if total_samples else 1000))
    last_progress = 0

    logging.info(
        "Fast delta stats path: streaming %d NetCDF files in contiguous chunks (chunk_steps=%d).",
        len(getattr(dataset, "files_paths")),
        int(chunk_steps),
    )
    for file_idx, path in enumerate(getattr(dataset, "files_paths")):
        if max_samples is not None and num_samples >= max_samples:
            break
        with dataset.nc.Dataset(path, "r") as ds:
            fields = ds["fields"]
            time_len = int(fields.shape[0])
            first_center = int(dt * int(n_history))
            last_center = int(time_len - dt - 1)
            if first_center > last_center:
                continue
            center = first_center
            while center <= last_center:
                if max_samples is not None and num_samples >= max_samples:
                    break
                stop = min(center + int(chunk_steps), last_center + 1)
                if max_samples is not None:
                    stop = min(stop, center + (max_samples - num_samples))
                current = np.asarray(fields[center:stop, out_channels, :, :], dtype=np.float32)
                next_state = np.asarray(fields[center + dt : stop + dt, out_channels, :, :], dtype=np.float32)
                delta = (next_state - current) / (state_std + 1.0e-8)
                delta64 = delta.astype(np.float64, copy=False)
                sum_delta += delta64.sum(axis=(0, 2, 3))
                sum_delta2 += np.square(delta64).sum(axis=(0, 2, 3))
                chunk_samples = int(delta.shape[0])
                count += int(chunk_samples * delta.shape[2] * delta.shape[3])
                num_samples += chunk_samples
                center = stop

                if num_samples - last_progress >= progress_every:
                    logging.info(
                        "Delta stats progress: %d/%d samples processed",
                        num_samples,
                        max_samples if max_samples is not None else total_samples,
                    )
                    last_progress = num_samples
        logging.info(
            "Delta stats file %d/%d done: %s | samples=%d/%d",
            file_idx + 1,
            len(getattr(dataset, "files_paths")),
            os.path.basename(str(path)),
            num_samples,
            max_samples if max_samples is not None else total_samples,
        )

    if count <= 0:
        raise RuntimeError("No samples were available for delta-statistics computation.")
    mean = sum_delta / float(count)
    variance = np.maximum(sum_delta2 / float(count) - mean * mean, 0.0)
    std = np.sqrt(variance)
    channels = [int(x) for x in out_channels.tolist()]
    return _save_delta_payload(output_path, mean, std, num_samples, count, channels, resolution_mode, std_floor=std_floor)


def compute_delta_stats(
    dataset_or_loader: Dataset | DataLoader,
    output_path: str | os.PathLike[str],
    max_batches: int | None = None,
    batch_size: int = 1,
    resolution_mode: str | None = None,
    output_channels: int | None = None,
    n_history: int | None = None,
    std_floor: float = 1.0e-5,
) -> dict[str, Any]:
    """Stream one-step normalized-state tendency statistics to an NPZ file."""
    if isinstance(dataset_or_loader, DataLoader):
        loader = dataset_or_loader
        dataset = loader.dataset
        loader_batch_size = int(loader.batch_size or batch_size)
    else:
        dataset = dataset_or_loader
        loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
        loader_batch_size = int(batch_size)

    config = getattr(dataset, "config", None)
    if output_channels is None:
        if config is None:
            raise ValueError("output_channels is required when dataset.config is unavailable.")
        output_channels = len(getattr(config, "out_channels"))
    if n_history is None:
        n_history = int(getattr(config, "n_history", 1)) if config is not None else 1
    if resolution_mode is None:
        resolution_mode = str(getattr(config, "resolution_mode", "")) if config is not None else ""
    channels = list(getattr(config, "out_channels", list(range(int(output_channels))))) if config is not None else list(range(int(output_channels)))

    if _can_fast_path_dataset(dataset):
        return _compute_delta_stats_fast_netcdf(
            dataset,
            output_path,
            max_batches=max_batches,
            batch_size=loader_batch_size,
            resolution_mode=str(resolution_mode),
            output_channels=int(output_channels),
            n_history=int(n_history),
            std_floor=float(std_floor),
        )

    sum_delta = np.zeros((int(output_channels),), dtype=np.float64)
    sum_delta2 = np.zeros((int(output_channels),), dtype=np.float64)
    count = 0
    num_samples = 0

    for batch_idx, (inp, target) in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        inp = inp.to(dtype=torch.float32)
        target_seq = _target_sequence(target.to(dtype=torch.float32))
        if target_seq.shape[1] < 1:
            raise ValueError("Delta statistics require at least one target step.")
        current = _current_state_from_input(inp, int(n_history), int(output_channels))
        next_state = target_seq[:, 0]
        if current.shape != next_state.shape:
            raise ValueError(f"Current/next shape mismatch: {tuple(current.shape)} vs {tuple(next_state.shape)}")
        delta = (next_state - current).double()
        sum_delta += delta.sum(dim=(0, 2, 3)).cpu().numpy()
        sum_delta2 += delta.pow(2).sum(dim=(0, 2, 3)).cpu().numpy()
        count += int(delta.shape[0] * delta.shape[2] * delta.shape[3])
        num_samples += int(delta.shape[0])

    if count <= 0:
        raise RuntimeError("No samples were available for delta-statistics computation.")

    mean = sum_delta / float(count)
    variance = np.maximum(sum_delta2 / float(count) - mean * mean, 0.0)
    std = np.sqrt(variance)
    return _save_delta_payload(
        output_path,
        mean,
        std,
        num_samples,
        count,
        [int(x) for x in channels],
        str(resolution_mode),
        std_floor=float(std_floor),
    )


def load_delta_stats(
    path: str | os.PathLike[str],
    output_channels: int,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    payload = np.load(path, allow_pickle=False)
    if "delta_mean" not in payload or "delta_std" not in payload:
        raise ValueError(f"Delta stats file lacks delta_mean/delta_std: {path}")
    delta_mean = np.asarray(payload["delta_mean"], dtype=np.float32).reshape(-1)
    delta_std = np.asarray(payload["delta_std"], dtype=np.float32).reshape(-1)
    if delta_mean.shape[0] != int(output_channels) or delta_std.shape[0] != int(output_channels):
        raise ValueError(
            f"Delta stats channel count mismatch: mean={delta_mean.shape[0]} std={delta_std.shape[0]} "
            f"expected={int(output_channels)} from {path}"
        )
    if not np.all(np.isfinite(delta_mean)) or not np.all(np.isfinite(delta_std)):
        raise ValueError(f"Delta stats contain NaN/Inf: {path}")
    if np.any(delta_std <= float(eps)):
        raise ValueError(f"Delta std must be finite and > eps={eps:g}: {path}")
    metadata = {key: payload[key].tolist() for key in payload.files if key not in {"delta_mean", "delta_std"}}
    return torch.as_tensor(delta_mean), torch.as_tensor(delta_std), metadata
