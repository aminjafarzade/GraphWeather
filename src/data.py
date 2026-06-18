from __future__ import annotations

import glob
import logging
import math
import os
import random
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class DataConfig:
    dt: int
    n_history: int
    in_channels: Sequence[int]
    out_channels: Sequence[int]
    crop_size_x: Optional[int] = None
    crop_size_y: Optional[int] = None
    roll: bool = False
    orography: bool = False
    orography_path: Optional[str] = None
    add_noise: bool = False
    noise_std: float = 0.0
    normalize: bool = True
    normalization: str = "zscore"
    global_means_path: Optional[str] = None
    global_stds_path: Optional[str] = None
    add_grid: bool = False
    gridtype: str = "linear"
    N_grid_channels: int = 2
    rollout_steps: int = 1
    batch_size: int = 1
    num_workers: int = 0


def _load_netcdf4():
    try:
        import netCDF4 as nc
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "netCDF4 is required for KAI_5 NetCDF data loading. "
            "Use an environment that has both torch and netCDF4 installed."
        ) from exc
    return nc


def _normalization_vectors(
    means: np.ndarray,
    stds: np.ndarray,
    channels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if means.ndim == 4:
        m = means.squeeze()[channels]
        s = stds.squeeze()[channels]
    elif means.ndim == 2:
        m = means[0, channels]
        s = stds[0, channels]
    elif means.ndim == 1:
        m = means[channels]
        s = stds[channels]
    else:
        raise ValueError(f"Unexpected normalization stats shape: {means.shape}")
    return m.astype(np.float32), s.astype(np.float32)


def _apply_normalization(
    img: np.ndarray,
    channels: np.ndarray,
    means: Optional[np.ndarray],
    stds: Optional[np.ndarray],
    enabled: bool,
    normalization: str,
) -> np.ndarray:
    if not enabled:
        return img
    if means is None or stds is None:
        raise ValueError("Normalization is enabled but means/stds were not loaded.")
    if normalization.lower() != "zscore":
        raise NotImplementedError("Only zscore normalization is supported.")

    m, s = _normalization_vectors(means, stds, channels)
    if img.ndim == 4:
        return (img - m.reshape(1, -1, 1, 1)) / (s.reshape(1, -1, 1, 1) + 1e-8)
    if img.ndim == 3:
        return (img - m.reshape(-1, 1, 1)) / (s.reshape(-1, 1, 1) + 1e-8)
    raise ValueError(f"Unexpected image ndim: {img.ndim}")


def _add_grid_channels(img: np.ndarray, gridtype: str, n_grid_channels: int) -> np.ndarray:
    add_time = False
    if img.ndim == 3:
        img = img[None, ...]
        add_time = True

    _, _, height, width = img.shape
    if gridtype == "linear":
        if n_grid_channels != 2:
            raise ValueError("linear grid requires N_grid_channels=2")
        lat_coord = np.linspace(-1.0, 1.0, height, dtype=np.float32)
        lon_coord = np.linspace(-1.0, 1.0, width, dtype=np.float32)
        lon_grid, lat_grid = np.meshgrid(lon_coord, lat_coord)
        grid = np.stack([lat_grid, lon_grid], axis=0)
    elif gridtype == "sinusoidal":
        if n_grid_channels != 4:
            raise ValueError("sinusoidal grid requires N_grid_channels=4")
        lat_coord = np.linspace(0.0, 2.0 * math.pi, height, dtype=np.float32)
        lon_coord = np.linspace(0.0, 2.0 * math.pi, width, dtype=np.float32)
        lon_grid, lat_grid = np.meshgrid(lon_coord, lat_coord)
        grid = np.stack(
            [np.sin(lat_grid), np.cos(lat_grid), np.sin(lon_grid), np.cos(lon_grid)],
            axis=0,
        ).astype(np.float32)
    else:
        raise ValueError(f"Unknown grid type '{gridtype}'")

    grid = np.repeat(grid[None, ...], img.shape[0], axis=0)
    out = np.concatenate([img, grid], axis=1)
    return out[0] if add_time else out


def _reshape_fields(
    img: np.ndarray,
    inp_or_tar: str,
    channels: np.ndarray,
    config: DataConfig,
    train: bool,
    crop_x: int,
    crop_y: int,
    rnd_x: int,
    rnd_y: int,
    y_roll: int,
    means: Optional[np.ndarray],
    stds: Optional[np.ndarray],
    orog: Optional[np.ndarray],
    add_noise: bool,
) -> torch.Tensor:
    if img.ndim == 3:
        img = img[None, ...]

    img = img[:, channels, :, :]
    img = _apply_normalization(
        img,
        channels=channels,
        means=means,
        stds=stds,
        enabled=config.normalize,
        normalization=config.normalization,
    )

    if inp_or_tar == "inp" and config.add_grid:
        img = _add_grid_channels(img, config.gridtype, config.N_grid_channels)

    if inp_or_tar == "inp" and config.orography:
        if orog is None:
            raise ValueError("orography=True but no orography field was loaded.")
        if orog.shape != img.shape[-2:]:
            raise ValueError(f"Orography shape {orog.shape} does not match {img.shape[-2:]}.")
        img = np.concatenate(
            [img, np.repeat(orog[None, None, :, :], img.shape[0], axis=0)],
            axis=1,
        )

    if y_roll:
        img = np.roll(img, shift=y_roll, axis=-1)

    if train and (config.crop_size_x or config.crop_size_y):
        img = img[:, :, rnd_x : rnd_x + crop_x, rnd_y : rnd_y + crop_y]
    else:
        img = img[:, :, :crop_x, :crop_y]

    if inp_or_tar == "inp":
        t, c, h, w = img.shape
        img = img.reshape(t * c, h, w)
    else:
        if img.shape[0] == 1:
            img = img[0]

    if add_noise:
        noise = np.random.normal(0.0, config.noise_std, size=img.shape).astype(np.float32)
        img = img + noise

    return torch.as_tensor(img, dtype=torch.float32)


class ClimateNetCDFDataset(Dataset):
    """KAI-style NetCDF dataset that can optionally return rollout target sequences."""

    def __init__(self, config: DataConfig, data_dir: str, train: bool = True):
        super().__init__()
        self.config = config
        self.data_dir = data_dir
        self.train = train
        self.nc = _load_netcdf4()
        self.in_channels = np.asarray(config.in_channels, dtype=np.int64)
        self.out_channels = np.asarray(config.out_channels, dtype=np.int64)

        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray] = None
        if config.normalize:
            if not config.global_means_path or not config.global_stds_path:
                raise ValueError("global_means_path and global_stds_path are required.")
            self.means = np.load(config.global_means_path)
            self.stds = np.load(config.global_stds_path)

        self.orography_field: Optional[np.ndarray] = None
        if config.orography:
            if not config.orography_path:
                raise ValueError("orography_path is required when orography=True.")
            with self.nc.Dataset(config.orography_path, "r") as ds:
                key = "orog" if "orog" in ds.variables else next(iter(ds.variables))
                self.orography_field = np.asarray(ds[key][:], dtype=np.float32).squeeze()

        self.files_paths = sorted(glob.glob(os.path.join(data_dir, "*.nc")))
        if not self.files_paths:
            raise FileNotFoundError(f"No .nc files found under {data_dir}")
        self._discover_shapes()
        self._build_index()

    def _discover_shapes(self) -> None:
        self._file_lengths: list[int] = []
        self.img_shape_x: Optional[int] = None
        self.img_shape_y: Optional[int] = None
        for path in self.files_paths:
            with self.nc.Dataset(path, "r") as ds:
                if "fields" not in ds.variables:
                    raise KeyError(f"{path} has no 'fields' variable")
                time_len, _, height, width = ds["fields"].shape
                self._file_lengths.append(int(time_len))
                if self.img_shape_x is None:
                    self.img_shape_x = int(height)
                    self.img_shape_y = int(width)
                elif (height, width) != (self.img_shape_x, self.img_shape_y):
                    raise ValueError(f"Inconsistent grid in {path}: {(height, width)}")
        logging.info(
            "Found %d NetCDF files under %s. Spatial dims: %d x %d.",
            len(self.files_paths),
            self.data_dir,
            self.img_shape_x,
            self.img_shape_y,
        )

    def _build_index(self) -> None:
        dt = int(self.config.dt)
        n_history = int(self.config.n_history)
        rollout_steps = int(max(1, self.config.rollout_steps))
        self._valid_starts: list[int] = []
        self._num_valid: list[int] = []
        for length in self._file_lengths:
            t_min = dt * n_history
            t_max = length - dt * rollout_steps - 1
            if t_min > t_max:
                self._valid_starts.append(0)
                self._num_valid.append(0)
            else:
                self._valid_starts.append(t_min)
                self._num_valid.append(t_max - t_min + 1)
        self._file_offsets = [0]
        for n_valid in self._num_valid:
            self._file_offsets.append(self._file_offsets[-1] + n_valid)
        self.n_samples_total = self._file_offsets[-1]
        if self.n_samples_total <= 0:
            raise RuntimeError("No valid samples found for the requested history/rollout.")
        logging.info("Total valid samples: %d", self.n_samples_total)

    def __len__(self) -> int:
        return self.n_samples_total

    def _locate_sample(self, idx: int) -> tuple[int, int]:
        if idx < 0 or idx >= self.n_samples_total:
            raise IndexError(idx)
        file_idx = 0
        while file_idx + 1 < len(self._file_offsets) and self._file_offsets[file_idx + 1] <= idx:
            file_idx += 1
        local_idx = idx - self._file_offsets[file_idx]
        return file_idx, self._valid_starts[file_idx] + local_idx

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        file_idx, center_idx = self._locate_sample(idx)
        dt = int(self.config.dt)
        n_history = int(self.config.n_history)
        rollout_steps = int(max(1, self.config.rollout_steps))

        crop_x = self.config.crop_size_x or int(self.img_shape_x)
        crop_y = self.config.crop_size_y or int(self.img_shape_y)
        if crop_x > self.img_shape_x or crop_y > self.img_shape_y:
            raise ValueError(f"Crop {(crop_x, crop_y)} is larger than grid {(self.img_shape_x, self.img_shape_y)}")

        if self.train and self.config.roll:
            y_roll = random.randint(0, int(self.img_shape_y) - 1)
        else:
            y_roll = 0

        if self.train and (self.config.crop_size_x or self.config.crop_size_y):
            rnd_x = random.randint(0, int(self.img_shape_x) - crop_x)
            rnd_y = random.randint(0, int(self.img_shape_y) - crop_y)
        else:
            rnd_x = 0
            rnd_y = 0

        path = self.files_paths[file_idx]
        with self.nc.Dataset(path, "r") as ds:
            fields = ds["fields"]
            input_start = center_idx - dt * n_history
            input_stop = center_idx + 1
            inp_seq = np.asarray(
                fields[input_start:input_stop:dt, :, :, :],
                dtype=np.float32,
            )
            target_times = [center_idx + dt * step for step in range(1, rollout_steps + 1)]
            tar_seq = np.asarray(fields[target_times, :, :, :], dtype=np.float32)

        inp = _reshape_fields(
            inp_seq,
            inp_or_tar="inp",
            channels=self.in_channels,
            config=self.config,
            train=self.train,
            crop_x=crop_x,
            crop_y=crop_y,
            rnd_x=rnd_x,
            rnd_y=rnd_y,
            y_roll=y_roll,
            means=self.means,
            stds=self.stds,
            orog=self.orography_field,
            add_noise=self.config.add_noise if self.train else False,
        )

        target_tensors = []
        for step in range(rollout_steps):
            target_tensors.append(
                _reshape_fields(
                    tar_seq[step],
                    inp_or_tar="tar",
                    channels=self.out_channels,
                    config=self.config,
                    train=self.train,
                    crop_x=crop_x,
                    crop_y=crop_y,
                    rnd_x=rnd_x,
                    rnd_y=rnd_y,
                    y_roll=y_roll,
                    means=self.means,
                    stds=self.stds,
                    orog=None,
                    add_noise=False,
                )
            )
        tar = target_tensors[0] if rollout_steps == 1 else torch.stack(target_tensors, dim=0)
        return inp, tar


def build_data_loader(
    config: DataConfig,
    data_dir: str,
    train: bool = True,
) -> tuple[DataLoader, ClimateNetCDFDataset]:
    dataset = ClimateNetCDFDataset(config=config, data_dir=data_dir, train=train)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=train,
        num_workers=config.num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, dataset
