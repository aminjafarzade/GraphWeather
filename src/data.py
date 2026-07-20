from __future__ import annotations

import glob
import logging
import math
import os
import random
from datetime import datetime
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
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: Optional[int] = None
    resolution_mode: str = "5p625"
    expected_grid_shape: Optional[Sequence[int]] = None
    return_metadata: bool = False


def _load_netcdf4():
    try:
        import netCDF4 as nc
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "netCDF4 is required for KAI_5 NetCDF data loading. "
            "Use an environment that has both torch and netCDF4 installed."
        ) from exc
    return nc


def _decode_channel_name(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _dayofyear_and_year_length(value: object) -> tuple[int, int]:
    if isinstance(value, np.datetime64):
        dt = value.astype("datetime64[ms]").astype(datetime)
        year = int(dt.year)
        doy = int(dt.timetuple().tm_yday)
    elif hasattr(value, "timetuple") and hasattr(value, "year"):
        year = int(getattr(value, "year"))
        doy = int(value.timetuple().tm_yday)
    else:
        raise TypeError(f"Cannot determine day-of-year for timestamp type {type(value)!r}.")
    is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    return doy, 366 if is_leap else 365


def _normalization_vectors(
    means: np.ndarray,
    stds: np.ndarray,
    channels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    channels = np.asarray(channels, dtype=np.int64)
    if means.ndim == 4:
        m = means.squeeze()[channels]
        s = stds.squeeze()[channels]
    elif means.ndim == 2:
        if means.shape[0] == 2 and means.shape[1] == channels.size and int(channels.max(initial=-1)) < channels.size:
            m = means[-1, channels]
            s = stds[-1, channels]
        else:
            m = means[0, channels]
            s = stds[0, channels]
    elif means.ndim == 1:
        if means.shape[0] == 2 * channels.size and int(channels.max(initial=-1)) < channels.size:
            m = means[channels.size + channels]
            s = stds[channels.size + channels]
        else:
            m = means[channels]
            s = stds[channels]
    else:
        raise ValueError(f"Unexpected normalization stats shape: {means.shape}")
    return m.astype(np.float32), s.astype(np.float32)


def _validate_stats_channels(
    means: np.ndarray,
    stds: np.ndarray,
    required_channels: np.ndarray,
) -> None:
    max_channel = int(required_channels.max()) if required_channels.size else -1
    for label, arr in (("global_means_path", means), ("global_stds_path", stds)):
        squeezed = np.asarray(arr).squeeze()
        if squeezed.ndim == 2:
            channel_count = int(squeezed.shape[-1])
        elif squeezed.ndim == 1:
            channel_count = int(squeezed.shape[0])
        elif squeezed.ndim == 4:
            channel_count = int(np.asarray(arr).shape[1])
        else:
            raise ValueError(f"Unexpected normalization stats shape for {label}: {arr.shape}")
        if max_channel >= channel_count:
            raise ValueError(
                f"{label} has {channel_count} channels, but channel index {max_channel} is required. "
                "Check resolution-specific normalization statistics."
            )


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
            required_channels = np.unique(np.concatenate([self.in_channels, self.out_channels]))
            _validate_stats_channels(self.means, self.stds, required_channels)

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
        self._discover_channel_names_and_times()
        self._build_index()

    def output_normalization_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        if self.means is None or self.stds is None:
            raise ValueError("Normalization vectors are unavailable because normalization is disabled.")
        return _normalization_vectors(self.means, self.stds, self.out_channels)

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
                    self._warn_coordinate_issues(ds, int(height), int(width), path)
                elif (height, width) != (self.img_shape_x, self.img_shape_y):
                    raise ValueError(f"Inconsistent grid in {path}: {(height, width)}")
        expected = self.config.expected_grid_shape
        if expected is not None:
            expected_shape = (int(expected[0]), int(expected[1]))
            actual_shape = (int(self.img_shape_x), int(self.img_shape_y))
            if actual_shape != expected_shape:
                raise ValueError(
                    f"Expected {self.config.resolution_mode} grid shape {expected_shape}, "
                    f"received {actual_shape}. Check resolution_mode and dataset path."
                )
        logging.info(
            "Found %d NetCDF files under %s. Spatial dims: %d x %d.",
            len(self.files_paths),
            self.data_dir,
            self.img_shape_x,
            self.img_shape_y,
        )

    def _discover_channel_names_and_times(self) -> None:
        self.channel_names: list[str] = []
        self._decoded_times_by_file: list[list[object] | None] = []
        for path in self.files_paths:
            decoded_times: list[object] | None = None
            with self.nc.Dataset(path, "r") as ds:
                if not self.channel_names and "channel" in ds.variables:
                    self.channel_names = [_decode_channel_name(x) for x in ds.variables["channel"][:]]
                if self.config.return_metadata:
                    if "time" not in ds.variables:
                        raise KeyError(
                            f"{path} has no time variable; extra temporal features require real time coordinates."
                        )
                    time_var = ds.variables["time"]
                    units = getattr(time_var, "units", None)
                    if not units:
                        raise ValueError(f"{path} time variable has no units attribute.")
                    calendar_name = getattr(time_var, "calendar", "standard")
                    decoded = self.nc.num2date(
                        np.asarray(time_var[:]),
                        units=units,
                        calendar=calendar_name,
                        only_use_cftime_datetimes=False,
                        only_use_python_datetimes=False,
                    )
                    decoded_times = list(decoded)
            self._decoded_times_by_file.append(decoded_times)
        if not self.channel_names:
            max_channel = int(max(self.in_channels.max(initial=0), self.out_channels.max(initial=0)))
            self.channel_names = [f"Var{i}" for i in range(max_channel + 1)]

    def _warn_coordinate_issues(self, ds: object, height: int, width: int, path: str) -> None:
        variables = getattr(ds, "variables", {})
        lat_key = "latitude" if "latitude" in variables else "lat" if "lat" in variables else None
        lon_key = "longitude" if "longitude" in variables else "lon" if "lon" in variables else None
        if lat_key is not None:
            latitudes = np.asarray(variables[lat_key][:], dtype=np.float64).reshape(-1)
            if latitudes.size != height:
                logging.warning("Latitude coordinate length %d does not match grid height %d in %s.", latitudes.size, height, path)
            if np.any(np.isclose(np.abs(latitudes), 90.0)):
                logging.warning("Latitude coordinates in %s include exact +/-90 degree centers.", path)
        if lon_key is not None:
            longitudes = np.asarray(variables[lon_key][:], dtype=np.float64).reshape(-1)
            if longitudes.size != width:
                logging.warning("Longitude coordinate length %d does not match grid width %d in %s.", longitudes.size, width, path)
            if np.any(np.isclose(longitudes, 0.0)) and np.any(np.isclose(longitudes, 360.0)):
                logging.warning("Longitude coordinates in %s include both 0 and 360 degrees.", path)

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

    def __getitem__(self, idx: int):
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
        input_time_indices: list[int]
        target_times: list[int]
        with self.nc.Dataset(path, "r") as ds:
            fields = ds["fields"]
            input_start = center_idx - dt * n_history
            input_stop = center_idx + 1
            input_time_indices = list(range(input_start, input_stop, dt))
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
        if self.config.return_metadata:
            decoded_times = self._decoded_times_by_file[file_idx]
            if decoded_times is None:
                raise RuntimeError("Dataset was configured to return metadata, but decoded times are unavailable.")
            target_doy: list[int] = []
            target_days: list[int] = []
            for time_idx in target_times:
                doy, days = _dayofyear_and_year_length(decoded_times[time_idx])
                target_doy.append(doy)
                target_days.append(days)
            input_doy: list[int] = []
            input_days: list[int] = []
            for time_idx in input_time_indices:
                doy, days = _dayofyear_and_year_length(decoded_times[time_idx])
                input_doy.append(doy)
                input_days.append(days)
            return {
                "input": inp,
                "target": tar,
                "input_indices": torch.as_tensor(input_time_indices, dtype=torch.long),
                "target_indices": torch.as_tensor(target_times, dtype=torch.long),
                "input_dayofyear": torch.as_tensor(input_doy, dtype=torch.long),
                "input_days_in_year": torch.as_tensor(input_days, dtype=torch.long),
                "target_dayofyear": torch.as_tensor(target_doy, dtype=torch.long),
                "target_days_in_year": torch.as_tensor(target_days, dtype=torch.long),
                "center_index": torch.as_tensor(center_idx, dtype=torch.long),
            }
        return inp, tar


def build_data_loader(
    config: DataConfig,
    data_dir: str,
    train: bool = True,
) -> tuple[DataLoader, ClimateNetCDFDataset]:
    dataset = ClimateNetCDFDataset(config=config, data_dir=data_dir, train=train)
    num_workers = int(config.num_workers)
    persistent_workers = bool(config.persistent_workers) and num_workers > 0
    loader_kwargs = {
        "batch_size": int(config.batch_size),
        "shuffle": bool(train),
        "num_workers": num_workers,
        "drop_last": bool(train),
        "pin_memory": bool(config.pin_memory),
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0 and config.prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = int(config.prefetch_factor)
    loader = DataLoader(
        dataset,
        **loader_kwargs,
    )
    logging.info(
        "DataLoader %s | target_rollout_steps=%d | batch_size=%d | num_workers=%d | "
        "pin_memory=%s | persistent_workers=%s | prefetch_factor=%s | drop_last=%s",
        "train" if train else "eval",
        int(config.rollout_steps),
        int(config.batch_size),
        num_workers,
        bool(config.pin_memory),
        persistent_workers,
        config.prefetch_factor if num_workers > 0 else None,
        bool(train),
    )
    return loader, dataset
