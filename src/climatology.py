from __future__ import annotations

import glob
import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def load_netcdf4():
    try:
        import netCDF4 as nc
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "netCDF4 is required for calendar-aware climatology and evaluation. "
            "Install requirements.txt in your environment."
        ) from exc
    return nc


def find_nc_files(path: str | os.PathLike[str]) -> list[str]:
    path = str(path)
    if os.path.isfile(path):
        return [path]
    files = sorted(glob.glob(os.path.join(path, "*.nc")))
    if not files:
        raise FileNotFoundError(f"No .nc files found under {path}")
    return files


def decode_time_values(ds: Any) -> list[Any]:
    if "time" not in ds.variables:
        raise KeyError("Dataset has no 'time' variable; calendar-aware evaluation requires real time coordinates.")
    nc = load_netcdf4()
    time_var = ds.variables["time"]
    units = getattr(time_var, "units", None)
    if not units:
        raise ValueError("Dataset time variable has no units attribute; cannot decode real timestamps.")
    calendar = getattr(time_var, "calendar", "standard")
    values = np.asarray(time_var[:])
    decoded = nc.num2date(
        values,
        units=units,
        calendar=calendar,
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=False,
    )
    return list(decoded)


def dayofyear(value: Any) -> int:
    if isinstance(value, np.datetime64):
        py_value = value.astype("datetime64[ms]").astype(datetime)
        return int(py_value.timetuple().tm_yday)
    if hasattr(value, "timetuple"):
        return int(value.timetuple().tm_yday)
    if hasattr(value, "dayofyr"):
        return int(value.dayofyr)
    if hasattr(value, "dayofyear"):
        return int(value.dayofyear)
    raise TypeError(f"Cannot determine day-of-year for timestamp type {type(value)!r}.")


def date_from_time(value: Any) -> date:
    if isinstance(value, np.datetime64):
        py_value = value.astype("datetime64[ms]").astype(datetime)
        return py_value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if all(hasattr(value, name) for name in ("year", "month", "day")):
        return date(int(value.year), int(value.month), int(value.day))
    raise TypeError(f"Cannot determine date for timestamp type {type(value)!r}.")


def format_time(value: Any | None) -> str | None:
    if value is None:
        return None
    d = date_from_time(value)
    return d.isoformat()


def parse_date(value: str | None) -> date | None:
    if value is None or str(value).strip() == "":
        return None
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def decode_channel_names(ds: Any) -> list[str]:
    if "channel" not in ds.variables:
        return []
    raw = ds.variables["channel"][:]
    names: list[str] = []
    for item in raw:
        if isinstance(item, bytes):
            names.append(item.decode("utf-8"))
        else:
            names.append(str(item))
    return names


@dataclass
class DayOfYearClimatology:
    values: np.ndarray
    dayofyears: np.ndarray
    counts: np.ndarray | None
    metadata: dict[str, Any]

    @property
    def available_dayofyears(self) -> set[int]:
        if self.counts is None:
            return {int(day) for day in self.dayofyears}
        return {int(day) for day, count in zip(self.dayofyears, self.counts) if int(count) > 0}


def _json_attr(value: Any, default: Any) -> Any:
    if value is None:
        return default
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return default


def load_dayofyear_climatology(
    path: str | os.PathLike[str],
    out_channels: Iterable[int],
    height: int,
    width: int,
) -> DayOfYearClimatology:
    path = str(path)
    out_channels_list = [int(x) for x in out_channels]
    if path.endswith(".npz"):
        payload = np.load(path)
        values = payload["climatology"].astype(np.float32)
        dayofyears = payload["dayofyear"].astype(np.int16) if "dayofyear" in payload else np.arange(1, values.shape[0] + 1, dtype=np.int16)
        counts = payload["count"].astype(np.int64) if "count" in payload else None
        metadata = {"path": path, "calendar_aware": bool("dayofyear" in payload), "format": "npz"}
    elif path.endswith(".npy"):
        values = np.load(path).astype(np.float32)
        dayofyears = np.arange(1, values.shape[0] + 1, dtype=np.int16)
        counts = None
        metadata = {"path": path, "calendar_aware": False, "format": "npy"}
    else:
        nc = load_netcdf4()
        with nc.Dataset(path, "r") as ds:
            key = "climatology" if "climatology" in ds.variables else next(iter(ds.variables))
            values = np.asarray(ds[key][:], dtype=np.float32)
            dayofyears = np.asarray(ds["dayofyear"][:], dtype=np.int16) if "dayofyear" in ds.variables else np.arange(1, values.shape[0] + 1, dtype=np.int16)
            counts = np.asarray(ds["count"][:], dtype=np.int64) if "count" in ds.variables else None
            metadata = {
                "path": path,
                "format": "netcdf",
                "calendar_aware": bool(getattr(ds, "calendar_aware", "false") == "true"),
                "dayofyear_from_real_time_coordinate": bool(
                    getattr(ds, "dayofyear_from_real_time_coordinate", "false") == "true"
                ),
                "source_split": getattr(ds, "source_split", None),
                "source_files": _json_attr(getattr(ds, "source_files", None), []),
                "source_years": _json_attr(getattr(ds, "source_years", None), []),
                "variable_names": _json_attr(getattr(ds, "variable_names", None), []),
                "source_channels": _json_attr(getattr(ds, "source_channels", None), []),
            }

    if values.ndim != 4:
        raise ValueError(f"Expected climatology with dims [dayofyear, channel, lat, lon], got {values.shape}")
    if values.shape[-2] < height or values.shape[-1] < width:
        raise ValueError(f"Climatology grid {values.shape[-2:]} is smaller than evaluation grid {(height, width)}.")

    if values.shape[1] > len(out_channels_list):
        max_channel = max(out_channels_list) if out_channels_list else -1
        if max_channel >= values.shape[1]:
            raise ValueError(
                f"Climatology has {values.shape[1]} channels but output channel {max_channel} was requested."
            )
        values = values[:, out_channels_list, :height, :width]
    elif values.shape[1] == len(out_channels_list):
        values = values[:, :, :height, :width]
    else:
        raise ValueError(
            f"Climatology channel count {values.shape[1]} cannot match out_channels length {len(out_channels_list)}."
        )

    metadata.setdefault("path", path)
    metadata.setdefault("calendar_aware", False)
    metadata.setdefault("dayofyear_from_real_time_coordinate", False)
    return DayOfYearClimatology(values=values.astype(np.float32), dayofyears=dayofyears, counts=counts, metadata=metadata)


def build_dayofyear_climatology(
    data_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    out_channels: Iterable[int],
    split: str,
    chunk_size: int = 32,
    logger: logging.Logger | Any = logging,
) -> dict[str, Any]:
    nc = load_netcdf4()
    files = find_nc_files(data_path)
    out_channels_list = [int(x) for x in out_channels]
    if not out_channels_list:
        raise ValueError("out_channels is required to build climatology.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    with nc.Dataset(files[0], "r") as first:
        fields = first["fields"]
        _, total_channels, height, width = fields.shape
        channel_names_all = decode_channel_names(first)
        lat_key = "latitude" if "latitude" in first.variables else "lat" if "lat" in first.variables else None
        lon_key = "longitude" if "longitude" in first.variables else "lon" if "lon" in first.variables else None
        latitudes = np.asarray(first.variables[lat_key][:], dtype=np.float32) if lat_key else np.arange(height, dtype=np.float32)
        longitudes = np.asarray(first.variables[lon_key][:], dtype=np.float32) if lon_key else np.arange(width, dtype=np.float32)
    max_channel = max(out_channels_list)
    if max_channel >= int(total_channels):
        raise ValueError(f"Requested output channel {max_channel}, but dataset has {total_channels} channels.")

    n_days = 366
    n_channels = len(out_channels_list)
    clim_sum = np.zeros((n_days, n_channels, int(height), int(width)), dtype=np.float64)
    clim_count = np.zeros((n_days,), dtype=np.int64)
    source_years: set[int] = set()
    total_steps = 0

    for file_idx, file_path in enumerate(files, start=1):
        with nc.Dataset(file_path, "r") as ds:
            fields = ds["fields"]
            times = decode_time_values(ds)
            if len(times) != fields.shape[0]:
                raise ValueError(f"Time length {len(times)} does not match fields length {fields.shape[0]} in {file_path}.")
            source_years.update(int(date_from_time(t).year) for t in times)
            for start in range(0, fields.shape[0], chunk_size):
                stop = min(start + chunk_size, fields.shape[0])
                chunk = np.asarray(fields[start:stop, out_channels_list, :height, :width], dtype=np.float32)
                day_indices = np.asarray([dayofyear(t) - 1 for t in times[start:stop]], dtype=np.int64)
                np.add.at(clim_sum, day_indices, chunk)
                np.add.at(clim_count, day_indices, 1)
                total_steps += int(chunk.shape[0])
        logger.info("Climatology source file %d/%d done: %s", file_idx, len(files), file_path)

    valid = clim_count > 0
    if not np.any(valid):
        raise RuntimeError("No valid timesteps found while building day-of-year climatology.")
    climatology = np.full_like(clim_sum, np.nan, dtype=np.float64)
    climatology[valid] = clim_sum[valid] / clim_count[valid].reshape(-1, 1, 1, 1)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_names = [
        channel_names_all[channel] if channel < len(channel_names_all) else f"channel_{channel}"
        for channel in out_channels_list
    ]
    with nc.Dataset(output_path, "w") as out:
        out.createDimension("dayofyear", n_days)
        out.createDimension("channel", n_channels)
        out.createDimension("latitude", int(height))
        out.createDimension("longitude", int(width))

        out.createVariable("dayofyear", "i2", ("dayofyear",))[:] = np.arange(1, n_days + 1, dtype=np.int16)
        out.createVariable("count", "i4", ("dayofyear",))[:] = clim_count.astype(np.int32)
        out.createVariable("channel", "i4", ("channel",))[:] = np.asarray(out_channels_list, dtype=np.int32)
        name_var = out.createVariable("channel_name", str, ("channel",))
        name_var[:] = np.asarray(selected_names, dtype=object)
        out.createVariable("latitude", "f4", ("latitude",))[:] = latitudes.astype(np.float32)
        out.createVariable("longitude", "f4", ("longitude",))[:] = longitudes.astype(np.float32)
        clim_var = out.createVariable(
            "climatology",
            "f4",
            ("dayofyear", "channel", "latitude", "longitude"),
            zlib=True,
            complevel=4,
            fill_value=np.float32(np.nan),
        )
        clim_var[:] = climatology.astype(np.float32)
        clim_var.long_name = "calendar day-of-year climatology from real time coordinates"

        out.calendar_aware = "true"
        out.dayofyear_from_real_time_coordinate = "true"
        out.source_split = str(split)
        out.source_path = str(data_path)
        out.source_files = json.dumps([str(path) for path in files])
        out.source_years = json.dumps(sorted(source_years))
        out.variable_names = json.dumps(selected_names)
        out.source_channels = json.dumps(out_channels_list)
        out.notes = "Climatology was grouped by decoded NetCDF time.dayofyear; no modulo timestep indexing was used."

    metadata = {
        "climatology_path": str(output_path),
        "climatology_source_split": str(split),
        "calendar_aware": True,
        "dayofyear_from_real_time_coordinate": True,
        "source_years": sorted(source_years),
        "source_files": [str(path) for path in files],
        "variable_names": selected_names,
        "source_channels": out_channels_list,
        "dayofyear_count_min": int(clim_count[valid].min()),
        "dayofyear_count_max": int(clim_count[valid].max()),
        "missing_dayofyears": [int(day) for day in np.arange(1, n_days + 1)[~valid]],
        "total_timesteps": int(total_steps),
    }
    logger.info("Saved calendar-aware day-of-year climatology: %s", output_path)
    return metadata
