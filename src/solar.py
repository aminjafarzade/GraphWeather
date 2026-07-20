"""Analytic top-of-atmosphere incident solar radiation (``tisr``).

REAL-INFERENCE INTERFACE for the prescribed-solar-forcing path. During training
and evaluation the true ``tisr`` for each rollout step's valid time is taken
from the target sequence (``target_handling.known_future_variables``). At real
inference time no future ERA5 exists, so the forcing must be computed from the
forecast valid time and the grid geometry instead. This module provides that
computation; pass the result to ``TargetHandling.apply(known_future_values=...)``.

``tisr`` is pure astronomy: solar constant x Earth-Sun eccentricity correction
x solar geometry. The daily ERA5-derived dataset used in this repo stores the
DAILY-MEAN top-of-atmosphere flux in ERA5's hourly-accumulation convention
(J m-2 accumulated over one hour, i.e. daily-mean W m-2 x 3600 s); the field is
longitude-independent. ``daily_mean_toa_insolation_wm2`` reproduces it up to
that fixed scale, which the z-score normalization absorbs anyway.

Solar-position terms use Spencer's (1971) Fourier series; accuracy is well
under 1% of the solar constant, far below the model's own error at any lead.
"""

from __future__ import annotations

import math

import numpy as np
import torch

SOLAR_CONSTANT_WM2 = 1361.0
# ERA5 "tisr" accumulation period: fluxes are accumulated over one hour (J m-2).
ERA5_ACCUMULATION_SECONDS = 3600.0


def _fractional_year_rad(day_of_year: torch.Tensor, days_in_year: torch.Tensor) -> torch.Tensor:
    return 2.0 * math.pi * (day_of_year - 1.0) / days_in_year


def solar_declination_rad(day_of_year: torch.Tensor, days_in_year: torch.Tensor) -> torch.Tensor:
    """Solar declination (radians) from the day of year, Spencer (1971)."""
    g = _fractional_year_rad(day_of_year, days_in_year)
    return (
        0.006918
        - 0.399912 * torch.cos(g)
        + 0.070257 * torch.sin(g)
        - 0.006758 * torch.cos(2.0 * g)
        + 0.000907 * torch.sin(2.0 * g)
        - 0.002697 * torch.cos(3.0 * g)
        + 0.00148 * torch.sin(3.0 * g)
    )


def eccentricity_correction_factor(day_of_year: torch.Tensor, days_in_year: torch.Tensor) -> torch.Tensor:
    """(r0/r)^2 Earth-Sun distance correction, Spencer (1971)."""
    g = _fractional_year_rad(day_of_year, days_in_year)
    return (
        1.000110
        + 0.034221 * torch.cos(g)
        + 0.001280 * torch.sin(g)
        + 0.000719 * torch.cos(2.0 * g)
        + 0.000077 * torch.sin(2.0 * g)
    )


def daily_mean_toa_insolation_wm2(
    day_of_year: torch.Tensor | float,
    days_in_year: torch.Tensor | float,
    latitudes_rad: torch.Tensor,
) -> torch.Tensor:
    """Daily-mean TOA insolation (W m-2) per latitude.

    Q = (S0/pi) * E0 * (w_s sin(phi) sin(delta) + cos(phi) cos(delta) sin(w_s))
    with w_s the sunset hour angle; handles polar day/night via clamping.
    Broadcasting: scalar day inputs give a [lat] result; tensor day inputs of
    shape [...] give [..., lat].
    """
    day = torch.as_tensor(day_of_year, dtype=torch.float64)
    days = torch.as_tensor(days_in_year, dtype=torch.float64)
    lat = latitudes_rad.to(dtype=torch.float64)
    delta = solar_declination_rad(day, days)
    e0 = eccentricity_correction_factor(day, days)
    delta = delta.reshape(delta.shape + (1,) * lat.dim())
    e0 = e0.reshape(e0.shape + (1,) * lat.dim())
    cos_ws = torch.clamp(-torch.tan(lat) * torch.tan(delta), -1.0, 1.0)
    ws = torch.acos(cos_ws)
    q = (SOLAR_CONSTANT_WM2 / math.pi) * e0 * (
        ws * torch.sin(lat) * torch.sin(delta) + torch.cos(lat) * torch.cos(delta) * torch.sin(ws)
    )
    return torch.clamp(q, min=0.0)


def instantaneous_toa_wm2(
    day_of_year: torch.Tensor | float,
    days_in_year: torch.Tensor | float,
    utc_seconds: torch.Tensor | float,
    latitudes_rad: torch.Tensor,
    longitudes_rad: torch.Tensor,
) -> torch.Tensor:
    """Instantaneous TOA flux S0 * E0 * max(0, cos(zenith)) on a [lat, lon] grid.

    Only needed for sub-daily timesteps; the daily dataset in this repo uses
    ``daily_mean_toa_insolation_wm2``. Hour angle uses mean solar time
    (UTC + longitude offset); the ~15 min equation-of-time wobble is ignored.
    """
    day = torch.as_tensor(day_of_year, dtype=torch.float64)
    days = torch.as_tensor(days_in_year, dtype=torch.float64)
    seconds = torch.as_tensor(utc_seconds, dtype=torch.float64)
    lat = latitudes_rad.to(dtype=torch.float64).reshape(-1, 1)
    lon = longitudes_rad.to(dtype=torch.float64).reshape(1, -1)
    delta = solar_declination_rad(day, days)
    e0 = eccentricity_correction_factor(day, days)
    solar_time_rad = 2.0 * math.pi * (seconds / 86400.0) + lon
    hour_angle = solar_time_rad - math.pi
    cos_zenith = torch.sin(lat) * torch.sin(delta) + torch.cos(lat) * torch.cos(delta) * torch.cos(hour_angle)
    return SOLAR_CONSTANT_WM2 * e0 * torch.clamp(cos_zenith, min=0.0)


class AnalyticSolarForcing:
    """Produce NORMALIZED ``tisr`` fields for ``TargetHandling.apply(known_future_values=...)``.

    Construct once from the grid latitudes and the dataset's per-channel z-score
    stats for ``tisr``, then call :meth:`known_future_values` with each rollout
    step's valid time. Example (real inference driver)::

        forcing = AnalyticSolarForcing(latitudes_deg, n_longitudes, tisr_mean, tisr_std)
        pred = target_handler.apply(
            pred_next=pred, current_state=current,
            known_future_values=forcing.known_future_values(valid_doy, days_in_year, batch_size=b),
        )
    """

    def __init__(
        self,
        latitudes_deg: np.ndarray | torch.Tensor,
        n_longitudes: int,
        tisr_mean: float,
        tisr_std: float,
        *,
        accumulation_seconds: float = ERA5_ACCUMULATION_SECONDS,
        variable: str = "tisr",
    ) -> None:
        self.latitudes_rad = torch.as_tensor(np.asarray(latitudes_deg), dtype=torch.float64) * math.pi / 180.0
        self.n_longitudes = int(n_longitudes)
        self.tisr_mean = float(tisr_mean)
        self.tisr_std = float(tisr_std)
        self.accumulation_seconds = float(accumulation_seconds)
        self.variable = str(variable)

    def physical_field(self, day_of_year: float, days_in_year: float = 365.0) -> torch.Tensor:
        """[lat, lon] field in the dataset's units (daily-mean W m-2 x accumulation period)."""
        per_lat = daily_mean_toa_insolation_wm2(day_of_year, days_in_year, self.latitudes_rad)
        field = per_lat.unsqueeze(-1).expand(-1, self.n_longitudes)
        return field * self.accumulation_seconds

    def normalized_field(
        self,
        day_of_year: float,
        days_in_year: float = 365.0,
        *,
        batch_size: int = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """[batch, lat, lon] z-score-normalized field ready for state injection."""
        field = (self.physical_field(day_of_year, days_in_year) - self.tisr_mean) / self.tisr_std
        return field.to(device=device, dtype=dtype).unsqueeze(0).expand(int(batch_size), -1, -1)

    def known_future_values(
        self,
        day_of_year: float,
        days_in_year: float = 365.0,
        *,
        batch_size: int = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> dict[str, torch.Tensor]:
        return {
            self.variable: self.normalized_field(
                day_of_year,
                days_in_year,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
        }
