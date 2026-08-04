from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.solar import (  # noqa: E402
    ERA5_ACCUMULATION_SECONDS,
    SOLAR_CONSTANT_WM2,
    AnalyticSolarForcing,
    daily_mean_toa_insolation_wm2,
    eccentricity_correction_factor,
    instantaneous_toa_wm2,
    solar_declination_rad,
)
from src.target_handling import TargetHandling  # noqa: E402

ERA5_TEST_FILE = Path("/home/amin/KAI_5/era5_67/test/2018.nc")


class _NullLogger:
    def info(self, *args, **kwargs) -> None:
        return None

    def warning(self, *args, **kwargs) -> None:
        return None


class SolarGeometryTest(unittest.TestCase):
    def test_declination_stays_within_tropics_and_flips_sign(self) -> None:
        days = torch.arange(1, 366, dtype=torch.float64)
        delta = solar_declination_rad(days, torch.full_like(days, 365.0))
        max_deg = float(delta.abs().max()) * 180.0 / math.pi
        self.assertLess(max_deg, 23.6)
        self.assertGreater(max_deg, 23.2)
        # Negative (southern summer) near Jan 1, positive near Jul 1.
        self.assertLess(float(delta[0]), 0.0)
        self.assertGreater(float(delta[181]), 0.0)

    def test_eccentricity_peaks_near_perihelion(self) -> None:
        days = torch.arange(1, 366, dtype=torch.float64)
        e0 = eccentricity_correction_factor(days, torch.full_like(days, 365.0))
        self.assertLess(int(days[e0.argmax()]), 15)  # perihelion ~Jan 3
        self.assertTrue(170 < int(days[e0.argmin()]) < 200)  # aphelion ~Jul 4
        self.assertAlmostEqual(float(e0.mean()), 1.0, places=2)

    def test_daily_mean_insolation_polar_and_annual_mean(self) -> None:
        lat = torch.tensor([-88.75, 0.0, 88.75], dtype=torch.float64) * math.pi / 180.0
        january = daily_mean_toa_insolation_wm2(1.0, 365.0, lat)
        july = daily_mean_toa_insolation_wm2(182.0, 365.0, lat)
        self.assertEqual(float(january[2]), 0.0)  # north polar night
        self.assertEqual(float(july[0]), 0.0)  # south polar night
        # Polar summer daily mean exceeds the equatorial one.
        self.assertGreater(float(january[0]), float(january[1]))
        self.assertGreater(float(july[2]), float(july[1]))
        # Area-weighted annual-global mean must be ~S0/4.
        lats = torch.linspace(-89.9, 89.9, 400, dtype=torch.float64) * math.pi / 180.0
        days = torch.arange(1, 366, dtype=torch.float64)
        q = daily_mean_toa_insolation_wm2(days, torch.full_like(days, 365.0), lats)
        weights = torch.cos(lats)
        global_mean = float((q.mean(dim=0) * weights).sum() / weights.sum())
        self.assertAlmostEqual(global_mean, SOLAR_CONSTANT_WM2 / 4.0, delta=SOLAR_CONSTANT_WM2 * 0.0025)

    def test_instantaneous_toa_is_zero_at_night_and_bounded(self) -> None:
        lat = torch.linspace(-89.0, 89.0, 72, dtype=torch.float64) * math.pi / 180.0
        lon = torch.linspace(0.0, 357.5, 144, dtype=torch.float64) * math.pi / 180.0
        field = instantaneous_toa_wm2(80.0, 365.0, 12 * 3600.0, lat, lon)
        self.assertEqual(tuple(field.shape), (72, 144))
        self.assertGreaterEqual(float(field.min()), 0.0)
        self.assertLessEqual(float(field.max()), SOLAR_CONSTANT_WM2 * 1.04)
        # At 12 UTC the subsolar longitude is ~0 deg; the antimeridian is night.
        self.assertEqual(float(field[36, 72]), 0.0)
        self.assertGreater(float(field[36, 0]), 1000.0)


@unittest.skipUnless(ERA5_TEST_FILE.is_file(), "ERA5 test file not available on this machine")
class SolarVsEra5Test(unittest.TestCase):
    def test_analytic_daily_mean_matches_dataset_tisr(self) -> None:
        import netCDF4 as nc

        ds = nc.Dataset(str(ERA5_TEST_FILE))
        try:
            channels = [str(x) for x in ds.variables["channel"][:]]
            tisr_idx = channels.index("tisr")
            lat_deg = np.asarray(ds.variables["latitude"][:], dtype=np.float64)
            days = [1, 60, 120, 181, 240, 300, 355]
            data = np.stack(
                [np.asarray(ds.variables["fields"][d - 1, tisr_idx], dtype=np.float64) for d in days]
            )
        finally:
            ds.close()
        lat_rad = torch.as_tensor(lat_deg) * math.pi / 180.0
        analytic = (
            daily_mean_toa_insolation_wm2(
                torch.tensor(days, dtype=torch.float64),
                torch.full((len(days),), 365.0, dtype=torch.float64),
                lat_rad,
            )
            * ERA5_ACCUMULATION_SECONDS
        )
        analytic = analytic.unsqueeze(-1).expand(-1, -1, data.shape[-1]).numpy()
        corr = np.corrcoef(analytic.ravel(), data.ravel())[0, 1]
        rel_rmse = float(np.sqrt(np.mean((analytic - data) ** 2)) / np.mean(np.abs(data)))
        self.assertGreater(corr, 0.999)
        self.assertLess(rel_rmse, 0.05)


class AnalyticSolarForcingIntegrationTest(unittest.TestCase):
    def _handler(self) -> TargetHandling:
        params = SimpleNamespace(
            extra_features={"enabled": False},
            target_handling={
                "enabled": True,
                "copy_variables": [],
                "known_future_variables": ["tisr"],
                "exclude_loss_variables": ["tisr"],
            },
            variable_metadata={"path": None},
            global_means_path="",
            global_stds_path="",
            experiment_dir="",
        )
        return TargetHandling.from_params(
            params,
            channel_names=["t2m", "tisr", "z500"],
            out_channels=[0, 1, 2],
            logger=_NullLogger(),
        )

    def test_known_future_values_inject_normalized_tisr(self) -> None:
        forcing = AnalyticSolarForcing(
            latitudes_deg=np.array([-45.0, 0.0, 45.0]),
            n_longitudes=4,
            tisr_mean=1.0e6,
            tisr_std=5.0e5,
        )
        handler = self._handler()
        pred = torch.full((2, 3, 3, 4), 7.0)
        current = torch.zeros_like(pred)
        values = forcing.known_future_values(100.0, 365.0, batch_size=2)
        out = handler.apply(pred_next=pred, current_state=current, known_future_values=values)
        self.assertEqual(out.shape, pred.shape)
        self.assertTrue(torch.equal(out[:, 1], values["tisr"].to(out.dtype)))
        self.assertTrue(torch.equal(out[:, 0], pred[:, 0]))
        self.assertTrue(torch.equal(out[:, 2], pred[:, 2]))

    def test_missing_sources_raise_actionable_error(self) -> None:
        handler = self._handler()
        pred = torch.zeros(1, 3, 2, 2)
        with self.assertRaises(ValueError) as ctx:
            handler.apply(pred_next=pred, current_state=pred)
        self.assertIn("known_future_values", str(ctx.exception))

    def test_missing_variable_in_known_future_values_raises(self) -> None:
        handler = self._handler()
        pred = torch.zeros(1, 3, 2, 2)
        with self.assertRaises(ValueError):
            handler.apply(
                pred_next=pred,
                current_state=pred,
                known_future_values={"wrong_name": torch.zeros(1, 2, 2)},
            )


if __name__ == "__main__":
    unittest.main()
