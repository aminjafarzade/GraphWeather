from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_zone_rmse_dominance import (  # noqa: E402
    ZONE_ORDER,
    _add_skill,
    _empty_accumulator,
    _finalize_metrics,
    _zone_masks,
    _zone_metadata,
)


class ZoneRmseDominanceMathTest(unittest.TestCase):
    def test_zone_masks_use_actual_latitudes(self) -> None:
        self.assertEqual(
            ZONE_ORDER,
            [
                "90S-75S",
                "75S-60S",
                "60S-45S",
                "45S-30S",
                "30S-15S",
                "15S-0",
                "0-15N",
                "15N-30N",
                "30N-45N",
                "45N-60N",
                "60N-75N",
                "75N-90N",
            ],
        )
        lats = np.asarray([-90.0, -75.0, -60.0, -0.1, 0.0, 14.9, 75.0, 90.0])
        masks = _zone_masks(lats)
        self.assertTrue(masks["90S-75S"][0])
        self.assertTrue(masks["75S-60S"][1])
        self.assertTrue(masks["60S-45S"][2])
        self.assertTrue(masks["15S-0"][3])
        self.assertTrue(masks["0-15N"][4])
        self.assertTrue(masks["0-15N"][5])
        self.assertTrue(masks["75N-90N"][6])
        self.assertTrue(masks["75N-90N"][7])
        coverage = sum(mask.astype(np.int64) for mask in masks.values())
        np.testing.assert_array_equal(coverage, np.ones_like(coverage))

    def test_dominance_percent_sums_to_100(self) -> None:
        lats = np.asarray([-82.5, -67.5, -52.5, -37.5, -22.5, -7.5, 7.5, 22.5, 37.5, 52.5, 67.5, 82.5])
        weights = np.ones_like(lats)
        meta = _zone_metadata(lats, weights, width=2)
        acc = _empty_accumulator(n_leads=2, n_vars=1)
        zone_sse = np.arange(1.0, len(ZONE_ORDER) + 1.0)
        acc["global_sse"][:] = np.asarray([[float(zone_sse.sum())], [float((2.0 * zone_sse).sum())]])
        acc["global_weight"][:] = float(len(ZONE_ORDER) * 2)
        for zone_idx, zone in enumerate(ZONE_ORDER):
            acc["zones"][zone]["sse"][:] = np.asarray([[zone_sse[zone_idx]], [2.0 * zone_sse[zone_idx]]])
            acc["zones"][zone]["weight"][:] = 2.0

        metrics = _finalize_metrics(acc, ["x"], meta)
        for lead_idx in range(2):
            total = sum(metrics["x"]["zones"][zone]["dominance_percent_by_lead"][lead_idx] for zone in ZONE_ORDER)
            self.assertAlmostEqual(total, 100.0)
        self.assertEqual(metrics["x"]["dominant_zone_day10"], "75N-90N")

    def test_skill_uses_mse_not_rmse(self) -> None:
        model_zones = {zone: {"mse_by_lead": [2.0]} for zone in ZONE_ORDER}
        persistence_zones = {zone: {"mse_by_lead": [4.0]} for zone in ZONE_ORDER}
        model_zones["75S-60S"] = {"mse_by_lead": [1.0]}
        persistence_zones["75S-60S"] = {"mse_by_lead": [4.0]}
        model_zones["75N-90N"] = {"mse_by_lead": [4.0]}
        persistence_zones["75N-90N"] = {"mse_by_lead": [2.0]}
        model = {
            "x": {
                "global_mse_by_lead": [4.0],
                "zones": model_zones,
            }
        }
        persistence = {
            "x": {
                "global_mse_by_lead": [8.0],
                "zones": persistence_zones,
            }
        }
        _add_skill(model, persistence)
        self.assertAlmostEqual(model["x"]["global_skill_by_lead"][0], 0.5)
        self.assertAlmostEqual(model["x"]["zones"]["75S-60S"]["skill_by_lead"][0], 0.75)
        self.assertAlmostEqual(model["x"]["zones"]["75N-90N"]["skill_by_lead"][0], -1.0)


if __name__ == "__main__":
    unittest.main()
