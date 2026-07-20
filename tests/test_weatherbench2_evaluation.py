from __future__ import annotations

import contextlib
import csv
import io
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator import GraphWeatherEvaluator, load_external_baselines  # noqa: E402
from src.target_handling import TargetHandling  # noqa: E402
from src.weatherbench2_metrics import (  # noqa: E402
    mean_per_ic_rmse,
    rmse_from_per_ic_mse,
    weatherbench2_package_mse,
    weighted_mse_per_ic,
)


class _CaptureLogger:
    def __init__(self) -> None:
        self.infos: list[str] = []
        self.warnings: list[str] = []

    def info(self, message, *args, **kwargs) -> None:
        self.infos.append(str(message) % args if args else str(message))

    def warning(self, message, *args, **kwargs) -> None:
        self.warnings.append(str(message) % args if args else str(message))


class WeatherBench2MetricMathTest(unittest.TestCase):
    def test_sqrt_last_rmse_not_mean_per_ic_rmse(self) -> None:
        predictions = np.zeros((2, 1, 1, 2, 1), dtype=np.float64)
        targets = np.asarray([[[[[0.0], [2.0]]]], [[[[0.0], [4.0]]]]], dtype=np.float64)
        per_ic_mse = weighted_mse_per_ic(predictions, targets, latitudes_deg=np.asarray([0.0, 60.0]))

        official = rmse_from_per_ic_mse(per_ic_mse)
        legacy = mean_per_ic_rmse(per_ic_mse)
        expected = np.sqrt(np.nanmean(per_ic_mse, axis=0))

        np.testing.assert_allclose(official, expected)
        self.assertNotAlmostEqual(float(official[0, 0]), float(legacy[0, 0]))

    def test_manual_fallback_matches_weatherbench2_if_installed(self) -> None:
        predictions = np.arange(2 * 3 * 2 * 4 * 5, dtype=np.float64).reshape(2, 3, 2, 4, 5)
        targets = predictions + np.linspace(0.1, 1.3, predictions.size, dtype=np.float64).reshape(predictions.shape)
        latitudes = np.linspace(-67.5, 67.5, 4)
        manual_mse = np.nanmean(weighted_mse_per_ic(predictions, targets, latitudes_deg=latitudes), axis=0)
        try:
            package_mse = weatherbench2_package_mse(predictions, targets, latitudes, ["x", "y"])
        except ModuleNotFoundError:
            self.skipTest("weatherbench2 is not installed")
        np.testing.assert_allclose(package_mse, manual_mse, rtol=1.0e-6, atol=1.0e-10)


class WeatherBench2EvaluatorBehaviorTest(unittest.TestCase):
    def test_lead0_is_excluded_from_summary_by_default(self) -> None:
        evaluator = object.__new__(GraphWeatherEvaluator)
        evaluator.cfg = SimpleNamespace(include_lead0_in_summary=False)
        metrics = {"rmse": np.zeros((3, 1), dtype=np.float64), "lead_times": [0, 1, 2]}
        self.assertEqual(GraphWeatherEvaluator._summary_lead_indices(evaluator, metrics), [1, 2])

        evaluator.cfg = SimpleNamespace(include_lead0_in_summary=True)
        self.assertEqual(GraphWeatherEvaluator._summary_lead_indices(evaluator, metrics), [0, 1, 2])

    def test_calendar_time_coordinate_uses_actual_valid_date(self) -> None:
        evaluator = object.__new__(GraphWeatherEvaluator)
        evaluator.logger = _CaptureLogger()
        evaluator._climatology_time_fallback_warned = False
        evaluator._time_values = [datetime(2020, 12, 31), datetime(2021, 1, 1)]

        self.assertEqual(GraphWeatherEvaluator._target_dayofyear(evaluator, 1), 1)
        self.assertEqual(evaluator.logger.warnings, [])

    def test_missing_time_coordinate_warns_and_uses_index_mod_365(self) -> None:
        evaluator = object.__new__(GraphWeatherEvaluator)
        evaluator.logger = _CaptureLogger()
        evaluator._climatology_time_fallback_warned = False
        evaluator._time_values = []

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(GraphWeatherEvaluator._target_dayofyear(evaluator, 365), 1)
        self.assertIn("WARNING: no time coordinate found; using index % 365 climatology lookup.", stdout.getvalue())
        self.assertIn("WARNING: no time coordinate found; using index % 365 climatology lookup.", evaluator.logger.warnings)

    def test_rollout_uses_forward_steps_delta_path_not_direct_call(self) -> None:
        class FakeModel:
            lead_conditioning_enabled = False

            def forward_steps(self, previous, current, aux_features=None):
                return current[:, :1] + 2.0

            def __call__(self, *args, **kwargs):
                raise AssertionError("direct model call should not be used")

        evaluator = object.__new__(GraphWeatherEvaluator)
        evaluator.model = FakeModel()
        previous = torch.zeros(1, 1, 1, 1)
        current = torch.ones(1, 1, 1, 1) * 3.0
        out = GraphWeatherEvaluator._forward_model_step(evaluator, previous, current, None, lead=1)
        self.assertEqual(float(out.item()), 5.0)

    def test_fixed_orography_copy_is_available_for_weatherbench2_rollout_path(self) -> None:
        params = SimpleNamespace(
            extra_features={"enabled": False},
            target_handling={"enabled": True, "copy_variables": ["orog"], "known_future_variables": [], "exclude_loss_variables": ["orog"]},
            variable_metadata={"path": None},
            global_means_path="",
            global_stds_path="",
            experiment_dir="",
        )
        handler = TargetHandling.from_params(
            params,
            channel_names=["t2m", "orog"],
            out_channels=[0, 1],
            logger=_CaptureLogger(),
        )
        pred = torch.zeros(1, 2, 1, 1)
        pred[:, 1] = -999.0
        initial = torch.zeros_like(pred)
        initial[:, 1] = 123.0
        out = handler.apply(pred_next=pred, current_state=initial, initial_state=initial, lead=1)
        self.assertEqual(float(out[:, 1].item()), 123.0)


class ExternalKaiCsvParsingTest(unittest.TestCase):
    def _write_csv(self, path: Path, lead_column: str) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["variable", lead_column, "rmse", "acc"])
            writer.writeheader()
            writer.writerow({"variable": "z500", lead_column: 1, "rmse": 10.0, "acc": 0.9})
            writer.writerow({"variable": "z500", lead_column: 2, "rmse": 20.0, "acc": 0.8})

    def test_parse_timestep_and_lead_time_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            timestep_csv = root / "kai_timestep.csv"
            lead_time_csv = root / "kai_lead_time.csv"
            self._write_csv(timestep_csv, "timestep")
            self._write_csv(lead_time_csv, "lead_time")

            timestep = load_external_baselines([str(timestep_csv)], ["Kai"], None, fixed_steps=2, logger=_CaptureLogger())[0]
            lead_time = load_external_baselines([str(lead_time_csv)], ["Kai"], None, fixed_steps=2, logger=_CaptureLogger())[0]

        np.testing.assert_allclose(timestep.curves["z500"]["rmse"], [10.0, 20.0])
        np.testing.assert_allclose(lead_time.curves["z500"]["rmse"], [10.0, 20.0])


if __name__ == "__main__":
    unittest.main()
