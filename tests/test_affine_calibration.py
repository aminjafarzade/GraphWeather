from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.affine_calibration import (  # noqa: E402
    apply_affine_calibration_to_tensor,
    empty_affine_accumulator,
    finalize_affine_coefficients,
    update_affine_accumulator,
    validate_fit_split,
)
from src.evaluator import EvalConfig  # noqa: E402


class AffineCalibrationMathTest(unittest.TestCase):
    def test_synthetic_affine_recovery(self) -> None:
        pred = torch.linspace(-2.0, 3.0, steps=2 * 1 * 3 * 4, dtype=torch.float32).reshape(2, 1, 3, 4)
        target = 2.0 * pred + 3.0
        accumulator = empty_affine_accumulator(n_leads=1, n_variables=1)
        update_affine_accumulator(
            accumulator,
            lead_idx=0,
            pred_norm=pred,
            target_norm=target,
            lat_weights=torch.tensor([0.5, 1.0, 0.5], dtype=torch.float32),
            variable_indices=[0],
        )
        payload = finalize_affine_coefficients(
            accumulator,
            [{"name": "x", "canonical_name": "x", "variable_idx": 0, "channel": 0}],
            {"fit_split": "valid", "eval_split": "test", "fixed_rollout_steps": 1},
        )
        coeff = payload["coefficients"]["x"]["1"]
        self.assertAlmostEqual(float(coeff["a"]), 2.0, places=6)
        self.assertAlmostEqual(float(coeff["b"]), 3.0, places=6)
        self.assertLess(float(coeff["fit_rmse_after"]), 1.0e-6)

    def test_no_leakage_rejects_test_fit_by_default(self) -> None:
        with self.assertRaisesRegex(ValueError, "Refusing to fit"):
            validate_fit_split("test")
        validate_fit_split("test", allow_fit_on_test=True)

    def test_shape_has_each_variable_and_lead(self) -> None:
        n_vars = 6
        n_leads = 10
        accumulator = empty_affine_accumulator(n_leads=n_leads, n_variables=n_vars)
        pred = torch.randn(1, n_vars, 2, 3)
        target = pred + 0.1
        for lead_idx in range(n_leads):
            update_affine_accumulator(
                accumulator,
                lead_idx=lead_idx,
                pred_norm=pred,
                target_norm=target,
                lat_weights=torch.ones(2),
                variable_indices=list(range(n_vars)),
            )
        variables = [
            {"name": f"v{i}", "canonical_name": f"v{i}", "variable_idx": i, "channel": i}
            for i in range(n_vars)
        ]
        payload = finalize_affine_coefficients(
            accumulator,
            variables,
            {"fit_split": "valid", "eval_split": "test", "fixed_rollout_steps": n_leads},
        )
        self.assertEqual(sorted(payload["coefficients"]), [f"v{i}" for i in range(n_vars)])
        for variable in variables:
            self.assertEqual(sorted(int(k) for k in payload["coefficients"][variable["canonical_name"]]), list(range(1, 11)))


class AffineCalibrationApplicationTest(unittest.TestCase):
    def test_output_only_application_does_not_mutate_rollout_prediction(self) -> None:
        pred = torch.ones(1, 2, 2, 2)
        resolved = {0: {1: (2.0, 3.0, "x")}}
        calibrated = apply_affine_calibration_to_tensor(pred, 1, resolved, in_place=False)
        self.assertTrue(torch.equal(pred, torch.ones_like(pred)))
        self.assertTrue(torch.equal(calibrated[:, 0], torch.full_like(calibrated[:, 0], 5.0)))
        self.assertTrue(torch.equal(calibrated[:, 1], torch.ones_like(calibrated[:, 1])))

    def test_autoregressive_state_application_uses_calibrated_next_state(self) -> None:
        pred_next = torch.ones(1, 1, 2, 2)
        resolved = {0: {1: (2.0, 3.0, "x")}}
        next_current = torch.zeros_like(pred_next)
        next_current[:, :1] = apply_affine_calibration_to_tensor(pred_next, 1, resolved, in_place=False)
        self.assertTrue(torch.equal(next_current[:, 0], torch.full_like(next_current[:, 0], 5.0)))

    def test_eval_config_default_keeps_normal_evaluation_uncalibrated(self) -> None:
        cfg = EvalConfig.from_params(SimpleNamespace())
        self.assertIsNone(cfg.affine_calibration_path)
        self.assertEqual(cfg.calibration_apply_mode, "output_only")


if __name__ == "__main__":
    unittest.main()
