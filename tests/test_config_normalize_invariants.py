"""Phase-0 safety net: micro-fixtures pinning the subtle config invariants.

Each test pins ONE normalization behaviour that a typed config schema could
silently drop. They exercise the real ``normalize_*`` functions and ``YParams``
from ``src.config`` (current code) and assert the exact present-day contract, so
if a refactor changes it the test fails loudly and specifically.

Run:  python -m unittest tests.test_config_normalize_invariants

Adds no production code; observes only.
"""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402

from src.config import (  # noqa: E402
    DEFAULT_TARGET_HANDLING,
    YParams,
    normalize_diagnostics_config_dict,
    normalize_target_handling_config_dict,
    normalize_training_config_dict,
)

# candidate (file, section, mode) triples known to fully resolve; the first that
# constructs is reused for the YParams-level fixtures (None-coercion, shim).
_RESOLVABLE_CANDIDATES = [
    ("configs/weather_dual_resolution_l3_hidden128.yaml", "raw_l3_hidden128", "2p5"),
    ("configs/weather_dual_resolution.yaml", "raw", "2p5"),
    ("configs/weather_dual_resolution_l3.yaml", "raw_l3", "2p5"),
]


def _first_resolvable():
    for rel, section, mode in _RESOLVABLE_CANDIDATES:
        path = PROJECT_ROOT / rel
        if not path.exists():
            continue
        try:
            YParams(str(path), section, resolution_mode=mode)
            return path, section, mode
        except Exception:
            continue
    return None


class TrainingRolloutInvariants(unittest.TestCase):
    def test_random_and_scheduled_force_load_only_false_on_both_surfaces(self):
        for mode in ("random", "scheduled"):
            r = normalize_training_config_dict({"rollout_mode": mode, "load_only_current_rollout": True})
            self.assertIs(r["load_only_current_rollout"], False, mode)
            self.assertIs(r["training"]["load_only_current_rollout"], False, mode)

    def test_non_forcing_mode_keeps_explicit_load_only_true(self):
        r = normalize_training_config_dict({"rollout_mode": "curriculum", "load_only_current_rollout": True})
        self.assertIs(r["load_only_current_rollout"], True)

    def test_dual_write_parity_and_int_coercion(self):
        r = normalize_training_config_dict({"training": {"rollout_mode": "fixed", "fixed_train_rollout_steps": "3"}})
        self.assertEqual(r["rollout_mode"], r["training"]["rollout_mode"])
        self.assertEqual(r["fixed_train_rollout_steps"], r["training"]["fixed_train_rollout_steps"])
        self.assertEqual(r["fixed_train_rollout_steps"], 3)
        self.assertIsInstance(r["fixed_train_rollout_steps"], int)

    def test_nested_training_takes_precedence_over_top_level(self):
        r = normalize_training_config_dict({"rollout_mode": "curriculum", "training": {"rollout_mode": "fixed"}})
        self.assertEqual(r["rollout_mode"], "fixed")

    def test_scheduled_phases_are_normalized(self):
        r = normalize_training_config_dict(
            {
                "scheduled_rollout": {
                    "phases": [
                        {"name": "a", "mode": "RANDOM", "min_horizon": "2", "max_horizon": "5"},
                        {"mode": "Fixed", "horizon": "7"},
                    ]
                }
            }
        )
        phases = r["scheduled_rollout"]["phases"]
        self.assertEqual(phases[0]["mode"], "random")
        self.assertEqual((phases[0]["min_horizon"], phases[0]["max_horizon"]), (2, 5))
        self.assertEqual(phases[1]["mode"], "fixed")
        self.assertEqual(phases[1]["horizon"], 7)


class TargetHandlingInvariants(unittest.TestCase):
    def test_missing_defaults_to_fixed_orography(self):
        r = normalize_target_handling_config_dict({})
        self.assertEqual(r["target_handling"], DEFAULT_TARGET_HANDLING)

    def test_comma_and_whitespace_string_splitting(self):
        r = normalize_target_handling_config_dict({"target_handling": {"copy_variables": "orog, lsm  foo"}})
        self.assertEqual(r["target_handling"]["copy_variables"], ["orog", "lsm", "foo"])

    def test_enabled_coerced_to_bool(self):
        r = normalize_target_handling_config_dict({"target_handling": {"enabled": 0}})
        self.assertIs(r["target_handling"]["enabled"], False)


class DiagnosticsInvariants(unittest.TestCase):
    def test_disabled_path_merges_baseline_but_discards_other_raw_keys(self):
        r = normalize_diagnostics_config_dict(
            {"diagnostics": {"enabled": False, "baseline_compare": {"enabled": True}, "rollout_horizons": [99]}}
        )
        diag = r["diagnostics"]
        self.assertIs(diag["enabled"], False)
        self.assertIs(diag["baseline_compare"]["enabled"], True)
        self.assertNotIn("rollout_horizons", diag)

    def test_enabled_path_filters_horizons_and_coerces_types(self):
        r = normalize_diagnostics_config_dict(
            {"diagnostics": {"enabled": True, "rollout_horizons": [1, 0, -2, 4], "log_every_epochs": "3"}}
        )
        diag = r["diagnostics"]
        self.assertEqual(diag["rollout_horizons"], [1, 4])
        self.assertEqual(diag["log_every_epochs"], 3)
        self.assertIsInstance(diag["log_every_epochs"], int)
        self.assertIsInstance(diag["wandb"]["enabled"], bool)
        self.assertIsInstance(diag["plots"]["enabled"], bool)


class YParamsCoercionAndShimContract(unittest.TestCase):
    """Behaviour the future thin-shim YParams must reproduce byte-for-byte."""

    @classmethod
    def setUpClass(cls):
        cls.resolvable = _first_resolvable()

    def _build_probe_yparams(self, extra: dict) -> YParams:
        assert self.resolvable is not None
        path, section, mode = self.resolvable
        root = yaml.safe_load(path.read_text(encoding="utf-8"))
        body = copy.deepcopy(root[section])
        body.update(extra)
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
        try:
            yaml.safe_dump({"probe": body}, tmp)
            tmp.close()
            return YParams(tmp.name, "probe", resolution_mode=mode)
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    def test_none_string_coerced_top_level_only(self):
        if self.resolvable is None:
            self.skipTest("no resolvable base config available")
        yp = self._build_probe_yparams(
            {"_probe_none_top": "None", "_probe_none_nested": {"inner": "None"}}
        )
        self.assertIsNone(yp["_probe_none_top"])
        self.assertEqual(yp["_probe_none_nested"]["inner"], "None")  # nested NOT coerced

    def test_access_semantics_and_identity(self):
        if self.resolvable is None:
            self.skipTest("no resolvable base config available")
        yp = self._build_probe_yparams({})
        self.assertEqual(yp["hidden_dim"], yp.hidden_dim)
        self.assertEqual(yp["hidden_dim"], yp.get("hidden_dim"))
        self.assertIn("hidden_dim", yp)
        self.assertEqual(yp.get("__definitely_absent__", 42), 42)
        self.assertIs(yp["model"], yp.model)  # dict and attr expose the SAME object

    def test_setitem_updates_both_surfaces_and_is_not_renormalized(self):
        if self.resolvable is None:
            self.skipTest("no resolvable base config available")
        yp = self._build_probe_yparams({"rollout_mode": "random"})
        # even for a forcing mode, a POST-init override sticks (no re-normalization)
        yp["load_only_current_rollout"] = True
        self.assertIs(yp["load_only_current_rollout"], True)
        self.assertIs(yp.load_only_current_rollout, True)

    def test_update_params(self):
        if self.resolvable is None:
            self.skipTest("no resolvable base config available")
        yp = self._build_probe_yparams({})
        yp.update_params({"__brand_new_key__": 7})
        self.assertEqual(yp["__brand_new_key__"], 7)
        self.assertEqual(yp.__getattribute__("__brand_new_key__"), 7)


if __name__ == "__main__":
    unittest.main()
