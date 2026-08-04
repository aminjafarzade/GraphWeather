from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.trainer import Trainer  # noqa: E402


# The real ERA5-67 2.5-degree channel order (67 channels: 6 surface + 5 vars x
# 12 pressure levels + orog). Used so the back-compat assertions run against the
# exact vector production builds.
ERA5_67_CHANNELS = [
    "t2m", "msl", "sp", "tcwv", "skt", "tisr",
    "u1000", "u925", "u850", "u800", "u700", "u600",
    "u500", "u400", "u300", "u200", "u100", "u50",
    "v1000", "v925", "v850", "v800", "v700", "v600",
    "v500", "v400", "v300", "v200", "v100", "v50",
    "t1000", "t925", "t850", "t800", "t700", "t600",
    "t500", "t400", "t300", "t200", "t100", "t50",
    "q1000", "q925", "q850", "q800", "q700", "q600",
    "q500", "q400", "q300", "q200", "q100", "q50",
    "z1000", "z925", "z850", "z800", "z700", "z600",
    "z500", "z400", "z300", "z200", "z100", "z50",
    "orog",
]


def legacy_weights(channel_names: list[str], num_channels: int, upweights=None) -> list[float]:
    """Independent re-implementation of the PRE-CHANGE pressure branch.

    Deliberately a separate copy of the old arithmetic (weight = level /
    mean_level, surface channels = 1.0) so the back-compat test compares the new
    code against the old formula rather than against itself.
    """
    import re

    upweights = dict(upweights or {})
    names = [str(x) for x in channel_names]
    levels = []
    for name in names:
        m = re.match(r"^[A-Za-z_]+(\d+)$", name)
        levels.append(int(m.group(1)) if m else None)
    present = sorted({lv for lv in levels if lv is not None})
    mean_level = (sum(present) / len(present)) if present else 1.0
    weights = [1.0] * num_channels
    for i, lv in enumerate(levels):
        if lv is not None:
            weights[i] = float(lv) / float(mean_level)
    for i, name in enumerate(names):
        if name in upweights:
            weights[i] *= float(upweights[name])
    return weights


def build(cfg: dict, channel_names=None, num_channels=None):
    """Call the real method on a bare Trainer shell (repo test convention)."""
    names = list(channel_names if channel_names is not None else ERA5_67_CHANNELS)
    n = int(num_channels if num_channels is not None else len(names))
    trainer = object.__new__(Trainer)
    trainer.loss_channel_weight_cfg = dict(cfg)
    trainer.params = {}
    weights = Trainer._build_loss_channel_weights(trainer, channel_names=names, num_channels=n)
    return weights, trainer


class LossChannelWeightBackCompatTest(unittest.TestCase):
    """min_level_weight=0.0 + reference_level=0.0 must be a no-op."""

    def test_explicit_zero_defaults_reproduce_legacy_weights(self) -> None:
        cfg = {
            "enabled": True,
            "pressure_weighting": True,
            "min_level_weight": 0.0,
            "reference_level": 0.0,
            "variable_upweights": {},
        }
        got, _ = build(cfg)
        want = legacy_weights(ERA5_67_CHANNELS, len(ERA5_67_CHANNELS))
        self.assertEqual(len(got), len(want))
        # exact equality, not almost-equal: the arithmetic must be untouched
        self.assertEqual(got, want)

    def test_absent_keys_reproduce_legacy_weights(self) -> None:
        """The control configs omit both keys entirely -- that path must not move."""
        cfg = {"enabled": True, "pressure_weighting": True, "variable_upweights": {}}
        got, _ = build(cfg)
        want = legacy_weights(ERA5_67_CHANNELS, len(ERA5_67_CHANNELS))
        self.assertEqual(got, want)

    def test_zero_defaults_reproduce_legacy_weights_with_upweights(self) -> None:
        """variable_upweights must still compose exactly as before."""
        ups = {"t2m": 2.0, "z500": 2.0}
        cfg = {
            "enabled": True,
            "pressure_weighting": True,
            "min_level_weight": 0.0,
            "reference_level": 0.0,
            "variable_upweights": ups,
        }
        got, _ = build(cfg)
        want = legacy_weights(ERA5_67_CHANNELS, len(ERA5_67_CHANNELS), upweights=ups)
        self.assertEqual(got, want)

    def test_disabled_still_returns_none(self) -> None:
        weights, _ = build({"enabled": False, "min_level_weight": 0.2, "reference_level": 1000})
        self.assertIsNone(weights)

    def test_pressure_weighting_off_leaves_surface_scale(self) -> None:
        cfg = {
            "enabled": True,
            "pressure_weighting": False,
            "min_level_weight": 0.2,
            "reference_level": 1000,
            "variable_upweights": {},
        }
        got, _ = build(cfg)
        self.assertEqual(got, [1.0] * len(ERA5_67_CHANNELS))


class LossChannelWeightFloorTest(unittest.TestCase):
    """min_level_weight / reference_level behaviour."""

    # The contract table as specified, to 2 decimals: weight = max(0.2, lv/1000).
    # 925 -> 0.925 exactly, which renders as "0.93"; both are asserted below.
    EXPECTED_DISPLAY = {
        50: "0.20", 100: "0.20", 200: "0.20", 250: "0.25", 300: "0.30", 400: "0.40",
        500: "0.50", 600: "0.60", 700: "0.70", 850: "0.85", 925: "0.93", 1000: "1.00",
    }

    def test_floor_with_reference_level_1000_table(self) -> None:
        levels = sorted(self.EXPECTED_DISPLAY)
        names = [f"z{lv}" for lv in levels]
        cfg = {
            "enabled": True,
            "pressure_weighting": True,
            "min_level_weight": 0.2,
            "reference_level": 1000,
            "variable_upweights": {},
        }
        got, _ = build(cfg, channel_names=names)
        for lv, w in zip(levels, got):
            # exact float identity with the documented formula
            self.assertEqual(w, max(0.2, lv / 1000.0), msg=f"level {lv} exact")
            # and the 2-decimal rendering matches the spec table
            self.assertEqual(f"{w:.2f}", self.EXPECTED_DISPLAY[lv], msg=f"level {lv} display")

    def test_reference_level_makes_weights_independent_of_level_set(self) -> None:
        """With reference_level>0 there is no mean normalization, so dropping
        levels must not move the surviving weights."""
        cfg = {
            "enabled": True, "pressure_weighting": True,
            "min_level_weight": 0.2, "reference_level": 1000, "variable_upweights": {},
        }
        full, _ = build(cfg, channel_names=["z1000", "z500", "z50"])
        subset, _ = build(cfg, channel_names=["z1000", "z500"])
        self.assertEqual(full[:2], subset)

    def test_floor_clamps_only_below_threshold(self) -> None:
        cfg = {
            "enabled": True, "pressure_weighting": True,
            "min_level_weight": 0.5, "reference_level": 1000, "variable_upweights": {},
        }
        got, _ = build(cfg, channel_names=["z100", "z500", "z1000"])
        self.assertEqual(got, [0.5, 0.5, 1.0])

    def test_reference_level_zero_falls_back_to_mean_level(self) -> None:
        """reference_level=0 keeps mean-level normalization; the floor still applies."""
        names = ["z100", "z500", "z1000"]
        mean_level = (100 + 500 + 1000) / 3.0
        cfg = {
            "enabled": True, "pressure_weighting": True,
            "min_level_weight": 0.4, "reference_level": 0.0, "variable_upweights": {},
        }
        got, _ = build(cfg, channel_names=names)
        want = [max(0.4, lv / mean_level) for lv in (100, 500, 1000)]
        self.assertEqual(got, want)

    def test_floor_applied_before_upweights(self) -> None:
        """The floor bounds the level term; upweights still multiply on top."""
        cfg = {
            "enabled": True, "pressure_weighting": True,
            "min_level_weight": 0.2, "reference_level": 1000,
            "variable_upweights": {"z50": 3.0},
        }
        got, _ = build(cfg, channel_names=["z50", "z1000"])
        self.assertEqual(got, [0.2 * 3.0, 1.0])


class LossChannelWeightProvenanceTest(unittest.TestCase):
    def test_resolved_keys_persisted_into_params(self) -> None:
        """Both keys must reach config_resolved.yaml (dumped from params)."""
        cfg = {
            "enabled": True, "pressure_weighting": True,
            "min_level_weight": 0.2, "reference_level": 1000, "variable_upweights": {},
        }
        _, trainer = build(cfg)
        persisted = trainer.params["loss_channel_weighting"]
        self.assertEqual(persisted["min_level_weight"], 0.2)
        self.assertEqual(persisted["reference_level"], 1000.0)

    def test_defaults_persisted_when_keys_absent(self) -> None:
        cfg = {"enabled": True, "pressure_weighting": True, "variable_upweights": {}}
        _, trainer = build(cfg)
        persisted = trainer.params["loss_channel_weighting"]
        self.assertEqual(persisted["min_level_weight"], 0.0)
        self.assertEqual(persisted["reference_level"], 0.0)

    def test_weight_table_is_logged_and_greppable(self) -> None:
        cfg = {
            "enabled": True, "pressure_weighting": True,
            "min_level_weight": 0.2, "reference_level": 1000, "variable_upweights": {},
        }
        with self.assertLogs(level="INFO") as captured:
            build(cfg, channel_names=["z1000", "z50", "t2m"])
        text = "\n".join(captured.output)
        self.assertIn("LOSS_CHANNEL_WEIGHT", text)
        self.assertIn("min_level_weight=0.2000", text)
        self.assertIn("reference_level=1000.0000", text)
        # grouped by variable, and surface channels labelled as such
        self.assertIn("var=t2m", text)
        self.assertIn("level=surface", text)
        self.assertIn("var=z", text)
        # one table row per channel
        self.assertEqual(sum("LOSS_CHANNEL_WEIGHT" in ln for ln in captured.output), 3)


if __name__ == "__main__":
    unittest.main()
