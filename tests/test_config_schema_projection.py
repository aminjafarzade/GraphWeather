"""Faithfulness tests for the additive typed projection ``src/config_schema.py``.

Proves the projection is a lossless read-model over the resolved config dict:
section views round-trip (``view.to_dict() == flat[section]``) and typed
accessors return the resolved values -- across every real config fixture plus a
synthetic case. Nothing here touches production code; ``config_schema`` is not
wired into ``YParams`` yet.

Run:  python -m unittest tests.test_config_schema_projection
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402

from src.config import YParams  # noqa: E402
from src.config_schema import AppConfig  # noqa: E402

MODES = [None, "2p5", "5p625"]


def _resolvable_configs():
    files = sorted((PROJECT_ROOT / "configs").glob("*.yaml"))
    files += sorted((PROJECT_ROOT / "configs" / "experiments").glob("*.yaml"))
    out = []
    for path in files:
        root = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(root, dict):
            continue
        for section in [k for k, v in root.items() if isinstance(v, dict)]:
            for mode in MODES:
                try:
                    yp = YParams(str(path), section, resolution_mode=mode)
                except Exception:
                    continue
                out.append((path.name, section, mode, yp))
    return out


class ProjectionFaithfulnessTest(unittest.TestCase):
    def test_projection_is_lossless_over_all_real_configs(self):
        fixtures = _resolvable_configs()
        self.assertGreater(len(fixtures), 0, "no resolvable configs found")
        for name, section, mode, yp in fixtures:
            msg = "%s::%s::%s" % (name, section, mode)
            app = AppConfig.from_yparams(yp)
            flat = yp.params
            with self.subTest(cfg=msg):
                # flat is the single source of truth (same object, verbatim)
                self.assertIs(app.flat, flat, msg)

                # section views round-trip losslessly
                self.assertEqual(app.model.to_dict(), flat.get("model") or {}, msg)
                self.assertEqual(app.target_handling.to_dict(), flat.get("target_handling") or {}, msg)
                self.assertEqual(app.diagnostics.to_dict(), flat.get("diagnostics") or {}, msg)
                self.assertEqual(app.rollout.random_rollout.to_dict(), flat.get("random_rollout") or {}, msg)
                self.assertEqual(app.rollout.scheduled_rollout.to_dict(), flat.get("scheduled_rollout") or {}, msg)

                # nested model sub-sections round-trip
                m = flat.get("model") or {}
                self.assertEqual(app.model.skip_fusion.to_dict(), m.get("skip_fusion") or {}, msg)
                self.assertEqual(app.model.pooling.to_dict(), m.get("pooling") or {}, msg)
                self.assertEqual(app.model.lead_conditioning.to_dict(), m.get("lead_conditioning") or {}, msg)
                if isinstance(m.get("l0_refine"), dict):
                    self.assertEqual(app.model.l0_refine.to_dict(), m["l0_refine"], msg)
                else:
                    self.assertIsNone(app.model.l0_refine, msg)

                # typed accessors equal the raw resolved values when present
                for k in ("hidden_dim", "num_heads", "use_l3", "num_graph_levels", "level_k_neighbors"):
                    if k in m:
                        self.assertEqual(getattr(app.model, k), m[k], "%s / model.%s" % (msg, k))
                for k in ("rollout_mode", "fixed_train_rollout_steps", "load_only_current_rollout"):
                    if k in flat:
                        self.assertEqual(getattr(app.rollout, k), flat[k], "%s / rollout.%s" % (msg, k))
                th = flat.get("target_handling") or {}
                if "enabled" in th:
                    self.assertEqual(app.target_handling.enabled, th["enabled"], msg)

                # dict-style delegation matches flat exactly
                for k in list(flat)[:6]:
                    self.assertEqual(app.get(k), flat[k], msg)
                    self.assertIn(k, app)


class ProjectionSemanticsTest(unittest.TestCase):
    def test_synthetic_projection(self):
        flat = {
            "model": {
                "hidden_dim": 128,
                "num_heads": 4,
                "skip_fusion": {"type": "gated", "init_scale": 0.5},
                "l0_refine": {"type": "nodewise_mlp", "mlp_expansion": 2},
            },
            "rollout_mode": "random",
            "fixed_train_rollout_steps": 1,
            "random_rollout": {"min_horizon": 1, "max_horizon": 5},
            "target_handling": {"enabled": True, "copy_variables": ["orog"]},
            "diagnostics": {"enabled": False, "baseline_compare": {"enabled": True}},
            "batch_size": 4,
        }
        app = AppConfig.from_resolved(flat)
        self.assertEqual(app.model.hidden_dim, 128)
        self.assertEqual(app.model.num_heads, 4)
        self.assertEqual(app.model.skip_fusion.type, "gated")
        self.assertEqual(app.model.skip_fusion.init_scale, 0.5)
        self.assertEqual(app.model.l0_refine.type, "nodewise_mlp")
        self.assertEqual(app.rollout.rollout_mode, "random")
        self.assertEqual(app.rollout.random_rollout.max_horizon, 5)
        self.assertEqual(app.target_handling.copy_variables, ["orog"])
        self.assertIs(app.diagnostics.enabled, False)
        self.assertIs(app.diagnostics.baseline_compare.enabled, True)
        self.assertEqual(app.data.batch_size, 4)
        self.assertEqual(app.model.to_dict(), flat["model"])  # lossless

    def test_defaults_and_absent_sections(self):
        app = AppConfig.from_resolved({"model": {}})
        self.assertIsNone(app.model.l0_refine)  # absent -> None
        self.assertEqual(app.model.skip_fusion.type, "default")  # default when absent
        self.assertEqual(app.model.pooling.mean_type, "mean")
        self.assertEqual(app.rollout.rollout_mode, "curriculum")
        self.assertEqual(app.diagnostics.to_dict(), {})


if __name__ == "__main__":
    unittest.main()
