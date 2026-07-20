from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import YParams  # noqa: E402
from src.resolution import canonicalize_resolution_mode, get_resolution_spec  # noqa: E402


class ResolutionSpecTest(unittest.TestCase):
    def test_aliases_and_shapes(self) -> None:
        self.assertEqual(canonicalize_resolution_mode("5.625"), "5p625")
        self.assertEqual(canonicalize_resolution_mode("5deg625"), "5p625")
        self.assertEqual(canonicalize_resolution_mode("2.5"), "2p5")
        self.assertEqual(canonicalize_resolution_mode("2deg5"), "2p5")
        self.assertEqual(get_resolution_spec("5p625").grid_shape, (32, 64))
        self.assertEqual(get_resolution_spec("2p5").grid_shape, (72, 144))
        self.assertEqual(get_resolution_spec("5p625").level_shapes, ((32, 64), (16, 32), (8, 16)))
        self.assertEqual(get_resolution_spec("2p5").level_shapes, ((72, 144), (36, 72), (18, 36)))

    def test_yaml_profile_overlay(self) -> None:
        cfg = PROJECT_ROOT / "configs" / "weather_dual_resolution.yaml"
        params = YParams(str(cfg), "raw", resolution_mode="2p5")
        self.assertEqual(params.resolution_mode, "2p5")
        self.assertEqual(tuple(params.grid_shape), (72, 144))
        self.assertEqual(params.batch_size, 4)
        self.assertEqual(params.gradient_accumulation_steps, 3)
        self.assertEqual(params.batch_size * params.gradient_accumulation_steps, 12)
        self.assertEqual(params.max_epochs, 50)
        self.assertEqual(params.lr_schedule_type, "warmup_cosine")
        self.assertAlmostEqual(params.lr, 1.0e-4)
        self.assertAlmostEqual(params.min_lr, 3.0e-6)
        self.assertEqual(params.warmup_epochs, 2)
        self.assertAlmostEqual(params.warmup_start_factor, 0.1)
        self.assertIn("graph_2p5_k8_hybrid_row_aware_v2.pt", params.graph_path)


if __name__ == "__main__":
    unittest.main()
