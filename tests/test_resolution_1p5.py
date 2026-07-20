from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import YParams  # noqa: E402
from src.resolution import (  # noqa: E402
    canonicalize_resolution_mode,
    coarsened_level_shapes,
    get_resolution_spec,
)


class Resolution1p5SpecTest(unittest.TestCase):
    def test_aliases_and_shapes(self) -> None:
        self.assertEqual(canonicalize_resolution_mode("1.5"), "1p5")
        self.assertEqual(canonicalize_resolution_mode("1p5"), "1p5")
        self.assertEqual(canonicalize_resolution_mode("1deg5"), "1p5")
        spec = get_resolution_spec("1p5")
        self.assertEqual(spec.grid_shape, (121, 240))
        self.assertEqual(spec.resolution_degrees, 1.5)
        self.assertEqual(spec.level_shapes, ((121, 240), (61, 120), (31, 60)))
        self.assertEqual(spec.node_counts, (29040, 7320, 1860))

    def test_coarsened_level_shapes(self) -> None:
        self.assertEqual(
            coarsened_level_shapes(121, 240, 4),
            [[121, 240], [61, 120], [31, 60], [16, 30]],
        )
        self.assertEqual(
            coarsened_level_shapes(121, 240, 5),
            [[121, 240], [61, 120], [31, 60], [16, 30], [8, 15]],
        )


class Resolution1p5ConfigTest(unittest.TestCase):
    def _load(self, yaml_name: str, config_name: str) -> YParams:
        cfg = PROJECT_ROOT / "configs" / yaml_name
        return YParams(str(cfg), config_name)

    def _assert_common(self, params: YParams) -> None:
        self.assertEqual(params.resolution_mode, "1p5")
        self.assertEqual(tuple(params.grid_shape), (121, 240))
        self.assertAlmostEqual(params.lr, 1.0e-4)
        self.assertAlmostEqual(params.min_lr, 3.0e-6)
        self.assertAlmostEqual(params.weight_decay, 1.0e-4)
        self.assertEqual(params.lr_schedule_type, "warmup_cosine")
        self.assertEqual(params.warmup_epochs, 2)
        self.assertAlmostEqual(params.warmup_start_factor, 0.1)
        self.assertEqual(params.batch_size, 4)
        self.assertEqual(params.gradient_accumulation_steps, 3)
        self.assertAlmostEqual(params.max_gradient_norm, 0.5)
        self.assertEqual(list(params.rollout_schedule), [1, 2, 4, 6, 8, 10])
        target_handling = dict(params.target_handling)
        self.assertTrue(target_handling["enabled"])
        self.assertEqual(list(target_handling["copy_variables"]), ["orog"])
        self.assertEqual(list(target_handling["known_future_variables"]), ["tisr"])
        self.assertEqual(list(target_handling["exclude_loss_variables"]), ["orog", "tisr"])
        self.assertIn("era5_67_1p5_delta_stats.npz", str(params.delta_stats_path))

    def test_l3_hidden128(self) -> None:
        params = self._load(
            "weather_dual_resolution_1p5_l3_hidden128_orog_tisr_fixed.yaml",
            "raw_1p5_l3_hidden128_orog_tisr_fixed",
        )
        self._assert_common(params)
        self.assertEqual(params.hidden_dim, 128)
        self.assertEqual(params.num_heads, 4)
        self.assertEqual(
            [list(shape) for shape in params.level_shapes],
            [[121, 240], [61, 120], [31, 60], [16, 30]],
        )
        self.assertEqual(list(params.node_counts), [29040, 7320, 1860, 480])
        self.assertEqual(list(params.level_k_neighbors), [8, 8, 8, 8])
        self.assertEqual(
            list(params.edge_counts),
            [29040 * 8, 7320 * 8, 1860 * 8, 480 * 8],
        )
        self.assertIn("graph_1p5_k8_hybrid_row_aware_L3_v3.pt", str(params.graph_path))

    def test_l4_hidden128(self) -> None:
        params = self._load(
            "weather_dual_resolution_1p5_l4_hidden128_orog_tisr_fixed.yaml",
            "raw_1p5_l4_hidden128_orog_tisr_fixed",
        )
        self._assert_common(params)
        self.assertEqual(params.hidden_dim, 128)
        self.assertEqual(params.num_heads, 4)
        self.assertEqual(
            [list(shape) for shape in params.level_shapes],
            [[121, 240], [61, 120], [31, 60], [16, 30], [8, 15]],
        )
        self.assertEqual(list(params.node_counts), [29040, 7320, 1860, 480, 120])
        self.assertEqual(list(params.level_k_neighbors), [8, 8, 8, 8, 24])
        self.assertEqual(
            list(params.edge_counts),
            [29040 * 8, 7320 * 8, 1860 * 8, 480 * 8, 120 * 24],
        )
        self.assertIn(
            "graph_1p5_k8_levelk8-8-8-8-24_hybrid_row_aware_v5.pt",
            str(params.graph_path),
        )

    def test_l4_hidden160(self) -> None:
        params = self._load(
            "weather_dual_resolution_1p5_l4_hidden160_orog_tisr_fixed.yaml",
            "raw_1p5_l4_hidden160_orog_tisr_fixed",
        )
        self._assert_common(params)
        self.assertEqual(params.hidden_dim, 160)
        self.assertEqual(params.num_heads, 5)
        self.assertEqual(list(params.node_counts), [29040, 7320, 1860, 480, 120])
        self.assertEqual(list(params.level_k_neighbors), [8, 8, 8, 8, 24])


if __name__ == "__main__":
    unittest.main()
