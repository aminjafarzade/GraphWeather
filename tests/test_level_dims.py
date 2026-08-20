from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.graph_builder import HYBRID_ROW_AWARE_KNN, build_graph_bundle, regular_lat_lon  # noqa: E402
from src.graph_bundle import GraphBundle  # noqa: E402
from src.models import GraphWeatherModel  # noqa: E402

# Parameter counts are grid-size independent (every layer is node-shared), so the
# small 5.625deg 4-level graph exercises the exact layer inventory of the 2.5deg
# L3 configs and pins their trainable-parameter counts.
FLAT_HIDDEN160_PARAMS = 3_844_932
INVERTED_W192_PARAMS = 2_852_257

# The standard L3 block layout of the 2.5deg dense-l3k24 experiment family.
SHARED_MODEL_KWARGS = dict(
    input_channels=134,
    output_channels=67,
    n_history=1,
    encoder_blocks=1,
    decoder_blocks=1,
    l0_blocks=2,
    l1_blocks=2,
    l2_blocks=1,
    l1_refine_blocks=1,
    l0_refine_blocks=1,
    num_graph_levels=4,
    use_l3=True,
    l3_blocks=1,
    l2_refine_after_l3_blocks=1,
)


def _trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class LevelDimsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        latitudes, longitudes = regular_lat_lon(
            lat_count=32,
            lon_count=64,
            resolution=5.625,
            lat_start=-87.1875,
            lon_start=-180.0,
        )
        cls.bundle = build_graph_bundle(
            latitudes,
            longitudes,
            k=8,
            resolution=5.625,
            connectivity_strategy=HYBRID_ROW_AWARE_KNN,
            num_graph_levels=4,
        )

    def _model(self, **overrides: object) -> GraphWeatherModel:
        kwargs = dict(
            graph=GraphBundle(self.bundle),
            grid_shape=(32, 64),
            **SHARED_MODEL_KWARGS,
        )
        kwargs.update(overrides)
        return GraphWeatherModel(**kwargs)

    def test_default_level_dims_keeps_flat_hidden160_count(self) -> None:
        model = self._model(hidden_dim=160, heads=5)
        self.assertEqual(_trainable_params(model), FLAT_HIDDEN160_PARAMS)
        self.assertEqual(model.level_dims, [160] * 4)
        self.assertEqual(model.level_heads, [5] * 4)

    def test_inverted_w192_parameter_count(self) -> None:
        model = self._model(
            hidden_dim=192,
            heads=6,
            level_dims=[192, 96, 48, 24],
            level_heads=[6, 6, 6, 6],
        )
        self.assertEqual(_trainable_params(model), INVERTED_W192_PARAMS)
        self.assertEqual(model.level_dims, [192, 96, 48, 24])
        self.assertEqual(model.level_heads, [6, 6, 6, 6])

    def test_inverted_forward_and_autoregressive_step(self) -> None:
        model = self._model(
            hidden_dim=192,
            heads=6,
            level_dims=[192, 96, 48, 24],
            level_heads=[6, 6, 6, 6],
        )
        inp = torch.randn(2, 134, 32, 64)
        with torch.no_grad():
            out = model(inp)
        self.assertEqual(tuple(out.shape), (2, 67, 32, 64))
        self.assertTrue(torch.isfinite(out).all().item())
        # One autoregressive step: [previous(67), current(67)] -> [current, prediction].
        with torch.no_grad():
            out2 = model(torch.cat([inp[:, 67:], out], dim=1))
        self.assertEqual(tuple(out2.shape), (2, 67, 32, 64))
        self.assertTrue(torch.isfinite(out2).all().item())

    def test_invalid_level_dims_raise(self) -> None:
        with self.assertRaises(ValueError):
            self._model(hidden_dim=192, heads=6, level_dims=[192, 96, 48])
        with self.assertRaises(ValueError):
            self._model(
                hidden_dim=192,
                heads=6,
                level_dims=[192, 96, 48, 24],
                level_heads=[6, 6, 6, 5],
            )

    def test_sum_skip_fusion_rejects_unequal_widths(self) -> None:
        with self.assertRaises(ValueError):
            self._model(
                hidden_dim=192,
                heads=6,
                level_dims=[192, 96, 48, 24],
                level_heads=[6, 6, 6, 6],
                skip_fusion={"type": "sum"},
            )


if __name__ == "__main__":
    unittest.main()
