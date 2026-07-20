from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import RolloutFeatureBuilder, dayofyear_sincos  # noqa: E402
from src.losses import LatitudeWeightedMSE  # noqa: E402
from src.models import GraphWeatherModel  # noqa: E402


class _NullLogger:
    def info(self, *args, **kwargs) -> None:
        return None

    def warning(self, *args, **kwargs) -> None:
        return None


def _fake_graph(height: int = 2, width: int = 2) -> SimpleNamespace:
    lat_lon = torch.tensor(
        [
            [-45.0, 0.0],
            [-45.0, 180.0],
            [45.0, 0.0],
            [45.0, 180.0],
        ],
        dtype=torch.float32,
    )
    return SimpleNamespace(L0=SimpleNamespace(height=height, width=width, lat_lon=lat_lon))


def _feature_params() -> SimpleNamespace:
    return SimpleNamespace(
        extra_features={
            "enabled": True,
            "spatial": {"lat_lon_sincos": True, "orography": True, "land_sea_mask": True},
            "temporal": {"dayofyear_sincos": True, "use_target_time": True},
            "known_forcings": {"enabled": True, "variables": ["tisr"]},
            "static_handling": {
                "copy_variables": ["orog"],
                "exclude_loss_variables": ["orog", "tisr"],
                "override_prediction_with_known": ["tisr"],
            },
        },
        static_fields={"path": None, "orography_name": "orog", "land_sea_mask_name": "lsm"},
        variable_metadata={"path": None},
        require_static_features=False,
        global_means_path="",
        global_stds_path="",
        experiment_dir="",
    )


class StaticFeatureBuilderTest(unittest.TestCase):
    def test_feature_shape_and_names(self) -> None:
        builder = RolloutFeatureBuilder.from_params(
            _feature_params(),
            graph=_fake_graph(),
            channel_names=["tisr", "orog", "lsm", "x"],
            out_channels=[0, 1, 2, 3],
            logger=_NullLogger(),
        )
        self.assertEqual(builder.aux_feature_dim, 9)
        self.assertEqual(
            builder.feature_names,
            [
                "sin_lat",
                "cos_lat",
                "sin_lon",
                "cos_lon",
                "orography",
                "land_sea_mask",
                "sin_dayofyear",
                "cos_dayofyear",
                "known_tisr",
            ],
        )

        current = torch.zeros(2, 4, 2, 2)
        current[:, 1] = 3.0
        current[:, 2] = 1.0
        target = torch.zeros(2, 4, 2, 2)
        target[:, 0] = 7.0
        aux = builder.build_step_features(
            current=current,
            target_norm=target,
            target_dayofyear=torch.tensor([[1], [182]]),
            target_days_in_year=torch.tensor([[365], [365]]),
            step_idx=0,
        )
        self.assertEqual(tuple(aux.shape), (2, 4, 9))
        self.assertTrue(torch.allclose(aux[:, :, 4], torch.full((2, 4), 3.0)))
        self.assertTrue(torch.allclose(aux[:, :, 5], torch.ones(2, 4)))
        self.assertTrue(torch.allclose(aux[:, :, 8], torch.full((2, 4), 7.0)))

    def test_known_forcing_and_static_overrides(self) -> None:
        builder = RolloutFeatureBuilder.from_params(
            _feature_params(),
            graph=_fake_graph(),
            channel_names=["tisr", "orog", "lsm", "x"],
            out_channels=[0, 1, 2, 3],
            logger=_NullLogger(),
        )
        pred = torch.zeros(1, 4, 2, 2)
        current = torch.zeros_like(pred)
        current[:, 1] = 11.0
        target = torch.zeros_like(pred)
        target[:, 0] = 22.0
        overridden = builder.apply_overrides(pred, current=current, target_norm=target)
        self.assertTrue(torch.equal(overridden[:, 0], target[:, 0]))
        self.assertTrue(torch.equal(overridden[:, 1], current[:, 1]))
        self.assertTrue(torch.equal(overridden[:, 2], pred[:, 2]))

        mask = builder.loss_channel_mask(4)
        self.assertEqual(mask.tolist(), [0.0, 0.0, 1.0, 1.0])

    def test_dayofyear_is_cyclic(self) -> None:
        jan1 = dayofyear_sincos(
            torch.tensor([1]),
            torch.tensor([365]),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        dec31 = dayofyear_sincos(
            torch.tensor([365]),
            torch.tensor([365]),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        self.assertLess(torch.linalg.vector_norm(jan1 - dec31).item(), 2.0 * math.pi / 365.0 + 1.0e-4)


class StaticFeatureModelAndLossTest(unittest.TestCase):
    def test_model_input_channels_include_aux_dim(self) -> None:
        graph = _fake_graph()
        model = GraphWeatherModel(
            graph=graph,
            grid_shape=(2, 2),
            input_channels=8,
            output_channels=4,
            n_history=1,
            hidden_dim=8,
            heads=1,
            encoder_blocks=0,
            decoder_blocks=0,
            l0_blocks=1,
            l1_blocks=1,
            l2_blocks=1,
            l1_refine_blocks=1,
            l0_refine_blocks=1,
            aux_feature_dim=9,
        )
        self.assertEqual(model.base_node_feature_channels, 8)
        self.assertEqual(model.total_node_feature_channels, 17)
        self.assertEqual(model.embed.in_features, 17)

    def test_loss_mask_excludes_channels(self) -> None:
        loss = LatitudeWeightedMSE(torch.tensor([0.0, 0.0]))
        pred = torch.zeros(1, 4, 2, 2)
        target = torch.zeros_like(pred)
        target[:, 0] = 100.0
        masked = loss(pred, target, channel_mask=torch.tensor([0.0, 1.0, 1.0, 1.0]))
        unmasked = loss(pred, target)
        self.assertAlmostEqual(float(masked.item()), 0.0)
        self.assertGreater(float(unmasked.item()), 0.0)


if __name__ == "__main__":
    unittest.main()
