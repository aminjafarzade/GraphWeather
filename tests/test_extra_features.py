from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import RolloutFeatureBuilder, dayofyear_sincos  # noqa: E402
from src.losses import LatitudeWeightedMSE  # noqa: E402


class _NullLogger:
    def info(self, *args, **kwargs) -> None:
        return None

    def warning(self, *args, **kwargs) -> None:
        return None


def _fake_graph(height: int = 3, width: int = 4) -> SimpleNamespace:
    lats = torch.linspace(-60.0, 60.0, height)
    lons = torch.linspace(0.0, 360.0 - 360.0 / width, width)
    lat_lon = torch.stack(
        [torch.tensor([float(lat), float(lon)]) for lat in lats for lon in lons],
        dim=0,
    )
    return SimpleNamespace(L0=SimpleNamespace(height=height, width=width, lat_lon=lat_lon))


def _params(enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        extra_features={
            "enabled": enabled,
            "spatial": {"lat_lon_sincos": enabled, "orography": enabled, "land_sea_mask": enabled},
            "temporal": {"dayofyear_sincos": enabled, "use_target_time": True},
            "known_forcings": {"enabled": enabled, "variables": ["tisr"] if enabled else []},
            "static_handling": {
                "copy_variables": ["orog"] if enabled else [],
                "exclude_loss_variables": ["orog", "tisr"] if enabled else [],
                "override_prediction_with_known": ["tisr"] if enabled else [],
            },
        },
        static_fields={"path": None, "orography_name": "orog", "land_sea_mask_name": "lsm"},
        variable_metadata={"path": None},
        require_static_features=False,
        global_means_path="",
        global_stds_path="",
        experiment_dir="",
    )


def _builder(enabled: bool = True) -> RolloutFeatureBuilder:
    return RolloutFeatureBuilder.from_params(
        _params(enabled=enabled),
        graph=_fake_graph(),
        channel_names=["tisr", "orog", "lsm", "t2m"],
        out_channels=[0, 1, 2, 3],
        logger=_NullLogger(),
    )


def test_lat_lon_feature_ordering_on_tiny_grid() -> None:
    builder = _builder()
    current = torch.zeros(1, 4, 3, 4)
    target = torch.zeros_like(current)
    aux = builder.build_step_features(
        current=current,
        target_norm=target,
        target_dayofyear=torch.tensor([[10]]),
        target_days_in_year=torch.tensor([[365]]),
        step_idx=0,
    )
    assert aux is not None
    columns = {name: builder.feature_names.index(name) for name in ["sin_lat", "cos_lat", "sin_lon", "cos_lon"]}
    lat_lon = _fake_graph().L0.lat_lon
    for node_id in [0, 3, 6, 8, 11]:
        lat = math.radians(float(lat_lon[node_id, 0]))
        lon = math.radians(float(lat_lon[node_id, 1]))
        expected = torch.tensor([math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon)])
        actual = torch.tensor([float(aux[0, node_id, columns[name]]) for name in columns])
        assert torch.allclose(actual, expected, atol=1.0e-6)


def test_dayofyear_feature_uses_target_step() -> None:
    builder = _builder()
    current = torch.zeros(1, 4, 3, 4)
    target = torch.zeros_like(current)
    target_dayofyear = torch.tensor([[31, 32]])
    target_days = torch.tensor([[365, 365]])
    aux = builder.build_step_features(
        current=current,
        target_norm=target,
        target_dayofyear=target_dayofyear,
        target_days_in_year=target_days,
        step_idx=1,
    )
    assert aux is not None
    sin_col = builder.feature_names.index("sin_dayofyear")
    cos_col = builder.feature_names.index("cos_dayofyear")
    expected = dayofyear_sincos(
        torch.tensor([32]),
        torch.tensor([365]),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )[0]
    actual = torch.tensor([float(aux[0, 0, sin_col]), float(aux[0, 0, cos_col])])
    assert torch.allclose(actual, expected, atol=1.0e-6)


def test_loss_mask_excludes_orog_and_tisr_only() -> None:
    builder = _builder()
    mask = builder.loss_channel_mask(4)
    assert mask is not None
    assert mask.tolist() == [0.0, 0.0, 1.0, 1.0]
    loss_fn = LatitudeWeightedMSE(torch.zeros(3))
    target = torch.zeros(1, 4, 3, 4)
    pred = target.clone()
    pred[:, 0] += 100000.0
    pred[:, 1] += 100000.0
    assert float(loss_fn(pred, target, channel_mask=mask).item()) == 0.0
    pred = target.clone()
    pred[:, 3] += 1.0
    assert float(loss_fn(pred, target, channel_mask=mask).item()) > 0.0


def test_orography_copy_override() -> None:
    builder = _builder()
    pred = torch.zeros(1, 4, 3, 4)
    current = torch.zeros_like(pred)
    current[:, 1] = 42.0
    target = torch.zeros_like(pred)
    overridden = builder.apply_overrides(pred, current=current, target_norm=target)
    assert torch.equal(overridden[:, 1], current[:, 1])
    assert torch.equal(overridden[:, 2], pred[:, 2])


def test_tisr_known_feature_and_override_use_target() -> None:
    builder = _builder()
    current = torch.zeros(1, 4, 3, 4)
    target = torch.zeros_like(current)
    target[:, 0] = 7.0
    aux = builder.build_step_features(
        current=current,
        target_norm=target,
        target_dayofyear=torch.tensor([[1]]),
        target_days_in_year=torch.tensor([[365]]),
        step_idx=0,
    )
    assert aux is not None
    known_col = builder.feature_names.index("known_tisr")
    assert torch.allclose(aux[0, :, known_col].reshape(3, 4), target[0, 0])
    pred = torch.zeros_like(target)
    overridden = builder.apply_overrides(pred, current=current, target_norm=target)
    assert torch.equal(overridden[:, 0], target[:, 0])


def test_backward_compatibility_when_extra_features_disabled() -> None:
    builder = _builder(enabled=False)
    assert builder.aux_feature_dim == 0
    current = torch.zeros(1, 4, 3, 4)
    target = torch.ones_like(current)
    assert builder.build_step_features(current=current, target_norm=target) is None
    pred = torch.full_like(current, 3.0)
    assert builder.apply_overrides(pred, current=current, target_norm=target) is pred
    assert builder.loss_channel_mask(4) is None
