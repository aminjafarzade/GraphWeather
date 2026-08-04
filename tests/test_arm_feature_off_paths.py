"""OFF-path guarantees for the five-arm feature set (arms C, D, E).

All five arms share one code state, so every new feature is config-gated and must
default to today's exact behaviour. These tests are the guard: they assert the
default/explicitly-off configurations are bit-identical to the pre-feature code,
both structurally (module types, parameter names and shapes) and numerically
(identical forward output under a fixed seed).

Arm G (variable_upweights) and arm H (pooling include_max / area_weighted) need no
code change, so their tests only assert the existing paths still build and run.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.graph_builder import HYBRID_ROW_AWARE_KNN, build_graph_bundle, cell_center_lat_lon  # noqa: E402
from src.graph_bundle import GraphBundle  # noqa: E402
from src.layers import (  # noqa: E402
    LocalGraphAttention,
    edge_encoding_is_off,
    expand_edge_features,
    expanded_edge_dim,
    resolve_edge_encoding,
)
from src.models import GraphWeatherModel  # noqa: E402
from src.pooling import MeanMaxPool  # noqa: E402
from src.trainer import Trainer  # noqa: E402

_BUNDLE_CACHE: dict[str, object] = {}


def _bundle_2p5_l3():
    """Same synthetic 2.5-degree L3 bundle the graph-builder tests use."""
    if "b" not in _BUNDLE_CACHE:
        latitudes, longitudes = cell_center_lat_lon(72, 144)
        _BUNDLE_CACHE["b"] = build_graph_bundle(
            latitudes,
            longitudes,
            k=8,
            resolution=2.5,
            connectivity_strategy=HYBRID_ROW_AWARE_KNN,
            resolution_mode="2p5",
            num_graph_levels=4,
        )
    return _BUNDLE_CACHE["b"]


def _model(**overrides) -> GraphWeatherModel:
    kwargs = dict(
        graph=GraphBundle(_bundle_2p5_l3()),
        grid_shape=(72, 144),
        input_channels=134,
        output_channels=67,
        n_history=1,
        hidden_dim=16,
        edge_dim=6,
        heads=4,
        encoder_blocks=1,
        decoder_blocks=1,
        l0_blocks=2,
        l1_blocks=2,
        l2_blocks=1,
        num_graph_levels=4,
        use_l3=True,
        l3_blocks=1,
        l2_refine_after_l3_blocks=1,
        l1_refine_blocks=1,
        l0_refine_blocks=1,
    )
    kwargs.update(overrides)
    return GraphWeatherModel(**kwargs)


def _seeded(fn):
    torch.manual_seed(1234)
    return fn()


def _forward(model: GraphWeatherModel) -> torch.Tensor:
    torch.manual_seed(99)
    x = torch.randn(1, 134, 72, 144)
    model.eval()
    with torch.no_grad():
        return model(x)


def _sig(model: nn.Module) -> list[tuple[str, tuple[int, ...]]]:
    return sorted((name, tuple(p.shape)) for name, p in model.state_dict().items())


# --------------------------------------------------------------------------- #
# ARM D -- edge_encoding
# --------------------------------------------------------------------------- #
class ArmDEdgeEncodingOffPathTest(unittest.TestCase):
    def test_defaults_are_all_off(self) -> None:
        r = resolve_edge_encoding(None)
        self.assertEqual(
            r, {"rbf_bins": 0, "bearing_harmonics": 0, "gate": False, "bias_mlp_hidden": 0}
        )
        self.assertTrue(edge_encoding_is_off(r))
        self.assertEqual(expanded_edge_dim(6, r), 6)

    def test_off_attention_module_structure_unchanged(self) -> None:
        """No gate params, and edge_bias stays the original Linear(edge_dim, heads)."""
        attn = LocalGraphAttention(160, edge_dim=6, heads=5)
        self.assertIsInstance(attn.edge_bias, nn.Linear)
        self.assertEqual(attn.edge_bias.in_features, 6)
        self.assertEqual(attn.edge_bias.out_features, 5)
        self.assertIsNone(attn.edge_gate_mlp)
        self.assertIsNone(attn.gate_scale)
        # 4 x (160^2+160) + 2 x (6*160+160) + (6*5+5)
        self.assertEqual(sum(p.numel() for p in attn.parameters()), 105315)

    def test_off_expansion_is_identity(self) -> None:
        ea = torch.randn(5, 8, 6)
        out = expand_edge_features(ea, resolve_edge_encoding(None))
        self.assertIs(out, ea)

    def test_unset_equals_explicitly_off_bitwise(self) -> None:
        a = _seeded(lambda: _model())
        b = _seeded(
            lambda: _model(
                edge_encoding={
                    "rbf_bins": 0,
                    "bearing_harmonics": 0,
                    "gate": False,
                    "bias_mlp_hidden": 0,
                }
            )
        )
        self.assertEqual(_sig(a), _sig(b))
        self.assertTrue(torch.equal(_forward(a), _forward(b)))

    def test_arm_d_on_builds_and_runs(self) -> None:
        m = _model(
            edge_encoding={
                "rbf_bins": 16,
                "bearing_harmonics": 2,
                "gate": True,
                "bias_mlp_hidden": 32,
            }
        )
        y = _forward(m)
        self.assertEqual(tuple(y.shape), (1, 67, 72, 144))
        self.assertTrue(torch.isfinite(y).all().item())

    def test_gate_is_exactly_one_before_training(self) -> None:
        """Arm D must start as the control: gate == 1.0 everywhere at init."""
        ee = {"rbf_bins": 16, "bearing_harmonics": 2, "gate": True, "bias_mlp_hidden": 32}
        attn = LocalGraphAttention(160, edge_dim=6, heads=5, edge_encoding=ee)
        feats = expand_edge_features(torch.rand(20, 8, 6) + 0.5, resolve_edge_encoding(ee))
        with torch.no_grad():
            gate = 1.0 + attn.gate_scale * torch.tanh(attn.edge_gate_mlp(feats))
        self.assertTrue(bool(torch.all(gate == 1.0)), f"gate range {gate.min()}..{gate.max()}")

    def test_expanded_width_and_harmonics(self) -> None:
        ee = resolve_edge_encoding(
            {"rbf_bins": 16, "bearing_harmonics": 2, "gate": True, "bias_mlp_hidden": 32}
        )
        self.assertEqual(expanded_edge_dim(6, ee), 26)  # 6 + 16 + 2*2
        # harmonics come from the multiple-angle recurrence on sin/cos(bearing)
        ang = torch.tensor([0.3, 1.1, 2.7])
        ea = torch.zeros(1, 3, 6)
        ea[0, :, 0] = 1.0
        ea[0, :, 1] = torch.sin(ang)
        ea[0, :, 2] = torch.cos(ang)
        feats = expand_edge_features(ea, ee)
        # layout: [rbf(16) | sin2,cos2 | sin3,cos3 | original 6]
        self.assertTrue(torch.allclose(feats[0, :, 16], torch.sin(2 * ang), atol=1e-6))
        self.assertTrue(torch.allclose(feats[0, :, 17], torch.cos(2 * ang), atol=1e-6))
        self.assertTrue(torch.allclose(feats[0, :, 18], torch.sin(3 * ang), atol=1e-6))
        self.assertTrue(torch.allclose(feats[0, :, 19], torch.cos(3 * ang), atol=1e-6))
        self.assertTrue(torch.allclose(feats[0, :, 20:], ea[0], atol=0))

    def test_expansion_is_cached_not_recomputed(self) -> None:
        ee = {"rbf_bins": 8, "bearing_harmonics": 1, "gate": True, "bias_mlp_hidden": 0}
        attn = LocalGraphAttention(16, edge_dim=6, heads=4, edge_encoding=ee)
        ea = torch.rand(10, 4, 6) + 0.5
        first = attn._edge_feats(ea)
        second = attn._edge_feats(ea)
        self.assertIs(first, second)
        self.assertEqual(len(attn._edge_feat_cache), 1)

    def test_rejects_degenerate_rbf_bins(self) -> None:
        with self.assertRaises(ValueError):
            resolve_edge_encoding({"rbf_bins": 1})
        with self.assertRaises(ValueError):
            resolve_edge_encoding({"rbf_bins": -3})


# --------------------------------------------------------------------------- #
# ARM E -- boundary_mlp / head_init_std
# --------------------------------------------------------------------------- #
class ArmEBoundaryMlpOffPathTest(unittest.TestCase):
    def test_off_keeps_single_linear_embed_and_head(self) -> None:
        m = _model()
        self.assertIsInstance(m.embed, nn.Linear)
        self.assertIsInstance(m.head, nn.Linear)
        self.assertEqual(m.embed.in_features, 134)

    def test_unset_equals_explicitly_off_bitwise(self) -> None:
        a = _seeded(lambda: _model())
        b = _seeded(lambda: _model(boundary_mlp=False, head_init_std=0.0))
        self.assertEqual(_sig(a), _sig(b))
        self.assertTrue(torch.equal(_forward(a), _forward(b)))

    def test_arm_e_on_builds_runs_and_uses_sequential(self) -> None:
        m = _model(boundary_mlp=True, head_init_std=1.0e-3)
        self.assertIsInstance(m.embed, nn.Sequential)
        self.assertIsInstance(m.head, nn.Sequential)
        self.assertIsInstance(m.embed[-1], nn.LayerNorm)
        self.assertIsInstance(m.head[-1], nn.Linear)
        y = _forward(m)
        self.assertEqual(tuple(y.shape), (1, 67, 72, 144))
        self.assertTrue(torch.isfinite(y).all().item())

    def test_head_init_std_applies_to_final_linear_both_forms(self) -> None:
        lin = _model(head_init_std=1.0e-3)
        self.assertLess(float(lin.head.weight.detach().std()), 5.0e-3)
        self.assertTrue(bool(torch.all(lin.head.bias == 0.0)))
        seq = _model(boundary_mlp=True, head_init_std=1.0e-3)
        self.assertLess(float(seq.head[-1].weight.detach().std()), 5.0e-3)
        self.assertTrue(bool(torch.all(seq.head[-1].bias == 0.0)))

    def test_head_init_std_zero_leaves_default_init(self) -> None:
        """std=0.0 must not touch the head (a normal_(std=0) would zero it)."""
        m = _model(head_init_std=0.0)
        self.assertGreater(float(m.head.weight.detach().std()), 1.0e-3)


# --------------------------------------------------------------------------- #
# ARM C -- inverse_tendency_variance
# --------------------------------------------------------------------------- #
ERA5_67 = [
    "t2m", "msl", "sp", "tcwv", "skt", "tisr",
    *[f"{v}{l}" for v in ("u", "v", "t", "q", "z")
      for l in (1000, 925, 850, 800, 700, 600, 500, 400, 300, 200, 100, 50)],
    "orog",
]


def _weights(cfg, delta_std=None, mask=None):
    t = object.__new__(Trainer)
    t.loss_channel_weight_cfg = dict(cfg)
    t.params = {}
    if delta_std is not None:
        t.loss_delta_std_norm = torch.as_tensor(delta_std, dtype=torch.float32)
    if mask is not None:
        t.loss_channel_mask = torch.as_tensor(mask, dtype=torch.float32)
    return Trainer._build_loss_channel_weights(t, channel_names=ERA5_67, num_channels=len(ERA5_67))


class ArmCInverseTendencyVarianceTest(unittest.TestCase):
    BASE = {"enabled": True, "pressure_weighting": False, "variable_upweights": {}}

    def test_off_reproduces_today_exactly(self) -> None:
        std = torch.rand(67) * 0.5 + 0.05
        without = _weights(self.BASE)
        with_off = _weights({**self.BASE, "inverse_tendency_variance": False}, delta_std=std)
        self.assertEqual(without, with_off)
        self.assertEqual(without, [1.0] * 67)  # pressure off, no upweights -> all ones

    def test_off_reproduces_today_exactly_with_pressure_on(self) -> None:
        std = torch.rand(67) * 0.5 + 0.05
        a = _weights({"enabled": True, "pressure_weighting": True, "variable_upweights": {}})
        b = _weights(
            {
                "enabled": True,
                "pressure_weighting": True,
                "variable_upweights": {},
                "inverse_tendency_variance": False,
            },
            delta_std=std,
        )
        self.assertEqual(a, b)

    def test_itv_requires_delta_stats(self) -> None:
        with self.assertRaisesRegex(ValueError, "inverse_tendency_variance"):
            _weights({**self.BASE, "inverse_tendency_variance": True})

    def test_itv_factor_clamp_and_exponent(self) -> None:
        std = [0.5] * 67
        std[0] = 0.001  # below the clamp
        w = _weights(
            {**self.BASE, "inverse_tendency_variance": True, "itv_exponent": 2.0, "itv_min_std": 0.01},
            delta_std=std,
        )
        # raw factors: clamped channel 1/0.01^2 = 1e4, others 1/0.5^2 = 4
        raw = [1.0 / (max(0.01, s) ** 2) for s in std]
        mean_raw = sum(raw) / len(raw)
        self.assertAlmostEqual(w[0], raw[0] / mean_raw, places=5)
        self.assertAlmostEqual(w[1], raw[1] / mean_raw, places=5)

    def test_itv_mean_over_scored_channels_is_one(self) -> None:
        std = (torch.rand(67) * 0.4 + 0.05).tolist()
        mask = [1.0] * 67
        mask[5] = 0.0   # tisr
        mask[66] = 0.0  # orog
        w = _weights(
            {**self.BASE, "inverse_tendency_variance": True}, delta_std=std, mask=mask
        )
        self.assertEqual(w[5], 0.0)
        self.assertEqual(w[66], 0.0)
        scored = [w[i] for i in range(67) if mask[i] > 0.0]
        self.assertEqual(len(scored), 65)
        self.assertAlmostEqual(sum(scored) / len(scored), 1.0, places=6)

    def test_itv_exponent_one_is_inverse_std(self) -> None:
        std = [0.2, 0.4] + [0.3] * 65
        w = _weights(
            {**self.BASE, "inverse_tendency_variance": True, "itv_exponent": 1.0},
            delta_std=std,
        )
        self.assertAlmostEqual(w[0] / w[1], (1 / 0.2) / (1 / 0.4), places=5)

    def test_resolved_keys_persisted(self) -> None:
        std = [0.3] * 67
        t = object.__new__(Trainer)
        t.loss_channel_weight_cfg = {**self.BASE, "inverse_tendency_variance": True}
        t.params = {}
        t.loss_delta_std_norm = torch.as_tensor(std, dtype=torch.float32)
        Trainer._build_loss_channel_weights(t, channel_names=ERA5_67, num_channels=67)
        p = t.params["loss_channel_weighting"]
        self.assertIs(p["inverse_tendency_variance"], True)
        self.assertEqual(p["itv_exponent"], 2.0)
        self.assertEqual(p["itv_min_std"], 0.01)


# --------------------------------------------------------------------------- #
# ARM H -- pooling (no code change; assert both settings work)
# --------------------------------------------------------------------------- #
class ArmHPoolingTest(unittest.TestCase):
    def test_include_max_sizes_projection(self) -> None:
        self.assertEqual(MeanMaxPool(32, {"include_max": True}).proj.in_features, 64)
        self.assertEqual(MeanMaxPool(32, {"include_max": False}).proj.in_features, 32)

    def test_both_include_max_settings_run(self) -> None:
        h = torch.randn(2, 8, 32)
        pool_map = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        for include_max in (True, False):
            out = MeanMaxPool(32, {"include_max": include_max})(h, pool_map, 4)
            self.assertEqual(tuple(out.shape), (2, 4, 32))
            self.assertTrue(torch.isfinite(out).all().item())

    def test_area_weighted_runs_and_needs_child_weights(self) -> None:
        h = torch.randn(2, 8, 32)
        pool_map = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        pool = MeanMaxPool(32, {"mean_type": "area_weighted", "include_max": False})
        with self.assertRaisesRegex(ValueError, "area_weighted"):
            pool(h, pool_map, 4)
        out = pool(h, pool_map, 4, torch.rand(8) + 0.1)
        self.assertEqual(tuple(out.shape), (2, 4, 32))
        self.assertTrue(torch.isfinite(out).all().item())


# --------------------------------------------------------------------------- #
# ARM G -- variable_upweights (no code change)
# --------------------------------------------------------------------------- #
class ArmGVariableUpweightsTest(unittest.TestCase):
    def test_upweights_apply_multiplicatively_on_top_of_pressure(self) -> None:
        cfg = {
            "enabled": True,
            "pressure_weighting": True,
            "variable_upweights": {"t2m": 2.0, "z500": 2.0},
        }
        with_up = _weights(cfg)
        without = _weights({**cfg, "variable_upweights": {}})
        i_t2m, i_z500 = ERA5_67.index("t2m"), ERA5_67.index("z500")
        self.assertAlmostEqual(with_up[i_t2m], without[i_t2m] * 2.0, places=6)
        self.assertAlmostEqual(with_up[i_z500], without[i_z500] * 2.0, places=6)
        for i in range(67):
            if i not in (i_t2m, i_z500):
                self.assertEqual(with_up[i], without[i])


if __name__ == "__main__":
    unittest.main()
