from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.graph_builder import (  # noqa: E402
    HYBRID_ROW_AWARE_KNN,
    build_graph_bundle,
    expected_graph_metadata,
    regular_lat_lon,
    validate_graph_cache_metadata,
)
from src.graph_bundle import GraphBundle  # noqa: E402
from src.models import GraphWeatherModel  # noqa: E402
from src.architecture import architecture_metadata, validate_checkpoint_architecture  # noqa: E402
from src.lead_conditioning import build_lead_conditioning_grid, lead_sincos_values  # noqa: E402
from src.layers import LocalGraphAttentionBlock, NodewiseRefineMLP  # noqa: E402
from src.pooling import ParentUnpoolFuse  # noqa: E402
from src.resolution import cell_center_lat_lon  # noqa: E402


def _neighbors(level: dict[str, object]) -> np.ndarray:
    edge_index = level["edge_index"]
    assert isinstance(edge_index, torch.Tensor)
    num_nodes = int(level["num_nodes"])
    k = int(level["k"])
    return edge_index[0].reshape(num_nodes, k).detach().cpu().numpy()


def _component_count(level: dict[str, object]) -> int:
    edge_index = level["edge_index"]
    assert isinstance(edge_index, torch.Tensor)
    num_nodes = int(level["num_nodes"])
    src = edge_index[0].detach().cpu().numpy()
    dst = edge_index[1].detach().cpu().numpy()
    adjacency = coo_matrix((np.ones(src.size, dtype=np.uint8), (src, dst)), shape=(num_nodes, num_nodes))
    num_components, _ = connected_components(adjacency, directed=False)
    return int(num_components)


class HybridRowAwareGraphBuilderTest(unittest.TestCase):
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
        )

    def test_neighbor_counts_no_self_loops_no_duplicates(self) -> None:
        for level_name, level in self.bundle["levels"].items():
            with self.subTest(level=level_name):
                neighbors = _neighbors(level)
                num_nodes = int(level["num_nodes"])
                self.assertEqual(neighbors.shape, (num_nodes, 8))
                for target in range(num_nodes):
                    row = neighbors[target].tolist()
                    self.assertNotIn(target, row)
                    self.assertEqual(len(set(row)), 8)

    def test_row_aware_cross_row_rules(self) -> None:
        for level_name, level in self.bundle["levels"].items():
            with self.subTest(level=level_name):
                neighbors = _neighbors(level)
                height = int(level["height"])
                width = int(level["width"])
                target_rows = np.arange(height * width) // width
                neighbor_rows = neighbors // width
                for target, target_row in enumerate(target_rows):
                    if target_row == 0:
                        self.assertGreaterEqual(int(np.sum(neighbor_rows[target] == 1)), 2)
                    elif target_row == height - 1:
                        self.assertGreaterEqual(int(np.sum(neighbor_rows[target] == height - 2)), 2)
                    else:
                        self.assertGreaterEqual(int(np.sum(neighbor_rows[target] == target_row - 1)), 1)
                        self.assertGreaterEqual(int(np.sum(neighbor_rows[target] == target_row + 1)), 1)

    def test_connected_components_and_edge_counts(self) -> None:
        expected_edges = {"L0": 16384, "L1": 4096, "L2": 1024}
        for level_name, level in self.bundle["levels"].items():
            with self.subTest(level=level_name):
                edge_index = level["edge_index"]
                assert isinstance(edge_index, torch.Tensor)
                self.assertEqual(int(edge_index.shape[1]), expected_edges[level_name])
                self.assertEqual(_component_count(level), 1)

    def test_deterministic_graph_build(self) -> None:
        latitudes, longitudes = regular_lat_lon(
            lat_count=32,
            lon_count=64,
            resolution=5.625,
            lat_start=-87.1875,
            lon_start=-180.0,
        )
        other = build_graph_bundle(
            latitudes,
            longitudes,
            k=8,
            resolution=5.625,
            connectivity_strategy=HYBRID_ROW_AWARE_KNN,
        )
        for level_name in ("L0", "L1", "L2"):
            with self.subTest(level=level_name):
                a = self.bundle["levels"][level_name]
                b = other["levels"][level_name]
                self.assertTrue(torch.equal(a["edge_index"], b["edge_index"]))
                self.assertTrue(torch.allclose(a["edge_attr"], b["edge_attr"]))

    def test_model_forward_smoke(self) -> None:
        graph = GraphBundle(self.bundle)
        model = GraphWeatherModel(
            graph=graph,
            grid_shape=(32, 64),
            input_channels=134,
            output_channels=67,
            n_history=1,
            hidden_dim=16,
            edge_dim=6,
            heads=4,
            encoder_blocks=1,
            decoder_blocks=1,
            l0_blocks=1,
            l1_blocks=1,
            l2_blocks=1,
            l1_refine_blocks=1,
            l0_refine_blocks=1,
        )
        x = torch.randn(1, 134, 32, 64)
        with torch.no_grad():
            y = model(x)
        self.assertEqual(tuple(y.shape), (1, 67, 32, 64))
        self.assertTrue(torch.isfinite(y).all().item())


class DualResolutionGraphBuilderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        latitudes, longitudes = cell_center_lat_lon(72, 144)
        cls.bundle_2p5 = build_graph_bundle(
            latitudes,
            longitudes,
            k=8,
            resolution=2.5,
            connectivity_strategy=HYBRID_ROW_AWARE_KNN,
            resolution_mode="2p5",
        )

    def test_2p5_graph_counts_and_pool_maps(self) -> None:
        expected_nodes = {"L0": 10368, "L1": 2592, "L2": 648}
        expected_edges = {"L0": 82944, "L1": 20736, "L2": 5184}
        for level_name, level in self.bundle_2p5["levels"].items():
            with self.subTest(level=level_name):
                self.assertEqual(int(level["num_nodes"]), expected_nodes[level_name])
                self.assertEqual(int(level["edge_index"].shape[1]), expected_edges[level_name])
                self.assertEqual(_component_count(level), 1)
        self.assertEqual(int(self.bundle_2p5["pool"]["L0_to_L1"].numel()), 10368)
        self.assertEqual(int(self.bundle_2p5["pool"]["L1_to_L2"].numel()), 2592)
        self.assertLess(int(self.bundle_2p5["pool"]["L0_to_L1"].max()), 2592)
        self.assertLess(int(self.bundle_2p5["pool"]["L1_to_L2"].max()), 648)
        graph = GraphBundle(self.bundle_2p5)
        self.assertEqual(graph.num_graph_levels, 3)
        self.assertFalse(graph.use_l3)
        self.assertIsNone(graph.level3)
        self.assertIsNone(graph.pool_l2_to_l3)

    def test_parameter_count_equal_between_modes(self) -> None:
        latitudes, longitudes = regular_lat_lon(
            lat_count=32,
            lon_count=64,
            resolution=5.625,
            lat_start=-87.1875,
            lon_start=-180.0,
        )
        bundle_5p625 = build_graph_bundle(
            latitudes,
            longitudes,
            k=8,
            resolution=5.625,
            connectivity_strategy=HYBRID_ROW_AWARE_KNN,
            resolution_mode="5p625",
        )
        model_kwargs = dict(
            input_channels=134,
            output_channels=67,
            n_history=1,
            hidden_dim=16,
            edge_dim=6,
            heads=4,
            encoder_blocks=1,
            decoder_blocks=1,
            l0_blocks=1,
            l1_blocks=1,
            l2_blocks=1,
            l1_refine_blocks=1,
            l0_refine_blocks=1,
        )
        model_5p625 = GraphWeatherModel(GraphBundle(bundle_5p625), grid_shape=(32, 64), **model_kwargs)
        model_2p5 = GraphWeatherModel(GraphBundle(self.bundle_2p5), grid_shape=(72, 144), **model_kwargs)
        params_5p625 = sum(p.numel() for p in model_5p625.parameters() if p.requires_grad)
        params_2p5 = sum(p.numel() for p in model_2p5.parameters() if p.requires_grad)
        self.assertEqual(params_5p625, params_2p5)
        x = torch.randn(1, 134, 72, 144)
        with torch.no_grad():
            y = model_2p5(x)
        self.assertEqual(tuple(y.shape), (1, 67, 72, 144))
        self.assertTrue(torch.isfinite(y).all().item())


class L3GraphUNetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        latitudes, longitudes = cell_center_lat_lon(72, 144)
        cls.bundle_2p5_l3 = build_graph_bundle(
            latitudes,
            longitudes,
            k=8,
            resolution=2.5,
            connectivity_strategy=HYBRID_ROW_AWARE_KNN,
            resolution_mode="2p5",
            num_graph_levels=4,
        )

    def test_2p5_l3_graph_counts_and_pool_maps(self) -> None:
        expected_nodes = {"L0": 10368, "L1": 2592, "L2": 648, "L3": 162}
        expected_edges = {"L0": 82944, "L1": 20736, "L2": 5184, "L3": 1296}
        self.assertEqual(self.bundle_2p5_l3["metadata"]["num_graph_levels"], 4)
        self.assertTrue(self.bundle_2p5_l3["metadata"]["use_l3"])
        self.assertEqual(
            self.bundle_2p5_l3["metadata"]["level_shapes"],
            [[72, 144], [36, 72], [18, 36], [9, 18]],
        )
        for level_name, level in self.bundle_2p5_l3["levels"].items():
            with self.subTest(level=level_name):
                self.assertEqual(int(level["num_nodes"]), expected_nodes[level_name])
                self.assertEqual(int(level["edge_index"].shape[1]), expected_edges[level_name])
                self.assertEqual(_component_count(level), 1)
        self.assertEqual(int(self.bundle_2p5_l3["pool"]["L0_to_L1"].numel()), 10368)
        self.assertEqual(int(self.bundle_2p5_l3["pool"]["L1_to_L2"].numel()), 2592)
        self.assertEqual(int(self.bundle_2p5_l3["pool"]["L2_to_L3"].numel()), 648)
        self.assertLess(int(self.bundle_2p5_l3["pool"]["L2_to_L3"].max()), 162)

    def test_l3_model_forward_shape(self) -> None:
        graph = GraphBundle(self.bundle_2p5_l3)
        model = GraphWeatherModel(
            graph=graph,
            grid_shape=(72, 144),
            input_channels=134,
            output_channels=67,
            n_history=1,
            hidden_dim=16,
            edge_dim=6,
            heads=4,
            encoder_blocks=1,
            decoder_blocks=1,
            l0_blocks=1,
            l1_blocks=1,
            l2_blocks=1,
            l1_refine_blocks=1,
            l0_refine_blocks=1,
            num_graph_levels=4,
            use_l3=True,
            l3_blocks=1,
            l2_refine_after_l3_blocks=1,
        )
        x = torch.randn(1, 134, 72, 144)
        with torch.no_grad():
            y = model(x)
        self.assertEqual(tuple(y.shape), (1, 67, 72, 144))
        self.assertTrue(torch.isfinite(y).all().item())

    def _l3_model(self, hidden_dim: int = 16, **overrides: object) -> GraphWeatherModel:
        graph = GraphBundle(self.bundle_2p5_l3)
        kwargs = dict(
            graph=graph,
            grid_shape=(72, 144),
            input_channels=134,
            output_channels=67,
            n_history=1,
            hidden_dim=hidden_dim,
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

    def test_l3_blocks3_model_forward_shape(self) -> None:
        model = self._l3_model(l3_blocks=3)
        x = torch.randn(1, 134, 72, 144)
        with torch.no_grad():
            y = model(x)
        self.assertEqual(tuple(y.shape), (1, 67, 72, 144))
        self.assertTrue(torch.isfinite(y).all().item())

    def test_l0_refine_defaults_to_attention(self) -> None:
        model = self._l3_model(hidden_dim=128)
        self.assertIsInstance(model.processor.l0_refine[0], LocalGraphAttentionBlock)
        metadata = architecture_metadata({"num_graph_levels": 4, "use_l3": True, "hidden_dim": 128})
        self.assertEqual(metadata["l0_refine"]["type"], "attention")

    def test_l0_refine_nodewise_mlp_replaces_only_first_block(self) -> None:
        model = self._l3_model(
            hidden_dim=128,
            l0_refine_blocks=2,
            l0_refine={
                "type": "nodewise_mlp",
                "mlp_expansion": 2,
                "dropout": 0.0,
                "residual_scale_init": 0.1,
                "learnable_residual_scale": True,
            },
        )
        first = model.processor.l0_refine[0]
        second = model.processor.l0_refine[1]
        self.assertIsInstance(first, NodewiseRefineMLP)
        self.assertIsInstance(second, LocalGraphAttentionBlock)
        self.assertEqual(first.dim, 128)
        self.assertEqual(first.hidden_dim, 256)
        self.assertAlmostEqual(float(first.residual_scale.detach().item()), 0.1, places=6)
        self.assertEqual(model.l0_refine_config["type"], "nodewise_mlp")

    def test_heavy_l3_unet_forward_shape(self) -> None:
        model = self._l3_model(
            l0_blocks=3,
            l1_blocks=2,
            l2_blocks=2,
            l3_blocks=2,
            l2_refine_after_l3_blocks=2,
            l1_refine_blocks=2,
            l0_refine_blocks=1,
        )
        x = torch.randn(1, 134, 72, 144)
        with torch.no_grad():
            y = model(x)
        self.assertEqual(tuple(y.shape), (1, 67, 72, 144))
        self.assertTrue(torch.isfinite(y).all().item())

    def test_l3_hidden128_model_forward_shape_and_head_dim(self) -> None:
        model = self._l3_model(hidden_dim=128)
        self.assertEqual(model.hidden_dim, 128)
        self.assertEqual(model.num_heads, 4)
        self.assertEqual(model.head_dim, 32)
        x = torch.randn(1, 134, 72, 144)
        with torch.no_grad():
            y = model(x)
        self.assertEqual(tuple(y.shape), (1, 67, 72, 144))
        self.assertTrue(torch.isfinite(y).all().item())

    def test_scalar_gated_skip_forward_shape_and_gate_initialization(self) -> None:
        model = self._l3_model(
            hidden_dim=128,
            skip_fusion={"type": "scalar_gated", "init_scale": 1.0, "max_scale": 2.0},
        )
        values = model.fusion_gate_values()
        self.assertEqual(
            sorted(values),
            [
                "l1_to_l0_skip",
                "l1_to_l0_up",
                "l2_to_l1_skip",
                "l2_to_l1_up",
                "l3_to_l2_skip",
                "l3_to_l2_up",
            ],
        )
        for value in values.values():
            self.assertAlmostEqual(value, 1.0, places=6)
        x = torch.randn(1, 134, 72, 144)
        with torch.no_grad():
            y = model(x)
        self.assertEqual(tuple(y.shape), (1, 67, 72, 144))
        self.assertTrue(torch.isfinite(y).all().item())

    def test_sum_skip_fusion_is_exact_parent_plus_skip(self) -> None:
        fusion = ParentUnpoolFuse(3, skip_fusion={"type": "sum"})
        h_coarse = torch.tensor(
            [[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]],
        )
        parent_map = torch.tensor([0, 0, 1, 1])
        h_skip = torch.tensor(
            [[[0.1, 0.2, 0.3], [1.0, 1.0, 1.0], [2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]],
        )

        actual = fusion(h_coarse, parent_map, h_skip)
        expected = h_skip + h_coarse[:, parent_map, :]

        self.assertTrue(torch.equal(actual, expected))
        self.assertIsNone(fusion.fuse)
        self.assertEqual(sum(parameter.numel() for parameter in fusion.parameters()), 0)

    def test_sum_skip_fusion_model_forward(self) -> None:
        model = self._l3_model(
            hidden_dim=16,
            skip_fusion={"type": "sum"},
        )
        self.assertEqual(model.skip_fusion["type"], "sum")
        self.assertEqual(model.fusion_gate_values(), {})
        x = torch.randn(1, 134, 72, 144)
        with torch.no_grad():
            y = model(x)
        self.assertEqual(tuple(y.shape), (1, 67, 72, 144))
        self.assertTrue(torch.isfinite(y).all().item())

    def test_scalar_gated_skip_gate_parameters_receive_gradients(self) -> None:
        model = self._l3_model(
            hidden_dim=16,
            skip_fusion={"type": "scalar_gated", "init_scale": 1.0, "max_scale": 2.0},
        )
        gate_params = [(name, param) for name, param in model.named_parameters() if "gate_logit" in name]
        self.assertEqual(len(gate_params), 6)
        self.assertTrue(all(param.requires_grad for _, param in gate_params))
        x = torch.randn(1, 134, 72, 144)
        y = model(x)
        y.mean().backward()
        for name, param in gate_params:
            with self.subTest(name=name):
                self.assertIsNotNone(param.grad)

    def test_default_skip_fusion_has_no_gate_parameters(self) -> None:
        model = self._l3_model(hidden_dim=16)
        self.assertEqual(model.skip_fusion["type"], "default")
        self.assertEqual(model.fusion_gate_values(), {})
        gate_params = [(name, param) for name, param in model.named_parameters() if "gate_logit" in name]
        self.assertEqual(gate_params, [])

    def test_scalar_gated_skip_parameter_count_adds_six(self) -> None:
        base = self._l3_model(hidden_dim=128)
        gated = self._l3_model(
            hidden_dim=128,
            skip_fusion={"type": "scalar_gated", "init_scale": 1.0, "max_scale": 2.0},
        )
        base_params = sum(p.numel() for p in base.parameters() if p.requires_grad)
        gated_params = sum(p.numel() for p in gated.parameters() if p.requires_grad)
        self.assertEqual(gated_params, base_params + 6)
        self.assertLess(gated_params, 3_000_000)

    def test_scalar_gated_pooling_forward_shape_and_gate_initialization(self) -> None:
        model = self._l3_model(
            hidden_dim=128,
            pooling={"type": "scalar_gated_meanmax", "init_scale": 1.0, "max_scale": 2.0},
        )
        values = model.pooling_gate_values()
        self.assertEqual(
            sorted(values),
            [
                "l0_to_l1_max",
                "l0_to_l1_mean",
                "l1_to_l2_max",
                "l1_to_l2_mean",
                "l2_to_l3_max",
                "l2_to_l3_mean",
            ],
        )
        for value in values.values():
            self.assertAlmostEqual(value, 1.0, places=6)
        x = torch.randn(1, 134, 72, 144)
        with torch.no_grad():
            y = model(x)
        self.assertEqual(tuple(y.shape), (1, 67, 72, 144))
        self.assertTrue(torch.isfinite(y).all().item())

    def test_scalar_gated_pooling_gate_parameters_receive_gradients(self) -> None:
        model = self._l3_model(
            hidden_dim=16,
            pooling={"type": "scalar_gated_meanmax", "init_scale": 1.0, "max_scale": 2.0},
        )
        gate_params = [
            (name, param)
            for name, param in model.named_parameters()
            if "mean_gate_logit" in name or "max_gate_logit" in name
        ]
        self.assertEqual(len(gate_params), 6)
        self.assertTrue(all(param.requires_grad for _, param in gate_params))
        x = torch.randn(1, 134, 72, 144)
        y = model(x)
        y.mean().backward()
        for name, param in gate_params:
            with self.subTest(name=name):
                self.assertIsNotNone(param.grad)

    def test_default_pooling_has_no_gate_parameters(self) -> None:
        model = self._l3_model(hidden_dim=16)
        self.assertEqual(model.pooling["type"], "default")
        self.assertEqual(model.pooling_gate_values(), {})
        gate_params = [
            (name, param)
            for name, param in model.named_parameters()
            if "mean_gate_logit" in name or "max_gate_logit" in name
        ]
        self.assertEqual(gate_params, [])

    def test_scalar_gated_pooling_parameter_count_adds_six(self) -> None:
        base = self._l3_model(hidden_dim=128)
        gated = self._l3_model(
            hidden_dim=128,
            pooling={"type": "scalar_gated_meanmax", "init_scale": 1.0, "max_scale": 2.0},
        )
        base_params = sum(p.numel() for p in base.parameters() if p.requires_grad)
        gated_params = sum(p.numel() for p in gated.parameters() if p.requires_grad)
        self.assertEqual(gated_params, base_params + 6)
        self.assertLess(gated_params, 3_000_000)

    def test_lead_conditioning_sincos_grid_values(self) -> None:
        for lead in (1, 5, 10):
            with self.subTest(lead=lead):
                lead_sin, lead_cos = lead_sincos_values(lead, 10, dtype=torch.float64, device=torch.device("cpu"))
                expected_angle = 2.0 * np.pi * float(lead) / 10.0
                self.assertAlmostEqual(float(lead_sin.item()), float(np.sin(expected_angle)), places=12)
                self.assertAlmostEqual(float(lead_cos.item()), float(np.cos(expected_angle)), places=12)
        grid = build_lead_conditioning_grid(
            5,
            batch_size=2,
            height=72,
            width=144,
            max_lead=10,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        self.assertEqual(tuple(grid.shape), (2, 2, 72, 144))
        self.assertTrue(torch.allclose(grid[:, 0], torch.zeros_like(grid[:, 0]), atol=1.0e-6))
        self.assertTrue(torch.allclose(grid[:, 1], -torch.ones_like(grid[:, 1]), atol=1.0e-6))

    def test_lead_conditioned_forward_shape_and_input_channels(self) -> None:
        model = self._l3_model(
            hidden_dim=128,
            input_channels=136,
            lead_conditioning={"enabled": True, "type": "sincos_concat", "max_lead": 10},
        )
        self.assertTrue(model.lead_conditioning_enabled)
        self.assertEqual(model.input_channels, 136)
        self.assertEqual(model.adapter.state_input_channels, 134)
        self.assertEqual(model.adapter.appended_context_channels, 2)
        self.assertEqual(model.embed.in_features, 136)
        x = torch.randn(1, 136, 72, 144)
        with torch.no_grad():
            y = model(x)
        self.assertEqual(tuple(y.shape), (1, 67, 72, 144))
        self.assertTrue(torch.isfinite(y).all().item())

    def test_default_model_still_uses_134_input_channels(self) -> None:
        model = self._l3_model(hidden_dim=128)
        self.assertFalse(model.lead_conditioning_enabled)
        self.assertEqual(model.input_channels, 134)
        self.assertEqual(model.adapter.state_input_channels, 134)
        self.assertEqual(model.adapter.appended_context_channels, 0)
        self.assertEqual(model.embed.in_features, 134)

    def test_l3_hidden96_model_forward_shape_and_head_dim(self) -> None:
        model = self._l3_model(hidden_dim=96)
        self.assertEqual(model.hidden_dim, 96)
        self.assertEqual(model.num_heads, 4)
        self.assertEqual(model.head_dim, 24)
        x = torch.randn(1, 134, 72, 144)
        with torch.no_grad():
            y = model(x)
        self.assertEqual(tuple(y.shape), (1, 67, 72, 144))
        self.assertTrue(torch.isfinite(y).all().item())

    def test_l3_hidden160_model_forward_shape_head_dim_and_parameter_cap(self) -> None:
        model = self._l3_model(hidden_dim=160, heads=5)
        self.assertEqual(model.hidden_dim, 160)
        self.assertEqual(model.num_heads, 5)
        self.assertEqual(model.head_dim, 32)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertLess(params, 4_000_000)
        x = torch.randn(1, 134, 72, 144)
        with torch.no_grad():
            y = model(x)
        self.assertEqual(tuple(y.shape), (1, 67, 72, 144))
        self.assertTrue(torch.isfinite(y).all().item())

    def test_l3_block_count_parameter_ordering(self) -> None:
        previous_l3 = self._l3_model(hidden_dim=96)
        l3_blocks3 = self._l3_model(hidden_dim=96, l3_blocks=3)
        heavy = self._l3_model(
            hidden_dim=96,
            l0_blocks=3,
            l1_blocks=2,
            l2_blocks=2,
            l3_blocks=2,
            l2_refine_after_l3_blocks=2,
            l1_refine_blocks=2,
            l0_refine_blocks=1,
        )
        previous_params = sum(p.numel() for p in previous_l3.parameters() if p.requires_grad)
        blocks3_params = sum(p.numel() for p in l3_blocks3.parameters() if p.requires_grad)
        heavy_params = sum(p.numel() for p in heavy.parameters() if p.requires_grad)
        print(
            f"previous L3 params: {previous_params}; "
            f"L3 blocks=3 params: {blocks3_params}; heavy L3 params: {heavy_params}"
        )
        self.assertEqual(previous_l3.processor.l3_blocks_count, 1)
        self.assertEqual(previous_l3.processor.l2_refine_after_l3_blocks_count, 1)
        self.assertEqual(previous_l3.processor.l1_refine_blocks_count, 1)
        self.assertEqual(previous_l3.processor.l0_refine_blocks_count, 1)
        self.assertGreater(blocks3_params, previous_params)
        self.assertGreater(heavy_params, blocks3_params)
        self.assertLess(previous_params, 3_000_000)
        self.assertLess(blocks3_params, 3_000_000)
        self.assertLess(heavy_params, 3_000_000)

    def test_l3_hidden128_parameter_count_and_metadata(self) -> None:
        previous_l3 = self._l3_model(hidden_dim=96)
        hidden128 = self._l3_model(hidden_dim=128)
        previous_params = sum(p.numel() for p in previous_l3.parameters() if p.requires_grad)
        hidden128_params = sum(p.numel() for p in hidden128.parameters() if p.requires_grad)
        metadata = architecture_metadata(
            {
                "num_graph_levels": 4,
                "use_l3": True,
                "hidden_dim": 128,
                "num_heads": 4,
                "k_neighbors": 8,
            },
            num_parameters=hidden128_params,
        )
        self.assertGreater(hidden128_params, previous_params)
        self.assertLess(hidden128_params, 3_000_000)
        self.assertEqual(metadata["hidden_dim"], 128)
        self.assertEqual(metadata["num_heads"], 4)
        self.assertEqual(metadata["head_dim"], 32)
        self.assertEqual(metadata["k_neighbors"], 8)
        self.assertEqual(metadata["num_parameters"], hidden128_params)

    def test_l3_hidden160_metadata(self) -> None:
        hidden160 = self._l3_model(hidden_dim=160, heads=5)
        hidden160_params = sum(p.numel() for p in hidden160.parameters() if p.requires_grad)
        metadata = architecture_metadata(
            {
                "num_graph_levels": 4,
                "use_l3": True,
                "hidden_dim": 160,
                "num_heads": 5,
                "k_neighbors": 8,
            },
            num_parameters=hidden160_params,
        )
        self.assertEqual(metadata["hidden_dim"], 160)
        self.assertEqual(metadata["num_heads"], 5)
        self.assertEqual(metadata["head_dim"], 32)
        self.assertLess(hidden160_params, 4_000_000)

    def test_hidden_dim_must_be_divisible_by_num_heads(self) -> None:
        with self.assertRaisesRegex(ValueError, "hidden_dim=130 must be divisible by num_heads=4"):
            self._l3_model(hidden_dim=130)

    def test_checkpoint_architecture_mismatch_is_clear(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Checkpoint architecture mismatch"):
            validate_checkpoint_architecture(
                {"num_graph_levels": 3, "use_l3": False},
                {"num_graph_levels": 4, "use_l3": True},
            )
        with self.assertRaisesRegex(RuntimeError, "Checkpoint architecture mismatch"):
            validate_checkpoint_architecture(
                {"num_graph_levels": 4, "use_l3": True},
                {"num_graph_levels": 3, "use_l3": False},
            )
        with self.assertRaisesRegex(
            RuntimeError,
            "checkpoint has l3_blocks=1 but current config has l3_blocks=3",
        ):
            validate_checkpoint_architecture(
                {"num_graph_levels": 4, "use_l3": True, "l3_blocks": 1},
                {"num_graph_levels": 4, "use_l3": True, "l3_blocks": 3},
            )
        with self.assertRaisesRegex(
            RuntimeError,
            "checkpoint has l3_blocks=3 but current config has l3_blocks=1",
        ):
            validate_checkpoint_architecture(
                {"num_graph_levels": 4, "use_l3": True, "l3_blocks": 3},
                {"num_graph_levels": 4, "use_l3": True, "l3_blocks": 1},
            )
        with self.assertRaisesRegex(
            RuntimeError,
            "checkpoint has l0_blocks=3 but current config has l0_blocks=2",
        ):
            validate_checkpoint_architecture(
                {
                    "num_graph_levels": 4,
                    "use_l3": True,
                    "l0_blocks": 3,
                    "l1_blocks": 2,
                    "l2_blocks": 2,
                    "l3_blocks": 2,
                    "l2_refine_after_l3_blocks": 2,
                    "l1_refine_blocks": 2,
                    "l0_refine_blocks": 1,
                },
                {"num_graph_levels": 4, "use_l3": True},
            )
        with self.assertRaisesRegex(
            RuntimeError,
            "checkpoint has hidden_dim=96 but current config has hidden_dim=128",
        ):
            validate_checkpoint_architecture(
                {"num_graph_levels": 4, "use_l3": True, "hidden_dim": 96},
                {"num_graph_levels": 4, "use_l3": True, "hidden_dim": 128},
            )
        with self.assertRaisesRegex(
            RuntimeError,
            "checkpoint has hidden_dim=128 but current config has hidden_dim=96",
        ):
            validate_checkpoint_architecture(
                {"num_graph_levels": 4, "use_l3": True, "hidden_dim": 128},
                {"num_graph_levels": 4, "use_l3": True, "hidden_dim": 96},
            )
        with self.assertRaisesRegex(
            RuntimeError,
            "checkpoint skip_fusion.type='default' but current config has skip_fusion.type='scalar_gated'",
        ):
            validate_checkpoint_architecture(
                {"num_graph_levels": 4, "use_l3": True, "skip_fusion": {"type": "default"}},
                {
                    "num_graph_levels": 4,
                    "use_l3": True,
                    "skip_fusion": {"type": "scalar_gated", "init_scale": 1.0, "max_scale": 2.0},
                },
            )
        with self.assertRaisesRegex(
            RuntimeError,
            "checkpoint pooling.type='default' but current config has pooling.type='scalar_gated_meanmax'",
        ):
            validate_checkpoint_architecture(
                {"num_graph_levels": 4, "use_l3": True, "pooling": {"type": "default"}},
                {
                    "num_graph_levels": 4,
                    "use_l3": True,
                    "pooling": {"type": "scalar_gated_meanmax", "init_scale": 1.0, "max_scale": 2.0},
                },
            )
        with self.assertRaisesRegex(
            RuntimeError,
            "checkpoint lead_conditioning.enabled=False but current config has lead_conditioning.enabled=True",
        ):
            validate_checkpoint_architecture(
                {
                    "num_graph_levels": 4,
                    "use_l3": True,
                    "input_channels": 134,
                    "output_channels": 67,
                    "lead_conditioning": {"enabled": False},
                },
                {
                    "num_graph_levels": 4,
                    "use_l3": True,
                    "input_channels": 136,
                    "output_channels": 67,
                    "lead_conditioning": {"enabled": True, "type": "sincos_concat", "max_lead": 10},
                },
            )
        with self.assertRaisesRegex(
            RuntimeError,
            "Checkpoint input_channels=134, current input_channels=136",
        ):
            validate_checkpoint_architecture(
                {
                    "num_graph_levels": 4,
                    "use_l3": True,
                    "input_channels": 134,
                    "output_channels": 67,
                    "lead_conditioning": {"enabled": True, "type": "sincos_concat", "max_lead": 10},
                },
                {
                    "num_graph_levels": 4,
                    "use_l3": True,
                    "input_channels": 136,
                    "output_channels": 67,
                    "lead_conditioning": {"enabled": True, "type": "sincos_concat", "max_lead": 10},
                },
            )
        with self.assertRaisesRegex(
            RuntimeError,
            "checkpoint l0_refine.type='attention' but current config has l0_refine.type='nodewise_mlp'",
        ):
            validate_checkpoint_architecture(
                {
                    "num_graph_levels": 4,
                    "use_l3": True,
                    "l0_refine": {"type": "attention"},
                },
                {
                    "num_graph_levels": 4,
                    "use_l3": True,
                    "l0_refine": {
                        "type": "nodewise_mlp",
                        "mlp_expansion": 2,
                        "dropout": 0.0,
                        "residual_scale_init": 0.1,
                        "learnable_residual_scale": True,
                    },
                },
            )

    def test_checkpoint_mixed_level_k_mismatch_is_clear(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            r"checkpoint has level_k_neighbors=\[8, 8, 8, 8\] "
            r"but current config has level_k_neighbors=\[8, 8, 8, 24\]",
        ):
            validate_checkpoint_architecture(
                {
                    "num_graph_levels": 4,
                    "use_l3": True,
                    "k_neighbors": 8,
                    "level_k_neighbors": [8, 8, 8, 8],
                },
                {
                    "num_graph_levels": 4,
                    "use_l3": True,
                    "k_neighbors": 8,
                    "level_k_neighbors": [8, 8, 8, 24],
                },
            )


class DenseL3K24GraphTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.latitudes, cls.longitudes = cell_center_lat_lon(72, 144)
        cls.bundle = build_graph_bundle(
            cls.latitudes,
            cls.longitudes,
            k=8,
            level_k_neighbors=[8, 8, 8, 24],
            resolution=2.5,
            connectivity_strategy=HYBRID_ROW_AWARE_KNN,
            resolution_mode="2p5",
            num_graph_levels=4,
        )

    def test_dense_l3_counts_and_metadata(self) -> None:
        expected_nodes = {"L0": 10368, "L1": 2592, "L2": 648, "L3": 162}
        expected_edges = {"L0": 82944, "L1": 20736, "L2": 5184, "L3": 3888}
        expected_k = {"L0": 8, "L1": 8, "L2": 8, "L3": 24}
        metadata = self.bundle["metadata"]
        self.assertEqual(metadata["num_graph_levels"], 4)
        self.assertTrue(metadata["use_l3"])
        self.assertEqual(metadata["level_shapes"], [[72, 144], [36, 72], [18, 36], [9, 18]])
        self.assertEqual(metadata["node_counts"], [10368, 2592, 648, 162])
        self.assertEqual(metadata["level_k_neighbors"], [8, 8, 8, 24])
        self.assertEqual(metadata["edge_counts"], [82944, 20736, 5184, 3888])
        self.assertEqual(metadata["connectivity_strategy"], HYBRID_ROW_AWARE_KNN)
        self.assertEqual(metadata["graph_format_version"], 4)
        for level_name, level in self.bundle["levels"].items():
            with self.subTest(level=level_name):
                self.assertEqual(int(level["num_nodes"]), expected_nodes[level_name])
                self.assertEqual(int(level["k"]), expected_k[level_name])
                self.assertEqual(int(level["edge_index"].shape[1]), expected_edges[level_name])
                self.assertEqual(_component_count(level), 1)

    def test_dense_l3_neighbor_uniqueness(self) -> None:
        l3 = self.bundle["levels"]["L3"]
        neighbors = _neighbors(l3)
        self.assertEqual(neighbors.shape, (162, 24))
        for target in range(162):
            row = neighbors[target].tolist()
            self.assertNotIn(target, row)
            self.assertEqual(len(set(row)), 24)

    def test_dense_l3_graph_cache_mismatch_detects_old_k8_graph(self) -> None:
        expected = expected_graph_metadata(
            self.latitudes,
            self.longitudes,
            k=8,
            level_k_neighbors=[8, 8, 8, 24],
            resolution=2.5,
            connectivity_strategy=HYBRID_ROW_AWARE_KNN,
            resolution_mode="2p5",
            num_graph_levels=4,
        )
        old = build_graph_bundle(
            self.latitudes,
            self.longitudes,
            k=8,
            resolution=2.5,
            connectivity_strategy=HYBRID_ROW_AWARE_KNN,
            resolution_mode="2p5",
            num_graph_levels=4,
        )
        mismatches = validate_graph_cache_metadata(old, expected)
        self.assertTrue(any("level_k_neighbors" in item for item in mismatches))
        self.assertTrue(any("edge_counts" in item for item in mismatches))

    def test_dense_l3_mixed_k_forward_hidden128(self) -> None:
        graph = GraphBundle(self.bundle)
        self.assertEqual(graph.L0.k, 8)
        self.assertEqual(graph.L1.k, 8)
        self.assertEqual(graph.L2.k, 8)
        self.assertIsNotNone(graph.L3)
        self.assertEqual(graph.L3.k, 24)
        model = GraphWeatherModel(
            graph=graph,
            grid_shape=(72, 144),
            input_channels=134,
            output_channels=67,
            n_history=1,
            hidden_dim=128,
            edge_dim=6,
            heads=4,
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
        x = torch.randn(1, 134, 72, 144)
        with torch.no_grad():
            y = model(x)
        self.assertEqual(tuple(y.shape), (1, 67, 72, 144))
        self.assertTrue(torch.isfinite(y).all().item())


class Ratio15L4GraphTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.latitudes, cls.longitudes = cell_center_lat_lon(72, 144)
        cls.level_shapes = [[72, 144], [48, 96], [32, 64], [21, 42], [14, 28]]
        cls.level_k = [8, 8, 8, 16, 24]
        cls.bundle = build_graph_bundle(
            cls.latitudes,
            cls.longitudes,
            k=8,
            level_k_neighbors=cls.level_k,
            level_shapes=cls.level_shapes,
            hierarchy_type="ratio15_l4",
            use_l4_ratio15=True,
            resolution=2.5,
            connectivity_strategy=HYBRID_ROW_AWARE_KNN,
            resolution_mode="2p5",
            num_graph_levels=5,
        )

    def test_ratio15_l4_counts_and_metadata(self) -> None:
        expected_nodes = {"L0": 10368, "L1": 4608, "L2": 2048, "L3": 882, "L4": 392}
        expected_edges = {"L0": 82944, "L1": 36864, "L2": 16384, "L3": 14112, "L4": 9408}
        expected_k = {"L0": 8, "L1": 8, "L2": 8, "L3": 16, "L4": 24}
        metadata = self.bundle["metadata"]
        self.assertEqual(metadata["hierarchy_type"], "ratio15_l4")
        self.assertTrue(metadata["use_l4_ratio15"])
        self.assertEqual(metadata["num_graph_levels"], 5)
        self.assertTrue(metadata["use_l3"])
        self.assertTrue(metadata["use_l4"])
        self.assertEqual(metadata["level_shapes"], self.level_shapes)
        self.assertEqual(metadata["node_counts"], [10368, 4608, 2048, 882, 392])
        self.assertEqual(metadata["level_k_neighbors"], self.level_k)
        self.assertEqual(metadata["edge_counts"], [82944, 36864, 16384, 14112, 9408])
        self.assertEqual(metadata["connectivity_strategy"], HYBRID_ROW_AWARE_KNN)
        self.assertEqual(metadata["pooling_map_strategy"], "proportional_parent_index")
        self.assertEqual(metadata["graph_format_version"], "ratio15_l4_v1")
        for level_name, level in self.bundle["levels"].items():
            with self.subTest(level=level_name):
                self.assertEqual(int(level["num_nodes"]), expected_nodes[level_name])
                self.assertEqual(int(level["k"]), expected_k[level_name])
                self.assertEqual(int(level["edge_index"].shape[1]), expected_edges[level_name])
                self.assertEqual(_component_count(level), 1)

    def test_ratio15_l4_pool_maps_are_valid_parent_indices(self) -> None:
        expected_lengths = {
            "L0_to_L1": 10368,
            "L1_to_L2": 4608,
            "L2_to_L3": 2048,
            "L3_to_L4": 882,
        }
        expected_parents = {
            "L0_to_L1": 4608,
            "L1_to_L2": 2048,
            "L2_to_L3": 882,
            "L3_to_L4": 392,
        }
        for name, pool_map in self.bundle["pool"].items():
            with self.subTest(pool=name):
                num_parents = expected_parents[name]
                self.assertEqual(int(pool_map.numel()), expected_lengths[name])
                self.assertGreaterEqual(int(pool_map.min()), 0)
                self.assertLess(int(pool_map.max()), num_parents)
                counts = torch.bincount(pool_map, minlength=num_parents)
                self.assertGreaterEqual(int(counts.min()), 1)
                self.assertGreaterEqual(int(counts.max()), 2)
        stats = self.bundle["metadata"]["pool_child_count_stats"]
        self.assertEqual(sorted(stats), ["L0_to_L1", "L1_to_L2", "L2_to_L3", "L3_to_L4"])

    def test_ratio15_l4_expected_metadata_matches_built_bundle(self) -> None:
        expected = expected_graph_metadata(
            self.latitudes,
            self.longitudes,
            k=8,
            level_k_neighbors=self.level_k,
            level_shapes=self.level_shapes,
            hierarchy_type="ratio15_l4",
            use_l4_ratio15=True,
            resolution=2.5,
            connectivity_strategy=HYBRID_ROW_AWARE_KNN,
            resolution_mode="2p5",
            num_graph_levels=5,
        )
        self.assertEqual(validate_graph_cache_metadata(self.bundle, expected), [])

    def test_ratio15_l4_graph_bundle_and_forward(self) -> None:
        graph = GraphBundle(self.bundle)
        self.assertEqual(graph.num_graph_levels, 5)
        self.assertTrue(graph.use_l3)
        self.assertTrue(graph.use_l4)
        self.assertIsNotNone(graph.level4)
        self.assertIsNotNone(graph.pool_l3_to_l4)
        self.assertEqual(graph.L4.num_nodes, 392)
        model = GraphWeatherModel(
            graph=graph,
            grid_shape=(72, 144),
            input_channels=134,
            output_channels=67,
            n_history=1,
            hidden_dim=16,
            edge_dim=6,
            heads=4,
            encoder_blocks=1,
            decoder_blocks=1,
            l0_blocks=1,
            l1_blocks=1,
            l2_blocks=1,
            l1_refine_blocks=1,
            l0_refine_blocks=1,
            num_graph_levels=5,
            l3_blocks=1,
            l4_blocks=1,
            l3_refine_after_l4_blocks=1,
            l2_refine_after_l3_blocks=1,
            pooling={"type": "parent_index_meanmax", "mean_type": "area_weighted", "include_max": True},
        )
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertLess(params, 3_100_000)
        x = torch.randn(1, 134, 72, 144)
        with torch.no_grad():
            y = model(x)
        self.assertEqual(tuple(y.shape), (1, 67, 72, 144))
        self.assertTrue(torch.isfinite(y).all().item())


class L4_72_36_24_18_9_GraphTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.latitudes, cls.longitudes = cell_center_lat_lon(72, 144)
        cls.level_shapes = [[72, 144], [36, 72], [24, 48], [18, 36], [9, 18]]
        cls.level_k = [8, 8, 8, 12, 24]
        cls.bundle = build_graph_bundle(
            cls.latitudes,
            cls.longitudes,
            k=8,
            level_k_neighbors=cls.level_k,
            level_shapes=cls.level_shapes,
            hierarchy_type="l4_72_36_24_18_9",
            resolution=2.5,
            connectivity_strategy=HYBRID_ROW_AWARE_KNN,
            resolution_mode="2p5",
            num_graph_levels=5,
        )

    def test_counts_and_metadata(self) -> None:
        expected_nodes = {"L0": 10368, "L1": 2592, "L2": 1152, "L3": 648, "L4": 162}
        expected_edges = {"L0": 82944, "L1": 20736, "L2": 9216, "L3": 7776, "L4": 3888}
        expected_k = {"L0": 8, "L1": 8, "L2": 8, "L3": 12, "L4": 24}
        metadata = self.bundle["metadata"]
        self.assertEqual(metadata["hierarchy_type"], "l4_72_36_24_18_9")
        self.assertFalse(metadata["use_l4_ratio15"])
        self.assertEqual(metadata["num_graph_levels"], 5)
        self.assertTrue(metadata["use_l3"])
        self.assertTrue(metadata["use_l4"])
        self.assertEqual(metadata["level_shapes"], self.level_shapes)
        self.assertEqual(metadata["node_counts"], [10368, 2592, 1152, 648, 162])
        self.assertEqual(metadata["level_k_neighbors"], self.level_k)
        self.assertEqual(metadata["edge_counts"], [82944, 20736, 9216, 7776, 3888])
        self.assertEqual(metadata["pooling_map_strategy"], "proportional_parent_index")
        self.assertEqual(metadata["graph_format_version"], "l4_72_36_24_18_9_v1")
        for level_name, level in self.bundle["levels"].items():
            with self.subTest(level=level_name):
                self.assertEqual(int(level["num_nodes"]), expected_nodes[level_name])
                self.assertEqual(int(level["k"]), expected_k[level_name])
                self.assertEqual(int(level["edge_index"].shape[1]), expected_edges[level_name])
                self.assertEqual(_component_count(level), 1)

    def test_pool_maps_are_valid(self) -> None:
        expected_lengths = {
            "L0_to_L1": 10368,
            "L1_to_L2": 2592,
            "L2_to_L3": 1152,
            "L3_to_L4": 648,
        }
        expected_parents = {
            "L0_to_L1": 2592,
            "L1_to_L2": 1152,
            "L2_to_L3": 648,
            "L3_to_L4": 162,
        }
        for name, pool_map in self.bundle["pool"].items():
            with self.subTest(pool=name):
                num_parents = expected_parents[name]
                self.assertEqual(int(pool_map.numel()), expected_lengths[name])
                self.assertGreaterEqual(int(pool_map.min()), 0)
                self.assertLess(int(pool_map.max()), num_parents)
                counts = torch.bincount(pool_map, minlength=num_parents)
                self.assertGreaterEqual(int(counts.min()), 1)
                self.assertGreaterEqual(int(counts.max()), 2)

    def test_expected_metadata_matches_built_bundle(self) -> None:
        expected = expected_graph_metadata(
            self.latitudes,
            self.longitudes,
            k=8,
            level_k_neighbors=self.level_k,
            level_shapes=self.level_shapes,
            hierarchy_type="l4_72_36_24_18_9",
            resolution=2.5,
            connectivity_strategy=HYBRID_ROW_AWARE_KNN,
            resolution_mode="2p5",
            num_graph_levels=5,
        )
        self.assertEqual(validate_graph_cache_metadata(self.bundle, expected), [])

    def test_forward(self) -> None:
        graph = GraphBundle(self.bundle)
        self.assertEqual(graph.num_graph_levels, 5)
        self.assertTrue(graph.use_l4)
        self.assertEqual(graph.L4.num_nodes, 162)
        model = GraphWeatherModel(
            graph=graph,
            grid_shape=(72, 144),
            input_channels=134,
            output_channels=67,
            n_history=1,
            hidden_dim=16,
            edge_dim=6,
            heads=4,
            encoder_blocks=1,
            decoder_blocks=1,
            l0_blocks=1,
            l1_blocks=1,
            l2_blocks=1,
            l1_refine_blocks=1,
            l0_refine_blocks=1,
            num_graph_levels=5,
            l3_blocks=1,
            l4_blocks=1,
            l3_refine_after_l4_blocks=1,
            l2_refine_after_l3_blocks=1,
            pooling={"type": "parent_index_meanmax", "mean_type": "area_weighted", "include_max": True},
        )
        x = torch.randn(1, 134, 72, 144)
        with torch.no_grad():
            y = model(x)
        self.assertEqual(tuple(y.shape), (1, 67, 72, 144))
        self.assertTrue(torch.isfinite(y).all().item())


if __name__ == "__main__":
    unittest.main()
