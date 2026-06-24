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
    regular_lat_lon,
)
from src.graph_bundle import GraphBundle  # noqa: E402
from src.models import GraphWeatherModel  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()
