"""P-mesh tests: grid<->icosphere-mesh encode/decode (mesh_builder + BipartiteMP +
edge-mask). CPU-only, no data. unittest-style (repo convention).

Verifies: icosphere counts/degree, the edge-mask makes the padded slot inert, a
full mesh-mode forward returns [B, out, H, W], and that a lat-lon-style level
(edge_mask=None) leaves LocalGraphAttention unchanged.
"""
from __future__ import annotations

import copy
import math
import logging
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import mesh_builder  # noqa: E402
from src.architecture import (  # noqa: E402
    architecture_metadata,
    normalize_model_config_dict,
    validate_checkpoint_architecture,
)
from src.config import YParams  # noqa: E402
from src.graph_bundle import GraphBundle, GraphLevel  # noqa: E402
from src.graph_builder import graph_topology_metadata  # noqa: E402
from src.evaluator import GraphWeatherEvaluator  # noqa: E402
from src.layers import LocalGraphAttention  # noqa: E402
from src.mesh_layers import BipartiteMP, FixedBipartiteRemap, MLPLayerNorm  # noqa: E402
from src.models import GraphWeatherModel  # noqa: E402


def _bundle(
    refinement=2,
    levels=3,
    grid=(24, 48),
    bipartite_edge_features=mesh_builder.LEGACY_BIPARTITE_EDGE_FEATURES,
    coarse_level_connectivity=mesh_builder.NATIVE_COARSE_CONNECTIVITY,
    grid_attention_k_neighbors=None,
    bipartite_mapping_type=mesh_builder.GRAPHCAST_RADIUS_MAPPING,
):
    return mesh_builder.build_mesh_bundle(
        refinement=refinement,
        num_graph_levels=levels,
        grid_shape=grid,
        resolution_mode="2p5",
        bipartite_edge_features=bipartite_edge_features,
        coarse_level_connectivity=coarse_level_connectivity,
        grid_attention_k_neighbors=grid_attention_k_neighbors,
        bipartite_mapping_type=bipartite_mapping_type,
    )


class IcosphereGeometryTest(unittest.TestCase):
    def test_public_build_icosphere_r1(self):
        xyz, lat_lon, native_edges, faces = mesh_builder.build_icosphere(1)
        self.assertEqual(tuple(xyz.shape), (42, 3))
        self.assertEqual(tuple(lat_lon.shape), (42, 2))
        self.assertEqual(tuple(native_edges.shape), (2, 120))
        self.assertEqual(tuple(faces.shape), (80, 3))
        self.assertTrue(torch.allclose(torch.linalg.vector_norm(xyz, dim=1), torch.ones(42), atol=1e-6))

    def test_node_counts_and_degrees(self):
        b = _bundle(refinement=2, levels=3)
        # M2, M1, M0
        self.assertEqual(b["levels"]["L0"]["num_nodes"], 162)
        self.assertEqual(b["levels"]["L1"]["num_nodes"], 42)
        self.assertEqual(b["levels"]["L2"]["num_nodes"], 12)
        # every level: exactly 12 degree-5 nodes -> 12 masked (dummy) slots
        for name in ("L0", "L1", "L2"):
            lvl = b["levels"][name]
            n = lvl["num_nodes"]
            self.assertEqual(tuple(lvl["edge_index"].shape), (2, n * 6))
            self.assertEqual(tuple(lvl["edge_mask"].shape), (n, 6))
            invalid = int((~lvl["edge_mask"]).sum().item())
            self.assertEqual(invalid, 12, f"{name}: expected 12 masked slots, got {invalid}")

    def test_metadata_is_mesh(self):
        b = _bundle()
        self.assertEqual(b["metadata"]["graph_mode"], "mesh")
        self.assertEqual(b["metadata"]["num_graph_levels"], 3)
        self.assertEqual(list(b["metadata"]["grid_shape"]), [24, 48])
        self.assertEqual(
            b["metadata"]["bipartite_edge_features"],
            mesh_builder.LEGACY_BIPARTITE_EDGE_FEATURES,
        )
        self.assertEqual(b["metadata"]["bipartite_edge_dim"], 6)

    def test_optional_grid_attention_graph_uses_requested_dense_grid_topology(self):
        H, W = 12, 24
        b = _bundle(
            refinement=2,
            levels=3,
            grid=(H, W),
            grid_attention_k_neighbors=8,
        )
        level = b["grid"]["attention_level"]
        self.assertEqual(level["num_nodes"], H * W)
        self.assertEqual(level["k"], 8)
        self.assertEqual(tuple(level["edge_index"].shape), (2, H * W * 8))
        self.assertEqual(tuple(level["edge_attr"].shape), (H * W * 8, 6))
        self.assertIs(b["metadata"]["grid_attention_graph"], True)
        self.assertEqual(b["metadata"]["grid_attention_k_neighbors"], 8)
        self.assertEqual(
            b["metadata"]["grid_attention_connectivity_strategy"],
            "hybrid_row_aware_knn",
        )

        graph = GraphBundle(b)
        self.assertIsNotNone(graph.grid_attention_graph)
        self.assertEqual(graph.grid_attention_graph.num_nodes, H * W)
        self.assertEqual(graph.grid_attention_graph.k, 8)

        without_grid_attention = _bundle(
            refinement=2,
            levels=3,
            grid=(H, W),
        )
        expected = mesh_builder.expected_mesh_metadata(
            refinement=2,
            num_graph_levels=3,
            grid_shape=(H, W),
            grid_lat_lon=b["grid"]["lat_lon"],
            g2m_radius_factor=0.6,
            resolution_mode="2p5",
            grid_attention_k_neighbors=8,
        )
        mismatches = mesh_builder.validate_mesh_cache_metadata(
            without_grid_attention,
            expected,
        )
        self.assertTrue(any("grid_attention_graph" in item for item in mismatches))

    def test_graphcast_edges_append_receiver_local_geometry(self):
        legacy = _bundle(refinement=2, levels=3, grid=(12, 24))
        graphcast = _bundle(
            refinement=2,
            levels=3,
            grid=(12, 24),
            bipartite_edge_features=mesh_builder.GRAPHCAST_BIPARTITE_EDGE_FEATURES,
        )
        for direction in ("g2m", "m2g"):
            legacy_attr = legacy[direction]["edge_attr"]
            graphcast_attr = graphcast[direction]["edge_attr"]
            self.assertEqual(tuple(graphcast_attr.shape), (legacy_attr.shape[0], 10))
            self.assertTrue(torch.equal(graphcast_attr[:, :6], legacy_attr))

            normalized_distance = graphcast_attr[:, 6]
            relative_local_xyz = graphcast_attr[:, 7:10]
            self.assertAlmostEqual(float(normalized_distance.max()), 1.0, places=6)
            self.assertTrue(bool((normalized_distance > 0.0).all()))
            self.assertTrue(
                torch.allclose(
                    torch.linalg.vector_norm(relative_local_xyz, dim=1),
                    normalized_distance,
                    atol=2.0e-6,
                    rtol=2.0e-6,
                )
            )
        self.assertEqual(graphcast["metadata"]["bipartite_edge_dim"], 10)

    def test_legacy_and_graphcast_edge_caches_are_distinct(self):
        legacy = _bundle(refinement=2, levels=3, grid=(12, 24))
        expected_graphcast = mesh_builder.expected_mesh_metadata(
            refinement=2,
            num_graph_levels=3,
            grid_shape=(12, 24),
            grid_lat_lon=legacy["grid"]["lat_lon"],
            g2m_radius_factor=0.6,
            resolution_mode="2p5",
            bipartite_edge_features=mesh_builder.GRAPHCAST_BIPARTITE_EDGE_FEATURES,
        )
        mismatches = mesh_builder.validate_mesh_cache_metadata(
            legacy,
            expected_graphcast,
        )
        self.assertTrue(any("bipartite_edge_features" in item for item in mismatches))
        self.assertTrue(any("bipartite_edge_dim" in item for item in mismatches))

    def test_fixed_spherical_weights_cover_every_destination_and_preserve_constants(self):
        height, width = 12, 24
        bundle = _bundle(
            refinement=2,
            levels=3,
            grid=(height, width),
            bipartite_mapping_type=mesh_builder.FIXED_SPHERICAL_MAPPING,
        )
        self.assertEqual(
            bundle["metadata"]["bipartite_mapping_type"],
            mesh_builder.FIXED_SPHERICAL_MAPPING,
        )
        for direction, destination_count in (
            ("g2m", bundle["levels"]["L0"]["num_nodes"]),
            ("m2g", height * width),
        ):
            edges = bundle[direction]
            self.assertIn("edge_weight", edges)
            sums = torch.zeros(destination_count)
            sums.index_add_(0, edges["edge_index"][1], edges["edge_weight"])
            self.assertTrue(
                torch.allclose(sums, torch.ones_like(sums), atol=2.0e-6, rtol=0.0)
            )

        graph = GraphBundle(bundle)
        remap = FixedBipartiteRemap()
        grid_constant = torch.ones(2, height * width, 3)
        mesh_constant = remap(
            grid_constant,
            graph.g2m_edge_index,
            graph.g2m_edge_weight,
            graph.L0.num_nodes,
        )
        grid_roundtrip = remap(
            mesh_constant,
            graph.m2g_edge_index,
            graph.m2g_edge_weight,
            height * width,
        )
        self.assertTrue(torch.allclose(mesh_constant, torch.ones_like(mesh_constant), atol=2.0e-6))
        self.assertTrue(torch.allclose(grid_roundtrip, torch.ones_like(grid_roundtrip), atol=2.0e-6))

    def test_fixed_spherical_and_radius_caches_are_distinct(self):
        radius = _bundle(refinement=2, levels=3, grid=(12, 24))
        expected_fixed = mesh_builder.expected_mesh_metadata(
            refinement=2,
            num_graph_levels=3,
            grid_shape=(12, 24),
            grid_lat_lon=radius["grid"]["lat_lon"],
            g2m_radius_factor=0.6,
            resolution_mode="2p5",
            bipartite_mapping_type=mesh_builder.FIXED_SPHERICAL_MAPPING,
        )
        mismatches = mesh_builder.validate_mesh_cache_metadata(
            radius,
            expected_fixed,
        )
        self.assertTrue(any("bipartite_mapping_type" in item for item in mismatches))

    def test_pool_map_uses_nearest_retained_parent(self):
        b = _bundle(refinement=2, levels=3)
        fine_xyz, _, _, _ = mesh_builder.build_icosphere(2)
        coarse_xyz, _, _, _ = mesh_builder.build_icosphere(1)
        parent = b["pool"]["L0_to_L1"]
        coarse_n = coarse_xyz.shape[0]

        self.assertTrue(torch.equal(parent[:coarse_n], torch.arange(coarse_n)))
        chosen_similarity = (fine_xyz[coarse_n:] * coarse_xyz[parent[coarse_n:]]).sum(dim=1)
        nearest_similarity = fine_xyz[coarse_n:] @ coarse_xyz.T
        self.assertTrue(
            torch.allclose(chosen_similarity, nearest_similarity.max(dim=1).values, atol=1e-6)
        )

    def test_full_m1_is_complete_without_changing_finer_levels(self):
        bundle = _bundle(
            refinement=3,
            levels=3,
            grid=(12, 24),
            coarse_level_connectivity=mesh_builder.FULL_M1_COARSE_CONNECTIVITY,
        )
        self.assertEqual(bundle["metadata"]["node_counts"], [642, 162, 42])
        self.assertEqual(bundle["metadata"]["level_k_neighbors"], [6, 6, 41])
        self.assertEqual(bundle["metadata"]["edge_counts"], [3852, 972, 1722])
        self.assertEqual(
            bundle["metadata"]["coarse_level_connectivity"],
            mesh_builder.FULL_M1_COARSE_CONNECTIVITY,
        )
        self.assertEqual(
            bundle["metadata"]["graph_connectivity_strategy"],
            "native_icosphere_with_full_m1",
        )

        self.assertEqual(bundle["levels"]["L0"]["k"], 6)
        self.assertEqual(bundle["levels"]["L1"]["k"], 6)
        coarse = bundle["levels"]["L2"]
        self.assertEqual(coarse["k"], 41)
        self.assertEqual(tuple(coarse["edge_index"].shape), (2, 1722))
        self.assertEqual(tuple(coarse["edge_mask"].shape), (42, 41))
        self.assertTrue(bool(coarse["edge_mask"].all()))
        self.assertTrue(torch.isfinite(coarse["edge_attr"]).all())

        src, dst = coarse["edge_index"]
        self.assertFalse(bool((src == dst).any()))
        all_nodes = set(range(42))
        for node in range(42):
            self.assertEqual(set(src[dst == node].tolist()), all_nodes - {node})

    def test_full_m1_requires_m1_as_the_coarsest_level(self):
        with self.assertRaisesRegex(ValueError, "coarsest level to be M1"):
            _bundle(
                refinement=4,
                levels=3,
                grid=(12, 24),
                coarse_level_connectivity=mesh_builder.FULL_M1_COARSE_CONNECTIVITY,
            )

    def test_native_and_full_m1_caches_are_distinct(self):
        native = _bundle(refinement=3, levels=3, grid=(12, 24))
        expected_full = mesh_builder.expected_mesh_metadata(
            refinement=3,
            num_graph_levels=3,
            grid_shape=(12, 24),
            grid_lat_lon=native["grid"]["lat_lon"],
            g2m_radius_factor=0.6,
            resolution_mode="2p5",
            coarse_level_connectivity=mesh_builder.FULL_M1_COARSE_CONNECTIVITY,
        )
        mismatches = mesh_builder.validate_mesh_cache_metadata(native, expected_full)
        self.assertTrue(any("coarse_level_connectivity" in item for item in mismatches))


class EdgeMaskInertnessTest(unittest.TestCase):
    @staticmethod
    def _legacy_forward(attn, h, graph):
        bsz, num_nodes, dim = h.shape
        k_neighbors = graph.k
        src = graph.edge_index[0].reshape(num_nodes, k_neighbors)
        q = attn.q_proj(h).reshape(bsz, num_nodes, attn.heads, attn.head_dim)
        key = attn.k_proj(h).reshape(bsz, num_nodes, attn.heads, attn.head_dim)
        value = attn.v_proj(h).reshape(bsz, num_nodes, attn.heads, attn.head_dim)
        key_src = key[:, src]
        value_src = value[:, src]
        edge_attr = graph.edge_attr.reshape(num_nodes, k_neighbors, -1)
        edge_key = attn.edge_k(edge_attr).reshape(num_nodes, k_neighbors, attn.heads, attn.head_dim)
        edge_value = attn.edge_v(edge_attr).reshape(num_nodes, k_neighbors, attn.heads, attn.head_dim)
        edge_bias = attn.edge_bias(edge_attr).reshape(num_nodes, k_neighbors, attn.heads)
        scores = (
            (q[:, :, None] * (key_src + edge_key[None])).sum(dim=-1)
            / math.sqrt(attn.head_dim)
        )
        weights = torch.softmax(scores + edge_bias[None], dim=2)
        out = (weights[..., None] * (value_src + edge_value[None])).sum(dim=2)
        return attn.out_proj(out.reshape(bsz, num_nodes, dim))

    @staticmethod
    def _ragged_forward(attn, h, graph):
        bsz, num_nodes, dim = h.shape
        k_neighbors = graph.k
        src = graph.edge_index[0].reshape(num_nodes, k_neighbors)
        valid = graph.edge_mask.reshape(num_nodes, k_neighbors)
        q = attn.q_proj(h).reshape(bsz, num_nodes, attn.heads, attn.head_dim)
        key = attn.k_proj(h).reshape(bsz, num_nodes, attn.heads, attn.head_dim)
        value = attn.v_proj(h).reshape(bsz, num_nodes, attn.heads, attn.head_dim)
        edge_attr = graph.edge_attr.reshape(num_nodes, k_neighbors, -1)
        result = torch.empty_like(q)
        for node in range(num_nodes):
            node_src = src[node, valid[node]]
            node_edge = edge_attr[node, valid[node]]
            edge_key = attn.edge_k(node_edge).reshape(-1, attn.heads, attn.head_dim)
            edge_value = attn.edge_v(node_edge).reshape(-1, attn.heads, attn.head_dim)
            edge_bias = attn.edge_bias(node_edge).reshape(-1, attn.heads)
            scores = (
                q[:, node, None]
                * (key[:, node_src] + edge_key[None])
            ).sum(dim=-1) / math.sqrt(attn.head_dim)
            weights = torch.softmax(scores + edge_bias[None], dim=1)
            result[:, node] = (
                weights[..., None] * (value[:, node_src] + edge_value[None])
            ).sum(dim=1)
        return attn.out_proj(result.reshape(bsz, num_nodes, dim))

    def test_mask_matches_true_ragged_neighborhood(self):
        lvl = _bundle(refinement=2, levels=3)["levels"]["L0"]
        torch.manual_seed(4)
        attn = LocalGraphAttention(dim=16, edge_dim=6, heads=4).eval()
        h = torch.randn(1, lvl["num_nodes"], 16)
        graph = GraphLevel(lvl)
        with torch.no_grad():
            padded = attn(h, graph)
            ragged = self._ragged_forward(attn, h, graph)
        self.assertTrue(torch.allclose(padded, ragged, atol=1e-6, rtol=1e-6))

    def test_masked_slot_contributes_nothing(self):
        b = _bundle(refinement=2, levels=3)
        lvl = b["levels"]["L0"]
        torch.manual_seed(0)
        attn = LocalGraphAttention(dim=16, edge_dim=6, heads=4).eval()
        h = torch.randn(2, lvl["num_nodes"], 16)

        g1 = GraphLevel(lvl)
        with torch.no_grad():
            out1 = attn(h, g1)

        # poison the masked (dummy) slots: garbage edge features + reroute their source
        poisoned = dict(lvl)
        ea = lvl["edge_attr"].clone()
        ei = lvl["edge_index"].clone()
        invalid = (~lvl["edge_mask"]).reshape(-1)
        ea[invalid] = 1.0e6
        ei[0, invalid] = 0  # point the dummy edges at node 0 (a real, non-self node for most)
        poisoned["edge_attr"] = ea
        poisoned["edge_index"] = ei
        g2 = GraphLevel(poisoned)
        with torch.no_grad():
            out2 = attn(h, g2)

        self.assertTrue(torch.allclose(out1, out2, atol=1e-6),
                        f"masked slot leaked: max diff {float((out1 - out2).abs().max())}")

    def test_none_mask_is_noop(self):
        # A level with no edge_mask must be byte-identical to the legacy formula.
        b = _bundle(refinement=2, levels=3)
        lvl = dict(b["levels"]["L0"])
        lvl.pop("edge_mask")
        g = GraphLevel(lvl)
        self.assertIsNone(g.edge_mask)
        attn = LocalGraphAttention(dim=16, edge_dim=6, heads=4).eval()
        h = torch.randn(1, lvl["num_nodes"], 16)
        with torch.no_grad():
            out = attn(h, g)
            legacy = self._legacy_forward(attn, h, g)
        self.assertEqual(tuple(out.shape), (1, lvl["num_nodes"], 16))
        # LocalGraphAttention now reduces via torch.matmul instead of (a * b).sum(dim),
        # so it no longer reproduces the elementwise formula bit-for-bit in float32.
        # Equivalence is exact in float64; see
        # tests/test_attention_matmul_edge_cache.py::TestNumericalEquivalence.
        self.assertTrue(
            torch.allclose(out, legacy, atol=1e-6),
            f"masked-free level diverged from the legacy formula: "
            f"max |delta| = {float((out - legacy).abs().max())}",
        )


class BipartiteMPTest(unittest.TestCase):
    def test_variable_degree_output_shape(self):
        layer = BipartiteMP(dim=16, edge_dim=6, mlp_hidden_ratio=2)
        h_src = torch.randn(2, 5, 16)
        h_dst = torch.randn(2, 3, 16)
        edge_index = torch.tensor([[0, 1, 2, 4, 0], [0, 0, 1, 1, 2]], dtype=torch.long)
        edge_attr = torch.randn(5, 6)
        out = layer(h_src, h_dst, edge_index, edge_attr)
        self.assertEqual(tuple(out.shape), (2, 3, 16))

    def test_mean_aggregation_divides_by_destination_degree(self):
        messages = torch.tensor([[[2.0], [4.0], [8.0]]])
        destinations = torch.tensor([0, 0, 1], dtype=torch.long)

        summed = BipartiteMP(dim=1, edge_dim=0, aggregation="sum")._aggregate_messages(
            messages,
            destinations,
            n_dst=3,
        )
        averaged = BipartiteMP(dim=1, edge_dim=0, aggregation="mean")._aggregate_messages(
            messages,
            destinations,
            n_dst=3,
        )

        self.assertTrue(torch.equal(summed, torch.tensor([[[6.0], [8.0], [0.0]]])))
        self.assertTrue(torch.equal(averaged, torch.tensor([[[3.0], [8.0], [0.0]]])))

    def test_invalid_aggregation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "aggregation must be 'sum' or 'mean'"):
            BipartiteMP(dim=16, aggregation="max")

    def test_graphcast_boundary_embeds_edges_and_normalizes_updates(self):
        layer = BipartiteMP(
            dim=16,
            edge_dim=10,
            mlp_hidden_ratio=2,
            aggregation="mean",
            graphcast_mlp=True,
        )
        self.assertIsInstance(layer.edge_embed, MLPLayerNorm)
        self.assertIsInstance(layer.msg, MLPLayerNorm)
        self.assertIsInstance(layer.upd, MLPLayerNorm)
        self.assertIsInstance(layer.edge_embed.net[-1], torch.nn.LayerNorm)
        self.assertIsInstance(layer.msg.net[-1], torch.nn.LayerNorm)
        self.assertIsInstance(layer.upd.net[-1], torch.nn.LayerNorm)

        h_src = torch.randn(2, 5, 16)
        h_dst = torch.randn(2, 3, 16)
        edge_index = torch.tensor([[0, 1, 2, 4, 0], [0, 0, 1, 1, 2]], dtype=torch.long)
        edge_attr = torch.randn(5, 10)
        out = layer(h_src, h_dst, edge_index, edge_attr)
        self.assertEqual(tuple(out.shape), tuple(h_dst.shape))
        self.assertTrue(torch.isfinite(out).all())
        self.assertFalse(torch.equal(out, h_dst))


class MeshConfigAndCheckpointTest(unittest.TestCase):
    def test_absent_mesh_block_does_not_change_resolved_grid_config(self):
        resolved = normalize_model_config_dict({"hidden_dim": 160, "num_heads": 5})
        self.assertNotIn("mesh_encoder", resolved)
        self.assertNotIn("mesh_encoder", resolved["model"])

    def test_mesh_block_is_normalized_on_both_surfaces(self):
        resolved = normalize_model_config_dict(
            {
                "hidden_dim": 160,
                "num_heads": 5,
                "num_graph_levels": 4,
                "mesh_encoder": {"enabled": True, "refinement": 5},
            }
        )
        expected = {
            "enabled": True,
            "refinement": 5,
            "g2m_radius_factor": 0.6,
            "mlp_hidden_ratio": 2,
            "aggregation": "sum",
            "boundary_type": "legacy",
            "grid_skip_mlp": False,
            "grid_attention_encoder_blocks": 0,
            "grid_attention_decoder_blocks": 0,
            "grid_attention_k_neighbors": 8,
            "coarse_level_connectivity": "native_icosphere",
        }
        self.assertEqual(resolved["mesh_encoder"], expected)
        self.assertEqual(resolved["model"]["mesh_encoder"], expected)
        self.assertEqual(resolved["node_counts"], [10242, 2562, 642, 162])

    def test_full_m1_resolves_counts_and_is_a_separate_checkpoint_family(self):
        native = {
            "hidden_dim": 160,
            "num_heads": 5,
            "num_graph_levels": 4,
            "mesh_encoder": {"enabled": True, "refinement": 4},
        }
        full_m1 = {
            **native,
            "mesh_encoder": {
                "enabled": True,
                "refinement": 4,
                "coarse_level_connectivity": "full_m1",
            },
        }
        resolved = normalize_model_config_dict(full_m1)
        self.assertEqual(resolved["node_counts"], [2562, 642, 162, 42])
        self.assertEqual(resolved["level_k_neighbors"], [6, 6, 6, 41])
        self.assertEqual(resolved["model"]["level_k_neighbors"], [6, 6, 6, 41])
        self.assertEqual(resolved["edge_counts"], [15372, 3852, 972, 1722])
        with self.assertRaisesRegex(RuntimeError, "mesh_encoder"):
            validate_checkpoint_architecture(
                architecture_metadata(native),
                full_m1,
            )

    def test_full_m1_config_rejects_a_non_m1_coarsest_level(self):
        with self.assertRaisesRegex(ValueError, "coarsest level to be M1"):
            normalize_model_config_dict(
                {
                    "hidden_dim": 160,
                    "num_heads": 5,
                    "num_graph_levels": 3,
                    "mesh_encoder": {
                        "enabled": True,
                        "refinement": 4,
                        "coarse_level_connectivity": "full_m1",
                    },
                }
            )

    def test_grid_and_mesh_checkpoint_families_are_rejected(self):
        grid = {"hidden_dim": 160, "num_heads": 5, "num_graph_levels": 4}
        mesh = {
            **grid,
            "mesh_encoder": {
                "enabled": True,
                "refinement": 5,
                "g2m_radius_factor": 0.6,
                "mlp_hidden_ratio": 2,
            },
        }
        grid_metadata = architecture_metadata(grid)
        mesh_metadata = architecture_metadata(mesh)
        with self.assertRaisesRegex(RuntimeError, "separate checkpoint families"):
            validate_checkpoint_architecture(grid_metadata, mesh)
        with self.assertRaisesRegex(RuntimeError, "separate checkpoint families"):
            validate_checkpoint_architecture(mesh_metadata, grid)

    def test_sum_and_mean_checkpoint_families_are_rejected(self):
        sum_config = {
            "hidden_dim": 160,
            "num_heads": 5,
            "num_graph_levels": 4,
            "mesh_encoder": {"enabled": True, "refinement": 4, "aggregation": "sum"},
        }
        mean_config = {
            **sum_config,
            "mesh_encoder": {"enabled": True, "refinement": 4, "aggregation": "mean"},
        }
        with self.assertRaisesRegex(RuntimeError, "mesh_encoder"):
            validate_checkpoint_architecture(architecture_metadata(sum_config), mean_config)

    def test_legacy_and_graphcast_boundary_checkpoint_families_are_rejected(self):
        legacy_config = {
            "hidden_dim": 160,
            "num_heads": 5,
            "num_graph_levels": 4,
            "mesh_encoder": {
                "enabled": True,
                "refinement": 4,
                "boundary_type": "legacy",
            },
        }
        graphcast_config = {
            **legacy_config,
            "mesh_encoder": {
                "enabled": True,
                "refinement": 4,
                "boundary_type": "graphcast_mlp",
            },
        }
        with self.assertRaisesRegex(RuntimeError, "mesh_encoder"):
            validate_checkpoint_architecture(
                architecture_metadata(legacy_config),
                graphcast_config,
            )

    def test_grid_skip_mlp_requires_graphcast_boundary(self):
        with self.assertRaisesRegex(ValueError, "boundary_type='graphcast_mlp'"):
            normalize_model_config_dict(
                {
                    "hidden_dim": 160,
                    "num_heads": 5,
                    "num_graph_levels": 4,
                    "mesh_encoder": {
                        "enabled": True,
                        "refinement": 4,
                        "boundary_type": "legacy",
                        "grid_skip_mlp": True,
                    },
                }
            )

    def test_grid_skip_mlp_is_a_separate_checkpoint_family(self):
        control = {
            "hidden_dim": 160,
            "num_heads": 5,
            "num_graph_levels": 4,
            "mesh_encoder": {
                "enabled": True,
                "refinement": 4,
                "boundary_type": "graphcast_mlp",
            },
        }
        grid_skip = copy.deepcopy(control)
        grid_skip["mesh_encoder"]["grid_skip_mlp"] = True
        with self.assertRaisesRegex(RuntimeError, "mesh_encoder"):
            validate_checkpoint_architecture(
                architecture_metadata(control),
                grid_skip,
            )

    def test_grid_attention_stem_is_a_separate_checkpoint_family(self):
        control = {
            "hidden_dim": 160,
            "num_heads": 5,
            "num_graph_levels": 3,
            "mesh_encoder": {
                "enabled": True,
                "refinement": 4,
                "boundary_type": "graphcast_mlp",
            },
        }
        grid_attention = copy.deepcopy(control)
        grid_attention["mesh_encoder"]["grid_attention_encoder_blocks"] = 2
        grid_attention["mesh_encoder"]["grid_attention_decoder_blocks"] = 2
        with self.assertRaisesRegex(RuntimeError, "mesh_encoder"):
            validate_checkpoint_architecture(
                architecture_metadata(control),
                grid_attention,
            )

    def test_fixed_spherical_s1x200_config_matches_dense_training_contract(self):
        dense_path = PROJECT_ROOT / "configs" / "experiments" / (
            "config_2p5_l3_hidden160_base_S1_200epoch_dense_l3k24.yaml"
        )
        fixed_path = PROJECT_ROOT / "configs" / "experiments" / (
            "config_2p5_l3_hidden160_base_S1_200epoch_icomeshm5_fixedspherical.yaml"
        )
        dense = YParams(
            str(dense_path),
            "s1only_2p5_l3_hidden160_base_200epoch_dense_l3k24",
            resolution_mode="2p5",
        )
        fixed = YParams(
            str(fixed_path),
            "2p5_l3_h160_icomeshm5_fixedspherical_s1x200",
            resolution_mode="2p5",
        )

        for key in (
            "lr",
            "min_lr",
            "weight_decay",
            "max_epochs",
            "batch_size",
            "gradient_accumulation_steps",
            "encoder_blocks",
            "decoder_blocks",
            "l0_blocks",
            "l1_blocks",
            "l2_blocks",
            "l3_blocks",
            "l2_refine_after_l3_blocks",
            "l1_refine_blocks",
            "l0_refine_blocks",
            "rollout_schedule",
            "rollout_stage_epochs",
            "valid_rollout_steps",
            "checkpoint_metric",
            "lr_schedule_type",
            "warmup_epochs",
            "target_handling",
        ):
            self.assertEqual(getattr(fixed, key), getattr(dense, key), key)
        self.assertEqual(fixed.node_counts, [10242, 2562, 642, 162])
        self.assertEqual(fixed.level_k_neighbors, [6, 6, 6, 6])
        self.assertEqual(fixed.mesh_encoder["boundary_type"], "fixed_spherical")
        self.assertIs(fixed.mesh_encoder["grid_skip_mlp"], False)
        self.assertEqual(fixed.mesh_encoder["grid_attention_encoder_blocks"], 0)
        self.assertEqual(fixed.mesh_encoder["grid_attention_decoder_blocks"], 0)
        self.assertEqual(fixed.model["mesh_encoder"], fixed.mesh_encoder)
        self.assertEqual(
            fixed.graph_path,
            "graphs/graph_2p5_icosphere_r5_l3_fixedspherical.pt",
        )

    def test_mean_tisrfix_experiment_config_resolves(self):
        config_path = PROJECT_ROOT / "configs" / "experiments" / (
            "2p5_l3_h160_icomeshm4_bipmponly_l0x2_l1x1_"
            "s1x120_currs2tos9x2_s10x5.yaml"
        )
        name = (
            "2p5_l3_h160_icomeshm4_bipmponly_meanagg_tisrfix_"
            "l0x2_l1x1_s1x120_currs2tos9x2_s10x5"
        )
        params = YParams(str(config_path), name, resolution_mode="2p5")
        self.assertEqual(params.mesh_encoder["aggregation"], "mean")
        self.assertEqual(params.model["mesh_encoder"]["aggregation"], "mean")
        self.assertEqual(params.target_handling["known_future_variables"], ["tisr"])
        self.assertEqual(params.target_handling["exclude_loss_variables"], ["orog", "tisr"])
        self.assertEqual(params.experiment_name, name)
        self.assertEqual(params.wandb["run_name"], name)

    def test_mean_tisrfix_fast_experiment_config_resolves(self):
        config_path = PROJECT_ROOT / "configs" / "experiments" / (
            "2p5_l3_h160_icomeshm4_bipmponly_l0x2_l1x1_"
            "s1x120_currs2tos9x2_s10x5.yaml"
        )
        name = (
            "2p5_l3_h160_icomeshm4_bipmponly_meanagg_tisrfix_fast_"
            "l0x2_l1x1_s1x120_currs2tos9x2_s10x5"
        )
        params = YParams(str(config_path), name, resolution_mode="2p5")

        self.assertEqual(params.mesh_encoder["aggregation"], "mean")
        self.assertEqual(params.target_handling["known_future_variables"], ["tisr"])
        self.assertEqual(params.batch_size, 6)
        self.assertEqual(params.gradient_accumulation_steps, 2)
        self.assertEqual(params.batch_size * params.gradient_accumulation_steps, 12)
        self.assertIs(params.load_only_current_rollout, True)
        self.assertIs(params.training["load_only_current_rollout"], True)
        self.assertIs(params.training["activation_checkpointing"], False)
        self.assertIs(params.training["checkpoint_rollout_steps"], False)
        self.assertEqual(params.rollout_stage_epochs, [120, 2, 2, 2, 2, 2, 2, 2, 2, 5])
        self.assertEqual(params.max_epochs, 141)
        self.assertEqual(params.experiment_name, name)
        self.assertEqual(params.wandb["run_name"], name)

    def test_graphcast_boundary_fast_experiment_config_resolves(self):
        config_path = PROJECT_ROOT / "configs" / "experiments" / (
            "2p5_l3_h160_icomeshm4_bipmponly_l0x2_l1x1_"
            "s1x120_currs2tos9x2_s10x5.yaml"
        )
        name = (
            "2p5_l3_h160_icomeshm4_bipmponly_meanagg_tisrfix_fast_gcboundary_"
            "l0x2_l1x1_s1x120_currs2tos9x2_s10x5"
        )
        params = YParams(str(config_path), name, resolution_mode="2p5")

        self.assertEqual(params.mesh_encoder["aggregation"], "mean")
        self.assertEqual(params.mesh_encoder["boundary_type"], "graphcast_mlp")
        self.assertEqual(params.model["mesh_encoder"], params.mesh_encoder)
        self.assertEqual(
            params.graph_path,
            "graphs/graph_2p5_icosphere_r4_l3_gclocal10.pt",
        )
        self.assertEqual(params.target_handling["known_future_variables"], ["tisr"])
        self.assertEqual(params.batch_size, 6)
        self.assertEqual(params.gradient_accumulation_steps, 2)
        self.assertIs(params.training["activation_checkpointing"], False)
        self.assertEqual(params.experiment_name, name)
        self.assertEqual(params.wandb["run_name"], name)

    def test_graphcast_boundary_grid_skip_mlp_experiment_config_resolves(self):
        config_path = PROJECT_ROOT / "configs" / "experiments" / (
            "2p5_l3_h160_icomeshm4_bipmponly_l0x2_l1x1_"
            "s1x120_currs2tos9x2_s10x5.yaml"
        )
        name = (
            "2p5_l3_h160_icomeshm4_bipmponly_meanagg_tisrfix_fast_"
            "gcboundary_gridskipmlp_l0x2_l1x1_s1x120_currs2tos9x2_s10x5"
        )
        params = YParams(str(config_path), name, resolution_mode="2p5")

        self.assertEqual(params.mesh_encoder["aggregation"], "mean")
        self.assertEqual(params.mesh_encoder["boundary_type"], "graphcast_mlp")
        self.assertIs(params.mesh_encoder["grid_skip_mlp"], True)
        self.assertEqual(
            params.mesh_encoder["coarse_level_connectivity"],
            "native_icosphere",
        )
        self.assertEqual(params.model["mesh_encoder"], params.mesh_encoder)
        self.assertEqual(
            params.graph_path,
            "graphs/graph_2p5_icosphere_r4_l3_gclocal10.pt",
        )
        self.assertEqual(params.experiment_name, name)
        self.assertEqual(params.wandb["run_name"], name)

    def test_dense_l3_matched_grid_attention_experiment_config_resolves(self):
        config_path = PROJECT_ROOT / "configs" / "experiments" / (
            "2p5_l3_h160_icomeshm4_bipmponly_l0x2_l1x1_"
            "s1x120_currs2tos9x2_s10x5.yaml"
        )
        name = (
            "2p5_l3_h160_icomeshm4_gcboundary_gridenc2dec2_densel3match_"
            "meanagg_tisrfix_s1x120_currs2tos9x2_s10x5"
        )
        params = YParams(str(config_path), name, resolution_mode="2p5")

        self.assertEqual(params.hidden_dim, 160)
        self.assertEqual(params.num_graph_levels, 3)
        self.assertIs(params.use_l3, False)
        self.assertEqual(params.node_counts, [2562, 642, 162])
        self.assertEqual(params.level_k_neighbors, [6, 6, 6])
        self.assertEqual(
            params.graph_path,
            "graphs/graph_2p5_icosphere_r4_l2_gclocal10_gridk8.pt",
        )
        self.assertEqual(params.mesh_encoder["grid_attention_encoder_blocks"], 2)
        self.assertEqual(params.mesh_encoder["grid_attention_decoder_blocks"], 2)
        self.assertEqual(params.mesh_encoder["grid_attention_k_neighbors"], 8)
        self.assertIs(params.mesh_encoder["grid_skip_mlp"], False)
        self.assertEqual(params.model["mesh_encoder"], params.mesh_encoder)
        self.assertEqual(params.batch_size, 4)
        self.assertEqual(params.gradient_accumulation_steps, 3)
        self.assertEqual(params.batch_size * params.gradient_accumulation_steps, 12)
        self.assertEqual(params.experiment_name, name)
        self.assertEqual(params.wandb["run_name"], name)

    def test_graphcast_boundary_full_m1_experiment_config_resolves(self):
        config_path = PROJECT_ROOT / "configs" / "experiments" / (
            "2p5_l3_h160_icomeshm4_bipmponly_l0x2_l1x1_"
            "s1x120_currs2tos9x2_s10x5.yaml"
        )
        name = (
            "2p5_l3_h160_icomeshm4_bipmponly_meanagg_tisrfix_fast_"
            "gcboundary_fullm1_l0x2_l1x1_s1x120_currs2tos9x2_s10x5"
        )
        params = YParams(str(config_path), name, resolution_mode="2p5")

        self.assertEqual(params.mesh_encoder["aggregation"], "mean")
        self.assertEqual(params.mesh_encoder["boundary_type"], "graphcast_mlp")
        self.assertEqual(params.mesh_encoder["coarse_level_connectivity"], "full_m1")
        self.assertEqual(params.model["mesh_encoder"], params.mesh_encoder)
        self.assertEqual(params.level_k_neighbors, [6, 6, 6, 41])
        self.assertEqual(params.edge_counts, [15372, 3852, 972, 1722])
        self.assertEqual(
            params.graph_path,
            "graphs/graph_2p5_icosphere_r4_l3_gclocal10_fullm1.pt",
        )
        self.assertEqual(params.target_handling["known_future_variables"], ["tisr"])
        self.assertEqual(params.batch_size, 6)
        self.assertEqual(params.gradient_accumulation_steps, 2)
        self.assertIs(params.training["activation_checkpointing"], False)
        self.assertEqual(params.experiment_name, name)
        self.assertEqual(params.wandb["run_name"], name)

    def test_every_mean_experiment_has_a_matched_sum_counterpart(self):
        config_path = PROJECT_ROOT / "configs" / "experiments" / (
            "2p5_l3_h160_icomeshm4_bipmponly_l0x2_l1x1_"
            "s1x120_currs2tos9x2_s10x5.yaml"
        )
        suffixes = (
            "tisrfix_l0x2_l1x1_s1x120_currs2tos9x2_s10x5",
            "tisrfix_fast_l0x2_l1x1_s1x120_currs2tos9x2_s10x5",
            "tisrfix_fast_gcboundary_l0x2_l1x1_s1x120_currs2tos9x2_s10x5",
            "tisrfix_fast_gcboundary_fullm1_l0x2_l1x1_s1x120_currs2tos9x2_s10x5",
        )

        def comparison_payload(params):
            payload = copy.deepcopy(params.params)
            payload.pop("experiment_name", None)
            payload.pop("run_name", None)
            payload["wandb"].pop("run_name", None)
            payload["wandb"].pop("tags", None)
            payload["mesh_encoder"]["aggregation"] = "<aggregation>"
            payload["model"]["mesh_encoder"]["aggregation"] = "<aggregation>"
            return payload

        for suffix in suffixes:
            with self.subTest(suffix=suffix):
                mean_name = f"2p5_l3_h160_icomeshm4_bipmponly_meanagg_{suffix}"
                sum_name = f"2p5_l3_h160_icomeshm4_bipmponly_sumagg_{suffix}"
                mean_params = YParams(
                    str(config_path),
                    mean_name,
                    resolution_mode="2p5",
                )
                sum_params = YParams(
                    str(config_path),
                    sum_name,
                    resolution_mode="2p5",
                )

                self.assertEqual(mean_params.mesh_encoder["aggregation"], "mean")
                self.assertEqual(sum_params.mesh_encoder["aggregation"], "sum")
                self.assertEqual(sum_params.model["mesh_encoder"], sum_params.mesh_encoder)
                self.assertEqual(
                    sum_params.target_handling["known_future_variables"],
                    ["tisr"],
                )
                self.assertEqual(
                    sum_params.target_handling["exclude_loss_variables"],
                    ["orog", "tisr"],
                )
                self.assertEqual(sum_params.experiment_name, sum_name)
                self.assertEqual(sum_params.wandb["run_name"], sum_name)
                self.assertEqual(
                    comparison_payload(mean_params),
                    comparison_payload(sum_params),
                )


class MeshEvaluatorGraphValidationTest(unittest.TestCase):
    @staticmethod
    def _evaluator(bundle, *, mesh_enabled=True):
        metadata = bundle["metadata"]
        evaluator = GraphWeatherEvaluator.__new__(GraphWeatherEvaluator)
        evaluator.logger = logging
        evaluator.cfg = SimpleNamespace(
            resolution_mode=str(metadata["resolution_mode"]),
            grid_shape=tuple(metadata["grid_shape"]),
            num_graph_levels=int(metadata["num_graph_levels"]),
            graph_format_version=3,
            mesh_format_version=1,
            hierarchy_type="standard",
            use_l4_ratio15=False,
            graph_connectivity_strategy="hybrid_row_aware_knn",
            k_neighbors=6,
            level_k_neighbors=list(
                metadata.get(
                    "level_k_neighbors",
                    [6] * int(metadata["num_graph_levels"]),
                )
            ),
            level_shapes=[[24, 48], [12, 24], [6, 12]],
            node_counts=list(metadata["node_counts"]),
            edge_counts=list(metadata["edge_counts"]),
            mesh_encoder={
                "enabled": mesh_enabled,
                "refinement": int(metadata["refinement"]),
                "g2m_radius_factor": float(metadata["g2m_radius_factor"]),
                "mlp_hidden_ratio": 2,
                "coarse_level_connectivity": metadata.get(
                    "coarse_level_connectivity",
                    mesh_builder.NATIVE_COARSE_CONNECTIVITY,
                ),
            },
        )
        return evaluator

    def test_mesh_uses_mesh_format_and_ignores_grid_only_metadata(self):
        bundle = _bundle(refinement=2, levels=3, grid=(24, 48))
        evaluator = self._evaluator(bundle)
        # Reproduces the evaluation config that failed: graph_format_version=3,
        # hierarchy_type=standard, and grid connectivity are irrelevant in mesh mode.
        evaluator._validate_graph_resolution(bundle["metadata"])

    def test_graphcast_edge_cache_contract_is_validated(self):
        bundle = _bundle(
            refinement=2,
            levels=3,
            grid=(24, 48),
            bipartite_edge_features=mesh_builder.GRAPHCAST_BIPARTITE_EDGE_FEATURES,
        )
        evaluator = self._evaluator(bundle)
        evaluator.cfg.mesh_encoder["boundary_type"] = "graphcast_mlp"
        evaluator._validate_graph_resolution(bundle["metadata"])

        evaluator.cfg.mesh_encoder["boundary_type"] = "legacy"
        with self.assertRaisesRegex(ValueError, "bipartite_edge_features"):
            evaluator._validate_graph_resolution(bundle["metadata"])

    def test_coarse_connectivity_cache_contract_is_validated(self):
        bundle = _bundle(
            refinement=3,
            levels=3,
            grid=(12, 24),
            coarse_level_connectivity=mesh_builder.FULL_M1_COARSE_CONNECTIVITY,
        )
        evaluator = self._evaluator(bundle)
        evaluator._validate_graph_resolution(bundle["metadata"])

        evaluator.cfg.mesh_encoder["coarse_level_connectivity"] = (
            mesh_builder.NATIVE_COARSE_CONNECTIVITY
        )
        with self.assertRaisesRegex(ValueError, "coarse_level_connectivity"):
            evaluator._validate_graph_resolution(bundle["metadata"])

    def test_mesh_format_mismatch_is_rejected(self):
        bundle = _bundle(refinement=2, levels=3, grid=(24, 48))
        evaluator = self._evaluator(bundle)
        evaluator.cfg.mesh_format_version = 2
        with self.assertRaisesRegex(ValueError, "mesh_format_version=1"):
            evaluator._validate_graph_resolution(bundle["metadata"])

    def test_grid_config_rejects_mesh_bundle(self):
        bundle = _bundle(refinement=2, levels=3, grid=(24, 48))
        evaluator = self._evaluator(bundle, mesh_enabled=False)
        with self.assertRaisesRegex(ValueError, "Graph cache mode mismatch"):
            evaluator._validate_graph_resolution(bundle["metadata"])

    def test_mesh_rollout_uses_data_grid_not_mesh_level_shape(self):
        bundle = _bundle(refinement=2, levels=3, grid=(24, 48))
        evaluator = self._evaluator(bundle)
        evaluator.model = SimpleNamespace(graph=GraphBundle(bundle))
        self.assertEqual(evaluator._model_grid_shape(), (24, 48))

    def test_grid_rollout_shape_path_is_unchanged(self):
        evaluator = GraphWeatherEvaluator.__new__(GraphWeatherEvaluator)
        evaluator.model = SimpleNamespace(
            graph=SimpleNamespace(
                graph_mode="grid",
                L0=SimpleNamespace(height=72, width=144),
            )
        )
        self.assertEqual(evaluator._model_grid_shape(), (72, 144))

    def test_mesh_checkpoint_and_cache_hierarchy_names_are_canonicalized(self):
        bundle = _bundle(refinement=2, levels=3, grid=(24, 48))
        cache_metadata = dict(bundle["metadata"])
        checkpoint_metadata = dict(cache_metadata)
        checkpoint_metadata["hierarchy_type"] = "standard"
        self.assertEqual(
            graph_topology_metadata(checkpoint_metadata),
            graph_topology_metadata(cache_metadata),
        )


class MeshForwardTest(unittest.TestCase):
    def test_fixed_spherical_boundary_has_no_boundary_parameters_and_backpropagates(self):
        H, W = 12, 24
        bundle = _bundle(
            refinement=2,
            levels=3,
            grid=(H, W),
            bipartite_mapping_type=mesh_builder.FIXED_SPHERICAL_MAPPING,
        )
        model = GraphWeatherModel(
            graph=GraphBundle(bundle),
            grid_shape=(H, W),
            input_channels=134,
            output_channels=67,
            n_history=1,
            hidden_dim=16,
            heads=4,
            num_graph_levels=3,
            use_l3=False,
            mesh_encoder={
                "enabled": True,
                "boundary_type": "fixed_spherical",
            },
        ).eval()

        self.assertIsInstance(model.embed, torch.nn.Linear)
        self.assertIsInstance(model.grid2mesh, FixedBipartiteRemap)
        self.assertIsInstance(model.mesh2grid, FixedBipartiteRemap)
        self.assertIsNone(model.mesh_node_init)
        self.assertIsNone(model.grid_skip_mlp)
        boundary_parameter_names = [
            name
            for name, _ in model.named_parameters()
            if any(
                component in name
                for component in (
                    "grid2mesh",
                    "mesh2grid",
                    "mesh_node_init",
                    "grid_skip_mlp",
                )
            )
        ]
        self.assertEqual(boundary_parameter_names, [])

        inp = torch.randn(1, 134, H, W, requires_grad=True)
        out = model(inp)
        self.assertEqual(tuple(out.shape), (1, 67, H, W))
        out.square().mean().backward()
        self.assertIsNotNone(inp.grad)
        self.assertTrue(torch.isfinite(inp.grad).all())
    def test_mesh_mode_forward_shape(self):
        H, W = 12, 24
        b = _bundle(refinement=2, levels=3, grid=(H, W))
        graph = GraphBundle(b)
        self.assertEqual(graph.graph_mode, "mesh")
        self.assertEqual((graph.grid_height, graph.grid_width), (H, W))

        model = GraphWeatherModel(
            graph=graph,
            grid_shape=(H, W),
            input_channels=134,
            output_channels=67,
            n_history=1,
            hidden_dim=16,
            heads=4,
            num_graph_levels=3,
            use_l3=False,
            mesh_encoder={"enabled": True, "mlp_hidden_ratio": 2, "aggregation": "mean"},
        ).eval()

        self.assertEqual(model.grid2mesh.aggregation, "mean")
        self.assertEqual(model.mesh2grid.aggregation, "mean")

        inp = torch.randn(1, 134, H, W)
        with torch.no_grad():
            out = model(inp)
        self.assertEqual(tuple(out.shape), (1, 67, H, W))
        self.assertTrue(torch.isfinite(out).all())

    def test_grid_shape_mismatch_raises(self):
        b = _bundle(refinement=2, levels=3, grid=(24, 48))
        graph = GraphBundle(b)
        with self.assertRaises(ValueError):
            GraphWeatherModel(
                graph=graph, grid_shape=(12, 24), input_channels=8, output_channels=2,
                n_history=1, hidden_dim=16, heads=4, num_graph_levels=3,
                mesh_encoder={"enabled": True},
            )

    def test_graphcast_mlp_boundary_forward_shape_and_modules(self):
        H, W = 12, 24
        bundle = _bundle(
            refinement=2,
            levels=3,
            grid=(H, W),
            bipartite_edge_features=mesh_builder.GRAPHCAST_BIPARTITE_EDGE_FEATURES,
        )
        graph = GraphBundle(bundle)
        model = GraphWeatherModel(
            graph=graph,
            grid_shape=(H, W),
            input_channels=134,
            output_channels=67,
            n_history=1,
            hidden_dim=16,
            heads=4,
            num_graph_levels=3,
            use_l3=False,
            mesh_encoder={
                "enabled": True,
                "mlp_hidden_ratio": 2,
                "aggregation": "mean",
                "boundary_type": "graphcast_mlp",
            },
        ).eval()

        self.assertIsInstance(model.embed, MLPLayerNorm)
        self.assertEqual(model.embed.in_dim, model.total_node_feature_channels + 4)
        self.assertIsInstance(model.mesh_node_init, MLPLayerNorm)
        self.assertIsNone(model.grid_skip_mlp)
        self.assertTrue(model.grid2mesh.graphcast_mlp)
        self.assertTrue(model.mesh2grid.graphcast_mlp)
        self.assertEqual(model.grid2mesh.edge_dim, 10)
        self.assertEqual(model.mesh2grid.edge_dim, 10)

        inp = torch.randn(1, 134, H, W)
        with torch.no_grad():
            out = model(inp)
        self.assertEqual(tuple(out.shape), (1, 67, H, W))
        self.assertTrue(torch.isfinite(out).all())

    def test_graphcast_grid_skip_mlp_is_carried_into_mesh2grid(self):
        H, W = 12, 24
        bundle = _bundle(
            refinement=2,
            levels=3,
            grid=(H, W),
            bipartite_edge_features=mesh_builder.GRAPHCAST_BIPARTITE_EDGE_FEATURES,
        )
        graph = GraphBundle(bundle)
        model = GraphWeatherModel(
            graph=graph,
            grid_shape=(H, W),
            input_channels=134,
            output_channels=67,
            n_history=1,
            hidden_dim=16,
            heads=4,
            num_graph_levels=3,
            use_l3=False,
            mesh_encoder={
                "enabled": True,
                "mlp_hidden_ratio": 2,
                "aggregation": "mean",
                "boundary_type": "graphcast_mlp",
                "grid_skip_mlp": True,
            },
        ).eval()

        self.assertIsInstance(model.grid_skip_mlp, MLPLayerNorm)
        diagnostics = {}

        class _Collector:
            def add_embedding(self, name, tensor):
                diagnostics[name] = tensor.detach().clone()

        mesh2grid_inputs = {}

        def _capture_mesh2grid_inputs(_module, args):
            mesh2grid_inputs["h_dst"] = args[1].detach().clone()

        handle = model.mesh2grid.register_forward_pre_hook(_capture_mesh2grid_inputs)
        try:
            inp = torch.randn(1, 134, H, W)
            with torch.no_grad():
                out = model(inp, diagnostics_collector=_Collector())
        finally:
            handle.remove()

        expected_skip = diagnostics["model/embed"] + model.grid_skip_mlp(
            diagnostics["model/embed"]
        )
        self.assertTrue(
            torch.allclose(
                diagnostics["model/grid_skip_encoded"],
                expected_skip,
                atol=1.0e-6,
                rtol=1.0e-6,
            )
        )
        self.assertTrue(
            torch.equal(
                mesh2grid_inputs["h_dst"],
                diagnostics["model/grid_skip_encoded"],
            )
        )
        self.assertEqual(tuple(out.shape), (1, 67, H, W))
        self.assertTrue(torch.isfinite(out).all())

    def test_grid_attention_encoder_output_is_carried_and_decoder_refines_it(self):
        H, W = 12, 24
        bundle = _bundle(
            refinement=2,
            levels=3,
            grid=(H, W),
            bipartite_edge_features=mesh_builder.GRAPHCAST_BIPARTITE_EDGE_FEATURES,
            grid_attention_k_neighbors=8,
        )
        graph = GraphBundle(bundle)
        model = GraphWeatherModel(
            graph=graph,
            grid_shape=(H, W),
            input_channels=134,
            output_channels=67,
            n_history=1,
            hidden_dim=16,
            heads=4,
            num_graph_levels=3,
            use_l3=False,
            mesh_encoder={
                "enabled": True,
                "mlp_hidden_ratio": 2,
                "aggregation": "mean",
                "boundary_type": "graphcast_mlp",
                "grid_attention_encoder_blocks": 2,
                "grid_attention_decoder_blocks": 2,
                "grid_attention_k_neighbors": 8,
            },
        ).eval()

        self.assertEqual(len(model.grid_encoder), 2)
        self.assertEqual(len(model.grid_decoder), 2)
        self.assertIsNone(model.grid_skip_mlp)
        diagnostics = {}

        class _Collector:
            def add_embedding(self, name, tensor):
                diagnostics[name] = tensor.detach().clone()

        boundary_inputs = {}

        def _capture_grid2mesh(_module, args):
            boundary_inputs["g2m_src"] = args[0].detach().clone()

        def _capture_mesh2grid(_module, args):
            boundary_inputs["m2g_dst"] = args[1].detach().clone()

        g2m_handle = model.grid2mesh.register_forward_pre_hook(_capture_grid2mesh)
        m2g_handle = model.mesh2grid.register_forward_pre_hook(_capture_mesh2grid)
        try:
            inp = torch.randn(1, 134, H, W)
            with torch.no_grad():
                out = model(inp, diagnostics_collector=_Collector())
        finally:
            g2m_handle.remove()
            m2g_handle.remove()

        processed_grid = diagnostics["model/grid_encoder.1"]
        self.assertTrue(torch.equal(boundary_inputs["g2m_src"], processed_grid))
        self.assertTrue(torch.equal(boundary_inputs["m2g_dst"], processed_grid))
        self.assertIn("model/grid_decoder.0", diagnostics)
        self.assertIn("model/grid_decoder.1", diagnostics)
        self.assertFalse(
            torch.equal(
                diagnostics["model/mesh2grid"],
                diagnostics["model/grid_decoder.1"],
            )
        )
        self.assertEqual(tuple(out.shape), (1, 67, H, W))
        self.assertTrue(torch.isfinite(out).all())

    def test_graphcast_mlp_boundary_rejects_legacy_edge_cache(self):
        H, W = 12, 24
        graph = GraphBundle(_bundle(refinement=2, levels=3, grid=(H, W)))
        with self.assertRaisesRegex(ValueError, "requires 10-D"):
            GraphWeatherModel(
                graph=graph,
                grid_shape=(H, W),
                input_channels=8,
                output_channels=2,
                n_history=1,
                hidden_dim=16,
                heads=4,
                num_graph_levels=3,
                mesh_encoder={
                    "enabled": True,
                    "boundary_type": "graphcast_mlp",
                },
            )


if __name__ == "__main__":
    unittest.main()
