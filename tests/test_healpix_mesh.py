"""HEALPix-as-mesh tests: lat-lon data grid, HEALPix mesh, conservative regrid.

This is the in-model regridding design: the data never leaves the lat-lon grid,
the model regrids to HEALPix between grid2mesh and mesh2grid, so delta statistics,
the latitude-weighted loss and the evaluator all stay lat-lon. Contrast with
``resolution_mode: hpx32`` (tests/test_healpix_native.py), where the data on disk
IS HEALPix.

Verifies: pixel counts and NEST quadtree pooling, true degree-8 adjacency with the
24 degree-7 pixels masked, conservative operators that are row-normalized and
reproduce the offline regrid exactly, a full lat-lon-in/lat-lon-out forward pass,
cache identity in both directions against icosphere bundles, and the config guards.

CPU-only, no data. Needs healpy (skipped otherwise). unittest-style (repo convention).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import mesh_builder  # noqa: E402
from src.architecture import resolve_mesh_encoder  # noqa: E402
from src.graph_bundle import GraphBundle  # noqa: E402
from src.mesh_layers import FixedBipartiteRemap  # noqa: E402
from src.models import GraphWeatherModel  # noqa: E402

try:  # healpy is an optional extra; every lat-lon path works without it.
    import healpy  # noqa: F401

    HAVE_HEALPY = True
except ImportError:  # pragma: no cover - dependency guard
    HAVE_HEALPY = False

# Small on purpose: nside 4 keeps 4 levels (4/2/1 would run out, so 3 levels) and
# the whole bundle builds in well under a second. The geometry facts under test
# (quadtree pooling, 24 degree-7 pixels, row-normalized operators) are exact at
# every nside, so a small one is not a weaker test.
NSIDE = 4
LEVELS = 3
GRID = (18, 36)

_BUNDLE_CACHE: dict = {}


def _bundle(nside=NSIDE, levels=LEVELS, grid=GRID, oversample=8, level_k=None):
    key = (nside, levels, grid, oversample, tuple(level_k) if level_k else None)
    if key not in _BUNDLE_CACHE:
        _BUNDLE_CACHE[key] = mesh_builder.build_healpix_mesh_bundle(
            nside=nside,
            num_graph_levels=levels,
            grid_shape=grid,
            grid_lat_lon=mesh_builder._grid_latlon(*grid),
            resolution_mode="test",
            regrid_oversample=oversample,
            regrid_cache_dir=str(PROJECT_ROOT / "tests" / "_healpix_regrid_cache"),
            level_k_neighbors=level_k,
        )
    return _BUNDLE_CACHE[key]


def _bundle_dense():
    """Coarsest level densified past the mesh degree, the dense-L3K24 shape.

    nside 8/4/2 -> 768/192/48 pixels, so k=24 fits at the coarsest level (the real
    config coarsens to 192 pixels with k=24). NSIDE=4 would bottom out at nside 1
    = 12 pixels, where k=24 cannot exist at all.
    """
    return _bundle(nside=8, level_k=[8, 8, 24])


@unittest.skipUnless(HAVE_HEALPY, "healpy is an optional extra")
class HealpixMeshGeometryTest(unittest.TestCase):
    def test_pixel_counts_are_a_quadtree(self):
        bundle = _bundle()
        counts = bundle["metadata"]["node_counts"]
        self.assertEqual(counts, [12 * (NSIDE >> i) ** 2 for i in range(LEVELS)])
        # Each coarsening divides the pixel count by exactly 4, not 2 per axis.
        for fine, coarse in zip(counts, counts[1:]):
            self.assertEqual(fine, 4 * coarse)

    def test_pool_map_is_nest_parent(self):
        bundle = _bundle()
        for i in range(LEVELS - 1):
            pool = bundle["pool"][f"L{i}_to_L{i + 1}"]
            expected = torch.arange(int(pool.numel()), dtype=torch.long) >> 2
            torch.testing.assert_close(pool, expected)
            # Exactly four children per parent, no exceptions.
            counts = torch.bincount(pool)
            self.assertTrue(bool((counts == 4).all()))

    def test_degree_distribution_per_level(self):
        """8 slots everywhere; the real degree depends on nside.

        nside >= 2: exactly 24 pixels have 7 neighbours (three-face corners), the
        rest have 8. nside == 1: the 12 base pixels all have degree 6. Asserting a
        flat "24 degree-7 pixels" would be wrong at the coarsest level, which is
        exactly the case a small test grid reaches.
        """
        bundle = _bundle()
        nsides = bundle["metadata"]["level_nsides"]
        self.assertEqual(nsides, [NSIDE >> i for i in range(LEVELS)])
        for i, nside in enumerate(nsides):
            level = bundle["levels"][f"L{i}"]
            self.assertEqual(int(level["k"]), 8)
            mask = level["edge_mask"].reshape(int(level["num_nodes"]), 8)
            degree = mask.sum(dim=1)
            with self.subTest(nside=nside):
                if nside == 1:
                    self.assertEqual(int(level["num_nodes"]), 12)
                    self.assertTrue(bool((degree == 6).all()))
                else:
                    self.assertEqual(int((degree == 7).sum()), 24)
                    self.assertEqual(int((degree == 8).sum()), int(level["num_nodes"]) - 24)
                # Whatever the degree, no node may be left with only padding --
                # GraphLevel rejects that, and attention would produce NaN.
                self.assertGreaterEqual(int(degree.min()), 6)

    def test_masked_slots_are_self_edges_and_have_zero_edge_attr(self):
        bundle = _bundle()
        level = bundle["levels"]["L0"]
        n, k = int(level["num_nodes"]), int(level["k"])
        flat_mask = level["edge_mask"].reshape(-1)
        src, dst = level["edge_index"][0], level["edge_index"][1]
        # Padding must be a self-edge so every gather stays in range...
        self.assertTrue(bool((src[~flat_mask] == dst[~flat_mask]).all()))
        # ...and inert, so it cannot leak geometry into the message.
        self.assertTrue(bool((level["edge_attr"][~flat_mask] == 0.0).all()))
        # Real slots must never be self-loops.
        self.assertFalse(bool((src[flat_mask] == dst[flat_mask]).any()))
        self.assertEqual(int(flat_mask.numel()), n * k)

    def test_densified_coarsest_level_falls_back_to_knn(self):
        """The dense-L3K24 recipe has to survive the substrate swap.

        Native adjacency has exactly 8 slots and cannot express k=24, so that level
        must switch to spherical kNN over the pixel centers rather than silently
        collapse to k=8 -- otherwise the HEALPix run is not architecture-matched to
        the lat-lon baseline it is being compared against.
        """
        bundle = _bundle_dense()
        metadata = bundle["metadata"]
        self.assertEqual(metadata["level_k_neighbors"], [8, 8, 24])
        self.assertEqual(
            metadata["level_connectivity_strategies"],
            {"L0": "native_healpix", "L1": "native_healpix", "L2": "pure_spherical_knn"},
        )
        # Native levels keep their mask; the densified one has none (kNN returns
        # exactly k real neighbours, so there is nothing to pad).
        self.assertIn("edge_mask", bundle["levels"]["L0"])
        self.assertNotIn("edge_mask", bundle["levels"]["L2"])
        self.assertEqual(int(bundle["levels"]["L2"]["k"]), 24)
        self.assertEqual(metadata["masked_edge_slots"][-1], 0)

    def test_level_k_below_the_mesh_degree_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "below the HEALPix mesh degree"):
            mesh_builder.build_healpix_mesh_bundle(
                nside=NSIDE, num_graph_levels=LEVELS, grid_shape=GRID,
                grid_lat_lon=mesh_builder._grid_latlon(*GRID),
                level_k_neighbors=[8, 8, 6],
            )

    def test_level_k_length_must_match_levels(self):
        with self.assertRaisesRegex(ValueError, "length num_graph_levels"):
            mesh_builder.build_healpix_mesh_bundle(
                nside=NSIDE, num_graph_levels=LEVELS, grid_shape=GRID,
                grid_lat_lon=mesh_builder._grid_latlon(*GRID),
                level_k_neighbors=[8, 8],
            )

    def test_mesh_levels_carry_no_grid_shape(self):
        # A mesh level has no (height, width); num_nodes is authoritative. This is
        # what distinguishes it from the hpx32-native level, which IS the data grid.
        bundle = _bundle()
        for i in range(LEVELS):
            self.assertEqual(int(bundle["levels"][f"L{i}"]["height"]), 0)
            self.assertEqual(int(bundle["levels"][f"L{i}"]["width"]), 0)
        self.assertEqual(int(bundle["grid"]["height"]), GRID[0])
        self.assertEqual(int(bundle["grid"]["width"]), GRID[1])


@unittest.skipUnless(HAVE_HEALPY, "healpy is an optional extra")
class ConservativeRegridTest(unittest.TestCase):
    def test_both_operators_are_row_normalized(self):
        bundle = _bundle()
        n_mesh = int(bundle["levels"]["L0"]["num_nodes"])
        n_grid = GRID[0] * GRID[1]
        for name, n_dst in (("g2m", n_mesh), ("m2g", n_grid)):
            edge_index = bundle[name]["edge_index"]
            weight = bundle[name]["edge_weight"]
            row_sum = torch.zeros(n_dst).index_add_(0, edge_index[1], weight)
            torch.testing.assert_close(
                row_sum, torch.ones(n_dst), atol=1.0e-5, rtol=0.0,
                msg=f"{name} rows must sum to 1 so a constant field is preserved",
            )

    def test_constant_field_survives_the_round_trip(self):
        # The property that makes the remap conservative: no artificial gradients
        # at the pole rows or at HEALPix face boundaries.
        bundle = _bundle()
        graph = GraphBundle(bundle)
        remap = FixedBipartiteRemap()
        n_grid = GRID[0] * GRID[1]
        ones = torch.ones(1, n_grid, 1)
        mesh = remap(ones, graph.g2m_edge_index, graph.g2m_edge_weight, graph.L0.num_nodes)
        back = remap(mesh, graph.m2g_edge_index, graph.m2g_edge_weight, n_grid)
        torch.testing.assert_close(mesh, torch.ones_like(mesh), atol=1.0e-5, rtol=0.0)
        torch.testing.assert_close(back, torch.ones_like(back), atol=1.0e-5, rtol=0.0)

    def test_in_model_remap_matches_the_offline_regrid(self):
        """The in-model weights and the offline NetCDF conversion must agree.

        Both come from the same operators, and this is what lets a HEALPix-native
        dataset and a HEALPix-mesh run be compared at all.
        """
        from src.healpix_regrid import (
            healpix_to_latlon,
            latlon_axes,
            latlon_to_healpix,
            load_or_build_operators,
        )

        bundle = _bundle()
        graph = GraphBundle(bundle)
        height, width = GRID
        lats, lons = latlon_axes(height, width)
        w_fwd, w_bwd = load_or_build_operators(
            height, width, NSIDE, 8,
            str(PROJECT_ROOT / "tests" / "_healpix_regrid_cache"), lats, lons,
        )
        rng = np.random.default_rng(0)
        field = rng.standard_normal((2, height, width)).astype(np.float32)
        remap = FixedBipartiteRemap()

        x = torch.from_numpy(field.reshape(2, height * width).T[None].copy())
        got = remap(x, graph.g2m_edge_index, graph.g2m_edge_weight, graph.L0.num_nodes)
        want = latlon_to_healpix(field, w_fwd)
        np.testing.assert_allclose(got[0].T.numpy(), want, atol=1.0e-5, rtol=0.0)

        y = torch.from_numpy(want.T[None].copy())
        got_back = remap(y, graph.m2g_edge_index, graph.m2g_edge_weight, height * width)
        want_back = healpix_to_latlon(want, w_bwd, height, width)
        np.testing.assert_allclose(
            got_back[0].T.numpy().reshape(2, height, width), want_back, atol=1.0e-5, rtol=0.0
        )

    def test_latitude_major_flattening_is_enforced(self):
        # Transposing the grid would silently produce wrong weights with no shape
        # error anywhere, so the builder must reject a longitude-major grid.
        height, width = GRID
        lat = np.linspace(-87.5, 87.5, height)
        lon = np.linspace(0.0, 350.0, width)
        lon_grid, lat_grid = np.meshgrid(lon, lat, indexing="ij")  # wrong order
        bad = np.deg2rad(
            np.stack([lat_grid.reshape(-1), lon_grid.reshape(-1)], axis=1)
        ).astype(np.float32)
        with self.assertRaisesRegex(ValueError, "latitude-major"):
            mesh_builder.build_healpix_mesh_bundle(
                nside=NSIDE, num_graph_levels=LEVELS, grid_shape=GRID, grid_lat_lon=bad,
            )

    def test_degenerate_grid_is_rejected(self):
        # (npix, 1) is the hpx32-NATIVE data shape; regridding from it is meaningless.
        with self.assertRaisesRegex(ValueError, "hpx32"):
            mesh_builder.build_healpix_mesh_bundle(
                nside=NSIDE, num_graph_levels=LEVELS, grid_shape=(192, 1),
            )


@unittest.skipUnless(HAVE_HEALPY, "healpy is an optional extra")
class HealpixMeshForwardTest(unittest.TestCase):
    def test_forward_is_latlon_in_latlon_out(self):
        bundle = _bundle()
        graph = GraphBundle(bundle)
        self.assertEqual(graph.graph_mode, "mesh")
        height, width = GRID
        model = GraphWeatherModel(
            graph,
            grid_shape=(height, width),
            input_channels=6,
            output_channels=3,
            hidden_dim=16,
            heads=2,
            edge_dim=6,
            num_graph_levels=LEVELS,
            mesh_encoder={
                "enabled": True,
                "mesh_type": "healpix",
                "boundary_type": "fixed_spherical",
                "nside": NSIDE,
            },
        )
        x = torch.randn(2, 6, height, width)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(tuple(out.shape), (2, 3, height, width))
        self.assertTrue(bool(torch.isfinite(out).all()))

    def test_gradient_flows_through_both_remaps(self):
        bundle = _bundle()
        graph = GraphBundle(bundle)
        height, width = GRID
        model = GraphWeatherModel(
            graph,
            grid_shape=(height, width),
            input_channels=6,
            output_channels=3,
            hidden_dim=16,
            heads=2,
            edge_dim=6,
            num_graph_levels=LEVELS,
            mesh_encoder={
                "enabled": True,
                "mesh_type": "healpix",
                "boundary_type": "fixed_spherical",
                "nside": NSIDE,
            },
        )
        x = torch.randn(2, 6, height, width, requires_grad=True)
        model(x).mean().backward()
        self.assertGreater(float(x.grad.abs().sum()), 0.0)

    def test_boundary_is_parameter_free(self):
        # fixed_spherical must add no learned weights at the boundary: the remap is
        # geometry, and this is what makes the round-trip error a hard floor.
        bundle = _bundle()
        graph = GraphBundle(bundle)
        model = GraphWeatherModel(
            graph,
            grid_shape=GRID,
            input_channels=6,
            output_channels=3,
            hidden_dim=16,
            heads=2,
            edge_dim=6,
            num_graph_levels=LEVELS,
            mesh_encoder={
                "enabled": True,
                "mesh_type": "healpix",
                "boundary_type": "fixed_spherical",
                "nside": NSIDE,
            },
        )
        self.assertEqual(sum(p.numel() for p in model.grid2mesh.parameters()), 0)
        self.assertEqual(sum(p.numel() for p in model.mesh2grid.parameters()), 0)


@unittest.skipUnless(HAVE_HEALPY, "healpy is an optional extra")
class HealpixMeshCacheIdentityTest(unittest.TestCase):
    def _expected(self, **overrides):
        kwargs = dict(
            nside=NSIDE,
            num_graph_levels=LEVELS,
            grid_shape=GRID,
            grid_lat_lon=mesh_builder._grid_latlon(*GRID),
            resolution_mode="test",
            regrid_oversample=8,
        )
        kwargs.update(overrides)
        return mesh_builder.expected_healpix_mesh_metadata(**kwargs)

    def test_matching_expectation_validates(self):
        raw = {"metadata": _bundle()["metadata"], "g2m": _bundle()["g2m"]}
        self.assertEqual(mesh_builder.validate_mesh_cache_metadata(raw, self._expected()), [])

    def test_nside_and_oversample_are_part_of_the_identity(self):
        raw = {"metadata": _bundle()["metadata"], "g2m": _bundle()["g2m"]}
        for overrides, key in (
            ({"nside": NSIDE * 2}, "nside"),
            ({"regrid_oversample": 4}, "regrid_oversample"),
            ({"level_k_neighbors": [8, 8, 24]}, "level_k_neighbors"),
        ):
            mismatches = mesh_builder.validate_mesh_cache_metadata(raw, self._expected(**overrides))
            self.assertTrue(
                any(m.startswith(key) for m in mismatches),
                f"{key} change must invalidate the cache, got {mismatches}",
            )

    def test_icosphere_expectation_rejects_a_healpix_cache(self):
        raw = {"metadata": _bundle()["metadata"], "g2m": _bundle()["g2m"]}
        expected = mesh_builder.expected_mesh_metadata(
            refinement=3,
            num_graph_levels=LEVELS,
            grid_shape=GRID,
            grid_lat_lon=mesh_builder._grid_latlon(*GRID),
            g2m_radius_factor=0.6,
            resolution_mode="test",
            bipartite_mapping_type=mesh_builder.bipartite_mapping_type("fixed_spherical"),
            bipartite_edge_features=mesh_builder.bipartite_edge_feature_set("fixed_spherical"),
        )
        mismatches = mesh_builder.validate_mesh_cache_metadata(raw, expected)
        self.assertTrue(mismatches)
        self.assertTrue(any(m.startswith("mesh_format_version") for m in mismatches))

    def test_icosphere_metadata_gains_no_healpix_keys(self):
        """The HEALPix identity keys must be absent on the icosphere path.

        They are validated with a None default, so an existing icosphere cache on
        disk keeps matching -- adding them unconditionally would silently rebuild
        every icosphere graph.
        """
        expected = mesh_builder.expected_mesh_metadata(
            refinement=3,
            num_graph_levels=LEVELS,
            grid_shape=GRID,
            grid_lat_lon=mesh_builder._grid_latlon(*GRID),
            g2m_radius_factor=0.6,
            resolution_mode="test",
        )
        for key in ("mesh_hierarchy", "nside", "regrid_oversample"):
            self.assertNotIn(key, expected)


class HealpixMeshResolveIdempotenceTest(unittest.TestCase):
    """resolve_mesh_encoder is applied to its OWN output; it must be idempotent.

    Regression for a real crash: ``MeshEncoderConfig.asdict()`` used to re-emit
    ``refinement``/``g2m_radius_factor``/``coarse_level_connectivity`` at their
    defaults even for a HEALPix mesh. ``architecture_metadata`` (trainer __init__)
    and ``validate_checkpoint_architecture`` both re-resolve an already-resolved
    config, so the second pass saw those as explicitly-set icosphere keys and
    raised "keys [...] only apply to mesh_type='icosphere'" -- training died before
    the first step.
    """

    RAW = {"enabled": True, "mesh_type": "healpix", "nside": 32,
           "boundary_type": "fixed_spherical"}

    def test_asdict_drops_icosphere_only_keys_for_healpix(self):
        resolved = resolve_mesh_encoder({"mesh_encoder": self.RAW}).asdict()
        for key in ("refinement", "g2m_radius_factor", "coarse_level_connectivity"):
            self.assertNotIn(key, resolved)
        self.assertEqual(resolved["mesh_type"], "healpix")

    def test_resolving_the_resolved_output_is_stable(self):
        current = dict(self.RAW)
        seen = []
        for _ in range(4):
            current = resolve_mesh_encoder({"mesh_encoder": current}).asdict()
            seen.append(current)
        for later in seen[1:]:
            self.assertEqual(seen[0], later)

    def test_architecture_metadata_and_checkpoint_validation_round_trip(self):
        from src.architecture import architecture_metadata, validate_checkpoint_architecture

        source = {
            "mesh_encoder": dict(self.RAW),
            "num_graph_levels": 4,
            "level_k_neighbors": [8, 8, 8, 8],
            "input_channels": 134,
            "output_channels": 67,
        }
        metadata = architecture_metadata(source)
        self.assertEqual(metadata["graph_mode"], "mesh")
        self.assertEqual(metadata["mesh_encoder"]["mesh_type"], "healpix")
        # This is the call that used to raise.
        validate_checkpoint_architecture(metadata, source)

    def test_icosphere_asdict_still_carries_its_keys(self):
        resolved = resolve_mesh_encoder(
            {"mesh_encoder": {"enabled": True, "refinement": 5, "boundary_type": "legacy"}}
        ).asdict()
        for key in ("refinement", "g2m_radius_factor", "coarse_level_connectivity"):
            self.assertIn(key, resolved)
        self.assertNotIn("mesh_type", resolved)


class HealpixMeshEvaluatorValidationTest(unittest.TestCase):
    """The evaluator's mesh-cache validation had five icosphere-only assumptions.

    All of them fired on a valid HEALPix bundle: int(refinement=None) TypeError,
    the barycentric-vs-conservative mapping type, normalizing 'native_healpix'
    against the icosphere enum, hierarchy_type != 'icosphere', and the
    coarse-connectivity-derived strategy. Eval is the stage right after training,
    so this is on the critical path.
    """

    @unittest.skipUnless(HAVE_HEALPY, "healpy is an optional extra")
    def _cfg(self, **overrides):
        from types import SimpleNamespace

        bundle = _bundle()
        metadata = bundle["metadata"]
        base = dict(
            mesh_encoder={"enabled": True, "mesh_type": "healpix", "nside": NSIDE,
                          "boundary_type": "fixed_spherical", "regrid_oversample": 8,
                          "grid_attention_encoder_blocks": 0,
                          "grid_attention_decoder_blocks": 0},
            mesh_format_version=mesh_builder.HEALPIX_MESH_FORMAT_VERSION,
            graph_format_version=3,
            level_k_neighbors=[8] * LEVELS,
            k_neighbors=8,
            num_graph_levels=LEVELS,
            grid_shape=GRID,
            resolution_mode="test",
            graph_connectivity_strategy="hybrid_row_aware_knn",
            hierarchy_type="standard",
            use_l4_ratio15=False,
            level_shapes=None,
            node_counts=metadata["node_counts"],
            edge_counts=metadata["edge_counts"],
        )
        base.update(overrides)
        return SimpleNamespace(**base), metadata

    def _validate(self, cfg, metadata):
        from src.evaluator import GraphWeatherEvaluator

        evaluator = object.__new__(GraphWeatherEvaluator)
        evaluator.cfg = cfg
        evaluator._validate_graph_resolution(metadata)

    @unittest.skipUnless(HAVE_HEALPY, "healpy is an optional extra")
    def test_valid_healpix_bundle_passes(self):
        cfg, metadata = self._cfg()
        self._validate(cfg, metadata)  # must not raise

    @unittest.skipUnless(HAVE_HEALPY, "healpy is an optional extra")
    def test_mismatched_nside_and_oversample_are_rejected(self):
        for key, bad in (("nside", NSIDE * 2), ("regrid_oversample", 4)):
            cfg, metadata = self._cfg()
            cfg.mesh_encoder = {**cfg.mesh_encoder, key: bad}
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    self._validate(cfg, metadata)

    @unittest.skipUnless(HAVE_HEALPY, "healpy is an optional extra")
    def test_icosphere_config_against_a_healpix_cache_gives_a_clear_error(self):
        """Used to be an opaque TypeError from int(refinement=None)."""
        cfg, metadata = self._cfg()
        cfg.mesh_encoder = {**cfg.mesh_encoder, "mesh_type": "icosphere", "refinement": 3}
        with self.assertRaisesRegex(ValueError, "no icosphere refinement"):
            self._validate(cfg, metadata)


class HealpixMeshConfigTest(unittest.TestCase):
    """Config-layer guards. No healpy needed -- nothing here builds geometry."""

    def test_config_layer_derives_healpix_counts(self):
        from src.architecture import normalize_model_config_dict

        resolved = normalize_model_config_dict(
            {
                "num_graph_levels": 4,
                "mesh_encoder": {
                    "enabled": True,
                    "mesh_type": "healpix",
                    "nside": 32,
                    "boundary_type": "fixed_spherical",
                },
            }
        )
        self.assertEqual(resolved["node_counts"], [12288, 3072, 768, 192])
        self.assertEqual(resolved["level_k_neighbors"], [8, 8, 8, 8])
        self.assertEqual(resolved["edge_counts"], [98304, 24576, 6144, 1536])
        # Pins the config layer's literal to the builder's constant.
        self.assertEqual(
            resolved["mesh_format_version"], mesh_builder.HEALPIX_MESH_FORMAT_VERSION
        )

    def test_icosphere_resolution_is_unchanged(self):
        # The three new keys must not appear in a resolved icosphere config, or
        # every existing resolved-config golden would shift.
        resolved = resolve_mesh_encoder(
            {"mesh_encoder": {"enabled": True, "refinement": 5, "boundary_type": "fixed_spherical"}}
        ).asdict()
        for key in ("mesh_type", "nside", "regrid_oversample"):
            self.assertNotIn(key, resolved)

    def test_healpix_resolution_emits_the_new_keys(self):
        resolved = resolve_mesh_encoder(
            {
                "mesh_encoder": {
                    "enabled": True,
                    "mesh_type": "healpix",
                    "nside": 32,
                    "boundary_type": "fixed_spherical",
                }
            }
        ).asdict()
        self.assertEqual(resolved["mesh_type"], "healpix")
        self.assertEqual(resolved["nside"], 32)
        self.assertEqual(resolved["regrid_oversample"], 8)

    def test_guards(self):
        cases = [
            ({"mesh_type": "healpix", "boundary_type": "graphcast_mlp"}, "fixed_spherical"),
            ({"mesh_type": "healpix", "boundary_type": "fixed_spherical", "nside": 24}, "power of two"),
            ({"mesh_type": "healpix", "boundary_type": "fixed_spherical", "refinement": 5}, "icosphere"),
            ({"mesh_type": "icosphere", "nside": 32}, "healpix"),
            ({"mesh_type": "cubed_sphere"}, "Unsupported mesh_encoder.mesh_type"),
        ]
        for cfg, pattern in cases:
            with self.subTest(cfg=cfg):
                with self.assertRaisesRegex(ValueError, pattern):
                    resolve_mesh_encoder({"mesh_encoder": {"enabled": True, **cfg}})

    def test_nside_too_small_for_the_level_count(self):
        from src.architecture import normalize_model_config_dict

        with self.assertRaisesRegex(ValueError, "too small for"):
            normalize_model_config_dict(
                {
                    "num_graph_levels": 5,
                    "mesh_encoder": {
                        "enabled": True,
                        "mesh_type": "healpix",
                        "nside": 8,
                        "boundary_type": "fixed_spherical",
                    },
                }
            )

    HPX_EMA_ITV_NAME = (
        "2p5_hpxmeshns32_l3_h160_hpxedges_s1x100lr5e4_currS2toS10x3_flatlr1e6"
        "_perfctl_compilefull_ema_itv"
    )
    BASELINE_NAME = (
        "2p5_l3_h160_densel3k24_s1x100lr1e3_currS2toS10x3_flatlr1e6"
        "_perfctl_compilefull_ema_itv"
    )
    BASELINE_FILE = (
        "config_2p5_l3_hidden160_s1x100_lr1e3_currS2toS10x3_flatlr1e6"
        "_dense_l3k24_perfctl_compilefull_ema_itv.yaml"
    )

    def _baseline_and_hpx(self):
        from src.config import YParams

        configs = PROJECT_ROOT / "configs" / "experiments"
        base = YParams(
            str(configs / self.BASELINE_FILE), self.BASELINE_NAME, resolution_mode="2p5"
        ).params
        hpx = YParams(
            str(configs / f"config_{self.HPX_EMA_ITV_NAME}.yaml"),
            self.HPX_EMA_ITV_NAME,
            resolution_mode="2p5",
        ).params
        return base, hpx

    def test_hpx_config_changes_only_substrate_edges_and_lr(self):
        """The controlled-comparison guarantee, enforced.

        Three intentional differences from the best-2.5-deg baseline: the HEALPix
        substrate, native HEALPix edges instead of the dense-L3 k=24 level, and a
        5.0e-4 peak LR. EVERYTHING else -- EMA, ITV, the epoch/curriculum schedule,
        the rest of the optimizer, and every architecture knob -- must match. If
        anyone edits one file and not the other, this fails.
        """
        base, hpx = self._baseline_and_hpx()
        SUBSTRATE = {
            "mesh_encoder", "mesh_format_version", "node_counts", "edge_counts",
            "graph_path", "model", "resolution_profiles",
        }
        # graph_format_version is DERIVED from level_k_neighbors (uniform k -> 3,
        # mixed -> 4), so it moves as a consequence of the edge change rather than
        # as a fourth decision. It is inert in mesh mode -- mesh bundles are
        # validated against mesh_format_version.
        EDGES = {"level_k_neighbors", "l3_k_neighbors", "graph_format_version"}
        LR = {"lr", "min_lr", "rollout_stage_lr_schedule"}
        IDENTITY = {"experiment_name", "run_name", "name", "wandb"}
        allowed = SUBSTRATE | EDGES | LR | IDENTITY
        for key in sorted(set(base) | set(hpx)):
            if key in allowed:
                continue
            self.assertEqual(
                base.get(key), hpx.get(key),
                f"{key!r} differs from the baseline but is not one of the three "
                "intended changes (substrate, edges, lr).",
            )

    def test_ema_itv_and_schedule_match_the_baseline(self):
        base, hpx = self._baseline_and_hpx()
        self.assertEqual(base["ema"], hpx["ema"])
        self.assertEqual(base["loss_channel_weighting"], hpx["loss_channel_weighting"])
        self.assertIs(hpx["ema"]["enabled"], True)
        self.assertIs(hpx["loss_channel_weighting"]["inverse_tendency_variance"], True)
        for key in (
            "weight_decay", "scheduler", "max_epochs", "batch_size",
            "gradient_accumulation_steps", "max_gradient_norm", "lr_schedule_type",
            "warmup_epochs", "warmup_start_factor", "rollout_stage_epochs",
            "rollout_schedule", "rollout_mode", "hidden_dim", "num_heads", "head_dim",
            "num_graph_levels", "encoder_blocks", "decoder_blocks", "l0_blocks",
            "l1_blocks", "l2_blocks", "l3_blocks", "use_delta_normalization",
            "delta_stats_path", "resolution_mode", "grid_shape", "train_data_path",
            "compile_scope", "attention_impl", "edge_projection_cache",
        ):
            self.assertEqual(base[key], hpx[key], f"{key} must match the baseline")

    def test_lr_is_5e4_in_both_the_top_level_and_the_stage_schedule(self):
        """The stage entry is authoritative under rollout_stage_warmup_cosine.

        ``_build_rollout_stage_warmup_cosine_scheduler`` builds purely from
        ``rollout_stage_lr_schedule``, so a config whose top-level ``lr`` disagrees
        with stage S=1 trains at the STAGE value while reporting the other. Both
        must say 5.0e-4 or the run is not the run the filename claims.
        """
        _, hpx = self._baseline_and_hpx()
        self.assertEqual(hpx["lr"], 5.0e-4)
        self.assertEqual(hpx["min_lr"], 2.5e-5)
        self.assertEqual(hpx["lr_schedule_type"], "rollout_stage_warmup_cosine")
        stage_one = [s for s in hpx["rollout_stage_lr_schedule"] if int(s["rollout_steps"]) == 1]
        self.assertEqual(len(stage_one), 1)
        self.assertEqual(float(stage_one[0]["max_lr"]), 5.0e-4)
        self.assertEqual(float(stage_one[0]["min_lr"]), 2.5e-5)
        # The curriculum stages stay at the baseline's flat 1.0e-6.
        for stage in hpx["rollout_stage_lr_schedule"]:
            if int(stage["rollout_steps"]) > 1:
                self.assertEqual(float(stage["max_lr"]), 1.0e-6)

    def test_edges_are_native_healpix_on_every_level(self):
        """No dense k=24 level: 24 is not HEALPix connectivity.

        Native adjacency has exactly 8 slots, so a k=24 level would have to be
        spherical kNN over the pixel centers. Declaring 8 everywhere keeps the whole
        U-Net on true HEALPix edges, and the derived edge_counts must be 8x the
        pixel counts.
        """
        base, hpx = self._baseline_and_hpx()
        self.assertEqual(hpx["level_k_neighbors"], [8, 8, 8, 8])
        self.assertEqual(hpx["l3_k_neighbors"], 8)
        self.assertEqual(base["level_k_neighbors"], [8, 8, 8, 24])  # what changed
        # Uniform k makes the derived lat-lon graph version 3 rather than 4.
        self.assertEqual(hpx["graph_format_version"], 3)
        self.assertEqual(base["graph_format_version"], 4)
        self.assertEqual(hpx["node_counts"], [12288, 3072, 768, 192])
        self.assertEqual(
            hpx["edge_counts"], [n * 8 for n in hpx["node_counts"]]
        )
        self.assertEqual(hpx["mesh_encoder"]["mesh_type"], "healpix")

    def test_built_bundle_matches_the_configs_declared_counts(self):
        """The YAML's node/edge counts must be what the builder actually produces."""
        if not HAVE_HEALPY:
            self.skipTest("healpy is an optional extra")
        _, hpx = self._baseline_and_hpx()
        bundle = _bundle(nside=8, levels=4, grid=GRID, level_k=[8, 8, 8, 8])
        metadata = bundle["metadata"]
        # Same shape of claim at the test nside: every level native, 8x edges.
        self.assertEqual(metadata["level_k_neighbors"], [8, 8, 8, 8])
        self.assertTrue(
            all(s == "native_healpix" for s in metadata["level_connectivity_strategies"].values()),
            metadata["level_connectivity_strategies"],
        )
        self.assertEqual(
            metadata["edge_counts"], [n * 8 for n in metadata["node_counts"]]
        )
        # ...and the config's own arithmetic is the same rule at nside 32.
        self.assertEqual(hpx["edge_counts"], [n * 8 for n in hpx["node_counts"]])

    def test_experiment_config_resolves(self):
        from src.config import YParams

        name = self.HPX_EMA_ITV_NAME
        params = YParams(
            str(PROJECT_ROOT / "configs" / "experiments" / f"config_{name}.yaml"),
            name,
            resolution_mode="2p5",
        )
        # The point of this design: the data stays lat-lon, so nothing outside the
        # model changes. Assert that explicitly -- it is the whole contract.
        self.assertEqual(params.params["resolution_mode"], "2p5")
        self.assertEqual(list(params.params["grid_shape"]), [72, 144])
        self.assertTrue(str(params.params["delta_stats_path"]).endswith("era5_67_2p5_delta_stats.npz"))
        self.assertIs(params.params["use_delta_normalization"], True)
        self.assertEqual(params.params["mesh_encoder"]["mesh_type"], "healpix")
        self.assertEqual(params.params["mesh_encoder"]["nside"], 32)
        self.assertEqual(params.params["mesh_encoder"]["boundary_type"], "fixed_spherical")
        self.assertEqual(params.params["node_counts"], [12288, 3072, 768, 192])
        self.assertEqual(params.params["level_k_neighbors"], [8, 8, 8, 8])
        self.assertEqual(params.params["graph_path"], "graphs/graph_2p5_healpix_ns32_l3.pt")


if __name__ == "__main__":
    unittest.main()
