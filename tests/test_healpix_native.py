"""HEALPix-native tests: ``resolution_mode: hpx32``, HEALPix data on disk.

The data itself is HEALPix, carried through the pipeline as a DEGENERATE lat-lon
grid of shape ``(npix, 1)`` so every ``[B, C, H, W]`` reshape, the GridNodeAdapter
and the normalization broadcast keep working unchanged. Contrast with
tests/test_healpix_mesh.py, the in-model-regrid design where the data stays
lat-lon and HEALPix is only the mesh.

Verifies: the quadtree level shapes (which the lat-lon per-axis halving would get
wrong), native adjacency and NEST pooling, a forward pass on the degenerate grid,
the loader's ``[T, C, npix]`` handling, and that every guard separating the two
grid kinds fires.

Also pins the property the whole design rests on: adding HEALPix must not perturb
any lat-lon graph bundle.

CPU-only. Geometry tests need healpy (skipped otherwise). unittest-style.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import graph_builder  # noqa: E402
from src.graph_bundle import GraphBundle  # noqa: E402
from src.models import GraphWeatherModel  # noqa: E402
from src.resolution import (  # noqa: E402
    cell_center_lat_lon,
    get_resolution_spec,
    grid_kind,
    healpix_level_nsides,
    healpix_npix,
    resolve_level_shapes_for_config,
)

try:
    import healpy  # noqa: F401

    HAVE_HEALPY = True
except ImportError:  # pragma: no cover - dependency guard
    HAVE_HEALPY = False


class HealpixResolutionSpecTest(unittest.TestCase):
    """No healpy needed: pixel counts are exact arithmetic."""

    def test_spec_is_a_degenerate_grid(self):
        spec = get_resolution_spec("hpx32")
        self.assertEqual(spec.name, "hpx32")
        self.assertEqual((spec.height, spec.width), (12288, 1))
        self.assertEqual(
            list(spec.level_shapes), [(12288, 1), (3072, 1), (768, 1), (192, 1)]
        )

    def test_level_shapes_are_a_quadtree_not_per_axis_halving(self):
        """The bug this guards: ceil(h/2) per axis would give 6144, not 3072.

        A HEALPix parent covers FOUR children (p >> 2), so the pixel count divides
        by 4 per level. Routing hpx32 through coarsened_level_shapes() would build a
        pyramid whose L1 has twice as many nodes as the pool map addresses.
        """
        shapes = resolve_level_shapes_for_config({}, "hpx32", num_graph_levels=4)
        self.assertEqual(shapes, [[12288, 1], [3072, 1], [768, 1], [192, 1]])
        for fine, coarse in zip(shapes, shapes[1:]):
            self.assertEqual(fine[0], 4 * coarse[0])

    def test_too_many_levels_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "HEALPix levels"):
            resolve_level_shapes_for_config({}, "hpx32", num_graph_levels=9)

    def test_grid_kind_defaults_to_latlon(self):
        self.assertEqual(grid_kind("hpx32"), "healpix")
        for mode in (None, "", "2p5", "5p625", "1p5"):
            self.assertEqual(grid_kind(mode), "latlon")

    def test_npix_rejects_non_power_of_two(self):
        self.assertEqual(healpix_npix(32), 12288)
        for bad in (0, 3, 24, -8):
            with self.assertRaises(ValueError):
                healpix_npix(bad)

    def test_level_nsides_halve(self):
        self.assertEqual(healpix_level_nsides(32, 4), (32, 16, 8, 4))
        with self.assertRaisesRegex(ValueError, "ran out of resolution"):
            healpix_level_nsides(4, 4)

    def test_cell_center_lat_lon_refuses_a_degenerate_grid(self):
        """width == 1 means HEALPix, so a linspace would be fabricated latitudes.

        Returning 12288 fake latitudes here would silently corrupt whatever consumed
        them -- edge features, climatology latitudes, or the loss weights.
        """
        with self.assertRaisesRegex(ValueError, "HEALPix"):
            cell_center_lat_lon(12288, 1)
        # The normal lat-lon case is untouched.
        lats, lons = cell_center_lat_lon(72, 144)
        self.assertEqual((lats.numel(), lons.numel()), (72, 144))


@unittest.skipUnless(HAVE_HEALPY, "healpy is an optional extra")
class HealpixNativeBundleTest(unittest.TestCase):
    NSIDE = 4
    LEVELS = 3

    def _bundle(self, level_k=None):
        spec = get_resolution_spec("hpx32")
        return graph_builder.build_graph_bundle(
            torch.zeros(1),
            torch.zeros(1),
            k=8,
            resolution=spec.resolution_degrees,
            connectivity_strategy=graph_builder.HEALPIX_NATIVE,
            resolution_mode="hpx32",
            num_graph_levels=4,
            level_k_neighbors=level_k,
        )

    def test_pool_map_is_the_nest_parent(self):
        pool = graph_builder.healpix_pool_map(192)
        torch.testing.assert_close(pool, torch.arange(192, dtype=torch.long) >> 2)
        self.assertTrue(bool((torch.bincount(pool) == 4).all()))
        with self.assertRaisesRegex(ValueError, "divisible by 4"):
            graph_builder.healpix_pool_map(10)

    def test_pixel_centers_are_lat_lon_radians(self):
        """Ordering trap: healpy's lonlat=True returns (lon, lat), the reverse of
        this repo's (lat, lon). Swapping them transposes the graph silently."""
        lat_lon = graph_builder.healpix_level_lat_lon(self.NSIDE)
        self.assertEqual(tuple(lat_lon.shape), (healpix_npix(self.NSIDE), 2))
        lat, lon = lat_lon[:, 0], lat_lon[:, 1]
        self.assertLessEqual(float(lat.abs().max()), np.pi / 2 + 1e-5)
        self.assertGreaterEqual(float(lon.min()), 0.0)
        self.assertLessEqual(float(lon.max()), 2 * np.pi + 1e-5)
        # A latitude range wider than a longitude range would mean they are swapped.
        self.assertGreater(float(lon.max() - lon.min()), float(lat.max() - lat.min()))

    def test_native_bundle_shapes_and_masking(self):
        bundle = self._bundle()
        self.assertEqual(bundle["metadata"]["grid_kind"], "healpix")
        self.assertEqual(bundle["metadata"]["ordering"], "nest")
        self.assertEqual(bundle["metadata"]["level_nsides"], [32, 16, 8, 4])
        self.assertEqual(
            bundle["metadata"]["graph_format_version"],
            graph_builder.GRAPH_FORMAT_VERSION_HPX,
        )
        for name, expected_nodes in zip(
            ("L0", "L1", "L2", "L3"), (12288, 3072, 768, 192)
        ):
            level = bundle["levels"][name]
            self.assertEqual(int(level["num_nodes"]), expected_nodes)
            # The data grid IS the pixel list here -- (npix, 1), not (0, 0).
            self.assertEqual((int(level["height"]), int(level["width"])), (expected_nodes, 1))
            diagnostics = level["neighbor_diagnostics"]
            self.assertEqual(diagnostics["degree_min"], 7)
            self.assertEqual(diagnostics["masked_edge_slots"], 24)
            self.assertEqual(diagnostics["degree_histogram"][7], 24)

    def test_a_densified_level_falls_back_to_spherical_knn(self):
        """Native adjacency has exactly 8 slots, so a level asking for k=24 cannot
        use it. That level must switch to kNN rather than silently get k=8."""
        bundle = self._bundle(level_k=[8, 8, 8, 24])
        strategies = bundle["metadata"]["level_connectivity_strategies"]
        self.assertEqual(strategies["L0"], graph_builder.HEALPIX_NATIVE)
        self.assertEqual(strategies["L3"], graph_builder.PURE_SPHERICAL_KNN)
        self.assertEqual(int(bundle["levels"]["L3"]["k"]), 24)
        self.assertNotIn("edge_mask", bundle["levels"]["L3"])

    def test_forward_pass_on_the_degenerate_grid(self):
        bundle = GraphBundle(self._bundle())
        model = GraphWeatherModel(
            bundle,
            grid_shape=(12288, 1),
            input_channels=6,
            output_channels=3,
            hidden_dim=16,
            heads=2,
            edge_dim=6,
            num_graph_levels=4,
            use_l3=True,
        )
        x = torch.randn(2, 6, 12288, 1)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(tuple(out.shape), (2, 3, 12288, 1))
        self.assertTrue(bool(torch.isfinite(out).all()))

    def test_strategy_and_grid_kind_must_agree(self):
        lats, lons = cell_center_lat_lon(72, 144)
        with self.assertRaisesRegex(ValueError, "requires a HEALPix"):
            graph_builder.build_graph_bundle(
                lats, lons, k=8, resolution=2.5,
                connectivity_strategy=graph_builder.HEALPIX_NATIVE,
                resolution_mode="2p5", num_graph_levels=3,
            )
        with self.assertRaisesRegex(ValueError, "Row-aware kNN is meaningless"):
            graph_builder.build_graph_bundle(
                torch.zeros(1), torch.zeros(1), k=8, resolution=1.8,
                connectivity_strategy=graph_builder.HYBRID_ROW_AWARE_KNN,
                resolution_mode="hpx32", num_graph_levels=3,
            )


class HealpixNativeLoaderGuardTest(unittest.TestCase):
    """The loader guards need no data: they fire during construction/validation."""

    def _config(self, **overrides):
        from src.data import DataConfig

        base = dict(
            in_channels=[0, 1],
            out_channels=[0, 1],
            normalization="none",
            resolution_mode="hpx32",
            grid_kind="healpix",
            add_grid=False,
            roll=False,
            crop_size_x=None,
            crop_size_y=None,
            dt=1,
            n_history=0,
        )
        base.update(overrides)
        return DataConfig(**base)

    def test_latlon_only_transforms_are_rejected(self):
        from src.data import _reject_latlon_only_transforms

        # A longitude roll or an (x, y) crop over (npix, 1) is not a rotation or a
        # window -- it reindexes pixels arbitrarily across face boundaries.
        with self.assertRaisesRegex(ValueError, "longitude axis"):
            _reject_latlon_only_transforms(self._config(roll=True))
        with self.assertRaisesRegex(ValueError, "crop"):
            _reject_latlon_only_transforms(self._config(crop_size_x=16))
        with self.assertRaisesRegex(ValueError, "pixel_lat_lon"):
            _reject_latlon_only_transforms(self._config(add_grid=True))
        # With real pixel centers supplied, add_grid is allowed.
        _reject_latlon_only_transforms(
            self._config(add_grid=True, pixel_lat_lon=np.zeros((192, 2), dtype=np.float32))
        )

    def test_grid_channels_use_real_pixel_centers(self):
        """The linspace sweep encodes GRID INDEX position, which at (npix, 1) would
        hand the model npix fake latitudes and a single longitude."""
        from src.data import _add_grid_channels

        npix = 12
        pixel_lat_lon = np.stack(
            [np.linspace(-1.0, 1.0, npix), np.linspace(0.0, 6.0, npix)], axis=1
        ).astype(np.float32)
        img = np.zeros((1, 2, npix, 1), dtype=np.float32)
        out = _add_grid_channels(img, "sinusoidal", 4, pixel_lat_lon=pixel_lat_lon)
        self.assertEqual(out.shape, (1, 6, npix, 1))
        np.testing.assert_allclose(
            out[0, 2, :, 0], np.sin(pixel_lat_lon[:, 0]), atol=1e-6
        )
        with self.assertRaisesRegex(ValueError, "rows but the field has"):
            _add_grid_channels(img, "sinusoidal", 4, pixel_lat_lon=pixel_lat_lon[:-1])


class LatLonUnaffectedTest(unittest.TestCase):
    """The backward-compatibility contract, pinned.

    Adding HEALPix must leave lat-lon bundles bit-identical and must not add a
    metadata key to them -- a new key would make validate_graph_cache_metadata
    report a mismatch and silently rebuild every lat-lon graph on disk.
    """

    def test_latlon_bundle_metadata_has_no_grid_kind_key(self):
        lats, lons = cell_center_lat_lon(18, 36)
        bundle = graph_builder.build_graph_bundle(
            lats, lons, k=8, resolution=10.0,
            connectivity_strategy=graph_builder.HYBRID_ROW_AWARE_KNN,
            resolution_mode="5p625", num_graph_levels=3,
        )
        metadata = bundle["metadata"]
        for key in ("grid_kind", "nside", "ordering", "level_nsides"):
            self.assertNotIn(
                key, metadata, f"{key} on a lat-lon bundle would invalidate every cache"
            )
        self.assertNotEqual(
            metadata["graph_format_version"], graph_builder.GRAPH_FORMAT_VERSION_HPX
        )
        # Absent therefore means lat-lon.
        self.assertEqual(metadata.get("grid_kind", "latlon"), "latlon")

    def test_latlon_levels_carry_no_edge_mask(self):
        # No mask -> LocalGraphAttention's masked_fill is skipped entirely, so the
        # lat-lon attention math is untouched by the HEALPix work.
        lats, lons = cell_center_lat_lon(18, 36)
        bundle = graph_builder.build_graph_bundle(
            lats, lons, k=8, resolution=10.0,
            connectivity_strategy=graph_builder.PURE_SPHERICAL_KNN,
            resolution_mode="5p625", num_graph_levels=3,
        )
        for name in ("L0", "L1", "L2"):
            self.assertNotIn("edge_mask", bundle["levels"][name])


if __name__ == "__main__":
    unittest.main()
