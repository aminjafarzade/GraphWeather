#!/usr/bin/env python3
"""Area-weighted regridding between a lat-lon grid and a HEALPix (NEST) map.

Builds two sparse operators from the exact spherical overlap between the two
tessellations and applies them to the repo's ``fields`` NetCDF layout:

    W_fwd : (npix, H*W)   lat-lon -> HEALPix   each HEALPix pixel is the
                                               area-weighted mean of the lat-lon
                                               cells overlapping it
    W_bwd : (H*W, npix)   HEALPix -> lat-lon   the transpose relationship, row
                                               normalized the other way

Both are row-normalized, so a constant field regrids to the same constant in
either direction and no artificial gradients appear at the pole rows or at
HEALPix face boundaries.

Overlap areas come from oversampling: each lat-lon cell is split into
``oversample**2`` sub-cells, each sub-cell's exact solid angle
``dlon * (sin(lat_hi) - sin(lat_lo))`` is accumulated into whichever HEALPix
pixel contains its center. That converges to the true overlap area and needs no
spherical-polygon clipping.

Usage
-----
    # convert a year of data, writing fields as [T, C, npix]
    python scripts/regrid_latlon_healpix.py --to-hpx  in.nc out.nc --nside 32

    # bring HEALPix predictions back to lat-lon for scoring
    python scripts/regrid_latlon_healpix.py --to-latlon in.nc out.nc --height 72 --width 144

    # per-variable round-trip error floor on real fields (report this next to
    # any short-lead comparison against a lat-lon baseline)
    python scripts/regrid_latlon_healpix.py --verify in.nc --nside 32

    # recompute normalization moments ON the HEALPix arrays (see plan section 0)
    python scripts/regrid_latlon_healpix.py --stats hpx_dir/ --out stats/

The importable API lives in ``src/healpix_regrid.py``: ``load_or_build_operators``,
``latlon_to_healpix``, ``healpix_to_latlon``, ``sparse_to_bipartite_edges``. This
file is only the CLI on top of it. ``mesh_builder.build_healpix_mesh_bundle``
consumes the SAME operators as fixed grid<->mesh weights, which is how the
in-model regridding path stays consistent with any data converted here.
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys
from typing import Iterable

import numpy as np

# The geometry/operator core lives in src/healpix_regrid.py so this CLI and the
# in-model path (mesh_builder.build_healpix_mesh_bundle, which turns the very
# same two operators into fixed grid<->mesh interpolation weights) share exactly
# one implementation. Duplicating it would let the offline data conversion and
# the in-model regrid drift apart silently.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.healpix_regrid import (  # noqa: E402
    DEFAULT_OVERSAMPLE,
    healpix_to_latlon,
    latlon_axes as cell_center_lat_lon,
    latlon_to_healpix,
    load_or_build_operators,
)
from src.resolution import healpix_npix  # noqa: E402
from src.resolution import require_healpy as _require_healpy  # noqa: E402

logger = logging.getLogger("regrid_latlon_healpix")


# --------------------------------------------------------------------------- #
# NetCDF conversion
# --------------------------------------------------------------------------- #

def _grid_axes(ds, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    lat_key = "latitude" if "latitude" in ds.variables else "lat" if "lat" in ds.variables else None
    lon_key = "longitude" if "longitude" in ds.variables else "lon" if "lon" in ds.variables else None
    if lat_key and lon_key:
        return (
            np.asarray(ds.variables[lat_key][:], dtype=np.float64),
            np.asarray(ds.variables[lon_key][:], dtype=np.float64),
        )
    logger.warning("No lat/lon variables found; assuming cell-centered %dx%d.", height, width)
    return cell_center_lat_lon(height, width)


def _copy_side_variables(src, dst, skip: Iterable[str]) -> None:
    """Carry channel names, time, and any other metadata across verbatim."""
    skip = set(skip)
    for name, var in src.variables.items():
        if name in skip:
            continue
        for dim in var.dimensions:
            if dim not in dst.dimensions:
                dst.createDimension(dim, len(src.dimensions[dim]))
        out = dst.createVariable(name, var.datatype, var.dimensions)
        out.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
        out[:] = var[:]


def convert_to_healpix(in_path: str, out_path: str, nside: int,
                       oversample: int = DEFAULT_OVERSAMPLE,
                       cache_dir: str = "data/stats", chunk: int = 32) -> None:
    import netCDF4 as nc

    with nc.Dataset(in_path, "r") as src:
        fields = src["fields"]
        n_t, n_c, height, width = fields.shape
        lats, lons = _grid_axes(src, height, width)
        w_fwd, _ = load_or_build_operators(height, width, nside, oversample, cache_dir, lats, lons)
        npix = healpix_npix(nside)

        with nc.Dataset(out_path, "w") as dst:
            dst.createDimension("time", n_t)
            dst.createDimension("channel", n_c)
            dst.createDimension("pixel", npix)
            out = dst.createVariable("fields", "f4", ("time", "channel", "pixel"),
                                     zlib=True, complevel=1)
            dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})
            # the attributes src/data.py keys the HEALPix loader path off
            dst.grid = "healpix"
            dst.nside = int(nside)
            dst.ordering = "nest"
            dst.regrid_source_grid = f"{int(height)}x{int(width)}"
            dst.regrid_oversample = int(oversample)
            dst.regrid_scheme = "area_weighted_row_normalized"

            hp = _require_healpy()
            lon_p, lat_p = hp.pix2ang(int(nside), np.arange(npix), nest=True, lonlat=True)
            dst.createVariable("pixel_lat", "f4", ("pixel",))[:] = lat_p.astype(np.float32)
            dst.createVariable("pixel_lon", "f4", ("pixel",))[:] = lon_p.astype(np.float32)
            _copy_side_variables(src, dst, skip={"fields", "latitude", "longitude", "lat", "lon"})

            for start in range(0, n_t, chunk):
                stop = min(start + chunk, n_t)
                block = np.asarray(fields[start:stop], dtype=np.float32)
                out[start:stop] = latlon_to_healpix(block, w_fwd).astype(np.float32)
            logger.info("Wrote %s: fields[%d, %d, %d]", out_path, n_t, n_c, npix)


def convert_to_latlon(in_path: str, out_path: str, height: int, width: int,
                      oversample: int = DEFAULT_OVERSAMPLE,
                      cache_dir: str = "data/stats", chunk: int = 32) -> None:
    import netCDF4 as nc

    with nc.Dataset(in_path, "r") as src:
        fields = src["fields"]
        n_t, n_c, npix = fields.shape[0], fields.shape[1], fields.shape[-1]
        nside = int(round(float(np.sqrt(npix / 12.0))))
        if healpix_npix(nside) != npix:
            raise ValueError(f"{npix} pixels is not a valid HEALPix map (12*nside^2).")
        lats, lons = cell_center_lat_lon(height, width)
        _, w_bwd = load_or_build_operators(height, width, nside, oversample, cache_dir, lats, lons)

        with nc.Dataset(out_path, "w") as dst:
            dst.createDimension("time", n_t)
            dst.createDimension("channel", n_c)
            dst.createDimension("lat", int(height))
            dst.createDimension("lon", int(width))
            out = dst.createVariable("fields", "f4", ("time", "channel", "lat", "lon"),
                                     zlib=True, complevel=1)
            dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})
            dst.grid = "latlon"
            dst.regrid_source_grid = f"healpix_nside{nside}_nest"
            dst.createVariable("latitude", "f4", ("lat",))[:] = lats.astype(np.float32)
            dst.createVariable("longitude", "f4", ("lon",))[:] = lons.astype(np.float32)
            _copy_side_variables(src, dst, skip={"fields", "pixel_lat", "pixel_lon"})

            for start in range(0, n_t, chunk):
                stop = min(start + chunk, n_t)
                block = np.asarray(fields[start:stop], dtype=np.float32)
                out[start:stop] = healpix_to_latlon(block, w_bwd, height, width).astype(np.float32)
            logger.info("Wrote %s: fields[%d, %d, %d, %d]", out_path, n_t, n_c, height, width)


# --------------------------------------------------------------------------- #
# verify + stats
# --------------------------------------------------------------------------- #

def verify_roundtrip(in_path: str, nside: int, oversample: int = DEFAULT_OVERSAMPLE,
                     cache_dir: str = "data/stats", n_times: int = 4,
                     channel_names: list[str] | None = None) -> list[dict]:
    """Per-channel lat-lon -> HEALPix -> lat-lon error, on real fields.

    This is the information floor of the grid change: no model can score better
    than this on the lat-lon grid, so report it beside any short-lead comparison
    against a lat-lon baseline.
    """
    import netCDF4 as nc

    rows: list[dict] = []
    with nc.Dataset(in_path, "r") as src:
        fields = src["fields"]
        n_t, n_c, height, width = fields.shape
        lats, lons = _grid_axes(src, height, width)
        w_fwd, w_bwd = load_or_build_operators(height, width, nside, oversample, cache_dir, lats, lons)
        take = min(int(n_times), int(n_t))
        block = np.asarray(fields[:take], dtype=np.float64)

        # cos-lat weights, so the floor is comparable to the repo's RMSE
        weight = np.cos(np.deg2rad(lats)).clip(0.0)
        weight = weight / weight.mean()
        weight = weight[None, None, :, None]

        back = healpix_to_latlon(latlon_to_healpix(block, w_fwd), w_bwd, height, width)
        err = back - block
        for c in range(n_c):
            rmse = float(np.sqrt((weight[:, 0] * err[:, c] ** 2).mean()))
            std = float(np.sqrt((weight[:, 0] * (block[:, c] - block[:, c].mean()) ** 2).mean()))
            rows.append({
                "channel": c,
                "name": (channel_names[c] if channel_names and c < len(channel_names) else str(c)),
                "roundtrip_rmse": rmse,
                "field_std": std,
                "rmse_over_std": (rmse / std) if std > 0 else float("nan"),
            })
    return rows


def compute_healpix_stats(hpx_dir: str, out_dir: str, chunk: int = 64) -> None:
    """Global per-channel mean/std over HEALPix files.

    On equal-area pixels the plain pixel moments ARE the area-weighted moments,
    which is exactly why lat-lon stats must not be reused: they over-weight the
    poles relative to a HEALPix map.
    """
    import netCDF4 as nc

    files = sorted(pathlib.Path(hpx_dir).glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No .nc files under {hpx_dir}")
    total = count = None
    sq = None
    for path in files:
        with nc.Dataset(path, "r") as ds:
            fields = ds["fields"]
            n_c = fields.shape[1]
            if total is None:
                total = np.zeros(n_c, dtype=np.float64)
                sq = np.zeros(n_c, dtype=np.float64)
                count = 0
            for start in range(0, fields.shape[0], chunk):
                stop = min(start + chunk, fields.shape[0])
                block = np.asarray(fields[start:stop], dtype=np.float64)
                total += block.sum(axis=(0, 2))
                sq += (block ** 2).sum(axis=(0, 2))
                count += block.shape[0] * block.shape[2]
        logger.info("stats: %s", path.name)
    mean = total / float(count)
    std = np.sqrt(np.maximum(sq / float(count) - mean ** 2, 0.0))
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "global_mean.npy", mean.reshape(1, -1, 1, 1).astype(np.float32))
    np.save(out / "global_std.npy", std.reshape(1, -1, 1, 1).astype(np.float32))
    logger.info("Wrote %s/global_mean.npy and global_std.npy (%d channels, %d pixel-samples)",
                out, mean.size, count)


# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--to-hpx", nargs=2, metavar=("IN", "OUT"))
    group.add_argument("--to-latlon", nargs=2, metavar=("IN", "OUT"))
    group.add_argument("--verify", metavar="IN")
    group.add_argument("--stats", metavar="HPX_DIR")
    parser.add_argument("--nside", type=int, default=32)
    parser.add_argument("--height", type=int, default=72)
    parser.add_argument("--width", type=int, default=144)
    parser.add_argument("--oversample", type=int, default=DEFAULT_OVERSAMPLE)
    parser.add_argument("--cache-dir", default="data/stats")
    parser.add_argument("--out", default="data/stats")
    parser.add_argument("--verify-times", type=int, default=4)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.to_hpx:
        convert_to_healpix(args.to_hpx[0], args.to_hpx[1], args.nside,
                           args.oversample, args.cache_dir)
    elif args.to_latlon:
        convert_to_latlon(args.to_latlon[0], args.to_latlon[1], args.height,
                          args.width, args.oversample, args.cache_dir)
    elif args.verify:
        rows = verify_roundtrip(args.verify, args.nside, args.oversample,
                                args.cache_dir, args.verify_times)
        print(f"{'ch':>4} {'name':<10} {'roundtrip_rmse':>15} {'field_std':>12} {'rmse/std':>10}")
        for row in rows:
            print(f"{row['channel']:>4} {row['name']:<10} {row['roundtrip_rmse']:>15.6g} "
                  f"{row['field_std']:>12.6g} {row['rmse_over_std']:>10.4f}")
    elif args.stats:
        compute_healpix_stats(args.stats, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
