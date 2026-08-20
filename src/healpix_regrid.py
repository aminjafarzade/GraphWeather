"""Conservative (area-weighted) regridding operators between lat-lon and HEALPix.

This is the geometry core that both users share:

  * ``scripts/regrid_latlon_healpix.py`` -- the offline CLI that converts NetCDF
    files for the HEALPix-native pipeline (``resolution_mode: hpx32``).
  * ``mesh_builder.build_healpix_mesh_bundle`` -- the in-model path, where the
    same two operators become the FIXED grid->mesh / mesh->grid interpolation
    weights of a mesh bundle. There the data never leaves the lat-lon grid: the
    model regrids to HEALPix on the way in and back on the way out.

Two operators are built from the exact spherical overlap between the two
tessellations:

    W_fwd : (npix, H*W)   lat-lon -> HEALPix
    W_bwd : (H*W, npix)   HEALPix -> lat-lon

Both are row-normalized, so each destination value is the area-weighted mean of
the sources overlapping it. A constant field therefore regrids to the same
constant in either direction, and no artificial gradients appear at the pole rows
or at HEALPix face boundaries.

Overlap areas come from oversampling: each lat-lon cell is split into
``oversample**2`` sub-cells, each sub-cell's exact solid angle
``dlon * (sin(lat_hi) - sin(lat_lo))`` is accumulated into whichever HEALPix
pixel contains its center. That converges to the true overlap area and needs no
spherical-polygon clipping.

NOTE on ordering: the lat-lon axis is flattened row-major as ``i * width + j``
(latitude-major), which is exactly what ``GridNodeAdapter`` and
``mesh_builder``'s ``grid_lat_lon`` produce. Flattening the other way would
silently transpose every regrid.
"""
from __future__ import annotations

import logging
import pathlib

import numpy as np

from .resolution import healpix_npix, require_healpy

logger = logging.getLogger(__name__)

DEFAULT_OVERSAMPLE = 8
CACHE_SUBDIR = "regrid_operators"

__all__ = [
    "DEFAULT_OVERSAMPLE",
    "build_operators",
    "build_overlap_matrix",
    "healpix_to_latlon",
    "latitude_edges",
    "latlon_axes",
    "latlon_to_healpix",
    "load_or_build_operators",
    "operator_cache_path",
    "sparse_to_bipartite_edges",
]


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #

def latlon_axes(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centered lat-lon axes in degrees, matching ``resolution.cell_center_lat_lon``.

    Kept separate from that function because this one returns float64 numpy (the
    overlap integration needs the precision) and because ``cell_center_lat_lon``
    deliberately rejects ``width == 1``, which is the HEALPix-native grid shape.
    """
    height, width = int(height), int(width)
    dlat = 180.0 / float(height)
    dlon = 360.0 / float(width)
    lats = np.linspace(-90.0 + dlat / 2.0, 90.0 - dlat / 2.0, height, dtype=np.float64)
    lons = np.linspace(0.0, 360.0 - dlon, width, dtype=np.float64)
    return lats, lons


def latitude_edges(lats: np.ndarray) -> np.ndarray:
    """Cell edges from centers, clamped to the poles.

    Handles both conventions in this repo: 2p5/5p625 are cell-centered and
    pole-free, while the kai_1p5 grid includes both exact poles (so its first and
    last cells are half-height and MUST NOT be treated as full cells).
    """
    lats = np.asarray(lats, dtype=np.float64)
    if lats.size < 2:
        raise ValueError("Need at least two latitudes to derive cell edges.")
    edges = np.empty(lats.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (lats[:-1] + lats[1:])
    edges[0] = lats[0] - 0.5 * (lats[1] - lats[0])
    edges[-1] = lats[-1] + 0.5 * (lats[-1] - lats[-2])
    return np.clip(edges, -90.0, 90.0)


def build_overlap_matrix(
    lats: np.ndarray,
    lons: np.ndarray,
    nside: int,
    oversample: int = DEFAULT_OVERSAMPLE,
):
    """Sparse (npix, H*W) matrix of overlap solid angle, in steradians."""
    from scipy import sparse

    hp = require_healpy()
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    height, width = lats.size, lons.size
    n = int(oversample)
    if n < 1:
        raise ValueError("oversample must be >= 1.")

    lat_edges = latitude_edges(lats)
    dlon = 360.0 / float(width)

    # sub-cell latitude edges/centers per row: (H, n+1) and (H, n)
    frac = np.linspace(0.0, 1.0, n + 1)
    sub_lat_edges = lat_edges[:-1, None] + frac[None, :] * (lat_edges[1:, None] - lat_edges[:-1, None])
    sub_lat_centers = 0.5 * (sub_lat_edges[:, :-1] + sub_lat_edges[:, 1:])

    # exact solid angle of each sub-cell: dlon_rad * (sin(hi) - sin(lo))
    sin_edges = np.sin(np.deg2rad(sub_lat_edges))
    dlon_sub_rad = np.deg2rad(dlon) / float(n)
    sub_area = dlon_sub_rad * (sin_edges[:, 1:] - sin_edges[:, :-1])      # (H, n)
    sub_area = np.abs(sub_area)

    # sub-cell longitude centers per column: (W, n)
    sub_lon_centers = (
        lons[:, None] - 0.5 * dlon + (np.arange(n)[None, :] + 0.5) * (dlon / float(n))
    ) % 360.0

    # full sample set, ordered (H, n_lat, W, n_lon) so the cell index is i*W + j
    lat_s = np.repeat(sub_lat_centers[:, :, None, None], width, axis=2)
    lat_s = np.repeat(lat_s, n, axis=3).ravel()
    lon_s = np.broadcast_to(sub_lon_centers[None, None, :, :], (height, n, width, n)).ravel()
    area_s = np.repeat(sub_area[:, :, None, None], width, axis=2)
    area_s = np.repeat(area_s, n, axis=3).ravel()

    cell_idx = np.arange(height * width, dtype=np.int64).reshape(height, 1, width, 1)
    cell_s = np.broadcast_to(cell_idx, (height, n, width, n)).ravel()

    pix_s = hp.ang2pix(int(nside), lon_s, lat_s, nest=True, lonlat=True).astype(np.int64)

    npix = healpix_npix(nside)
    overlap = sparse.coo_matrix(
        (area_s, (pix_s, cell_s)), shape=(npix, height * width)
    ).tocsr()
    overlap.sum_duplicates()
    return overlap


def _row_normalize(matrix, what: str, nside: int, oversample: int):
    """Row-normalize, filling empty rows by nearest-neighbour so no row is zero."""
    from scipy import sparse

    row_sum = np.asarray(matrix.sum(axis=1)).ravel()
    empty = np.flatnonzero(row_sum <= 0.0)
    if empty.size:
        logger.warning(
            "%s: %d of %d rows had no overlap at oversample=%d; raise --oversample "
            "for a smoother operator (falling back to nearest-neighbour for them).",
            what, empty.size, row_sum.size, oversample,
        )
    normalized = matrix.multiply(
        sparse.csr_matrix(1.0 / np.where(row_sum > 0.0, row_sum, 1.0)[:, None])
    ).tocsr()
    return normalized, empty


def build_operators(
    height: int,
    width: int,
    nside: int,
    oversample: int = DEFAULT_OVERSAMPLE,
    lats: np.ndarray | None = None,
    lons: np.ndarray | None = None,
):
    """(W_fwd, W_bwd) as row-normalized CSR matrices."""
    from scipy import sparse

    hp = require_healpy()
    if lats is None or lons is None:
        lats, lons = latlon_axes(height, width)
    overlap = build_overlap_matrix(lats, lons, nside, oversample=oversample)

    w_fwd, empty_fwd = _row_normalize(overlap, "lat-lon -> HEALPix", nside, oversample)
    w_bwd, empty_bwd = _row_normalize(overlap.T.tocsr(), "HEALPix -> lat-lon", nside, oversample)

    # nearest-neighbour repair for any pixel/cell the sampling missed
    if empty_fwd.size:
        lon_p, lat_p = hp.pix2ang(int(nside), empty_fwd, nest=True, lonlat=True)
        lat_i = np.abs(np.asarray(lats)[:, None] - lat_p[None, :]).argmin(axis=0)
        lon_j = np.abs(((np.asarray(lons)[:, None] - lon_p[None, :] + 180.0) % 360.0) - 180.0).argmin(axis=0)
        repair = sparse.coo_matrix(
            (np.ones(empty_fwd.size), (empty_fwd, lat_i * int(width) + lon_j)),
            shape=w_fwd.shape,
        ).tocsr()
        w_fwd = (w_fwd + repair).tocsr()
    if empty_bwd.size:
        flat_lats = np.repeat(np.asarray(lats), int(width))
        flat_lons = np.tile(np.asarray(lons), int(height))
        pix = hp.ang2pix(int(nside), flat_lons[empty_bwd], flat_lats[empty_bwd], nest=True, lonlat=True)
        repair = sparse.coo_matrix(
            (np.ones(empty_bwd.size), (empty_bwd, pix.astype(np.int64))), shape=w_bwd.shape
        ).tocsr()
        w_bwd = (w_bwd + repair).tocsr()
    return w_fwd, w_bwd


# --------------------------------------------------------------------------- #
# operator cache
# --------------------------------------------------------------------------- #

def operator_cache_path(cache_dir: str | pathlib.Path, height: int, width: int,
                        nside: int, oversample: int) -> pathlib.Path:
    name = f"hpx{int(nside)}-nest_from_{int(height)}x{int(width)}_os{int(oversample)}.npz"
    return pathlib.Path(cache_dir) / CACHE_SUBDIR / name


def load_or_build_operators(
    height: int,
    width: int,
    nside: int,
    oversample: int = DEFAULT_OVERSAMPLE,
    cache_dir: str | pathlib.Path = "data/stats",
    lats: np.ndarray | None = None,
    lons: np.ndarray | None = None,
):
    """Cached (W_fwd, W_bwd). Building at 2p5/nside 32 takes a few seconds."""
    from scipy import sparse

    path = operator_cache_path(cache_dir, height, width, nside, oversample)
    if path.exists():
        with np.load(path, allow_pickle=False) as blob:
            w_fwd = sparse.csr_matrix(
                (blob["fwd_data"], blob["fwd_indices"], blob["fwd_indptr"]), shape=tuple(blob["fwd_shape"])
            )
            w_bwd = sparse.csr_matrix(
                (blob["bwd_data"], blob["bwd_indices"], blob["bwd_indptr"]), shape=tuple(blob["bwd_shape"])
            )
        return w_fwd, w_bwd

    w_fwd, w_bwd = build_operators(height, width, nside, oversample, lats=lats, lons=lons)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        fwd_data=w_fwd.data, fwd_indices=w_fwd.indices, fwd_indptr=w_fwd.indptr,
        fwd_shape=np.asarray(w_fwd.shape),
        bwd_data=w_bwd.data, bwd_indices=w_bwd.indices, bwd_indptr=w_bwd.indptr,
        bwd_shape=np.asarray(w_bwd.shape),
    )
    logger.info("Wrote regrid operators: %s", path)
    return w_fwd, w_bwd


# --------------------------------------------------------------------------- #
# application
# --------------------------------------------------------------------------- #

def latlon_to_healpix(field: np.ndarray, w_fwd) -> np.ndarray:
    """[..., H, W] -> [..., npix]."""
    field = np.asarray(field)
    npix, ncell = w_fwd.shape
    lead = field.shape[:-2]
    flat = field.reshape(-1, ncell).astype(np.float64, copy=False)
    out = (w_fwd @ flat.T).T
    return out.reshape(*lead, npix).astype(field.dtype, copy=False)


def healpix_to_latlon(field: np.ndarray, w_bwd, height: int, width: int) -> np.ndarray:
    """[..., npix] -> [..., H, W]."""
    field = np.asarray(field)
    ncell, npix = w_bwd.shape
    if int(height) * int(width) != ncell:
        raise ValueError(f"Operator expects {ncell} lat-lon cells, got {height}x{width}.")
    lead = field.shape[:-1]
    flat = field.reshape(-1, npix).astype(np.float64, copy=False)
    out = (w_bwd @ flat.T).T
    return out.reshape(*lead, int(height), int(width)).astype(field.dtype, copy=False)


# --------------------------------------------------------------------------- #
# bipartite edge form (in-model regridding)
# --------------------------------------------------------------------------- #

def sparse_to_bipartite_edges(matrix) -> tuple[np.ndarray, np.ndarray]:
    """Row-normalized sparse operator -> ``(edge_index [2, E], edge_weight [E])``.

    Row ``r`` of the operator is DESTINATION node ``r``; its stored columns are
    the source nodes and the stored values are that row's interpolation weights.
    ``mesh_layers.FixedBipartiteRemap`` computes
    ``out[dst] = sum_e weight[e] * x_src[src[e]]``, which is exactly the sparse
    matrix-vector product -- so a row-normalized matrix reproduces the offline
    regrid of ``latlon_to_healpix`` / ``healpix_to_latlon`` to float precision.

    ``edge_index`` is ``[src, dst]`` to match the repo's convention everywhere
    else (``edge_index[0]`` gathers, ``edge_index[1]`` scatters).
    """
    coo = matrix.tocoo()
    n_dst, n_src = matrix.shape
    if coo.nnz == 0:
        raise ValueError("Regrid operator has no non-zero entries.")
    covered = np.zeros(int(n_dst), dtype=bool)
    covered[coo.row] = True
    if not covered.all():
        raise ValueError(
            f"{int((~covered).sum())} of {int(n_dst)} destination nodes have no "
            "regrid weight; FixedBipartiteRemap would emit zeros there."
        )
    edge_index = np.stack(
        [coo.col.astype(np.int64), coo.row.astype(np.int64)],
        axis=0,
    )
    return edge_index, coo.data.astype(np.float32)
