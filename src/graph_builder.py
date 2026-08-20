from __future__ import annotations

import argparse
import glob
import hashlib
import math
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .architecture import resolve_graph_architecture, resolve_level_k_neighbors
from .resolution import (
    L4_72_36_24_18_9_2P5_LEVEL_SHAPES,
    RESOLUTION_SPECS,
    RATIO15_L4_2P5_LEVEL_SHAPES,
    cell_center_lat_lon,
    canonicalize_resolution_mode,
    default_graph_path,
    get_resolution_spec,
    graph_format_version_for_level_k,
)


GRAPH_FORMAT_VERSION = 2
GRAPH_FORMAT_VERSION_L3 = 3
# HEALPix bundles carry their own format version so a lat-lon cache and a HEALPix
# cache can never satisfy each other's validation, in either direction.
GRAPH_FORMAT_VERSION_HPX = "hpx_nest_v1"
PURE_SPHERICAL_KNN = "pure_spherical_knn"
HYBRID_ROW_AWARE_KNN = "hybrid_row_aware_knn"
# True HEALPix adjacency (8 slots, 24 pixels padded to degree 7 and masked).
# HEALPix grids only; rejected on lat-lon grids, which have no such adjacency.
HEALPIX_NATIVE = "healpix_native"
SUPPORTED_CONNECTIVITY_STRATEGIES = {PURE_SPHERICAL_KNN, HYBRID_ROW_AWARE_KNN, HEALPIX_NATIVE}


def regular_lat_lon(
    lat_count: int,
    lon_count: int,
    resolution: float,
    lat_start: Optional[float] = None,
    lon_start: float = -180.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if lat_start is None:
        lat_start = -90.0 + resolution / 2.0
    latitudes = lat_start + resolution * torch.arange(lat_count, dtype=torch.float32)
    longitudes = lon_start + resolution * torch.arange(lon_count, dtype=torch.float32)
    return latitudes, longitudes


def lat_lon_to_3d(lat_rad: torch.Tensor, lon_rad: torch.Tensor) -> torch.Tensor:
    x = torch.cos(lat_rad) * torch.cos(lon_rad)
    y = torch.cos(lat_rad) * torch.sin(lon_rad)
    z = torch.sin(lat_rad)
    return torch.stack([x, y, z], dim=-1)


def grid_nodes(latitudes_deg: torch.Tensor, longitudes_deg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    lat_grid, lon_grid = torch.meshgrid(latitudes_deg, longitudes_deg, indexing="ij")
    lat_lon = torch.stack([lat_grid.reshape(-1), lon_grid.reshape(-1)], dim=-1)
    lat_rad = torch.deg2rad(lat_lon[:, 0])
    lon_rad = torch.deg2rad(lat_lon[:, 1])
    coords = lat_lon_to_3d(lat_rad, lon_rad)
    return F.normalize(coords, dim=-1), torch.stack([lat_rad, lon_rad], dim=-1)


def _wrap_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def edge_features(lat_lon: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    src = edge_index[0]
    dst = edge_index[1]
    lat_i = lat_lon[dst, 0]
    lon_i = lat_lon[dst, 1]
    lat_j = lat_lon[src, 0]
    lon_j = lat_lon[src, 1]
    dlon = _wrap_pi(lon_j - lon_i)
    dlat = lat_j - lat_i

    cos_central = (
        torch.sin(lat_i) * torch.sin(lat_j)
        + torch.cos(lat_i) * torch.cos(lat_j) * torch.cos(dlon)
    ).clamp(-1.0, 1.0)
    distance = torch.acos(cos_central)

    y = torch.sin(dlon) * torch.cos(lat_j)
    x = torch.cos(lat_i) * torch.sin(lat_j) - torch.sin(lat_i) * torch.cos(lat_j) * torch.cos(dlon)
    bearing = torch.atan2(y, x)

    return torch.stack(
        [
            distance,
            torch.sin(bearing),
            torch.cos(bearing),
            dlat,
            torch.sin(dlon),
            torch.cos(dlon),
        ],
        dim=-1,
    ).to(torch.float32)


def _as_numpy_xyz(coords: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(coords, torch.Tensor):
        xyz = coords.detach().cpu().numpy()
    else:
        xyz = np.asarray(coords)
    xyz = np.asarray(xyz, dtype=np.float64)
    norm = np.linalg.norm(xyz, axis=1, keepdims=True)
    if np.any(norm <= 0.0):
        raise ValueError("All graph coordinates must have non-zero norm.")
    return xyz / norm


def coordinate_hash_from_lat_lon(lat_lon: torch.Tensor) -> str:
    arr = lat_lon.detach().cpu().numpy().astype(np.float64, copy=False)
    arr = np.ascontiguousarray(arr)
    digest = hashlib.sha256()
    digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    digest.update(arr.tobytes())
    return digest.hexdigest()


def _candidate_count(num_nodes: int, k: int) -> int:
    return min(num_nodes, max(2 * int(k) + 1, int(k) + 8))


def _sort_candidate_ids_by_distance(
    target_xyz: np.ndarray,
    xyz: np.ndarray,
    candidate_ids: np.ndarray,
    target_id: int,
) -> np.ndarray:
    ids = np.asarray(candidate_ids, dtype=np.int64).reshape(-1)
    ids = ids[(ids >= 0) & (ids < xyz.shape[0]) & (ids != int(target_id))]
    if ids.size == 0:
        return ids
    squared_distance = np.sum((xyz[ids] - target_xyz[None, :]) ** 2, axis=1)
    order = np.lexsort((ids, squared_distance))
    return ids[order]


def _query_target_candidates(
    tree: cKDTree,
    xyz: np.ndarray,
    target_id: int,
    candidate_count: int,
) -> np.ndarray:
    count = min(int(candidate_count), int(xyz.shape[0]))
    _, indices = tree.query(xyz[int(target_id)], k=count)
    return _sort_candidate_ids_by_distance(
        target_xyz=xyz[int(target_id)],
        xyz=xyz,
        candidate_ids=np.asarray(indices),
        target_id=int(target_id),
    )


def _query_all_candidates(tree: cKDTree, xyz: np.ndarray, k: int) -> list[np.ndarray]:
    num_nodes = int(xyz.shape[0])
    count = _candidate_count(num_nodes, int(k))
    _, indices = tree.query(xyz, k=count)
    indices = np.asarray(indices)
    if indices.ndim == 1:
        indices = indices[:, None]
    return [
        _sort_candidate_ids_by_distance(
            target_xyz=xyz[target_id],
            xyz=xyz,
            candidate_ids=indices[target_id],
            target_id=target_id,
        )
        for target_id in range(num_nodes)
    ]


def _global_target_candidates(xyz: np.ndarray, target_id: int) -> np.ndarray:
    ids = np.arange(xyz.shape[0], dtype=np.int64)
    return _sort_candidate_ids_by_distance(
        target_xyz=xyz[int(target_id)],
        xyz=xyz,
        candidate_ids=ids,
        target_id=int(target_id),
    )


def nearest_nodes_in_row(
    target_xyz: np.ndarray,
    candidate_xyz: np.ndarray,
    candidate_ids: np.ndarray,
    count: int,
) -> list[int]:
    if int(count) <= 0:
        return []
    ids = np.asarray(candidate_ids, dtype=np.int64)
    if int(count) > ids.size:
        raise ValueError(f"Requested {count} row neighbors from only {ids.size} candidates.")
    squared_distance = np.sum((candidate_xyz - target_xyz[None, :]) ** 2, axis=1)
    order = np.lexsort((ids, squared_distance))
    return [int(x) for x in ids[order[: int(count)]]]


def build_spherical_knn_neighbors(coords: torch.Tensor | np.ndarray, k: int) -> np.ndarray:
    xyz = _as_numpy_xyz(coords)
    num_nodes = int(xyz.shape[0])
    k = int(k)
    if k >= num_nodes:
        raise ValueError(f"k={k} must be smaller than num_nodes={num_nodes}")

    tree = cKDTree(xyz)
    base_candidates = _query_all_candidates(tree, xyz, k)
    neighbors = np.empty((num_nodes, k), dtype=np.int64)
    for target_id in range(num_nodes):
        chosen: list[int] = []
        seen = {int(target_id)}
        for source_id in base_candidates[target_id]:
            source = int(source_id)
            if source in seen:
                continue
            chosen.append(source)
            seen.add(source)
            if len(chosen) == k:
                break

        candidate_count = _candidate_count(num_nodes, k)
        while len(chosen) < k and candidate_count < num_nodes:
            candidate_count = min(num_nodes, max(candidate_count + 1, candidate_count * 2))
            for source_id in _query_target_candidates(tree, xyz, target_id, candidate_count):
                source = int(source_id)
                if source in seen:
                    continue
                chosen.append(source)
                seen.add(source)
                if len(chosen) == k:
                    break

        if len(chosen) < k:
            for source_id in _global_target_candidates(xyz, target_id):
                source = int(source_id)
                if source in seen:
                    continue
                chosen.append(source)
                seen.add(source)
                if len(chosen) == k:
                    break

        if len(chosen) != k:
            raise RuntimeError(f"Could not find exactly {k} neighbors for node {target_id}.")
        neighbors[target_id] = np.asarray(chosen, dtype=np.int64)
    return neighbors


def build_hybrid_row_aware_neighbors(
    xyz: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    height: int,
    width: int,
    k: int,
    interior_north_count: int = 1,
    interior_south_count: int = 1,
    polar_adjacent_count: int = 2,
) -> np.ndarray:
    """
    Return an integer array with shape [num_nodes, k].

    Each row contains exactly k unique source-neighbor node IDs
    for the corresponding target node.
    """
    del latitudes, longitudes  # Row membership is defined by node_id = row * width + column.
    xyz = _as_numpy_xyz(xyz)
    height = int(height)
    width = int(width)
    k = int(k)
    num_nodes = int(xyz.shape[0])
    if num_nodes != height * width:
        raise ValueError(f"Expected height * width = {height * width} nodes, got {num_nodes}.")
    if k >= num_nodes:
        raise ValueError(f"k={k} must be smaller than num_nodes={num_nodes}")
    if height < 2:
        raise ValueError("Hybrid row-aware kNN requires at least two latitude rows.")
    max_forced = max(
        int(interior_north_count) + int(interior_south_count),
        int(polar_adjacent_count),
    )
    if max_forced > k:
        raise ValueError(f"Forced row-aware neighbors ({max_forced}) cannot exceed k={k}.")

    tree = cKDTree(xyz)
    base_candidates = _query_all_candidates(tree, xyz, k)
    neighbors = np.empty((num_nodes, k), dtype=np.int64)

    for target_id in range(num_nodes):
        target_row, _ = divmod(int(target_id), width)
        forced_neighbors: list[int] = []
        if target_row == 0:
            row_ids = np.arange(width, 2 * width, dtype=np.int64)
            forced_neighbors.extend(
                nearest_nodes_in_row(xyz[target_id], xyz[row_ids], row_ids, int(polar_adjacent_count))
            )
        elif target_row == height - 1:
            row_ids = np.arange((height - 2) * width, (height - 1) * width, dtype=np.int64)
            forced_neighbors.extend(
                nearest_nodes_in_row(xyz[target_id], xyz[row_ids], row_ids, int(polar_adjacent_count))
            )
        else:
            north_ids = np.arange((target_row - 1) * width, target_row * width, dtype=np.int64)
            south_ids = np.arange((target_row + 1) * width, (target_row + 2) * width, dtype=np.int64)
            forced_neighbors.extend(
                nearest_nodes_in_row(xyz[target_id], xyz[north_ids], north_ids, int(interior_north_count))
            )
            forced_neighbors.extend(
                nearest_nodes_in_row(xyz[target_id], xyz[south_ids], south_ids, int(interior_south_count))
            )

        chosen: list[int] = []
        seen = {int(target_id)}
        for source_id in forced_neighbors:
            source = int(source_id)
            if source in seen:
                continue
            chosen.append(source)
            seen.add(source)

        for source_id in base_candidates[target_id]:
            source = int(source_id)
            if source in seen:
                continue
            chosen.append(source)
            seen.add(source)
            if len(chosen) == k:
                break

        candidate_count = _candidate_count(num_nodes, k)
        while len(chosen) < k and candidate_count < num_nodes:
            candidate_count = min(num_nodes, max(candidate_count + 1, candidate_count * 2))
            for source_id in _query_target_candidates(tree, xyz, target_id, candidate_count):
                source = int(source_id)
                if source in seen:
                    continue
                chosen.append(source)
                seen.add(source)
                if len(chosen) == k:
                    break

        if len(chosen) < k:
            for source_id in _global_target_candidates(xyz, target_id):
                source = int(source_id)
                if source in seen:
                    continue
                chosen.append(source)
                seen.add(source)
                if len(chosen) == k:
                    break

        if len(chosen) != k:
            raise RuntimeError(f"Could not find exactly {k} neighbors for node {target_id}.")
        neighbors[target_id] = np.asarray(chosen, dtype=np.int64)

    return neighbors


def neighbors_to_edge_index(neighbors: np.ndarray) -> torch.Tensor:
    neighbors = np.asarray(neighbors, dtype=np.int64)
    if neighbors.ndim != 2:
        raise ValueError(f"Expected neighbors with shape [num_nodes, k], got {neighbors.shape}.")
    num_nodes, k = neighbors.shape
    targets = np.repeat(np.arange(num_nodes, dtype=np.int64), int(k))
    sources = neighbors.reshape(-1)
    return torch.as_tensor(np.stack([sources, targets], axis=0), dtype=torch.long)


def _connected_component_count(edge_index: torch.Tensor, num_nodes: int) -> int:
    src = edge_index[0].detach().cpu().numpy()
    dst = edge_index[1].detach().cpu().numpy()
    adjacency = coo_matrix(
        (np.ones(src.shape[0], dtype=np.uint8), (src, dst)),
        shape=(int(num_nodes), int(num_nodes)),
    )
    num_components, _ = connected_components(adjacency, directed=False)
    return int(num_components)


def neighbor_diagnostics(
    neighbors: np.ndarray,
    edge_index: torch.Tensor,
    height: int,
    width: int,
    interior_north_count: int = 1,
    interior_south_count: int = 1,
    polar_adjacent_count: int = 2,
    enforce_row_aware: bool = False,
) -> dict[str, Any]:
    neighbors = np.asarray(neighbors, dtype=np.int64)
    num_nodes, k = neighbors.shape
    height = int(height)
    width = int(width)
    target_ids = np.arange(num_nodes, dtype=np.int64)
    # A MESH level has no grid shape and passes (0, 0). The row statistics below
    # divide the node index by `width` to recover a latitude row, which is both a
    # divide-by-zero and meaningless there -- a mesh node is not in any row. Treat
    # every node as its own row so the counts come out as "no row structure"
    # (same_row = k, cross_row = 0) instead of warning and reporting garbage.
    has_rows = width > 0
    row_divisor = width if has_rows else 1
    target_rows = (target_ids // row_divisor)[:, None]
    neighbor_rows = neighbors // row_divisor
    if not has_rows:
        # Collapse to a single row: no neighbour can be "north" or "south" of another.
        target_rows = np.zeros_like(target_rows)
        neighbor_rows = np.zeros_like(neighbor_rows)

    unique_counts = np.asarray([len(set(row.tolist())) for row in neighbors], dtype=np.int64)
    duplicate_neighbors = int(np.sum(k - unique_counts))
    self_loops = int(np.sum(neighbors == target_ids[:, None]))
    same_row = np.sum(neighbor_rows == target_rows, axis=1)
    north_row = np.sum(neighbor_rows == target_rows - 1, axis=1)
    south_row = np.sum(neighbor_rows == target_rows + 1, axis=1)
    cross_row = k - same_row
    other_row = k - same_row - north_row - south_row
    components = _connected_component_count(edge_index, num_nodes)

    if has_rows:
        interior_mask = (target_ids // width > 0) & (target_ids // width < height - 1)
        north_edge_adjacent = np.sum(neighbor_rows[:width] == 1, axis=1) if height > 1 else np.asarray([], dtype=np.int64)
        south_start = (height - 1) * width
        south_edge_adjacent = (
            np.sum(neighbor_rows[south_start:] == height - 2, axis=1) if height > 1 else np.asarray([], dtype=np.int64)
        )
    else:
        interior_mask = np.zeros(num_nodes, dtype=bool)
        north_edge_adjacent = np.asarray([], dtype=np.int64)
        south_edge_adjacent = np.asarray([], dtype=np.int64)

    diagnostics: dict[str, Any] = {
        "nodes": int(num_nodes),
        "edges": int(edge_index.shape[1]),
        "connected_components": int(components),
        "minimum_unique_neighbors": int(unique_counts.min()),
        "maximum_unique_neighbors": int(unique_counts.max()),
        "self_loops": int(self_loops),
        "duplicate_neighbors": int(duplicate_neighbors),
        "min_same_row_neighbors": int(same_row.min()),
        "max_same_row_neighbors": int(same_row.max()),
        "mean_same_row_neighbors": float(np.mean(same_row)),
        "min_cross_row_neighbors": int(cross_row.min()),
        "max_cross_row_neighbors": int(cross_row.max()),
        "mean_cross_row_neighbors": float(np.mean(cross_row)),
        "min_north_row_neighbors": int(north_row.min()),
        "min_south_row_neighbors": int(south_row.min()),
        "mean_other_row_neighbors": float(np.mean(other_row)),
        "north_edge_min_adjacent_neighbors": int(north_edge_adjacent.min()) if north_edge_adjacent.size else 0,
        "south_edge_min_adjacent_neighbors": int(south_edge_adjacent.min()) if south_edge_adjacent.size else 0,
        "interior_min_north_row_neighbors": int(north_row[interior_mask].min()) if np.any(interior_mask) else 0,
        "interior_min_south_row_neighbors": int(south_row[interior_mask].min()) if np.any(interior_mask) else 0,
    }

    if self_loops != 0:
        raise AssertionError(f"Graph contains {self_loops} self loops.")
    if duplicate_neighbors != 0:
        raise AssertionError(f"Graph contains {duplicate_neighbors} duplicate neighbor entries.")
    if int(unique_counts.min()) != k or int(unique_counts.max()) != k:
        raise AssertionError("Every node must have exactly k unique neighbors.")

    if enforce_row_aware:
        if diagnostics["interior_min_north_row_neighbors"] < int(interior_north_count):
            raise AssertionError("Interior rows are missing required row - 1 neighbors.")
        if diagnostics["interior_min_south_row_neighbors"] < int(interior_south_count):
            raise AssertionError("Interior rows are missing required row + 1 neighbors.")
        if diagnostics["north_edge_min_adjacent_neighbors"] < int(polar_adjacent_count):
            raise AssertionError("First latitude row is missing required adjacent-row neighbors.")
        if diagnostics["south_edge_min_adjacent_neighbors"] < int(polar_adjacent_count):
            raise AssertionError("Final latitude row is missing required adjacent-row neighbors.")
        if diagnostics["connected_components"] != 1:
            raise AssertionError(f"Expected one connected component, got {diagnostics['connected_components']}.")

    return diagnostics


def spherical_knn_graph(coords: torch.Tensor, lat_lon: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    neighbors = build_spherical_knn_neighbors(coords, k)
    edge_index = neighbors_to_edge_index(neighbors)
    return edge_index, edge_features(lat_lon, edge_index)


def make_pool_map(fine_height: int, fine_width: int) -> tuple[torch.Tensor, int, int]:
    coarse_height = math.ceil(fine_height / 2)
    coarse_width = math.ceil(fine_width / 2)
    return make_parent_index_pool_map(fine_height, fine_width, coarse_height, coarse_width)


def make_parent_index_pool_map(
    fine_height: int,
    fine_width: int,
    coarse_height: int,
    coarse_width: int,
) -> tuple[torch.Tensor, int, int]:
    fine_height = int(fine_height)
    fine_width = int(fine_width)
    coarse_height = int(coarse_height)
    coarse_width = int(coarse_width)
    if min(fine_height, fine_width, coarse_height, coarse_width) <= 0:
        raise ValueError(
            f"Grid shapes must be positive, got fine=({fine_height}, {fine_width}) "
            f"coarse=({coarse_height}, {coarse_width})."
        )
    fine_i = torch.arange(fine_height).repeat_interleave(fine_width)
    fine_j = torch.arange(fine_width).repeat(fine_height)
    parent_i = torch.div(fine_i * coarse_height, fine_height, rounding_mode="floor").clamp_max(coarse_height - 1)
    parent_j = torch.div(fine_j * coarse_width, fine_width, rounding_mode="floor").clamp_max(coarse_width - 1)
    pool_map = parent_i * coarse_width + parent_j
    validate_pool_map(pool_map, fine_height * fine_width, coarse_height * coarse_width)
    return pool_map.to(torch.long), coarse_height, coarse_width


def validate_pool_map(pool_map: torch.Tensor, num_fine: int, num_coarse: int) -> torch.Tensor:
    pool_map = pool_map.to(torch.long).reshape(-1)
    num_fine = int(num_fine)
    num_coarse = int(num_coarse)
    if int(pool_map.numel()) != num_fine:
        raise ValueError(f"Expected pool map length {num_fine}, got {int(pool_map.numel())}.")
    if num_coarse <= 0:
        raise ValueError("num_coarse must be positive.")
    if int(pool_map.min().item()) < 0 or int(pool_map.max().item()) >= num_coarse:
        raise ValueError(
            f"Pool map parent IDs must be in [0, {num_coarse - 1}], "
            f"got min={int(pool_map.min().item())} max={int(pool_map.max().item())}."
        )
    counts = torch.bincount(pool_map, minlength=num_coarse)
    empty = torch.nonzero(counts == 0, as_tuple=False).reshape(-1)
    if int(empty.numel()) > 0:
        preview = empty[:10].tolist()
        raise ValueError(f"Pool map leaves {int(empty.numel())} coarse parents without children; first IDs: {preview}.")
    return counts.to(torch.long)


def coarsen_coords(
    fine_coords: torch.Tensor,
    pool_map: torch.Tensor,
    coarse_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    coarse = torch.zeros(coarse_nodes, 3, dtype=fine_coords.dtype)
    coarse.index_add_(0, pool_map, fine_coords)
    counts = torch.bincount(pool_map, minlength=coarse_nodes).to(coarse.dtype).clamp_min(1.0)
    coarse = F.normalize(coarse / counts[:, None], dim=-1)
    lat = torch.asin(coarse[:, 2].clamp(-1.0, 1.0))
    lon = torch.atan2(coarse[:, 1], coarse[:, 0])
    return coarse, torch.stack([lat, lon], dim=-1)


def graph_format_version_for_levels(
    num_graph_levels: int,
    level_k_neighbors: list[int] | tuple[int, ...] | None = None,
    k_neighbors: int = 8,
    hierarchy_type: str | None = None,
    use_l4_ratio15: bool = False,
) -> int | str:
    if level_k_neighbors is None:
        if bool(use_l4_ratio15) or str(hierarchy_type or "").strip().lower() == "ratio15_l4":
            return "ratio15_l4_v1"
        if str(hierarchy_type or "").strip().lower() == "l4_72_36_24_18_9":
            return "l4_72_36_24_18_9_v1"
        if int(num_graph_levels) == 4:
            return GRAPH_FORMAT_VERSION_L3
        if int(num_graph_levels) == 5:
            return 5
        return GRAPH_FORMAT_VERSION
    return graph_format_version_for_level_k(
        int(num_graph_levels),
        level_k_neighbors=level_k_neighbors,
        k_neighbors=int(k_neighbors),
        hierarchy_type=hierarchy_type,
        use_l4_ratio15=use_l4_ratio15,
    )


def _normalize_level_shapes(
    base_height: int,
    base_width: int,
    num_graph_levels: int,
    level_shapes: list[list[int]] | tuple[tuple[int, int], ...] | None = None,
    hierarchy_type: str | None = None,
    use_l4_ratio15: bool = False,
) -> list[list[int]] | None:
    hierarchy = str(hierarchy_type or "standard").strip().lower()
    if level_shapes is None and (bool(use_l4_ratio15) or hierarchy == "ratio15_l4"):
        if (int(base_height), int(base_width)) != RATIO15_L4_2P5_LEVEL_SHAPES[0]:
            raise ValueError(
                "hierarchy_type='ratio15_l4' requires an L0 2.5-degree grid "
                f"{RATIO15_L4_2P5_LEVEL_SHAPES[0]}, got {(int(base_height), int(base_width))}."
            )
        level_shapes = RATIO15_L4_2P5_LEVEL_SHAPES
    if level_shapes is None and hierarchy == "l4_72_36_24_18_9":
        if (int(base_height), int(base_width)) != L4_72_36_24_18_9_2P5_LEVEL_SHAPES[0]:
            raise ValueError(
                "hierarchy_type='l4_72_36_24_18_9' requires an L0 2.5-degree grid "
                f"{L4_72_36_24_18_9_2P5_LEVEL_SHAPES[0]}, got {(int(base_height), int(base_width))}."
            )
        level_shapes = L4_72_36_24_18_9_2P5_LEVEL_SHAPES
    if level_shapes is None:
        return None
    shapes = [[int(shape[0]), int(shape[1])] for shape in level_shapes]
    if len(shapes) != int(num_graph_levels):
        raise ValueError(f"level_shapes must have length num_graph_levels={int(num_graph_levels)}, got {len(shapes)}.")
    if shapes[0] != [int(base_height), int(base_width)]:
        raise ValueError(f"level_shapes[0]={shapes[0]} does not match L0 grid {[int(base_height), int(base_width)]}.")
    for idx, (height, width) in enumerate(shapes):
        if height <= 0 or width <= 0:
            raise ValueError(f"level_shapes[{idx}] must be positive, got {[height, width]}.")
        if idx > 0:
            prev_height, prev_width = shapes[idx - 1]
            if height > prev_height or width > prev_width:
                raise ValueError(
                    f"level_shapes[{idx}]={shapes[idx]} cannot be finer than previous level {shapes[idx - 1]}."
                )
    return shapes


def _regular_level_lat_lon(
    height: int,
    width: int,
    reference_longitudes_deg: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    height = int(height)
    width = int(width)
    dlat = 180.0 / float(height)
    dlon = 360.0 / float(width)
    lon_start = float(reference_longitudes_deg[0].detach().cpu().item()) if int(reference_longitudes_deg.numel()) else 0.0
    latitudes = torch.linspace(-90.0 + dlat / 2.0, 90.0 - dlat / 2.0, height, dtype=torch.float32)
    longitudes = lon_start + dlon * torch.arange(width, dtype=torch.float32)
    return latitudes, longitudes


def _coordinate_pyramid(
    latitudes_deg: torch.Tensor,
    longitudes_deg: torch.Tensor,
    num_graph_levels: int,
    level_shapes: list[list[int]] | tuple[tuple[int, int], ...] | None = None,
    hierarchy_type: str | None = None,
    use_l4_ratio15: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    num_graph_levels = int(num_graph_levels)
    if num_graph_levels not in {3, 4, 5}:
        raise ValueError(f"num_graph_levels must be 3, 4, or 5, got {num_graph_levels}.")
    height0 = int(latitudes_deg.numel())
    width0 = int(longitudes_deg.numel())
    shapes = _normalize_level_shapes(
        height0,
        width0,
        num_graph_levels,
        level_shapes=level_shapes,
        hierarchy_type=hierarchy_type,
        use_l4_ratio15=use_l4_ratio15,
    )
    coords0, latlon0 = grid_nodes(latitudes_deg, longitudes_deg)
    levels: list[dict[str, Any]] = [
        {
            "name": "L0",
            "height": height0,
            "width": width0,
            "coords": coords0,
            "lat_lon": latlon0,
        }
    ]
    pools: dict[str, torch.Tensor] = {}
    if shapes is not None:
        levels = []
        for idx, (height, width) in enumerate(shapes):
            if idx == 0:
                level_latitudes = latitudes_deg.to(torch.float32)
                level_longitudes = longitudes_deg.to(torch.float32)
                coords, latlon = coords0, latlon0
            else:
                level_latitudes, level_longitudes = _regular_level_lat_lon(height, width, longitudes_deg)
                coords, latlon = grid_nodes(level_latitudes, level_longitudes)
                fine_height, fine_width = shapes[idx - 1]
                pool_map, _, _ = make_parent_index_pool_map(fine_height, fine_width, height, width)
                pools[f"L{idx - 1}_to_L{idx}"] = pool_map.to(torch.long)
            levels.append(
                {
                    "name": f"L{idx}",
                    "height": int(height),
                    "width": int(width),
                    "coords": coords,
                    "lat_lon": latlon,
                }
            )
        return levels, pools

    for source_idx in range(num_graph_levels - 1):
        source = levels[-1]
        pool_map, coarse_height, coarse_width = make_pool_map(int(source["height"]), int(source["width"]))
        coords, latlon = coarsen_coords(source["coords"], pool_map, coarse_height * coarse_width)
        pools[f"L{source_idx}_to_L{source_idx + 1}"] = pool_map.to(torch.long)
        levels.append(
            {
                "name": f"L{source_idx + 1}",
                "height": int(coarse_height),
                "width": int(coarse_width),
                "coords": coords,
                "lat_lon": latlon,
            }
        )
    return levels, pools


def _require_healpy():
    from .resolution import require_healpy

    return require_healpy()


def healpix_level_lat_lon(nside: int) -> torch.Tensor:
    """HEALPix pixel centers as ``(npix, 2)`` of ``(lat, lon)`` in RADIANS.

    Matches the convention ``grid_nodes`` returns, so ``edge_features`` and
    ``build_spherical_knn_neighbors`` consume it unchanged.

    Note the argument order below: with ``lonlat=True`` healpy returns
    ``(lon, lat)`` in DEGREES -- the opposite order from this repo's ``(lat, lon)``
    and a silent source of transposed graphs if swapped.
    """
    hp = _require_healpy()
    from .resolution import healpix_npix

    npix = healpix_npix(nside)
    lon_deg, lat_deg = hp.pix2ang(int(nside), np.arange(npix), nest=True, lonlat=True)
    lat = torch.deg2rad(torch.as_tensor(np.asarray(lat_deg), dtype=torch.float32))
    lon = torch.deg2rad(torch.as_tensor(np.asarray(lon_deg), dtype=torch.float32))
    return torch.stack([lat, lon], dim=-1)


def healpix_native_neighbours(nside: int) -> tuple[np.ndarray, np.ndarray]:
    """True HEALPix adjacency as ``(npix, 8)`` sources plus a validity mask.

    Degrees, measured against healpy (which reports a missing slot as ``-1``):

      * ``nside >= 2``: every pixel has 8 geometric neighbours except exactly 24,
        which have 7 -- they sit at the corners where only three base faces meet.
      * ``nside == 1``: the 12 base pixels are a special case and ALL have degree
        6, so two of the eight slots are padding on every pixel. Only relevant as
        a coarsest U-Net level; it is handled, not rejected.

    Invalid slots are padded with a **self-edge** and marked invalid, exactly the
    way mesh_builder pads the icosphere's degree-5 vertices. The self index keeps
    every gather in range while ``LocalGraphAttention`` masks the slot to ``-inf``
    before softmax, so it contributes nothing to the attention weights or the
    value sum.

    This is what spherical kNN cannot express: kNN always returns exactly k nodes,
    so at the low-degree pixels it silently promotes a non-adjacent pixel to a
    neighbour rather than admitting the degree is lower.
    """
    hp = _require_healpy()
    from .resolution import healpix_npix

    npix = healpix_npix(nside)
    raw = np.asarray(hp.get_all_neighbours(int(nside), np.arange(npix), nest=True))
    if raw.shape != (8, npix):
        raise RuntimeError(f"Expected healpy neighbours of shape (8, {npix}), got {raw.shape}.")
    neighbours = raw.T.astype(np.int64).copy()          # (npix, 8), node-major
    mask = neighbours >= 0
    self_index = np.broadcast_to(np.arange(npix, dtype=np.int64)[:, None], neighbours.shape)
    neighbours = np.where(mask, neighbours, self_index)
    return neighbours, mask


def _healpix_make_level(
    nside: int,
    coords: torch.Tensor,
    lat_lon: torch.Tensor,
    k: int,
) -> dict[str, torch.Tensor | int]:
    """Level dict from native HEALPix adjacency, with ``edge_mask``.

    Deliberately does NOT go through ``make_level``: ``neighbor_diagnostics``
    asserts zero self-loops and exactly k unique neighbours per node, both of
    which the padded low-degree pixels violate by construction.
    """
    from .resolution import healpix_npix

    if int(k) != 8:
        raise ValueError(
            f"Native HEALPix adjacency has exactly 8 slots per pixel, so k must be 8, got {k}. "
            "Use graph_connectivity_strategy='pure_spherical_knn' for a different k."
        )
    npix = healpix_npix(nside)
    neighbours, mask = healpix_native_neighbours(nside)
    edge_index = neighbors_to_edge_index(neighbours)
    edge_attr = edge_features(lat_lon, edge_index)
    flat_mask = torch.as_tensor(mask.reshape(-1), dtype=torch.bool)
    edge_attr = edge_attr.clone()
    edge_attr[~flat_mask] = 0.0                        # inert (masked) slots
    degree = mask.sum(axis=1)
    return {
        "height": int(npix),
        "width": 1,
        "num_nodes": int(npix),
        "k": int(k),
        "coords": coords.to(torch.float32),
        "lat_lon": lat_lon.to(torch.float32),
        "edge_index": edge_index.to(torch.long),
        "edge_attr": edge_attr.to(torch.float32),
        "edge_mask": flat_mask,
        "neighbor_diagnostics": {
            "strategy": HEALPIX_NATIVE,
            "nside": int(nside),
            "num_nodes": int(npix),
            "k": int(k),
            "degree_min": int(degree.min()),
            "degree_max": int(degree.max()),
            # Full histogram rather than hardcoded 7/8 counts: nside 1 is uniform
            # degree 6, so a fixed pair of buckets would read as all-zero there.
            "degree_histogram": {
                int(value): int(count)
                for value, count in zip(*np.unique(degree, return_counts=True))
            },
            "masked_edge_slots": int((~mask).sum()),
        },
    }


def healpix_pool_map(npix_fine: int) -> torch.Tensor:
    """Fine-to-coarse parent index for a HEALPix quadtree.

    In NEST ordering the parent of pixel ``p`` at ``nside`` is exactly ``p >> 2``
    at ``nside // 2`` -- four children per parent, no exceptions and no boundary
    cases. That is the whole reason NEST is required: with RING ordering there is
    no such relation and the pool map would need a coordinate lookup.
    """
    npix_fine = int(npix_fine)
    if npix_fine % 4 != 0:
        raise ValueError(f"HEALPix npix must be divisible by 4 to pool, got {npix_fine}.")
    return torch.arange(npix_fine, dtype=torch.long) >> 2


def _healpix_pyramid(nside: int, num_graph_levels: int) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    """Pixel centers and pool maps per level. Mirrors ``_coordinate_pyramid``."""
    from .resolution import healpix_level_nsides, healpix_npix

    sides = healpix_level_nsides(int(nside), int(num_graph_levels))
    levels: list[dict[str, Any]] = []
    pools: dict[str, torch.Tensor] = {}
    for idx, side in enumerate(sides):
        lat_lon = healpix_level_lat_lon(side)
        coords = F.normalize(lat_lon_to_3d(lat_lon[:, 0], lat_lon[:, 1]), dim=-1)
        npix = healpix_npix(side)
        levels.append(
            {
                "name": f"L{idx}",
                # carried as a degenerate (npix, 1) lat-lon grid; see resolution.py
                "height": int(npix),
                "width": 1,
                "coords": coords,
                "lat_lon": lat_lon,
                "nside": int(side),
            }
        )
        if idx > 0:
            fine_npix = healpix_npix(sides[idx - 1])
            pool_map = healpix_pool_map(fine_npix)
            validate_pool_map(pool_map, fine_npix, npix)
            pools[f"L{idx - 1}_to_L{idx}"] = pool_map
    return levels, pools


def make_level(
    coords: torch.Tensor,
    lat_lon: torch.Tensor,
    height: int,
    width: int,
    k: int,
    connectivity_strategy: str = HYBRID_ROW_AWARE_KNN,
    row_aware_knn: dict[str, int] | None = None,
) -> dict[str, torch.Tensor | int]:
    row_cfg = normalize_row_aware_config(row_aware_knn)
    connectivity_strategy = normalize_connectivity_strategy(connectivity_strategy)
    if connectivity_strategy == PURE_SPHERICAL_KNN:
        neighbors = build_spherical_knn_neighbors(coords, k)
    elif connectivity_strategy == HYBRID_ROW_AWARE_KNN:
        neighbors = build_hybrid_row_aware_neighbors(
            xyz=_as_numpy_xyz(coords),
            latitudes=lat_lon[:, 0].detach().cpu().numpy(),
            longitudes=lat_lon[:, 1].detach().cpu().numpy(),
            height=height,
            width=width,
            k=k,
            interior_north_count=row_cfg["interior_neighbors_from_north_row"],
            interior_south_count=row_cfg["interior_neighbors_from_south_row"],
            polar_adjacent_count=row_cfg["polar_neighbors_from_adjacent_row"],
        )
    else:
        raise ValueError(f"Unsupported graph connectivity strategy: {connectivity_strategy}")

    edge_index = neighbors_to_edge_index(neighbors)
    diagnostics = neighbor_diagnostics(
        neighbors,
        edge_index,
        height=height,
        width=width,
        interior_north_count=row_cfg["interior_neighbors_from_north_row"],
        interior_south_count=row_cfg["interior_neighbors_from_south_row"],
        polar_adjacent_count=row_cfg["polar_neighbors_from_adjacent_row"],
        enforce_row_aware=(connectivity_strategy == HYBRID_ROW_AWARE_KNN),
    )
    edge_attr = edge_features(lat_lon, edge_index)
    return {
        "height": int(height),
        "width": int(width),
        "num_nodes": int(coords.shape[0]),
        "k": int(k),
        "coords": coords.to(torch.float32),
        "lat_lon": lat_lon.to(torch.float32),
        "edge_index": edge_index.to(torch.long),
        "edge_attr": edge_attr.to(torch.float32),
        "neighbor_diagnostics": diagnostics,
    }


def normalize_connectivity_strategy(strategy: str | None) -> str:
    value = str(strategy or HYBRID_ROW_AWARE_KNN).strip()
    if value not in SUPPORTED_CONNECTIVITY_STRATEGIES:
        available = ", ".join(sorted(SUPPORTED_CONNECTIVITY_STRATEGIES))
        raise ValueError(f"Unsupported graph_connectivity_strategy '{value}'. Available: {available}")
    return value


def normalize_row_aware_config(row_aware_knn: dict[str, Any] | None) -> dict[str, int]:
    raw = dict(row_aware_knn or {})
    return {
        "enabled": bool(raw.get("enabled", True)),
        "interior_neighbors_from_north_row": int(raw.get("interior_neighbors_from_north_row", 1)),
        "interior_neighbors_from_south_row": int(raw.get("interior_neighbors_from_south_row", 1)),
        "polar_neighbors_from_adjacent_row": int(raw.get("polar_neighbors_from_adjacent_row", 2)),
    }


def _coordinate_metadata(
    latitudes_deg: torch.Tensor,
    longitudes_deg: torch.Tensor,
    k: int,
    resolution: float,
    connectivity_strategy: str,
    row_aware_knn: dict[str, Any] | None,
    resolution_mode: str | None = None,
    num_graph_levels: int = 3,
    level_k_neighbors: list[int] | tuple[int, ...] | None = None,
    level_shapes: list[list[int]] | tuple[tuple[int, int], ...] | None = None,
    hierarchy_type: str | None = None,
    use_l4_ratio15: bool = False,
    pyramid_override: list[dict[str, Any]] | None = None,
    pools_override: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    height0 = int(latitudes_deg.numel())
    width0 = int(longitudes_deg.numel())
    num_graph_levels = int(num_graph_levels)
    hierarchy = str(hierarchy_type or "standard").strip().lower()
    if bool(use_l4_ratio15):
        hierarchy = "ratio15_l4"
    if pyramid_override is not None:
        # HEALPix: the caller already built the quadtree pyramid, and rebuilding it
        # through _coordinate_pyramid would take the lat-lon branch and produce the
        # wrong level shapes and coordinate hashes.
        pyramid = pyramid_override
        pools = pools_override if pools_override is not None else {}
    else:
        pyramid, pools = _coordinate_pyramid(
            latitudes_deg,
            longitudes_deg,
            num_graph_levels=num_graph_levels,
            level_shapes=level_shapes,
            hierarchy_type=hierarchy,
            use_l4_ratio15=use_l4_ratio15,
        )
    strategy = normalize_connectivity_strategy(connectivity_strategy)
    row_cfg = normalize_row_aware_config(row_aware_knn)
    level_k_values = list(
        resolve_level_k_neighbors(
            {
                "k_neighbors": int(k),
                **({"level_k_neighbors": list(level_k_neighbors)} if level_k_neighbors is not None else {}),
                "num_graph_levels": int(num_graph_levels),
            },
            num_graph_levels=int(num_graph_levels),
            k_neighbors=int(k),
        )
    )
    levels = {
        str(level["name"]): [int(level["height"]), int(level["width"])]
        for level in pyramid
    }
    level_shapes = [[int(level["height"]), int(level["width"])] for level in pyramid]
    node_counts = [int(level["height"]) * int(level["width"]) for level in pyramid]
    edge_counts = [int(count) * int(level_k) for count, level_k in zip(node_counts, level_k_values)]
    mode = resolution_mode
    if mode is None:
        for spec in RESOLUTION_SPECS.values():
            if (height0, width0) == spec.grid_shape and abs(float(resolution) - spec.resolution_degrees) < 1.0e-6:
                mode = spec.name
                break
    mode = canonicalize_resolution_mode(mode) if mode else None
    level_hashes = {
        str(level["name"]): coordinate_hash_from_lat_lon(level["lat_lon"])
        for level in pyramid
    }
    return {
        "graph_format_version": graph_format_version_for_levels(
            num_graph_levels,
            level_k_neighbors=level_k_values,
            k_neighbors=int(k),
            hierarchy_type=hierarchy,
            use_l4_ratio15=use_l4_ratio15,
        ),
        "hierarchy_type": hierarchy,
        "num_graph_levels": int(num_graph_levels),
        "use_l3": bool(num_graph_levels >= 4),
        "use_l4": bool(num_graph_levels >= 5),
        "use_l4_ratio15": bool(hierarchy == "ratio15_l4" or use_l4_ratio15),
        "resolution_mode": mode,
        "resolution": float(resolution),
        "resolution_degrees": float(resolution),
        "grid_shape": [height0, width0],
        "level_shapes": level_shapes,
        "node_counts": node_counts,
        "edge_counts": edge_counts,
        "lat_count": height0,
        "lon_count": width0,
        "k": int(k),
        "graph_k": int(k),
        "k_neighbors": int(k),
        "level_k_neighbors": level_k_values,
        "connectivity_strategy": strategy,
        "graph_connectivity_strategy": strategy,
        "row_aware_knn": row_cfg,
        "interior_north_count": int(row_cfg["interior_neighbors_from_north_row"]),
        "interior_south_count": int(row_cfg["interior_neighbors_from_south_row"]),
        "polar_adjacent_count": int(row_cfg["polar_neighbors_from_adjacent_row"]),
        "node_order": "lat_index * num_lon + lon_index",
        "levels": levels,
        "pool_maps": [f"L{idx}_to_L{idx + 1}" for idx in range(num_graph_levels - 1)],
        "pooling_map_strategy": "proportional_parent_index",
        "pool_child_counts": {
            name: validate_pool_map(
                pool_map,
                int(pyramid[idx]["height"]) * int(pyramid[idx]["width"]),
                int(pyramid[idx + 1]["height"]) * int(pyramid[idx + 1]["width"]),
            ).tolist()
            for idx, (name, pool_map) in enumerate(pools.items())
        },
        "pool_child_count_stats": {
            name: {
                "min": int(counts.min().item()),
                "max": int(counts.max().item()),
                "mean": float(counts.to(torch.float32).mean().item()),
            }
            for name, counts in (
                (
                    name,
                    validate_pool_map(
                        pool_map,
                        int(pyramid[idx]["height"]) * int(pyramid[idx]["width"]),
                        int(pyramid[idx + 1]["height"]) * int(pyramid[idx + 1]["width"]),
                    ),
                )
                for idx, (name, pool_map) in enumerate(pools.items())
            )
        },
        "connectivity_strategy_name": strategy,
        "graph_coordinate_hash": level_hashes["L0"],
        "coordinate_hash": level_hashes["L0"],
        "level_coordinate_hashes": level_hashes,
    }


def expected_graph_metadata(
    latitudes_deg: torch.Tensor,
    longitudes_deg: torch.Tensor,
    k: int = 8,
    resolution: float = 5.625,
    connectivity_strategy: str = HYBRID_ROW_AWARE_KNN,
    row_aware_knn: dict[str, Any] | None = None,
    resolution_mode: str | None = None,
    num_graph_levels: int = 3,
    level_k_neighbors: list[int] | tuple[int, ...] | None = None,
    level_shapes: list[list[int]] | tuple[tuple[int, int], ...] | None = None,
    hierarchy_type: str | None = None,
    use_l4_ratio15: bool = False,
) -> dict[str, Any]:
    return _coordinate_metadata(
        latitudes_deg=latitudes_deg,
        longitudes_deg=longitudes_deg,
        k=k,
        resolution=resolution,
        connectivity_strategy=connectivity_strategy,
        row_aware_knn=row_aware_knn,
        resolution_mode=resolution_mode,
        num_graph_levels=num_graph_levels,
        level_k_neighbors=level_k_neighbors,
        level_shapes=level_shapes,
        hierarchy_type=hierarchy_type,
        use_l4_ratio15=use_l4_ratio15,
    )


def graph_topology_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    graph_mode = str(metadata.get("graph_mode", "grid"))
    num_graph_levels = int(metadata.get("num_graph_levels", len(metadata.get("levels", {}) or {}) or 3))
    use_l3 = bool(metadata.get("use_l3", num_graph_levels >= 4))
    use_l4 = bool(metadata.get("use_l4", num_graph_levels >= 5))
    graph_k = int(metadata.get("graph_k", metadata.get("k_neighbors", metadata.get("k", 0))))
    raw_level_k = metadata.get("level_k_neighbors", None)
    if raw_level_k is None and graph_k > 0:
        raw_level_k = [graph_k for _ in range(num_graph_levels)]
    level_k = [int(x) for x in raw_level_k] if raw_level_k is not None else None
    topology = {
        "resolution_mode": metadata.get("resolution_mode"),
        "hierarchy_type": metadata.get("hierarchy_type", "standard"),
        "graph_connectivity_strategy": metadata.get("graph_connectivity_strategy", metadata.get("connectivity_strategy")),
        "graph_format_version": metadata.get("graph_format_version", 0),
        "graph_k": graph_k,
        "level_k_neighbors": level_k,
        "edge_counts": metadata.get("edge_counts"),
        "graph_coordinate_hash": metadata.get("graph_coordinate_hash", metadata.get("coordinate_hash")),
        "num_graph_levels": num_graph_levels,
        "use_l3": use_l3,
        "use_l4": use_l4,
        "use_l4_ratio15": bool(metadata.get("use_l4_ratio15", False)),
    }
    if graph_mode == "mesh":
        # ``hierarchy_type`` in checkpoint architecture metadata describes the
        # processor layout ("standard").  In a mesh cache it describes physical
        # connectivity ("icosphere").  These are intentionally different concepts;
        # canonicalize the cache-side value so an existing mesh checkpoint is not
        # rejected merely because architecture metadata was written afterward.
        topology["hierarchy_type"] = "icosphere"
        topology.update(
            {
                "graph_mode": "mesh",
                "mesh_format_version": metadata.get("mesh_format_version"),
                "refinement": metadata.get("refinement"),
                "grid_shape": metadata.get("grid_shape"),
                "grid_coordinate_hash": metadata.get("grid_coordinate_hash"),
                "bipartite_mapping_type": metadata.get(
                    "bipartite_mapping_type",
                    "graphcast_radius",
                ),
                "g2m_radius_factor": (
                    metadata.get("g2m_radius_factor")
                    if metadata.get(
                        "bipartite_mapping_type",
                        "graphcast_radius",
                    )
                    == "graphcast_radius"
                    else None
                ),
                "coarse_level_connectivity": metadata.get("coarse_level_connectivity"),
                "grid_attention_graph": metadata.get("grid_attention_graph", False),
                "grid_attention_k_neighbors": metadata.get("grid_attention_k_neighbors"),
                "grid_attention_connectivity_strategy": metadata.get(
                    "grid_attention_connectivity_strategy"
                ),
                "grid_attention_reference_graph": metadata.get(
                    "grid_attention_reference_graph"
                ),
                "grid_attention_reference_sha256": metadata.get(
                    "grid_attention_reference_sha256"
                ),
            }
        )
    return topology


def validate_graph_cache_metadata(bundle: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    actual = dict(bundle.get("metadata", {}))
    mismatches: list[str] = []
    actual_num_levels = int(actual.get("num_graph_levels", len(actual.get("levels", {}) or {}) or 3))
    expected_num_levels = int(expected.get("num_graph_levels", len(expected.get("levels", {}) or {}) or 3))
    actual_use_l3 = bool(actual.get("use_l3", actual_num_levels >= 4))
    expected_use_l3 = bool(expected.get("use_l3", expected_num_levels >= 4))
    actual_use_l4 = bool(actual.get("use_l4", actual_num_levels >= 5))
    expected_use_l4 = bool(expected.get("use_l4", expected_num_levels >= 5))
    comparisons = [
        ("graph_format_version", "graph_format_version"),
        ("hierarchy_type", "hierarchy_type"),
        ("use_l4_ratio15", "use_l4_ratio15"),
        ("resolution_mode", "resolution_mode"),
        ("connectivity_strategy", "connectivity_strategy"),
        ("pooling_map_strategy", "pooling_map_strategy"),
        ("k", "k"),
        ("lat_count", "lat_count"),
        ("lon_count", "lon_count"),
        ("graph_coordinate_hash", "graph_coordinate_hash"),
    ]
    for actual_key, expected_key in comparisons:
        if actual.get(actual_key) != expected.get(expected_key):
            mismatches.append(
                f"{actual_key}: cache={actual.get(actual_key)!r}, expected={expected.get(expected_key)!r}"
            )
    if actual_num_levels != expected_num_levels:
        mismatches.append(f"num_graph_levels: cache={actual_num_levels!r}, expected={expected_num_levels!r}")
    if actual_use_l3 != expected_use_l3:
        mismatches.append(f"use_l3: cache={actual_use_l3!r}, expected={expected_use_l3!r}")
    if actual_use_l4 != expected_use_l4:
        mismatches.append(f"use_l4: cache={actual_use_l4!r}, expected={expected_use_l4!r}")
    if actual.get("levels") != expected.get("levels"):
        mismatches.append(f"levels: cache={actual.get('levels')!r}, expected={expected.get('levels')!r}")
    actual_graph_k = int(actual.get("graph_k", actual.get("k", 0)))
    expected_graph_k = int(expected.get("graph_k", expected.get("k", 0)))
    if actual_graph_k != expected_graph_k:
        mismatches.append(f"graph_k: cache={actual_graph_k!r}, expected={expected_graph_k!r}")
    actual_level_k = actual.get("level_k_neighbors", None)
    if actual_level_k is None and actual_graph_k > 0:
        actual_level_k = [actual_graph_k for _ in range(actual_num_levels)]
    expected_level_k = expected.get("level_k_neighbors", None)
    if expected_level_k is None and expected_graph_k > 0:
        expected_level_k = [expected_graph_k for _ in range(expected_num_levels)]
    if [int(x) for x in actual_level_k or []] != [int(x) for x in expected_level_k or []]:
        mismatches.append(f"level_k_neighbors: cache={actual_level_k!r}, expected={expected_level_k!r}")
    for key in ("grid_shape", "level_shapes", "node_counts", "edge_counts"):
        if actual.get(key) != expected.get(key):
            mismatches.append(f"{key}: cache={actual.get(key)!r}, expected={expected.get(key)!r}")
    if actual.get("level_coordinate_hashes") != expected.get("level_coordinate_hashes"):
        mismatches.append("level_coordinate_hashes differ")
    if expected_num_levels >= 4 and actual.get("pool_maps") != expected.get("pool_maps"):
        mismatches.append(f"pool_maps: cache={actual.get('pool_maps')!r}, expected={expected.get('pool_maps')!r}")
    if actual.get("connectivity_strategy") == HYBRID_ROW_AWARE_KNN:
        actual_row = normalize_row_aware_config(actual.get("row_aware_knn", {}))
        expected_row = normalize_row_aware_config(expected.get("row_aware_knn", {}))
        if actual_row != expected_row:
            mismatches.append(f"row_aware_knn: cache={actual_row!r}, expected={expected_row!r}")
    return mismatches


def load_raw_graph_bundle(path: str, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_graph_bundle(
    latitudes_deg: torch.Tensor,
    longitudes_deg: torch.Tensor,
    k: int = 8,
    resolution: float = 5.625,
    connectivity_strategy: str = HYBRID_ROW_AWARE_KNN,
    row_aware_knn: dict[str, Any] | None = None,
    resolution_mode: str | None = None,
    num_graph_levels: int = 3,
    level_k_neighbors: list[int] | tuple[int, ...] | None = None,
    level_shapes: list[list[int]] | tuple[tuple[int, int], ...] | None = None,
    hierarchy_type: str | None = None,
    use_l4_ratio15: bool = False,
) -> dict[str, object]:
    num_graph_levels = int(num_graph_levels)
    hierarchy = str(hierarchy_type or "standard").strip().lower()
    if bool(use_l4_ratio15):
        hierarchy = "ratio15_l4"
    strategy = normalize_connectivity_strategy(connectivity_strategy)

    from .resolution import grid_kind as _grid_kind

    kind = _grid_kind(resolution_mode) if resolution_mode is not None else "latlon"
    if kind == "healpix":
        from .resolution import healpix_nside_for_mode

        if strategy not in {HEALPIX_NATIVE, PURE_SPHERICAL_KNN}:
            raise ValueError(
                f"resolution_mode='{resolution_mode}' is a HEALPix grid and requires "
                f"graph_connectivity_strategy '{HEALPIX_NATIVE}' (recommended: true "
                f"adjacency, degree-7 pixels masked) or '{PURE_SPHERICAL_KNN}', got "
                f"'{strategy}'. Row-aware kNN is meaningless on a (npix, 1) grid: "
                "every row is one pixel."
            )
        if hierarchy != "standard":
            raise ValueError(
                f"HEALPix grids support hierarchy_type='standard' only, got '{hierarchy}'."
            )
        nside = healpix_nside_for_mode(resolution_mode)
        pyramid, pools = _healpix_pyramid(nside, num_graph_levels)
    else:
        if strategy == HEALPIX_NATIVE:
            raise ValueError(
                f"graph_connectivity_strategy='{HEALPIX_NATIVE}' requires a HEALPix "
                "resolution_mode (e.g. hpx32); a lat-lon grid has no HEALPix adjacency."
            )
        nside = None
        pyramid, pools = _coordinate_pyramid(
            latitudes_deg,
            longitudes_deg,
            num_graph_levels=num_graph_levels,
            level_shapes=level_shapes,
            hierarchy_type=hierarchy,
            use_l4_ratio15=use_l4_ratio15,
        )
    row_cfg = normalize_row_aware_config(row_aware_knn)
    level_k_values = list(
        resolve_level_k_neighbors(
            {
                "k_neighbors": int(k),
                **({"level_k_neighbors": list(level_k_neighbors)} if level_k_neighbors is not None else {}),
                "num_graph_levels": num_graph_levels,
            },
            num_graph_levels=num_graph_levels,
            k_neighbors=int(k),
        )
    )
    if strategy == HEALPIX_NATIVE:
        # Per-level dispatch. Native adjacency has exactly 8 slots, so a level that
        # asks for a different k (e.g. the dense coarsest level, k=24) falls back to
        # spherical kNN for that level only. Levels at k=8 keep true adjacency and
        # the degree-7 masking; deliberately densified levels keep their reach.
        levels = {}
        level_strategies: dict[str, str] = {}
        for idx, level in enumerate(pyramid):
            name = str(level["name"])
            k_level = int(level_k_values[idx])
            if k_level == 8:
                levels[name] = _healpix_make_level(
                    int(level["nside"]), level["coords"], level["lat_lon"], k_level
                )
                level_strategies[name] = HEALPIX_NATIVE
            else:
                levels[name] = make_level(
                    level["coords"],
                    level["lat_lon"],
                    int(level["height"]),
                    int(level["width"]),
                    k_level,
                    PURE_SPHERICAL_KNN,
                    row_cfg,
                )
                level_strategies[name] = PURE_SPHERICAL_KNN
    else:
        level_strategies = {}
        levels = {
            str(level["name"]): make_level(
                level["coords"],
                level["lat_lon"],
                int(level["height"]),
                int(level["width"]),
                int(level_k_values[idx]),
                strategy,
                row_cfg,
            )
            for idx, level in enumerate(pyramid)
        }
    metadata = _coordinate_metadata(
        latitudes_deg=latitudes_deg,
        longitudes_deg=longitudes_deg,
        k=k,
        resolution=resolution,
        connectivity_strategy=strategy,
        row_aware_knn=row_cfg,
        resolution_mode=resolution_mode,
        num_graph_levels=num_graph_levels,
        level_k_neighbors=level_k_values,
        level_shapes=level_shapes,
        hierarchy_type=hierarchy,
        use_l4_ratio15=use_l4_ratio15,
        pyramid_override=pyramid if kind == "healpix" else None,
        pools_override=pools if kind == "healpix" else None,
    )
    metadata["diagnostics"] = {
        name: dict(level["neighbor_diagnostics"])
        for name, level in levels.items()
    }
    if kind == "healpix":
        # Distinct cache identity, so a lat-lon bundle and a HEALPix bundle can
        # never be loaded in place of one another: the format version differs and
        # the per-level coordinate hashes differ.
        metadata["grid_kind"] = "healpix"
        metadata["nside"] = int(nside)
        metadata["ordering"] = "nest"
        metadata["pooling_map_strategy"] = "nest_quadtree"
        metadata["level_nsides"] = [int(level["nside"]) for level in pyramid]
        metadata["level_connectivity_strategies"] = dict(level_strategies)
        metadata["graph_format_version"] = GRAPH_FORMAT_VERSION_HPX
    # NOTE: lat-lon bundles deliberately get NO "grid_kind" key. Adding one would
    # change the metadata of every existing cache, so validate_graph_cache_metadata
    # would report a mismatch and silently rebuild every lat-lon graph on disk.
    # Absent therefore means lat-lon: read it as metadata.get("grid_kind", "latlon").

    return {
        "metadata": metadata,
        "levels": levels,
        "pool": pools,
    }


def lat_lon_from_netcdf(path_or_dir: str) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        import netCDF4 as nc
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("netCDF4 is required when building a graph from NetCDF coordinates.") from exc

    path = path_or_dir
    if os.path.isdir(path_or_dir):
        matches = sorted(glob.glob(os.path.join(path_or_dir, "*.nc")))
        if not matches:
            raise FileNotFoundError(f"No .nc files found under {path_or_dir}")
        path = matches[0]

    with nc.Dataset(path, "r") as ds:
        lat_key = "latitude" if "latitude" in ds.variables else "lat"
        lon_key = "longitude" if "longitude" in ds.variables else "lon"
        latitudes = torch.as_tensor(ds[lat_key][:], dtype=torch.float32)
        longitudes = torch.as_tensor(ds[lon_key][:], dtype=torch.float32)
    return latitudes, longitudes


def save_graph(bundle: dict[str, object], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output_path)


def graph_path_for_resolution(
    resolution: float,
    k: int,
    connectivity_strategy: str,
    graph_format_version: int = GRAPH_FORMAT_VERSION,
    resolution_mode: str | None = None,
    num_graph_levels: int = 3,
    level_k_neighbors: list[int] | tuple[int, ...] | None = None,
    hierarchy_type: str | None = None,
    use_l4_ratio15: bool = False,
) -> str:
    graph_format_version = graph_format_version_for_levels(
        num_graph_levels,
        level_k_neighbors=level_k_neighbors,
        k_neighbors=int(k),
        hierarchy_type=hierarchy_type,
        use_l4_ratio15=use_l4_ratio15,
    )
    if resolution_mode is None:
        for spec in RESOLUTION_SPECS.values():
            if abs(float(resolution) - spec.resolution_degrees) < 1.0e-8:
                resolution_mode = spec.name
                break
    if resolution_mode is not None:
        return default_graph_path(
            resolution_mode,
            k=int(k),
            strategy=normalize_connectivity_strategy(connectivity_strategy),
            graph_format_version=graph_format_version,
            num_graph_levels=int(num_graph_levels),
            level_k_neighbors=level_k_neighbors,
            hierarchy_type=hierarchy_type,
            use_l4_ratio15=use_l4_ratio15,
        )
    resolution_stem = str(float(resolution)).replace(".", "p")
    strategy_stem = normalize_connectivity_strategy(connectivity_strategy)
    if strategy_stem == HYBRID_ROW_AWARE_KNN:
        strategy_stem = "hybrid_row_aware"
    hierarchy = str(hierarchy_type or "standard").strip().lower()
    values = [int(x) for x in level_k_neighbors] if level_k_neighbors is not None else [int(k)] * int(num_graph_levels)
    if bool(use_l4_ratio15) or hierarchy == "ratio15_l4":
        joined = "_".join(str(value) for value in values)
        return f"graphs/graph_{resolution_stem}_ratio15_L4_k{joined}_{strategy_stem}_v1.pt"
    if hierarchy == "l4_72_36_24_18_9":
        joined = "_".join(str(value) for value in values)
        return f"graphs/graph_{resolution_stem}_l4_72_36_24_18_9_k{joined}_{strategy_stem}_v1.pt"
    level_stem = "_L3" if int(num_graph_levels) == 4 else ""
    k_stem = f"k{int(k)}"
    if level_k_neighbors is not None:
        if len(values) == int(num_graph_levels) and any(value != int(k) for value in values):
            if int(num_graph_levels) == 4 and values[:-1] == [int(k), int(k), int(k)]:
                k_stem = f"k{int(k)}_l3k{values[-1]}"
            else:
                k_stem = f"k{int(k)}_levelk{'-'.join(str(value) for value in values)}"
    return f"graphs/graph_{resolution_stem}_{k_stem}_{strategy_stem}{level_stem}_v{int(graph_format_version)}.pt"


def print_graph_diagnostics(bundle: dict[str, Any]) -> None:
    metadata = dict(bundle.get("metadata", {}))
    print(f"k = {metadata.get('k')}")
    print(f"level_k_neighbors = {metadata.get('level_k_neighbors')}")
    print(f"Graph U-Net levels = {metadata.get('num_graph_levels', len(bundle.get('levels', {})))}")
    for level_name in sorted(bundle["levels"], key=lambda name: int(str(name)[1:]) if str(name).startswith("L") else 0):
        level = bundle["levels"][level_name]
        diag = dict(level.get("neighbor_diagnostics", metadata.get("diagnostics", {}).get(level_name, {})))
        print()
        print(f"{level_name}:")
        print(f"  grid = {int(level['height'])} x {int(level['width'])}")
        print(f"  nodes = {int(level['num_nodes'])}")
        print(f"  k = {int(level['k'])}")
        print(f"  edges = {int(level['edge_index'].shape[1])}")
        print(f"  connected components = {diag.get('connected_components')}")
        print(f"  minimum unique neighbors = {diag.get('minimum_unique_neighbors')}")
        print(f"  maximum unique neighbors = {diag.get('maximum_unique_neighbors')}")
        print(f"  self loops = {diag.get('self_loops')}")
        print(f"  duplicate neighbors = {diag.get('duplicate_neighbors')}")
        print(f"  minimum cross-row neighbors per node = {diag.get('min_cross_row_neighbors')}")
        print(f"  north edge-row minimum adjacent-row neighbors = {diag.get('north_edge_min_adjacent_neighbors')}")
        print(f"  south edge-row minimum adjacent-row neighbors = {diag.get('south_edge_min_adjacent_neighbors')}")
    pool_stats = dict(metadata.get("pool_child_count_stats", {}) or {})
    if pool_stats:
        print()
        print("Pool child counts:")
        for name in sorted(pool_stats, key=lambda item: int(str(item).split("_to_")[0][1:])):
            stats = dict(pool_stats[name])
            print(
                f"  {name}: min={stats.get('min')} max={stats.get('max')} "
                f"mean={float(stats.get('mean', 0.0)):.3f}"
            )
    print()
    print("No self-loops")
    print("No duplicate neighbors")
    print("All nodes have exactly their configured per-level k neighbors")
    if metadata.get("connectivity_strategy") == HYBRID_ROW_AWARE_KNN:
        print("Polar rows connected to adjacent latitude rows")


def _looks_like_yaml(value: str) -> bool:
    return value.endswith((".yaml", ".yml")) or os.path.isfile(value)


def _resolve_cli_config(args: argparse.Namespace) -> tuple[Any | None, str | None, str | None]:
    yaml_path = args.yaml_config
    config_name = args.config_name
    if args.config:
        if _looks_like_yaml(args.config):
            yaml_path = args.config
            if config_name is None:
                if Path(args.config).name == "weather_dual_resolution.yaml":
                    config_name = "raw"
                elif Path(args.config).name == "weather_dual_resolution_l3.yaml":
                    config_name = "raw_l3"
                elif Path(args.config).name == "weather_dual_resolution_l3_stage_warmup_cosine.yaml":
                    config_name = "raw_l3_stage_warmup_cosine"
                elif Path(args.config).name == "weather_dual_resolution_l3_blocks3.yaml":
                    config_name = "raw_l3_blocks3"
                elif Path(args.config).name == "weather_dual_resolution_l3_heavy_unet.yaml":
                    config_name = "raw_l3_heavy_unet"
                elif Path(args.config).name == "weather_dual_resolution_l3_full_rollout.yaml":
                    config_name = "raw_l3_full_rollout"
                elif Path(args.config).name == "weather_dual_resolution_l3_hidden128.yaml":
                    config_name = "raw_l3_hidden128"
                elif Path(args.config).name == "weather_dual_resolution_l3_hidden160.yaml":
                    config_name = "raw_l3_hidden160"
                elif Path(args.config).name == "weather_dual_resolution_l3_hidden128_dense_l3k24_fixed_orog.yaml":
                    config_name = "raw_l3_hidden128_dense_l3k24_fixed_orog"
                elif Path(args.config).name == "weather_dual_resolution_l3_hidden128_scalar_gated_skip_fixed_orog.yaml":
                    config_name = "raw_l3_hidden128_scalar_gated_skip_fixed_orog"
                elif Path(args.config).name == "weather_dual_resolution_l3_hidden128_scalar_gated_pooling_fixed_orog.yaml":
                    config_name = "raw_l3_hidden128_scalar_gated_pooling_fixed_orog"
                elif Path(args.config).name == "weather_dual_resolution_l3_hidden128_lead_conditioned_fixed_orog.yaml":
                    config_name = "raw_l3_hidden128_lead_conditioned_fixed_orog"
                elif Path(args.config).name == "weather_dual_resolution_l4_ratio15_hidden128_fixed_orog.yaml":
                    config_name = "raw_l4_ratio15_hidden128_fixed_orog"
                elif Path(args.config).name == "weather_dual_resolution_l4_hidden128_72_36_24_18_9_fixed_orog.yaml":
                    config_name = "raw_l4_hidden128_72_36_24_18_9_fixed_orog"
        elif config_name is None:
            config_name = args.config
    if yaml_path is None and config_name is None:
        return None, None, None
    if yaml_path is None:
        yaml_path = str(Path(__file__).resolve().parent.parent / "configs" / "gnn_5p625.yaml")
    if config_name is None:
        config_name = "raw_5p625"
    try:
        from .config import YParams
    except ImportError:
        from config import YParams  # type: ignore
    params = YParams(os.path.abspath(yaml_path), config_name, resolution_mode=getattr(args, "resolution_mode", None))
    return params, os.path.abspath(yaml_path), config_name


def _resolution_defaults(resolution_mode: str | None) -> dict[str, float | int]:
    if resolution_mode is None:
        return {}
    spec = get_resolution_spec(resolution_mode)
    return {
        "resolution": spec.resolution_degrees,
        "lat_count": spec.height,
        "lon_count": spec.width,
        "lat_start": None,
        "lon_start": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    parser.add_argument("--data", default=None, help="NetCDF file or directory with latitude/longitude coordinates")
    parser.add_argument("--config", default=None, help="YAML path or config section name.")
    parser.add_argument("--yaml_config", default=None, help="Explicit YAML config path.")
    parser.add_argument("--config_name", default=None, help="YAML section name, default raw_5p625.")
    parser.add_argument("--resolution_mode", default=None, help="Preset grid mode, e.g. 5p625.")
    parser.add_argument("--resolution", type=float, default=5.625)
    parser.add_argument("--lat-count", type=int, default=32)
    parser.add_argument("--lon-count", type=int, default=64)
    parser.add_argument("--lat-start", type=float, default=None)
    parser.add_argument("--lon-start", type=float, default=-180.0)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument(
        "--level_k_neighbors",
        default=None,
        help="Optional comma/space separated per-level k values, e.g. '8,8,8,24'.",
    )
    parser.add_argument("--l3_k_neighbors", type=int, default=None)
    parser.add_argument("--num_graph_levels", type=int, default=None, choices=[3, 4, 5])
    parser.add_argument(
        "--graph_connectivity_strategy",
        default=None,
        choices=sorted(SUPPORTED_CONNECTIVITY_STRATEGIES),
    )
    parser.add_argument("--force_rebuild", action="store_true")
    args = parser.parse_args()
    cli_num_graph_levels = args.num_graph_levels

    params, _, _ = _resolve_cli_config(args)
    defaults = _resolution_defaults(args.resolution_mode)
    if defaults:
        args.resolution = float(defaults["resolution"])
        args.lat_count = int(defaults["lat_count"])
        args.lon_count = int(defaults["lon_count"])
        args.lat_start = None if defaults["lat_start"] is None else float(defaults["lat_start"])
        args.lon_start = float(defaults["lon_start"])

    if params is not None:
        args.resolution = float(getattr(params, "resolution", args.resolution))
        args.k = int(getattr(params, "k_neighbors", args.k))
        if args.level_k_neighbors is None and getattr(params, "level_k_neighbors", None) is not None:
            args.level_k_neighbors = ",".join(str(x) for x in getattr(params, "level_k_neighbors"))
        if args.l3_k_neighbors is None and getattr(params, "l3_k_neighbors", None) is not None:
            args.l3_k_neighbors = int(getattr(params, "l3_k_neighbors"))
        if args.num_graph_levels is None:
            args.num_graph_levels = int(getattr(params, "num_graph_levels", 3))
        candidate_data = getattr(params, "train_data_path", None)
        if args.data is None and candidate_data and os.path.exists(str(candidate_data)):
            args.data = str(candidate_data)
        args.resolution_mode = getattr(params, "resolution_mode", args.resolution_mode)
    args.num_graph_levels = int(args.num_graph_levels or 3)

    strategy = normalize_connectivity_strategy(
        args.graph_connectivity_strategy
        or (getattr(params, "graph_connectivity_strategy", None) if params is not None else None)
        or HYBRID_ROW_AWARE_KNN
    )
    row_aware_knn = normalize_row_aware_config(getattr(params, "row_aware_knn", {}) if params is not None else {})
    level_shapes = getattr(params, "level_shapes", None) if params is not None else None
    hierarchy_type = getattr(params, "hierarchy_type", None) if params is not None else None
    use_l4_ratio15 = bool(getattr(params, "use_l4_ratio15", False)) if params is not None else False
    level_k_source: dict[str, Any] = {
        "k_neighbors": int(args.k),
        "num_graph_levels": int(args.num_graph_levels or 3),
    }
    if args.level_k_neighbors is not None:
        level_k_source["level_k_neighbors"] = args.level_k_neighbors
    elif args.l3_k_neighbors is not None:
        level_k_source["l3_k_neighbors"] = int(args.l3_k_neighbors)
    level_k_neighbors = list(
        resolve_level_k_neighbors(
            level_k_source,
            num_graph_levels=int(args.num_graph_levels or 3),
            k_neighbors=int(args.k),
        )
    )

    if args.data:
        latitudes, longitudes = lat_lon_from_netcdf(args.data)
    elif args.resolution_mode:
        spec = get_resolution_spec(args.resolution_mode)
        latitudes, longitudes = cell_center_lat_lon(spec.height, spec.width)
    else:
        latitudes, longitudes = regular_lat_lon(
            lat_count=args.lat_count,
            lon_count=args.lon_count,
            resolution=args.resolution,
            lat_start=args.lat_start,
            lon_start=args.lon_start,
        )

    mesh_cfg = dict(getattr(params, "mesh_encoder", {}) or {}) if params is not None else {}
    output = args.output
    if output is None and params is not None and (
        cli_num_graph_levels is None or int(cli_num_graph_levels) == int(getattr(params, "num_graph_levels", 3))
    ):
        output = str(getattr(params, "graph_path", ""))
    if not output:
        output = graph_path_for_resolution(
            args.resolution,
            args.k,
            strategy,
            resolution_mode=args.resolution_mode,
            num_graph_levels=args.num_graph_levels,
            level_k_neighbors=level_k_neighbors,
            hierarchy_type=hierarchy_type,
            use_l4_ratio15=use_l4_ratio15,
        )
    output = os.path.abspath(output)

    # Icosphere mesh mode (mesh_encoder.enabled): build a mesh bundle instead of
    # the lat-lon graph, using the data's own lat/lon axes (row-major over lat,lon).
    if bool(mesh_cfg.get("enabled", False)):
        from .mesh_builder import (
            bipartite_edge_feature_set,
            bipartite_mapping_type,
            build_and_save,
            expected_mesh_metadata,
            validate_mesh_cache_metadata,
        )
        lat = latitudes.detach().cpu().numpy() if hasattr(latitudes, "detach") else np.asarray(latitudes)
        lon = longitudes.detach().cpu().numpy() if hasattr(longitudes, "detach") else np.asarray(longitudes)
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
        grid_ll = np.deg2rad(np.stack([lat_grid.reshape(-1), lon_grid.reshape(-1)], axis=1)).astype(np.float32)
        bipartite_features = bipartite_edge_feature_set(
            mesh_cfg.get("boundary_type", "legacy")
        )
        bipartite_mapping = bipartite_mapping_type(
            mesh_cfg.get("boundary_type", "legacy")
        )
        grid_attention_enabled = (
            int(mesh_cfg.get("grid_attention_encoder_blocks", 0)) > 0
            or int(mesh_cfg.get("grid_attention_decoder_blocks", 0)) > 0
        )
        grid_attention_k = (
            int(mesh_cfg.get("grid_attention_k_neighbors", 8))
            if grid_attention_enabled
            else None
        )
        grid_attention_reference = (
            getattr(params, "grid_attention_reference_graph_path", None)
            if params is not None
            else None
        )
        expected_mesh = expected_mesh_metadata(
            refinement=int(mesh_cfg.get("refinement", 5)),
            num_graph_levels=int(args.num_graph_levels),
            grid_shape=(int(lat.shape[0]), int(lon.shape[0])),
            grid_lat_lon=grid_ll,
            g2m_radius_factor=float(mesh_cfg.get("g2m_radius_factor", 0.6)),
            resolution_mode=args.resolution_mode,
            bipartite_mapping_type=bipartite_mapping,
            bipartite_edge_features=bipartite_features,
            coarse_level_connectivity=str(
                mesh_cfg.get("coarse_level_connectivity", "native_icosphere")
            ),
            grid_attention_k_neighbors=grid_attention_k,
            grid_attention_connectivity_strategy=strategy,
            grid_attention_row_aware_knn=row_aware_knn,
            grid_attention_reference_graph_path=grid_attention_reference,
        )
        if os.path.isfile(output) and not args.force_rebuild:
            cached = load_raw_graph_bundle(output)
            mismatches = validate_mesh_cache_metadata(cached, expected_mesh)
            if not mismatches:
                print(f"Mesh graph cache already present: {output}")
                return
            print(f"Mesh graph cache metadata mismatch for {output}; rebuilding.")
            for mismatch in mismatches:
                print(f"  {mismatch}")
        meta = build_and_save(
            output,
            refinement=int(mesh_cfg.get("refinement", 5)),
            num_graph_levels=int(args.num_graph_levels or 4),
            grid_shape=(int(lat.shape[0]), int(lon.shape[0])),
            grid_lat_lon=grid_ll,
            g2m_radius_factor=float(mesh_cfg.get("g2m_radius_factor", 0.6)),
            resolution_mode=args.resolution_mode,
            bipartite_mapping_type=bipartite_mapping,
            bipartite_edge_features=bipartite_features,
            coarse_level_connectivity=str(
                mesh_cfg.get("coarse_level_connectivity", "native_icosphere")
            ),
            grid_attention_k_neighbors=grid_attention_k,
            grid_attention_connectivity_strategy=strategy,
            grid_attention_row_aware_knn=row_aware_knn,
            grid_attention_reference_graph_path=grid_attention_reference,
        )
        print(f"saved icosphere mesh bundle {output} : {meta}")
        return

    expected = expected_graph_metadata(
        latitudes,
        longitudes,
        k=args.k,
        resolution=args.resolution,
        connectivity_strategy=strategy,
        row_aware_knn=row_aware_knn,
        resolution_mode=args.resolution_mode,
        num_graph_levels=args.num_graph_levels,
        level_k_neighbors=level_k_neighbors,
        level_shapes=level_shapes,
        hierarchy_type=hierarchy_type,
        use_l4_ratio15=use_l4_ratio15,
    )
    if os.path.isfile(output) and not args.force_rebuild:
        cached = load_raw_graph_bundle(output)
        mismatches = validate_graph_cache_metadata(cached, expected)
        if not mismatches:
            print(f"Graph cache already valid: {output}")
            print_graph_diagnostics(cached)
            return
        print(f"Graph cache metadata mismatch for {output}; rebuilding.")
        for mismatch in mismatches:
            print(f"  {mismatch}")

    if strategy == HYBRID_ROW_AWARE_KNN:
        print("Building hybrid row-aware spherical kNN graph")
    else:
        print("Building pure spherical kNN graph")
    bundle = build_graph_bundle(
        latitudes,
        longitudes,
        k=args.k,
        resolution=args.resolution,
        connectivity_strategy=strategy,
        row_aware_knn=row_aware_knn,
        resolution_mode=args.resolution_mode,
        num_graph_levels=args.num_graph_levels,
        level_k_neighbors=level_k_neighbors,
        level_shapes=level_shapes,
        hierarchy_type=hierarchy_type,
        use_l4_ratio15=use_l4_ratio15,
    )
    save_graph(bundle, output)
    print_graph_diagnostics(bundle)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
