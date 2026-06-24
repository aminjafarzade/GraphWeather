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

from .resolution import (
    RESOLUTION_SPECS,
    cell_center_lat_lon,
    canonicalize_resolution_mode,
    default_graph_path,
    get_resolution_spec,
)


GRAPH_FORMAT_VERSION = 2
PURE_SPHERICAL_KNN = "pure_spherical_knn"
HYBRID_ROW_AWARE_KNN = "hybrid_row_aware_knn"
SUPPORTED_CONNECTIVITY_STRATEGIES = {PURE_SPHERICAL_KNN, HYBRID_ROW_AWARE_KNN}


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
    target_rows = (target_ids // width)[:, None]
    neighbor_rows = neighbors // width

    unique_counts = np.asarray([len(set(row.tolist())) for row in neighbors], dtype=np.int64)
    duplicate_neighbors = int(np.sum(k - unique_counts))
    self_loops = int(np.sum(neighbors == target_ids[:, None]))
    same_row = np.sum(neighbor_rows == target_rows, axis=1)
    north_row = np.sum(neighbor_rows == target_rows - 1, axis=1)
    south_row = np.sum(neighbor_rows == target_rows + 1, axis=1)
    cross_row = k - same_row
    other_row = k - same_row - north_row - south_row
    components = _connected_component_count(edge_index, num_nodes)

    interior_mask = (target_ids // width > 0) & (target_ids // width < height - 1)
    north_edge_adjacent = np.sum(neighbor_rows[:width] == 1, axis=1) if height > 1 else np.asarray([], dtype=np.int64)
    south_start = (height - 1) * width
    south_edge_adjacent = (
        np.sum(neighbor_rows[south_start:] == height - 2, axis=1) if height > 1 else np.asarray([], dtype=np.int64)
    )

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
    fine_i = torch.arange(fine_height).repeat_interleave(fine_width)
    fine_j = torch.arange(fine_width).repeat(fine_height)
    pool_map = (fine_i // 2) * coarse_width + (fine_j // 2)
    return pool_map.to(torch.long), coarse_height, coarse_width


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
) -> dict[str, Any]:
    height0 = int(latitudes_deg.numel())
    width0 = int(longitudes_deg.numel())
    coords0, latlon0 = grid_nodes(latitudes_deg, longitudes_deg)
    pool01, height1, width1 = make_pool_map(height0, width0)
    coords1, latlon1 = coarsen_coords(coords0, pool01, height1 * width1)
    pool12, height2, width2 = make_pool_map(height1, width1)
    _, latlon2 = coarsen_coords(coords1, pool12, height2 * width2)
    strategy = normalize_connectivity_strategy(connectivity_strategy)
    row_cfg = normalize_row_aware_config(row_aware_knn)
    levels = {
        "L0": [height0, width0],
        "L1": [height1, width1],
        "L2": [height2, width2],
    }
    level_shapes = [[height0, width0], [height1, width1], [height2, width2]]
    node_counts = [height0 * width0, height1 * width1, height2 * width2]
    edge_counts = [int(count) * int(k) for count in node_counts]
    mode = resolution_mode
    if mode is None:
        for spec in RESOLUTION_SPECS.values():
            if (height0, width0) == spec.grid_shape and abs(float(resolution) - spec.resolution_degrees) < 1.0e-6:
                mode = spec.name
                break
    mode = canonicalize_resolution_mode(mode) if mode else None
    level_hashes = {
        "L0": coordinate_hash_from_lat_lon(latlon0),
        "L1": coordinate_hash_from_lat_lon(latlon1),
        "L2": coordinate_hash_from_lat_lon(latlon2),
    }
    return {
        "graph_format_version": GRAPH_FORMAT_VERSION,
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
        "connectivity_strategy": strategy,
        "graph_connectivity_strategy": strategy,
        "row_aware_knn": row_cfg,
        "interior_north_count": int(row_cfg["interior_neighbors_from_north_row"]),
        "interior_south_count": int(row_cfg["interior_neighbors_from_south_row"]),
        "polar_adjacent_count": int(row_cfg["polar_neighbors_from_adjacent_row"]),
        "node_order": "lat_index * num_lon + lon_index",
        "levels": levels,
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
) -> dict[str, Any]:
    return _coordinate_metadata(
        latitudes_deg=latitudes_deg,
        longitudes_deg=longitudes_deg,
        k=k,
        resolution=resolution,
        connectivity_strategy=connectivity_strategy,
        row_aware_knn=row_aware_knn,
        resolution_mode=resolution_mode,
    )


def graph_topology_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "resolution_mode": metadata.get("resolution_mode"),
        "graph_connectivity_strategy": metadata.get("graph_connectivity_strategy", metadata.get("connectivity_strategy")),
        "graph_format_version": int(metadata.get("graph_format_version", 0)),
        "graph_k": int(metadata.get("graph_k", metadata.get("k", 0))),
        "graph_coordinate_hash": metadata.get("graph_coordinate_hash", metadata.get("coordinate_hash")),
    }


def validate_graph_cache_metadata(bundle: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    actual = dict(bundle.get("metadata", {}))
    mismatches: list[str] = []
    comparisons = [
        ("graph_format_version", "graph_format_version"),
        ("resolution_mode", "resolution_mode"),
        ("connectivity_strategy", "connectivity_strategy"),
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
    if actual.get("levels") != expected.get("levels"):
        mismatches.append(f"levels: cache={actual.get('levels')!r}, expected={expected.get('levels')!r}")
    for key in ("grid_shape", "level_shapes", "node_counts", "edge_counts"):
        if actual.get(key) != expected.get(key):
            mismatches.append(f"{key}: cache={actual.get(key)!r}, expected={expected.get(key)!r}")
    if actual.get("level_coordinate_hashes") != expected.get("level_coordinate_hashes"):
        mismatches.append("level_coordinate_hashes differ")
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
) -> dict[str, object]:
    height0 = int(latitudes_deg.numel())
    width0 = int(longitudes_deg.numel())
    coords0, latlon0 = grid_nodes(latitudes_deg, longitudes_deg)
    pool01, height1, width1 = make_pool_map(height0, width0)
    coords1, latlon1 = coarsen_coords(coords0, pool01, height1 * width1)
    pool12, height2, width2 = make_pool_map(height1, width1)
    coords2, latlon2 = coarsen_coords(coords1, pool12, height2 * width2)

    strategy = normalize_connectivity_strategy(connectivity_strategy)
    row_cfg = normalize_row_aware_config(row_aware_knn)
    levels = {
        "L0": make_level(coords0, latlon0, height0, width0, k, strategy, row_cfg),
        "L1": make_level(coords1, latlon1, height1, width1, k, strategy, row_cfg),
        "L2": make_level(coords2, latlon2, height2, width2, k, strategy, row_cfg),
    }
    metadata = _coordinate_metadata(
        latitudes_deg=latitudes_deg,
        longitudes_deg=longitudes_deg,
        k=k,
        resolution=resolution,
        connectivity_strategy=strategy,
        row_aware_knn=row_cfg,
        resolution_mode=resolution_mode,
    )
    metadata["diagnostics"] = {
        name: dict(level["neighbor_diagnostics"])
        for name, level in levels.items()
    }

    return {
        "metadata": metadata,
        "levels": {
            "L0": levels["L0"],
            "L1": levels["L1"],
            "L2": levels["L2"],
        },
        "pool": {
            "L0_to_L1": pool01.to(torch.long),
            "L1_to_L2": pool12.to(torch.long),
        },
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
) -> str:
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
        )
    resolution_stem = str(float(resolution)).replace(".", "p")
    strategy_stem = normalize_connectivity_strategy(connectivity_strategy)
    if strategy_stem == HYBRID_ROW_AWARE_KNN:
        strategy_stem = "hybrid_row_aware"
    return f"graphs/graph_{resolution_stem}_k{int(k)}_{strategy_stem}_v{int(graph_format_version)}.pt"


def print_graph_diagnostics(bundle: dict[str, Any]) -> None:
    metadata = dict(bundle.get("metadata", {}))
    print(f"k = {metadata.get('k')}")
    for level_name in ("L0", "L1", "L2"):
        level = bundle["levels"][level_name]
        diag = dict(level.get("neighbor_diagnostics", metadata.get("diagnostics", {}).get(level_name, {})))
        print()
        print(f"{level_name}:")
        print(f"  grid = {int(level['height'])} x {int(level['width'])}")
        print(f"  nodes = {int(level['num_nodes'])}")
        print(f"  edges = {int(level['edge_index'].shape[1])}")
        print(f"  connected components = {diag.get('connected_components')}")
        print(f"  minimum unique neighbors = {diag.get('minimum_unique_neighbors')}")
        print(f"  maximum unique neighbors = {diag.get('maximum_unique_neighbors')}")
        print(f"  self loops = {diag.get('self_loops')}")
        print(f"  duplicate neighbors = {diag.get('duplicate_neighbors')}")
        print(f"  minimum cross-row neighbors per node = {diag.get('min_cross_row_neighbors')}")
        print(f"  north edge-row minimum adjacent-row neighbors = {diag.get('north_edge_min_adjacent_neighbors')}")
        print(f"  south edge-row minimum adjacent-row neighbors = {diag.get('south_edge_min_adjacent_neighbors')}")
    print()
    print("No self-loops")
    print("No duplicate neighbors")
    print("All nodes have exactly 8 neighbors" if int(metadata.get("k", 0)) == 8 else "All nodes have exactly k neighbors")
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
            if config_name is None and Path(args.config).name == "weather_dual_resolution.yaml":
                config_name = "raw"
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
        "--graph_connectivity_strategy",
        default=None,
        choices=sorted(SUPPORTED_CONNECTIVITY_STRATEGIES),
    )
    parser.add_argument("--force_rebuild", action="store_true")
    args = parser.parse_args()

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
        candidate_data = getattr(params, "train_data_path", None)
        if args.data is None and candidate_data and os.path.exists(str(candidate_data)):
            args.data = str(candidate_data)
        args.resolution_mode = getattr(params, "resolution_mode", args.resolution_mode)

    strategy = normalize_connectivity_strategy(
        args.graph_connectivity_strategy
        or (getattr(params, "graph_connectivity_strategy", None) if params is not None else None)
        or HYBRID_ROW_AWARE_KNN
    )
    row_aware_knn = normalize_row_aware_config(getattr(params, "row_aware_knn", {}) if params is not None else {})

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

    output = args.output
    if output is None and params is not None:
        output = str(getattr(params, "graph_path", ""))
    if not output:
        output = graph_path_for_resolution(args.resolution, args.k, strategy, resolution_mode=args.resolution_mode)
    output = os.path.abspath(output)

    expected = expected_graph_metadata(
        latitudes,
        longitudes,
        k=args.k,
        resolution=args.resolution,
        connectivity_strategy=strategy,
        row_aware_knn=row_aware_knn,
        resolution_mode=args.resolution_mode,
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
    )
    save_graph(bundle, output)
    print_graph_diagnostics(bundle)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
