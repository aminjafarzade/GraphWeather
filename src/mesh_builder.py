"""Build an icosphere (geodesic) mesh bundle for the GraphCast-style encode/decode.

The bundle is load-compatible with graph_bundle.GraphBundle (metadata.graph_mode
== "mesh"). Mesh levels L0..L{n-1} are icosphere refinements M_r, M_{r-1}, ...;
by default their edges are the icosphere's OWN triangulation (degree 6 except the
12 original icosahedron vertices at degree 5), padded to k=6 with a validity mask.
An opt-in ``full_m1`` topology makes only a coarsest M1 level fully connected.
Pool maps come from the refinement hierarchy. grid<->mesh edges follow GraphCast:
grid->mesh either follows GraphCast's radius rule or uses parameter-free
spherical barycentric remapping. Mesh->grid uses the 3 vertices of the
containing triangle; the fixed remapping path also stores their interpolation
weights.
"""
from __future__ import annotations

import math
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree

from .graph_builder import (
    coordinate_hash_from_lat_lon,
    edge_features,
    make_level,
    normalize_connectivity_strategy,
    normalize_row_aware_config,
)

_K = 6  # uniform padded degree for the mesh levels (icosphere max degree)
MESH_FORMAT_VERSION = 1
LEGACY_BIPARTITE_EDGE_FEATURES = "legacy_spherical_6"
GRAPHCAST_BIPARTITE_EDGE_FEATURES = "legacy_plus_receiver_local_10"
GRAPHCAST_RADIUS_MAPPING = "graphcast_radius"
FIXED_SPHERICAL_MAPPING = "fixed_spherical_barycentric"
FIXED_SPHERICAL_METHOD = "spherical_triangle_area_barycentric"
FIXED_SPHERICAL_POLE_HANDLING = "longitude_ring_mean"
NATIVE_COARSE_CONNECTIVITY = "native_icosphere"
FULL_M1_COARSE_CONNECTIVITY = "full_m1"
SUPPORTED_COARSE_LEVEL_CONNECTIVITY = {
    NATIVE_COARSE_CONNECTIVITY,
    FULL_M1_COARSE_CONNECTIVITY,
}


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_grid_attention_reference(
    path: str | Path,
    *,
    grid_shape: tuple[int, int],
    grid_lat_lon: np.ndarray | torch.Tensor,
    k_neighbors: int,
    connectivity_strategy: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reference_path = Path(path).expanduser().resolve()
    if not reference_path.is_file():
        raise FileNotFoundError(
            f"Grid-attention reference graph not found: {reference_path}"
        )
    try:
        raw = torch.load(reference_path, map_location="cpu", weights_only=True)
    except TypeError:
        raw = torch.load(reference_path, map_location="cpu")
    try:
        level = raw["levels"]["L0"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Grid-attention reference graph has no levels['L0']: {reference_path}"
        ) from exc

    H, W = int(grid_shape[0]), int(grid_shape[1])
    if (
        int(level.get("height", 0)) != H
        or int(level.get("width", 0)) != W
        or int(level.get("num_nodes", 0)) != H * W
    ):
        raise ValueError(
            "Grid-attention reference L0 does not match the mesh data grid: "
            f"reference={(level.get('height'), level.get('width'), level.get('num_nodes'))}, "
            f"grid={(H, W, H * W)}."
        )
    if int(level.get("k", 0)) != int(k_neighbors):
        raise ValueError(
            "Grid-attention reference L0 k does not match the requested value: "
            f"reference={level.get('k')}, requested={int(k_neighbors)}."
        )
    reference_lat_lon = torch.as_tensor(level["lat_lon"], dtype=torch.float32)
    expected_lat_lon = torch.as_tensor(grid_lat_lon, dtype=torch.float32).reshape(-1, 2)
    if not torch.allclose(
        reference_lat_lon,
        expected_lat_lon,
        atol=1.0e-6,
        rtol=0.0,
    ):
        max_difference = float(
            (reference_lat_lon - expected_lat_lon).abs().max().item()
        )
        raise ValueError(
            "Grid-attention reference L0 coordinates do not match the mesh data "
            f"grid ordering (max abs difference={max_difference:.6g})."
        )
    reference_metadata = dict(raw.get("metadata", {}) or {})
    reference_strategy = str(
        reference_metadata.get(
            "graph_connectivity_strategy",
            reference_metadata.get("connectivity_strategy", ""),
        )
    )
    expected_strategy = normalize_connectivity_strategy(connectivity_strategy)
    if reference_strategy != expected_strategy:
        raise ValueError(
            "Grid-attention reference connectivity does not match the requested "
            f"strategy: reference={reference_strategy!r}, requested={expected_strategy!r}."
        )
    provenance = {
        "grid_attention_reference_graph": reference_path.name,
        "grid_attention_reference_sha256": _sha256_file(reference_path),
        "grid_attention_reference_coordinate_hash": reference_metadata.get(
            "graph_coordinate_hash",
            reference_metadata.get("coordinate_hash"),
        ),
    }
    return dict(level), provenance


def bipartite_edge_feature_set(boundary_type: str) -> str:
    boundary = str(boundary_type).strip().lower()
    if boundary in {"legacy", "fixed_spherical"}:
        return LEGACY_BIPARTITE_EDGE_FEATURES
    if boundary == "graphcast_mlp":
        return GRAPHCAST_BIPARTITE_EDGE_FEATURES
    raise ValueError(
        "Unsupported mesh boundary_type="
        f"{boundary_type!r}; expected 'legacy', 'graphcast_mlp', or 'fixed_spherical'."
    )


def bipartite_mapping_type(boundary_type: str) -> str:
    boundary = str(boundary_type).strip().lower()
    if boundary in {"legacy", "graphcast_mlp"}:
        return GRAPHCAST_RADIUS_MAPPING
    if boundary == "fixed_spherical":
        return FIXED_SPHERICAL_MAPPING
    raise ValueError(
        "Unsupported mesh boundary_type="
        f"{boundary_type!r}; expected 'legacy', 'graphcast_mlp', or 'fixed_spherical'."
    )


def normalize_bipartite_mapping_type(value: str) -> str:
    mapping_type = str(value).strip().lower()
    supported = {GRAPHCAST_RADIUS_MAPPING, FIXED_SPHERICAL_MAPPING}
    if mapping_type not in supported:
        available = ", ".join(sorted(supported))
        raise ValueError(
            f"Unsupported bipartite_mapping_type={value!r}; expected one of: {available}."
        )
    return mapping_type


def bipartite_edge_feature_dim(feature_set: str) -> int:
    feature_set = str(feature_set)
    if feature_set == LEGACY_BIPARTITE_EDGE_FEATURES:
        return 6
    if feature_set == GRAPHCAST_BIPARTITE_EDGE_FEATURES:
        return 10
    raise ValueError(f"Unsupported bipartite edge feature set {feature_set!r}.")


def normalize_coarse_level_connectivity(value: str) -> str:
    connectivity = str(value).strip().lower()
    if connectivity not in SUPPORTED_COARSE_LEVEL_CONNECTIVITY:
        available = ", ".join(sorted(SUPPORTED_COARSE_LEVEL_CONNECTIVITY))
        raise ValueError(
            f"Unsupported coarse_level_connectivity={value!r}; expected one of: {available}."
        )
    return connectivity


def mesh_connectivity_strategy(coarse_level_connectivity: str) -> str:
    connectivity = normalize_coarse_level_connectivity(coarse_level_connectivity)
    if connectivity == FULL_M1_COARSE_CONNECTIVITY:
        return "native_icosphere_with_full_m1"
    return NATIVE_COARSE_CONNECTIVITY


# --------------------------------------------------------------------------- #
# icosphere geometry
# --------------------------------------------------------------------------- #
def _icosahedron() -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    t = (1.0 + math.sqrt(5.0)) / 2.0
    v = np.array(
        [
            (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
            (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
            (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
        ],
        dtype=np.float64,
    )
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]
    return v, faces


def _subdivide(verts: np.ndarray, faces: list) -> tuple[np.ndarray, list, np.ndarray]:
    """One 4-way subdivision. Existing vertex indices are preserved (prefix);
    new edge-midpoint vertices are appended. Returns (new_verts, new_faces,
    parent) where parent[i] is i's index in the COARSE level (identity for
    retained vertices, an endpoint for new midpoints) -> the pool/unpool map."""
    n0 = verts.shape[0]
    new_verts = [verts[i] for i in range(n0)]
    parent = list(range(n0))
    cache: dict[tuple[int, int], int] = {}

    def midpoint(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        idx = cache.get(key)
        if idx is not None:
            return idx
        m = new_verts[a] + new_verts[b]
        m = m / np.linalg.norm(m)
        idx = len(new_verts)
        new_verts.append(m)
        # The projected midpoint is equally close to its two retained endpoints;
        # choose the lower endpoint as the deterministic nearest-parent tie break.
        parent.append(key[0])
        cache[key] = idx
        return idx

    new_faces = []
    for a, b, c in faces:
        ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
        new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
    return np.asarray(new_verts, dtype=np.float64), new_faces, np.asarray(parent, dtype=np.int64)


def _hierarchy(refinement: int) -> list[dict]:
    """levels[m] = {verts, faces, parent(to m-1)} for m = 0..refinement."""
    if int(refinement) < 0:
        raise ValueError(f"refinement must be >= 0, got {refinement}.")
    verts, faces = _icosahedron()
    levels = [{"verts": verts, "faces": faces, "parent": None}]
    for _ in range(int(refinement)):
        nv, nf, parent = _subdivide(levels[-1]["verts"], levels[-1]["faces"])
        levels.append({"verts": nv, "faces": nf, "parent": parent})
    return levels


def _xyz_to_latlon(xyz: np.ndarray) -> np.ndarray:
    z = np.clip(xyz[:, 2], -1.0, 1.0)
    lat = np.arcsin(z)
    lon = np.arctan2(xyz[:, 1], xyz[:, 0])
    return np.stack([lat, lon], axis=1).astype(np.float32)


def _adjacency(n: int, faces: list) -> list[list[int]]:
    adj: list[set] = [set() for _ in range(n)]
    for a, b, c in faces:
        adj[a].update((b, c))
        adj[b].update((a, c))
        adj[c].update((a, b))
    return [sorted(s) for s in adj]


def _native_edges(faces: list[tuple[int, int, int]] | np.ndarray) -> np.ndarray:
    edges: set[tuple[int, int]] = set()
    for a, b, c in faces:
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            edges.add((u, v) if u < v else (v, u))
    return np.asarray(sorted(edges), dtype=np.int64).T


def build_icosphere(
    refinement: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(xyz, lat_lon, native_edges, faces)`` for refinement ``r``.

    ``native_edges`` contains each undirected triangulation edge once, with shape
    ``[2,E]``. Vertex indices are hierarchical: all vertices from M(r-1) are the
    prefix of M(r), followed by the deduplicated projected edge midpoints.
    """

    geometry = _hierarchy(int(refinement))[-1]
    xyz = np.asarray(geometry["verts"], dtype=np.float32)
    faces = np.asarray(geometry["faces"], dtype=np.int64)
    expected_nodes = 10 * (4 ** int(refinement)) + 2
    if int(xyz.shape[0]) != expected_nodes:
        raise AssertionError(
            f"Icosphere M{int(refinement)} has {xyz.shape[0]} nodes, expected {expected_nodes}."
        )
    return (
        torch.from_numpy(xyz),
        torch.from_numpy(_xyz_to_latlon(xyz)),
        torch.from_numpy(_native_edges(faces)),
        torch.from_numpy(faces),
    )


def _level_dict(
    verts: np.ndarray,
    faces: list,
    connectivity: str = NATIVE_COARSE_CONNECTIVITY,
) -> dict[str, Any]:
    n = verts.shape[0]
    lat_lon = _xyz_to_latlon(verts)
    connectivity = normalize_coarse_level_connectivity(connectivity)
    if connectivity == FULL_M1_COARSE_CONNECTIVITY:
        k = n - 1
        nodes = np.arange(n, dtype=np.int64)
        src = np.stack([nodes[nodes != dst] for dst in range(n)], axis=0)
        mask = np.ones((n, k), dtype=bool)
    else:
        k = _K
        adj = _adjacency(n, faces)
        src = np.empty((n, k), dtype=np.int64)
        mask = np.zeros((n, k), dtype=bool)
        for i in range(n):
            nb = adj[i]
            deg = len(nb)  # 5 (the 12 original verts) or 6
            for j in range(k):
                if j < deg:
                    src[i, j] = nb[j]
                    mask[i, j] = True
                else:
                    src[i, j] = i          # dummy self-edge, masked out
    dst = np.repeat(np.arange(n, dtype=np.int64), k)
    edge_index = np.stack([src.reshape(-1), dst], axis=0)
    ll_t = torch.from_numpy(lat_lon)
    edge_attr = edge_features(ll_t, torch.from_numpy(edge_index)).numpy()
    edge_attr[~mask.reshape(-1)] = 0.0  # inert (masked) slots
    return {
        "height": 0,
        "width": 0,
        "num_nodes": int(n),
        "k": int(k),
        "coords": torch.from_numpy(verts.astype(np.float32)),
        "lat_lon": ll_t,
        "edge_index": torch.from_numpy(edge_index),
        "edge_attr": torch.from_numpy(edge_attr.astype(np.float32)),
        "edge_mask": torch.from_numpy(mask),
    }


# --------------------------------------------------------------------------- #
# grid geometry
# --------------------------------------------------------------------------- #
def _grid_latlon(height: int, width: int) -> np.ndarray:
    """Row-major (node = row*W + col) lat/lon in radians, matching resolution.py
    (lat south->north linspace(-90+dlat/2, 90-dlat/2), lon 0->360-dlon) and the
    GridNodeAdapter's permute(0,2,3,1).reshape node order."""
    dlat, dlon = 180.0 / height, 360.0 / width
    lats = np.linspace(-90.0 + dlat / 2.0, 90.0 - dlat / 2.0, height)
    lons = np.linspace(0.0, 360.0 - dlon, width)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    ll_deg = np.stack([lat_grid.reshape(-1), lon_grid.reshape(-1)], axis=1)
    return np.deg2rad(ll_deg).astype(np.float32)


def _latlon_to_xyz(lat_lon: np.ndarray) -> np.ndarray:
    lat, lon = lat_lon[:, 0], lat_lon[:, 1]
    return np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1
    ).astype(np.float64)


def _receiver_local_geometry(
    src_ll: np.ndarray,
    dst_ll: np.ndarray,
    edge_index: np.ndarray,
) -> np.ndarray:
    """GraphCast-style [normalized distance, receiver-local relative xyz].

    Each receiver is rotated to longitude=0 and latitude=0, making the local
    coordinate axes radial/east/north. Distance and relative coordinates share
    the maximum chord length of this bipartite edge set as their normalization.
    """

    src_xyz = _latlon_to_xyz(src_ll)[edge_index[0]]
    dst_xyz = _latlon_to_xyz(dst_ll)[edge_index[1]]
    relative = src_xyz - dst_xyz
    distance = np.linalg.norm(relative, axis=1)
    normalization = float(distance.max(initial=0.0))
    if not math.isfinite(normalization) or normalization <= 0.0:
        raise ValueError("Bipartite edge geometry has no positive finite distance.")

    dst_edge_ll = dst_ll[edge_index[1]]
    lat = dst_edge_ll[:, 0].astype(np.float64)
    lon = dst_edge_ll[:, 1].astype(np.float64)
    radial = np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        axis=1,
    )
    east = np.stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)], axis=1)
    north = np.stack(
        [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
        axis=1,
    )
    relative_local = np.stack(
        [
            np.sum(relative * radial, axis=1),
            np.sum(relative * east, axis=1),
            np.sum(relative * north, axis=1),
        ],
        axis=1,
    )
    return np.concatenate(
        [(distance / normalization)[:, None], relative_local / normalization],
        axis=1,
    ).astype(np.float32)


def _bipartite_edge_attr(
    src_ll: np.ndarray,
    dst_ll: np.ndarray,
    edge_index: np.ndarray,
    feature_set: str = LEGACY_BIPARTITE_EDGE_FEATURES,
) -> np.ndarray:
    """Existing spherical features, optionally plus receiver-local geometry."""

    n_src = src_ll.shape[0]
    combined = np.concatenate([src_ll, dst_ll], axis=0)
    ei_off = np.stack([edge_index[0], edge_index[1] + n_src], axis=0)
    legacy = edge_features(
        torch.from_numpy(combined),
        torch.from_numpy(ei_off),
    ).numpy().astype(np.float32)
    if feature_set == LEGACY_BIPARTITE_EDGE_FEATURES:
        return legacy
    if feature_set == GRAPHCAST_BIPARTITE_EDGE_FEATURES:
        local = _receiver_local_geometry(src_ll, dst_ll, edge_index)
        return np.concatenate([legacy, local], axis=1).astype(np.float32)
    raise ValueError(f"Unsupported bipartite edge feature set {feature_set!r}.")


def _point_in_spherical_triangle(p, a, b, c, eps: float = 5e-7) -> bool:
    """Test the minor spherical triangle using inward-oriented edge half-spaces."""

    ab = np.cross(a, b)
    bc = np.cross(b, c)
    ca = np.cross(c, a)
    return (
        float(np.dot(ab, p)) * float(np.dot(ab, c)) >= -eps
        and float(np.dot(bc, p)) * float(np.dot(bc, a)) >= -eps
        and float(np.dot(ca, p)) * float(np.dot(ca, b)) >= -eps
    )


def _spherical_triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Area of a minor unit-sphere triangle in steradians."""

    numerator = abs(float(np.dot(a, np.cross(b, c))))
    denominator = 1.0 + float(np.dot(a, b) + np.dot(b, c) + np.dot(c, a))
    return 2.0 * math.atan2(numerator, denominator)


def _spherical_barycentric_weights(
    p: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> np.ndarray:
    """Normalized spherical-area coordinates for ``p`` inside ``(a,b,c)``."""

    total = _spherical_triangle_area(a, b, c)
    if not math.isfinite(total) or total <= 1.0e-15:
        raise ValueError("Cannot interpolate on a degenerate spherical triangle.")
    weights = np.asarray(
        [
            _spherical_triangle_area(p, b, c),
            _spherical_triangle_area(p, c, a),
            _spherical_triangle_area(p, a, b),
        ],
        dtype=np.float64,
    )
    weights = np.clip(weights, 0.0, None)
    weight_sum = float(weights.sum())
    if not math.isfinite(weight_sum) or weight_sum <= 0.0:
        raise ValueError("Spherical barycentric weights are not finite and positive.")
    return weights / weight_sum


def _locate_containing_faces(
    points: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """Return the three containing triangulation vertices for every point."""

    points = np.asarray(points, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    vertex_tree = cKDTree(vertices)
    face_centers = vertices[faces].sum(axis=1)
    face_centers /= np.linalg.norm(face_centers, axis=1, keepdims=True)
    face_tree = cKDTree(face_centers)
    incident: list[list[int]] = [[] for _ in range(vertices.shape[0])]
    for face_index, (a, b, c) in enumerate(faces):
        incident[int(a)].append(face_index)
        incident[int(b)].append(face_index)
        incident[int(c)].append(face_index)

    containing = np.empty((points.shape[0], 3), dtype=np.int64)
    for point_index, p in enumerate(points):
        triangle = None
        seen_faces: set[int] = set()
        nearest_vertices = np.atleast_1d(
            vertex_tree.query(p, k=min(8, vertices.shape[0]))[1]
        )
        for nearest_vertex in nearest_vertices:
            for face_index in incident[int(nearest_vertex)]:
                if face_index in seen_faces:
                    continue
                seen_faces.add(face_index)
                a, b, c = faces[face_index]
                if _point_in_spherical_triangle(
                    p,
                    vertices[a],
                    vertices[b],
                    vertices[c],
                ):
                    triangle = faces[face_index]
                    break
            if triangle is not None:
                break
        if triangle is None:
            candidate_faces = np.atleast_1d(
                face_tree.query(p, k=min(64, faces.shape[0]))[1]
            )
            for face_index in candidate_faces:
                a, b, c = faces[int(face_index)]
                if _point_in_spherical_triangle(
                    p,
                    vertices[a],
                    vertices[b],
                    vertices[c],
                ):
                    triangle = faces[int(face_index)]
                    break
        if triangle is None:
            raise RuntimeError(
                f"Could not locate a containing spherical triangle for point {point_index}."
            )
        containing[point_index] = triangle
    return containing


def _structured_grid_triangulation(
    grid_ll: np.ndarray,
    grid_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Triangulate a periodic cell-centred latitude-longitude grid.

    The two missing poles are represented by synthetic vertices. Their field
    values are defined as the longitude mean of the nearest grid ring when the
    interpolation matrix is expanded back onto real grid nodes.
    """

    height, width = int(grid_shape[0]), int(grid_shape[1])
    ll = np.asarray(grid_ll, dtype=np.float64).reshape(height, width, 2)
    latitudes = ll[:, 0, 0]
    if not np.all(np.diff(latitudes) > 0.0):
        raise ValueError(
            "Fixed spherical remapping requires grid latitudes in strictly "
            "south-to-north order."
        )
    if not np.allclose(ll[:, :, 0], latitudes[:, None], atol=2.0e-6, rtol=0.0):
        raise ValueError("Grid latitude must be constant across every longitude row.")
    longitudes = np.unwrap(ll[0, :, 1])
    if not np.allclose(
        np.unwrap(ll[:, :, 1], axis=1),
        longitudes[None, :],
        atol=2.0e-6,
        rtol=0.0,
    ):
        raise ValueError("All grid rows must share the same longitude coordinates.")
    longitude_steps = np.diff(longitudes)
    if not np.all(longitude_steps > 0.0):
        raise ValueError("Grid longitudes must be strictly increasing and periodic.")
    expected_step = 2.0 * math.pi / width
    if not np.allclose(
        longitude_steps,
        expected_step,
        atol=2.0e-6,
        rtol=0.0,
    ) or not math.isclose(
        float(longitudes[0] + 2.0 * math.pi - longitudes[-1]),
        expected_step,
        rel_tol=0.0,
        abs_tol=2.0e-6,
    ):
        raise ValueError(
            "Fixed spherical remapping requires an evenly spaced periodic longitude axis."
        )

    grid_xyz = _latlon_to_xyz(np.asarray(grid_ll, dtype=np.float64))
    south_pole = height * width
    north_pole = south_pole + 1
    vertices = np.concatenate(
        [
            grid_xyz,
            np.asarray([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]], dtype=np.float64),
        ],
        axis=0,
    )
    faces: list[tuple[int, int, int]] = []
    for column in range(width):
        next_column = (column + 1) % width
        faces.append((south_pole, next_column, column))
    for row in range(height - 1):
        south_offset = row * width
        north_offset = (row + 1) * width
        for column in range(width):
            next_column = (column + 1) % width
            southwest = south_offset + column
            southeast = south_offset + next_column
            northwest = north_offset + column
            northeast = north_offset + next_column
            faces.append((southwest, southeast, northeast))
            faces.append((southwest, northeast, northwest))
    north_offset = (height - 1) * width
    for column in range(width):
        next_column = (column + 1) % width
        faces.append((north_pole, north_offset + column, north_offset + next_column))
    return vertices, np.asarray(faces, dtype=np.int64), south_pole, north_pole


def _fixed_grid_to_mesh_edges(
    grid_ll: np.ndarray,
    grid_shape: tuple[int, int],
    mesh_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build raw-channel grid->mesh spherical interpolation weights."""

    height, width = int(grid_shape[0]), int(grid_shape[1])
    vertices, faces, south_pole, north_pole = _structured_grid_triangulation(
        grid_ll,
        grid_shape,
    )
    triangles = _locate_containing_faces(mesh_xyz, vertices, faces)
    source: list[int] = []
    destination: list[int] = []
    edge_weight: list[float] = []
    south_ring = range(0, width)
    north_ring = range((height - 1) * width, height * width)

    for mesh_index, triangle in enumerate(triangles):
        a, b, c = (int(value) for value in triangle)
        weights = _spherical_barycentric_weights(
            mesh_xyz[mesh_index],
            vertices[a],
            vertices[b],
            vertices[c],
        )
        contributions: dict[int, float] = {}
        for vertex, weight in zip((a, b, c), weights):
            if vertex == south_pole:
                share = float(weight) / width
                for grid_index in south_ring:
                    contributions[grid_index] = contributions.get(grid_index, 0.0) + share
            elif vertex == north_pole:
                share = float(weight) / width
                for grid_index in north_ring:
                    contributions[grid_index] = contributions.get(grid_index, 0.0) + share
            else:
                contributions[vertex] = contributions.get(vertex, 0.0) + float(weight)
        contribution_sum = float(sum(contributions.values()))
        if not math.isfinite(contribution_sum) or contribution_sum <= 0.0:
            raise RuntimeError(
                f"Invalid fixed grid->mesh weights for mesh node {mesh_index}."
            )
        for grid_index in sorted(contributions):
            normalized_weight = contributions[grid_index] / contribution_sum
            if normalized_weight <= 1.0e-14:
                continue
            source.append(grid_index)
            destination.append(mesh_index)
            edge_weight.append(normalized_weight)

    edge_index = np.stack(
        [np.asarray(source, dtype=np.int64), np.asarray(destination, dtype=np.int64)],
        axis=0,
    )
    return edge_index, np.asarray(edge_weight, dtype=np.float32)


def _fixed_mesh_to_grid_weights(
    grid_xyz: np.ndarray,
    mesh_xyz: np.ndarray,
    mesh_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build raw-channel mesh->grid spherical interpolation weights."""

    triangles = _locate_containing_faces(grid_xyz, mesh_xyz, mesh_faces)
    destination = np.repeat(
        np.arange(grid_xyz.shape[0], dtype=np.int64),
        3,
    )
    edge_index = np.stack([triangles.reshape(-1), destination], axis=0)
    weights = np.empty((grid_xyz.shape[0], 3), dtype=np.float64)
    for grid_index, (a, b, c) in enumerate(triangles):
        weights[grid_index] = _spherical_barycentric_weights(
            grid_xyz[grid_index],
            mesh_xyz[a],
            mesh_xyz[b],
            mesh_xyz[c],
        )
    return edge_index, weights.reshape(-1).astype(np.float32)


def mesh_graph_path(resolution_mode: str, refinement: int, num_graph_levels: int) -> str:
    return (
        f"graphs/graph_{str(resolution_mode)}_icosphere_r{int(refinement)}_"
        f"l{int(num_graph_levels) - 1}.pt"
    )


def expected_mesh_metadata(
    *,
    refinement: int,
    num_graph_levels: int,
    grid_shape: tuple[int, int],
    grid_lat_lon: np.ndarray | torch.Tensor,
    g2m_radius_factor: float,
    resolution_mode: str | None,
    bipartite_mapping_type: str = GRAPHCAST_RADIUS_MAPPING,
    bipartite_edge_features: str = LEGACY_BIPARTITE_EDGE_FEATURES,
    coarse_level_connectivity: str = NATIVE_COARSE_CONNECTIVITY,
    grid_attention_k_neighbors: int | None = None,
    grid_attention_connectivity_strategy: str = "hybrid_row_aware_knn",
    grid_attention_row_aware_knn: dict[str, Any] | None = None,
    grid_attention_reference_graph_path: str | None = None,
) -> dict[str, Any]:
    grid_ll = torch.as_tensor(grid_lat_lon, dtype=torch.float32).reshape(-1, 2)
    coarse_level_connectivity = normalize_coarse_level_connectivity(
        coarse_level_connectivity
    )
    bipartite_mapping_type = normalize_bipartite_mapping_type(
        bipartite_mapping_type
    )
    grid_attention_enabled = grid_attention_k_neighbors is not None
    metadata = {
        "graph_mode": "mesh",
        "mesh_format_version": MESH_FORMAT_VERSION,
        "resolution_mode": resolution_mode,
        "refinement": int(refinement),
        "num_graph_levels": int(num_graph_levels),
        "grid_shape": [int(grid_shape[0]), int(grid_shape[1])],
        "grid_coordinate_hash": coordinate_hash_from_lat_lon(grid_ll),
        "g2m_radius_factor": float(g2m_radius_factor),
        "bipartite_mapping_type": bipartite_mapping_type,
        "coarse_level_connectivity": coarse_level_connectivity,
        "bipartite_edge_features": str(bipartite_edge_features),
        "bipartite_edge_dim": bipartite_edge_feature_dim(bipartite_edge_features),
        "grid_attention_graph": bool(grid_attention_enabled),
    }
    if bipartite_mapping_type == FIXED_SPHERICAL_MAPPING:
        metadata.update(
            {
                "fixed_spherical_method": FIXED_SPHERICAL_METHOD,
                "fixed_spherical_pole_handling": FIXED_SPHERICAL_POLE_HANDLING,
                "fixed_spherical_channel_policy": "shared_weights_no_channel_mixing",
            }
        )
    if grid_attention_enabled:
        metadata.update(
            {
                "grid_attention_k_neighbors": int(grid_attention_k_neighbors),
                "grid_attention_connectivity_strategy": normalize_connectivity_strategy(
                    grid_attention_connectivity_strategy
                ),
                "grid_attention_row_aware_knn": normalize_row_aware_config(
                    grid_attention_row_aware_knn
                ),
            }
        )
        if grid_attention_reference_graph_path is not None:
            _, provenance = _load_grid_attention_reference(
                grid_attention_reference_graph_path,
                grid_shape=grid_shape,
                grid_lat_lon=grid_ll,
                k_neighbors=int(grid_attention_k_neighbors),
                connectivity_strategy=grid_attention_connectivity_strategy,
            )
            metadata.update(provenance)
    return metadata


def validate_mesh_cache_metadata(
    bundle: dict[str, Any],
    expected: dict[str, Any],
) -> list[str]:
    actual = dict(bundle.get("metadata", {}) or {})
    mismatches: list[str] = []
    for key in (
        "graph_mode",
        "mesh_format_version",
        "resolution_mode",
        "refinement",
        "num_graph_levels",
        "grid_shape",
        "grid_coordinate_hash",
        "bipartite_mapping_type",
        "coarse_level_connectivity",
    ):
        actual_value = actual.get(
            key,
            GRAPHCAST_RADIUS_MAPPING if key == "bipartite_mapping_type" else None,
        )
        if actual_value != expected.get(key):
            mismatches.append(f"{key}: cache={actual_value!r}, expected={expected.get(key)!r}")
    if expected.get("bipartite_mapping_type") == FIXED_SPHERICAL_MAPPING:
        for key in (
            "fixed_spherical_method",
            "fixed_spherical_pole_handling",
            "fixed_spherical_channel_policy",
        ):
            if actual.get(key) != expected.get(key):
                mismatches.append(
                    f"{key}: cache={actual.get(key)!r}, expected={expected.get(key)!r}"
                )
    actual_edge_features = actual.get(
        "bipartite_edge_features",
        LEGACY_BIPARTITE_EDGE_FEATURES,
    )
    if actual_edge_features != expected.get("bipartite_edge_features"):
        mismatches.append(
            "bipartite_edge_features: "
            f"cache={actual_edge_features!r}, expected={expected.get('bipartite_edge_features')!r}"
        )
    actual_edge_dim = actual.get("bipartite_edge_dim", None)
    if actual_edge_dim is None:
        try:
            actual_edge_dim = int(bundle["g2m"]["edge_attr"].shape[1])
        except (KeyError, TypeError, AttributeError, IndexError):
            actual_edge_dim = None
    if actual_edge_dim != expected.get("bipartite_edge_dim"):
        mismatches.append(
            "bipartite_edge_dim: "
            f"cache={actual_edge_dim!r}, expected={expected.get('bipartite_edge_dim')!r}"
        )
    actual_grid_attention = bool(
        actual.get(
            "grid_attention_graph",
            isinstance(bundle.get("grid", None), dict)
            and "attention_level" in bundle["grid"],
        )
    )
    expected_grid_attention = bool(expected.get("grid_attention_graph", False))
    if actual_grid_attention != expected_grid_attention:
        mismatches.append(
            "grid_attention_graph: "
            f"cache={actual_grid_attention!r}, expected={expected_grid_attention!r}"
        )
    if expected_grid_attention:
        for key in (
            "grid_attention_k_neighbors",
            "grid_attention_connectivity_strategy",
            "grid_attention_row_aware_knn",
            "grid_attention_reference_graph",
            "grid_attention_reference_sha256",
            "grid_attention_reference_coordinate_hash",
        ):
            if actual.get(key) != expected.get(key):
                mismatches.append(
                    f"{key}: cache={actual.get(key)!r}, expected={expected.get(key)!r}"
                )
    if expected.get("bipartite_mapping_type") == GRAPHCAST_RADIUS_MAPPING:
        if not math.isclose(
            float(actual.get("g2m_radius_factor", float("nan"))),
            float(expected["g2m_radius_factor"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            mismatches.append(
                "g2m_radius_factor: "
                f"cache={actual.get('g2m_radius_factor')!r}, expected={expected['g2m_radius_factor']!r}"
            )
    return mismatches


# --------------------------------------------------------------------------- #
# bundle
# --------------------------------------------------------------------------- #
def build_mesh_bundle(
    refinement: int,
    num_graph_levels: int,
    grid_shape: tuple[int, int],
    grid_lat_lon: np.ndarray | torch.Tensor | None = None,
    g2m_radius_factor: float = 0.6,
    resolution_mode: str | None = None,
    bipartite_mapping_type: str = GRAPHCAST_RADIUS_MAPPING,
    bipartite_edge_features: str = LEGACY_BIPARTITE_EDGE_FEATURES,
    coarse_level_connectivity: str = NATIVE_COARSE_CONNECTIVITY,
    grid_attention_k_neighbors: int | None = None,
    grid_attention_connectivity_strategy: str = "hybrid_row_aware_knn",
    grid_attention_row_aware_knn: dict[str, Any] | None = None,
    grid_attention_reference_graph_path: str | None = None,
) -> dict[str, Any]:
    r = int(refinement)
    L = int(num_graph_levels)
    if L < 3 or L > 5:
        raise ValueError(f"num_graph_levels must be 3, 4, or 5; got {L}.")
    if r - (L - 1) < 0:
        raise ValueError(f"refinement={r} too small for {L} levels (need >= {L - 1}).")
    coarse_level_connectivity = normalize_coarse_level_connectivity(
        coarse_level_connectivity
    )
    bipartite_mapping_type = normalize_bipartite_mapping_type(
        bipartite_mapping_type
    )
    coarsest_refinement = r - (L - 1)
    if (
        coarse_level_connectivity == FULL_M1_COARSE_CONNECTIVITY
        and coarsest_refinement != 1
    ):
        raise ValueError(
            "coarse_level_connectivity='full_m1' requires the coarsest level "
            f"to be M1, got M{coarsest_refinement} "
            f"(refinement={r}, num_graph_levels={L})."
        )
    H, W = int(grid_shape[0]), int(grid_shape[1])

    hierarchy = _hierarchy(r)  # M0..Mr

    # U-Net levels: L0 = finest (M_r), ..., L{L-1} = M_{r-L+1}
    levels: dict[str, Any] = {}
    pool: dict[str, Any] = {}
    node_counts = []
    for i in range(L):
        m = r - i
        level_connectivity = (
            coarse_level_connectivity
            if i == L - 1
            else NATIVE_COARSE_CONNECTIVITY
        )
        lvl = _level_dict(
            hierarchy[m]["verts"],
            hierarchy[m]["faces"],
            connectivity=level_connectivity,
        )
        levels[f"L{i}"] = lvl
        node_counts.append(lvl["num_nodes"])
        if i < L - 1:
            # parent map of M_m (fine) -> M_{m-1} (coarse) = hierarchy[m]["parent"]
            pool[f"L{i}_to_L{i + 1}"] = torch.from_numpy(hierarchy[m]["parent"])

    # finest mesh (= L0) geometry for the bipartite edges
    mesh_verts = hierarchy[r]["verts"]
    mesh_faces = hierarchy[r]["faces"]
    mesh_xyz = mesh_verts.astype(np.float64)
    mesh_ll = _xyz_to_latlon(mesh_verts)
    M = mesh_verts.shape[0]

    # grid geometry (row-major); prefer the authoritative grid lat/lon from the caller
    if grid_lat_lon is None:
        grid_ll = _grid_latlon(H, W)
    else:
        grid_ll = (grid_lat_lon.detach().cpu().numpy() if isinstance(grid_lat_lon, torch.Tensor)
                   else np.asarray(grid_lat_lon)).astype(np.float32)
        if grid_ll.shape != (H * W, 2):
            raise ValueError(f"grid_lat_lon must be [{H * W}, 2], got {grid_ll.shape}.")
    if not np.isfinite(grid_ll).all():
        raise ValueError("grid_lat_lon contains non-finite values.")
    grid_xyz = _latlon_to_xyz(grid_ll)
    grid_attention_level = None
    grid_attention_provenance: dict[str, Any] = {}
    if grid_attention_k_neighbors is not None:
        if grid_attention_reference_graph_path is not None:
            grid_attention_level, grid_attention_provenance = (
                _load_grid_attention_reference(
                    grid_attention_reference_graph_path,
                    grid_shape=(H, W),
                    grid_lat_lon=grid_ll,
                    k_neighbors=int(grid_attention_k_neighbors),
                    connectivity_strategy=grid_attention_connectivity_strategy,
                )
            )
        else:
            grid_attention_level = make_level(
                torch.from_numpy(grid_xyz.astype(np.float32)),
                torch.from_numpy(grid_ll),
                H,
                W,
                int(grid_attention_k_neighbors),
                connectivity_strategy=grid_attention_connectivity_strategy,
                row_aware_knn=grid_attention_row_aware_knn,
            )

    # GraphCast radius: factor times the maximum native edge length on the
    # finest mesh. Chord distance is equivalent for the cKDTree radius query.
    finest_native_edges = _native_edges(mesh_faces)
    a = mesh_xyz[finest_native_edges[0]]
    b = mesh_xyz[finest_native_edges[1]]
    edge_gc = np.arccos(np.clip(np.sum(a * b, axis=1), -1.0, 1.0))
    finest_edge = float(edge_gc.max())
    radius_gc = float(g2m_radius_factor) * finest_edge
    chord = 2.0 * math.sin(radius_gc / 2.0)

    g2m_weight = None
    if bipartite_mapping_type == FIXED_SPHERICAL_MAPPING:
        # Every raw channel uses this same parameter-free interpolation matrix.
        g2m_ei, g2m_weight = _fixed_grid_to_mesh_edges(
            grid_ll,
            (H, W),
            mesh_xyz,
        )
    else:
        # GraphCast radius rule; degree is intentionally variable.
        grid_tree = cKDTree(grid_xyz)
        g2m_src, g2m_dst = [], []
        for mi in range(M):
            idxs = grid_tree.query_ball_point(mesh_xyz[mi], chord)
            for gi in idxs:
                g2m_src.append(int(gi))
                g2m_dst.append(mi)
        if not g2m_src:
            raise ValueError(
                f"grid->mesh radius rule produced no edges (factor={g2m_radius_factor}, radius={radius_gc})."
            )
        g2m_ei = np.stack(
            [
                np.asarray(g2m_src, dtype=np.int64),
                np.asarray(g2m_dst, dtype=np.int64),
            ],
            axis=0,
        )
    g2m_ea = _bipartite_edge_attr(
        grid_ll,
        mesh_ll,
        g2m_ei,
        feature_set=bipartite_edge_features,
    )

    # --- mesh -> grid: exactly the 3 vertices of the containing native face. ---
    face_array = np.asarray(mesh_faces, dtype=np.int64)
    m2g_weight = None
    if bipartite_mapping_type == FIXED_SPHERICAL_MAPPING:
        m2g_ei, m2g_weight = _fixed_mesh_to_grid_weights(
            grid_xyz,
            mesh_xyz,
            face_array,
        )
    else:
        triangles = _locate_containing_faces(grid_xyz, mesh_xyz, face_array)
        m2g_ei = np.stack(
            [
                triangles.reshape(-1),
                np.repeat(
                    np.arange(grid_xyz.shape[0], dtype=np.int64),
                    3,
                ),
            ],
            axis=0,
        )
    m2g_ea = _bipartite_edge_attr(
        mesh_ll,
        grid_ll,
        m2g_ei,
        feature_set=bipartite_edge_features,
    )

    # mesh node input features: [sin lat, cos lat, sin lon, cos lon]
    mesh_static = np.stack(
        [np.sin(mesh_ll[:, 0]), np.cos(mesh_ll[:, 0]), np.sin(mesh_ll[:, 1]), np.cos(mesh_ll[:, 1])],
        axis=1,
    ).astype(np.float32)

    grid_ll_tensor = torch.from_numpy(grid_ll)
    metadata = expected_mesh_metadata(
        refinement=r,
        num_graph_levels=L,
        grid_shape=(H, W),
        grid_lat_lon=grid_ll_tensor,
        g2m_radius_factor=float(g2m_radius_factor),
        resolution_mode=resolution_mode,
        bipartite_mapping_type=bipartite_mapping_type,
        bipartite_edge_features=bipartite_edge_features,
        coarse_level_connectivity=coarse_level_connectivity,
        grid_attention_k_neighbors=grid_attention_k_neighbors,
        grid_attention_connectivity_strategy=grid_attention_connectivity_strategy,
        grid_attention_row_aware_knn=grid_attention_row_aware_knn,
        grid_attention_reference_graph_path=grid_attention_reference_graph_path,
    )
    edge_counts = [int(levels[f"L{i}"]["edge_index"].shape[1]) for i in range(L)]
    level_k_neighbors = [int(levels[f"L{i}"]["k"]) for i in range(L)]
    connectivity_strategy = mesh_connectivity_strategy(coarse_level_connectivity)
    metadata.update(
        {
            "hierarchy_type": "icosphere",
            "use_l3": L >= 4,
            "use_l4": L >= 5,
            "mesh_static_dim": int(mesh_static.shape[1]),
            "node_counts": node_counts,
            "edge_counts": edge_counts,
            "face_counts": [int(len(hierarchy[r - i]["faces"])) for i in range(L)],
            "graph_k": _K,
            "k": _K,
            "k_neighbors": _K,
            "level_k_neighbors": level_k_neighbors,
            "connectivity_strategy": connectivity_strategy,
            "graph_connectivity_strategy": connectivity_strategy,
            "pooling_map_strategy": "icosphere_refinement_parent",
            "finest_mesh_max_edge_length_radians": finest_edge,
            "g2m_radius_radians": (
                radius_gc
                if bipartite_mapping_type == GRAPHCAST_RADIUS_MAPPING
                else None
            ),
            "g2m_edge_count": int(g2m_ei.shape[1]),
            "m2g_edge_count": int(m2g_ei.shape[1]),
        }
    )
    if grid_attention_level is not None:
        metadata["grid_attention_edge_count"] = int(
            grid_attention_level["edge_index"].shape[1]
        )
        metadata.update(grid_attention_provenance)
    grid_bundle = {
        "lat_lon": grid_ll_tensor,
        "coords": torch.from_numpy(grid_xyz.astype(np.float32)),
        "height": H,
        "width": W,
    }
    if grid_attention_level is not None:
        grid_bundle["attention_level"] = grid_attention_level
    g2m_bundle = {
        "edge_index": torch.from_numpy(g2m_ei),
        "edge_attr": torch.from_numpy(g2m_ea),
    }
    m2g_bundle = {
        "edge_index": torch.from_numpy(m2g_ei),
        "edge_attr": torch.from_numpy(m2g_ea),
    }
    if g2m_weight is not None:
        g2m_bundle["edge_weight"] = torch.from_numpy(g2m_weight)
    if m2g_weight is not None:
        m2g_bundle["edge_weight"] = torch.from_numpy(m2g_weight)
    return {
        "levels": levels,
        "pool": pool,
        "grid": grid_bundle,
        "g2m": g2m_bundle,
        "m2g": m2g_bundle,
        "mesh_static": torch.from_numpy(mesh_static),
        "metadata": metadata,
    }


def build_and_save(path: str, **kwargs: Any) -> dict[str, Any]:
    bundle = build_mesh_bundle(**kwargs)
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)
    return bundle["metadata"]
