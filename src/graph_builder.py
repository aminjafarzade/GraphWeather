from __future__ import annotations

import argparse
import glob
import math
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


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


def spherical_knn_graph(coords: torch.Tensor, lat_lon: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    num_nodes = coords.shape[0]
    if k >= num_nodes:
        raise ValueError(f"k={k} must be smaller than num_nodes={num_nodes}")
    similarity = coords @ coords.T
    similarity.fill_diagonal_(-float("inf"))
    neighbors = torch.topk(similarity, k=k, dim=1).indices.to(torch.long)
    target = torch.arange(num_nodes, dtype=torch.long).repeat_interleave(k)
    source = neighbors.reshape(-1)
    edge_index = torch.stack([source, target], dim=0)
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
) -> dict[str, torch.Tensor | int]:
    edge_index, edge_attr = spherical_knn_graph(coords, lat_lon, k)
    return {
        "height": int(height),
        "width": int(width),
        "num_nodes": int(coords.shape[0]),
        "k": int(k),
        "coords": coords.to(torch.float32),
        "lat_lon": lat_lon.to(torch.float32),
        "edge_index": edge_index.to(torch.long),
        "edge_attr": edge_attr.to(torch.float32),
    }


def build_graph_bundle(
    latitudes_deg: torch.Tensor,
    longitudes_deg: torch.Tensor,
    k: int = 8,
    resolution: float = 5.625,
) -> dict[str, object]:
    height0 = int(latitudes_deg.numel())
    width0 = int(longitudes_deg.numel())
    coords0, latlon0 = grid_nodes(latitudes_deg, longitudes_deg)
    pool01, height1, width1 = make_pool_map(height0, width0)
    coords1, latlon1 = coarsen_coords(coords0, pool01, height1 * width1)
    pool12, height2, width2 = make_pool_map(height1, width1)
    coords2, latlon2 = coarsen_coords(coords1, pool12, height2 * width2)

    return {
        "metadata": {
            "resolution": float(resolution),
            "lat_count": height0,
            "lon_count": width0,
            "k": int(k),
            "node_order": "lat_index * num_lon + lon_index",
            "levels": {
                "L0": [height0, width0],
                "L1": [height1, width1],
                "L2": [height2, width2],
            },
        },
        "levels": {
            "L0": make_level(coords0, latlon0, height0, width0, k),
            "L1": make_level(coords1, latlon1, height1, width1, k),
            "L2": make_level(coords2, latlon2, height2, width2, k),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--data", default=None, help="NetCDF file or directory with latitude/longitude coordinates")
    parser.add_argument("--resolution", type=float, default=5.625)
    parser.add_argument("--lat-count", type=int, default=32)
    parser.add_argument("--lon-count", type=int, default=64)
    parser.add_argument("--lat-start", type=float, default=None)
    parser.add_argument("--lon-start", type=float, default=-180.0)
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()

    if args.data:
        latitudes, longitudes = lat_lon_from_netcdf(args.data)
    else:
        latitudes, longitudes = regular_lat_lon(
            lat_count=args.lat_count,
            lon_count=args.lon_count,
            resolution=args.resolution,
            lat_start=args.lat_start,
            lon_start=args.lon_start,
        )

    bundle = build_graph_bundle(latitudes, longitudes, k=args.k, resolution=args.resolution)
    save_graph(bundle, args.output)
    meta = bundle["metadata"]
    print(f"saved {args.output}")
    print(f"L0/L1/L2: {meta['levels']}")


if __name__ == "__main__":
    main()

