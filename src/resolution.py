from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class ResolutionSpec:
    name: str
    resolution_degrees: float
    height: int
    width: int
    level_shapes: tuple[tuple[int, int], ...]

    @property
    def grid_shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def node_counts(self) -> tuple[int, ...]:
        return tuple(height * width for height, width in self.level_shapes)

    def edge_counts(self, k: int) -> tuple[int, ...]:
        return tuple(count * int(k) for count in self.node_counts)


RESOLUTION_SPECS: dict[str, ResolutionSpec] = {
    "5p625": ResolutionSpec(
        name="5p625",
        resolution_degrees=5.625,
        height=32,
        width=64,
        level_shapes=((32, 64), (16, 32), (8, 16)),
    ),
    "2p5": ResolutionSpec(
        name="2p5",
        resolution_degrees=2.5,
        height=72,
        width=144,
        level_shapes=((72, 144), (36, 72), (18, 36)),
    ),
}

RESOLUTION_ALIASES: dict[str, str] = {
    "5.625": "5p625",
    "5p625": "5p625",
    "5deg625": "5p625",
    "2.5": "2p5",
    "2p5": "2p5",
    "2deg5": "2p5",
}


def canonicalize_resolution_mode(mode: Any | None) -> str:
    if mode is None or str(mode).strip() == "":
        return "5p625"
    key = str(mode).strip().lower().replace("_", "").replace("-", "")
    if key not in RESOLUTION_ALIASES:
        available = ", ".join(sorted(RESOLUTION_ALIASES))
        raise ValueError(f"Unsupported resolution_mode '{mode}'. Supported aliases: {available}")
    return RESOLUTION_ALIASES[key]


def get_resolution_spec(mode: Any | None) -> ResolutionSpec:
    return RESOLUTION_SPECS[canonicalize_resolution_mode(mode)]


def cell_center_lat_lon(height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    height = int(height)
    width = int(width)
    dlat = 180.0 / float(height)
    dlon = 360.0 / float(width)
    latitudes = np.linspace(-90.0 + dlat / 2.0, 90.0 - dlat / 2.0, height, dtype=np.float32)
    longitudes = np.linspace(0.0, 360.0 - dlon, width, dtype=np.float32)
    return torch.as_tensor(latitudes), torch.as_tensor(longitudes)


def default_graph_path(mode: str, k: int, strategy: str, graph_format_version: int) -> str:
    strategy_stem = str(strategy)
    if strategy_stem == "hybrid_row_aware_knn":
        strategy_stem = "hybrid_row_aware"
    return f"graphs/graph_{canonicalize_resolution_mode(mode)}_k{int(k)}_{strategy_stem}_v{int(graph_format_version)}.pt"


def _derive_split_paths(profile: dict[str, Any]) -> dict[str, Any]:
    merged = dict(profile)
    data_root = merged.get("data_root", None)
    if data_root:
        root = str(data_root)
        merged.setdefault("train_data_path", os.path.join(root, "train"))
        merged.setdefault("valid_data_path", os.path.join(root, "valid"))
        merged.setdefault("test_dataset_path", os.path.join(root, "test"))
    normalization_stats_path = merged.pop("normalization_stats_path", None)
    if normalization_stats_path:
        stats_root = Path(str(normalization_stats_path))
        merged.setdefault("global_means_path", str(stats_root.with_name("global_mean.npy")))
        merged.setdefault("global_stds_path", str(stats_root.with_name("global_std.npy")))
    return merged


def apply_resolution_profile(params: dict[str, Any], cli_resolution_mode: Any | None = None) -> dict[str, Any]:
    """Resolve the active resolution and overlay mode-specific profile values.

    Precedence is explicit CLI mode, YAML resolution_mode, then the 5p625 default.
    The returned dict is a copy; the input is not mutated.
    """
    resolved = copy.deepcopy(dict(params))
    mode_source = cli_resolution_mode if cli_resolution_mode is not None else resolved.get("resolution_mode", None)
    mode = canonicalize_resolution_mode(mode_source)
    spec = get_resolution_spec(mode)
    profiles = resolved.get("resolution_profiles", {}) or {}
    profile = profiles.get(mode, {}) if isinstance(profiles, dict) else {}
    if profile:
        resolved.update(_derive_split_paths(profile))

    k = int(resolved.get("k_neighbors", resolved.get("k", 8)))
    strategy = str(resolved.get("graph_connectivity_strategy", "hybrid_row_aware_knn"))
    resolved["resolution_mode"] = spec.name
    resolved["resolution"] = float(spec.resolution_degrees)
    resolved["resolution_degrees"] = float(spec.resolution_degrees)
    resolved["grid_shape"] = [int(spec.height), int(spec.width)]
    resolved["expected_grid_shape"] = [int(spec.height), int(spec.width)]
    resolved["level_shapes"] = [[int(h), int(w)] for h, w in spec.level_shapes]
    resolved["node_counts"] = [int(x) for x in spec.node_counts]
    resolved["edge_counts"] = [int(x) for x in spec.edge_counts(k)]
    resolved["k_neighbors"] = k
    if not resolved.get("graph_path"):
        resolved["graph_path"] = default_graph_path(spec.name, k, strategy, graph_format_version=2)
    return resolved


def resolution_metadata(mode: Any, k: int, num_parameters: int | None = None) -> dict[str, Any]:
    spec = get_resolution_spec(mode)
    metadata: dict[str, Any] = {
        "resolution_mode": spec.name,
        "resolution_degrees": float(spec.resolution_degrees),
        "grid_shape": [int(spec.height), int(spec.width)],
        "level_shapes": [[int(h), int(w)] for h, w in spec.level_shapes],
        "graph_k": int(k),
    }
    if num_parameters is not None:
        metadata["num_parameters"] = int(num_parameters)
    return metadata
