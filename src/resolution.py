from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .architecture import resolve_graph_architecture


RATIO15_L4_2P5_LEVEL_SHAPES: tuple[tuple[int, int], ...] = (
    (72, 144),
    (48, 96),
    (32, 64),
    (21, 42),
    (14, 28),
)

L4_72_36_24_18_9_2P5_LEVEL_SHAPES: tuple[tuple[int, int], ...] = (
    (72, 144),
    (36, 72),
    (24, 48),
    (18, 36),
    (9, 18),
)


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

    def edge_counts(self, k: int | list[int] | tuple[int, ...]) -> tuple[int, ...]:
        if isinstance(k, (list, tuple)):
            if len(k) != len(self.node_counts):
                raise ValueError(f"Expected {len(self.node_counts)} k values, got {len(k)}.")
            return tuple(count * int(k_value) for count, k_value in zip(self.node_counts, k))
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
    "1p5": ResolutionSpec(
        name="1p5",
        resolution_degrees=1.5,
        # The kai_1p5 dataset grid includes both exact poles: latitudes -90..+90
        # inclusive at 1.5 degrees = 121 rows (unlike 2p5/5p625, which are
        # cell-centered and pole-free).
        height=121,
        width=240,
        level_shapes=((121, 240), (61, 120), (31, 60)),
    ),
}

RESOLUTION_ALIASES: dict[str, str] = {
    "5.625": "5p625",
    "5p625": "5p625",
    "5deg625": "5p625",
    "2.5": "2p5",
    "2p5": "2p5",
    "2deg5": "2p5",
    "1.5": "1p5",
    "1p5": "1p5",
    "1deg5": "1p5",
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


def default_graph_path(
    mode: str,
    k: int,
    strategy: str,
    graph_format_version: int | str,
    num_graph_levels: int = 3,
    level_k_neighbors: list[int] | tuple[int, ...] | None = None,
    hierarchy_type: str | None = None,
    use_l4_ratio15: bool = False,
) -> str:
    strategy_stem = str(strategy)
    if strategy_stem == "hybrid_row_aware_knn":
        strategy_stem = "hybrid_row_aware"
    hierarchy = str(hierarchy_type or "standard").strip().lower()
    values = [int(x) for x in (level_k_neighbors or [int(k)] * int(num_graph_levels))]
    if bool(use_l4_ratio15) or hierarchy == "ratio15_l4":
        joined = "_".join(str(value) for value in values)
        return (
            f"graphs/graph_{canonicalize_resolution_mode(mode)}_ratio15_L4_k{joined}_"
            f"{strategy_stem}_v1.pt"
        )
    if hierarchy == "l4_72_36_24_18_9":
        joined = "_".join(str(value) for value in values)
        return (
            f"graphs/graph_{canonicalize_resolution_mode(mode)}_l4_72_36_24_18_9_k{joined}_"
            f"{strategy_stem}_v1.pt"
        )

    level_stem = "_L3" if int(num_graph_levels) == 4 else ""
    k_stem = f"k{int(k)}"
    if level_k_neighbors is not None:
        if len(values) == int(num_graph_levels) and any(value != int(k) for value in values):
            if int(num_graph_levels) == 4 and values[:-1] == [int(k), int(k), int(k)]:
                k_stem = f"k{int(k)}_l3k{values[-1]}"
            else:
                joined = "-".join(str(value) for value in values)
                k_stem = f"k{int(k)}_levelk{joined}"
    return (
        f"graphs/graph_{canonicalize_resolution_mode(mode)}_{k_stem}_"
        f"{strategy_stem}{level_stem}_v{int(graph_format_version)}.pt"
    )


def graph_format_version_for_level_k(
    num_graph_levels: int,
    level_k_neighbors: list[int] | tuple[int, ...] | None = None,
    k_neighbors: int = 8,
    hierarchy_type: str | None = None,
    use_l4_ratio15: bool = False,
) -> int | str:
    levels = int(num_graph_levels)
    hierarchy = str(hierarchy_type or "standard").strip().lower()
    if bool(use_l4_ratio15) or hierarchy == "ratio15_l4":
        return "ratio15_l4_v1"
    if hierarchy == "l4_72_36_24_18_9":
        return "l4_72_36_24_18_9_v1"
    if levels == 4:
        values = [int(x) for x in (level_k_neighbors or [int(k_neighbors)] * levels)]
        if any(value != int(k_neighbors) for value in values):
            return 4
        return 3
    if levels == 5:
        return 5
    return 2


def coarsened_level_shapes(height: int, width: int, num_graph_levels: int) -> list[list[int]]:
    shapes: list[list[int]] = []
    h = int(height)
    w = int(width)
    for _ in range(int(num_graph_levels)):
        shapes.append([h, w])
        h = int(np.ceil(float(h) / 2.0))
        w = int(np.ceil(float(w) / 2.0))
    return shapes


def _shape_list(raw_shapes: Any) -> list[list[int]] | None:
    if raw_shapes is None:
        return None
    shapes = [[int(shape[0]), int(shape[1])] for shape in list(raw_shapes)]
    if not shapes:
        raise ValueError("level_shapes must not be empty.")
    for shape in shapes:
        if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
            raise ValueError(f"Invalid level shape: {shape!r}")
    return shapes


def resolve_level_shapes_for_config(
    params: dict[str, Any],
    mode: str,
    num_graph_levels: int,
    hierarchy_type: str = "standard",
    use_l4_ratio15: bool = False,
) -> list[list[int]]:
    spec = get_resolution_spec(mode)
    model_cfg = params.get("model", {}) if isinstance(params.get("model", {}), dict) else {}
    model_shapes = _shape_list(model_cfg.get("level_shapes", None))
    top_level_shapes = _shape_list(params.get("level_shapes", None))
    raw_shapes = model_shapes if model_shapes is not None else top_level_shapes
    if raw_shapes is not None:
        if len(raw_shapes) != int(num_graph_levels):
            if model_shapes is not None:
                raise ValueError(
                    f"model.level_shapes must have length num_graph_levels={int(num_graph_levels)}, "
                    f"got {len(raw_shapes)}."
                )
            raw_shapes = None
        elif tuple(raw_shapes[0]) != tuple(spec.grid_shape):
            raise ValueError(
                f"level_shapes[0]={raw_shapes[0]} does not match {spec.name} grid shape {list(spec.grid_shape)}."
            )
        else:
            return raw_shapes

    hierarchy = str(hierarchy_type or "standard").strip().lower()
    if bool(use_l4_ratio15) or hierarchy == "ratio15_l4":
        if spec.name != "2p5":
            raise ValueError("hierarchy_type='ratio15_l4' is only defined for resolution_mode='2p5'.")
        if int(num_graph_levels) != 5:
            raise ValueError("hierarchy_type='ratio15_l4' requires num_graph_levels=5.")
        return [[int(h), int(w)] for h, w in RATIO15_L4_2P5_LEVEL_SHAPES]
    if hierarchy == "l4_72_36_24_18_9":
        if spec.name != "2p5":
            raise ValueError("hierarchy_type='l4_72_36_24_18_9' is only defined for resolution_mode='2p5'.")
        if int(num_graph_levels) != 5:
            raise ValueError("hierarchy_type='l4_72_36_24_18_9' requires num_graph_levels=5.")
        return [[int(h), int(w)] for h, w in L4_72_36_24_18_9_2P5_LEVEL_SHAPES]

    return coarsened_level_shapes(spec.height, spec.width, int(num_graph_levels))


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
    graph_path_before_profile = resolved.get("graph_path", None)
    if profile:
        resolved.update(_derive_split_paths(profile))

    arch = resolve_graph_architecture(resolved)
    model_cfg = resolved.get("model", {}) if isinstance(resolved.get("model", {}), dict) else {}
    k = int(model_cfg.get("k_neighbors", resolved.get("k_neighbors", resolved.get("k", 8))))
    level_k_neighbors = [int(x) for x in arch.level_k_neighbors]
    strategy = str(resolved.get("graph_connectivity_strategy", "hybrid_row_aware_knn"))
    graph_format_version = graph_format_version_for_level_k(
        arch.num_graph_levels,
        level_k_neighbors=level_k_neighbors,
        k_neighbors=k,
        hierarchy_type=arch.hierarchy_type,
        use_l4_ratio15=arch.use_l4_ratio15,
    )
    level_shapes = resolve_level_shapes_for_config(
        resolved,
        spec.name,
        arch.num_graph_levels,
        hierarchy_type=arch.hierarchy_type,
        use_l4_ratio15=arch.use_l4_ratio15,
    )
    node_counts = [int(h) * int(w) for h, w in level_shapes]
    resolved["hierarchy_type"] = arch.hierarchy_type
    resolved["use_l4_ratio15"] = bool(arch.use_l4_ratio15)
    resolved["resolution_mode"] = spec.name
    resolved["resolution"] = float(spec.resolution_degrees)
    resolved["resolution_degrees"] = float(spec.resolution_degrees)
    resolved["grid_shape"] = [int(spec.height), int(spec.width)]
    resolved["expected_grid_shape"] = [int(spec.height), int(spec.width)]
    resolved["level_shapes"] = level_shapes
    resolved["node_counts"] = node_counts
    resolved["edge_counts"] = [int(count) * int(level_k) for count, level_k in zip(node_counts, level_k_neighbors)]
    resolved["k_neighbors"] = k
    resolved["level_k_neighbors"] = level_k_neighbors
    resolved["graph_format_version"] = graph_format_version
    if arch.num_graph_levels >= 4:
        resolved["l3_k_neighbors"] = int(level_k_neighbors[3])
    if arch.num_graph_levels >= 5:
        resolved["l4_k_neighbors"] = int(level_k_neighbors[4])
    if graph_path_before_profile is None and (arch.num_graph_levels >= 4 or not resolved.get("graph_path")):
        resolved["graph_path"] = default_graph_path(
            spec.name,
            k,
            strategy,
            graph_format_version=graph_format_version,
            num_graph_levels=arch.num_graph_levels,
            level_k_neighbors=level_k_neighbors,
            hierarchy_type=arch.hierarchy_type,
            use_l4_ratio15=arch.use_l4_ratio15,
        )
    return resolved


def resolution_metadata(
    mode: Any,
    k: int,
    num_parameters: int | None = None,
    num_graph_levels: int = 3,
    level_k_neighbors: list[int] | tuple[int, ...] | None = None,
    level_shapes: list[list[int]] | tuple[tuple[int, int], ...] | None = None,
    hierarchy_type: str = "standard",
    use_l4_ratio15: bool = False,
) -> dict[str, Any]:
    spec = get_resolution_spec(mode)
    resolved_shapes = _shape_list(level_shapes)
    if resolved_shapes is None:
        resolved_shapes = resolve_level_shapes_for_config(
            {},
            spec.name,
            int(num_graph_levels),
            hierarchy_type=hierarchy_type,
            use_l4_ratio15=use_l4_ratio15,
        )
    values = [int(x) for x in (level_k_neighbors or [int(k)] * int(num_graph_levels))]
    node_counts = [int(h) * int(w) for h, w in resolved_shapes]
    metadata: dict[str, Any] = {
        "hierarchy_type": str(hierarchy_type or "standard"),
        "use_l4_ratio15": bool(use_l4_ratio15),
        "resolution_mode": spec.name,
        "resolution_degrees": float(spec.resolution_degrees),
        "grid_shape": [int(spec.height), int(spec.width)],
        "level_shapes": resolved_shapes,
        "graph_k": int(k),
        "level_k_neighbors": values,
        "node_counts": node_counts,
        "edge_counts": [int(count) * int(level_k) for count, level_k in zip(node_counts, values)],
    }
    if num_parameters is not None:
        metadata["num_parameters"] = int(num_parameters)
    return metadata
