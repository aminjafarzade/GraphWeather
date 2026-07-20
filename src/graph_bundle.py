from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn

from .graph_builder import validate_graph_cache_metadata


class GraphLevel(nn.Module):
    def __init__(self, level_dict: dict[str, Any]):
        super().__init__()
        self.height = int(level_dict["height"])
        self.width = int(level_dict["width"])
        self.num_nodes = int(level_dict["num_nodes"])
        self.k = int(level_dict["k"])
        self.register_buffer("coords", level_dict["coords"].to(torch.float32), persistent=False)
        self.register_buffer("lat_lon", level_dict["lat_lon"].to(torch.float32), persistent=False)
        self.register_buffer("edge_index", level_dict["edge_index"].to(torch.long), persistent=False)
        self.register_buffer("edge_attr", level_dict["edge_attr"].to(torch.float32), persistent=False)


class GraphBundle(nn.Module):
    """Graph tensors registered as module buffers so model.to(device) moves them."""

    def __init__(self, bundle: dict[str, Any]):
        super().__init__()
        levels = bundle["levels"]
        self.metadata = dict(bundle.get("metadata", {}))
        self.num_graph_levels = int(self.metadata.get("num_graph_levels", len(levels) or 3))
        if self.num_graph_levels not in {3, 4, 5}:
            raise ValueError(f"GraphBundle only supports 3, 4, or 5 levels, got {self.num_graph_levels}.")
        self.use_l3 = bool(self.metadata.get("use_l3", self.num_graph_levels >= 4))
        self.use_l4 = bool(self.metadata.get("use_l4", self.num_graph_levels >= 5))
        self.use_l4_ratio15 = bool(self.metadata.get("use_l4_ratio15", False))
        self.L0 = GraphLevel(levels["L0"])
        self.L1 = GraphLevel(levels["L1"])
        self.L2 = GraphLevel(levels["L2"])
        if self.num_graph_levels >= 4:
            if "L3" not in levels:
                raise ValueError(f"GraphBundle metadata says num_graph_levels={self.num_graph_levels}, but levels['L3'] is missing.")
            self.L3 = GraphLevel(levels["L3"])
        else:
            self.L3 = None
        if self.num_graph_levels >= 5:
            if "L4" not in levels:
                raise ValueError("GraphBundle metadata says num_graph_levels=5, but levels['L4'] is missing.")
            self.L4 = GraphLevel(levels["L4"])
        else:
            self.L4 = None
        pool = bundle["pool"]
        self.register_buffer("pool_L0_to_L1", pool["L0_to_L1"].to(torch.long), persistent=False)
        self.register_buffer("pool_L1_to_L2", pool["L1_to_L2"].to(torch.long), persistent=False)
        if self.num_graph_levels >= 4:
            if "L2_to_L3" not in pool:
                raise ValueError(f"GraphBundle metadata says num_graph_levels={self.num_graph_levels}, but pool['L2_to_L3'] is missing.")
            self.register_buffer("pool_L2_to_L3", pool["L2_to_L3"].to(torch.long), persistent=False)
        else:
            self.pool_L2_to_L3 = None
        if self.num_graph_levels >= 5:
            if "L3_to_L4" not in pool:
                raise ValueError("GraphBundle metadata says num_graph_levels=5, but pool['L3_to_L4'] is missing.")
            self.register_buffer("pool_L3_to_L4", pool["L3_to_L4"].to(torch.long), persistent=False)
        else:
            self.pool_L3_to_L4 = None

    @property
    def level0(self) -> GraphLevel:
        return self.L0

    @property
    def level1(self) -> GraphLevel:
        return self.L1

    @property
    def level2(self) -> GraphLevel:
        return self.L2

    @property
    def level3(self) -> GraphLevel | None:
        return self.L3

    @property
    def level4(self) -> GraphLevel | None:
        return self.L4

    @property
    def pool_l0_to_l1(self) -> torch.Tensor:
        return self.pool_L0_to_L1

    @property
    def pool_l1_to_l2(self) -> torch.Tensor:
        return self.pool_L1_to_L2

    @property
    def pool_l2_to_l3(self) -> torch.Tensor | None:
        return self.pool_L2_to_L3

    @property
    def pool_l3_to_l4(self) -> torch.Tensor | None:
        return self.pool_L3_to_L4


def load_graph_bundle(
    path: str,
    map_location: str | torch.device = "cpu",
    expected_metadata: dict[str, Any] | None = None,
) -> GraphBundle:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Graph bundle not found: {path}")
    try:
        raw = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        raw = torch.load(path, map_location=map_location)
    if expected_metadata is not None:
        mismatches = validate_graph_cache_metadata(raw, expected_metadata)
        if mismatches:
            details = "\n  ".join(mismatches)
            raise ValueError(f"Graph bundle metadata mismatch for {path}:\n  {details}")
    return GraphBundle(raw)
