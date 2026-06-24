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
        self.L0 = GraphLevel(levels["L0"])
        self.L1 = GraphLevel(levels["L1"])
        self.L2 = GraphLevel(levels["L2"])
        self.metadata = dict(bundle.get("metadata", {}))
        pool = bundle["pool"]
        self.register_buffer("pool_L0_to_L1", pool["L0_to_L1"].to(torch.long), persistent=False)
        self.register_buffer("pool_L1_to_L2", pool["L1_to_L2"].to(torch.long), persistent=False)


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
