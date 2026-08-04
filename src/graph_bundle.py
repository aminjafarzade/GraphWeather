from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn

from .graph_builder import validate_graph_cache_metadata


class GraphLevel(nn.Module):
    def __init__(self, level_dict: dict[str, Any]):
        super().__init__()
        # height/width are informational; mesh (icosphere) levels have no grid shape,
        # so num_nodes is authoritative and height/width default to 0.
        self.height = int(level_dict.get("height", 0))
        self.width = int(level_dict.get("width", 0))
        self.num_nodes = int(level_dict["num_nodes"])
        self.k = int(level_dict["k"])
        self.register_buffer("coords", level_dict["coords"].to(torch.float32), persistent=False)
        self.register_buffer("lat_lon", level_dict["lat_lon"].to(torch.float32), persistent=False)
        self.register_buffer("edge_index", level_dict["edge_index"].to(torch.long), persistent=False)
        self.register_buffer("edge_attr", level_dict["edge_attr"].to(torch.float32), persistent=False)
        # Optional per-edge validity mask (bool, [num_nodes*k] or [num_nodes,k]).
        # Present only for mesh levels (padded icosphere degree-5 nodes). None on
        # lat-lon levels -> LocalGraphAttention's masked_fill is skipped entirely.
        edge_mask = level_dict.get("edge_mask", None)
        if edge_mask is not None:
            edge_mask = edge_mask.to(torch.bool)
            if int(edge_mask.numel()) != self.num_nodes * self.k:
                raise ValueError(
                    f"edge_mask must contain num_nodes*k={self.num_nodes * self.k} entries, "
                    f"got shape {tuple(edge_mask.shape)}."
                )
            if not bool(edge_mask.reshape(self.num_nodes, self.k).any(dim=1).all()):
                raise ValueError("Every graph node must have at least one valid incoming edge.")
            self.register_buffer("edge_mask", edge_mask, persistent=False)
        else:
            self.edge_mask = None


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

        # --- mesh (icosphere) mode: the data grid + bipartite grid<->mesh edges ---
        # Present only when metadata.graph_mode == "mesh"; grid bundles are unchanged.
        self.graph_mode = str(self.metadata.get("graph_mode", "grid"))
        if self.graph_mode == "mesh":
            grid = bundle["grid"]
            self.grid_height = int(grid.get("height", 0))
            self.grid_width = int(grid.get("width", 0))
            grid_lat_lon = grid["lat_lon"].to(torch.float32)
            self.register_buffer("grid_lat_lon", grid_lat_lon, persistent=False)
            self.register_buffer(
                "grid_static",
                torch.stack(
                    [
                        torch.sin(grid_lat_lon[:, 0]),
                        torch.cos(grid_lat_lon[:, 0]),
                        torch.sin(grid_lat_lon[:, 1]),
                        torch.cos(grid_lat_lon[:, 1]),
                    ],
                    dim=1,
                ),
                persistent=False,
            )
            grid_coords = grid.get("coords", None)
            if grid_coords is not None:
                self.register_buffer("grid_coords", grid_coords.to(torch.float32), persistent=False)
            else:
                self.grid_coords = None
            grid_attention_level = grid.get("attention_level", None)
            if grid_attention_level is not None:
                self.grid_attention_graph = GraphLevel(grid_attention_level)
                if (
                    self.grid_attention_graph.height != self.grid_height
                    or self.grid_attention_graph.width != self.grid_width
                    or self.grid_attention_graph.num_nodes != self.grid_height * self.grid_width
                ):
                    raise ValueError(
                        "Mesh bundle grid attention graph does not match its data grid: "
                        f"level={(self.grid_attention_graph.height, self.grid_attention_graph.width, self.grid_attention_graph.num_nodes)}, "
                        f"grid={(self.grid_height, self.grid_width, self.grid_height * self.grid_width)}."
                    )
            else:
                self.grid_attention_graph = None
            g2m, m2g = bundle["g2m"], bundle["m2g"]
            self.register_buffer("g2m_edge_index", g2m["edge_index"].to(torch.long), persistent=False)
            self.register_buffer("g2m_edge_attr", g2m["edge_attr"].to(torch.float32), persistent=False)
            g2m_edge_weight = g2m.get("edge_weight", None)
            if g2m_edge_weight is not None:
                self.register_buffer(
                    "g2m_edge_weight",
                    g2m_edge_weight.to(torch.float32),
                    persistent=False,
                )
            else:
                self.g2m_edge_weight = None
            self.register_buffer("m2g_edge_index", m2g["edge_index"].to(torch.long), persistent=False)
            self.register_buffer("m2g_edge_attr", m2g["edge_attr"].to(torch.float32), persistent=False)
            m2g_edge_weight = m2g.get("edge_weight", None)
            if m2g_edge_weight is not None:
                self.register_buffer(
                    "m2g_edge_weight",
                    m2g_edge_weight.to(torch.float32),
                    persistent=False,
                )
            else:
                self.m2g_edge_weight = None
            self.register_buffer("mesh_static", bundle["mesh_static"].to(torch.float32), persistent=False)
        else:
            self.grid_height = 0
            self.grid_width = 0
            self.grid_attention_graph = None
            self.g2m_edge_weight = None
            self.m2g_edge_weight = None

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
    raw_meta = raw.get("metadata", {}) if isinstance(raw, dict) else {}
    # Mesh (icosphere) bundles don't share the lat-lon grid's node/edge counts, so the
    # grid-oriented cache validation doesn't apply; the mesh forward validates its own shapes.
    if expected_metadata is not None and str(raw_meta.get("graph_mode", "grid")) != "mesh":
        mismatches = validate_graph_cache_metadata(raw, expected_metadata)
        if mismatches:
            details = "\n  ".join(mismatches)
            raise ValueError(f"Graph bundle metadata mismatch for {path}:\n  {details}")
    return GraphBundle(raw)
