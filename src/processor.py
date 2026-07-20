from __future__ import annotations

import logging

import torch
from torch import nn

from .architecture import resolve_graph_architecture
from .graph_bundle import GraphBundle
from .layers import LocalGraphAttentionBlock, NodewiseRefineMLP
from .pooling import MeanMaxPool, ParentUnpoolFuse


class GraphUNetProcessor(nn.Module):
    def __init__(
        self,
        dim: int,
        graph: GraphBundle,
        edge_dim: int = 6,
        heads: int = 4,
        l0_blocks: int = 2,
        l1_blocks: int = 2,
        l2_blocks: int = 1,
        l1_refine_blocks: int = 1,
        l0_refine_blocks: int = 1,
        num_graph_levels: int = 3,
        use_l3: bool = False,
        l3_blocks: int = 1,
        l4_blocks: int = 1,
        l3_refine_after_l4_blocks: int = 1,
        l2_refine_after_l3_blocks: int = 1,
        skip_fusion: dict | None = None,
        pooling: dict | None = None,
        l0_refine: dict | None = None,
    ):
        super().__init__()
        self.graph = graph
        derived_use_l3 = bool(use_l3) or int(num_graph_levels) >= 4
        arch = resolve_graph_architecture(
            {
                "num_graph_levels": num_graph_levels,
                "use_l3": derived_use_l3,
                "l0_blocks": l0_blocks,
                "l1_blocks": l1_blocks,
                "l2_blocks": l2_blocks,
                "l3_blocks": l3_blocks,
                "l4_blocks": l4_blocks,
                "l3_refine_after_l4_blocks": l3_refine_after_l4_blocks,
                "l2_refine_after_l3_blocks": l2_refine_after_l3_blocks,
                "l1_refine_blocks": l1_refine_blocks,
                "l0_refine_blocks": l0_refine_blocks,
                "skip_fusion": skip_fusion,
                "pooling": pooling,
                "l0_refine": l0_refine,
            }
        )
        self.skip_fusion = arch.skip_fusion.asdict()
        self.pooling = arch.pooling.asdict()
        self.l0_refine_config = arch.l0_refine.asdict()
        self.num_graph_levels = arch.num_graph_levels
        self.use_l3 = arch.use_l3
        self.use_l4 = bool(arch.num_graph_levels >= 5)
        self.l0_blocks_count = int(arch.l0_blocks)
        self.l1_blocks_count = int(arch.l1_blocks)
        self.l2_blocks_count = int(arch.l2_blocks)
        self.l3_blocks_count = int(arch.l3_blocks)
        self.l4_blocks_count = int(arch.l4_blocks)
        self.l3_refine_after_l4_blocks_count = int(arch.l3_refine_after_l4_blocks)
        self.l2_refine_after_l3_blocks_count = int(arch.l2_refine_after_l3_blocks)
        self.l1_refine_blocks_count = int(arch.l1_refine_blocks)
        self.l0_refine_blocks_count = int(arch.l0_refine_blocks)
        if self.use_l3:
            if getattr(graph, "L3", None) is None or getattr(graph, "pool_L2_to_L3", None) is None:
                raise ValueError("L3 processor path requested, but the graph bundle has no L3 level/pool map.")
        if self.use_l4:
            if getattr(graph, "L4", None) is None or getattr(graph, "pool_L3_to_L4", None) is None:
                raise ValueError("L4 processor path requested, but the graph bundle has no L4 level/pool map.")
        self.l0_blocks = nn.ModuleList(
            [LocalGraphAttentionBlock(dim, edge_dim=edge_dim, heads=heads) for _ in range(arch.l0_blocks)]
        )
        self.pool01 = MeanMaxPool(dim, pooling=self.pooling, name="l0_to_l1")
        self.l1_blocks = nn.ModuleList(
            [LocalGraphAttentionBlock(dim, edge_dim=edge_dim, heads=heads) for _ in range(arch.l1_blocks)]
        )
        self.pool12 = MeanMaxPool(dim, pooling=self.pooling, name="l1_to_l2")
        self.l2_blocks = nn.ModuleList(
            [LocalGraphAttentionBlock(dim, edge_dim=edge_dim, heads=heads) for _ in range(arch.l2_blocks)]
        )
        if self.use_l3:
            self.pool23 = MeanMaxPool(dim, pooling=self.pooling, name="l2_to_l3")
            self.l3_blocks = nn.ModuleList(
                [LocalGraphAttentionBlock(dim, edge_dim=edge_dim, heads=heads) for _ in range(arch.l3_blocks)]
            )
            if self.use_l4:
                self.pool34 = MeanMaxPool(dim, pooling=self.pooling, name="l3_to_l4")
                self.l4_blocks = nn.ModuleList(
                    [LocalGraphAttentionBlock(dim, edge_dim=edge_dim, heads=heads) for _ in range(arch.l4_blocks)]
                )
                self.unpool43 = ParentUnpoolFuse(dim, skip_fusion=self.skip_fusion, name="l4_to_l3")
                self.l3_refine_after_l4 = nn.ModuleList(
                    [
                        LocalGraphAttentionBlock(dim, edge_dim=edge_dim, heads=heads)
                        for _ in range(arch.l3_refine_after_l4_blocks)
                    ]
                )
            self.unpool32 = ParentUnpoolFuse(dim, skip_fusion=self.skip_fusion, name="l3_to_l2")
            self.l2_refine_after_l3 = nn.ModuleList(
                [
                    LocalGraphAttentionBlock(dim, edge_dim=edge_dim, heads=heads)
                    for _ in range(arch.l2_refine_after_l3_blocks)
                ]
            )
        self.unpool21 = ParentUnpoolFuse(dim, skip_fusion=self.skip_fusion, name="l2_to_l1")
        self.l1_refine = nn.ModuleList(
            [LocalGraphAttentionBlock(dim, edge_dim=edge_dim, heads=heads) for _ in range(arch.l1_refine_blocks)]
        )
        self.unpool10 = ParentUnpoolFuse(dim, skip_fusion=self.skip_fusion, name="l1_to_l0")
        self.l0_refine = nn.ModuleList(
            [
                self._make_l0_refine_block(
                    idx,
                    dim=dim,
                    edge_dim=edge_dim,
                    heads=heads,
                    config=self.l0_refine_config,
                )
                for idx in range(arch.l0_refine_blocks)
            ]
        )
        if self.l0_refine_config["type"] == "nodewise_mlp":
            hidden = int(dim) * int(self.l0_refine_config["mlp_expansion"])
            logging.info("Using l0_refine type: nodewise_mlp")
            logging.info(
                "NodewiseRefineMLP dim=%d hidden=%d residual_scale_init=%.6g",
                int(dim),
                hidden,
                float(self.l0_refine_config["residual_scale_init"]),
            )

    @staticmethod
    def _make_l0_refine_block(
        idx: int,
        *,
        dim: int,
        edge_dim: int,
        heads: int,
        config: dict,
    ) -> nn.Module:
        # The ablation intentionally replaces only processor/l0_refine.0.
        if idx == 0 and str(config.get("type", "attention")).strip().lower() == "nodewise_mlp":
            return NodewiseRefineMLP(
                dim,
                expansion=int(config.get("mlp_expansion", 2)),
                dropout=float(config.get("dropout", 0.0)),
                residual_scale_init=float(config.get("residual_scale_init", 0.1)),
                learnable_residual_scale=bool(config.get("learnable_residual_scale", True)),
            )
        return LocalGraphAttentionBlock(dim, edge_dim=edge_dim, heads=heads)

    def _pool_weights(self, level_name: str) -> torch.Tensor | None:
        if str(self.pooling.get("mean_type", "mean")).lower() != "area_weighted":
            return None
        level = getattr(self.graph, level_name)
        return torch.cos(level.lat_lon[:, 0]).clamp_min(0.0)

    @staticmethod
    def _add_diag(diagnostics_collector: object | None, name: str, tensor: torch.Tensor) -> None:
        if diagnostics_collector is not None and hasattr(diagnostics_collector, "add_embedding"):
            diagnostics_collector.add_embedding(name, tensor)

    def forward(self, h0: torch.Tensor, diagnostics_collector: object | None = None) -> torch.Tensor:
        self._add_diag(diagnostics_collector, "processor/input", h0)
        for idx, block in enumerate(self.l0_blocks):
            h0 = block(
                h0,
                self.graph.L0,
                diagnostics_collector=diagnostics_collector,
                diagnostics_name=f"processor/l0_blocks.{idx}",
            )
            self._add_diag(diagnostics_collector, f"processor/l0_blocks.{idx}", h0)
        skip0 = h0

        h1 = self.pool01(h0, self.graph.pool_L0_to_L1, self.graph.L1.num_nodes, self._pool_weights("L0"))
        self._add_diag(diagnostics_collector, "processor/pool01", h1)
        for idx, block in enumerate(self.l1_blocks):
            h1 = block(
                h1,
                self.graph.L1,
                diagnostics_collector=diagnostics_collector,
                diagnostics_name=f"processor/l1_blocks.{idx}",
            )
            self._add_diag(diagnostics_collector, f"processor/l1_blocks.{idx}", h1)
        skip1 = h1

        h2 = self.pool12(h1, self.graph.pool_L1_to_L2, self.graph.L2.num_nodes, self._pool_weights("L1"))
        self._add_diag(diagnostics_collector, "processor/pool12", h2)
        for idx, block in enumerate(self.l2_blocks):
            h2 = block(
                h2,
                self.graph.L2,
                diagnostics_collector=diagnostics_collector,
                diagnostics_name=f"processor/l2_blocks.{idx}",
            )
            self._add_diag(diagnostics_collector, f"processor/l2_blocks.{idx}", h2)

        if self.use_l3:
            skip2 = h2
            h3 = self.pool23(h2, self.graph.pool_L2_to_L3, self.graph.L3.num_nodes, self._pool_weights("L2"))
            self._add_diag(diagnostics_collector, "processor/pool23", h3)
            for idx, block in enumerate(self.l3_blocks):
                h3 = block(
                    h3,
                    self.graph.L3,
                    diagnostics_collector=diagnostics_collector,
                    diagnostics_name=f"processor/l3_blocks.{idx}",
                )
                self._add_diag(diagnostics_collector, f"processor/l3_blocks.{idx}", h3)
            if self.use_l4:
                skip3 = h3
                h4 = self.pool34(h3, self.graph.pool_L3_to_L4, self.graph.L4.num_nodes, self._pool_weights("L3"))
                self._add_diag(diagnostics_collector, "processor/pool34", h4)
                for idx, block in enumerate(self.l4_blocks):
                    h4 = block(
                        h4,
                        self.graph.L4,
                        diagnostics_collector=diagnostics_collector,
                        diagnostics_name=f"processor/l4_blocks.{idx}",
                    )
                    self._add_diag(diagnostics_collector, f"processor/l4_blocks.{idx}", h4)
                h3 = self.unpool43(h4, self.graph.pool_L3_to_L4, skip3)
                self._add_diag(diagnostics_collector, "processor/unpool43", h3)
                for idx, block in enumerate(self.l3_refine_after_l4):
                    h3 = block(
                        h3,
                        self.graph.L3,
                        diagnostics_collector=diagnostics_collector,
                        diagnostics_name=f"processor/l3_refine_after_l4.{idx}",
                    )
                    self._add_diag(diagnostics_collector, f"processor/l3_refine_after_l4.{idx}", h3)
            h2 = self.unpool32(h3, self.graph.pool_L2_to_L3, skip2)
            self._add_diag(diagnostics_collector, "processor/unpool32", h2)
            for idx, block in enumerate(self.l2_refine_after_l3):
                h2 = block(
                    h2,
                    self.graph.L2,
                    diagnostics_collector=diagnostics_collector,
                    diagnostics_name=f"processor/l2_refine_after_l3.{idx}",
                )
                self._add_diag(diagnostics_collector, f"processor/l2_refine_after_l3.{idx}", h2)

        h1 = self.unpool21(h2, self.graph.pool_L1_to_L2, skip1)
        self._add_diag(diagnostics_collector, "processor/unpool21", h1)
        for idx, block in enumerate(self.l1_refine):
            h1 = block(
                h1,
                self.graph.L1,
                diagnostics_collector=diagnostics_collector,
                diagnostics_name=f"processor/l1_refine.{idx}",
            )
            self._add_diag(diagnostics_collector, f"processor/l1_refine.{idx}", h1)

        h0 = self.unpool10(h1, self.graph.pool_L0_to_L1, skip0)
        self._add_diag(diagnostics_collector, "processor/unpool10", h0)
        for idx, block in enumerate(self.l0_refine):
            h0 = block(
                h0,
                self.graph.L0,
                diagnostics_collector=diagnostics_collector,
                diagnostics_name=f"processor/l0_refine.{idx}",
            )
            self._add_diag(diagnostics_collector, f"processor/l0_refine.{idx}", h0)
        self._add_diag(diagnostics_collector, "processor/output", h0)
        return h0

    def fusion_gate_values(self) -> dict[str, float]:
        values: dict[str, float] = {}
        modules = []
        if self.use_l4:
            modules.append(("l4_to_l3", self.unpool43))
        if self.use_l3:
            modules.append(("l3_to_l2", self.unpool32))
        modules.extend(
            [
                ("l2_to_l1", self.unpool21),
                ("l1_to_l0", self.unpool10),
            ]
        )
        for prefix, module in modules:
            for branch, value in module.gate_values().items():
                values[f"{prefix}_{branch}"] = float(value)
        return values

    def pooling_gate_values(self) -> dict[str, float]:
        values: dict[str, float] = {}
        modules = [
            ("l0_to_l1", self.pool01),
            ("l1_to_l2", self.pool12),
        ]
        if self.use_l3:
            modules.append(("l2_to_l3", self.pool23))
        if self.use_l4:
            modules.append(("l3_to_l4", self.pool34))
        for prefix, module in modules:
            for branch, value in module.gate_values().items():
                values[f"{prefix}_{branch}"] = float(value)
        return values

    def l0_refine_residual_scale_values(self) -> dict[str, float]:
        values: dict[str, float] = {}
        for idx, module in enumerate(self.l0_refine):
            residual_scale = getattr(module, "residual_scale", None)
            if residual_scale is None:
                continue
            values[str(idx)] = float(residual_scale.detach().float().cpu().item())
        return values
