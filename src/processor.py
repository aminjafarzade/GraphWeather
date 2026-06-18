from __future__ import annotations

import torch
from torch import nn

from .graph_bundle import GraphBundle
from .layers import LocalGraphAttentionBlock
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
    ):
        super().__init__()
        self.graph = graph
        self.l0_blocks = nn.ModuleList(
            [LocalGraphAttentionBlock(dim, edge_dim=edge_dim, heads=heads) for _ in range(l0_blocks)]
        )
        self.pool01 = MeanMaxPool(dim)
        self.l1_blocks = nn.ModuleList(
            [LocalGraphAttentionBlock(dim, edge_dim=edge_dim, heads=heads) for _ in range(l1_blocks)]
        )
        self.pool12 = MeanMaxPool(dim)
        self.l2_blocks = nn.ModuleList(
            [LocalGraphAttentionBlock(dim, edge_dim=edge_dim, heads=heads) for _ in range(l2_blocks)]
        )
        self.unpool21 = ParentUnpoolFuse(dim)
        self.l1_refine = nn.ModuleList(
            [LocalGraphAttentionBlock(dim, edge_dim=edge_dim, heads=heads) for _ in range(l1_refine_blocks)]
        )
        self.unpool10 = ParentUnpoolFuse(dim)
        self.l0_refine = nn.ModuleList(
            [LocalGraphAttentionBlock(dim, edge_dim=edge_dim, heads=heads) for _ in range(l0_refine_blocks)]
        )

    def forward(self, h0: torch.Tensor) -> torch.Tensor:
        for block in self.l0_blocks:
            h0 = block(h0, self.graph.L0)
        skip0 = h0

        h1 = self.pool01(h0, self.graph.pool_L0_to_L1, self.graph.L1.num_nodes)
        for block in self.l1_blocks:
            h1 = block(h1, self.graph.L1)
        skip1 = h1

        h2 = self.pool12(h1, self.graph.pool_L1_to_L2, self.graph.L2.num_nodes)
        for block in self.l2_blocks:
            h2 = block(h2, self.graph.L2)

        h1 = self.unpool21(h2, self.graph.pool_L1_to_L2, skip1)
        for block in self.l1_refine:
            h1 = block(h1, self.graph.L1)

        h0 = self.unpool10(h1, self.graph.pool_L0_to_L1, skip0)
        for block in self.l0_refine:
            h0 = block(h0, self.graph.L0)
        return h0

