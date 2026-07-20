from __future__ import annotations

import math

import torch
from torch import nn

from .graph_bundle import GraphLevel


class MLP(nn.Module):
    def __init__(self, dim: int, ratio: int = 4):
        super().__init__()
        hidden = int(dim * ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NodewiseRefineMLP(nn.Module):
    """Node-wise residual MLP used to ablate graph attention in one refine block."""

    def __init__(
        self,
        dim: int,
        expansion: int = 2,
        dropout: float = 0.0,
        residual_scale_init: float = 0.1,
        learnable_residual_scale: bool = True,
    ):
        super().__init__()
        dim = int(dim)
        expansion = int(expansion)
        hidden = dim * expansion
        self.dim = dim
        self.hidden_dim = hidden
        self.expansion = expansion
        self.dropout = float(dropout)
        self.residual_scale_init = float(residual_scale_init)
        self.learnable_residual_scale = bool(learnable_residual_scale)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, dim),
        )
        scale = torch.tensor(float(residual_scale_init), dtype=torch.float32)
        if learnable_residual_scale:
            self.residual_scale = nn.Parameter(scale)
        else:
            self.register_buffer("residual_scale", scale)

    def forward(self, x: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        return x + self.residual_scale.to(device=x.device, dtype=x.dtype) * self.mlp(self.norm(x))


class LocalGraphAttention(nn.Module):
    """Fixed-k local graph attention over sorted incoming kNN edges."""

    def __init__(self, dim: int, edge_dim: int = 6, heads: int = 4):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = dim // heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.edge_k = nn.Linear(edge_dim, dim)
        self.edge_v = nn.Linear(edge_dim, dim)
        self.edge_bias = nn.Linear(edge_dim, heads)
        self.out_proj = nn.Linear(dim, dim)

    def forward(
        self,
        h: torch.Tensor,
        graph: GraphLevel,
        diagnostics_collector: object | None = None,
        diagnostics_name: str | None = None,
    ) -> torch.Tensor:
        bsz, num_nodes, dim = h.shape
        if num_nodes != graph.num_nodes:
            raise ValueError(f"Expected {graph.num_nodes} nodes, got {num_nodes}")
        k_neighbors = graph.k
        edge_count = graph.edge_index.shape[1]
        if edge_count != num_nodes * k_neighbors:
            raise ValueError("Graph edges must be sorted as exactly k incoming edges per target node.")

        src = graph.edge_index[0].reshape(num_nodes, k_neighbors)
        q = self.q_proj(h).reshape(bsz, num_nodes, self.heads, self.head_dim)
        k = self.k_proj(h).reshape(bsz, num_nodes, self.heads, self.head_dim)
        v = self.v_proj(h).reshape(bsz, num_nodes, self.heads, self.head_dim)

        k_src = k[:, src, :, :]
        v_src = v[:, src, :, :]
        edge_attr = graph.edge_attr.reshape(num_nodes, k_neighbors, -1)
        edge_k = self.edge_k(edge_attr).reshape(num_nodes, k_neighbors, self.heads, self.head_dim)
        edge_v = self.edge_v(edge_attr).reshape(num_nodes, k_neighbors, self.heads, self.head_dim)
        edge_bias = self.edge_bias(edge_attr).reshape(num_nodes, k_neighbors, self.heads)

        q_tgt = q[:, :, None, :, :]
        scores = ((q_tgt * (k_src + edge_k[None, ...])).sum(dim=-1) / math.sqrt(self.head_dim))
        scores = scores + edge_bias[None, ...]
        attn = torch.softmax(scores, dim=2)
        if diagnostics_collector is not None and hasattr(diagnostics_collector, "add_attention"):
            diagnostics_collector.add_attention(
                diagnostics_name or "attention",
                attn,
                edge_index=graph.edge_index,
                num_nodes=graph.num_nodes,
            )
        out = (attn[..., None] * (v_src + edge_v[None, ...])).sum(dim=2)
        out = out.reshape(bsz, num_nodes, dim)
        return self.out_proj(out)


class LocalGraphAttentionBlock(nn.Module):
    def __init__(self, dim: int, edge_dim: int = 6, heads: int = 4, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LocalGraphAttention(dim, edge_dim=edge_dim, heads=heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, ratio=mlp_ratio)

    def forward(
        self,
        h: torch.Tensor,
        graph: GraphLevel,
        diagnostics_collector: object | None = None,
        diagnostics_name: str | None = None,
    ) -> torch.Tensor:
        h = h + self.attn(
            self.norm1(h),
            graph,
            diagnostics_collector=diagnostics_collector,
            diagnostics_name=diagnostics_name,
        )
        h = h + self.mlp(self.norm2(h))
        return h
