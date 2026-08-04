from __future__ import annotations

import torch
from torch import nn


class FixedBipartiteRemap(nn.Module):
    """Apply one fixed, channel-independent sparse remapping matrix.

    The edge weights are precomputed geometric interpolation weights.  This
    module has no parameters and deliberately performs no feature embedding,
    message MLP, residual update, or channel mixing.
    """

    def forward(
        self,
        x_src: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        n_dst: int,
    ) -> torch.Tensor:
        if x_src.dim() != 3:
            raise ValueError(f"Expected x_src [B,N,C], got {tuple(x_src.shape)}.")
        if edge_index.dim() != 2 or tuple(edge_index.shape[:1]) != (2,):
            raise ValueError(
                f"edge_index must have shape [2,E], got {tuple(edge_index.shape)}."
            )
        if edge_weight.dim() != 1 or int(edge_weight.shape[0]) != int(
            edge_index.shape[1]
        ):
            raise ValueError(
                "edge_weight must have shape [E] matching edge_index; "
                f"got {tuple(edge_weight.shape)} for E={edge_index.shape[1]}."
            )
        n_dst = int(n_dst)
        if n_dst <= 0:
            raise ValueError(f"n_dst must be positive, got {n_dst}.")

        src = edge_index[0].to(device=x_src.device).long()
        dst = edge_index[1].to(device=x_src.device).long()
        if src.numel() and (
            int(src.min()) < 0 or int(src.max()) >= int(x_src.shape[1])
        ):
            raise ValueError(
                f"Source indices exceed x_src node count {x_src.shape[1]}."
            )
        if dst.numel() and (int(dst.min()) < 0 or int(dst.max()) >= n_dst):
            raise ValueError(f"Destination indices exceed n_dst={n_dst}.")

        weight = edge_weight.to(device=x_src.device, dtype=x_src.dtype)
        out = x_src.new_zeros((int(x_src.shape[0]), n_dst, int(x_src.shape[2])))
        out.index_add_(1, dst, x_src[:, src] * weight.view(1, -1, 1))
        return out


class MLPLayerNorm(nn.Module):
    """Two-layer MLP followed by feature-wise LayerNorm.

    This is the boundary embedding/update primitive used by the opt-in
    GraphCast-style grid<->mesh path. The legacy path keeps its original linear
    and GELU-MLP modules so existing checkpoint parameter names and shapes remain
    unchanged.
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden)
        self.out_dim = int(out_dim)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _EdgeMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BipartiteMP(nn.Module):
    """One GraphCast-style bipartite message-passing step between two node sets.

    Messages flow src -> dst over a fixed, precomputed edge set with per-edge
    features. Aggregation is either a raw destination sum or a destination-degree
    mean. The latter prevents radius-gathered grid->mesh nodes with many incoming
    edges from receiving proportionally larger updates. The destination is updated
    RESIDUALLY, so pass its current embedding as ``h_dst`` (for the decoder that is
    the grid embedding carried through — a GraphCast-style skip).
    """

    def __init__(
        self,
        dim: int,
        edge_dim: int = 6,
        mlp_hidden_ratio: int = 2,
        *,
        hidden_ratio: int | None = None,
        aggregation: str = "sum",
        graphcast_mlp: bool = False,
    ):
        super().__init__()
        dim = int(dim)
        ratio = int(mlp_hidden_ratio if hidden_ratio is None else hidden_ratio)
        if ratio < 1:
            raise ValueError(f"mlp_hidden_ratio must be >= 1, got {ratio}.")
        aggregation = str(aggregation).strip().lower()
        if aggregation not in {"sum", "mean"}:
            raise ValueError(f"aggregation must be 'sum' or 'mean', got {aggregation!r}.")
        hidden = int(dim * ratio)
        self.dim = dim
        self.edge_dim = int(edge_dim)
        self.mlp_hidden_ratio = ratio
        self.aggregation = aggregation
        self.graphcast_mlp = bool(graphcast_mlp)
        if self.graphcast_mlp:
            self.edge_embed = MLPLayerNorm(int(edge_dim), hidden, dim)
            self.msg = MLPLayerNorm(3 * dim, hidden, dim)
            self.upd = MLPLayerNorm(2 * dim, hidden, dim)
        else:
            self.edge_embed = None
            self.msg = _EdgeMLP(2 * dim + int(edge_dim), hidden, dim)
            self.upd = _EdgeMLP(2 * dim, hidden, dim)

    def _aggregate_messages(
        self,
        messages: torch.Tensor,
        destination_index: torch.Tensor,
        n_dst: int,
    ) -> torch.Tensor:
        bsz, _, dim = messages.shape
        agg = torch.zeros(
            bsz,
            int(n_dst),
            dim,
            device=messages.device,
            dtype=messages.dtype,
        )
        agg.index_add_(1, destination_index, messages)
        if self.aggregation == "mean":
            degree = torch.bincount(destination_index, minlength=int(n_dst))
            degree = degree.clamp_min(1).to(device=messages.device, dtype=messages.dtype)
            agg = agg / degree.view(1, -1, 1)
        return agg

    def forward(
        self,
        h_src: torch.Tensor,       # [B, Ns, D]
        h_dst: torch.Tensor,       # [B, Nd, D]
        edge_index: torch.Tensor,  # [2, E] : row 0 = src node idx, row 1 = dst node idx
        edge_attr: torch.Tensor,   # [E, edge_dim]
    ) -> torch.Tensor:
        if h_src.dim() != 3 or h_dst.dim() != 3:
            raise ValueError(f"Expected [B,N,D] tensors, got {tuple(h_src.shape)} and {tuple(h_dst.shape)}")
        if h_src.shape[0] != h_dst.shape[0] or h_src.shape[-1] != h_dst.shape[-1]:
            raise ValueError(
                f"Source/destination batch and hidden dimensions must match, got "
                f"{tuple(h_src.shape)} and {tuple(h_dst.shape)}."
            )
        if int(h_dst.shape[-1]) != self.dim:
            raise ValueError(f"Expected hidden dimension {self.dim}, got {h_dst.shape[-1]}.")
        if edge_index.dim() != 2 or tuple(edge_index.shape[:1]) != (2,):
            raise ValueError(f"edge_index must have shape [2,E], got {tuple(edge_index.shape)}.")
        if edge_attr.dim() != 2 or int(edge_attr.shape[0]) != int(edge_index.shape[1]):
            raise ValueError(
                f"edge_attr must have shape [E,{self.edge_dim}] for E={edge_index.shape[1]}, "
                f"got {tuple(edge_attr.shape)}."
            )
        if int(edge_attr.shape[1]) != self.edge_dim:
            raise ValueError(f"Expected edge_attr width {self.edge_dim}, got {edge_attr.shape[1]}.")
        bsz, n_dst, _ = h_dst.shape
        s = edge_index[0].to(device=h_dst.device).long()
        d = edge_index[1].to(device=h_dst.device).long()
        ea = edge_attr.to(device=h_dst.device, dtype=h_dst.dtype).unsqueeze(0).expand(bsz, -1, -1)
        if self.edge_embed is not None:
            edge_h = self.edge_embed(ea)
            message_input = torch.cat([h_src[:, s], h_dst[:, d], edge_h], dim=-1)
        else:
            message_input = torch.cat([h_src[:, s], h_dst[:, d], ea], dim=-1)
        m = self.msg(message_input)  # [B, E, D]
        agg = self._aggregate_messages(m, d, int(n_dst))
        return h_dst + self.upd(torch.cat([h_dst, agg], dim=-1))
