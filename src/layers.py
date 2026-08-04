from __future__ import annotations

import logging
import math

import torch
from torch import nn

from .graph_bundle import GraphLevel

ATTENTION_IMPLS = ("elementwise", "matmul")

# Which LocalGraphAttention contraction runs by default, reported in the model
# summary so a run's logs say which kernel path produced them.
#
# "elementwise" is (a * b).sum(dim) in the [B,N,k,H,d] layout -- the formulation the
# 200-epoch hidden-160 control was trained with. "matmul" is the head-major
# torch.matmul form. They are numerically equivalent (exact in float64); the default
# is set from measurement, not preference.
#
# Measured on a B200, 2.5deg L0 block (B=12, N=10368, dim=160, heads=5, k=8),
# forward+backward, bf16:
#     eager  elementwise   7.84 ms   3.25 GB
#     eager  matmul       17.27 ms   2.97 GB   0.45x
# The contraction is [.., 1, d] @ [.., d, k] -- a batched *gemv* with m=1 over
# B*N*H = 622k matrices. No tensor-core tile applies at m=1, so it loses to the
# bandwidth-bound elementwise form despite touching less memory. matmul only pulls
# ahead at L3 in fp32 (k=24, 1 of 11 blocks).
#
# The transient the matmul was meant to remove is what a fusing compiler removes for
# free: compiled elementwise is 1.82 ms / 1.36 GB -- 4.3x faster AND 58% less memory
# than eager elementwise, and 8.6x faster than eager matmul. See
# scripts/dev/bench_attention_impl.py.
ATTENTION_IMPL_DEFAULT = "elementwise"

# Back-compat alias for callers that only want the default's name.
ATTENTION_IMPL = ATTENTION_IMPL_DEFAULT


def resolve_attention_impl(attention_impl: str | None) -> str:
    name = str(attention_impl or ATTENTION_IMPL_DEFAULT).strip().lower()
    if name not in ATTENTION_IMPLS:
        raise ValueError(f"attention_impl must be one of {ATTENTION_IMPLS}, got {name!r}")
    return name


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


def resolve_edge_encoding(edge_encoding: dict | None) -> dict:
    """Normalize the optional ``model.edge_encoding`` block.

    Every field defaults to the value that keeps ``LocalGraphAttention``
    bit-identical to the pre-feature behaviour: no feature expansion, no gate,
    and the original ``Linear(edge_dim, heads)`` attention bias.
    """
    raw = dict(edge_encoding or {})
    resolved = {
        "rbf_bins": int(raw.get("rbf_bins", 0) or 0),
        "bearing_harmonics": int(raw.get("bearing_harmonics", 0) or 0),
        "gate": bool(raw.get("gate", False)),
        "bias_mlp_hidden": int(raw.get("bias_mlp_hidden", 0) or 0),
    }
    if resolved["rbf_bins"] < 0:
        raise ValueError(f"edge_encoding.rbf_bins must be >= 0, got {resolved['rbf_bins']}")
    if resolved["rbf_bins"] == 1:
        raise ValueError("edge_encoding.rbf_bins must be 0 (off) or >= 2; 1 bin has no spacing.")
    if resolved["bearing_harmonics"] < 0:
        raise ValueError(
            f"edge_encoding.bearing_harmonics must be >= 0, got {resolved['bearing_harmonics']}"
        )
    if resolved["bias_mlp_hidden"] < 0:
        raise ValueError(
            f"edge_encoding.bias_mlp_hidden must be >= 0, got {resolved['bias_mlp_hidden']}"
        )
    return resolved


def edge_encoding_is_off(resolved: dict) -> bool:
    """True when the resolved block requests today's exact behaviour."""
    return (
        int(resolved["rbf_bins"]) == 0
        and int(resolved["bearing_harmonics"]) == 0
        and not bool(resolved["gate"])
        and int(resolved["bias_mlp_hidden"]) == 0
    )


def expanded_edge_dim(edge_dim: int, resolved: dict) -> int:
    """Width of the expanded edge-feature vector for a given config."""
    return (
        int(edge_dim)
        + int(resolved["rbf_bins"])
        + 2 * int(resolved["bearing_harmonics"])
    )


def expand_edge_features(edge_attr: torch.Tensor, resolved: dict) -> torch.Tensor:
    """Expand raw 6-dim edge features to [..., expanded_dim].

    ``edge_attr`` is [num_nodes, k, edge_dim] with the graph-builder layout
    ``[great_circle_distance, sin(bearing), cos(bearing), dlat, sin(dlon), cos(dlon)]``.
    The original features are always concatenated last so the raw signal is kept.
    """
    rbf_bins = int(resolved["rbf_bins"])
    harmonics = int(resolved["bearing_harmonics"])
    if rbf_bins == 0 and harmonics == 0:
        return edge_attr
    parts: list[torch.Tensor] = []
    if rbf_bins > 0:
        dist = edge_attr[..., 0]
        # Per-level max: edge_attr is fixed per graph level, so this is a constant.
        denom = dist.detach().max().clamp_min(1.0e-12)
        dist_norm = (dist / denom).unsqueeze(-1)
        centres = torch.linspace(0.0, 1.0, rbf_bins, device=edge_attr.device, dtype=edge_attr.dtype)
        spacing = 1.0 / float(rbf_bins - 1)
        gamma = 1.0 / (2.0 * spacing * spacing)
        # dist_norm is [..., 1] and centres is [rbf_bins]; broadcasting gives [..., rbf_bins].
        parts.append(torch.exp(-gamma * (dist_norm - centres) ** 2))
    if harmonics > 0:
        sin_b = edge_attr[..., 1]
        cos_b = edge_attr[..., 2]
        sin_n, cos_n = sin_b, cos_b
        for _ in range(harmonics):
            # multiple-angle recurrence: sin((n+1)x), cos((n+1)x) from sin(nx), cos(nx)
            sin_next = sin_n * cos_b + cos_n * sin_b
            cos_next = cos_n * cos_b - sin_n * sin_b
            sin_n, cos_n = sin_next, cos_next
            parts.append(torch.stack([sin_n, cos_n], dim=-1))
    parts.append(edge_attr)
    return torch.cat(parts, dim=-1)


class LocalGraphAttention(nn.Module):
    """Fixed-k local graph attention over sorted incoming kNN edges."""

    def __init__(
        self,
        dim: int,
        edge_dim: int = 6,
        heads: int = 4,
        edge_encoding: dict | None = None,
        attention_impl: str | None = None,
    ):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = dim // heads
        self.attention_impl = resolve_attention_impl(attention_impl)
        self.edge_dim = int(edge_dim)
        self.edge_encoding = resolve_edge_encoding(edge_encoding)
        self.edge_encoding_off = edge_encoding_is_off(self.edge_encoding)
        self.expanded_edge_dim = expanded_edge_dim(self.edge_dim, self.edge_encoding)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.edge_k = nn.Linear(edge_dim, dim)
        self.edge_v = nn.Linear(edge_dim, dim)
        bias_hidden = int(self.edge_encoding["bias_mlp_hidden"])
        if bias_hidden > 0:
            self.edge_bias = nn.Sequential(
                nn.Linear(self.expanded_edge_dim, bias_hidden),
                nn.GELU(),
                nn.Linear(bias_hidden, heads),
            )
        else:
            self.edge_bias = nn.Linear(edge_dim, heads)
        if bool(self.edge_encoding["gate"]):
            self.edge_gate_mlp = nn.Linear(self.expanded_edge_dim, dim)
            # Zero weight (not std 1e-2) so tanh(0)=0 and the gate is EXACTLY 1.0 at
            # init -- the stated premise "at init the layer is identical to today's".
            # A std-1e-2 weight would leave gate = 1 +- O(1e-2), which is not an
            # identity. This still trains from step one: d(out)/dW is proportional to
            # gate_scale * sech^2(0) * feats * v_src, which is non-zero.
            nn.init.zeros_(self.edge_gate_mlp.weight)
            nn.init.zeros_(self.edge_gate_mlp.bias)
            self.gate_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        else:
            self.edge_gate_mlp = None
            self.register_parameter("gate_scale", None)
        self.out_proj = nn.Linear(dim, dim)
        self._edge_feat_cache: dict[tuple, torch.Tensor] = {}
        # Rollout-scoped cache for the static edge projections. ``None`` disables
        # caching (recompute every call, the historical behaviour); a dict enables
        # it. Ownership belongs to GraphWeatherModel.edge_cache_scope(), which
        # warms it before the rollout and always clears it afterwards -- this
        # module never enables it on its own.
        self._edge_cache: dict[int, tuple] | None = None

    def _edge_feats(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """Expanded features for this level, computed once and cached.

        ``edge_attr`` is a fixed non-persistent buffer per graph level, so the
        expansion is a constant. Without this cache it would be recomputed for
        every block, rollout step and micro-batch.
        """
        if self.edge_encoding["rbf_bins"] == 0 and self.edge_encoding["bearing_harmonics"] == 0:
            return edge_attr
        key = (
            int(edge_attr.data_ptr()),
            tuple(edge_attr.shape),
            str(edge_attr.dtype),
            str(edge_attr.device),
        )
        cached = self._edge_feat_cache.get(key)
        if cached is None:
            cached = expand_edge_features(edge_attr, self.edge_encoding)
            self._edge_feat_cache[key] = cached
        return cached

    def _edge_static(self, graph: GraphLevel) -> tuple:
        """Every per-graph static tensor: ``(edge_k, edge_v, edge_bias, gate)``.

        Producing all of them here, rather than partly inside ``forward``, is what
        keeps the compiled path clean. ``_edge_feats`` keys its memo on
        ``edge_attr.data_ptr()``, a host-side read that dynamo cannot trace: calling
        it inside a compiled region graph-breaks, dynamo then compiles
        ``LocalGraphAttentionBlock.forward`` as its own frame, and that frame gets
        specialized once per ``diagnostics_name`` until it blows recompile_limit and
        falls back to eager. ``warm_edge_cache()`` calls this from outside every
        compiled region, so the expansion runs once per rollout in plain eager.

        ``gate`` is ``None`` unless edge gating is on. Unlike edge_k/edge_v it *is*
        retained per rollout step when computed inline (a ``mul`` saves both
        operands), so caching it also removes real memory, not just recompute.
        """
        if self._edge_cache is not None:
            hit = self._edge_cache.get(id(graph))
            if hit is not None and hit[0] is graph:
                return hit[1]
        num_nodes = int(graph.num_nodes)
        k_neighbors = int(graph.k)
        edge_attr = graph.edge_attr.reshape(num_nodes, k_neighbors, -1)
        needs_feats = (
            int(self.edge_encoding["bias_mlp_hidden"]) > 0 or self.edge_gate_mlp is not None
        )
        edge_feats = self._edge_feats(edge_attr) if needs_feats else edge_attr
        bias_input = (
            edge_feats if int(self.edge_encoding["bias_mlp_hidden"]) > 0 else edge_attr
        )
        edge_k = (
            self.edge_k(edge_attr)
            .reshape(num_nodes, k_neighbors, self.heads, self.head_dim)
            .transpose(1, 2)
        )
        edge_v = (
            self.edge_v(edge_attr)
            .reshape(num_nodes, k_neighbors, self.heads, self.head_dim)
            .transpose(1, 2)
        )
        edge_bias = (
            self.edge_bias(bias_input)
            .reshape(num_nodes, k_neighbors, self.heads)
            .transpose(1, 2)
        )
        gate = None
        if self.edge_gate_mlp is not None:
            # gate == 1.0 exactly at init (zero weight x tanh(0 bias) -> 0), so the
            # layer starts identical to the ungated path. Once gate_scale grows past 1
            # the gate can go negative, which lets the otherwise convex softmax
            # aggregation represent a signed stencil (i.e. a derivative).
            gate_raw = self.edge_gate_mlp(edge_feats).reshape(
                num_nodes, k_neighbors, self.heads, self.head_dim
            )
            gate_scale = self.gate_scale.to(device=gate_raw.device, dtype=gate_raw.dtype)
            gate = 1.0 + gate_scale * torch.tanh(gate_raw)
        terms = (edge_k, edge_v, edge_bias, gate)
        if self._edge_cache is not None:
            self._edge_cache[id(graph)] = (graph, terms)
        return terms

    def _edge_terms(self, graph: GraphLevel) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Head-major static edge projections ``[N,H,k,d]``, ``[N,H,k,d]``, ``[N,H,k]``.

        ``graph.edge_attr`` is built once by the graph builder and never changes,
        so within a single optimizer step these three tensors are byte-identical
        at every autoregressive rollout step. Recomputing them per step makes
        autograd retain one copy per step (~4.4 GB at 2.5deg/hidden-160/S10);
        sharing one copy collapses that and gives edge_k/edge_v/edge_bias a single
        forward whose gradient accumulates from every consumer.

        Keyed on ``id(graph)`` so a module reused across levels can never be
        served another level's geometry. The stored graph reference makes an
        ``id()`` collision after a GC impossible to mistake for a hit.
        """
        edge_k, edge_v, edge_bias, _ = self._edge_static(graph)
        return edge_k, edge_v, edge_bias

    def _validate(self, h: torch.Tensor, graph: GraphLevel) -> tuple[int, int, int, int]:
        bsz, num_nodes, dim = h.shape
        if num_nodes != graph.num_nodes:
            raise ValueError(f"Expected {graph.num_nodes} nodes, got {num_nodes}")
        k_neighbors = graph.k
        edge_count = graph.edge_index.shape[1]
        if edge_count != num_nodes * k_neighbors:
            raise ValueError("Graph edges must be sorted as exactly k incoming edges per target node.")
        return bsz, num_nodes, dim, k_neighbors

    def _gate(self, graph: GraphLevel) -> torch.Tensor:
        """Neighbour-major ``[N,k,H,d]`` value gate. Off in every config but edge_gate."""
        gate = self._edge_static(graph)[3]
        if gate is None:
            raise RuntimeError("_gate called on a module without edge gating enabled.")
        return gate

    def forward(
        self,
        h: torch.Tensor,
        graph: GraphLevel,
        diagnostics_collector: object | None = None,
        diagnostics_name: str | None = None,
    ) -> torch.Tensor:
        if self.attention_impl == "matmul":
            return self._forward_matmul(h, graph, diagnostics_collector, diagnostics_name)
        return self._forward_elementwise(h, graph, diagnostics_collector, diagnostics_name)

    def _forward_elementwise(
        self,
        h: torch.Tensor,
        graph: GraphLevel,
        diagnostics_collector: object | None = None,
        diagnostics_name: str | None = None,
    ) -> torch.Tensor:
        """[B,N,k,H,d] elementwise contraction. Default; see ATTENTION_IMPL_DEFAULT.

        Same math as ``_forward_reference``, but it reads its edge projections from
        the shared ``_edge_terms`` cache instead of recomputing them per rollout step.
        """
        bsz, num_nodes, dim, k_neighbors = self._validate(h, graph)
        src = graph.edge_index[0].reshape(num_nodes, k_neighbors)
        q = self.q_proj(h).reshape(bsz, num_nodes, self.heads, self.head_dim)
        k = self.k_proj(h).reshape(bsz, num_nodes, self.heads, self.head_dim)
        v = self.v_proj(h).reshape(bsz, num_nodes, self.heads, self.head_dim)

        # _edge_terms is head-major; transposing back to neighbour-major is a view.
        edge_k, edge_v, edge_bias = self._edge_terms(graph)
        edge_k = edge_k.transpose(1, 2)
        edge_v = edge_v.transpose(1, 2)
        edge_bias = edge_bias.transpose(1, 2)

        k_src = k[:, src]
        scores = (q[:, :, None] * (k_src + edge_k[None])).sum(dim=-1) / math.sqrt(self.head_dim)
        scores = scores + edge_bias[None]
        if graph.edge_mask is not None:
            scores = scores.masked_fill(
                ~graph.edge_mask.reshape(1, num_nodes, k_neighbors, 1),
                float("-inf"),
            )
        attn = torch.softmax(scores, dim=2)  # dim=2 is the neighbour axis in this layout
        if diagnostics_collector is not None and hasattr(diagnostics_collector, "add_attention"):
            # src/diagnostics/ consumes attention weights as [B, N, k, H].
            diagnostics_collector.add_attention(
                diagnostics_name or "attention",
                attn,
                edge_index=graph.edge_index,
                num_nodes=graph.num_nodes,
            )
        v_src = v[:, src]
        if self.edge_gate_mlp is not None:
            v_src = self._gate(graph)[None] * v_src
        out = (attn[..., None] * (v_src + edge_v[None])).sum(dim=2)
        return self.out_proj(out.reshape(bsz, num_nodes, dim))

    def _forward_matmul(
        self,
        h: torch.Tensor,
        graph: GraphLevel,
        diagnostics_collector: object | None = None,
        diagnostics_name: str | None = None,
    ) -> torch.Tensor:
        """Head-major torch.matmul contraction. Lower peak memory, slower in eager."""
        bsz, num_nodes, dim, k_neighbors = self._validate(h, graph)
        src = graph.edge_index[0].reshape(num_nodes, k_neighbors)
        q = self.q_proj(h).reshape(bsz, num_nodes, self.heads, self.head_dim)
        k = self.k_proj(h).reshape(bsz, num_nodes, self.heads, self.head_dim)
        v = self.v_proj(h).reshape(bsz, num_nodes, self.heads, self.head_dim)

        edge_k, edge_v, edge_bias = self._edge_terms(graph)

        # Head-major layout: (B, N, H) are batch dims and (k, head_dim) are the
        # matmul axes. The gathered-and-shifted key tensor below is the only
        # [B,N,H,k,d] transient; the elementwise form additionally materializes
        # the full q*k product at the same size before reducing it.
        k_src = k[:, src].transpose(2, 3) + edge_k
        scores = torch.matmul(q.unsqueeze(-2), k_src.transpose(-1, -2)).squeeze(-2)
        scores = scores / math.sqrt(self.head_dim) + edge_bias
        if graph.edge_mask is not None:
            scores = scores.masked_fill(
                ~graph.edge_mask.reshape(1, num_nodes, 1, k_neighbors),
                float("-inf"),
            )
        # dim=-1 is the neighbour axis here. It was dim=2 under the [B,N,k,H]
        # layout; using dim=2 now would normalize across heads and still train.
        attn = torch.softmax(scores, dim=-1)
        if diagnostics_collector is not None and hasattr(diagnostics_collector, "add_attention"):
            # src/diagnostics/ consumes attention weights as [B, N, k, H].
            diagnostics_collector.add_attention(
                diagnostics_name or "attention",
                attn.transpose(2, 3),
                edge_index=graph.edge_index,
                num_nodes=graph.num_nodes,
            )
        v_src = v[:, src].transpose(2, 3)
        if self.edge_gate_mlp is not None:
            v_src = self._gate(graph).transpose(1, 2) * v_src
        v_src = v_src + edge_v
        out = torch.matmul(attn.unsqueeze(-2), v_src).squeeze(-2)
        return self.out_proj(out.reshape(bsz, num_nodes, dim))

    def _forward_reference(
        self,
        h: torch.Tensor,
        graph: GraphLevel,
        diagnostics_collector: object | None = None,
        diagnostics_name: str | None = None,
    ) -> torch.Tensor:
        """Pre-refactor elementwise implementation. Test-only equivalence oracle.

        Kept verbatim, including the uncached edge projections, so the tests in
        tests/test_attention_matmul_edge_cache.py check both the layout change
        and _edge_terms against the code this run's control was trained with.
        """
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
        edge_feats = self._edge_feats(edge_attr)
        edge_k = self.edge_k(edge_attr).reshape(num_nodes, k_neighbors, self.heads, self.head_dim)
        edge_v = self.edge_v(edge_attr).reshape(num_nodes, k_neighbors, self.heads, self.head_dim)
        bias_input = edge_feats if int(self.edge_encoding["bias_mlp_hidden"]) > 0 else edge_attr
        edge_bias = self.edge_bias(bias_input).reshape(num_nodes, k_neighbors, self.heads)

        q_tgt = q[:, :, None, :, :]
        scores = ((q_tgt * (k_src + edge_k[None, ...])).sum(dim=-1) / math.sqrt(self.head_dim))
        scores = scores + edge_bias[None, ...]
        if graph.edge_mask is not None:
            scores = scores.masked_fill(
                ~graph.edge_mask[None].reshape(1, num_nodes, k_neighbors, 1),
                float("-inf"),
            )
        attn = torch.softmax(scores, dim=2)
        if diagnostics_collector is not None and hasattr(diagnostics_collector, "add_attention"):
            diagnostics_collector.add_attention(
                diagnostics_name or "attention",
                attn,
                edge_index=graph.edge_index,
                num_nodes=graph.num_nodes,
            )
        if self.edge_gate_mlp is None:
            out = (attn[..., None] * (v_src + edge_v[None, ...])).sum(dim=2)
        else:
            # gate == 1.0 exactly at init (weight std 1e-2 x tanh(0 bias) -> 0), so the
            # layer starts identical to the ungated path. Once gate_scale grows past 1
            # the gate can go negative, which lets the otherwise convex softmax
            # aggregation represent a signed stencil (i.e. a derivative).
            gate_raw = self.edge_gate_mlp(edge_feats).reshape(
                num_nodes, k_neighbors, self.heads, self.head_dim
            )
            gate_scale = self.gate_scale.to(device=gate_raw.device, dtype=gate_raw.dtype)
            gate = 1.0 + gate_scale * torch.tanh(gate_raw)
            out = (attn[..., None] * (gate[None, ...] * v_src + edge_v[None, ...])).sum(dim=2)
        out = out.reshape(bsz, num_nodes, dim)
        return self.out_proj(out)


class LocalGraphAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        edge_dim: int = 6,
        heads: int = 4,
        mlp_ratio: int = 4,
        edge_encoding: dict | None = None,
        attention_impl: str | None = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LocalGraphAttention(
            dim,
            edge_dim=edge_dim,
            heads=heads,
            edge_encoding=edge_encoding,
            attention_impl=attention_impl,
        )
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
