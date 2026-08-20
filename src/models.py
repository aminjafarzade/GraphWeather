from __future__ import annotations

import contextlib
import logging
import math
from collections.abc import Iterator

import torch
from torch import nn

from .architecture import resolve_graph_architecture
from .batch_adapter import GridNodeAdapter
from .graph_bundle import GraphBundle, GraphLevel
from .layers import LocalGraphAttention, LocalGraphAttentionBlock, resolve_attention_impl
from .lead_conditioning import build_lead_conditioning_grid
from .mesh_layers import BipartiteMP, FixedBipartiteRemap, MLPLayerNorm
from .processor import GraphUNetProcessor


def _diag_name(collector: object | None, name: str) -> str | None:
    """The name only matters when a collector will read it.

    LocalGraphAttention uses ``diagnostics_name`` solely inside its
    ``diagnostics_collector is not None`` branch, so with diagnostics off the string
    is dead. Passing a distinct one per block anyway makes torch.compile specialize
    ``LocalGraphAttentionBlock.forward`` once per block, and 11 blocks against the
    default ``recompile_limit`` of 8 tips that frame back into eager.
    """
    return name if collector is not None else None


class GraphWeatherModel(nn.Module):
    """Direct-grid spherical Graph U-Net that predicts a state tendency."""

    def __init__(
        self,
        graph: GraphBundle,
        grid_shape: tuple[int, int],
        input_channels: int,
        output_channels: int,
        n_history: int = 1,
        hidden_dim: int = 96,
        edge_dim: int = 6,
        heads: int = 4,
        k_neighbors: int = 8,
        level_k_neighbors: list[int] | tuple[int, ...] | None = None,
        level_dims: list[int] | tuple[int, ...] | None = None,
        level_heads: list[int] | tuple[int, ...] | None = None,
        encoder_blocks: int = 1,
        decoder_blocks: int = 1,
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
        lead_conditioning: dict | None = None,
        use_delta_normalization: bool = False,
        delta_mean: torch.Tensor | None = None,
        delta_std: torch.Tensor | None = None,
        delta_norm_center: bool = False,
        delta_norm_eps: float = 1.0e-6,
        aux_feature_dim: int = 0,
        mesh_encoder: dict | None = None,
        edge_encoding: dict | None = None,
        attention_impl: str | None = None,
        boundary_mlp: bool = False,
        head_init_std: float = 0.0,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim)
        heads = int(heads)
        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={heads}.")
        self.edge_encoding = dict(edge_encoding or {})
        self.attention_impl = resolve_attention_impl(attention_impl)
        self.boundary_mlp = bool(boundary_mlp)
        self.head_init_std = float(head_init_std)
        self.mesh_encoder = dict(mesh_encoder or {})
        self._mesh_cfg = self.mesh_encoder
        self.mesh_mode = bool(self.mesh_encoder.get("enabled", False))
        self.mesh_boundary_type = str(
            self.mesh_encoder.get("boundary_type", "legacy")
        ).strip().lower()
        if self.mesh_boundary_type not in {
            "legacy",
            "graphcast_mlp",
            "fixed_spherical",
        }:
            raise ValueError(
                "mesh_encoder.boundary_type must be 'legacy', 'graphcast_mlp', "
                "or 'fixed_spherical', "
                f"got {self.mesh_boundary_type!r}."
            )
        self.graphcast_mesh_boundary = self.mesh_mode and self.mesh_boundary_type == "graphcast_mlp"
        self.fixed_spherical_boundary = (
            self.mesh_mode and self.mesh_boundary_type == "fixed_spherical"
        )
        self.grid_skip_mlp_enabled = self.mesh_mode and bool(
            self.mesh_encoder.get("grid_skip_mlp", False)
        )
        self.grid_attention_encoder_blocks = (
            int(self.mesh_encoder.get("grid_attention_encoder_blocks", 0))
            if self.mesh_mode
            else 0
        )
        self.grid_attention_decoder_blocks = (
            int(self.mesh_encoder.get("grid_attention_decoder_blocks", 0))
            if self.mesh_mode
            else 0
        )
        self.grid_attention_k_neighbors = (
            int(self.mesh_encoder.get("grid_attention_k_neighbors", 8))
            if self.mesh_mode
            else 0
        )
        if self.grid_attention_encoder_blocks < 0 or self.grid_attention_decoder_blocks < 0:
            raise ValueError("Mesh grid-attention block counts must be >= 0.")
        if self.grid_skip_mlp_enabled and not self.graphcast_mesh_boundary:
            raise ValueError(
                "mesh_encoder.grid_skip_mlp=true requires "
                "mesh_encoder.boundary_type='graphcast_mlp'."
            )
        # With grid-attention blocks the fixed remap moves embedded hidden features
        # instead of raw channels, so the grid gets message passing BEFORE the
        # grid->mesh interpolation and AFTER the mesh->grid one, and the head runs on
        # the grid (the output delta is never interpolated). Without blocks the
        # historical raw-channel behavior is preserved bit-for-bit.
        self.grid_message_passing = bool(
            self.fixed_spherical_boundary
            and (
                self.grid_attention_encoder_blocks
                or self.grid_attention_decoder_blocks
            )
        )
        if self.mesh_mode:
            # In mesh mode graph.L0 is the icosphere; the data grid lives on the bundle.
            if str(getattr(graph, "graph_mode", "grid")) != "mesh":
                raise ValueError("mesh_encoder.enabled=true but the graph bundle is not a mesh bundle (metadata.graph_mode != 'mesh').")
            gh, gw = int(getattr(graph, "grid_height", 0)), int(getattr(graph, "grid_width", 0))
            if (gh, gw) != (int(grid_shape[0]), int(grid_shape[1])):
                raise ValueError(f"Mesh bundle grid {(gh, gw)} does not match data grid {tuple(grid_shape)}")
            if (
                self.grid_attention_encoder_blocks or self.grid_attention_decoder_blocks
            ) and getattr(graph, "grid_attention_graph", None) is None:
                raise ValueError(
                    "Mesh grid-attention blocks require a mesh bundle containing "
                    "grid['attention_level']."
                )
            if (
                self.grid_attention_encoder_blocks or self.grid_attention_decoder_blocks
            ) and int(graph.grid_attention_graph.k) != self.grid_attention_k_neighbors:
                raise ValueError(
                    "Mesh grid-attention graph k does not match the model config: "
                    f"graph={graph.grid_attention_graph.k}, "
                    f"mesh_encoder.grid_attention_k_neighbors={self.grid_attention_k_neighbors}."
                )
            if self.fixed_spherical_boundary and (
                getattr(graph, "g2m_edge_weight", None) is None
                or getattr(graph, "m2g_edge_weight", None) is None
            ):
                raise ValueError(
                    "mesh_encoder.boundary_type='fixed_spherical' requires "
                    "g2m.edge_weight and m2g.edge_weight in the mesh graph bundle."
                )
        else:
            if graph.L0.height != grid_shape[0] or graph.L0.width != grid_shape[1]:
                raise ValueError(
                    f"Graph L0 grid {(graph.L0.height, graph.L0.width)} does not match data grid {grid_shape}"
                )
        self.graph = graph
        self.hidden_dim = hidden_dim
        self.num_heads = heads
        self.head_dim = hidden_dim // heads
        self.use_delta_normalization = bool(use_delta_normalization)
        self.delta_norm_center = bool(delta_norm_center)
        self.delta_norm_eps = float(delta_norm_eps)
        self.aux_feature_dim = int(aux_feature_dim)
        graph_level_k = None
        if isinstance(getattr(graph, "metadata", None), dict):
            graph_level_k = graph.metadata.get("level_k_neighbors", None)
        resolved_level_k = level_k_neighbors if level_k_neighbors is not None else graph_level_k
        derived_use_l3 = bool(use_l3) or int(num_graph_levels) >= 4
        arch_settings = {
            "hidden_dim": hidden_dim,
            "num_heads": heads,
            "k_neighbors": k_neighbors,
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
            "lead_conditioning": lead_conditioning,
        }
        if resolved_level_k is not None:
            arch_settings["level_k_neighbors"] = [int(x) for x in resolved_level_k]
        arch = resolve_graph_architecture(
            arch_settings
        )
        self.skip_fusion = arch.skip_fusion.asdict()
        self.pooling = arch.pooling.asdict()
        self.l0_refine_config = arch.l0_refine.asdict()
        self.lead_conditioning = arch.lead_conditioning.asdict()
        self.k_neighbors = int(arch.k_neighbors)
        self.level_k_neighbors = list(arch.level_k_neighbors)
        self.lead_conditioning_enabled = bool(arch.lead_conditioning.enabled)
        self.lead_conditioning_type = str(arch.lead_conditioning.type)
        self.lead_conditioning_max_lead = int(arch.lead_conditioning.max_lead)
        self.lead_conditioning_added_input_channels = int(arch.lead_conditioning.added_input_channels)
        self.num_graph_levels = int(arch.num_graph_levels)
        self.use_l3 = bool(arch.use_l3)
        self.use_l4 = bool(arch.num_graph_levels >= 5)
        self.l0_blocks = int(arch.l0_blocks)
        self.l1_blocks = int(arch.l1_blocks)
        self.l2_blocks = int(arch.l2_blocks)
        self.l3_blocks = int(arch.l3_blocks)
        self.l4_blocks = int(arch.l4_blocks)
        self.l3_refine_after_l4_blocks = int(arch.l3_refine_after_l4_blocks)
        self.l2_refine_after_l3_blocks = int(arch.l2_refine_after_l3_blocks)
        self.l1_refine_blocks = int(arch.l1_refine_blocks)
        self.l0_refine_blocks = int(arch.l0_refine_blocks)
        self.l0_blocks_count = int(arch.l0_blocks)
        self.l1_blocks_count = int(arch.l1_blocks)
        self.l2_blocks_count = int(arch.l2_blocks)
        self.l3_blocks_count = int(arch.l3_blocks)
        self.l4_blocks_count = int(arch.l4_blocks)
        self.l3_refine_after_l4_blocks_count = int(arch.l3_refine_after_l4_blocks)
        self.l2_refine_after_l3_blocks_count = int(arch.l2_refine_after_l3_blocks)
        self.l1_refine_blocks_count = int(arch.l1_refine_blocks)
        self.l0_refine_blocks_count = int(arch.l0_refine_blocks)
        n_levels = int(arch.num_graph_levels)
        dims = [int(hidden_dim)] * n_levels if level_dims is None else [int(x) for x in level_dims]
        hds = [int(heads)] * n_levels if level_heads is None else [int(x) for x in level_heads]
        if len(dims) != n_levels:
            raise ValueError(f"level_dims must have length num_graph_levels={n_levels}, got {len(dims)}.")
        if len(hds) != n_levels:
            raise ValueError(f"level_heads must have length num_graph_levels={n_levels}, got {len(hds)}.")
        for level_idx, (level_dim, level_head_count) in enumerate(zip(dims, hds)):
            if level_dim < 1:
                raise ValueError(f"level_dims[{level_idx}]={level_dim} must be >= 1.")
            if level_head_count < 1:
                raise ValueError(f"level_heads[{level_idx}]={level_head_count} must be >= 1.")
            if level_dim % level_head_count != 0:
                raise ValueError(
                    f"level_dims[{level_idx}]={level_dim} must be divisible by "
                    f"level_heads[{level_idx}]={level_head_count}."
                )
        # The grid<->mesh transfer only ever touches L0, so it is only L0 that has to
        # match the grid-side width. Levels 1..N-1 are free: GraphUNetProcessor already
        # projects between differing widths (MeanMaxPool(out_dim=dims[i+1]) on the way
        # down, ParentUnpoolFuse(coarse_dim=dims[i+1]) on the way up). Previously this
        # required every level to equal hidden_dim, which blocked tapered ("inverted")
        # level_dims on mesh graphs for no architectural reason. Uniform level_dims are
        # unaffected: dims[0] == hidden_dim holds there exactly as before.
        if self.mesh_mode and dims[0] != hidden_dim:
            raise ValueError(
                "mesh_encoder.enabled=true feeds the grid<->mesh transfer straight into "
                f"L0, so level_dims[0]={dims[0]} must equal hidden_dim={hidden_dim}. "
                f"Coarser levels may differ; got level_dims={dims}."
            )
        self.level_dims = list(dims)
        self.level_heads = list(hds)
        d0, h0 = dims[0], hds[0]
        self.adapter = GridNodeAdapter(
            grid_shape=grid_shape,
            input_channels=input_channels,
            output_channels=output_channels,
            n_history=n_history,
            appended_context_channels=self.lead_conditioning_added_input_channels,
        )
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.base_node_feature_channels = int(self.adapter.node_feature_channels)
        self.total_node_feature_channels = int(self.base_node_feature_channels + self.aux_feature_dim)
        self.grid_static_dim = 4 if self.graphcast_mesh_boundary else 0
        self.embedding_input_channels = self.total_node_feature_channels + self.grid_static_dim
        if self.graphcast_mesh_boundary:
            boundary_hidden = d0 * int(self._mesh_cfg.get("mlp_hidden_ratio", 2))
            self.embed = MLPLayerNorm(
                self.embedding_input_channels,
                boundary_hidden,
                d0,
            )
        elif self.boundary_mlp:
            # Arm E: two-layer GELU encoder with output LayerNorm. Only the
            # non-GraphCast-boundary path is swapped; the mesh path above keeps
            # its MLPLayerNorm so existing checkpoints stay loadable.
            self.embed = nn.Sequential(
                nn.Linear(self.total_node_feature_channels, d0),
                nn.GELU(),
                nn.Linear(d0, d0),
                nn.LayerNorm(d0),
            )
        else:
            self.embed = nn.Linear(self.total_node_feature_channels, d0)
        self.encoder = nn.ModuleList(
            [LocalGraphAttentionBlock(d0, edge_dim=edge_dim, heads=h0, edge_encoding=self.edge_encoding, attention_impl=self.attention_impl) for _ in range(encoder_blocks)]
        )
        self.processor = GraphUNetProcessor(
            hidden_dim,
            graph=graph,
            edge_dim=edge_dim,
            heads=heads,
            l0_blocks=arch.l0_blocks,
            l1_blocks=arch.l1_blocks,
            l2_blocks=arch.l2_blocks,
            l1_refine_blocks=arch.l1_refine_blocks,
            l0_refine_blocks=arch.l0_refine_blocks,
            num_graph_levels=arch.num_graph_levels,
            use_l3=arch.use_l3,
            l3_blocks=arch.l3_blocks,
            l4_blocks=arch.l4_blocks,
            l3_refine_after_l4_blocks=arch.l3_refine_after_l4_blocks,
            l2_refine_after_l3_blocks=arch.l2_refine_after_l3_blocks,
            level_dims=self.level_dims,
            level_heads=self.level_heads,
            skip_fusion=self.skip_fusion,
            pooling=self.pooling,
            l0_refine=self.l0_refine_config,
            edge_encoding=self.edge_encoding,
            attention_impl=self.attention_impl,
        )
        self.decoder = nn.ModuleList(
            [LocalGraphAttentionBlock(d0, edge_dim=edge_dim, heads=h0, edge_encoding=self.edge_encoding, attention_impl=self.attention_impl) for _ in range(decoder_blocks)]
        )
        self.grid_encoder = nn.ModuleList(
            [
                LocalGraphAttentionBlock(
                    d0, edge_dim=edge_dim, heads=h0,
                    edge_encoding=self.edge_encoding, attention_impl=self.attention_impl,
                )
                for _ in range(self.grid_attention_encoder_blocks)
            ]
        )
        self.grid_decoder = nn.ModuleList(
            [
                LocalGraphAttentionBlock(
                    d0, edge_dim=edge_dim, heads=h0,
                    edge_encoding=self.edge_encoding, attention_impl=self.attention_impl,
                )
                for _ in range(self.grid_attention_decoder_blocks)
            ]
        )
        if self.boundary_mlp:
            self.head = nn.Sequential(
                nn.Linear(d0, d0),
                nn.GELU(),
                nn.Linear(d0, output_channels),
            )
        else:
            self.head = nn.Linear(d0, output_channels)
        if self.head_init_std > 0.0:
            # Small-init the FINAL projection so the initial delta is near zero.
            # Works for both the Linear and Sequential head forms.
            final_linear = self.head if isinstance(self.head, nn.Linear) else self.head[-1]
            if not isinstance(final_linear, nn.Linear):
                raise TypeError(
                    f"head_init_std expects the final head module to be nn.Linear, got {type(final_linear)}"
                )
            nn.init.normal_(final_linear.weight, mean=0.0, std=self.head_init_std)
            if final_linear.bias is not None:
                nn.init.zeros_(final_linear.bias)
        mean = torch.zeros(output_channels, dtype=torch.float32) if delta_mean is None else delta_mean.detach().to(torch.float32).reshape(-1)
        std = torch.ones(output_channels, dtype=torch.float32) if delta_std is None else delta_std.detach().to(torch.float32).reshape(-1)
        if mean.numel() != output_channels or std.numel() != output_channels:
            raise ValueError(
                f"Delta stats must have length {output_channels}; got mean={mean.numel()} std={std.numel()}"
            )
        if torch.any(~torch.isfinite(mean)) or torch.any(~torch.isfinite(std)):
            raise ValueError("Delta stats must be finite.")
        if torch.any(std <= self.delta_norm_eps):
            raise ValueError(f"Delta std must be > eps={self.delta_norm_eps:g}.")
        self.register_buffer("delta_mean", mean)
        self.register_buffer("delta_std", std)

        # GraphCast-style bipartite grid<->mesh encoder/decoder (mesh mode only).
        # embed/encoder/processor/decoder/head and the delta+residual are reused as-is;
        # only the domain crossing is new. Grid mode never builds these.
        if self.mesh_mode and self.fixed_spherical_boundary:
            self.grid_skip_mlp = None
            self.mesh_node_init = None
            self.grid2mesh = FixedBipartiteRemap()
            self.mesh2grid = FixedBipartiteRemap()
            # Long grid skip around the mesh round trip. The conservative remap is
            # low-pass in both directions, so without a path that bypasses it the
            # grid-scale detail the grid_encoder just built is destroyed before the
            # grid_decoder can use it. One learnable scalar, gated like
            # pooling.SkipFusion: scale = max_scale * sigmoid(logit), init at
            # skip_fusion.init_scale.
            if self.grid_message_passing:
                init_scale = float(self.skip_fusion.get("init_scale", 1.0))
                max_scale = float(self.skip_fusion.get("max_scale", 2.0))
                if not 0.0 < init_scale < max_scale:
                    raise ValueError(
                        f"skip_fusion init_scale={init_scale:g} must be > 0 and "
                        f"< max_scale={max_scale:g} for the mesh grid skip."
                    )
                self.grid_skip_max_scale = max_scale
                ratio = init_scale / max_scale
                self.grid_skip_logit = nn.Parameter(
                    torch.tensor(math.log(ratio / (1.0 - ratio)), dtype=torch.float32)
                )
            else:
                self.grid_skip_max_scale = 0.0
                self.register_parameter("grid_skip_logit", None)
        elif self.mesh_mode:
            mesh_static_dim = int(self.graph.mesh_static.shape[-1])
            mlp_ratio = int(self._mesh_cfg.get("mlp_hidden_ratio", 2))
            aggregation = str(self._mesh_cfg.get("aggregation", "sum")).strip().lower()
            self.grid_skip_mlp = (
                MLPLayerNorm(
                    hidden_dim,
                    hidden_dim * mlp_ratio,
                    hidden_dim,
                )
                if self.grid_skip_mlp_enabled
                else None
            )
            if self.graphcast_mesh_boundary:
                self.mesh_node_init = MLPLayerNorm(
                    mesh_static_dim,
                    hidden_dim * mlp_ratio,
                    hidden_dim,
                )
                g2m_edge_dim = int(self.graph.g2m_edge_attr.shape[-1])
                m2g_edge_dim = int(self.graph.m2g_edge_attr.shape[-1])
                if g2m_edge_dim != 10 or m2g_edge_dim != 10:
                    raise ValueError(
                        "mesh_encoder.boundary_type='graphcast_mlp' requires 10-D "
                        "bipartite edge features (legacy 6-D + normalized distance "
                        f"+ receiver-local relative xyz), got g2m={g2m_edge_dim}, m2g={m2g_edge_dim}."
                    )
            else:
                self.mesh_node_init = nn.Linear(mesh_static_dim, hidden_dim)
                g2m_edge_dim = int(edge_dim)
                m2g_edge_dim = int(edge_dim)
            self.grid2mesh = BipartiteMP(
                hidden_dim,
                edge_dim=g2m_edge_dim,
                mlp_hidden_ratio=mlp_ratio,
                aggregation=aggregation,
                graphcast_mlp=self.graphcast_mesh_boundary,
            )
            self.mesh2grid = BipartiteMP(
                hidden_dim,
                edge_dim=m2g_edge_dim,
                mlp_hidden_ratio=mlp_ratio,
                aggregation=aggregation,
                graphcast_mlp=self.graphcast_mesh_boundary,
            )

    # --- rollout-scoped cache for the static edge projections -----------------
    # graph.edge_attr never changes, so edge_k/edge_v/edge_bias are constant
    # across the autoregressive steps of one optimizer step. See
    # LocalGraphAttention._edge_terms for why sharing one copy matters.

    def _attention_modules(self) -> list[LocalGraphAttention]:
        return [m for m in self.modules() if isinstance(m, LocalGraphAttention)]

    def edge_cache_levels(self) -> list[tuple[GraphLevel | None, nn.ModuleList]]:
        """Every ``(graph level, attention blocks)`` pair the forward pass uses.

        The encoder and decoder are attention blocks in the grid configuration,
        so they carry edge projections exactly like the processor blocks do and
        must be warmed too.
        """
        # torch.compile(model.processor) wraps the submodule; edge_cache_levels
        # lives on the original.
        processor = getattr(self.processor, "_orig_mod", self.processor)
        pairs: list[tuple[GraphLevel | None, nn.ModuleList]] = []
        grid_graph = getattr(self.graph, "grid_attention_graph", None)
        pairs.append((grid_graph, self.grid_encoder))
        pairs.append((self.graph.L0, self.encoder))
        pairs.extend(processor.edge_cache_levels())
        pairs.append((self.graph.L0, self.decoder))
        pairs.append((grid_graph, self.grid_decoder))
        return pairs

    @property
    def edge_cache_enabled(self) -> bool:
        return any(m._edge_cache is not None for m in self._attention_modules())

    def enable_edge_cache(self) -> None:
        for module in self._attention_modules():
            module._edge_cache = {}

    def clear_edge_cache(self) -> None:
        for module in self._attention_modules():
            module._edge_cache = None

    def warm_edge_cache(self) -> int:
        """Populate every enabled edge cache. Returns the number of modules warmed."""
        attention_modules = self._attention_modules()
        disabled = [m for m in attention_modules if m._edge_cache is None]
        if disabled:
            logging.warning(
                "warm_edge_cache(): %d/%d LocalGraphAttention modules have no cache "
                "enabled and were skipped; call enable_edge_cache() first.",
                len(disabled),
                len(attention_modules),
            )
        warmed: set[int] = set()
        for level, blocks in self.edge_cache_levels():
            if level is None:
                continue
            for block in blocks:
                # NodewiseRefineMLP can stand in for an l0_refine block and has no .attn.
                attn = getattr(block, "attn", None)
                if not isinstance(attn, LocalGraphAttention) or attn._edge_cache is None:
                    continue
                attn._edge_terms(level)
                warmed.add(id(attn))
        expected = len(attention_modules) - len(disabled)
        if len(warmed) != expected:
            logging.warning(
                "warm_edge_cache(): warmed %d of %d cache-enabled attention modules. An "
                "un-warmed module populates its cache lazily, which is unsafe once "
                "checkpoint_rollout_steps is on: step 2 would reuse tensors whose graph "
                "step 1's checkpoint region already freed.",
                len(warmed),
                expected,
            )
        return len(warmed)

    @contextlib.contextmanager
    def edge_cache_scope(self, warm: bool = True) -> Iterator[None]:
        """Share the static edge projections for the duration of this block.

        Two constraints decide where the scope goes, and they pull in opposite
        directions:

        * Warming must happen **inside** the autocast context the rollout runs in,
          or the cached projections carry the wrong dtype -- and **outside** every
          ``torch.utils.checkpoint`` region, so the cached tensors are ordinary
          inputs to each recomputation rather than tensors created inside step 1's
          region.
        * The scope must not close before ``backward()``. Non-reentrant checkpoint
          recomputes during backward and pairs saved tensors positionally, so a
          cache that is live on the way in and cleared on the way out changes the
          recomputed op sequence and raises ``CheckpointError``.

        ``warm=True`` suits callers that do their forward and (optional) backward
        entirely inside the scope, already under autocast -- validation, evaluation,
        tests. Training uses ``warm=False`` around forward *and* backward, and warms
        from inside the autocast region via ``Trainer._warm_edge_cache()``.

        The cache must never survive ``optimizer.step()``, hence the unconditional
        clear.
        """
        self.enable_edge_cache()
        try:
            if warm:
                self.warm_edge_cache()
            yield
        finally:
            self.clear_edge_cache()

    def set_delta_normalization(
        self,
        delta_mean: torch.Tensor,
        delta_std: torch.Tensor,
        enabled: bool = True,
        center: bool | None = None,
        eps: float | None = None,
    ) -> None:
        eps_value = self.delta_norm_eps if eps is None else float(eps)
        mean = delta_mean.detach().to(device=self.delta_mean.device, dtype=torch.float32).reshape(-1)
        std = delta_std.detach().to(device=self.delta_std.device, dtype=torch.float32).reshape(-1)
        if mean.numel() != self.output_channels or std.numel() != self.output_channels:
            raise ValueError(
                f"Delta stats must have length {self.output_channels}; got mean={mean.numel()} std={std.numel()}"
            )
        if torch.any(~torch.isfinite(mean)) or torch.any(~torch.isfinite(std)):
            raise ValueError("Delta stats must be finite.")
        if torch.any(std <= eps_value):
            raise ValueError(f"Delta std must be > eps={eps_value:g}.")
        self.delta_mean.copy_(mean)
        self.delta_std.copy_(std)
        self.use_delta_normalization = bool(enabled)
        if center is not None:
            self.delta_norm_center = bool(center)
        self.delta_norm_eps = eps_value

    def denormalize_delta_nodes(self, delta_hat_nodes: torch.Tensor) -> torch.Tensor:
        if not self.use_delta_normalization:
            return delta_hat_nodes
        delta_std = self.delta_std.to(device=delta_hat_nodes.device, dtype=delta_hat_nodes.dtype).view(1, 1, -1)
        delta_nodes = delta_hat_nodes * delta_std
        if self.delta_norm_center:
            delta_mean = self.delta_mean.to(device=delta_hat_nodes.device, dtype=delta_hat_nodes.dtype).view(1, 1, -1)
            delta_nodes = delta_nodes + delta_mean
        return delta_nodes

    @staticmethod
    def _add_diag(diagnostics_collector: object | None, name: str, tensor: torch.Tensor) -> None:
        if diagnostics_collector is not None and hasattr(diagnostics_collector, "add_embedding"):
            diagnostics_collector.add_embedding(name, tensor)

    def predict_delta_nodes_from_steps(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        aux_features: torch.Tensor | None = None,
        lead: int | torch.Tensor | None = None,
        diagnostics_collector: object | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context_grid = None
        if self.lead_conditioning_enabled:
            if lead is None:
                raise ValueError("lead is required when lead_conditioning.enabled=true.")
            context_grid = build_lead_conditioning_grid(
                lead,
                batch_size=int(current.shape[0]),
                height=int(current.shape[-2]),
                width=int(current.shape[-1]),
                max_lead=self.lead_conditioning_max_lead,
                dtype=current.dtype,
                device=current.device,
            )
        node_x, current_state_nodes = self.adapter.to_node_features_from_steps(
            previous,
            current,
            context_grid=context_grid,
        )
        return self._predict_delta_nodes_from_node_features(
            node_x,
            current_state_nodes,
            aux_features=aux_features,
            diagnostics_collector=diagnostics_collector,
        )

    def _predict_delta_nodes_from_node_features(
        self,
        node_x: torch.Tensor,
        current_state_nodes: torch.Tensor,
        aux_features: torch.Tensor | None = None,
        diagnostics_collector: object | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._add_diag(diagnostics_collector, "model/node_input", node_x)
        if aux_features is not None:
            if aux_features.dim() != 3:
                raise ValueError(f"Expected aux_features [B,N,F], got {tuple(aux_features.shape)}")
            if tuple(aux_features.shape[:2]) != tuple(node_x.shape[:2]):
                raise ValueError(
                    f"Auxiliary feature shape {tuple(aux_features.shape[:2])} does not match node input {tuple(node_x.shape[:2])}"
                )
            if int(aux_features.shape[-1]) != int(self.aux_feature_dim):
                raise ValueError(f"Expected {self.aux_feature_dim} auxiliary channels, got {aux_features.shape[-1]}")
            node_x = torch.cat([node_x, aux_features.to(device=node_x.device, dtype=node_x.dtype)], dim=-1)
        elif self.aux_feature_dim:
            raise ValueError(f"Model was initialized with aux_feature_dim={self.aux_feature_dim}, but no aux_features were provided.")
        self._add_diag(diagnostics_collector, "model/node_input_with_aux", node_x)
        if self.graphcast_mesh_boundary:
            grid_static = self.graph.grid_static.to(device=node_x.device, dtype=node_x.dtype)
            grid_static = grid_static.unsqueeze(0).expand(int(node_x.shape[0]), -1, -1)
            node_x = torch.cat([node_x, grid_static], dim=-1)
            self._add_diag(diagnostics_collector, "model/grid_input_with_geometry", node_x)
        if self.fixed_spherical_boundary and not self.grid_message_passing:
            node_x = self.grid2mesh(
                node_x,
                self.graph.g2m_edge_index,
                self.graph.g2m_edge_weight,
                self.graph.L0.num_nodes,
            )
            self._add_diag(diagnostics_collector, "model/grid2mesh_raw", node_x)
        h = self.embed(node_x)
        self._add_diag(diagnostics_collector, "model/embed", h)
        if self.mesh_mode:
            if self.grid_message_passing:
                # Grid message passing -> conservative remap of HIDDEN features ->
                # mesh U-Net -> conservative remap back -> grid message passing.
                # embed/encoder/processor/decoder/head are reused unchanged; only
                # what the fixed remap carries and where the head runs differ.
                for idx, block in enumerate(self.grid_encoder):
                    h = block(
                        h,
                        self.graph.grid_attention_graph,
                        diagnostics_collector=diagnostics_collector,
                        diagnostics_name=_diag_name(diagnostics_collector, f"model/grid_encoder.{idx}"),
                    )
                    self._add_diag(diagnostics_collector, f"model/grid_encoder.{idx}", h)
                h_grid_skip = h
                h_mesh = self.grid2mesh(
                    h,
                    self.graph.g2m_edge_index,
                    self.graph.g2m_edge_weight,
                    self.graph.L0.num_nodes,
                )
                self._add_diag(diagnostics_collector, "model/grid2mesh", h_mesh)
                for idx, block in enumerate(self.encoder):
                    h_mesh = block(
                        h_mesh,
                        self.graph.L0,
                        diagnostics_collector=diagnostics_collector,
                        diagnostics_name=_diag_name(diagnostics_collector, f"model/encoder.{idx}"),
                    )
                    self._add_diag(diagnostics_collector, f"model/encoder.{idx}", h_mesh)
                h_mesh = self.processor(h_mesh, diagnostics_collector=diagnostics_collector)
                for idx, block in enumerate(self.decoder):
                    h_mesh = block(
                        h_mesh,
                        self.graph.L0,
                        diagnostics_collector=diagnostics_collector,
                        diagnostics_name=_diag_name(diagnostics_collector, f"model/decoder.{idx}"),
                    )
                    self._add_diag(diagnostics_collector, f"model/decoder.{idx}", h_mesh)
                h = self.mesh2grid(
                    h_mesh,
                    self.graph.m2g_edge_index,
                    self.graph.m2g_edge_weight,
                    int(h_grid_skip.shape[1]),
                )
                self._add_diag(diagnostics_collector, "model/mesh2grid", h)
                grid_skip_scale = self.grid_skip_max_scale * torch.sigmoid(
                    self.grid_skip_logit.to(device=h.device, dtype=h.dtype)
                )
                h = h + grid_skip_scale * h_grid_skip
                self._add_diag(diagnostics_collector, "model/grid_skip_fused", h)
                for idx, block in enumerate(self.grid_decoder):
                    h = block(
                        h,
                        self.graph.grid_attention_graph,
                        diagnostics_collector=diagnostics_collector,
                        diagnostics_name=_diag_name(diagnostics_collector, f"model/grid_decoder.{idx}"),
                    )
                    self._add_diag(diagnostics_collector, f"model/grid_decoder.{idx}", h)
            elif self.fixed_spherical_boundary:
                for idx, block in enumerate(self.encoder):
                    h = block(
                        h,
                        self.graph.L0,
                        diagnostics_collector=diagnostics_collector,
                        diagnostics_name=_diag_name(diagnostics_collector, f"model/encoder.{idx}"),
                    )
                    self._add_diag(diagnostics_collector, f"model/encoder.{idx}", h)
                h = self.processor(h, diagnostics_collector=diagnostics_collector)
                for idx, block in enumerate(self.decoder):
                    h = block(
                        h,
                        self.graph.L0,
                        diagnostics_collector=diagnostics_collector,
                        diagnostics_name=_diag_name(diagnostics_collector, f"model/decoder.{idx}"),
                    )
                    self._add_diag(diagnostics_collector, f"model/decoder.{idx}", h)
            else:
                bsz = h.shape[0]
                for idx, block in enumerate(self.grid_encoder):
                    h = block(
                        h,
                        self.graph.grid_attention_graph,
                        diagnostics_collector=diagnostics_collector,
                        diagnostics_name=_diag_name(diagnostics_collector, f"model/grid_encoder.{idx}"),
                    )
                    self._add_diag(diagnostics_collector, f"model/grid_encoder.{idx}", h)
                # The decoder already has a long grid skip through its residual
                # destination update. Optionally enrich the carried grid latent with
                # the encoder-side node MLP used by GraphCast's Grid2Mesh GNN. Keep
                # Grid2Mesh messages sourced from the original embedding: both
                # branches are outputs of the same encoder message-passing step.
                h_grid_skip = h
                if self.grid_skip_mlp is not None:
                    h_grid_skip = h + self.grid_skip_mlp(h)
                    self._add_diag(
                        diagnostics_collector,
                        "model/grid_skip_encoded",
                        h_grid_skip,
                    )
                mesh_static = self.graph.mesh_static.to(device=h.device, dtype=h.dtype)
                h_mesh = self.mesh_node_init(mesh_static).unsqueeze(0).expand(bsz, -1, -1)
                h_mesh = self.grid2mesh(h, h_mesh, self.graph.g2m_edge_index, self.graph.g2m_edge_attr)
                self._add_diag(diagnostics_collector, "model/grid2mesh", h_mesh)
                for idx, block in enumerate(self.encoder):
                    h_mesh = block(
                        h_mesh,
                        self.graph.L0,
                        diagnostics_collector=diagnostics_collector,
                        diagnostics_name=_diag_name(diagnostics_collector, f"model/encoder.{idx}"),
                    )
                    self._add_diag(diagnostics_collector, f"model/encoder.{idx}", h_mesh)
                h_mesh = self.processor(h_mesh, diagnostics_collector=diagnostics_collector)
                for idx, block in enumerate(self.decoder):
                    h_mesh = block(
                        h_mesh,
                        self.graph.L0,
                        diagnostics_collector=diagnostics_collector,
                        diagnostics_name=_diag_name(diagnostics_collector, f"model/decoder.{idx}"),
                    )
                    self._add_diag(diagnostics_collector, f"model/decoder.{idx}", h_mesh)
                h = self.mesh2grid(
                    h_mesh,
                    h_grid_skip,
                    self.graph.m2g_edge_index,
                    self.graph.m2g_edge_attr,
                )
                self._add_diag(diagnostics_collector, "model/mesh2grid", h)
                for idx, block in enumerate(self.grid_decoder):
                    h = block(
                        h,
                        self.graph.grid_attention_graph,
                        diagnostics_collector=diagnostics_collector,
                        diagnostics_name=_diag_name(diagnostics_collector, f"model/grid_decoder.{idx}"),
                    )
                    self._add_diag(diagnostics_collector, f"model/grid_decoder.{idx}", h)
        else:
            for idx, block in enumerate(self.encoder):
                h = block(
                    h,
                    self.graph.L0,
                    diagnostics_collector=diagnostics_collector,
                    diagnostics_name=_diag_name(diagnostics_collector, f"model/encoder.{idx}"),
                )
                self._add_diag(diagnostics_collector, f"model/encoder.{idx}", h)
            h = self.processor(h, diagnostics_collector=diagnostics_collector)
            for idx, block in enumerate(self.decoder):
                h = block(
                    h,
                    self.graph.L0,
                    diagnostics_collector=diagnostics_collector,
                    diagnostics_name=_diag_name(diagnostics_collector, f"model/decoder.{idx}"),
                )
                self._add_diag(diagnostics_collector, f"model/decoder.{idx}", h)
        delta_hat = self.head(h)
        self._add_diag(diagnostics_collector, "model/head_delta_normalized", delta_hat)
        if self.fixed_spherical_boundary and not self.grid_message_passing:
            delta_hat = self.mesh2grid(
                delta_hat,
                self.graph.m2g_edge_index,
                self.graph.m2g_edge_weight,
                int(current_state_nodes.shape[1]),
            )
            self._add_diag(
                diagnostics_collector,
                "model/mesh2grid_delta_normalized",
                delta_hat,
            )
        return delta_hat, current_state_nodes

    def forward_steps(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        aux_features: torch.Tensor | None = None,
        lead: int | torch.Tensor | None = None,
        diagnostics_collector: object | None = None,
    ) -> torch.Tensor:
        delta_nodes, current_state_nodes = self.predict_delta_nodes_from_steps(
            previous,
            current,
            aux_features=aux_features,
            lead=lead,
            diagnostics_collector=diagnostics_collector,
        )
        delta_nodes = self.denormalize_delta_nodes(delta_nodes)
        self._add_diag(diagnostics_collector, "model/head_delta_denormalized", delta_nodes)
        pred_nodes = current_state_nodes + delta_nodes
        self._add_diag(diagnostics_collector, "model/pred_nodes", pred_nodes)
        return self.adapter.nodes_to_grid(pred_nodes)

    def forward(self, inp: torch.Tensor, diagnostics_collector: object | None = None) -> torch.Tensor:
        node_x, current_state_nodes = self.adapter.to_node_features(inp)
        delta_nodes, current_state_nodes = self._predict_delta_nodes_from_node_features(
            node_x,
            current_state_nodes,
            diagnostics_collector=diagnostics_collector,
        )
        delta_nodes = self.denormalize_delta_nodes(delta_nodes)
        self._add_diag(diagnostics_collector, "model/head_delta_denormalized", delta_nodes)
        pred_nodes = current_state_nodes + delta_nodes
        self._add_diag(diagnostics_collector, "model/pred_nodes", pred_nodes)
        return self.adapter.nodes_to_grid(pred_nodes)

    def fusion_gate_values(self) -> dict[str, float]:
        return self.processor.fusion_gate_values()

    def pooling_gate_values(self) -> dict[str, float]:
        return self.processor.pooling_gate_values()

    def l0_refine_residual_scale_values(self) -> dict[str, float]:
        return self.processor.l0_refine_residual_scale_values()
