from __future__ import annotations

import torch
from torch import nn

from .architecture import resolve_graph_architecture
from .batch_adapter import GridNodeAdapter
from .graph_bundle import GraphBundle
from .layers import LocalGraphAttentionBlock
from .lead_conditioning import build_lead_conditioning_grid
from .processor import GraphUNetProcessor


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
    ):
        super().__init__()
        hidden_dim = int(hidden_dim)
        heads = int(heads)
        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={heads}.")
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
        self.embed = nn.Linear(self.total_node_feature_channels, hidden_dim)
        self.encoder = nn.ModuleList(
            [LocalGraphAttentionBlock(hidden_dim, edge_dim=edge_dim, heads=heads) for _ in range(encoder_blocks)]
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
            skip_fusion=self.skip_fusion,
            pooling=self.pooling,
            l0_refine=self.l0_refine_config,
        )
        self.decoder = nn.ModuleList(
            [LocalGraphAttentionBlock(hidden_dim, edge_dim=edge_dim, heads=heads) for _ in range(decoder_blocks)]
        )
        self.head = nn.Linear(hidden_dim, output_channels)
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
        h = self.embed(node_x)
        self._add_diag(diagnostics_collector, "model/embed", h)
        for idx, block in enumerate(self.encoder):
            h = block(
                h,
                self.graph.L0,
                diagnostics_collector=diagnostics_collector,
                diagnostics_name=f"model/encoder.{idx}",
            )
            self._add_diag(diagnostics_collector, f"model/encoder.{idx}", h)
        h = self.processor(h, diagnostics_collector=diagnostics_collector)
        for idx, block in enumerate(self.decoder):
            h = block(
                h,
                self.graph.L0,
                diagnostics_collector=diagnostics_collector,
                diagnostics_name=f"model/decoder.{idx}",
            )
            self._add_diag(diagnostics_collector, f"model/decoder.{idx}", h)
        delta_hat = self.head(h)
        self._add_diag(diagnostics_collector, "model/head_delta_normalized", delta_hat)
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
