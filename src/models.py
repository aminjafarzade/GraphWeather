from __future__ import annotations

import torch
from torch import nn

from .batch_adapter import GridNodeAdapter
from .graph_bundle import GraphBundle
from .layers import LocalGraphAttentionBlock
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
        encoder_blocks: int = 1,
        decoder_blocks: int = 1,
        l0_blocks: int = 2,
        l1_blocks: int = 2,
        l2_blocks: int = 1,
        l1_refine_blocks: int = 1,
        l0_refine_blocks: int = 1,
        use_delta_normalization: bool = False,
        delta_mean: torch.Tensor | None = None,
        delta_std: torch.Tensor | None = None,
        delta_norm_center: bool = False,
        delta_norm_eps: float = 1.0e-6,
    ):
        super().__init__()
        if graph.L0.height != grid_shape[0] or graph.L0.width != grid_shape[1]:
            raise ValueError(
                f"Graph L0 grid {(graph.L0.height, graph.L0.width)} does not match data grid {grid_shape}"
            )
        self.graph = graph
        self.use_delta_normalization = bool(use_delta_normalization)
        self.delta_norm_center = bool(delta_norm_center)
        self.delta_norm_eps = float(delta_norm_eps)
        self.adapter = GridNodeAdapter(
            grid_shape=grid_shape,
            input_channels=input_channels,
            output_channels=output_channels,
            n_history=n_history,
        )
        self.output_channels = int(output_channels)
        self.embed = nn.Linear(self.adapter.node_feature_channels, hidden_dim)
        self.encoder = nn.ModuleList(
            [LocalGraphAttentionBlock(hidden_dim, edge_dim=edge_dim, heads=heads) for _ in range(encoder_blocks)]
        )
        self.processor = GraphUNetProcessor(
            hidden_dim,
            graph=graph,
            edge_dim=edge_dim,
            heads=heads,
            l0_blocks=l0_blocks,
            l1_blocks=l1_blocks,
            l2_blocks=l2_blocks,
            l1_refine_blocks=l1_refine_blocks,
            l0_refine_blocks=l0_refine_blocks,
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

    def predict_delta_nodes_from_steps(self, previous: torch.Tensor, current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        node_x, current_state_nodes = self.adapter.to_node_features_from_steps(previous, current)
        h = self.embed(node_x)
        for block in self.encoder:
            h = block(h, self.graph.L0)
        h = self.processor(h)
        for block in self.decoder:
            h = block(h, self.graph.L0)
        return self.head(h), current_state_nodes

    def forward_steps(self, previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        delta_nodes, current_state_nodes = self.predict_delta_nodes_from_steps(previous, current)
        delta_nodes = self.denormalize_delta_nodes(delta_nodes)
        pred_nodes = current_state_nodes + delta_nodes
        return self.adapter.nodes_to_grid(pred_nodes)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        previous, current = self.adapter.extract_two_steps(inp)
        return self.forward_steps(previous, current)
