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
    ):
        super().__init__()
        if graph.L0.height != grid_shape[0] or graph.L0.width != grid_shape[1]:
            raise ValueError(
                f"Graph L0 grid {(graph.L0.height, graph.L0.width)} does not match data grid {grid_shape}"
            )
        self.graph = graph
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
        pred_nodes = current_state_nodes + delta_nodes
        return self.adapter.nodes_to_grid(pred_nodes)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        previous, current = self.adapter.extract_two_steps(inp)
        return self.forward_steps(previous, current)

