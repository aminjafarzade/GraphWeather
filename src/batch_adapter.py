from __future__ import annotations

import torch
from torch import nn


class GridNodeAdapter(nn.Module):
    """Convert KAI-style grid tensors to graph node tensors and back."""

    def __init__(
        self,
        grid_shape: tuple[int, int],
        input_channels: int,
        output_channels: int,
        n_history: int,
    ):
        super().__init__()
        self.height, self.width = int(grid_shape[0]), int(grid_shape[1])
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.n_history = int(n_history)
        self.num_input_steps = self.n_history + 1
        if self.input_channels % self.num_input_steps != 0:
            raise ValueError(
                f"input_channels={input_channels} is not divisible by n_history+1={self.num_input_steps}"
            )
        self.per_step_channels = self.input_channels // self.num_input_steps
        if self.output_channels > self.per_step_channels:
            raise ValueError(
                f"output_channels={output_channels} exceeds per-step input channels={self.per_step_channels}"
            )
        self.node_feature_channels = 2 * self.per_step_channels

    @property
    def num_nodes(self) -> int:
        return self.height * self.width

    def grid_to_nodes(self, x_grid: torch.Tensor) -> torch.Tensor:
        if x_grid.dim() != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(x_grid.shape)}")
        bsz, _, height, width = x_grid.shape
        if (height, width) != (self.height, self.width):
            raise ValueError(f"Expected grid {(self.height, self.width)}, got {(height, width)}")
        return x_grid.permute(0, 2, 3, 1).reshape(bsz, self.num_nodes, x_grid.shape[1])

    def nodes_to_grid(self, x_nodes: torch.Tensor) -> torch.Tensor:
        if x_nodes.dim() != 3:
            raise ValueError(f"Expected [B,N,C], got {tuple(x_nodes.shape)}")
        bsz, nodes, channels = x_nodes.shape
        if nodes != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {nodes}")
        return x_nodes.reshape(bsz, self.height, self.width, channels).permute(0, 3, 1, 2)

    def split_steps(self, inp: torch.Tensor) -> torch.Tensor:
        if inp.dim() != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(inp.shape)}")
        bsz, channels, height, width = inp.shape
        if channels != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} channels, got {channels}")
        if (height, width) != (self.height, self.width):
            raise ValueError(f"Expected grid {(self.height, self.width)}, got {(height, width)}")
        return inp.reshape(bsz, self.num_input_steps, self.per_step_channels, height, width)

    def extract_two_steps(self, inp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        steps = self.split_steps(inp)
        current = steps[:, -1]
        previous = steps[:, -2] if steps.shape[1] >= 2 else current
        return previous, current

    def steps_to_input(self, previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        if previous.shape != current.shape:
            raise ValueError(f"Step shape mismatch: {tuple(previous.shape)} vs {tuple(current.shape)}")
        if previous.shape[1] != self.per_step_channels:
            raise ValueError(f"Expected {self.per_step_channels} step channels, got {previous.shape[1]}")
        return torch.cat([previous, current], dim=1)

    def to_node_features_from_steps(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prev_nodes = self.grid_to_nodes(previous)
        cur_nodes = self.grid_to_nodes(current)
        node_x = torch.cat([prev_nodes, cur_nodes], dim=-1)
        current_state_nodes = cur_nodes[:, :, : self.output_channels]
        return node_x, current_state_nodes

    def to_node_features(self, inp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        previous, current = self.extract_two_steps(inp)
        return self.to_node_features_from_steps(previous, current)

    def check_reconstruction(self, x_grid: torch.Tensor) -> float:
        recon = self.nodes_to_grid(self.grid_to_nodes(x_grid))
        return torch.max(torch.abs(recon - x_grid)).item()

