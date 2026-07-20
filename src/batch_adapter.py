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
        appended_context_channels: int = 0,
    ):
        super().__init__()
        self.height, self.width = int(grid_shape[0]), int(grid_shape[1])
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.n_history = int(n_history)
        self.appended_context_channels = int(appended_context_channels)
        if self.appended_context_channels < 0:
            raise ValueError(f"appended_context_channels must be >= 0, got {self.appended_context_channels}.")
        self.state_input_channels = self.input_channels - self.appended_context_channels
        if self.state_input_channels <= 0:
            raise ValueError(
                f"input_channels={input_channels} must exceed appended_context_channels={appended_context_channels}."
            )
        self.num_input_steps = self.n_history + 1
        if self.state_input_channels % self.num_input_steps != 0:
            raise ValueError(
                f"state_input_channels={self.state_input_channels} is not divisible by "
                f"n_history+1={self.num_input_steps}"
            )
        self.per_step_channels = self.state_input_channels // self.num_input_steps
        if self.output_channels > self.per_step_channels:
            raise ValueError(
                f"output_channels={output_channels} exceeds per-step input channels={self.per_step_channels}"
            )
        self.node_feature_channels = 2 * self.per_step_channels + self.appended_context_channels

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

    def _split_state_and_context(self, inp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if inp.dim() != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(inp.shape)}")
        bsz, channels, height, width = inp.shape
        if channels == self.input_channels:
            state = inp[:, : self.state_input_channels]
            context = inp[:, self.state_input_channels :] if self.appended_context_channels else None
        elif self.appended_context_channels and channels == self.state_input_channels:
            state = inp
            context = None
        else:
            expected = (
                f"{self.input_channels}"
                if not self.appended_context_channels
                else f"{self.input_channels} full channels or {self.state_input_channels} state channels"
            )
            raise ValueError(f"Expected {expected}, got {channels}")
        if (height, width) != (self.height, self.width):
            raise ValueError(f"Expected grid {(self.height, self.width)}, got {(height, width)}")
        return state, context

    def split_steps(self, inp: torch.Tensor) -> torch.Tensor:
        state, _ = self._split_state_and_context(inp)
        bsz, _, height, width = state.shape
        return state.reshape(bsz, self.num_input_steps, self.per_step_channels, height, width)

    def extract_two_steps(self, inp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        steps = self.split_steps(inp)
        current = steps[:, -1]
        previous = steps[:, -2] if steps.shape[1] >= 2 else current
        return previous, current

    def steps_to_input(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        context_grid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if previous.shape != current.shape:
            raise ValueError(f"Step shape mismatch: {tuple(previous.shape)} vs {tuple(current.shape)}")
        if previous.shape[1] != self.per_step_channels:
            raise ValueError(f"Expected {self.per_step_channels} step channels, got {previous.shape[1]}")
        state = torch.cat([previous, current], dim=1)
        if self.appended_context_channels:
            if context_grid is None:
                raise ValueError(f"context_grid is required for {self.appended_context_channels} appended channels.")
            if context_grid.shape[1] != self.appended_context_channels:
                raise ValueError(
                    f"Expected {self.appended_context_channels} context channels, got {context_grid.shape[1]}"
                )
            return torch.cat([state, context_grid.to(device=state.device, dtype=state.dtype)], dim=1)
        return state

    def to_node_features_from_steps(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        context_grid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if previous.shape != current.shape:
            raise ValueError(f"Step shape mismatch: {tuple(previous.shape)} vs {tuple(current.shape)}")
        if previous.shape[1] != self.per_step_channels:
            raise ValueError(f"Expected {self.per_step_channels} step channels, got {previous.shape[1]}")
        prev_nodes = self.grid_to_nodes(previous)
        cur_nodes = self.grid_to_nodes(current)
        parts = [prev_nodes, cur_nodes]
        if self.appended_context_channels:
            if context_grid is None:
                raise ValueError(f"context_grid is required for {self.appended_context_channels} appended channels.")
            if context_grid.shape[1] != self.appended_context_channels:
                raise ValueError(
                    f"Expected {self.appended_context_channels} context channels, got {context_grid.shape[1]}"
                )
            parts.append(self.grid_to_nodes(context_grid.to(device=current.device, dtype=current.dtype)))
        node_x = torch.cat(parts, dim=-1)
        current_state_nodes = cur_nodes[:, :, : self.output_channels]
        return node_x, current_state_nodes

    def to_node_features(self, inp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state, context = self._split_state_and_context(inp)
        steps = self.split_steps(state)
        current = steps[:, -1]
        previous = steps[:, -2] if steps.shape[1] >= 2 else current
        return self.to_node_features_from_steps(previous, current, context_grid=context)

    def check_reconstruction(self, x_grid: torch.Tensor) -> float:
        recon = self.nodes_to_grid(self.grid_to_nodes(x_grid))
        return torch.max(torch.abs(recon - x_grid)).item()
