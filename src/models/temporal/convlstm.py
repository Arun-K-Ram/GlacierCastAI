"""
ConvLSTM for spatiotemporal glacier sequence modelling.

Processes a sequence of spatial feature maps (one per timestep)
and outputs a final hidden state for the decoder.

Reference:
    Shi et al. (2015) "Convolutional LSTM Network: A Machine
    Learning Approach for Precipitation Nowcasting"
    https://arxiv.org/abs/1506.04214

Used as:
    - Ablation baseline vs Temporal Transformer
    - Faster to train, lower memory footprint
    - Better on shorter sequences (T <= 6)
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell implementing all four gates."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
    ):
        """
        Args:
            input_dim: Channels of input feature map.
            hidden_dim: Channels of hidden state.
            kernel_size: Convolutional kernel size.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        # All four gates computed in one convolution for efficiency
        self.gates = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

        # Layer norm on cell state for training stability
        self.cell_norm = nn.GroupNorm(1, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, input_dim, H, W) input at current timestep.
            h: (B, hidden_dim, H, W) previous hidden state.
            c: (B, hidden_dim, H, W) previous cell state.

        Returns:
            h_next: (B, hidden_dim, H, W)
            c_next: (B, hidden_dim, H, W)
        """
        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined)

        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)   # input gate
        f = torch.sigmoid(f)   # forget gate
        o = torch.sigmoid(o)   # output gate
        g = torch.tanh(g)      # cell gate

        c_next = f * c + i * g
        c_next = self.cell_norm(c_next)
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        return h, c


class ConvLSTM(nn.Module):
    """
    Multi-layer ConvLSTM encoder for temporal satellite sequences.

    Accepts a sequence of encoded spatial feature maps and returns
    the final hidden state used by the decoder head.

    Climate context is injected at each timestep as a spatial bias -
    a simple but effective fusion strategy for tabular + spatial data.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
        climate_dim: int = 0,
    ):
        """
        Args:
            input_dim: Channels of input spatial features.
            hidden_dim: ConvLSTM hidden state channels.
            num_layers: Number of stacked ConvLSTM layers.
            kernel_size: Convolutional kernel size.
            dropout: Spatial dropout between layers.
            climate_dim: Climate feature dimension. If > 0,
                         a projection layer fuses climate into input.
        """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Climate projection: (B, climate_dim) → (B, input_dim)
        self.climate_proj = None
        if climate_dim > 0:
            self.climate_proj = nn.Linear(climate_dim, input_dim)

        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            in_dim = input_dim if layer == 0 else hidden_dim
            self.cells.append(ConvLSTMCell(in_dim, hidden_dim, kernel_size))

        self.dropout = nn.Dropout2d(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        climate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W) sequence of spatial feature maps.
            climate: (B, T, climate_dim) optional climate per timestep.

        Returns:
            (B, hidden_dim, H, W) final hidden state.
        """
        B, T, C, H, W = x.shape
        device = x.device

        states = [
            cell.init_hidden(B, H, W, device)
            for cell in self.cells
        ]

        for t in range(T):
            inp = x[:, t]   # (B, C, H, W)

            # Inject climate as spatial bias
            if self.climate_proj is not None and climate is not None:
                ctx = self.climate_proj(climate[:, t])          # (B, C)
                ctx = ctx[:, :, None, None].expand(-1, -1, H, W)
                inp = inp + ctx

            for layer_idx, cell in enumerate(self.cells):
                h, c = states[layer_idx]
                h, c = cell(inp, h, c)
                states[layer_idx] = (h, c)
                inp = self.dropout(h)

        return states[-1][0]    # (B, hidden_dim, H, W)