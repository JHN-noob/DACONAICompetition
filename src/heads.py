from __future__ import annotations

import torch
import torch.nn as nn


def build_activation(name: str) -> nn.Module:
    if name.lower() == "relu":
        return nn.ReLU()
    return nn.GELU()


class ViewAdapter(nn.Module):
    def __init__(self, input_dim: int, drop_rate: float, activation: str = "gelu") -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            build_activation(activation),
            nn.Dropout(drop_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, drop_rate: float, activation: str = "gelu") -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            build_activation(activation),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
