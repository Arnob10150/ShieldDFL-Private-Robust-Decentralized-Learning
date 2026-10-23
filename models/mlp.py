from __future__ import annotations

import copy

import torch
from torch import nn


class FlowMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        hidden_dims: tuple[int, int] = (128, 64),
        embedding_dim: int = 32,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        h1, h2 = hidden_dims
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, embedding_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def clone_model(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)


def state_dict_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

