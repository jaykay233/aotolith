from __future__ import annotations

import torch


class TinyCliModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.tensor(
                [[1.0, -1.0], [0.5, 2.0], [1.5, -0.25]],
                dtype=torch.float32,
            )
        )
        self.bias = torch.nn.Parameter(torch.tensor([0.25, -0.5], dtype=torch.float32))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.relu(value @ self.weight + self.bias)


def scaled_relu(value: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.relu(value * scale)


class MaskedCliModule(torch.nn.Module):
    """A module whose forward takes a tensor keyword argument."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.tensor(
                [[1.0, -1.0], [0.5, 2.0], [1.5, -0.25]],
                dtype=torch.float32,
            )
        )
        self.bias = torch.nn.Parameter(torch.tensor([0.25, -0.5], dtype=torch.float32))

    def forward(
        self, value: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        output = value @ self.weight + self.bias
        if mask is not None:
            output = output * mask
        return torch.relu(output)
