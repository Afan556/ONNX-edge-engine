from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from edgeopt.data import NDArray
from edgeopt.export import export_to_onnx


class TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.fc(self.pool(torch.relu(self.conv(x))).flatten(1))
        return out


class TinyMatmul(nn.Module):
    """Plain MatMul so dynamic quantization has something it supports on CPU."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.weight = nn.Parameter(torch.randn(16, 16, generator=gen))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ self.weight)


def cnn_inputs() -> tuple[torch.Tensor, ...]:
    return (torch.randn(1, 3, 8, 8),)


def matmul_inputs() -> tuple[torch.Tensor, ...]:
    return (torch.randn(1, 16),)


@pytest.fixture
def cnn_path(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    return export_to_onnx(
        TinyCNN().eval(), cnn_inputs(), ["input"], ["logits"], tmp_path / "cnn.onnx"
    )


@pytest.fixture
def matmul_path(tmp_path: Path) -> Path:
    return export_to_onnx(
        TinyMatmul().eval(), matmul_inputs(), ["input"], ["logits"], tmp_path / "mm.onnx"
    )


@pytest.fixture
def cnn_feeds() -> list[dict[str, NDArray]]:
    rng = np.random.default_rng(0)
    return [{"input": rng.standard_normal((1, 3, 8, 8)).astype(np.float32)} for _ in range(8)]
