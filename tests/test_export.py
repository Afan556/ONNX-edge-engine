from pathlib import Path

import pytest
import torch
from torch import nn

from conftest import TinyMatmul, matmul_inputs
from edgeopt.exceptions import ExportError
from edgeopt.export import export_to_onnx, verify_export
from edgeopt.session import create_session


def test_export_produces_runnable_model(cnn_path: Path) -> None:
    assert cnn_path.exists()
    assert create_session(cnn_path).get_inputs()[0].name == "input"


class Broken(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("cannot trace me")


def test_export_failure_is_wrapped(tmp_path: Path) -> None:
    with pytest.raises(ExportError, match="cannot trace me"):
        export_to_onnx(Broken(), (torch.zeros(1),), ["x"], ["y"], tmp_path / "b.onnx")


def test_checker_failure_is_wrapped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(path: str) -> None:
        raise ValueError("bad graph")

    monkeypatch.setattr("onnx.checker.check_model", boom)
    with pytest.raises(ExportError, match="bad graph"):
        export_to_onnx(TinyMatmul().eval(), matmul_inputs(), ["input"], ["y"], tmp_path / "m.onnx")


def test_parity_mismatch_detected(matmul_path: Path) -> None:
    different = TinyMatmul(seed=1).eval()  # different weights than the exported model
    with pytest.raises(ExportError, match="differs"):
        verify_export(matmul_path, different, matmul_inputs(), ["input"], atol=1e-3)


def test_parity_unrunnable_model(tmp_path: Path) -> None:
    with pytest.raises(ExportError, match="not runnable"):
        verify_export(tmp_path / "nope.onnx", TinyMatmul(), matmul_inputs(), ["input"])
