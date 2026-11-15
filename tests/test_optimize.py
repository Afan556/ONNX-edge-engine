from pathlib import Path

import numpy as np
import pytest

from edgeopt.exceptions import OptimizationError
from edgeopt.graph import file_size_mb, op_counts
from edgeopt.optimize import optimize_graph
from edgeopt.session import create_session


def test_optimize_writes_runnable_smaller_or_equal_graph(cnn_path: Path, tmp_path: Path) -> None:
    out = optimize_graph(cnn_path, tmp_path / "opt" / "cnn_opt.onnx")
    assert out.exists()
    assert sum(op_counts(out).values()) <= sum(op_counts(cnn_path).values())
    assert file_size_mb(out) > 0
    session = create_session(out)
    assert session.run(None, {"input": np.zeros((1, 3, 8, 8), np.float32)})[0].shape == (1, 3)


def test_level_none_rejected(cnn_path: Path, tmp_path: Path) -> None:
    with pytest.raises(OptimizationError, match="none"):
        optimize_graph(cnn_path, tmp_path / "o.onnx", "none")


def test_missing_input_wrapped(tmp_path: Path) -> None:
    with pytest.raises(OptimizationError):
        optimize_graph(tmp_path / "missing.onnx", tmp_path / "o.onnx")
