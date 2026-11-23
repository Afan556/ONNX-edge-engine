from pathlib import Path
from typing import Any

import numpy as np
import pytest

from edgeopt.data import NDArray
from edgeopt.exceptions import QuantizationError
from edgeopt.graph import op_counts
from edgeopt.quantize import (
    FeedCalibrationReader,
    quantize_dynamic_int8,
    quantize_static_int8,
)
from edgeopt.session import create_session


def test_reader_iterates_and_rewinds(cnn_feeds: list[dict[str, NDArray]]) -> None:
    reader = FeedCalibrationReader(cnn_feeds)
    seen = 0
    while reader.get_next() is not None:
        seen += 1
    assert seen == len(cnn_feeds)
    reader.rewind()
    assert reader.get_next() is not None


def test_reader_rejects_empty() -> None:
    with pytest.raises(QuantizationError):
        FeedCalibrationReader([])


def test_dynamic_quantization(matmul_path: Path, tmp_path: Path) -> None:
    out = quantize_dynamic_int8(matmul_path, tmp_path / "q" / "mm_dyn.onnx")
    assert "MatMulInteger" in op_counts(out)
    session = create_session(out)
    result = session.run(None, {"input": np.ones((1, 16), np.float32)})
    assert result[0].shape == (1, 16)


def test_dynamic_failure_wrapped(tmp_path: Path) -> None:
    with pytest.raises(QuantizationError):
        quantize_dynamic_int8(tmp_path / "missing.onnx", tmp_path / "o.onnx")


@pytest.mark.parametrize("method", ["minmax", "entropy", "percentile"])
def test_static_quantization(
    cnn_path: Path, tmp_path: Path, cnn_feeds: list[dict[str, NDArray]], method: Any
) -> None:
    out = quantize_static_int8(cnn_path, tmp_path / "cnn_int8.onnx", cnn_feeds, calibration=method)
    counts = op_counts(out)
    assert "QuantizeLinear" in counts and "DequantizeLinear" in counts
    session = create_session(out)
    assert session.run(None, cnn_feeds[0])[0].shape == (1, 3)


def test_static_unknown_calibration(
    cnn_path: Path, tmp_path: Path, cnn_feeds: list[dict[str, NDArray]]
) -> None:
    method: Any = "magic"
    with pytest.raises(QuantizationError, match="Unknown calibration"):
        quantize_static_int8(cnn_path, tmp_path / "o.onnx", cnn_feeds, calibration=method)


def test_static_failure_wrapped(tmp_path: Path, cnn_feeds: list[dict[str, NDArray]]) -> None:
    with pytest.raises(QuantizationError):
        quantize_static_int8(tmp_path / "missing.onnx", tmp_path / "o.onnx", cnn_feeds)


def test_static_op_type_restriction(
    cnn_path: Path, tmp_path: Path, cnn_feeds: list[dict[str, NDArray]]
) -> None:
    out = quantize_static_int8(
        cnn_path, tmp_path / "o.onnx", cnn_feeds, op_types_to_quantize=["Conv"]
    )
    assert "QuantizeLinear" in op_counts(out)
