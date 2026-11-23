"""INT8 quantization (dynamic and static) via ONNX Runtime."""

import logging
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from edgeopt.data import NDArray
from edgeopt.exceptions import QuantizationError
from edgeopt.graph import file_size_mb

logger = logging.getLogger(__name__)

Calibration = Literal["minmax", "entropy", "percentile"]
_CALIBRATION = {
    "minmax": CalibrationMethod.MinMax,
    "entropy": CalibrationMethod.Entropy,
    "percentile": CalibrationMethod.Percentile,
}


class FeedCalibrationReader(CalibrationDataReader):
    """Feeds pre-built {input_name: array} dicts to ORT's calibrator, one at a time."""

    def __init__(self, feeds: Sequence[Mapping[str, NDArray]]) -> None:
        if not feeds:
            raise QuantizationError("Static quantization needs at least one calibration sample")
        self._feeds = list(feeds)
        self._iter = iter(self._feeds)

    def get_next(self) -> dict[str, NDArray] | None:
        item = next(self._iter, None)
        return None if item is None else dict(item)

    def rewind(self) -> None:
        self._iter = iter(self._feeds)


def quantize_dynamic_int8(input_path: Path, output_path: Path) -> Path:
    """Weights -> INT8 offline; activations quantized on the fly. No calibration data."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        quantize_dynamic(
            model_input=str(input_path),
            model_output=str(output_path),
            weight_type=QuantType.QInt8,
        )
    except Exception as exc:
        raise QuantizationError(f"Dynamic quantization failed: {exc}") from exc
    logger.info(
        "Dynamic INT8: %.1f -> %.1f MB", file_size_mb(input_path), file_size_mb(output_path)
    )
    return output_path


def quantize_static_int8(
    input_path: Path,
    output_path: Path,
    calibration_feeds: Sequence[Mapping[str, NDArray]],
    calibration: Calibration = "minmax",
    per_channel: bool = True,
    op_types_to_quantize: Sequence[str] | None = None,
) -> Path:
    """QDQ static quantization. Calibration feeds must come from the TRAIN split, never eval."""
    if calibration not in _CALIBRATION:
        raise QuantizationError(f"Unknown calibration {calibration!r}; use {sorted(_CALIBRATION)}")
    reader = FeedCalibrationReader(calibration_feeds)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Static INT8: %d calibration samples, method=%s", len(calibration_feeds), calibration
    )
    try:
        with tempfile.TemporaryDirectory() as tmp:
            prepared = Path(tmp) / "prepared.onnx"
            quant_pre_process(str(input_path), str(prepared), skip_symbolic_shape=True)
            quantize_static(
                model_input=str(prepared),
                model_output=str(output_path),
                calibration_data_reader=reader,
                quant_format=QuantFormat.QDQ,
                per_channel=per_channel,
                weight_type=QuantType.QInt8,
                activation_type=QuantType.QUInt8,
                calibrate_method=_CALIBRATION[calibration],
                op_types_to_quantize=None
                if op_types_to_quantize is None
                else list(op_types_to_quantize),
            )
    except Exception as exc:
        raise QuantizationError(f"Static quantization failed: {exc}") from exc
    logger.info("Static INT8: %.1f -> %.1f MB", file_size_mb(input_path), file_size_mb(output_path))
    return output_path
