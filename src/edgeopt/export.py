"""PyTorch -> ONNX export with structural and numerical verification."""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from torch import nn

from edgeopt.exceptions import ExportError, InferenceError
from edgeopt.session import create_session

logger = logging.getLogger(__name__)

DEFAULT_OPSET = 17


def verify_export(
    onnx_path: Path,
    model: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    input_names: Sequence[str],
    atol: float = 1e-3,
) -> None:
    """Run the same inputs through PyTorch and ORT; raise if outputs diverge."""
    try:
        session = create_session(onnx_path, "none", intra_op_threads=1)
        feeds = {n: t.detach().numpy() for n, t in zip(input_names, example_inputs, strict=True)}
        ort_outputs = session.run(None, feeds)
    except InferenceError as exc:
        raise ExportError(f"Exported model is not runnable: {exc}") from exc
    with torch.no_grad():
        reference: Any = model(*example_inputs)
    ref_outputs: tuple[Any, ...] = reference if isinstance(reference, tuple) else (reference,)
    if len(ref_outputs) != len(ort_outputs):
        raise ExportError(f"Output count differs: torch={len(ref_outputs)} onnx={len(ort_outputs)}")
    for i, (ref, got) in enumerate(zip(ref_outputs, ort_outputs, strict=True)):
        max_diff = float(np.max(np.abs(ref.detach().numpy() - got)))
        if not max_diff <= atol:  # also catches NaN
            raise ExportError(f"Output {i} differs from PyTorch: max abs diff {max_diff:.2e}")
        logger.info("Export parity OK for output %d (max abs diff %.2e)", i, max_diff)


def export_to_onnx(
    model: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    input_names: Sequence[str],
    output_names: Sequence[str],
    output_path: Path,
    opset: int = DEFAULT_OPSET,
    atol: float = 1e-3,
) -> Path:
    """Export with fixed input shapes, validate the graph, and check numerical parity."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Exporting to %s (opset %d)", output_path, opset)
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                example_inputs,
                str(output_path),
                input_names=list(input_names),
                output_names=list(output_names),
                opset_version=opset,
                do_constant_folding=True,
                dynamo=False,
            )
    except Exception as exc:
        raise ExportError(f"torch.onnx.export failed: {exc}") from exc
    try:
        onnx.checker.check_model(str(output_path))
    except Exception as exc:
        raise ExportError(f"ONNX checker rejected the model: {exc}") from exc
    verify_export(output_path, model, example_inputs, input_names, atol)
    return output_path
