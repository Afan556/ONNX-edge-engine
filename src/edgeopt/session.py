"""Single place where ONNX Runtime sessions are created."""

import logging
from pathlib import Path
from typing import Literal

import onnxruntime as ort

from edgeopt.exceptions import InferenceError

logger = logging.getLogger(__name__)

OptLevel = Literal["none", "basic", "extended", "all"]

_LEVELS = {
    "none": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}


def create_session(
    path: Path,
    opt_level: OptLevel = "all",
    intra_op_threads: int | None = None,
    optimized_model_path: Path | None = None,
) -> ort.InferenceSession:
    """CPU-only session. If optimized_model_path is set, ORT writes the optimized graph there."""
    if opt_level not in _LEVELS:
        raise InferenceError(f"Unknown optimization level {opt_level!r}; use {sorted(_LEVELS)}")
    options = ort.SessionOptions()
    options.graph_optimization_level = _LEVELS[opt_level]
    options.log_severity_level = 3  # errors only
    if intra_op_threads is not None:
        options.intra_op_num_threads = intra_op_threads
    if optimized_model_path is not None:
        options.optimized_model_filepath = str(optimized_model_path)
    try:
        return ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
    except Exception as exc:
        raise InferenceError(f"Could not create session for {path}: {exc}") from exc
