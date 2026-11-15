"""ORT graph optimizations (constant folding, operator fusion) saved as a new ONNX file."""

import logging
from pathlib import Path

from edgeopt.exceptions import InferenceError, OptimizationError
from edgeopt.graph import op_counts
from edgeopt.session import OptLevel, create_session

logger = logging.getLogger(__name__)


def optimize_graph(input_path: Path, output_path: Path, level: OptLevel = "extended") -> Path:
    """Default is 'extended': 'all' bakes in hardware-specific layouts and hurts portability."""
    if level == "none":
        raise OptimizationError("Level 'none' performs no optimization")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        create_session(input_path, level, optimized_model_path=output_path)
    except InferenceError as exc:
        raise OptimizationError(f"Optimization failed: {exc}") from exc
    if not output_path.exists():
        raise OptimizationError(f"ORT did not write an optimized model to {output_path}")
    before = sum(op_counts(input_path).values())
    after = sum(op_counts(output_path).values())
    logger.info("Optimized graph (%s): %d -> %d nodes", level, before, after)
    return output_path
