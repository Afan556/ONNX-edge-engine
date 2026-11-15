"""Small helpers for inspecting ONNX files."""

from collections import Counter
from pathlib import Path

import onnx


def op_counts(path: Path) -> Counter[str]:
    """Count graph nodes by op type (e.g. to see Conv+Relu fusion or QuantizeLinear nodes)."""
    model = onnx.load(str(path), load_external_data=False)
    return Counter(node.op_type for node in model.graph.node)


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)
