"""Builds every optimized variant of a model into artifacts/<model>/."""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from edgeopt.data import NDArray
from edgeopt.exceptions import ConfigError
from edgeopt.export import DEFAULT_OPSET, export_to_onnx
from edgeopt.graph import file_size_mb, op_counts
from edgeopt.models import ModelSpec, Task, example_inputs
from edgeopt.optimize import optimize_graph
from edgeopt.quantize import Calibration, quantize_dynamic_int8, quantize_static_int8

logger = logging.getLogger(__name__)

ALL_VARIANTS = ("fp32", "fp32_opt", "int8_dynamic", "int8_static")

# Dynamic quantization suits transformers; for conv nets it is usually slow or unsupported on CPU.
DEFAULT_VARIANTS: dict[Task, tuple[str, ...]] = {
    "vision": ("fp32", "fp32_opt", "int8_static"),
    "nlp": ("fp32", "fp32_opt", "int8_dynamic", "int8_static"),
}


@dataclass(frozen=True)
class BuiltVariant:
    name: str
    path: Path
    size_mb: float
    node_count: int


def _describe(name: str, path: Path) -> BuiltVariant:
    return BuiltVariant(name, path, file_size_mb(path), sum(op_counts(path).values()))


def build_variants(
    spec: ModelSpec,
    model: nn.Module,
    calibration_feeds: Sequence[Mapping[str, NDArray]],
    out_dir: Path,
    variants: Sequence[str] | None = None,
    inputs: tuple[torch.Tensor, ...] | None = None,
    opset: int = DEFAULT_OPSET,
    calibration: Calibration = "minmax",
    op_types: Sequence[str] | None = None,
) -> dict[str, BuiltVariant]:
    """Export fp32 (always), then derive each requested variant from that exported model."""
    requested = tuple(variants) if variants is not None else DEFAULT_VARIANTS[spec.task]
    unknown = set(requested) - set(ALL_VARIANTS)
    if unknown:
        raise ConfigError(f"Unknown variants {sorted(unknown)}; choose from {ALL_VARIANTS}")

    model_dir = out_dir / spec.name
    fp32 = export_to_onnx(
        model,
        inputs if inputs is not None else example_inputs(spec),
        spec.input_names,
        spec.output_names,
        model_dir / "fp32.onnx",
        opset,
    )
    paths: dict[str, Path] = {"fp32": fp32}
    for name in requested:
        if name == "fp32":
            continue
        target = model_dir / f"{name}.onnx"
        if name == "fp32_opt":
            optimize_graph(fp32, target, "extended")
        elif name == "int8_dynamic":
            quantize_dynamic_int8(fp32, target)
        else:
            quantize_static_int8(
                fp32, target, calibration_feeds, calibration, op_types_to_quantize=op_types
            )
        paths[name] = target
    built = {name: _describe(name, path) for name, path in paths.items()}
    for v in built.values():
        logger.info("%-13s %8.1f MB  %5d nodes", v.name, v.size_mb, v.node_count)
    return built
