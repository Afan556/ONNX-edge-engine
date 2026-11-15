"""Model registry and loading for the supported PyTorch models."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from edgeopt.exceptions import ModelLoadError

logger = logging.getLogger(__name__)

Task = Literal["vision", "nlp"]
TEXT_MAX_LENGTH = 128


@dataclass(frozen=True)
class ModelSpec:
    name: str
    task: Task
    source: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]


REGISTRY: dict[str, ModelSpec] = {
    "resnet18": ModelSpec(
        name="resnet18",
        task="vision",
        source="torchvision:ResNet18_Weights.IMAGENET1K_V1",
        input_names=("input",),
        output_names=("logits",),
    ),
    "distilbert-sst2": ModelSpec(
        name="distilbert-sst2",
        task="nlp",
        source="distilbert-base-uncased-finetuned-sst-2-english",
        input_names=("input_ids", "attention_mask"),
        output_names=("logits",),
    ),
}


def get_spec(name: str) -> ModelSpec:
    try:
        return REGISTRY[name]
    except KeyError:
        raise ModelLoadError(f"Unknown model {name!r}. Available: {sorted(REGISTRY)}") from None


def _load_resnet18() -> nn.Module:
    from torchvision.models import ResNet18_Weights, resnet18

    model: nn.Module = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    return model


def _load_distilbert() -> nn.Module:
    from transformers import AutoModelForSequenceClassification

    # return_dict=False -> plain tuple outputs, which export cleanly to ONNX.
    model: nn.Module = AutoModelForSequenceClassification.from_pretrained(
        REGISTRY["distilbert-sst2"].source, return_dict=False, attn_implementation="eager"
    )
    return model


_LOADERS: dict[str, Callable[[], nn.Module]] = {
    "resnet18": _load_resnet18,
    "distilbert-sst2": _load_distilbert,
}


def load_model(name: str) -> tuple[ModelSpec, nn.Module]:
    """Load a registered model in eval mode."""
    spec = get_spec(name)
    logger.info("Loading model %s from %s", spec.name, spec.source)
    try:
        model = _LOADERS[spec.name]()
    except Exception as exc:
        raise ModelLoadError(f"Failed to load {spec.name!r}: {exc}") from exc
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Loaded %s (%.1fM parameters)", spec.name, n_params / 1e6)
    return spec, model


def example_inputs(spec: ModelSpec) -> tuple[torch.Tensor, ...]:
    """Dummy inputs used for ONNX tracing and smoke tests."""
    if spec.task == "vision":
        return (torch.randn(1, 3, 224, 224),)
    return (
        torch.randint(0, 1000, (1, TEXT_MAX_LENGTH), dtype=torch.int64),
        torch.ones(1, TEXT_MAX_LENGTH, dtype=torch.int64),
    )
