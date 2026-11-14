"""Calibration and evaluation datasets, pre-formatted as ONNX Runtime feeds."""

import logging
import random
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from PIL import Image

from edgeopt.exceptions import DatasetError
from edgeopt.models import TEXT_MAX_LENGTH, get_spec

logger = logging.getLogger(__name__)

NDArray = npt.NDArray[Any]

# Imagenette's 10 classes (sorted by wnid) -> their indices in ImageNet-1k.
IMAGENETTE_TO_IMAGENET: tuple[int, ...] = (0, 217, 482, 491, 497, 566, 569, 571, 574, 701)

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


@dataclass(frozen=True)
class EvalDataset:
    """Samples as ORT feeds. class_subset restricts logits before argmax (vision only)."""

    name: str
    feeds: list[dict[str, NDArray]]
    labels: list[int]
    class_subset: tuple[int, ...] | None = None

    def __len__(self) -> int:
        return len(self.labels)


def preprocess_image(img: Image.Image) -> NDArray:
    """Resize(256) -> CenterCrop(224) -> normalize. Returns float32 (1, 3, 224, 224)."""
    img = img.convert("RGB")
    width, height = img.size
    scale = 256 / min(width, height)
    img = img.resize(
        (max(224, round(width * scale)), max(224, round(height * scale))),
        Image.Resampling.BILINEAR,
    )
    width, height = img.size
    left, top = (width - 224) // 2, (height - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr.transpose(2, 0, 1) - _MEAN) / _STD
    return arr[np.newaxis].astype(np.float32)


def build_vision_dataset(
    samples: Iterable[tuple[Image.Image, int]], name: str, limit: int | None = None
) -> EvalDataset:
    feeds: list[dict[str, NDArray]] = []
    labels: list[int] = []
    for img, label in samples:
        if limit is not None and len(feeds) >= limit:
            break
        feeds.append({"input": preprocess_image(img)})
        labels.append(int(label))
    if not feeds:
        raise DatasetError(f"Dataset {name!r} produced no samples")
    return EvalDataset(name, feeds, labels, IMAGENETTE_TO_IMAGENET)


def build_text_dataset(
    rows: Iterable[Mapping[str, Any]],
    tokenizer: Callable[..., Mapping[str, Any]],
    name: str,
    limit: int | None = None,
) -> EvalDataset:
    feeds: list[dict[str, NDArray]] = []
    labels: list[int] = []
    for row in rows:
        if limit is not None and len(feeds) >= limit:
            break
        enc = tokenizer(
            row["sentence"],
            padding="max_length",
            truncation=True,
            max_length=TEXT_MAX_LENGTH,
            return_tensors="np",
        )
        feeds.append(
            {
                "input_ids": np.asarray(enc["input_ids"], dtype=np.int64),
                "attention_mask": np.asarray(enc["attention_mask"], dtype=np.int64),
            }
        )
        labels.append(int(row["label"]))
    if not feeds:
        raise DatasetError(f"Dataset {name!r} produced no samples")
    return EvalDataset(name, feeds, labels)


def load_vision_dataset(
    split: Literal["train", "val"], root: Path, limit: int | None = None, seed: int = 0
) -> EvalDataset:
    """Imagenette 160px. With a limit, samples are shuffled (seeded) so classes are mixed."""
    logger.info("Loading Imagenette split=%s limit=%s", split, limit)
    try:
        from torchvision.datasets import Imagenette

        ds = Imagenette(root=str(root), split=split, size="160px", download=True)
        indices = list(range(len(ds)))
        if limit is not None:
            random.Random(seed).shuffle(indices)
    except Exception as exc:
        raise DatasetError(f"Failed to load Imagenette: {exc}") from exc
    return build_vision_dataset((ds[i] for i in indices), f"imagenette-{split}", limit)


def load_text_dataset(
    split: Literal["train", "validation"], limit: int | None = None, seed: int = 0
) -> EvalDataset:
    """GLUE SST-2, tokenized with the DistilBERT tokenizer."""
    logger.info("Loading SST-2 split=%s limit=%s", split, limit)
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(get_spec("distilbert-sst2").source)
        raw = load_dataset("nyu-mll/glue", "sst2", split=split)
        if limit is not None:
            raw = raw.shuffle(seed=seed)
    except Exception as exc:
        raise DatasetError(f"Failed to load SST-2: {exc}") from exc
    return build_text_dataset(raw, tokenizer, f"sst2-{split}", limit)
