from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from edgeopt.data import (
    IMAGENETTE_TO_IMAGENET,
    build_text_dataset,
    build_vision_dataset,
    load_text_dataset,
    load_vision_dataset,
    preprocess_image,
)
from edgeopt.exceptions import DatasetError


def _img() -> Image.Image:
    return Image.new("RGB", (300, 200), (128, 64, 32))


def fake_tokenizer(text: str, **kwargs: Any) -> dict[str, Any]:
    assert kwargs["max_length"] == 128
    ids = np.zeros((1, 128), dtype=np.int64)
    ids[0, 0] = len(text)
    return {"input_ids": ids, "attention_mask": np.ones_like(ids)}


def test_preprocess_shape_and_dtype() -> None:
    out = preprocess_image(_img())
    assert out.shape == (1, 3, 224, 224)
    assert out.dtype == np.float32


def test_preprocess_handles_non_rgb_and_small_images() -> None:
    out = preprocess_image(Image.new("L", (50, 50), 10))
    assert out.shape == (1, 3, 224, 224)


def test_build_vision_respects_limit() -> None:
    ds = build_vision_dataset([(_img(), 1)] * 5, "t", limit=3)
    assert len(ds) == 3
    assert ds.class_subset == IMAGENETTE_TO_IMAGENET
    assert set(ds.feeds[0]) == {"input"}


def test_build_vision_empty_raises() -> None:
    with pytest.raises(DatasetError):
        build_vision_dataset([], "empty")


def test_build_text_respects_limit() -> None:
    rows = [{"sentence": "good", "label": 1}, {"sentence": "bad", "label": 0}]
    ds = build_text_dataset(rows, fake_tokenizer, "t", limit=1)
    assert len(ds) == 1
    assert ds.labels == [1]
    assert ds.feeds[0]["input_ids"].dtype == np.int64
    assert ds.class_subset is None


def test_build_text_empty_raises() -> None:
    with pytest.raises(DatasetError):
        build_text_dataset([], fake_tokenizer, "empty")


def test_load_vision_wraps_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:
        raise OSError("no network")

    monkeypatch.setattr("torchvision.datasets.Imagenette", boom)
    with pytest.raises(DatasetError, match="no network"):
        load_vision_dataset("val", tmp_path)


def test_load_text_wraps_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:
        raise OSError("no network")

    monkeypatch.setattr("datasets.load_dataset", boom)
    with pytest.raises(DatasetError, match="no network"):
        load_text_dataset("validation")


@pytest.mark.integration
def test_real_vision_dataset() -> None:
    ds = load_vision_dataset("val", Path("data"), limit=8)
    assert len(ds) == 8
    assert all(0 <= y < 10 for y in ds.labels)


@pytest.mark.integration
def test_real_text_dataset() -> None:
    ds = load_text_dataset("validation", limit=8)
    assert len(ds) == 8
    assert all(y in (0, 1) for y in ds.labels)
