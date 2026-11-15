import pytest
import torch
from torch import nn

from edgeopt import models
from edgeopt.exceptions import ModelLoadError
from edgeopt.models import example_inputs, get_spec, load_model


def test_get_spec_unknown() -> None:
    with pytest.raises(ModelLoadError, match="Available"):
        get_spec("nope")


def test_load_model_sets_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(models._LOADERS, "resnet18", lambda: nn.Linear(2, 2))
    spec, model = load_model("resnet18")
    assert spec.name == "resnet18"
    assert not model.training


def test_load_model_wraps_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> nn.Module:
        raise OSError("offline")

    monkeypatch.setitem(models._LOADERS, "resnet18", boom)
    with pytest.raises(ModelLoadError, match="offline"):
        load_model("resnet18")


def test_example_inputs_shapes() -> None:
    (img,) = example_inputs(get_spec("resnet18"))
    assert img.shape == (1, 3, 224, 224)
    ids, mask = example_inputs(get_spec("distilbert-sst2"))
    assert ids.shape == mask.shape == (1, 128)
    assert ids.dtype == torch.int64


@pytest.mark.integration
def test_real_resnet18_forward() -> None:
    spec, model = load_model("resnet18")
    with torch.no_grad():
        out = model(*example_inputs(spec))
    assert out.shape == (1, 1000)


@pytest.mark.integration
def test_real_distilbert_forward() -> None:
    spec, model = load_model("distilbert-sst2")
    with torch.no_grad():
        out = model(*example_inputs(spec))
    assert out[0].shape == (1, 2)
