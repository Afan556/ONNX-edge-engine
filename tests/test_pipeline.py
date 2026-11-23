from pathlib import Path

import pytest

from conftest import TinyCNN, TinyMatmul, cnn_inputs, matmul_inputs
from edgeopt.data import NDArray
from edgeopt.exceptions import ConfigError
from edgeopt.models import ModelSpec
from edgeopt.pipeline import build_variants

VISION = ModelSpec("tiny-cnn", "vision", "test", ("input",), ("logits",))
NLP = ModelSpec("tiny-mm", "nlp", "test", ("input",), ("logits",))


def test_vision_defaults(tmp_path: Path, cnn_feeds: list[dict[str, NDArray]]) -> None:
    built = build_variants(VISION, TinyCNN().eval(), cnn_feeds, tmp_path, inputs=cnn_inputs())
    assert set(built) == {"fp32", "fp32_opt", "int8_static"}
    assert all(v.path.exists() and v.size_mb > 0 and v.node_count > 0 for v in built.values())
    assert built["fp32"].path == tmp_path / "tiny-cnn" / "fp32.onnx"


def test_nlp_dynamic_variant(tmp_path: Path) -> None:
    built = build_variants(
        NLP, TinyMatmul().eval(), [], tmp_path, variants=["int8_dynamic"], inputs=matmul_inputs()
    )
    assert set(built) == {"fp32", "int8_dynamic"}  # fp32 is always built as the base


def test_unknown_variant_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Unknown variants"):
        build_variants(NLP, TinyMatmul().eval(), [], tmp_path, variants=["int4"])


def test_static_options_forwarded(tmp_path: Path, cnn_feeds: list[dict[str, NDArray]]) -> None:
    built = build_variants(
        VISION,
        TinyCNN().eval(),
        cnn_feeds,
        tmp_path,
        variants=["int8_static"],
        inputs=cnn_inputs(),
        calibration="percentile",
        op_types=["Conv"],
    )
    assert set(built) == {"fp32", "int8_static"}
