import json
import runpy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from conftest import TinyCNN
from edgeopt import cli
from edgeopt.arraytypes import NDArray
from edgeopt.data import EvalDataset
from edgeopt.exceptions import ConfigError
from edgeopt.models import ModelSpec

SPEC = ModelSpec("resnet18", "vision", "test", ("input",), ("logits",))


def _feeds(n: int, seed: int = 0) -> list[dict[str, NDArray]]:
    rng = np.random.default_rng(seed)
    return [{"input": rng.standard_normal((1, 3, 224, 224)).astype(np.float32)} for _ in range(n)]


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, str]:
    monkeypatch.setattr(cli, "load_model", lambda name: (SPEC, TinyCNN().eval()))
    monkeypatch.setattr(
        cli, "load_calibration", lambda spec, data_dir, n: EvalDataset("c", _feeds(8), [0] * 8)
    )
    monkeypatch.setattr(
        cli,
        "load_eval",
        lambda spec, data_dir, limit: EvalDataset("e", _feeds(12, 1), [i % 3 for i in range(12)]),
    )
    return {"ARTIFACT_DIR": str(tmp_path / "artifacts"), "DATA_DIR": str(tmp_path / "data")}


def test_list_models(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["list-models"], {}) == 0
    out = capsys.readouterr().out
    assert "resnet18" in out and "distilbert-sst2" in out


def test_build_then_benchmark(env: dict[str, str], tmp_path: Path) -> None:
    assert cli.main(["build", "resnet18", "--variants", "fp32_opt", "int8_static"], env) == 0
    out_dir = tmp_path / "artifacts" / "resnet18"
    assert {p.stem for p in out_dir.glob("*.onnx")} == {"fp32", "fp32_opt", "int8_static"}
    argv = ["benchmark", "resnet18", "--variants", "int8_static", "--runs", "5", "--warmup", "1"]
    assert cli.main(argv, env) == 0
    report = json.loads((out_dir / "benchmark.json").read_text())
    assert [r["name"] for r in report["results"]] == ["fp32", "int8_static"]


def test_run_all_with_static_options(
    env: dict[str, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "r.json"
    argv = [
        "run-all", "resnet18", "--variants", "int8_static",
        "--calibration", "percentile", "--op-types", "Conv",
        "--runs", "5", "--warmup", "1", "--output", str(out),
    ]  # fmt: skip
    assert cli.main(argv, env) == 0
    assert "speedup" in capsys.readouterr().out
    assert json.loads(out.read_text())["n_eval_samples"] == 12


def test_build_skips_calibration_when_not_needed(
    env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("calibration data should not be loaded")

    monkeypatch.setattr(cli, "load_calibration", boom)
    assert cli.main(["build", "resnet18", "--variants", "fp32_opt"], env) == 0


def test_benchmark_without_build_fails_cleanly(
    env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["benchmark", "resnet18"], env) == 1
    assert "build variants first" in capsys.readouterr().err


def test_invalid_log_level(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["list-models"], {"LOG_LEVEL": "LOUD"}) == 1
    assert "LOG_LEVEL" in capsys.readouterr().err


def test_unknown_model_is_usage_error() -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["build", "gpt-9"], {})
    assert exc.value.code == 2


def test_select_variants(tmp_path: Path) -> None:
    avail = {"fp32": tmp_path / "a", "fp32_opt": tmp_path / "b", "int8_static": tmp_path / "c"}
    assert cli.select_variants(avail, None) == avail
    assert list(cli.select_variants(avail, ["int8_static"])) == ["fp32", "int8_static"]
    with pytest.raises(ConfigError, match="not built"):
        cli.select_variants(avail, ["int8_dynamic"])
    with pytest.raises(ConfigError, match="fp32"):
        cli.select_variants({"int8_static": tmp_path}, ["int8_static"])


def test_python_dash_m_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["edgeopt", "list-models"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("edgeopt.__main__", run_name="__main__")
    assert exc.value.code == 0
