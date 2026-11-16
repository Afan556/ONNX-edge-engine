import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from edgeopt import benchmark
from edgeopt.benchmark import (
    benchmark_variants,
    discover_variants,
    evaluate_accuracy,
    measure_in_subprocess,
    render_table,
)
from edgeopt.data import EvalDataset, NDArray
from edgeopt.exceptions import BenchmarkError
from edgeopt.graph import file_size_mb
from edgeopt.quantize import quantize_static_int8
from edgeopt.session import create_session
from edgeopt.stats import LatencyStats, Measurement


def _predictions(
    path: Path, feeds: list[dict[str, NDArray]], subset: list[int] | None
) -> list[int]:
    session = create_session(path)
    out: list[int] = []
    for feed in feeds:
        logits = np.asarray(session.run(None, feed)[0])[0]
        out.append(int(np.argmax(logits if subset is None else logits[subset])))
    return out


def _fake_measurement(p50: float) -> Measurement:
    lat = LatencyStats(10, p50, 0.0, p50, p50, p50, p50, p50)
    return Measurement(lat, load_ms=1.0, baseline_rss_mb=100.0, peak_rss_mb=150.0)


# ---- subprocess measurement -------------------------------------------------


def test_real_subprocess_measurement(cnn_path: Path, cnn_feeds: list[dict[str, NDArray]]) -> None:
    m = measure_in_subprocess(cnn_path, cnn_feeds[0], warmup=1, runs=5)
    assert m.latency.runs == 5
    assert m.peak_rss_mb >= m.baseline_rss_mb > 0


def test_subprocess_failure_reported(tmp_path: Path, cnn_feeds: list[dict[str, NDArray]]) -> None:
    with pytest.raises(BenchmarkError, match="worker failed"):
        measure_in_subprocess(tmp_path / "missing.onnx", cnn_feeds[0], warmup=0, runs=1)


def _patch_run(monkeypatch: pytest.MonkeyPatch, stdout: str, code: int = 0) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=code, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)


@pytest.mark.parametrize(
    ("stdout", "match"),
    [("", "no output"), ("not json", "not JSON"), ('{"x": 1}', "Malformed")],
)
def test_subprocess_bad_output(
    monkeypatch: pytest.MonkeyPatch,
    cnn_feeds: list[dict[str, NDArray]],
    stdout: str,
    match: str,
) -> None:
    _patch_run(monkeypatch, stdout)
    with pytest.raises(BenchmarkError, match=match):
        measure_in_subprocess(Path("m.onnx"), cnn_feeds[0])


def test_subprocess_timeout(
    monkeypatch: pytest.MonkeyPatch, cnn_feeds: list[dict[str, NDArray]]
) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd="x", timeout=1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(BenchmarkError, match="timed out"):
        measure_in_subprocess(Path("m.onnx"), cnn_feeds[0])


# ---- accuracy ---------------------------------------------------------------


def test_accuracy_perfect_and_zero(cnn_path: Path, cnn_feeds: list[dict[str, NDArray]]) -> None:
    preds = _predictions(cnn_path, cnn_feeds, None)
    assert evaluate_accuracy(cnn_path, EvalDataset("t", cnn_feeds, preds)) == 1.0
    wrong = [(p + 1) % 3 for p in preds]
    assert evaluate_accuracy(cnn_path, EvalDataset("t", cnn_feeds, wrong)) == 0.0


def test_accuracy_respects_class_subset(
    cnn_path: Path, cnn_feeds: list[dict[str, NDArray]]
) -> None:
    labels = _predictions(cnn_path, cnn_feeds, [1, 2])
    ds = EvalDataset("t", cnn_feeds, labels, class_subset=(1, 2))
    assert evaluate_accuracy(cnn_path, ds) == 1.0


def test_accuracy_empty_dataset(cnn_path: Path) -> None:
    with pytest.raises(BenchmarkError, match="empty"):
        evaluate_accuracy(cnn_path, EvalDataset("t", [], []))


def test_accuracy_wraps_failures(tmp_path: Path, cnn_feeds: list[dict[str, NDArray]]) -> None:
    ds = EvalDataset("t", cnn_feeds, [0] * len(cnn_feeds))
    with pytest.raises(BenchmarkError, match="Accuracy evaluation failed"):
        evaluate_accuracy(tmp_path / "missing.onnx", ds)


# ---- orchestration ----------------------------------------------------------


def test_discover_variants(tmp_path: Path) -> None:
    (tmp_path / "fp32.onnx").touch()
    (tmp_path / "int8_static.onnx").touch()
    (tmp_path / "notes.txt").touch()
    assert set(discover_variants(tmp_path)) == {"fp32", "int8_static"}


def test_discover_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkError, match="build variants"):
        discover_variants(tmp_path / "nope")


def test_benchmark_variants_computes_deltas(
    cnn_path: Path,
    tmp_path: Path,
    cnn_feeds: list[dict[str, NDArray]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    int8 = quantize_static_int8(cnn_path, tmp_path / "int8_static.onnx", cnn_feeds)
    labels = _predictions(cnn_path, cnn_feeds, None)
    ds = EvalDataset("tiny", cnn_feeds, labels)

    def fake_measure(path: Path, *args: Any, **kwargs: Any) -> Measurement:
        return _fake_measurement(5.0 if "int8" in path.name else 10.0)

    monkeypatch.setattr(benchmark, "measure_in_subprocess", fake_measure)
    report = benchmark_variants("tiny", {"int8_static": int8, "fp32": cnn_path}, ds)

    assert [r.name for r in report.results] == ["fp32", "int8_static"]
    base, quant = report.results
    assert base.speedup_vs_fp32 == pytest.approx(1.0)
    assert base.accuracy == 1.0 and base.accuracy_drop_pts == 0.0
    assert quant.speedup_vs_fp32 == pytest.approx(2.0)
    expected = (1 - file_size_mb(int8) / file_size_mb(cnn_path)) * 100
    assert quant.size_reduction_pct == pytest.approx(expected)
    assert quant.accuracy_drop_pts >= 0.0

    out = report.write_json(tmp_path / "reports" / "b.json")
    assert json.loads(out.read_text())["model"] == "tiny"
    table = render_table(report)
    assert "fp32" in table and "int8_static" in table and "speedup" in table


def test_benchmark_requires_baseline(tmp_path: Path, cnn_feeds: list[dict[str, NDArray]]) -> None:
    ds = EvalDataset("t", cnn_feeds, [0] * len(cnn_feeds))
    with pytest.raises(BenchmarkError, match="baseline"):
        benchmark_variants("m", {"int8_static": tmp_path / "x.onnx"}, ds)
