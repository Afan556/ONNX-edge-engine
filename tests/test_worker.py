import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from edgeopt.data import NDArray
from edgeopt.exceptions import BenchmarkError
from edgeopt.worker import load_feed, main, measure, save_feed


def test_worker_does_not_import_torch() -> None:
    code = "import sys, edgeopt.worker; sys.exit(int('torch' in sys.modules))"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_measure_returns_sane_numbers(cnn_path: Path, cnn_feeds: list[dict[str, NDArray]]) -> None:
    m = measure(cnn_path, cnn_feeds[0], warmup=1, runs=5, threads=1)
    assert m.latency.runs == 5
    assert 0 < m.latency.min_ms <= m.latency.p50_ms <= m.latency.max_ms
    assert m.load_ms > 0
    assert 0 < m.baseline_rss_mb <= m.peak_rss_mb


@pytest.mark.parametrize(("warmup", "runs"), [(0, 0), (-1, 5)])
def test_measure_rejects_bad_counts(
    cnn_path: Path, cnn_feeds: list[dict[str, NDArray]], warmup: int, runs: int
) -> None:
    with pytest.raises(BenchmarkError, match="runs >= 1"):
        measure(cnn_path, cnn_feeds[0], warmup=warmup, runs=runs, threads=1)


def test_measure_wraps_inference_failure(cnn_path: Path) -> None:
    bad_feed = {"wrong_name": np.zeros((1, 3, 8, 8), dtype=np.float32)}
    with pytest.raises(BenchmarkError, match="Inference failed"):
        measure(cnn_path, bad_feed, warmup=0, runs=1, threads=1)


def test_load_feed_roundtrip(tmp_path: Path, cnn_feeds: list[dict[str, NDArray]]) -> None:
    path = tmp_path / "f.npz"
    save_feed(path, cnn_feeds[0])
    loaded = load_feed(path)
    assert set(loaded) == {"input"}
    assert np.array_equal(loaded["input"], cnn_feeds[0]["input"])


def test_load_feed_missing(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkError, match="feed file"):
        load_feed(tmp_path / "nope.npz")


def test_main_prints_json(
    cnn_path: Path,
    tmp_path: Path,
    cnn_feeds: list[dict[str, NDArray]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    feed = tmp_path / "f.npz"
    save_feed(feed, cnn_feeds[0])
    argv = ["--model", str(cnn_path), "--feed", str(feed), "--warmup", "1", "--runs", "5"]
    assert main(argv) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["latency"]["runs"] == 5


def test_main_reports_failure(
    tmp_path: Path, cnn_feeds: list[dict[str, NDArray]], capsys: pytest.CaptureFixture[str]
) -> None:
    feed = tmp_path / "f.npz"
    save_feed(feed, cnn_feeds[0])
    code = main(["--model", str(tmp_path / "missing.onnx"), "--feed", str(feed)])
    assert code == 1
    assert "missing.onnx" in capsys.readouterr().err


def test_main_missing_feed(tmp_path: Path, cnn_path: Path) -> None:
    assert main(["--model", str(cnn_path), "--feed", str(tmp_path / "nope.npz")]) == 1
