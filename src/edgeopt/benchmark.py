"""Benchmark orchestration: isolated latency/RAM, accuracy, and deltas vs the fp32 baseline."""

import json
import logging
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from edgeopt.data import EvalDataset, NDArray
from edgeopt.exceptions import BenchmarkError
from edgeopt.graph import file_size_mb
from edgeopt.session import create_session
from edgeopt.stats import Measurement
from edgeopt.worker import save_feed

logger = logging.getLogger(__name__)

DEFAULT_WARMUP = 20
DEFAULT_RUNS = 200
BASELINE = "fp32"


@dataclass(frozen=True)
class VariantResult:
    name: str
    size_mb: float
    accuracy: float
    measurement: Measurement
    speedup_vs_fp32: float
    size_reduction_pct: float
    accuracy_drop_pts: float


@dataclass(frozen=True)
class BenchmarkReport:
    model: str
    dataset: str
    n_eval_samples: int
    threads: int
    warmup: int
    runs: int
    results: list[VariantResult]

    def write_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path


def measure_in_subprocess(
    model_path: Path,
    feed: Mapping[str, NDArray],
    warmup: int = DEFAULT_WARMUP,
    runs: int = DEFAULT_RUNS,
    threads: int = 1,
    timeout_s: float = 600.0,
) -> Measurement:
    """Measure latency and peak RAM in a fresh interpreter so variants cannot interfere."""
    with tempfile.TemporaryDirectory() as tmp:
        feed_path = Path(tmp) / "feed.npz"
        save_feed(feed_path, feed)
        cmd = [
            sys.executable, "-m", "edgeopt.worker",
            "--model", str(model_path),
            "--feed", str(feed_path),
            "--warmup", str(warmup),
            "--runs", str(runs),
            "--threads", str(threads),
        ]  # fmt: skip
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_s, check=False
            )
        except subprocess.TimeoutExpired as exc:
            raise BenchmarkError(f"Benchmark of {model_path} timed out after {timeout_s}s") from exc
    if proc.returncode != 0:
        tail = proc.stderr.strip()[-500:]
        raise BenchmarkError(f"Benchmark worker failed (exit {proc.returncode}): {tail}")
    lines = proc.stdout.strip().splitlines()
    if not lines:
        raise BenchmarkError("Benchmark worker produced no output")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"Benchmark worker output is not JSON: {exc}") from exc
    return Measurement.from_dict(payload)


def evaluate_accuracy(model_path: Path, dataset: EvalDataset, threads: int = 1) -> float:
    """Top-1 accuracy. For vision, logits are restricted to the dataset's class subset first."""
    if len(dataset) == 0:
        raise BenchmarkError(f"Dataset {dataset.name!r} is empty")
    correct = 0
    try:
        session = create_session(model_path, "all", intra_op_threads=threads)
        for feed, label in zip(dataset.feeds, dataset.labels, strict=True):
            logits = np.asarray(session.run(None, feed)[0])[0]
            if dataset.class_subset is not None:
                logits = logits[list(dataset.class_subset)]
            correct += int(int(np.argmax(logits)) == label)
    except Exception as exc:
        raise BenchmarkError(f"Accuracy evaluation failed for {model_path}: {exc}") from exc
    return correct / len(dataset)


def discover_variants(model_dir: Path) -> dict[str, Path]:
    """Map variant name -> file for every .onnx in a model's artifact directory."""
    if not model_dir.is_dir():
        raise BenchmarkError(f"No artifact directory at {model_dir}; build variants first")
    return {p.stem: p for p in sorted(model_dir.glob("*.onnx"))}


def benchmark_variants(
    model: str,
    paths: Mapping[str, Path],
    dataset: EvalDataset,
    warmup: int = DEFAULT_WARMUP,
    runs: int = DEFAULT_RUNS,
    threads: int = 1,
) -> BenchmarkReport:
    if BASELINE not in paths:
        raise BenchmarkError(f"A baseline variant named {BASELINE!r} is required")
    names = [BASELINE, *sorted(n for n in paths if n != BASELINE)]
    raw: dict[str, tuple[float, float, Measurement]] = {}
    for name in names:
        path = paths[name]
        logger.info("Benchmarking %s (%s)", name, path)
        accuracy = evaluate_accuracy(path, dataset, threads)
        measurement = measure_in_subprocess(path, dataset.feeds[0], warmup, runs, threads)
        raw[name] = (file_size_mb(path), accuracy, measurement)
        logger.info("%s: acc=%.4f p50=%.2f ms", name, accuracy, measurement.latency.p50_ms)

    base_size, base_acc, base_meas = raw[BASELINE]
    results = [
        VariantResult(
            name=name,
            size_mb=size,
            accuracy=acc,
            measurement=meas,
            speedup_vs_fp32=base_meas.latency.p50_ms / meas.latency.p50_ms,
            size_reduction_pct=(1 - size / base_size) * 100,
            accuracy_drop_pts=(base_acc - acc) * 100,
        )
        for name, (size, acc, meas) in raw.items()
    ]
    return BenchmarkReport(model, dataset.name, len(dataset), threads, warmup, runs, results)


def render_table(report: BenchmarkReport) -> str:
    """Plain-ASCII table (safe for any terminal) for logs and the README."""
    header = (
        f"{'variant':<13}{'size MB':>9}{'p50 ms':>9}{'p95 ms':>9}{'p99 ms':>9}"
        f"{'load ms':>9}{'RAM MB':>9}{'acc %':>8}{'speedup':>9}{'acc drop':>10}"
    )
    rows = [header, "-" * len(header)]
    for r in report.results:
        lat = r.measurement.latency
        rows.append(
            f"{r.name:<13}{r.size_mb:>9.1f}{lat.p50_ms:>9.2f}{lat.p95_ms:>9.2f}{lat.p99_ms:>9.2f}"
            f"{r.measurement.load_ms:>9.0f}{r.measurement.peak_rss_mb:>9.0f}"
            f"{r.accuracy * 100:>8.2f}{r.speedup_vs_fp32:>8.2f}x{r.accuracy_drop_pts:>10.2f}"
        )
    rows.append(
        f"model={report.model} dataset={report.dataset} n={report.n_eval_samples} "
        f"threads={report.threads} runs={report.runs} (warmup {report.warmup})"
    )
    return "\n".join(rows)
