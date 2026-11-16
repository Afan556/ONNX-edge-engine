"""Subprocess entry point: measure ONE ONNX model in a fresh interpreter.

Usage: python -m edgeopt.worker --model m.onnx --feed f.npz [--warmup N --runs N --threads N]
Prints a single JSON object to stdout. Diagnostics go to stderr.
"""
from typing import Any
import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np

from edgeopt.data import NDArray
from edgeopt.exceptions import BenchmarkError, EdgeOptError
from edgeopt.memory import peak_rss_mb
from edgeopt.session import create_session
from edgeopt.stats import Measurement, summarize_latencies

_NS_PER_MS = 1e6

def save_feed(path: Path, feed: Mapping[str, NDArray]) -> None:
    """Write a feed dict to an .npz file (the worker's input format)."""
    arrays: dict[str, Any] = dict(feed)
    np.savez(path, **arrays)


def load_feed(path: Path) -> dict[str, NDArray]:
    try:
        with np.load(path) as data:
            return {key: data[key] for key in data.files}
    except (OSError, ValueError) as exc:
        raise BenchmarkError(f"Could not read feed file {path}: {exc}") from exc


def measure(
    model_path: Path, feed: Mapping[str, NDArray], warmup: int, runs: int, threads: int
) -> Measurement:
    if runs < 1 or warmup < 0:
        raise BenchmarkError(f"Need runs >= 1 and warmup >= 0 (got runs={runs}, warmup={warmup})")
    baseline = peak_rss_mb()  # interpreter + imports, before any model is loaded
    started = time.perf_counter_ns()
    session = create_session(model_path, "all", intra_op_threads=threads)
    load_ms = (time.perf_counter_ns() - started) / _NS_PER_MS
    latencies: list[float] = []
    try:
        for _ in range(warmup):
            session.run(None, dict(feed))
        for _ in range(runs):
            t0 = time.perf_counter_ns()
            session.run(None, dict(feed))
            latencies.append((time.perf_counter_ns() - t0) / _NS_PER_MS)
    except Exception as exc:
        raise BenchmarkError(f"Inference failed during benchmark: {exc}") from exc
    return Measurement(
        latency=summarize_latencies(latencies),
        load_ms=load_ms,
        baseline_rss_mb=baseline,
        peak_rss_mb=peak_rss_mb(),
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="edgeopt.worker")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--feed", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = measure(args.model, load_feed(args.feed), args.warmup, args.runs, args.threads)
    except EdgeOptError as exc:
        print(f"worker failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(asdict(result)))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
