"""Latency statistics and the measurement record shared by the worker and the benchmarker."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from edgeopt.exceptions import BenchmarkError


@dataclass(frozen=True)
class LatencyStats:
    runs: int
    mean_ms: float
    std_ms: float
    min_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float


@dataclass(frozen=True)
class Measurement:
    """One isolated run of one model: latency, session load time, and memory."""

    latency: LatencyStats
    load_ms: float
    baseline_rss_mb: float
    peak_rss_mb: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Measurement":
        try:
            return cls(
                latency=LatencyStats(**data["latency"]),
                load_ms=float(data["load_ms"]),
                baseline_rss_mb=float(data["baseline_rss_mb"]),
                peak_rss_mb=float(data["peak_rss_mb"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise BenchmarkError(f"Malformed worker output: {exc!r}") from exc


def summarize_latencies(latencies_ms: Sequence[float]) -> LatencyStats:
    if not latencies_ms:
        raise BenchmarkError("No latency samples to summarize")
    arr = np.asarray(latencies_ms, dtype=np.float64)
    p50, p95, p99 = (float(v) for v in np.percentile(arr, [50, 95, 99]))
    return LatencyStats(
        runs=len(arr),
        mean_ms=float(arr.mean()),
        std_ms=float(arr.std()),
        min_ms=float(min(latencies_ms)),
        p50_ms=p50,
        p95_ms=p95,
        p99_ms=p99,
        max_ms=float(max(latencies_ms)),
    )
