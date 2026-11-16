from dataclasses import asdict

import pytest

from edgeopt.exceptions import BenchmarkError
from edgeopt.stats import LatencyStats, Measurement, summarize_latencies


def test_summarize_known_values() -> None:
    s = summarize_latencies([float(i) for i in range(1, 101)])
    assert s.runs == 100
    assert s.p50_ms == pytest.approx(50.5)
    assert s.p95_ms == pytest.approx(95.05)
    assert s.p99_ms == pytest.approx(99.01)
    assert (s.min_ms, s.max_ms) == (1.0, 100.0)


def test_summarize_empty_raises() -> None:
    with pytest.raises(BenchmarkError):
        summarize_latencies([])


def _measurement() -> Measurement:
    lat = LatencyStats(5, 1.0, 0.1, 0.9, 1.0, 1.1, 1.2, 1.3)
    return Measurement(lat, load_ms=12.0, baseline_rss_mb=100.0, peak_rss_mb=130.0)


def test_measurement_roundtrip() -> None:
    original = _measurement()
    assert Measurement.from_dict(asdict(original)) == original


@pytest.mark.parametrize("payload", [{}, {"latency": {"runs": 1}}, {"latency": 3}])
def test_measurement_malformed(payload: dict[str, object]) -> None:
    with pytest.raises(BenchmarkError, match="Malformed"):
        Measurement.from_dict(payload)
