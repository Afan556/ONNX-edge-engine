import sys
from types import ModuleType, SimpleNamespace

import pytest

from edgeopt import memory


def test_real_peak_is_plausible() -> None:
    assert memory.peak_rss_bytes() > 10 * 1024 * 1024
    assert memory.peak_rss_mb() > 10


def test_windows_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeProcess:
        def memory_info(self) -> SimpleNamespace:
            return SimpleNamespace(peak_wset=123)

    monkeypatch.setattr("psutil.Process", FakeProcess)
    assert memory._peak_rss_windows() == 123


@pytest.mark.parametrize(("platform", "expected"), [("linux", 2048 * 1024), ("darwin", 2048)])
def test_posix_reader(monkeypatch: pytest.MonkeyPatch, platform: str, expected: int) -> None:
    fake = ModuleType("resource")
    fake.__dict__.update(RUSAGE_SELF=0, getrusage=lambda who: SimpleNamespace(ru_maxrss=2048))
    monkeypatch.setitem(sys.modules, "resource", fake)
    monkeypatch.setattr(sys, "platform", platform)
    assert memory._peak_rss_posix() == expected


@pytest.mark.parametrize(("platform", "expected"), [("win32", 1), ("linux", 2)])
def test_dispatch(monkeypatch: pytest.MonkeyPatch, platform: str, expected: int) -> None:
    monkeypatch.setattr(memory, "_peak_rss_windows", lambda: 1)
    monkeypatch.setattr(memory, "_peak_rss_posix", lambda: 2)
    monkeypatch.setattr(sys, "platform", platform)
    assert memory.peak_rss_bytes() == expected
