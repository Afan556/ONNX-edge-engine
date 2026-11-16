"""Peak resident memory of the current process, per platform."""

import sys
from importlib import import_module

import psutil

_MB = 1024 * 1024


def _peak_rss_windows() -> int:
    return int(psutil.Process().memory_info().peak_wset)


def _peak_rss_posix() -> int:
    # Imported lazily: the 'resource' module does not exist on Windows.
    resource = import_module("resource")
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes, Linux reports kilobytes.
    return peak if sys.platform == "darwin" else peak * 1024


def peak_rss_bytes() -> int:
    """Highest resident set size this process has reached so far, in bytes."""
    if sys.platform == "win32":
        return _peak_rss_windows()
    return _peak_rss_posix()


def peak_rss_mb() -> float:
    return peak_rss_bytes() / _MB
