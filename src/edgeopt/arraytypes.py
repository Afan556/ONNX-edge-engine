"""Shared NumPy array alias, kept torch-free so the benchmark worker stays lean."""

from typing import Any, TypeAlias

import numpy.typing as npt

NDArray: TypeAlias = npt.NDArray[Any]
