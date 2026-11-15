from pathlib import Path
from typing import Any

import numpy as np
import pytest

from edgeopt.exceptions import InferenceError
from edgeopt.session import create_session


def test_session_runs(cnn_path: Path) -> None:
    session = create_session(cnn_path, "all", intra_op_threads=1)
    out = session.run(None, {"input": np.zeros((1, 3, 8, 8), dtype=np.float32)})
    assert out[0].shape == (1, 3)


def test_unknown_level_raises(cnn_path: Path) -> None:
    level: Any = "turbo"
    with pytest.raises(InferenceError, match="Unknown"):
        create_session(cnn_path, level)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(InferenceError):
        create_session(tmp_path / "missing.onnx")
