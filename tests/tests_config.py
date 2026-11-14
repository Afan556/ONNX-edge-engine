from pathlib import Path

import pytest

from edgeopt.config import Settings
from edgeopt.exceptions import ConfigError


def test_defaults() -> None:
    s = Settings.from_env({})
    assert s.aws_region == "us-east-1"
    assert s.s3_artifact_bucket is None
    assert s.log_level == "INFO"
    assert s.data_dir == Path("data")


def test_overrides() -> None:
    s = Settings.from_env(
        {"AWS_REGION": "eu-west-1", "S3_ARTIFACT_BUCKET": "b", "LOG_LEVEL": "debug"}
    )
    assert (s.aws_region, s.s3_artifact_bucket, s.log_level) == ("eu-west-1", "b", "DEBUG")


def test_invalid_log_level() -> None:
    with pytest.raises(ConfigError):
        Settings.from_env({"LOG_LEVEL": "LOUD"})


def test_reads_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_REGION", "ap-south-1")
    assert Settings.from_env().aws_region == "ap-south-1"
