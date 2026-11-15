"""Environment-driven configuration. No secrets live in code."""

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from edgeopt.exceptions import ConfigError

_VALID_LEVELS = frozenset(logging._nameToLevel.keys())


@dataclass(frozen=True)
class Settings:
    aws_region: str
    s3_artifact_bucket: str | None
    log_level: str
    artifact_dir: Path
    data_dir: Path

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        source = os.environ if env is None else env
        level = source.get("LOG_LEVEL", "INFO").upper()
        if level not in _VALID_LEVELS:
            raise ConfigError(
                f"Invalid LOG_LEVEL {level!r}; expected one of {sorted(_VALID_LEVELS)}"
            )
        return cls(
            aws_region=source.get("AWS_REGION", "us-east-1"),
            s3_artifact_bucket=source.get("S3_ARTIFACT_BUCKET") or None,
            log_level=level,
            artifact_dir=Path(source.get("ARTIFACT_DIR", "artifacts")),
            data_dir=Path(source.get("DATA_DIR", "data")),
        )
