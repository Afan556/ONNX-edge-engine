import logging

from edgeopt.logging_config import configure_logging


def test_configure_is_idempotent() -> None:
    configure_logging("DEBUG")
    configure_logging("INFO")
    logger = logging.getLogger("edgeopt")
    assert len(logger.handlers) == 1
    assert logger.level == logging.INFO
