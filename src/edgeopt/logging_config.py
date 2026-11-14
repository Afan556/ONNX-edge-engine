"""Central logging setup. Library code only calls logging.getLogger(__name__)."""

import logging

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Configure the 'edgeopt' logger. Safe to call more than once."""
    logger = logging.getLogger("edgeopt")
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
