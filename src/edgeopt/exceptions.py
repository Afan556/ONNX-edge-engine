"""Project-specific exceptions so callers can catch failures precisely."""


class EdgeOptError(Exception):
    """Base class for all edgeopt errors."""


class ConfigError(EdgeOptError):
    """Invalid or missing configuration."""


class ModelLoadError(EdgeOptError):
    """A model could not be found or loaded."""


class DatasetError(EdgeOptError):
    """A dataset could not be downloaded, read, or was empty."""
