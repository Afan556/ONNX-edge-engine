"""Project-specific exceptions so callers can catch failures precisely."""


class EdgeOptError(Exception):
    """Base class for all edgeopt errors."""


class ConfigError(EdgeOptError):
    """Invalid or missing configuration."""


class ModelLoadError(EdgeOptError):
    """A model could not be found or loaded."""


class DatasetError(EdgeOptError):
    """A dataset could not be downloaded, read, or was empty."""


class ExportError(EdgeOptError):
    """PyTorch -> ONNX export or its verification failed."""


class OptimizationError(EdgeOptError):
    """ONNX graph optimization failed."""


class QuantizationError(EdgeOptError):
    """INT8 quantization failed."""


class InferenceError(EdgeOptError):
    """An ONNX Runtime session could not be created or run."""
