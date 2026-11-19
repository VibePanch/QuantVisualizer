"""Framework-specific model backends."""

from quant_visualizer.backends.base import ModelBackend
from quant_visualizer.backends.tensorflow_backend import TensorFlowBackend

__all__ = ["ModelBackend", "TensorFlowBackend"]
