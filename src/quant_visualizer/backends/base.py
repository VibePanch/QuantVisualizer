"""Framework adapter contract for model inspection and compression."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from quant_visualizer.graph.model_graph import ModelGraph


class ModelBackend(ABC):
    """Common interface implemented by framework-specific model adapters."""

    @abstractmethod
    def load_model(self, source: str | Path, **kwargs: Any) -> Any:
        """Load a model from a framework-supported source."""

    @abstractmethod
    def get_layers(self) -> list[Any]:
        """Return the model's layers in execution/display order."""

    @abstractmethod
    def build_graph(self) -> ModelGraph:
        """Build a framework-independent graph representation of the model."""

    @abstractmethod
    def get_weights(self, layer_name: str) -> list[Any]:
        """Return the weights belonging to one named layer."""

    @abstractmethod
    def set_layers(self, layer_names: Sequence[str]) -> list[Any]:
        """Choose layers targeted by subsequent quantization and pruning."""

    @abstractmethod
    def apply_quantization(
        self,
        destination: str | Path,
        *,
        representative_dataset: Callable[[], Iterable[Any]] | None = None,
        full_integer: bool = False,
    ) -> Path:
        """Convert and save a quantized model using the backend's quantizer."""

    @abstractmethod
    def apply_pruning(
        self, sparsity: float = 0.5, layer_names: Sequence[str] | None = None
    ) -> dict[str, int]:
        """Prune selected layer weights in place."""

    @abstractmethod
    def export_model(self, destination: str | Path, **kwargs: Any) -> Path:
        """Save the current model to a framework-supported destination."""
