"""Minimal adapter for TensorFlow Keras models."""

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from quant_visualizer.backends.base import ModelBackend
from quant_visualizer.graph import ModelGraph, Node


class TensorFlowBackend(ModelBackend):
    """Inspect and apply simple weight operations to a Keras model."""

    def __init__(self) -> None:
        self.model: Any | None = None
        self._selected_layer_names: tuple[str, ...] = ()

    @staticmethod
    def _tensorflow() -> Any:
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError(
                "TensorFlowBackend requires TensorFlow; install "
                "quant-visualizer[tensorflow]."
            ) from exc
        return tf

    def _require_model(self) -> Any:
        if self.model is None:
            raise RuntimeError("Load a model before calling this operation.")
        return self.model

    def load_model(self, source: str | Path, **kwargs: Any) -> Any:
        """Load a saved Keras model from a path."""
        tf = self._tensorflow()
        self.model = tf.keras.models.load_model(str(source), **kwargs)
        self._selected_layer_names = tuple(
            layer.name for layer in self.model.layers
        )
        return self.model

    def get_layers(self) -> list[Any]:
        """Return Keras layers in the model's declared order."""
        return list(self._require_model().layers)

    def build_graph(self) -> ModelGraph:
        """Build a simple layer-order graph for a Keras model.

        This is useful for Sequential models. Branched Functional models need
        tensor-connectivity tracing for exact graph edges.
        """
        layers = self.get_layers()
        nodes = []
        for layer in layers:
            try:
                input_shape = layer.input_shape
            except (AttributeError, RuntimeError, ValueError):
                input_shape = None
            try:
                output_shape = layer.output_shape
            except (AttributeError, RuntimeError, ValueError):
                output_shape = None
            nodes.append(
                Node(
                    id=layer.name,
                    type=layer.__class__.__name__,
                    input_shape=input_shape,
                    output_shape=output_shape,
                    parameters=int(layer.count_params()),
                )
            )
        edges = [
            (layers[i].name, layers[i + 1].name)
            for i in range(len(layers) - 1)
        ]
        return ModelGraph(nodes=nodes, edges=edges)

    def get_weights(self, layer_name: str) -> list[Any]:
        """Return NumPy weight arrays for a named layer."""
        return self._get_layer(layer_name).get_weights()

    def set_layers(self, layer_names: Sequence[str]) -> list[Any]:
        """Set default target layers for quantization and pruning."""
        available = {layer.name for layer in self._require_model().layers}
        names = tuple(dict.fromkeys(layer_names))
        unknown = set(names) - available
        if unknown:
            raise KeyError(f"Unknown layer name(s): {', '.join(sorted(unknown))}")
        self._selected_layer_names = names
        return [self._get_layer(name) for name in names]

    def apply_quantization(
        self,
        destination: str | Path,
        *,
        representative_dataset: Callable[[], Iterable[Any]] | None = None,
        full_integer: bool = False,
    ) -> Path:
        """Convert the whole Keras model to a quantized TensorFlow Lite file.

        By default this uses TensorFlow Lite post-training dynamic-range
        quantization. Pass a representative dataset and ``full_integer=True``
        to request full integer (int8) quantization.
        """
        if full_integer and representative_dataset is None:
            raise ValueError(
                "full_integer=True requires a representative_dataset callable"
            )
        path = Path(destination)
        if path.suffix.lower() != ".tflite":
            raise ValueError("TensorFlow Lite quantized models must use a .tflite path")

        tf = self._tensorflow()
        converter = tf.lite.TFLiteConverter.from_keras_model(self._require_model())
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if representative_dataset is not None:
            converter.representative_dataset = representative_dataset
        if full_integer:
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

        tflite_model = converter.convert()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(tflite_model)
        return path

    def apply_pruning(
        self, sparsity: float = 0.5, layer_names: Sequence[str] | None = None
    ) -> dict[str, int]:
        """Zero small-magnitude values per weight array in place."""
        if not 0.0 <= sparsity <= 1.0:
            raise ValueError("sparsity must be between 0 and 1")
        import numpy as np

        pruned: dict[str, int] = {}
        for layer in self._target_layers(layer_names):
            weights = layer.get_weights()
            updated = []
            zeroed = 0
            for weight in weights:
                if not np.issubdtype(weight.dtype, np.floating) or not weight.size:
                    updated.append(weight)
                    continue
                threshold = np.quantile(np.abs(weight), sparsity)
                mask = np.abs(weight) <= threshold
                zeroed += int(np.count_nonzero(mask & (weight != 0)))
                updated.append(np.where(mask, 0, weight))
            if weights:
                layer.set_weights(updated)
                pruned[layer.name] = zeroed
        return pruned

    def export_model(self, destination: str | Path, **kwargs: Any) -> Path:
        """Save the current Keras model to the requested path."""
        path = Path(destination)
        self._require_model().save(str(path), **kwargs)
        return path

    def _get_layer(self, layer_name: str) -> Any:
        for layer in self.get_layers():
            if layer.name == layer_name:
                return layer
        raise KeyError(f"No layer named {layer_name!r}")

    def _target_layers(self, layer_names: Sequence[str] | None) -> list[Any]:
        names = (
            tuple(layer_names)
            if layer_names is not None
            else self._selected_layer_names
        )
        # An empty selection defaults to all layers.
        return (
            [self._get_layer(name) for name in names]
            if names
            else self.get_layers()
        )
