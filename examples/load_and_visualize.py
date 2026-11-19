"""Load a Keras model, inspect its graph, then quantize and prune weights."""

from pathlib import Path

from quant_visualizer.backends import TensorFlowBackend


def main() -> None:
    backend = TensorFlowBackend()
    backend.load_model("path/to/saved_model.keras")

    graph = backend.build_graph()
    print(f"Loaded {len(graph.nodes)} layers and {len(graph.edges)} edges.")
    for node in graph.nodes:
        print(f"{node.id}: {node.type}, {node.parameters} parameters")

    if graph.nodes:
        first = graph.nodes[0].id
        for weight in backend.get_weights(first):
            print(f"{first}: shape={weight.shape}, dtype={weight.dtype}")

    # TFLite conversion quantizes the model as a whole; it does not target
    # individual selected layers. The default is dynamic-range quantization.
    tflite_path = backend.apply_quantization(
        Path("artifacts") / "quantized_model.tflite"
    )
    print("Quantized TensorFlow Lite model:", tflite_path)

    weighted_layers = [
        layer.name for layer in backend.get_layers() if layer.get_weights()
    ]
    if not weighted_layers:
        print("The model has no weighted layers to prune.")
        return
    backend.set_layers(weighted_layers)
    print("Pruned values:", backend.apply_pruning(sparsity=0.2))
    output_dir = Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    backend.export_model(output_dir / "pruned_model.keras")


if __name__ == "__main__":
    main()
