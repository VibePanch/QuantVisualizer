# Quant Visualizer

A starter toolkit for inspecting neural network layers and experimenting with
weight quantization and pruning.

## TensorFlow/Keras starter

Install the package and its TensorFlow backend:

```bash
pip install -e ".[tensorflow]"
```

The example in `examples/load_and_visualize.py` loads a saved Keras model,
builds a layer-order graph, reads layer weights, converts the whole model to a
TensorFlow Lite model with post-training dynamic-range quantization, prunes
selected Keras layer weights, and exports the pruned Keras model. Full integer
TFLite quantization can be requested by providing a representative dataset.

The current graph builder connects layers in declared order, which suits
Sequential models. Functional models with branches need connectivity tracing
before their graph edges are exact. Quantization is handled by the TensorFlow
Lite converter. Pruning zeros small-magnitude values in the selected Keras
layers.

## MNIST example and TensorFlow Lite visualization

Install the TensorFlow extra, then run `python examples/mnist.py`. This trains
a compact MNIST classifier and writes both `artifacts/mnist/mnist.keras` and
`artifacts/mnist/mnist.tflite`. The TFLite file can be loaded directly with
`TensorFlowBackend.load_model("artifacts/mnist/mnist.tflite")`; its graph
contains the operators and their tensor shapes and connectivity. TFLite models
are available for visualization and weight inspection. Quantization, pruning,
and Keras export require loading a `.keras` model instead.

`configs/default.yaml` points to the generated TFLite model. Run the MNIST
example first to create the model files, then inspect it with
`python examples/load_and_visualize.py artifacts/mnist/mnist.tflite`.
