"""Train a small MNIST classifier and export Keras and TensorFlow Lite models.

Run with ``python examples/mnist.py`` after installing ``.[tensorflow]``.
The generated files are written to ``artifacts/mnist/``.
"""

from pathlib import Path

import tensorflow as tf


OUTPUT_DIR = Path("artifacts/mnist")


def main() -> None:
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(28, 28), name="image"),
            tf.keras.layers.Flatten(name="flatten"),
            tf.keras.layers.Dense(64, activation="relu", name="hidden"),
            tf.keras.layers.Dense(10, activation="softmax", name="digits"),
        ],
        name="mnist_classifier",
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_images, train_labels, epochs=3, validation_split=0.1)
    print("Test metrics:", model.evaluate(test_images, test_labels, verbose=2))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    keras_path = OUTPUT_DIR / "mnist.keras"
    tflite_path = OUTPUT_DIR / "mnist.tflite"
    model.save(keras_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_path.write_bytes(converter.convert())
    print(f"Saved Keras model: {keras_path}")
    print(f"Saved TFLite model: {tflite_path}")


if __name__ == "__main__":
    main()
