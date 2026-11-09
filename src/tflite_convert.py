"""
TensorFlow Lite Conversion and Optimization Pipeline
AcenAAr Technologies Pvt. Ltd. & Joy University
Author: Yaragarla Veerababu (2023BTIA009)

Requirements:
- Convert trained Keras model to TensorFlow Lite (.tflite).
- Apply post-training dynamic range quantization.
- Log model sizes (original Keras vs TFLite vs Quantized TFLite).
- Verify compatibility for edge deployment on Raspberry Pi 5.
"""

import sys
import os
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import tensorflow as tf
from tensorflow import keras

def convert_to_tflite():
    """Converts the Keras model to standard and quantized TFLite formats."""
    print("=" * 70)
    print("Converting MobileNetV2 Model to TensorFlow Lite for Raspberry Pi 5")
    print("=" * 70)

    if not config.KERAS_MODEL_PATH.exists():
        raise FileNotFoundError(f"Keras model not found at {config.KERAS_MODEL_PATH}. Train the model first.")

    print(f"Loading Keras model from {config.KERAS_MODEL_PATH}...")
    model = keras.models.load_model(str(config.KERAS_MODEL_PATH))

    # 1. Standard TFLite Conversion (Float32)
    print("\n[Step 1/2] Converting to Standard Float32 TFLite Model...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(config.TFLITE_MODEL_PATH, "wb") as f:
        f.write(tflite_model)
    print(f"Standard TFLite model saved to: {config.TFLITE_MODEL_PATH}")

    # 2. Post-Training Quantized TFLite (Dynamic Range Quantization)
    print("\n[Step 2/2] Converting to Post-Training Quantized TFLite Model...")
    converter_quant = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter_quant.convert()

    with open(config.TFLITE_QUANT_MODEL_PATH, "wb") as f:
        f.write(tflite_quant_model)
    print(f"Quantized TFLite model saved to: {config.TFLITE_QUANT_MODEL_PATH}")

    # 3. Model Size Comparison
    keras_size_mb = os.path.getsize(config.KERAS_MODEL_PATH) / (1024 * 1024)
    tflite_size_mb = os.path.getsize(config.TFLITE_MODEL_PATH) / (1024 * 1024)
    quant_size_mb = os.path.getsize(config.TFLITE_QUANT_MODEL_PATH) / (1024 * 1024)

    print("\n" + "=" * 50)
    print("MODEL COMPRESSION & EDGE READINESS SUMMARY")
    print("=" * 50)
    print(f"Original Keras Model (.keras) : {keras_size_mb:.2f} MB")
    print(f"Standard TFLite Model (.tflite): {tflite_size_mb:.2f} MB  (Saved {((keras_size_mb - tflite_size_mb)/keras_size_mb)*100:.1f}%)")
    print(f"Quantized TFLite Model (INT8)  : {quant_size_mb:.2f} MB  (Saved {((keras_size_mb - quant_size_mb)/keras_size_mb)*100:.1f}%)")
    print("=" * 50)
    print("Optimization Complete! Ready for Raspberry Pi 5 edge execution.")

    return {
        "keras_size_mb": keras_size_mb,
        "tflite_size_mb": tflite_size_mb,
        "quant_size_mb": quant_size_mb
    }

if __name__ == "__main__":
    convert_to_tflite()
