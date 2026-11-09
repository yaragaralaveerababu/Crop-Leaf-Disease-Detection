"""
Standalone Inference and Prediction Engine using TensorFlow Lite / LiteRT
AcenAAr Technologies Pvt. Ltd. & Joy University
Author: Yaragarla Veerababu (2023BTIA009)

Requirements:
- Load TensorFlow Lite model on Raspberry Pi 5 / Windows simulation.
- Support LiteRT, tflite_runtime, or TensorFlow Lite interpreter.
- Accept image path or NumPy array.
- Resize to 224x224, normalize to [0, 1].
- Measure inference time in milliseconds (latency_ms).
- Return predicted class, condition, confidence, disease description, and treatment.
- Determine smart spray activation based on confidence threshold.
- Automatically log prediction to SQLite database.
- Save annotated prediction output image.
"""

import sys
import os
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Union, Tuple
import numpy as np
import cv2

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.database import log_prediction

# Determine available TFLite Interpreter backend
_INTERPRETER_TYPE = None
try:
    from ai_edge_litert.interpreter import Interpreter
    _INTERPRETER_TYPE = "ai_edge_litert"
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
        _INTERPRETER_TYPE = "tflite_runtime"
    except ImportError:
        try:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
            _INTERPRETER_TYPE = "tensorflow.lite"
        except ImportError:
            Interpreter = None
            _INTERPRETER_TYPE = "none"

class CropLeafPredictor:
    def __init__(self, model_path: Union[str, Path] = None, use_quantized: bool = True):
        """
        Initializes the TFLite / LiteRT inference engine.
        """
        if model_path is None:
            if use_quantized and config.TFLITE_QUANT_MODEL_PATH.exists():
                model_path = config.TFLITE_QUANT_MODEL_PATH
            elif config.TFLITE_MODEL_PATH.exists():
                model_path = config.TFLITE_MODEL_PATH
            else:
                model_path = config.KERAS_MODEL_PATH

        self.model_path = Path(model_path)
        self.backend = _INTERPRETER_TYPE
        self.interpreter = None
        self.keras_model = None

        if self.model_path.suffix == ".tflite" and Interpreter is not None:
            self.interpreter = Interpreter(model_path=str(self.model_path))
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.input_shape = self.input_details[0]['shape']
            self.input_dtype = self.input_details[0]['dtype']
            print(f"[InferenceEngine] Loaded TFLite model via {_INTERPRETER_TYPE}: {self.model_path.name}")
        else:
            # Fallback to Keras model if TFLite not available
            import tensorflow as tf
            from tensorflow import keras
            self.keras_model = keras.models.load_model(str(config.KERAS_MODEL_PATH))
            self.backend = "keras"
            print(f"[InferenceEngine] Loaded Keras model: {config.KERAS_MODEL_PATH.name}")

    def preprocess_image(self, image_input: Union[str, Path, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads and prepares image for model inference.
        Returns (preprocessed_batch, original_bgr_image).
        """
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found at {img_path}")
            orig_bgr = cv2.imread(str(img_path))
            if orig_bgr is None:
                raise ValueError(f"Could not decode image at {img_path}")
        elif isinstance(image_input, np.ndarray):
            orig_bgr = image_input.copy()
            if len(orig_bgr.shape) == 2:
                orig_bgr = cv2.cvtColor(orig_bgr, cv2.COLOR_GRAY2BGR)
        else:
            raise TypeError("Unsupported image input type.")

        # Resize to 224x224
        resized = cv2.resize(orig_bgr, (config.IMG_WIDTH, config.IMG_HEIGHT))
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        # Add batch dimension -> (1, 224, 224, 3)
        batch = np.expand_dims(normalized, axis=0)
        return batch, orig_bgr

    def predict(
        self,
        image_input: Union[str, Path, np.ndarray],
        device_id: str = config.DEFAULT_DEVICE_ID,
        log_to_db: bool = True,
        save_annotated_image: bool = True
    ) -> Dict[str, Any]:
        """
        Executes inference, measures latency, extracts disease info,
        evaluates spray threshold, and logs to SQLite.
        """
        batch, orig_bgr = self.preprocess_image(image_input)

        start_time = time.perf_counter()

        if self.interpreter is not None:
            # TFLite inference
            input_idx = self.input_details[0]['index']
            output_idx = self.output_details[0]['index']
            
            # Handle quantized input if required
            if self.input_dtype == np.uint8 or self.input_dtype == np.int8:
                scale, zero_point = self.input_details[0]['quantization']
                quant_batch = (batch / scale + zero_point).astype(self.input_dtype)
                self.interpreter.set_tensor(input_idx, quant_batch)
            else:
                self.interpreter.set_tensor(input_idx, batch.astype(np.float32))

            self.interpreter.invoke()
            raw_output = self.interpreter.get_tensor(output_idx)[0]
            
            # Dequantize output if necessary
            if self.output_details[0]['dtype'] == np.uint8 or self.output_details[0]['dtype'] == np.int8:
                out_scale, out_zero = self.output_details[0]['quantization']
                probabilities = (raw_output.astype(np.float32) - out_zero) * out_scale
            else:
                probabilities = raw_output
        else:
            # Keras inference
            raw_output = self.keras_model.predict(batch, verbose=0)
            probabilities = raw_output[0]

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Extract predictions
        predicted_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_idx])
        predicted_class = config.CLASS_NAMES[predicted_idx]

        disease_meta = config.DISEASE_INFO.get(predicted_class, {
            "crop": "Unknown",
            "condition": "Unknown",
            "description": "Information not available.",
            "treatment": "Consult agricultural extension specialist.",
            "prevention": "Routine sanitation.",
            "spray_required": False
        })

        condition = disease_meta["condition"]
        spray_required = disease_meta["spray_required"]

        # Spray logic: Spray only if diseased AND confidence >= threshold
        if spray_required and confidence >= config.CONFIDENCE_THRESHOLD:
            spray_action = "SPRAY_ACTIVATED"
            spray_duration_ms = 1500
        else:
            spray_action = "SPRAY_STANDBY"
            spray_duration_ms = 0

        # Class probabilities map
        class_probabilities = {
            config.CLASS_NAMES[i]: float(round(probabilities[i] * 100, 2))
            for i in range(len(config.CLASS_NAMES))
        }

        # Annotated image generation
        annotated_path = None
        if save_annotated_image:
            annotated_img = orig_bgr.copy()
            h, w = annotated_img.shape[:2]
            
            # Banner at top
            banner_color = (30, 130, 30) if condition == "Healthy" else (30, 30, 180) # BGR
            cv2.rectangle(annotated_img, (0, 0), (w, 55), banner_color, -1)
            
            title_text = f"{predicted_class.replace('___', ' - ')} | Conf: {confidence*100:.1f}%"
            cv2.putText(annotated_img, title_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            sub_text = f"Action: {spray_action} | Latency: {latency_ms:.1f}ms | Device: {device_id}"
            cv2.putText(annotated_img, sub_text, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)

            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            filename = f"pred_{timestamp_str}_{predicted_class}.jpg"
            out_file = config.PREDICTION_RESULTS_DIR / filename
            cv2.imwrite(str(out_file), annotated_img)
            annotated_path = str(out_file)

        # Log prediction to SQLite Database
        prediction_id = None
        if log_to_db:
            db_img_path = annotated_path if annotated_path else (str(image_input) if isinstance(image_input, (str, Path)) else "in_memory_capture.jpg")
            prediction_id = log_prediction(
                image_path=db_img_path,
                predicted_class=predicted_class,
                confidence_score=confidence,
                inference_time_ms=latency_ms,
                device_id=device_id
            )

        return {
            "prediction_id": prediction_id,
            "predicted_class": predicted_class,
            "condition": condition,
            "confidence_percentage": round(confidence * 100, 2),
            "confidence_score": round(confidence, 4),
            "inference_time_ms": round(latency_ms, 2),
            "spray_action": spray_action,
            "spray_duration_ms": spray_duration_ms,
            "crop": disease_meta["crop"],
            "description": disease_meta["description"],
            "treatment": disease_meta["treatment"],
            "prevention": disease_meta["prevention"],
            "device_id": device_id,
            "annotated_image_path": annotated_path,
            "class_probabilities": class_probabilities,
            "backend": self.backend
        }

# Global singleton predictor instance
_PREDICTOR_INSTANCE = None

def get_predictor() -> CropLeafPredictor:
    """Returns or lazily creates a singleton predictor instance."""
    global _PREDICTOR_INSTANCE
    if _PREDICTOR_INSTANCE is None:
        _PREDICTOR_INSTANCE = CropLeafPredictor()
    return _PREDICTOR_INSTANCE

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop Leaf Disease Inference")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--quantized", action="store_true", default=True, help="Use quantized TFLite model")
    args = parser.parse_args()

    # If no image provided, pick one from test set
    if not args.image:
        test_samples = list(config.TEST_DIR.glob("*/*.jpg"))
        if test_samples:
            target_image = str(test_samples[0])
            print(f"No image supplied. Selecting sample from test dataset: {target_image}")
        else:
            print("Please supply an image path with --image <path>")
            sys.exit(1)
    else:
        target_image = args.image

    predictor = CropLeafPredictor(use_quantized=args.quantized)
    result = predictor.predict(target_image)

    print("\n" + "=" * 55)
    print("INFERENCE RESULT")
    print("=" * 55)
    print(f"Predicted Class : {result['predicted_class']}")
    print(f"Condition       : {result['condition']}")
    print(f"Confidence      : {result['confidence_percentage']}%")
    print(f"Inference Time  : {result['inference_time_ms']} ms")
    print(f"Spray Decision  : {result['spray_action']} ({result['spray_duration_ms']} ms)")
    print(f"Backend Used    : {result['backend']}")
    print(f"Treatment       : {result['treatment']}")
    print("=" * 55)
