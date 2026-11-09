"""
Model Evaluation and Metrics Generation Pipeline
AcenAAr Technologies Pvt. Ltd. & Joy University
Author: Yaragarla Veerababu (2023BTIA009)

Requirements:
- Evaluate on unseen test dataset.
- Generate and save:
  1. Training & Validation Accuracy Graph -> outputs/accuracy.png
  2. Training & Validation Loss Graph -> outputs/loss.png
  3. Confusion Matrix -> outputs/confusion_matrix.png
  4. Classification Report -> outputs/classification_report.txt
- Calculate: Test Accuracy, Precision, Recall, F1-Score (macro avg).
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.preprocess import get_data_generators

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_curves(history: Dict[str, list]):
    """Plots and saves accuracy and loss curves."""
    epochs_range = range(1, len(history["accuracy"]) + 1)

    # 1. Accuracy Plot
    plt.figure(figsize=(9, 6))
    plt.plot(epochs_range, [x * 100 for x in history["accuracy"]], "bo-", label="Training Accuracy", linewidth=2, markersize=6)
    plt.plot(epochs_range, [x * 100 for x in history["val_accuracy"]], "rs-", label="Validation Accuracy", linewidth=2, markersize=6)
    plt.axvline(x=config.STAGE1_EPOCHS + 0.5, color="gray", linestyle="--", label="Fine-Tuning Stage 2 Starts")
    plt.title("AcenAAr MobileNetV2 - Training and Validation Accuracy", fontsize=14, fontweight="bold")
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", fontsize=11)
    acc_path = config.OUTPUTS_DIR / "accuracy.png"
    plt.savefig(str(acc_path), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved accuracy graph to: {acc_path}")

    # 2. Loss Plot
    plt.figure(figsize=(9, 6))
    plt.plot(epochs_range, history["loss"], "bo-", label="Training Loss", linewidth=2, markersize=6)
    plt.plot(epochs_range, history["val_loss"], "rs-", label="Validation Loss", linewidth=2, markersize=6)
    plt.axvline(x=config.STAGE1_EPOCHS + 0.5, color="gray", linestyle="--", label="Fine-Tuning Stage 2 Starts")
    plt.title("AcenAAr MobileNetV2 - Training and Validation Loss", fontsize=14, fontweight="bold")
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Categorical Crossentropy Loss", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=11)
    loss_path = config.OUTPUTS_DIR / "loss.png"
    plt.savefig(str(loss_path), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved loss graph to: {loss_path}")

def evaluate_model() -> Dict[str, Any]:
    """Evaluates the saved Keras model on the test dataset."""
    print("=" * 70)
    print("Evaluating Trained Crop Leaf Disease Classification Model")
    print("=" * 70)

    # 1. Load Model
    if not config.KERAS_MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found at {config.KERAS_MODEL_PATH}. Please run src/train.py first.")

    print(f"Loading trained model from {config.KERAS_MODEL_PATH}...")
    model = keras.models.load_model(str(config.KERAS_MODEL_PATH))

    # 2. Load History
    history_file = config.MODELS_DIR / "training_history.json"
    if history_file.exists():
        with open(history_file, "r") as f:
            history = json.load(f)
        plot_training_curves(history)
    else:
        print("Warning: training_history.json not found. Skipping curve plotting.")
        history = None

    # 3. Load Test Data
    _, _, test_gen = get_data_generators(batch_size=config.BATCH_SIZE)
    test_gen.reset()

    # Predict probabilities
    print(f"Running inference on {test_gen.samples} test images...")
    predictions = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())

    # 4. Metrics Computation
    acc = accuracy_score(y_true, y_pred)
    prec, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")

    print("\n" + "=" * 50)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 50)
    print(f"Test Accuracy  : {acc * 100:.2f}%")
    print(f"Precision (Macro): {prec:.4f}")
    print(f"Recall (Macro)   : {recall:.4f}")
    print(f"F1-Score (Macro) : {f1:.4f}")

    # 5. Classification Report
    clean_labels = [c.replace("___", " - ").replace("_", " ") for c in class_labels]
    report_text = classification_report(y_true, y_pred, target_names=clean_labels, digits=4)
    print("\nClassification Report:\n", report_text)

    report_path = config.OUTPUTS_DIR / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write("AcenAAr Technologies - Crop Leaf Disease Detection System\n")
        f.write("Evaluation on Unseen Test Dataset\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Overall Test Accuracy: {acc * 100:.2f}%\n")
        f.write(f"Macro Precision       : {prec:.4f}\n")
        f.write(f"Macro Recall          : {recall:.4f}\n")
        f.write(f"Macro F1-Score        : {f1:.4f}\n\n")
        f.write("Detailed Class-Wise Performance:\n")
        f.write(report_text)
    print(f"Saved classification report to: {report_path}")

    # 6. Confusion Matrix Heatmap
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=clean_labels,
        yticklabels=clean_labels
    )
    plt.title("AcenAAr Smart Spray Robot - Test Set Confusion Matrix", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=11, fontweight="bold")
    plt.ylabel("True Class", fontsize=11, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_path = config.OUTPUTS_DIR / "confusion_matrix.png"
    plt.savefig(str(cm_path), dpi=200)
    plt.close()
    print(f"Saved confusion matrix heatmap to: {cm_path}")

    # Save summary metrics to JSON
    summary_metrics = {
        "test_accuracy": float(round(acc, 4)),
        "precision_macro": float(round(prec, 4)),
        "recall_macro": float(round(recall, 4)),
        "f1_score_macro": float(round(f1, 4)),
        "final_train_accuracy": float(round(history["accuracy"][-1], 4)) if history else None,
        "final_val_accuracy": float(round(history["val_accuracy"][-1], 4)) if history else None,
        "final_train_loss": float(round(history["loss"][-1], 4)) if history else None,
        "final_val_loss": float(round(history["val_loss"][-1], 4)) if history else None
    }
    with open(config.OUTPUTS_DIR / "evaluation_metrics.json", "w") as f:
        json.dump(summary_metrics, f, indent=4)

    return summary_metrics

if __name__ == "__main__":
    evaluate_model()
