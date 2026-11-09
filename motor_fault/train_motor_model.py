"""
Motor Fault Model Training and Comparison Engine (Random Forest vs SVM)
AcenAAr Technologies Pvt. Ltd. & Joy University
Author: Yaragarla Veerababu (2023BTIA009)

Requirements:
- Train Random Forest and Support Vector Machine (SVM) on extracted sensor features.
- Evaluate: Accuracy, Precision, Recall, F1-Score.
- Generate and save:
  1. Motor condition prediction functionality.
  2. Classification report -> outputs/motor_fault/motor_classification_report.txt.
  3. Confusion matrix -> outputs/motor_fault/confusion_matrices.png.
  4. Model comparison graph -> outputs/motor_fault/model_comparison.png.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_compare_models():
    print("=" * 70)
    print("Secondary Project: Sensor-Based Motor Fault Classification")
    print("AcenAAr Edge AI Predictive Maintenance Module")
    print("=" * 70)

    features_csv = config.MOTOR_FAULT_DIR / "motor_features.csv"
    if not features_csv.exists():
        from motor_fault.feature_extraction import process_raw_dataset
        df = process_raw_dataset()
    else:
        df = pd.read_csv(features_csv)

    feature_cols = [
        "vib_rms", "vib_peak", "vib_std", "vib_kurtosis", "crest_factor",
        "fft_energy_1x_unbalance", "fft_energy_bearing_band",
        "cur_mean", "cur_rms", "cur_std", "temperature_degC"
    ]
    X = df[feature_cols].values
    y = df["label"].values
    class_names = [c.replace("_", " ") for c in df["condition"].unique()]

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    scaler_path = config.MOTOR_FAULT_DIR / "motor_scaler.joblib"
    joblib.dump(scaler, scaler_path)

    # 1. Random Forest Classifier
    print("\n[1/3] Training Random Forest Classifier...")
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    rf_clf.fit(X_train, y_train)  # Tree-based doesn't strictly need scaling
    rf_preds = rf_clf.predict(X_test)

    rf_acc = accuracy_score(y_test, rf_preds)
    rf_p, rf_r, rf_f1, _ = precision_recall_fscore_support(y_test, rf_preds, average="macro")
    print(f"Random Forest -> Accuracy: {rf_acc*100:.2f}% | Precision: {rf_p:.4f} | Recall: {rf_r:.4f} | F1: {rf_f1:.4f}")

    # 2. Support Vector Machine (SVM) Classifier
    print("\n[2/3] Training Support Vector Machine (RBF Kernel)...")
    svm_clf = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42)
    svm_clf.fit(X_train_scaled, y_train)
    svm_preds = svm_clf.predict(X_test_scaled)

    svm_acc = accuracy_score(y_test, svm_preds)
    svm_p, svm_r, svm_f1, _ = precision_recall_fscore_support(y_test, svm_preds, average="macro")
    print(f"Support Vector Machine -> Accuracy: {svm_acc*100:.2f}% | Precision: {svm_p:.4f} | Recall: {svm_r:.4f} | F1: {svm_f1:.4f}")

    # Save models
    joblib.dump(rf_clf, config.MOTOR_FAULT_DIR / "rf_motor_model.joblib")
    joblib.dump(svm_clf, config.MOTOR_FAULT_DIR / "svm_motor_model.joblib")

    # 3. Save Classification Reports
    report_file = config.MOTOR_OUTPUTS_DIR / "motor_classification_report.txt"
    with open(report_file, "w") as f:
        f.write("AcenAAr Predictive Maintenance: Motor Fault Classification Report\n")
        f.write("=" * 65 + "\n\n")
        f.write("1. RANDOM FOREST CLASSIFIER PERFORMANCE:\n")
        f.write(f"Overall Accuracy: {rf_acc*100:.2f}%\n")
        f.write(classification_report(y_test, rf_preds, target_names=class_names, digits=4))
        f.write("\n" + "=" * 65 + "\n\n")
        f.write("2. SUPPORT VECTOR MACHINE (SVM) PERFORMANCE:\n")
        f.write(f"Overall Accuracy: {svm_acc*100:.2f}%\n")
        f.write(classification_report(y_test, svm_preds, target_names=class_names, digits=4))
    print(f"\nSaved combined classification report to: {report_file}")

    # 4. Side-by-Side Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Motor Condition Classification: Confusion Matrix Comparison", fontsize=14, fontweight="bold")

    cm_rf = confusion_matrix(y_test, rf_preds)
    sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens", ax=axes[0],
                xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title(f"Random Forest (Accuracy: {rf_acc*100:.1f}%)", fontweight="bold")
    axes[0].set_xlabel("Predicted Condition")
    axes[0].set_ylabel("True Condition")
    axes[0].tick_params(axis='x', rotation=30)

    cm_svm = confusion_matrix(y_test, svm_preds)
    sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Purples", ax=axes[1],
                xticklabels=class_names, yticklabels=class_names)
    axes[1].set_title(f"Support Vector Machine (Accuracy: {svm_acc*100:.1f}%)", fontweight="bold")
    axes[1].set_xlabel("Predicted Condition")
    axes[1].set_ylabel("True Condition")
    axes[1].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    cm_path = config.MOTOR_OUTPUTS_DIR / "confusion_matrices.png"
    plt.savefig(str(cm_path), dpi=200)
    plt.close()
    print(f"Saved confusion matrix comparison to: {cm_path}")

    # 5. Model Comparison Bar Chart
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Random Forest": [rf_acc, rf_p, rf_r, rf_f1],
        "SVM (RBF)": [svm_acc, svm_p, svm_r, svm_f1]
    })

    plt.figure(figsize=(9, 5))
    x = np.arange(len(metrics_df["Metric"]))
    width = 0.35

    plt.bar(x - width/2, metrics_df["Random Forest"] * 100, width, label="Random Forest", color="#2ecc71", edgecolor="black")
    plt.bar(x + width/2, metrics_df["SVM (RBF)"] * 100, width, label="Support Vector Machine (SVM)", color="#9b59b6", edgecolor="black")

    plt.ylabel("Score (%)", fontsize=12, fontweight="bold")
    plt.title("Model Performance Comparison: Random Forest vs SVM\n(MPU6050 + INA219 + DS18B20 Features)", fontsize=13, fontweight="bold")
    plt.xticks(x, metrics_df["Metric"], fontsize=11, fontweight="bold")
    plt.ylim(80, 105)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)

    for i in x:
        plt.text(i - width/2, (metrics_df.loc[i, "Random Forest"] * 100) + 0.5,
                 f"{metrics_df.loc[i, 'Random Forest']*100:.1f}%", ha='center', fontsize=9, fontweight='bold')
        plt.text(i + width/2, (metrics_df.loc[i, "SVM (RBF)"] * 100) + 0.5,
                 f"{metrics_df.loc[i, 'SVM (RBF)']*100:.1f}%", ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    comp_path = config.MOTOR_OUTPUTS_DIR / "model_comparison.png"
    plt.savefig(str(comp_path), dpi=200)
    plt.close()
    print(f"Saved model comparison chart to: {comp_path}")

    # Summary JSON
    summary = {
        "Random_Forest": {
            "accuracy": round(rf_acc, 4),
            "precision": round(rf_p, 4),
            "recall": round(rf_r, 4),
            "f1_score": round(rf_f1, 4)
        },
        "Support_Vector_Machine": {
            "accuracy": round(svm_acc, 4),
            "precision": round(svm_p, 4),
            "recall": round(svm_r, 4),
            "f1_score": round(svm_f1, 4)
        }
    }
    with open(config.MOTOR_OUTPUTS_DIR / "motor_metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    return summary

def predict_motor_condition(vibration_signal: np.ndarray, current_signal: np.ndarray, temp_degC: float, model_type: str = "rf") -> dict:
    """Predicts motor health condition given raw window data."""
    from motor_fault.feature_extraction import extract_features_from_window
    feats = extract_features_from_window(vibration_signal, current_signal, temp_degC)
    feature_cols = [
        "vib_rms", "vib_peak", "vib_std", "vib_kurtosis", "crest_factor",
        "fft_energy_1x_unbalance", "fft_energy_bearing_band",
        "cur_mean", "cur_rms", "cur_std", "temperature_degC"
    ]
    raw_vec = np.array([[feats[col] for col in feature_cols]])

    conditions = ["Normal", "Mechanical_Imbalance", "Bearing_Fault", "Electrical_Overload"]

    if model_type.lower() == "rf":
        model = joblib.load(config.MOTOR_FAULT_DIR / "rf_motor_model.joblib")
        pred_idx = model.predict(raw_vec)[0]
        probs = model.predict_proba(raw_vec)[0]
    else:
        model = joblib.load(config.MOTOR_FAULT_DIR / "svm_motor_model.joblib")
        scaler = joblib.load(config.MOTOR_FAULT_DIR / "motor_scaler.joblib")
        scaled_vec = scaler.transform(raw_vec)
        pred_idx = model.predict(scaled_vec)[0]
        probs = model.predict_proba(scaled_vec)[0]

    return {
        "predicted_condition": conditions[pred_idx],
        "confidence": float(round(probs[pred_idx] * 100, 2)),
        "all_probabilities": {conditions[i]: float(round(probs[i] * 100, 2)) for i in range(len(conditions))},
        "extracted_features": feats
    }

if __name__ == "__main__":
    train_and_compare_models()
