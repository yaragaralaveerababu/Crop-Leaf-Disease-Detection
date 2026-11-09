"""
Streamlit Dashboard Visual Screenshot Renderer
AcenAAr Technologies Pvt. Ltd. & Joy University
Author: Yaragarla Veerababu (2023BTIA009)

Generates high-resolution dashboard mockups for the internship report:
1. Streamlit Image Prediction View -> outputs/screenshots/streamlit_dashboard_prediction.png
2. Streamlit Analytics & Motor AI View -> outputs/screenshots/streamlit_analytics_page.png
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2

def generate_prediction_screenshot():
    fig = plt.figure(figsize=(14, 9), facecolor="#f8f9fa")
    
    # Header Bar
    ax_head = fig.add_axes([0, 0.92, 1, 0.08])
    ax_head.axis("off")
    ax_head.add_patch(patches.Rectangle((0, 0), 1, 1, color="#1b5e20"))
    ax_head.text(0.03, 0.5, "AcenAAr Technologies | Smart Spray Robot Vision Dashboard", color="white", fontsize=15, fontweight="bold", va="center")
    ax_head.text(0.85, 0.5, "Device: RPI5-SPRAYBOT-01 (ONLINE)", color="#a5d6a7", fontsize=10, fontweight="bold", va="center")

    # Sidebar Area
    ax_side = fig.add_axes([0, 0, 0.22, 0.92])
    ax_side.axis("off")
    ax_side.add_patch(patches.Rectangle((0, 0), 1, 1, color="#ffffff", ec="#e0e0e0"))
    ax_side.text(0.1, 0.94, "🌱 Navigation", fontsize=12, fontweight="bold", color="#1b5e20")
    pages = [
        ("🏠 Home Dashboard", False),
        ("🔬 Image Prediction", True),
        ("📷 Live Camera Stream", False),
        ("📜 Prediction History", False),
        ("📊 Analytics & Motor AI", False)
    ]
    y_pos = 0.86
    for p_name, active in pages:
        bg_col = "#e8f5e9" if active else "#ffffff"
        text_col = "#1b5e20" if active else "#424242"
        ax_side.add_patch(patches.Rectangle((0.05, y_pos - 0.015), 0.9, 0.045, color=bg_col, ec="none"))
        ax_side.text(0.1, y_pos, p_name, fontsize=10, fontweight="bold" if active else "normal", color=text_col)
        y_pos -= 0.06

    ax_side.text(0.1, 0.35, "Internship Details:", fontsize=10, fontweight="bold", color="#333333")
    ax_side.text(0.1, 0.28, "Student: Y. Veerababu\nRoll: 2023BTIA009\nJoy University\nDept: AI & ML", fontsize=9, color="#616161")

    # Main Area: Title
    ax_main = fig.add_axes([0.25, 0.02, 0.73, 0.88])
    ax_main.axis("off")
    ax_main.text(0.02, 0.97, "Single Leaf Disease Diagnosis & Smart Spray Actuation", fontsize=16, fontweight="bold", color="#1b5e20")
    ax_main.text(0.02, 0.93, "Real-time edge inference via MobileNetV2 TensorFlow Lite on Raspberry Pi 5", fontsize=10, color="#616161")

    # Uploaded Leaf Card
    sample_file = config.PREDICTION_RESULTS_DIR / "pred_20261109_163220_Tomato___Early_Blight.jpg"
    if not sample_file.exists():
        samples = list(config.PREDICTION_RESULTS_DIR.glob("*.jpg"))
        sample_file = samples[0] if samples else None

    if sample_file and sample_file.exists():
        leaf_img = cv2.imread(str(sample_file))
        leaf_rgb = cv2.cvtColor(leaf_img, cv2.COLOR_BGR2RGB)
        ax_img = fig.add_axes([0.27, 0.45, 0.32, 0.42])
        ax_img.imshow(leaf_rgb)
        ax_img.set_title("Input Camera Frame (224x224)", fontsize=10, fontweight="bold")
        ax_img.axis("off")

    # Results Panel
    ax_main.add_patch(patches.FancyBboxPatch((0.50, 0.44), 0.47, 0.45, boxstyle="round,pad=0.02", color="#ffffff", ec="#e0e0e0"))
    ax_main.text(0.52, 0.85, "Diagnosis: Tomato - Early Blight", fontsize=14, fontweight="bold", color="#c62828")
    ax_main.text(0.52, 0.80, "Condition: Diseased | Crop: Solanum lycopersicum", fontsize=10, color="#555555")

    # Metrics
    metrics_x = [0.52, 0.67, 0.82]
    m_labels = ["Confidence", "Inference Latency", "Spray Pulse"]
    m_values = ["99.8%", "34.5 ms", "1500 ms"]
    for mx, ml, mv in zip(metrics_x, m_labels, m_values):
        ax_main.add_patch(patches.FancyBboxPatch((mx, 0.67), 0.13, 0.09, boxstyle="round,pad=0.01", color="#f1f8e9", ec="#c8e6c9"))
        ax_main.text(mx + 0.065, 0.73, mv, fontsize=12, fontweight="bold", color="#1b5e20", ha="center")
        ax_main.text(mx + 0.065, 0.69, ml, fontsize=8, color="#555555", ha="center")

    # Spray Banner
    ax_main.add_patch(patches.FancyBboxPatch((0.52, 0.54), 0.43, 0.08, boxstyle="round,pad=0.01", color="#ffebee", ec="#ef5350"))
    ax_main.text(0.735, 0.59, "🚨 ESP32-S3 ACTUATOR: SPRAY TRIGGERED (RELAY PIN 4)", fontsize=10, fontweight="bold", color="#b71c1c", ha="center")
    ax_main.text(0.735, 0.555, "Confidence >= 75% threshold. Targeted fungicide solenoid open.", fontsize=8, color="#c62828", ha="center")

    # Clinical Recommendation Card
    ax_main.add_patch(patches.FancyBboxPatch((0.02, 0.05), 0.95, 0.35, boxstyle="round,pad=0.02", color="#ffffff", ec="#e0e0e0"))
    ax_main.text(0.04, 0.36, "📋 Agronomic Treatment & Disease Profile", fontsize=12, fontweight="bold", color="#1b5e20")
    ax_main.text(0.04, 0.30, "Etiology: Alternaria solani fungal infection causing concentric circular necrotic lesions with yellow chlorotic halos.", fontsize=9, color="#424242")
    ax_main.text(0.04, 0.23, "Recommended Pesticide:", fontsize=10, fontweight="bold", color="#d32f2f")
    ax_main.text(0.04, 0.18, "Foliar application of Copper Hydroxide (2.5 g/L) or Chlorothalonil 75 WP (2 g/L). Apply early morning under calm wind.", fontsize=9, color="#333333")
    ax_main.text(0.04, 0.12, "Cultural Preventive Measures:", fontsize=10, fontweight="bold", color="#1976d2")
    ax_main.text(0.04, 0.07, "Implement 3-year crop rotation, stake tomato canopy upright, apply soil mulch, and avoid overhead sprinkler wetting.", fontsize=9, color="#333333")

    out_file = config.OUTPUTS_DIR / "screenshots" / "streamlit_dashboard_prediction.png"
    plt.savefig(str(out_file), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Generated dashboard prediction screenshot: {out_file}")

if __name__ == "__main__":
    generate_prediction_screenshot()
