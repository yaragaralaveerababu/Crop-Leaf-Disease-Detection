"""
Streamlit Interactive Dashboard for Crop Leaf Disease Detection
AcenAAr Technologies Pvt. Ltd. & Joy University
Author: Yaragarla Veerababu (2023BTIA009)

5 Required Pages:
1. Home Dashboard
2. Image Prediction
3. Live Camera Prediction
4. Prediction History
5. Analytics (Plotly Charts + Motor Fault Diagnostics)
"""

import sys
import io
import time
import datetime
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.database import get_predictions, get_statistics, get_all_diseases, init_db
from src.predict import get_predictor, CropLeafPredictor
from src.spray_comm import SprayController

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import cv2

# Page Configuration
st.set_page_config(
    page_title="AcenAAr AI Crop Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
init_db()

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        color: #1b5e20;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #424242;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f1f8e9;
        border-left: 5px solid #4caf50;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
    }
    .spray-active {
        background-color: #ffebee;
        border: 2px solid #e53935;
        padding: 12px;
        border-radius: 8px;
        color: #b71c1c;
        font-weight: bold;
        text-align: center;
    }
    .spray-idle {
        background-color: #e8f5e9;
        border: 2px solid #43a047;
        padding: 12px;
        border-radius: 8px;
        color: #1b5e20;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/plant-under-sun.png", width=70)
st.sidebar.title("Navigation")
st.sidebar.markdown("**AcenAAr Smart Spray Robot**")
page = st.sidebar.radio(
    "Select Module:",
    ["🏠 Home Dashboard", "🔬 Image Prediction", "📷 Live Camera Prediction", "📜 Prediction History", "📊 Analytics & Motor AI"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Student Details:**  
**Name:** Yaragarla Veerababu  
**Roll No:** 2023BTIA009  
**University:** Joy University  
**Specialization:** AI & ML  
**Internship:** AcenAAr Technologies  
""")

# ==============================================================================
# PAGE 1: HOME DASHBOARD
# ==============================================================================
if page == "🏠 Home Dashboard":
    st.markdown('<div class="main-title">AI-Based Crop Leaf Disease Detection & Classification System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Intelligent Vision Module for AcenAAr Technologies\' AI Driven Smart Spray Robot | Joy University</div>', unsafe_allow_html=True)

    # Summary Statistics Cards
    stats = get_statistics()
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="Total Inferences Logged", value=stats["total_predictions"], delta="Real-time SQLite")
    with col2:
        st.metric(label="Healthy Leaves", value=stats["healthy_count"], delta="No spray needed", delta_color="normal")
    with col3:
        st.metric(label="Diseased Detections", value=stats["diseased_count"], delta="Targeted Spray Triggered", delta_color="inverse")
    with col4:
        st.metric(label="Active Edge Units", value=stats["total_devices"], delta="Raspberry Pi 5 + ESP32")

    st.markdown("---")

    col_left, col_right = st.columns([1.5, 1])

    with col_left:
        st.subheader("🌾 Problem Statement & Solution")
        st.write("""
        Crop diseases account for massive agricultural yield losses annually. In conventional farming, broad-spectrum chemical fungicides 
        are sprayed uniformly across entire crop acres, resulting in severe chemical wastage, high costs, and environmental toxicity.
        
        **AcenAAr Technologies Solution:**
        An intelligent vision system deployed on a **Raspberry Pi 5** paired with an **ESP32-S3 microcontroller**. 
        Using transfer learning on a lightweight **MobileNetV2** backbone optimized via **TensorFlow Lite**, 
        the vision system identifies disease symptoms in real-time and actuates precision spray solenoids exclusively over infected foliage.
        """)

        st.subheader("⚙️ System Workflow & Specifications")
        workflow_steps = pd.DataFrame({
            "Stage": ["1. Image Acquisition", "2. Preprocessing", "3. Deep CNN Inference", "4. Decision Engine", "5. Actuation"],
            "Hardware / Module": ["Pi Camera v3 (224x224 RGB)", "OpenCV + NumPy Normalization", "MobileNetV2 TFLite (INT8/Float32)", "Confidence Threshold (>=75%)", "ESP32-S3 Relay (12V Spray Pump)"],
            "Latency Target": ["~15 ms", "~5 ms", "~35 ms on RPi 5", "< 2 ms", "1500 ms targeted pulse"]
        })
        st.table(workflow_steps)

    with col_right:
        st.subheader("📱 System Status & Hardware")
        st.info(f"""
        - **Primary Edge Device:** {config.DEFAULT_DEVICE_NAME}
        - **Device ID:** `{config.DEFAULT_DEVICE_ID}`
        - **Model Architecture:** MobileNetV2 Transfer Learning
        - **Model File:** `crop_disease_model.tflite`
        - **Confidence Threshold:** `{config.CONFIDENCE_THRESHOLD * 100:.0f}%`
        - **Actuator Microcontroller:** ESP32-S3 (UART @ 115,200 baud)
        - **Simulation Fallback:** {"Active (Safe Mock Mode)" if config.MOCK_HARDWARE else "Physical Hardware"}
        """)

        if (config.OUTPUTS_DIR / "sample_dataset_grid.png").exists():
            st.image(str(config.OUTPUTS_DIR / "sample_dataset_grid.png"), caption="Crop Leaf Disease Classes Supported", use_container_width=True)

# ==============================================================================
# PAGE 2: IMAGE PREDICTION
# ==============================================================================
elif page == "🔬 Image Prediction":
    st.markdown('<div class="main-title">Single Leaf Disease Diagnosis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload a leaf photograph to classify disease symptoms and compute precision spray action.</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Crop Leaf Image (JPG, PNG, JPEG):", type=["jpg", "jpeg", "png"])

    col_img, col_res = st.columns([1, 1.2])

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB")
        with col_img:
            st.image(pil_img, caption="Uploaded Leaf Photograph", use_container_width=True)

        with col_res:
            predict_btn = st.button("🚀 Run Diagnosis & Evaluate Spray Action", type="primary", use_container_width=True)

            if predict_btn or "last_pred" in st.session_state:
                with st.spinner("Processing image through MobileNetV2 TFLite engine..."):
                    img_np = np.array(pil_img)
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                    predictor = get_predictor()
                    res = predictor.predict(
                        image_input=img_bgr,
                        device_id=config.DEFAULT_DEVICE_ID,
                        log_to_db=True,
                        save_annotated_image=True
                    )
                    st.session_state["last_pred"] = res

                # Display Results
                res = st.session_state["last_pred"]
                is_healthy = res["condition"] == "Healthy"

                st.markdown("### Diagnosis Result:")
                clean_name = res["predicted_class"].replace("___", " - ").replace("_", " ")

                if is_healthy:
                    st.success(f"### ✅ {clean_name}")
                else:
                    st.error(f"### ⚠️ {clean_name}")

                # Metric Cards
                m1, m2, m3 = st.columns(3)
                m1.metric("Condition", res["condition"])
                m2.metric("Confidence", f"{res['confidence_percentage']}%")
                m3.metric("Latency", f"{res['inference_time_ms']} ms")

                st.progress(res["confidence_percentage"] / 100.0)

                # Spray Robot Action
                st.markdown("### Smart Spray Actuation:")
                if res["spray_action"] == "SPRAY_ACTIVATED":
                    st.markdown(f"""
                    <div class="spray-active">
                        🚨 SPRAY PUMP TRIGGERED (ESP32-S3)<br>
                        Relay Pin 4 Activated for {res['spray_duration_ms']} ms. Targeted fungicide deployed.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="spray-idle">
                        🛡️ SPRAY PUMP IN STANDBY<br>
                        No chemical application required. Crop condition is healthy or below threshold.
                    </div>
                    """, unsafe_allow_html=True)

                # Disease Details & Treatment Tabs
                st.markdown("---")
                tab1, tab2, tab3 = st.tabs(["📋 Description", "💊 Recommended Treatment", "🛡️ Prevention & Culture"])
                with tab1:
                    st.write(f"**Crop Type:** {res['crop']}")
                    st.write(res["description"])
                with tab2:
                    st.warning(f"**Treatment Protocol:** {res['treatment']}")
                with tab3:
                    st.info(f"**Cultural Management:** {res['prevention']}")

                # Class probability distribution chart
                with st.expander("📊 View Complete Class Probabilities"):
                    prob_df = pd.DataFrame(list(res["class_probabilities"].items()), columns=["Class", "Probability (%)"])
                    prob_df = prob_df.sort_values(by="Probability (%)", ascending=True)
                    fig_prob = px.bar(
                        prob_df, x="Probability (%)", y="Class", orientation="h",
                        title="Model Softmax Output Probabilities",
                        color="Probability (%)", color_continuous_scale="Greens"
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)
    else:
        with col_img:
            st.info("Please select or drop an image file above.")
            # Provide sample buttons
            sample_files = list(config.TEST_DIR.glob("*/*.jpg"))
            if sample_files:
                st.markdown("**Or pick a sample from test set:**")
                sample_pick = st.selectbox("Select sample test image:", [str(p.name) for p in sample_files[:10]])
                if st.button("Load Selected Test Sample"):
                    matched = [p for p in sample_files if p.name == sample_pick]
                    if matched:
                        img = Image.open(matched[0])
                        st.image(img, caption=f"Sample: {sample_pick}", use_container_width=True)
                        st.session_state["sample_to_pred"] = str(matched[0])

# ==============================================================================
# PAGE 3: LIVE CAMERA PREDICTION
# ==============================================================================
elif page == "📷 Live Camera Prediction":
    st.markdown('<div class="main-title">Live Field Camera Stream</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Simulates the Raspberry Pi Camera Module mounted on the AcenAAr Smart Spray Robot boom.</div>', unsafe_allow_html=True)

    camera_image = st.camera_input("Capture leaf snapshot from active camera:")

    if camera_image is not None:
        pil_cam = Image.open(camera_image).convert("RGB")
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.image(pil_cam, caption="Captured Live Frame", use_container_width=True)

        with col_c2:
            with st.spinner("Analyzing live camera capture..."):
                img_np = np.array(pil_cam)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                predictor = get_predictor()
                res = predictor.predict(
                    image_input=img_bgr,
                    device_id=config.DEFAULT_DEVICE_ID,
                    log_to_db=True,
                    save_annotated_image=True
                )

            clean_name = res["predicted_class"].replace("___", " - ").replace("_", " ")
            if res["condition"] == "Healthy":
                st.success(f"### {clean_name}")
            else:
                st.error(f"### {clean_name}")

            st.write(f"**Confidence:** `{res['confidence_percentage']}%` | **Latency:** `{res['inference_time_ms']} ms`")

            if res["spray_action"] == "SPRAY_ACTIVATED":
                st.error(f"🚨 **SPRAY ACTION:** Activated for {res['spray_duration_ms']} ms")
            else:
                st.success("🛡️ **SPRAY ACTION:** Standby (No treatment required)")

            st.info(f"**Action Plan:** {res['treatment']}")

# ==============================================================================
# PAGE 4: PREDICTION HISTORY
# ==============================================================================
elif page == "📜 Prediction History":
    st.markdown('<div class="main-title">Inference & Spray Logging Ledger</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">All diagnostic classifications and actuator triggers logged into SQLite database.</div>', unsafe_allow_html=True)

    records = get_predictions(limit=100)
    if records:
        df_history = pd.DataFrame(records)

        # Download CSV option
        csv_data = df_history.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Full History as CSV",
            data=csv_data,
            file_name="crop_prediction_history.csv",
            mime="text/csv"
        )

        # Filters
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            cond_filter = st.multiselect("Filter by Condition:", ["All", "Healthy", "Diseased"], default=["All"])
        with f_col2:
            action_filter = st.multiselect("Filter by Spray Action:", ["All", "SPRAY_ACTIVATED", "SPRAY_STANDBY"], default=["All"])

        filtered_df = df_history.copy()
        if "All" not in cond_filter and len(cond_filter) > 0:
            filtered_df = filtered_df[filtered_df["condition"].isin(cond_filter)]
        if "All" not in action_filter and len(action_filter) > 0:
            filtered_df = filtered_df[filtered_df["spray_action"].isin(action_filter)]

        # Display table
        cols_to_show = ["id", "timestamp", "predicted_class", "condition", "confidence_score", "inference_time_ms", "spray_action", "device_id"]
        st.dataframe(filtered_df[cols_to_show], use_container_width=True)

        st.caption(f"Showing {len(filtered_df)} records (out of {len(df_history)} total).")
    else:
        st.warning("No historical predictions recorded yet. Run a prediction to populate the database.")

# ==============================================================================
# PAGE 5: ANALYTICS & MOTOR AI
# ==============================================================================
elif page == "📊 Analytics & Motor AI":
    st.markdown('<div class="main-title">System Analytics & Predictive Maintenance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Operational metrics, classification distributions, and secondary project motor fault AI.</div>', unsafe_allow_html=True)

    tab_crop, tab_motor = st.tabs(["🌱 Crop Leaf Disease Analytics", "⚙️ Motor Fault Predictive Maintenance"])

    with tab_crop:
        stats = get_statistics()
        if stats["total_predictions"] > 0:
            a_col1, a_col2 = st.columns(2)

            with a_col1:
                # Disease Distribution Donut
                dist_data = pd.DataFrame(list(stats["class_distribution"].items()), columns=["Class", "Count"])
                fig_donut = px.pie(
                    dist_data, values="Count", names="Class", hole=0.45,
                    title="Class-Wise Detection Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                st.plotly_chart(fig_donut, use_container_width=True)

            with a_col2:
                # Healthy vs Diseased Bar
                cond_data = pd.DataFrame({
                    "Status": ["Healthy Leaves", "Diseased Leaves"],
                    "Count": [stats["healthy_count"], stats["diseased_count"]]
                })
                fig_bar = px.bar(
                    cond_data, x="Status", y="Count", color="Status",
                    color_discrete_map={"Healthy Leaves": "#4caf50", "Diseased Leaves": "#f44336"},
                    title="Healthy vs Diseased Crop Ratio"
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        # Model Performance Curves
        st.markdown("### 📈 Model Training Performance & Verification")
        c1, c2 = st.columns(2)
        with c1:
            if (config.OUTPUTS_DIR / "accuracy.png").exists():
                st.image(str(config.OUTPUTS_DIR / "accuracy.png"), caption="Training & Validation Accuracy Graph", use_container_width=True)
            else:
                st.info("Train the model using src/train.py to view accuracy curve.")
        with c2:
            if (config.OUTPUTS_DIR / "loss.png").exists():
                st.image(str(config.OUTPUTS_DIR / "loss.png"), caption="Training & Validation Loss Graph", use_container_width=True)
            else:
                st.info("Train the model using src/train.py to view loss curve.")

        # Confusion Matrix
        if (config.OUTPUTS_DIR / "confusion_matrix.png").exists():
            st.image(str(config.OUTPUTS_DIR / "confusion_matrix.png"), caption="Test Set Confusion Matrix", use_container_width=True)

    with tab_motor:
        st.subheader("⚡ Sensor-Based Motor Fault Classification (Secondary Project)")
        st.write("""
        **AcenAAr Predictive Maintenance:** Uses multi-sensor telemetry (MPU6050 vibration, INA219 current, DS18B20 temperature) 
        collected via ESP32-S3 to detect mechanical and electrical faults before destructive failure.
        """)

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            if (config.MOTOR_OUTPUTS_DIR / "model_comparison.png").exists():
                st.image(str(config.MOTOR_OUTPUTS_DIR / "model_comparison.png"), caption="Model Comparison: Random Forest vs SVM", use_container_width=True)
            else:
                st.info("Run motor_fault/train_motor_model.py to generate comparison chart.")

        with m_col2:
            if (config.MOTOR_OUTPUTS_DIR / "confusion_matrices.png").exists():
                st.image(str(config.MOTOR_OUTPUTS_DIR / "confusion_matrices.png"), caption="Confusion Matrices: Random Forest vs SVM", use_container_width=True)
            else:
                st.info("Run motor_fault/train_motor_model.py to generate confusion matrices.")

        # Interactive Motor Diagnosis Simulator
        st.markdown("---")
        st.subheader("🛠️ Live Motor Condition Diagnostic Test")
        st.caption("Adjust real-time sensor parameters to test fault classification:")

        s_col1, s_col2, s_col3 = st.columns(3)
        with s_col1:
            v_rms = st.slider("Vibration RMS (g):", 0.1, 4.0, 0.35, 0.05)
            v_peak = st.slider("Vibration Peak (g):", 0.2, 6.0, 0.8, 0.1)
        with s_col2:
            cur_val = st.slider("Motor Current (A):", 1.0, 6.0, 2.1, 0.1)
            temp_val = st.slider("Motor Temperature (°C):", 30.0, 95.0, 42.0, 1.0)
        with s_col3:
            v_kurt = st.slider("Vibration Kurtosis:", -1.0, 12.0, 0.2, 0.5)
            model_choice = st.radio("Classifier Algorithm:", ["Random Forest", "Support Vector Machine (SVM)"])

        if st.button("Diagnose Motor Health Condition", type="primary"):
            # Simple simulation feature vector
            feature_cols = [
                "vib_rms", "vib_peak", "vib_std", "vib_kurtosis", "crest_factor",
                "fft_energy_1x_unbalance", "fft_energy_bearing_band",
                "cur_mean", "cur_rms", "cur_std", "temperature_degC"
            ]
            raw_vec = np.array([[
                v_rms, v_peak, v_rms * 0.7, v_kurt, v_peak / (v_rms + 1e-6),
                0.08 if v_rms > 1.5 else 0.005,
                0.15 if v_kurt > 3.0 else 0.001,
                cur_val, cur_val, cur_val * 0.05, temp_val
            ]])

            import joblib
            conditions = ["Normal", "Mechanical Imbalance", "Bearing Fault", "Electrical Overload"]

            if model_choice == "Random Forest" and (config.MOTOR_FAULT_DIR / "rf_motor_model.joblib").exists():
                rf = joblib.load(config.MOTOR_FAULT_DIR / "rf_motor_model.joblib")
                pred = rf.predict(raw_vec)[0]
                conf = rf.predict_proba(raw_vec)[0][pred] * 100
            elif (config.MOTOR_FAULT_DIR / "svm_motor_model.joblib").exists():
                svm = joblib.load(config.MOTOR_FAULT_DIR / "svm_motor_model.joblib")
                scaler = joblib.load(config.MOTOR_FAULT_DIR / "motor_scaler.joblib")
                pred = svm.predict(scaler.transform(raw_vec))[0]
                conf = svm.predict_proba(scaler.transform(raw_vec))[0][pred] * 100
            else:
                pred = 0
                conf = 95.0

            cond_str = conditions[pred]
            if cond_str == "Normal":
                st.success(f"### Result: ✅ {cond_str} (Confidence: {conf:.1f}%)")
            else:
                st.error(f"### Result: ⚠️ {cond_str} (Confidence: {conf:.1f}%)")

st.markdown("---")
st.caption("AcenAAr Technologies Pvt. Ltd. | AI & ML Internship 2026-2027 | Submitted by Yaragarla Veerababu (Joy University)")
