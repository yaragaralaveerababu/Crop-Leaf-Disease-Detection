"""
FastAPI RESTful Backend Service for Crop Leaf Disease Detection
AcenAAr Technologies Pvt. Ltd. & Joy University
Author: Yaragarla Veerababu (2023BTIA009)

Endpoints:
- GET /         : System Overview and API Documentation
- GET /health   : Health check, database status, and model readiness
- POST /predict : Upload crop leaf image, run inference, log to DB, return JSON
- GET /history  : Query previous prediction logs from SQLite
- GET /diseases : List supported disease metadata and treatment guidelines
- POST /motor/predict : Sensor-based motor fault diagnosis
"""

import sys
import io
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.database import get_predictions, get_statistics, get_all_diseases, init_db
from src.predict import get_predictor, CropLeafPredictor
from src.spray_comm import SprayController

from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2

app = FastAPI(
    title="AI-Based Crop Leaf Disease Detection API",
    description="Vision Module API for AcenAAr Smart Spray Robot (Joy University B.Tech Internship)",
    version="1.0.0"
)

# Enable CORS for cross-origin frontend dashboard integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database tables and spray controller
init_db()
spray_controller = SprayController(mock_mode=True)

@app.get("/")
def home():
    """Project Overview and API Specifications."""
    return {
        "project": "AI-Based Crop Leaf Disease Detection and Classification System",
        "organization": "AcenAAr Technologies Pvt. Ltd.",
        "academic_institution": "Joy University, School of Computational Intelligence",
        "student_author": "Yaragarla Veerababu (2023BTIA009)",
        "deployment_target": "Raspberry Pi 5 + ESP32-S3 Smart Spray Robot",
        "model_architecture": "MobileNetV2 Transfer Learning with TensorFlow Lite Optimization",
        "status": "Online",
        "available_endpoints": {
            "GET /": "Home page with system metadata",
            "GET /health": "System status, DB health, model engine details",
            "POST /predict": "Upload leaf image for disease diagnosis and spray actuation",
            "GET /history": "Query SQLite database for past predictions",
            "GET /diseases": "List all 8 supported disease categories and treatments",
            "POST /motor/predict": "Diagnose industrial motor health from sensor features"
        }
    }

@app.get("/health")
def health_check():
    """System health check and hardware readiness."""
    predictor = get_predictor()
    stats = get_statistics()
    return {
        "status": "Healthy",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_engine": predictor.backend,
        "database_status": "Connected (SQLite)",
        "default_device_id": config.DEFAULT_DEVICE_ID,
        "mock_hardware_mode": config.MOCK_HARDWARE,
        "total_historical_predictions": stats["total_predictions"]
    }

@app.post("/predict")
def predict_crop_leaf(
    file: UploadFile = File(...),
    device_id: str = Form(config.DEFAULT_DEVICE_ID)
):
    """
    Accepts crop leaf image upload (JPEG/PNG), processes inference using
    TensorFlow Lite, evaluates confidence threshold, actuates spray robot,
    logs to SQLite, and returns diagnosis in JSON.
    """
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image (JPEG/PNG).")

    dev_id = str(device_id.default) if hasattr(device_id, "default") else str(device_id)

    # Read bytes into memory
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Unable to decode image file.")

    try:
        predictor = get_predictor()
        # Run inference
        result = predictor.predict(
            image_input=img_bgr,
            device_id=dev_id,
            log_to_db=True,
            save_annotated_image=True
        )

        # Trigger spray if needed
        spray_res = spray_controller.evaluate_and_act(
            result["predicted_class"],
            result["confidence_score"]
        )

        response = {
            "filename": file.filename,
            "prediction_id": result["prediction_id"],
            "predicted_class": result["predicted_class"],
            "condition": result["condition"],
            "confidence_percentage": result["confidence_percentage"],
            "confidence_score": result["confidence_score"],
            "inference_time_ms": result["inference_time_ms"],
            "spray_action": result["spray_action"],
            "spray_status": spray_res,
            "crop": result["crop"],
            "description": result["description"],
            "recommended_treatment": result["treatment"],
            "prevention": result["prevention"],
            "device_id": dev_id,
            "backend": result["backend"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        return JSONResponse(status_code=200, content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline error: {str(e)}")

@app.get("/history")
def get_prediction_history(limit: int = Query(50, ge=1, le=500)):
    """Retrieves previous prediction records from SQLite database."""
    try:
        records = get_predictions(limit=limit)
        stats = get_statistics()
        return {
            "total_records_returned": len(records),
            "summary_statistics": stats,
            "predictions": records
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query error: {str(e)}")

@app.get("/diseases")
def list_diseases():
    """Lists all supported crop classes, clinical descriptions, and recommended treatments."""
    diseases = get_all_diseases()
    return {
        "supported_classes_count": len(diseases),
        "classes": diseases
    }

@app.post("/motor/predict")
def predict_motor(
    vib_rms: float = Form(0.3),
    vib_peak: float = Form(0.8),
    vib_kurtosis: float = Form(0.0),
    fft_energy_1x: float = Form(0.01),
    fft_energy_bearing: float = Form(0.001),
    current_rms: float = Form(2.1),
    temperature: float = Form(41.0),
    model_type: str = Form("rf")
):
    """Diagnoses motor fault from input sensor features."""
    try:
        import joblib
        feature_cols = [
            "vib_rms", "vib_peak", "vib_std", "vib_kurtosis", "crest_factor",
            "fft_energy_1x_unbalance", "fft_energy_bearing_band",
            "cur_mean", "cur_rms", "cur_std", "temperature_degC"
        ]
        # Approximation of vector
        raw_vec = np.array([[
            vib_rms, vib_peak, vib_rms * 0.7, vib_kurtosis, vib_peak / (vib_rms + 1e-6),
            fft_energy_1x, fft_energy_bearing, current_rms, current_rms, current_rms * 0.05, temperature
        ]])
        
        conditions = ["Normal", "Mechanical_Imbalance", "Bearing_Fault", "Electrical_Overload"]
        if model_type.lower() == "rf" and (config.MOTOR_FAULT_DIR / "rf_motor_model.joblib").exists():
            model = joblib.load(config.MOTOR_FAULT_DIR / "rf_motor_model.joblib")
            pred_idx = model.predict(raw_vec)[0]
            probs = model.predict_proba(raw_vec)[0]
        elif (config.MOTOR_FAULT_DIR / "svm_motor_model.joblib").exists():
            model = joblib.load(config.MOTOR_FAULT_DIR / "svm_motor_model.joblib")
            scaler = joblib.load(config.MOTOR_FAULT_DIR / "motor_scaler.joblib")
            scaled_vec = scaler.transform(raw_vec)
            pred_idx = model.predict(scaled_vec)[0]
            probs = model.predict_proba(scaled_vec)[0]
        else:
            return {"status": "Model not yet trained. Run motor_fault/train_motor_model.py first."}

        return {
            "predicted_condition": conditions[pred_idx],
            "confidence_percentage": round(probs[pred_idx] * 100, 2),
            "probabilities": {conditions[i]: round(probs[i] * 100, 2) for i in range(len(conditions))},
            "model_type": model_type.upper()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=config.API_HOST, port=config.API_PORT, reload=True)
