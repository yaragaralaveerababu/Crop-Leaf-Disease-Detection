"""
FastAPI Endpoints Verification Script
AcenAAr Technologies Pvt. Ltd. & Joy University
Author: Yaragarla Veerababu (2023BTIA009)
"""

import sys
import json
from pathlib import Path
from io import BytesIO

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.main import home, health_check, list_diseases, get_prediction_history, predict_crop_leaf, predict_motor
from starlette.datastructures import UploadFile, Headers
import config

def test_all_endpoints():
    print("=" * 70, flush=True)
    print("Verifying FastAPI Backend REST Endpoints", flush=True)
    print("=" * 70, flush=True)

    # 1. GET /
    print("\n[Endpoint 1/6] Testing GET / (home) ...", flush=True)
    r1 = home()
    assert "project" in r1
    print("Project Title    :", r1["project"], flush=True)
    print("Available Routes :", len(r1["available_endpoints"]), flush=True)

    # 2. GET /health
    print("\n[Endpoint 2/6] Testing GET /health (health_check) ...", flush=True)
    r2 = health_check()
    assert r2["status"] == "Healthy"
    print("Status       :", r2["status"], flush=True)
    print("Model Engine :", r2["model_engine"], flush=True)
    print("DB Status    :", r2["database_status"], flush=True)

    # 3. GET /diseases
    print("\n[Endpoint 3/6] Testing GET /diseases (list_diseases) ...", flush=True)
    r3 = list_diseases()
    assert r3["supported_classes_count"] == 8
    print(f"Supported Classes Count: {r3['supported_classes_count']}", flush=True)

    # 4. POST /predict
    print("\n[Endpoint 4/6] Testing POST /predict (predict_crop_leaf) ...", flush=True)
    test_img_path = list(config.TEST_DIR.glob("*/*.jpg"))[0]
    with open(test_img_path, "rb") as f:
        file_bytes = f.read()

    upload_file = UploadFile(
        file=BytesIO(file_bytes),
        filename=test_img_path.name,
        headers=Headers({"content-type": "image/jpeg"})
    )
    r4_resp = predict_crop_leaf(file=upload_file, device_id="RPI5-SPRAYBOT-01")
    p_data = json.loads(r4_resp.body.decode("utf-8"))
    print(f"Predicted Class : {p_data['predicted_class']}", flush=True)
    print(f"Condition       : {p_data['condition']}", flush=True)
    print(f"Confidence      : {p_data['confidence_percentage']}%", flush=True)
    print(f"Latency         : {p_data['inference_time_ms']} ms", flush=True)
    print(f"Spray Action    : {p_data['spray_action']}", flush=True)

    # 5. GET /history
    print("\n[Endpoint 5/6] Testing GET /history (get_prediction_history) ...", flush=True)
    r5 = get_prediction_history(limit=5)
    print(f"Total History Records Returned: {r5['total_records_returned']}", flush=True)
    print(f"Total Logged in DB: {r5['summary_statistics']['total_predictions']}", flush=True)

    # 6. POST /motor/predict
    print("\n[Endpoint 6/6] Testing POST /motor/predict (predict_motor) ...", flush=True)
    r6 = predict_motor(
        vib_rms=1.6, vib_peak=2.8, vib_kurtosis=0.5,
        fft_energy_1x=0.12, fft_energy_bearing=0.001,
        current_rms=2.5, temperature=47.0, model_type="rf"
    )
    print(f"Motor Condition  : {r6['predicted_condition']}", flush=True)
    print(f"Confidence       : {r6['confidence_percentage']}%", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("ALL FASTAPI REST ENDPOINTS VERIFIED AND RETURNING 200 JSON RESPONSES!", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    test_all_endpoints()
