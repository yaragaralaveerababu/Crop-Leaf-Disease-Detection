"""
Batch Test and Prediction Populator
AcenAAr Technologies Pvt. Ltd. & Joy University
Author: Yaragarla Veerababu (2023BTIA009)
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.predict import get_predictor

def run_batch_tests():
    predictor = get_predictor()
    print("=" * 80)
    print("Executing Batch Predictions on Test Samples across All Crop Disease Classes")
    print("=" * 80)

    for class_name in config.CLASS_NAMES:
        class_test_dir = config.TEST_DIR / class_name
        test_files = sorted(list(class_test_dir.glob("*.jpg")))
        # Run 2 samples per class
        for sample_path in test_files[:2]:
            res = predictor.predict(str(sample_path))
            cls = res["predicted_class"]
            conf = res["confidence_percentage"]
            lat = res["inference_time_ms"]
            act = res["spray_action"]
            print(f"Sample: {sample_path.name:<32} -> Pred: {cls:<25} | Conf: {conf:>6.2f}% | Latency: {lat:>5.1f}ms | Spray: {act}")

    print("=" * 80)
    print(f"Saved annotated result images to: {config.PREDICTION_RESULTS_DIR}")
    print(f"All predictions logged into SQLite database: {config.DATABASE_PATH}")

if __name__ == "__main__":
    run_batch_tests()
