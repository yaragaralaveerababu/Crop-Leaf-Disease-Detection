"""
Configuration Settings for AI-Based Crop Leaf Disease Detection and Classification System
AcenAAr Technologies Pvt. Ltd. & Joy University
Author: Yaragarla Veerababu (2023BTIA009)
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"

MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
PREDICTION_RESULTS_DIR = OUTPUTS_DIR / "prediction_results"
MOTOR_FAULT_DIR = BASE_DIR / "motor_fault"
MOTOR_OUTPUTS_DIR = OUTPUTS_DIR / "motor_fault"
DATABASE_DIR = BASE_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "crop_monitoring.db"

# Model File Paths
KERAS_MODEL_PATH = MODELS_DIR / "crop_disease_model.keras"
TFLITE_MODEL_PATH = MODELS_DIR / "crop_disease_model.tflite"
TFLITE_QUANT_MODEL_PATH = MODELS_DIR / "crop_disease_model_quant.tflite"

# Ensure directories exist
for directory in [DATASET_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, MODELS_DIR, 
                  OUTPUTS_DIR, PREDICTION_RESULTS_DIR, MOTOR_OUTPUTS_DIR, DATABASE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Image & Training Hyperparameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
BATCH_SIZE = 32
STAGE1_EPOCHS = 5
STAGE2_EPOCHS = 5
STAGE1_LR = 1e-3
STAGE2_LR = 1e-5
CONFIDENCE_THRESHOLD = 0.75

# Supported Crop Disease Classes
CLASS_NAMES = [
    "Corn___Common_Rust",
    "Corn___Healthy",
    "Potato___Early_Blight",
    "Potato___Healthy",
    "Potato___Late_Blight",
    "Tomato___Early_Blight",
    "Tomato___Healthy",
    "Tomato___Late_Blight",
]

NUM_CLASSES = len(CLASS_NAMES)

# Detailed Disease Information & Recommended Treatments
DISEASE_INFO = {
    "Corn___Common_Rust": {
        "crop": "Corn (Maize)",
        "condition": "Diseased",
        "description": "Fungal disease caused by Puccinia sorghi. Characterized by golden-brown to cinnamon-brown pustules on both upper and lower leaf surfaces.",
        "treatment": "Apply fungicides containing azoxystrobin or pyraclostrobin upon early symptom onset. Plant resistant corn hybrids.",
        "prevention": "Rotate crops with non-host species, practice residue management, and maintain optimal crop spacing to reduce canopy humidity.",
        "spray_required": True,
    },
    "Corn___Healthy": {
        "crop": "Corn (Maize)",
        "condition": "Healthy",
        "description": "Leaf exhibits uniform green coloration with normal parallel venation, sturdy cuticle, and no signs of lesions, necrosis, or chlorosis.",
        "treatment": "No pesticide required. Continue regular irrigation and balanced nitrogen-phosphorus-potassium (NPK) fertilization.",
        "prevention": "Routine scouting, weed control, and balanced soil moisture management.",
        "spray_required": False,
    },
    "Potato___Early_Blight": {
        "crop": "Potato",
        "condition": "Diseased",
        "description": "Caused by Alternaria solani. Produces dark brown to black circular lesions with concentric rings ('target-board' pattern), typically starting on older foliage.",
        "treatment": "Foliar spray of chlorothalonil, mancozeb, or copper-based fungicides. Remove heavily infected lower leaves.",
        "prevention": "Ensure 3-year crop rotation, avoid overhead irrigation, space rows properly for air circulation, and apply organic mulch.",
        "spray_required": True,
    },
    "Potato___Healthy": {
        "crop": "Potato",
        "condition": "Healthy",
        "description": "Foliage shows vibrant dark green color, crisp compound leaf margins, intact epidermis, and complete absence of pathogen activity.",
        "treatment": "No treatment required. Maintain scheduled drip irrigation and hilling practices.",
        "prevention": "Regular field inspection, certified disease-free seed tubers, and optimal soil drainage.",
        "spray_required": False,
    },
    "Potato___Late_Blight": {
        "crop": "Potato",
        "condition": "Diseased",
        "description": "Devastating oomycete pathogen Phytophthora infestans. Displays irregular water-soaked pale-to-dark green lesions that rapidly turn brown/black with white mold under humid conditions.",
        "treatment": "Immediate application of systemic fungicides such as metalaxyl, cymoxanil, or mandipropamid. Destroy severely infected canopy.",
        "prevention": "Plant certified resistant varieties, eliminate volunteer potato cull piles, and monitor weather alerts for cool/damp blight conditions.",
        "spray_required": True,
    },
    "Tomato___Early_Blight": {
        "crop": "Tomato",
        "condition": "Diseased",
        "description": "Caused by Alternaria linariae (solani). Brown to black necrotic spots with concentric rings surrounded by yellow chlorotic halos, causing defoliation from base upward.",
        "treatment": "Apply copper octanoate, chlorothalonil, or Bacillus subtilis bio-fungicide. Prune lower diseased foliage at early stage.",
        "prevention": "Mulch base to stop soil splash, stake plants, drip irrigate at soil level, and follow solanaceous crop rotation.",
        "spray_required": True,
    },
    "Tomato___Healthy": {
        "crop": "Tomato",
        "condition": "Healthy",
        "description": "Leaflets show vibrant green coloration, well-defined serrated margins, healthy trichome density, and vigorous vascular development.",
        "treatment": "No fungicide intervention needed. Maintain balanced macronutrient supply and consistent soil moisture.",
        "prevention": "Prune suckers for airflow, stake plants upright, and conduct weekly scouting.",
        "spray_required": False,
    },
    "Tomato___Late_Blight": {
        "crop": "Tomato",
        "condition": "Diseased",
        "description": "Aggressive infection caused by Phytophthora infestans. Appears as dark greasy water-soaked lesions on leaves and stems, with white fuzzy sporangiophores in humid weather.",
        "treatment": "Prompt application of copper-based or systemic phosphonate fungicides. Quarantine and destroy severely affected plants immediately.",
        "prevention": "Avoid wet leaves by drip irrigating, plant late-blight resistant cultivars, and maintain wide plant spacing.",
        "spray_required": True,
    },
}

# Edge & Hardware Configuration
DEFAULT_DEVICE_ID = "RPI5-SPRAYBOT-01"
DEFAULT_DEVICE_NAME = "AcenAAr Smart Spray Robot Vision Unit"
DEFAULT_DEVICE_TYPE = "Raspberry Pi 5 + Pi Camera v3 + ESP32-S3"

SERIAL_PORT = os.environ.get("SPRAY_SERIAL_PORT", "COM3")
SERIAL_BAUD = 115200
MOCK_HARDWARE = True  # Safe simulation mode when hardware is not physically attached

# FastAPI Configuration
API_HOST = "127.0.0.1"
API_PORT = 8000

# Streamlit Configuration
DASHBOARD_PORT = 8501
