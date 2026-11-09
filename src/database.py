"""
SQLite Database Handler for Crop Leaf Disease Detection System
AcenAAr Technologies Pvt. Ltd. & Joy University
Author: Yaragarla Veerababu (2023BTIA009)
"""

import sqlite3
import datetime
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

def get_connection() -> sqlite3.Connection:
    """Creates and returns a connection to the SQLite database."""
    conn = sqlite3.connect(str(config.DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes database schema and populates initial reference data."""
    conn = get_connection()
    cursor = conn.cursor()

    # Table 1: devices
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS devices (
            device_id TEXT PRIMARY KEY,
            device_name TEXT NOT NULL,
            device_type TEXT NOT NULL,
            status TEXT DEFAULT 'Active',
            registered_at TEXT NOT NULL,
            last_active TEXT NOT NULL
        )
    """)

    # Table 2: diseases
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS diseases (
            disease_id INTEGER PRIMARY KEY AUTOINCREMENT,
            disease_name TEXT UNIQUE NOT NULL,
            crop_name TEXT NOT NULL,
            condition TEXT NOT NULL,
            description TEXT NOT NULL,
            recommended_treatment TEXT NOT NULL,
            prevention TEXT NOT NULL,
            spray_required INTEGER NOT NULL
        )
    """)

    # Table 3: predictions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            image_path TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            condition TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            inference_time_ms REAL NOT NULL,
            spray_action TEXT NOT NULL,
            device_id TEXT NOT NULL,
            FOREIGN KEY (device_id) REFERENCES devices (device_id)
        )
    """)

    # Insert default device if not present
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT OR IGNORE INTO devices (device_id, device_name, device_type, status, registered_at, last_active)
        VALUES (?, ?, ?, 'Online', ?, ?)
    """, (config.DEFAULT_DEVICE_ID, config.DEFAULT_DEVICE_NAME, config.DEFAULT_DEVICE_TYPE, now_str, now_str))

    # Populate disease references
    for d_name, d_meta in config.DISEASE_INFO.items():
        cursor.execute("""
            INSERT OR REPLACE INTO diseases 
            (disease_name, crop_name, condition, description, recommended_treatment, prevention, spray_required)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            d_name,
            d_meta["crop"],
            d_meta["condition"],
            d_meta["description"],
            d_meta["treatment"],
            d_meta["prevention"],
            1 if d_meta["spray_required"] else 0
        ))

    conn.commit()
    conn.close()

def log_prediction(
    image_path: str,
    predicted_class: str,
    confidence_score: float,
    inference_time_ms: float = 0.0,
    device_id: str = config.DEFAULT_DEVICE_ID
) -> int:
    """Logs a single inference prediction to the database."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()

    meta = config.DISEASE_INFO.get(predicted_class, {
        "condition": "Unknown",
        "spray_required": False
    })
    condition = meta.get("condition", "Unknown")
    spray_required = meta.get("spray_required", False)

    # Determine spray action based on confidence threshold and condition
    if spray_required and confidence_score >= (config.CONFIDENCE_THRESHOLD * 100 if confidence_score > 1 else config.CONFIDENCE_THRESHOLD):
        spray_action = "SPRAY_ACTIVATED"
    else:
        spray_action = "SPRAY_STANDBY"

    # Normalize confidence to 0.0 - 1.0 float if passed as percentage
    norm_conf = confidence_score / 100.0 if confidence_score > 1.0 else confidence_score

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO predictions 
        (timestamp, image_path, predicted_class, condition, confidence_score, inference_time_ms, spray_action, device_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now_str,
        str(image_path),
        predicted_class,
        condition,
        round(float(norm_conf), 4),
        round(float(inference_time_ms), 2),
        spray_action,
        device_id
    ))

    prediction_id = cursor.lastrowid

    # Update device last active
    cursor.execute("""
        UPDATE devices SET last_active = ? WHERE device_id = ?
    """, (now_str, device_id))

    conn.commit()
    conn.close()
    return prediction_id

def get_predictions(limit: int = 100) -> List[Dict[str, Any]]:
    """Retrieves recent predictions as dictionaries."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT p.id, p.timestamp, p.image_path, p.predicted_class, p.condition,
               p.confidence_score, p.inference_time_ms, p.spray_action, p.device_id,
               d.recommended_treatment, d.description
        FROM predictions p
        LEFT JOIN diseases d ON p.predicted_class = d.disease_name
        ORDER BY p.id DESC
        LIMIT ?
    """, (limit,))

    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def get_statistics() -> Dict[str, Any]:
    """Retrieves analytics summary for dashboard counters and graphs."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_predictions = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE condition = 'Healthy'")
    healthy_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE condition = 'Diseased'")
    diseased_count = cursor.fetchone()[0]

    cursor.execute("""
        SELECT predicted_class, COUNT(*) as count 
        FROM predictions 
        GROUP BY predicted_class 
        ORDER BY count DESC
    """)
    class_distribution = {row["predicted_class"]: row["count"] for row in cursor.fetchall()}

    cursor.execute("SELECT COUNT(*) FROM devices")
    total_devices = cursor.fetchone()[0]

    conn.close()
    return {
        "total_predictions": total_predictions,
        "healthy_count": healthy_count,
        "diseased_count": diseased_count,
        "class_distribution": class_distribution,
        "total_devices": total_devices
    }

def get_all_diseases() -> List[Dict[str, Any]]:
    """Retrieves all registered diseases and treatments."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM diseases ORDER BY disease_name ASC")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully at:", config.DATABASE_PATH)
    stats = get_statistics()
    print("Initial Statistics:", stats)
