"""
Feature Extraction Engine for Sensor-Based Motor Fault Classification
AcenAAr Technologies Pvt. Ltd. & Joy University
Author: Yaragarla Veerababu (2023BTIA009)

Features Extracted from Time-Series Signals:
1. RMS (Root Mean Square) - Overall energy/power of vibration and current.
2. Peak Value - Maximum instantaneous amplitude.
3. Standard Deviation - Spread and variance around mean.
4. Kurtosis - Peakedness/spikiness (sensitive indicator for bearing fault shocks).
5. FFT Band Energy:
   - Low Frequency Band (10 - 50 Hz): 1X Rotational unbalance detection.
   - High Frequency Band (400 - 900 Hz): Bearing defect impact ringing.
6. Current Statistics (Mean, RMS, Std) - Detects electromagnetic overload.
7. Temperature (°C) - Measures thermal stress from prolonged overload or severe bearing friction.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

def extract_features_from_window(vibration: np.ndarray, current: np.ndarray, temp: float, fs: int = 1000) -> dict:
    """Computes statistical and spectral features from a 1-second sensor window."""
    # 1. Vibration Time-Domain Features
    vib_rms = np.sqrt(np.mean(np.square(vibration)))
    vib_peak = np.max(np.abs(vibration))
    vib_std = np.std(vibration)
    vib_kurtosis = stats.kurtosis(vibration)
    crest_factor = vib_peak / (vib_rms + 1e-6)

    # 2. Vibration Frequency-Domain Features (FFT)
    n = len(vibration)
    fft_vals = np.abs(np.fft.rfft(vibration)) / n
    freqs = np.fft.rfftfreq(n, d=1.0/fs)

    # Low frequency band: 15 to 45 Hz (1X unbalance around 30 Hz)
    idx_low = np.where((freqs >= 15) & (freqs <= 45))[0]
    fft_energy_low = np.sum(np.square(fft_vals[idx_low])) if len(idx_low) > 0 else 0.0

    # High frequency band: 400 to 900 Hz (bearing impact resonance)
    idx_high = np.where((freqs >= 400) & (freqs <= 900))[0]
    fft_energy_high = np.sum(np.square(fft_vals[idx_high])) if len(idx_high) > 0 else 0.0

    # 3. Current Sensor Features (INA219)
    cur_mean = np.mean(current)
    cur_rms = np.sqrt(np.mean(np.square(current)))
    cur_std = np.std(current)

    return {
        "vib_rms": float(round(vib_rms, 4)),
        "vib_peak": float(round(vib_peak, 4)),
        "vib_std": float(round(vib_std, 4)),
        "vib_kurtosis": float(round(vib_kurtosis, 4)),
        "crest_factor": float(round(crest_factor, 4)),
        "fft_energy_1x_unbalance": float(round(fft_energy_low, 6)),
        "fft_energy_bearing_band": float(round(fft_energy_high, 6)),
        "cur_mean": float(round(cur_mean, 4)),
        "cur_rms": float(round(cur_rms, 4)),
        "cur_std": float(round(cur_std, 4)),
        "temperature_degC": float(round(temp, 2))
    }

def process_raw_dataset() -> pd.DataFrame:
    """Reads raw sensor windows, extracts all features, and saves feature matrix."""
    raw_path = config.MOTOR_FAULT_DIR / "raw_simulated_sensor_windows.json"
    if not raw_path.exists():
        from motor_fault.data_collection import generate_sensor_signals
        generate_sensor_signals()

    with open(raw_path, "r") as f:
        data = json.load(f)

    feature_rows = []
    print(f"Extracting features from {len(data)} sensor windows...")

    for item in data:
        vib = np.array(item["vibration_signal"], dtype=np.float32)
        cur = np.array(item["current_signal"], dtype=np.float32)
        temp = float(item["temperature_degC"])

        feats = extract_features_from_window(vib, cur, temp)
        feats["condition"] = item["condition"]
        feats["label"] = item["label"]
        feature_rows.append(feats)

    df_features = pd.DataFrame(feature_rows)
    features_csv = config.MOTOR_FAULT_DIR / "motor_features.csv"
    df_features.to_csv(features_csv, index=False)
    print(f"Feature extraction complete! Saved {len(df_features)} records to: {features_csv}")
    print("\nFeature Summary:\n", df_features.groupby("condition").mean())
    return df_features

if __name__ == "__main__":
    process_raw_dataset()
