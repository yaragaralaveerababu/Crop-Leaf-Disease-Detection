"""
Sensor Data Collection and Simulation for Motor Fault Classification
AcenAAr Technologies Pvt. Ltd. & Joy University
Author: Yaragarla Veerababu (2023BTIA009)

Sensors Modeled:
1. MPU6050 3-Axis Accelerometer (Vibration signature in g)
2. INA219 High-Side DC/AC Current Sensor (Motor load current in Amperes)
3. DS18B20 One-Wire Digital Temperature Sensor (Casing temperature in °C)
4. ESP32-S3 Microcontroller data sampling interface.

Motor Conditions Simulated:
- Condition 0: Normal Operation (healthy baseline)
- Condition 1: Mechanical Imbalance (1X rotational unbalance vibration)
- Condition 2: Bearing Fault (high kurtosis impulse shocks & friction)
- Condition 3: Electrical Overload (high winding current & thermal dissipation)

NOTE: As per project guidelines, this dataset is generated through physical 
modeling to simulate ESP32-S3 sensor acquisition for offline training.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

SAMPLING_RATE_HZ = 1000  # 1 kHz sampling rate
MOTOR_RPM = 1800         # 1800 RPM = 30 Hz rotational frequency (f_rot)
F_ROT = MOTOR_RPM / 60.0

CONDITIONS = [
    "Normal",
    "Mechanical_Imbalance",
    "Bearing_Fault",
    "Electrical_Overload"
]

def generate_sensor_signals(
    num_samples_per_condition: int = 150,
    window_seconds: float = 1.0
) -> pd.DataFrame:
    """
    Simulates raw time-series sensor windows across 4 operational states.
    """
    np.random.seed(42)
    n_points = int(SAMPLING_RATE_HZ * window_seconds)
    t = np.linspace(0, window_seconds, n_points, endpoint=False)

    records = []

    print(f"Generating sensor signals: {num_samples_per_condition} windows per condition ({len(CONDITIONS)} conditions)...")

    for cond_idx, condition in enumerate(CONDITIONS):
        for sample_i in range(num_samples_per_condition):
            noise = np.random.normal(0, 0.1, n_points)

            if condition == "Normal":
                # Healthy baseline: subtle baseline vibration, 2.0A nominal current, 38-42°C temp
                vibration = 0.25 * np.sin(2 * np.pi * F_ROT * t) + noise * 0.15
                current = np.random.normal(2.05, 0.08, n_points)
                temperature = np.random.normal(40.2, 0.5)

            elif condition == "Mechanical_Imbalance":
                # Prominent 1X harmonic vibration (centrifugal unbalance force), current slightly elevated
                unbalance_amp = np.random.uniform(1.8, 2.6)
                vibration = unbalance_amp * np.sin(2 * np.pi * F_ROT * t + np.random.uniform(0, np.pi)) + noise * 0.25
                current = np.random.normal(2.55, 0.12, n_points)
                temperature = np.random.normal(46.5, 0.7)

            elif condition == "Bearing_Fault":
                # High frequency impulse spikes / shock wave impacts with high kurtosis
                # Ball-pass frequency inner/outer race around 120-180 Hz
                carrier = 0.4 * np.sin(2 * np.pi * F_ROT * t)
                impact_rate = 140.0  # Hz impact frequency
                impact_pulses = np.zeros(n_points)
                impact_indices = np.arange(0, n_points, int(SAMPLING_RATE_HZ / impact_rate))
                for idx in impact_indices:
                    decay_len = min(40, n_points - idx)
                    decay = np.exp(-np.linspace(0, 3, decay_len))
                    impact_pulses[idx:idx+decay_len] += np.random.uniform(2.5, 4.2) * decay * np.sin(2 * np.pi * 800 * t[:decay_len])
                vibration = carrier + impact_pulses + noise * 0.35
                current = np.random.normal(2.35, 0.15, n_points)
                temperature = np.random.normal(49.8, 0.8)

            elif condition == "Electrical_Overload":
                # Excessive current (>4A), strong thermal rise, 2X line frequency vibration (100/120 Hz)
                vibration = 0.8 * np.sin(2 * np.pi * 100 * t) + 0.4 * np.sin(2 * np.pi * F_ROT * t) + noise * 0.2
                current = np.random.normal(4.65, 0.25, n_points)
                temperature = np.random.normal(72.5, 1.5)

            record = {
                "sample_id": f"{condition}_{sample_i:04d}",
                "condition": condition,
                "label": cond_idx,
                "temperature_degC": temperature,
                "vibration_signal": vibration.tolist(),
                "current_signal": current.tolist()
            }
            records.append(record)

    df = pd.DataFrame(records)
    output_csv = config.MOTOR_FAULT_DIR / "raw_simulated_sensor_windows.json"
    df.to_json(output_csv, orient="records", indent=2)
    print(f"Sensor dataset generated and saved to: {output_csv}")
    return df

if __name__ == "__main__":
    generate_sensor_signals()
