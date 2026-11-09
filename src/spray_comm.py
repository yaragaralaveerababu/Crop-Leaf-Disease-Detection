"""
Raspberry Pi 5 to ESP32-S3 Serial Communication & Spray Actuation Module
AcenAAr Technologies Pvt. Ltd. & Joy University
Author: Yaragarla Veerababu (2023BTIA009)

Features:
- Communicates with ESP32-S3 over UART serial.
- Includes safety checks and hardware simulation mock mode.
- Validates confidence threshold and disease condition before triggering spray.
- Prevents redundant spraying and protects hardware against over-actuation.
"""

import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SprayController")

# Optional PySerial import
try:
    import serial
    HAVE_PYSERIAL = True
except ImportError:
    HAVE_PYSERIAL = False
    logger.info("pyserial not installed; operating in hardware simulation mode.")

class SprayController:
    def __init__(self, port: str = config.SERIAL_PORT, baudrate: int = config.SERIAL_BAUD, mock_mode: bool = config.MOCK_HARDWARE):
        self.port = port
        self.baudrate = baudrate
        self.mock_mode = mock_mode
        self.serial_conn = None
        self.last_spray_time = 0.0
        self.min_interval_sec = 2.0  # Debounce limit to avoid multiple triggers on the same plant
        self.total_sprays_triggered = 0

        self._connect()

    def _connect(self):
        """Attempts connection to ESP32-S3 serial port, falls back to simulation."""
        if not self.mock_mode and HAVE_PYSERIAL:
            try:
                self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1.0)
                time.sleep(2.0)  # Wait for ESP32 reboot upon DTR assertion
                logger.info(f"Connected to ESP32-S3 on {self.port} at {self.baudrate} baud.")
            except Exception as e:
                logger.warning(f"Could not connect to physical serial port {self.port}: {e}. Falling back to MOCK mode.")
                self.mock_mode = True
        else:
            self.mock_mode = True
            logger.info("SprayController initialized in SAFE SIMULATION (MOCK) mode.")

    def trigger_spray(self, duration_ms: int = 1500) -> Dict[str, Any]:
        """
        Sends spray command to ESP32-S3 or simulates actuation.
        Enforces maximum safety duration and interval debouncing.
        """
        now = time.time()
        if (now - self.last_spray_time) < self.min_interval_sec:
            logger.warning(f"Debounce active. Last spray was only {now - self.last_spray_time:.1f}s ago. Command suppressed.")
            return {
                "status": "SUPPRESSED",
                "reason": "Debounce interval active",
                "duration_ms": 0,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

        # Cap safety duration
        duration_ms = min(duration_ms, 3000)
        self.last_spray_time = now
        self.total_sprays_triggered += 1

        if self.mock_mode or not self.serial_conn:
            logger.info(f"[SIMULATION] ESP32-S3 Relay ON for {duration_ms} ms -> Targeted fungicide applied.")
            time.sleep(0.05)  # Simulate non-blocking turnaround
            return {
                "status": "SIMULATED_SUCCESS",
                "mode": "MockSimulation",
                "duration_ms": duration_ms,
                "relay_pin": 4,
                "spray_count": self.total_sprays_triggered,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        else:
            try:
                cmd = f"SPRAY_ON:{duration_ms}\n".encode("utf-8")
                self.serial_conn.write(cmd)
                response = self.serial_conn.readline().decode("utf-8").strip()
                logger.info(f"[HARDWARE] ESP32-S3 Response: {response}")
                return {
                    "status": "HARDWARE_SUCCESS",
                    "mode": "PhysicalESP32",
                    "response": response,
                    "duration_ms": duration_ms,
                    "spray_count": self.total_sprays_triggered,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            except Exception as e:
                logger.error(f"Serial transmission failed: {e}")
                return {
                    "status": "FAILED",
                    "error": str(e),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }

    def evaluate_and_act(self, predicted_class: str, confidence_score: float) -> Dict[str, Any]:
        """
        Evaluates model predictions and conditionally triggers targeted spray.
        Safety Check:
        1. Condition must be Diseased.
        2. Confidence must exceed CONFIDENCE_THRESHOLD (default 0.75).
        """
        meta = config.DISEASE_INFO.get(predicted_class, {"condition": "Unknown", "spray_required": False})
        is_diseased = meta.get("condition") == "Diseased"
        spray_required = meta.get("spray_required", False)

        norm_conf = confidence_score / 100.0 if confidence_score > 1.0 else confidence_score

        if is_diseased and spray_required and norm_conf >= config.CONFIDENCE_THRESHOLD:
            logger.info(f"Triggering spray for {predicted_class} (Confidence: {norm_conf*100:.1f}%)")
            result = self.trigger_spray(duration_ms=1500)
            result["decision"] = "SPRAY_TRIGGERED"
            return result
        else:
            reason = "Crop is Healthy" if not is_diseased else f"Confidence ({norm_conf*100:.1f}%) below threshold ({config.CONFIDENCE_THRESHOLD*100:.1f}%)"
            logger.info(f"No spray actuation required: {reason}")
            return {
                "decision": "STANDBY",
                "reason": reason,
                "status": "IDLE",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

    def close(self):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            logger.info("Closed ESP32-S3 serial port connection.")

if __name__ == "__main__":
    controller = SprayController(mock_mode=True)
    print("Testing Spray Controller with Healthy Leaf:")
    print(controller.evaluate_and_act("Tomato___Healthy", 0.95))
    print("\nTesting Spray Controller with Diseased Leaf:")
    print(controller.evaluate_and_act("Tomato___Early_Blight", 0.88))
