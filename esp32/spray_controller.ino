/*
 * ============================================================================
 * AcenAAr Technologies Pvt. Ltd. - AI Driven Smart Spray Robot
 * ESP32-S3 Precision Spray Actuator Controller Firmware
 * Author: Yaragarla Veerababu (2023BTIA009)
 * Specialization: AI & ML (Joy University & AcenAAr Technologies)
 * ============================================================================
 * 
 * Hardware Pin Mapping:
 * - RELAY_PIN    : GPIO 4  (Drives 12V Diaphragm Chemical Spray Pump via Optocoupled Relay)
 * - STATUS_LED   : GPIO 2  (Visual Feedback for Active Spray / Ready State)
 * - BUZZER_PIN   : GPIO 5  (Audible Alert when spray triggers)
 * - RX/TX (UART0): GPIO 44 (RX), GPIO 43 (TX) connected to Raspberry Pi 5 UART
 *
 * Communication Protocol (Raspberry Pi 5 -> ESP32-S3):
 * - "SPRAY_ON:<duration_ms>\n" : Actuate relay for specified milliseconds.
 * - "SPRAY_OFF\n"              : Emergency stop or manual pump shutoff.
 * - "STATUS\n"                 : Query hardware relay & watchdog health.
 * - "PING\n"                   : Heartbeat check -> replies "PONG\n".
 *
 * Safety Features:
 * - Max Spray Duration Watchdog (Caps any spray burst to max 3000 ms to avoid pesticide overuse).
 * - Active-Low relay protection with power-on failsafe.
 */

#define RELAY_PIN    4
#define STATUS_LED   2
#define BUZZER_PIN   5
#define MAX_SPRAY_MS 3000  // Maximum safety cutoff in milliseconds

// State variables
bool isSpraying = false;
unsigned long sprayStartTime = 0;
unsigned long sprayDuration = 0;

void setup() {
  // Initialize Serial Communication at 115200 baud with Raspberry Pi 5
  Serial.begin(115200);

  // Initialize GPIO Pins
  pinMode(RELAY_PIN, OUTPUT);
  pinMode(STATUS_LED, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);

  // Initial Safe State: Pump OFF (Relay OFF is HIGH on active-low relays)
  digitalWrite(RELAY_PIN, HIGH);
  digitalWrite(STATUS_LED, LOW);
  digitalWrite(BUZZER_PIN, LOW);

  // Power-on beep indication
  digitalWrite(BUZZER_PIN, HIGH);
  digitalWrite(STATUS_LED, HIGH);
  delay(150);
  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(STATUS_LED, LOW);

  Serial.println("[ESP32-S3] Smart Spray Actuator Controller Ready. Awaiting commands...");
}

void loop() {
  // 1. Process Incoming Serial Commands
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd.startsWith("SPRAY_ON:")) {
      int duration = cmd.substring(9).toInt();
      if (duration <= 0) duration = 1000;
      if (duration > MAX_SPRAY_MS) duration = MAX_SPRAY_MS; // Watchdog cap

      startSpray(duration);
    } 
    else if (cmd.equalsIgnoreCase("SPRAY_OFF")) {
      stopSpray("MANUAL_COMMAND");
    } 
    else if (cmd.equalsIgnoreCase("STATUS")) {
      Serial.printf("[STATUS] Spraying: %s | RelayPin: %d | WatchdogCap: %d ms\n", 
                    isSpraying ? "ACTIVE" : "STANDBY", digitalRead(RELAY_PIN), MAX_SPRAY_MS);
    } 
    else if (cmd.equalsIgnoreCase("PING")) {
      Serial.println("PONG");
    } 
    else {
      Serial.printf("[ERROR] Unrecognized Command: %s\n", cmd.c_str());
    }
  }

  // 2. Hardware Watchdog & Timer Execution
  if (isSpraying) {
    unsigned long elapsed = millis() - sprayStartTime;
    if (elapsed >= sprayDuration) {
      stopSpray("TIMER_ELAPSED");
    }
  }
}

void startSpray(unsigned long durationMs) {
  isSpraying = true;
  sprayStartTime = millis();
  sprayDuration = durationMs;

  // Activate Relay (Active-Low)
  digitalWrite(RELAY_PIN, LOW);
  digitalWrite(STATUS_LED, HIGH);
  digitalWrite(BUZZER_PIN, HIGH);
  delay(50);
  digitalWrite(BUZZER_PIN, LOW); // Short chirp

  Serial.printf("[EVENT] SPRAY_STARTED: Duration = %lu ms\n", durationMs);
}

void stopSpray(const char* reason) {
  // Deactivate Relay
  digitalWrite(RELAY_PIN, HIGH);
  digitalWrite(STATUS_LED, LOW);
  digitalWrite(BUZZER_PIN, LOW);

  isSpraying = false;
  Serial.printf("[EVENT] SPRAY_STOPPED: Reason = %s\n", reason);
}
