/*
 * ============================================================
 *  ECG Heartbeat Sensor — V3.1 (BPM Fix)
 *  Board:  ESP32 Dev Module
 *  Sensor: AD8232 ECG Module
 *  IDE:    Arduino IDE
 * ============================================================
 *
 *  WHAT CHANGED FROM V3:
 *  ─────────────────────
 *  ✓ Beat detection now uses DET amplitude directly
 *    (squared derivative was not triggering properly)
 *  ✓ Simple peak detection: finds when signal rises above
 *    threshold then falls back — that's one beat
 *  ✓ Fixed threshold based on observed signal (peaks 200-500)
 *  ✓ Added BPM output to serial for easy reading
 *  ✓ BEAT line now spikes to 500 on each detected beat
 *    (visible on plotter)
 *  ✓ All V3 filtering retained (notch + bandpass + MA)
 *
 *  WIRING:  (same as before)
 *  AD8232 OUTPUT  → GPIO 34
 *  AD8232 LO+     → GPIO 32
 *  AD8232 LO-     → GPIO 33
 *  AD8232 3.3V    → ESP32 3.3V
 *  AD8232 GND     → ESP32 GND
 * ============================================================
 */

#define ECG_PIN    34
#define LO_PLUS    32
#define LO_MINUS   33

// ── Sampling ────────────────────────────────────────────────
const int   FS     = 150;
const int   DT_MS  = 1000 / FS;

// ── Software gain ───────────────────────────────────────────
const float GAIN   = 2.0f;

// ── Filter coefficients (same as V3 — working great) ────────
const float ALPHA_HP   = 0.99f;
const float ALPHA_LP1  = 0.08f;
const float ALPHA_LP2  = 0.15f;

// ── 50 Hz Notch filter (same as V3) ────────────────────────
const float NOTCH_R    = 0.85f;
const float NOTCH_COS  = -0.5f;
const float NOTCH_A1   = -2.0f * NOTCH_COS;
const float NOTCH_A2   = 1.0f;
const float NOTCH_B1   = -2.0f * NOTCH_R * NOTCH_COS;
const float NOTCH_B2   = NOTCH_R * NOTCH_R;

// ── Moving average ──────────────────────────────────────────
const int   MA_SIZE    = 5;

// ── Beat detection (REWRITTEN — amplitude based) ────────────
const int   BEAT_COOLDOWN_MS   = 400;    // Min ms between beats
const int   BPM_MIN            = 35;
const int   BPM_MAX            = 200;
const int   BPM_AVG_SIZE       = 8;
const float BEAT_THRESHOLD_INIT = 120.0f; // Initial threshold
const float PEAK_TRACK_ATTACK  = 0.3f;   // How fast peak tracker rises
const float PEAK_TRACK_DECAY   = 0.998f; // How slow peak tracker falls
const float THRESH_RATIO       = 0.40f;  // Threshold = 40% of tracked peak

// ── Calibration ─────────────────────────────────────────────
const int   CALIBRATION_MS     = 4000;

// ── Lead-off ────────────────────────────────────────────────
const int   LEADOFF_GRACE_MS   = 3000;
const int   LEADOFF_CHECK_MS   = 500;

// ── Serial output ───────────────────────────────────────────
const int   PRINT_EVERY_N      = 1;

// ════════════════════════════════════════════════════════════
//  STATE VARIABLES
// ════════════════════════════════════════════════════════════

// Filter states
float hp          = 0.0f;
float lp1         = 0.0f;
float lp2         = 0.0f;
float prev_x      = 0.0f;
float baseline    = 0.0f;

// Notch filter state
float notch_x1    = 0.0f;
float notch_x2    = 0.0f;
float notch_y1    = 0.0f;
float notch_y2    = 0.0f;

// Moving average
float maBuffer[5] = {0};
int   maIndex     = 0;
float maSum       = 0.0f;

// Beat detection — simple and reliable
float           trackedPeak      = 200.0f;  // Tracks recent peak amplitude
float           beatThreshold    = 120.0f;  // Dynamic threshold
bool            aboveThreshold   = false;    // Currently above threshold?
float           currentPeakVal   = 0.0f;    // Peak value in current beat
unsigned long   lastBeatTime     = 0;
unsigned long   beatIntervals[8];
int             beatIndex        = 0;
int             beatCount        = 0;
float           currentBPM       = 0.0f;
bool            beatPulse        = false;
int             beatDisplayCount = 0;        // For showing beat spike on plotter

// Calibration
bool            calibrating      = true;
unsigned long   calibStartTime   = 0;
float           calibMax         = 0.0f;

// Lead-off
unsigned long   leadoffStartTime = 0;
bool            leadsConnected   = true;
unsigned long   lastLeadoffCheck = 0;

// Counter
int             sampleCount      = 0;

// ════════════════════════════════════════════════════════════
//  16x OVERSAMPLING
// ════════════════════════════════════════════════════════════
static int readECG_oversample() {
  long s = 0;
  for (int i = 0; i < 16; i++) {
    s += analogRead(ECG_PIN);
    delayMicroseconds(100);
  }
  return (int)(s >> 4);
}

// ════════════════════════════════════════════════════════════
//  50 Hz NOTCH FILTER
// ════════════════════════════════════════════════════════════
float applyNotchFilter(float x) {
  float y = x + NOTCH_A1 * notch_x1 + NOTCH_A2 * notch_x2
              - NOTCH_B1 * notch_y1 - NOTCH_B2 * notch_y2;
  notch_x2 = notch_x1;
  notch_x1 = x;
  notch_y2 = notch_y1;
  notch_y1 = y;
  return y;
}

// ════════════════════════════════════════════════════════════
//  MOVING AVERAGE
// ════════════════════════════════════════════════════════════
float applyMovingAverage(float newVal) {
  maSum -= maBuffer[maIndex];
  maBuffer[maIndex] = newVal;
  maSum += newVal;
  maIndex = (maIndex + 1) % MA_SIZE;
  return maSum / (float)MA_SIZE;
}

// ════════════════════════════════════════════════════════════
//  BPM CALCULATION
// ════════════════════════════════════════════════════════════
float calculateBPM() {
  if (beatCount < 2) return 0.0f;

  int count = min(beatCount, (int)BPM_AVG_SIZE);
  unsigned long sum = 0;
  int validCount = 0;

  for (int i = 0; i < count; i++) {
    if (beatIntervals[i] >= 300 && beatIntervals[i] <= 1700) {
      sum += beatIntervals[i];
      validCount++;
    }
  }
  if (validCount < 2) return currentBPM;

  float avgInterval = (float)sum / (float)validCount;
  float bpm = 60000.0f / avgInterval;

  if (bpm < BPM_MIN || bpm > BPM_MAX) return currentBPM;
  return bpm;
}

// ════════════════════════════════════════════════════════════
//  SETUP
// ════════════════════════════════════════════════════════════
void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println("=================================");
  Serial.println("  ECG Monitor V3.1 (BPM Fix)");
  Serial.println("=================================");
  Serial.println("BOOT");

  pinMode(LO_PLUS,  INPUT);
  pinMode(LO_MINUS, INPUT);

  analogReadResolution(12);
  analogSetPinAttenuation(ECG_PIN, ADC_11db);

  for (int i = 0; i < BPM_AVG_SIZE; i++) beatIntervals[i] = 0;
  for (int i = 0; i < MA_SIZE; i++) maBuffer[i] = 0;

  calibStartTime = millis();
  lastBeatTime = millis();

  Serial.println("Calibrating... sit still for 4 seconds.");
  Serial.println("FORMAT: DET,DIS,BPM,BEAT");
}

// ════════════════════════════════════════════════════════════
//  MAIN LOOP
// ════════════════════════════════════════════════════════════
void loop() {
  unsigned long now = millis();

  // ── Timing ──────────────────────────────────────────────
  static unsigned long lastSampleTime = 0;
  if (now - lastSampleTime < (unsigned long)DT_MS) {
    yield();
    return;
  }
  lastSampleTime = now;

  // ── Lead-off check ──────────────────────────────────────
  if (now - lastLeadoffCheck >= (unsigned long)LEADOFF_CHECK_MS) {
    lastLeadoffCheck = now;
    bool leadoff = digitalRead(LO_PLUS) || digitalRead(LO_MINUS);

    if (leadoff) {
      if (leadoffStartTime == 0) leadoffStartTime = now;
      if (now - leadoffStartTime > (unsigned long)LEADOFF_GRACE_MS) {
        if (leadsConnected) {
          leadsConnected = false;
          Serial.println("STATUS:LEADS_OFF");
        }
      }
    } else {
      leadoffStartTime = 0;
      if (!leadsConnected) {
        leadsConnected = true;
        Serial.println("STATUS:LEADS_ON");
        hp = 0; lp1 = 0; lp2 = 0; prev_x = 0;
        notch_x1 = 0; notch_x2 = 0; notch_y1 = 0; notch_y2 = 0;
        maSum = 0; maIndex = 0;
        for (int i = 0; i < MA_SIZE; i++) maBuffer[i] = 0;
        trackedPeak = 200.0f;
        beatThreshold = BEAT_THRESHOLD_INIT;
        aboveThreshold = false;
        calibrating = true;
        calibStartTime = now;
        calibMax = 0;
      }
    }
  }

  if (!leadsConnected) {
    sampleCount++;
    if (sampleCount % (FS / 2) == 0) {
      Serial.println("DET:0,DIS:0,BPM:0,BEAT:0");
    }
    return;
  }

  // ── Read sensor ─────────────────────────────────────────
  int raw = readECG_oversample();
  float x = (float)raw;

  // ── FILTER PIPELINE (same as V3) ────────────────────────
  float notched = applyNotchFilter(x);
  hp = ALPHA_HP * (hp + notched - prev_x);
  prev_x = notched;
  lp1 = lp1 + ALPHA_LP1 * (hp - lp1);
  lp2 = lp2 + ALPHA_LP2 * (lp1 - lp2);
  float amplified = lp2 * GAIN;
  float smoothed = applyMovingAverage(amplified);

  baseline = 0.998f * baseline + 0.002f * x;

  // ── BEAT DETECTION (simple peak finding) ────────────────
  //
  // Logic:
  // 1. Signal rises above threshold → mark "above"
  // 2. Track the maximum value while above
  // 3. Signal falls below threshold → that was one beat
  // 4. Use the peak value to update the adaptive threshold
  // 5. Enforce cooldown between beats
  //
  beatPulse = false;

  if (!aboveThreshold) {
    // Waiting for signal to cross above threshold
    if (smoothed > beatThreshold) {
      aboveThreshold = true;
      currentPeakVal = smoothed;
    }
  } else {
    // We're in a peak — track the maximum
    if (smoothed > currentPeakVal) {
      currentPeakVal = smoothed;
    }

    // Check if signal has fallen back below threshold
    if (smoothed < beatThreshold * 0.5f) {
      aboveThreshold = false;

      // This was a complete peak — register as beat
      unsigned long timeSinceLastBeat = now - lastBeatTime;

      if (timeSinceLastBeat > (unsigned long)BEAT_COOLDOWN_MS) {
        beatPulse = true;
        beatDisplayCount = 3;  // Show beat marker for 3 print cycles

        // Update peak tracker (fast attack, slow decay)
        trackedPeak = trackedPeak + PEAK_TRACK_ATTACK * (currentPeakVal - trackedPeak);

        // Update threshold
        beatThreshold = trackedPeak * THRESH_RATIO;
        if (beatThreshold < 50.0f) beatThreshold = 50.0f;   // Minimum
        if (beatThreshold > 300.0f) beatThreshold = 300.0f;  // Maximum

        // Store interval and calculate BPM
        beatIntervals[beatIndex] = timeSinceLastBeat;
        beatIndex = (beatIndex + 1) % BPM_AVG_SIZE;
        if (beatCount < BPM_AVG_SIZE) beatCount++;

        lastBeatTime = now;
        currentBPM = calculateBPM();
      }

      currentPeakVal = 0.0f;
    }
  }

  // Slowly decay the tracked peak so threshold adapts downward
  // if signal amplitude decreases
  trackedPeak *= PEAK_TRACK_DECAY;
  if (trackedPeak < 100.0f) trackedPeak = 100.0f;

  // ── CALIBRATION ─────────────────────────────────────────
  if (calibrating) {
    if (smoothed > calibMax) calibMax = smoothed;

    if (now - calibStartTime > (unsigned long)CALIBRATION_MS) {
      calibrating = false;

      if (calibMax > 50.0f) {
        trackedPeak = calibMax;
        beatThreshold = calibMax * THRESH_RATIO;
        if (beatThreshold < 50.0f) beatThreshold = 50.0f;
      }

      Serial.print("STATUS:CALIBRATED,PEAK:");
      Serial.print((int)calibMax);
      Serial.print(",THRESH:");
      Serial.println((int)beatThreshold);
    }
  }

  // ── Serial output ───────────────────────────────────────
  sampleCount++;
  if (sampleCount % PRINT_EVERY_N == 0) {
    int det = (int)smoothed;
    int dis = (int)(baseline + smoothed);
    int bpm = (int)(currentBPM + 0.5f);

    // Beat marker: spike to 500 when beat detected (visible on plotter)
    int beat = 0;
    if (beatDisplayCount > 0) {
      beat = 500;
      beatDisplayCount--;
    }

    Serial.print("DET:");
    Serial.print(det);
    Serial.print(",DIS:");
    Serial.print(dis);
    Serial.print(",BPM:");
    Serial.print(bpm);
    Serial.print(",BEAT:");
    Serial.println(beat);
  }

  if (sampleCount > 100000) sampleCount = 0;
}
