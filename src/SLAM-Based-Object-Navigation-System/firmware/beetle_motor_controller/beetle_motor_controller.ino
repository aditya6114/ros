/*
 * Beetle ESP32-S3 motor controller.
 *
 * USB serial protocol, 115200 baud:
 *   Pi -> ESP32: CMD <linear_m_per_s> <angular_rad_per_s>\n
 *   ESP32 -> Pi: ENC <left_ticks> <right_ticks> <millis>\n
 * A command is valid for 300 ms.  On timeout both MDD20A channels are stopped.
 * GPIO 8/9 is the front-left encoder (left odometry); GPIO 10/11 is the
 * rear-right encoder (right odometry).  The other motors follow their side.
 */

#include <Arduino.h>

// Right MDD20A: rear on channel 1, front on channel 2.
constexpr int RIGHT_DIR1 = 5;
constexpr int RIGHT_DIR2 = 16;
constexpr int RIGHT_PWM1 = 4;
constexpr int RIGHT_PWM2 = 15;
// Left MDD20A: rear on channel 1, front on channel 2.
constexpr int LEFT_DIR1 = 7;
constexpr int LEFT_DIR2 = 18;
constexpr int LEFT_PWM1 = 6;
constexpr int LEFT_PWM2 = 17;

constexpr int LEFT_ENC_A = 8;   // FL white
constexpr int LEFT_ENC_B = 9;   // FL yellow
constexpr int RIGHT_ENC_A = 10; // RR white
constexpr int RIGHT_ENC_B = 11; // RR yellow

constexpr float WHEEL_DIAMETER_M = 0.106f;
constexpr float TRACK_WIDTH_M = 0.276f;
// Set this to the number of counts observed for exactly one wheel revolution.
// 5200 is used from the supplied encoder specification.  Do not multiply by 4
// again unless a one-revolution test proves that 5200 is only the base PPR.
constexpr float COUNTS_PER_WHEEL_REV = 5200.0f;
constexpr float METERS_PER_COUNT = PI * WHEEL_DIAMETER_M / COUNTS_PER_WHEEL_REV;

constexpr uint32_t PWM_FREQ_HZ = 20000;
constexpr uint8_t PWM_BITS = 10;
constexpr int PWM_MAX = (1 << PWM_BITS) - 1;
constexpr uint32_t CONTROL_PERIOD_MS = 20;
constexpr uint32_t COMMAND_TIMEOUT_MS = 300;

// Tune these with the wheels raised first, then on the floor at low speed.
constexpr float KP = 1100.0f;
constexpr float KI = 250.0f;
constexpr float KD = 0.0f;
constexpr float MAX_WHEEL_SPEED_MPS = 0.45f;

volatile int64_t left_ticks = 0;
volatile int64_t right_ticks = 0;
volatile uint8_t left_encoder_state = 0;
volatile uint8_t right_encoder_state = 0;

float command_linear = 0.0f;
float command_angular = 0.0f;
uint32_t last_command_ms = 0;
uint32_t last_control_ms = 0;
int64_t previous_left_ticks = 0;
int64_t previous_right_ticks = 0;

struct PID {
  float integral = 0.0f;
  float previous_error = 0.0f;
  float update(float target, float measured, float dt) {
    const float error = target - measured;
    integral = constrain(integral + error * dt, -1.0f, 1.0f);
    const float derivative = (error - previous_error) / dt;
    previous_error = error;
    return KP * error + KI * integral + KD * derivative;
  }
  void reset() { integral = 0.0f; previous_error = 0.0f; }
};
PID left_pid, right_pid;

// Valid quadrature transitions, indexed by old AB state then new AB state.
constexpr int8_t QUADRATURE_DELTA[16] = {0, -1, 1, 0, 1, 0, 0, -1,
                                          -1, 0, 0, 1, 0, 1, -1, 0};

void IRAM_ATTR leftEncoderISR() {
  const uint8_t state = (digitalRead(LEFT_ENC_A) << 1) | digitalRead(LEFT_ENC_B);
  left_ticks += QUADRATURE_DELTA[(left_encoder_state << 2) | state];
  left_encoder_state = state;
}

void IRAM_ATTR rightEncoderISR() {
  const uint8_t state = (digitalRead(RIGHT_ENC_A) << 1) | digitalRead(RIGHT_ENC_B);
  right_ticks += QUADRATURE_DELTA[(right_encoder_state << 2) | state];
  right_encoder_state = state;
}

void setSideMotor(bool left, int pwm) {
  pwm = constrain(pwm, -PWM_MAX, PWM_MAX);
  const bool forward = pwm >= 0;
  const int duty = abs(pwm);
  const int dir1 = left ? LEFT_DIR1 : RIGHT_DIR1;
  const int dir2 = left ? LEFT_DIR2 : RIGHT_DIR2;
  const int channel1 = left ? 2 : 0;
  const int channel2 = left ? 3 : 1;
  digitalWrite(dir1, forward ? HIGH : LOW);
  digitalWrite(dir2, forward ? HIGH : LOW);
  ledcWrite(channel1, duty);
  ledcWrite(channel2, duty);
}

void stopMotors() {
  setSideMotor(true, 0);
  setSideMotor(false, 0);
  left_pid.reset();
  right_pid.reset();
}

void readCommand() {
  static char line[80];
  static size_t used = 0;
  while (Serial.available()) {
    const char c = static_cast<char>(Serial.read());
    if (c == '\n' || c == '\r') {
      line[used] = '\0';
      float linear, angular;
      if (sscanf(line, "CMD %f %f", &linear, &angular) == 2) {
        command_linear = constrain(linear, -MAX_WHEEL_SPEED_MPS, MAX_WHEEL_SPEED_MPS);
        command_angular = constrain(angular, -3.0f, 3.0f);
        last_command_ms = millis();
      }
      used = 0;
    } else if (used < sizeof(line) - 1) {
      line[used++] = c;
    } else {
      used = 0;
    }
  }
}

void setup() {
  Serial.begin(115200);
  for (int pin : {RIGHT_DIR1, RIGHT_DIR2, LEFT_DIR1, LEFT_DIR2}) pinMode(pin, OUTPUT);
  ledcSetup(0, PWM_FREQ_HZ, PWM_BITS); ledcAttachPin(RIGHT_PWM1, 0);
  ledcSetup(1, PWM_FREQ_HZ, PWM_BITS); ledcAttachPin(RIGHT_PWM2, 1);
  ledcSetup(2, PWM_FREQ_HZ, PWM_BITS); ledcAttachPin(LEFT_PWM1, 2);
  ledcSetup(3, PWM_FREQ_HZ, PWM_BITS); ledcAttachPin(LEFT_PWM2, 3);
  pinMode(LEFT_ENC_A, INPUT_PULLUP); pinMode(LEFT_ENC_B, INPUT_PULLUP);
  pinMode(RIGHT_ENC_A, INPUT_PULLUP); pinMode(RIGHT_ENC_B, INPUT_PULLUP);
  left_encoder_state = (digitalRead(LEFT_ENC_A) << 1) | digitalRead(LEFT_ENC_B);
  right_encoder_state = (digitalRead(RIGHT_ENC_A) << 1) | digitalRead(RIGHT_ENC_B);
  attachInterrupt(digitalPinToInterrupt(LEFT_ENC_A), leftEncoderISR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(LEFT_ENC_B), leftEncoderISR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(RIGHT_ENC_A), rightEncoderISR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(RIGHT_ENC_B), rightEncoderISR, CHANGE);
  stopMotors();
  last_command_ms = millis();
  last_control_ms = millis();
}

void loop() {
  readCommand();
  const uint32_t now = millis();
  if (now - last_control_ms < CONTROL_PERIOD_MS) return;
  const float dt = (now - last_control_ms) / 1000.0f;
  last_control_ms = now;

  noInterrupts();
  const int64_t left_now = left_ticks;
  const int64_t right_now = right_ticks;
  interrupts();
  const float left_measured = (left_now - previous_left_ticks) * METERS_PER_COUNT / dt;
  const float right_measured = (right_now - previous_right_ticks) * METERS_PER_COUNT / dt;
  previous_left_ticks = left_now;
  previous_right_ticks = right_now;

  if (now - last_command_ms > COMMAND_TIMEOUT_MS) {
    command_linear = command_angular = 0.0f;
    stopMotors();
  } else {
    const float left_target = command_linear - command_angular * TRACK_WIDTH_M * 0.5f;
    const float right_target = command_linear + command_angular * TRACK_WIDTH_M * 0.5f;
    setSideMotor(true, lroundf(left_pid.update(left_target, left_measured, dt)));
    setSideMotor(false, lroundf(right_pid.update(right_target, right_measured, dt)));
  }
  Serial.printf("ENC %lld %lld %lu\n", left_now, right_now, static_cast<unsigned long>(now));
}
