/*
 * OMNI Motor Controller
 * Arduino Uno — Cytron MDD10A + AS5600 encoders via PCA9546
 *
 * Serial protocol (115200 baud):
 *   Pi → Arduino: "C,<left_mps>,<right_mps>\n"   velocity command
 *   Pi → Arduino: "R\n"                            reset odometry
 *   Arduino → Pi: "O,<x>,<y>,<th>,<vx>,<vth>,<lv>,<rv>\n"  20 Hz odometry
 *
 * Wiring — Cytron MDD10A:
 *   Left  motor: PWM→D5, DIR→D4
 *   Right motor: PWM→D6, DIR→D7
 *
 * Wiring — PCA9546 I2C mux (0x70):
 *   Left  AS5600: channel 0  (select byte 0x01)
 *   Right AS5600: channel 3  (select byte 0x08)
 *
 * Tune before first drive: WHEEL_RADIUS_M, WHEEL_SEP_M, PID gains.
 */

#include <Wire.h>
#include <math.h>

// ── Robot geometry (MEASURE and update) ──────────────────────────────────────
static const float WHEEL_RADIUS_M  = 0.050f;   // 100mm diameter wheel → 50mm radius
static const float WHEEL_SEP_M     = 0.300f;   // 300mm centre-to-centre track width

// ── Hardware pins (Cytron MDD10A) ────────────────────────────────────────────
static const uint8_t L_PWM = 5;
static const uint8_t L_DIR = 4;
static const uint8_t R_PWM = 6;
static const uint8_t R_DIR = 7;

// ── I2C addresses ─────────────────────────────────────────────────────────────
static const uint8_t PCA9546_ADDR = 0x70;
static const uint8_t AS5600_ADDR  = 0x36;
static const uint8_t AS5600_ANGLE_H = 0x0C;   // raw 12-bit angle MSB

static const uint8_t MUX_LEFT  = 0x01;   // channel 0
static const uint8_t MUX_RIGHT = 0x08;   // channel 3

// ── Timing ────────────────────────────────────────────────────────────────────
static const uint32_t PID_INTERVAL_MS  = 10;   // 100 Hz PID
static const uint32_t ODOM_INTERVAL_MS = 53;
static const uint32_t CMD_TIMEOUT_MS   = 500;  // stop if no command received

// ── PID gains — FF dominates, PID only slowly trims steady-state error ────────
static const float KP = 1.0f;
static const float KI = 0.5f;
static const float KD = 0.0f;
static const float INTEGRAL_MAX = 40.0f;
static const float VEL_ALPHA    = 0.25f;

// ── Motor PWM parameters ──────────────────────────────────────────────────────
// MIN_PWM: below this the worm gears stall — output 0 (deadband), not MIN_PWM
static const int   MIN_PWM  = 60;
// FF_GAIN: feedforward PWM per rad/s, derived from May 2025 bench data:
//   PWM=120 → ~13000 counts/2s → ~10 rad/s  →  120/10 = 12
//   Wheel radius corrected to 50mm — same encoder-to-PWM ratio, gain unchanged
static const float FF_GAIN  = 12.0f;

// ── Stall detection ───────────────────────────────────────────────────────────
// A stall is declared when the motor is commanded above MIN_PWM but the encoder
// reports near-zero velocity for STALL_COUNT consecutive PID cycles (100Hz).
// 30 cycles = 300ms — long enough to ignore brief slowdowns, short enough to
// detect a genuine mechanical stall before the motors overheat.
static const float   STALL_VEL_THRESHOLD = 0.3f;  // rad/s — below this = not moving
static const uint8_t STALL_COUNT_THRESHOLD = 30;   // PID cycles before declaring stall
static uint8_t left_stall_count  = 0;
static uint8_t right_stall_count = 0;
static bool    stall_detected    = false;

// ── Encoder sign convention (validated May 2025) ──────────────────────────────
// M1 left : forward = negative counts → negate to get forward-positive frame
// M2 right: forward = positive counts → no change
static const float L_ENC_DIR = -1.0f;
static const float R_ENC_DIR =  1.0f;

// ── State ─────────────────────────────────────────────────────────────────────
struct WheelState {
    float target_rads;    // commanded rad/s
    float actual_rads;    // measured rad/s
    float integral;
    float prev_error;
    uint16_t prev_angle;  // last AS5600 reading
    float cumulative_rad; // total radians for odometry
};

WheelState left_wheel  = {};
WheelState right_wheel = {};

// Cumulative encoder snapshots for odometry delta
float prev_left_cumulative_rad  = 0.0f;
float prev_right_cumulative_rad = 0.0f;

// Odometry pose
float odom_x   = 0.0f;
float odom_y   = 0.0f;
float odom_th  = 0.0f;
float odom_vx  = 0.0f;
float odom_vth = 0.0f;

uint32_t last_pid_ms   = 0;
uint32_t last_odom_ms  = 0;
uint32_t last_cmd_ms   = 0;

// Serial line buffer
char serial_buf[64];
uint8_t serial_pos = 0;

// ── PCA9546 mux select ────────────────────────────────────────────────────────
static void mux_select(uint8_t mask) {
    Wire.beginTransmission(PCA9546_ADDR);
    Wire.write(mask);
    Wire.endTransmission();
    delayMicroseconds(5);   // required settling time — omitting this causes I2C read failures
}

// ── AS5600 read — returns 0xFFFF on I2C failure ───────────────────────────────
static uint16_t as5600_read_angle() {
    Wire.beginTransmission(AS5600_ADDR);
    Wire.write(AS5600_ANGLE_H);
    if (Wire.endTransmission(false) != 0) return 0xFFFF;
    Wire.requestFrom(AS5600_ADDR, (uint8_t)2);
    if (Wire.available() < 2) return 0xFFFF;
    uint16_t angle = ((uint16_t)Wire.read() << 8) | Wire.read();
    return angle & 0x0FFF;
}

// ── Encoder delta — handles 12-bit wraparound ────────────────────────────────
static int16_t angle_delta(uint16_t cur, uint16_t prev) {
    int16_t d = (int16_t)cur - (int16_t)prev;
    if (d >  2048) d -= 4096;
    if (d < -2048) d += 4096;
    return d;
}

// ── Cytron MDD10A drive ───────────────────────────────────────────────────────
static void drive_motor(uint8_t pwm_pin, uint8_t dir_pin, float command) {
    int pwm = (int)fabs(command);
    if (pwm > 255) pwm = 255;
    // Deadband: outputs below stall threshold become 0 — worm gear holds position.
    // Do NOT snap to MIN_PWM here; that causes bang-bang oscillation.
    if (pwm < MIN_PWM) pwm = 0;
    digitalWrite(dir_pin, command >= 0.0f ? HIGH : LOW);
    analogWrite(pwm_pin, pwm);
}

// ── PID update for one wheel ──────────────────────────────────────────────────
static float pid_update(WheelState &w, float dt) {
    // Hard stop: worm gears are self-locking — bypass PID entirely when target is zero.
    // Without this, encoder noise drives a runaway feedback loop on startup.
    if (w.target_rads == 0.0f) {
        w.integral   = 0.0f;
        w.prev_error = 0.0f;
        return 0.0f;
    }
    float ff    = w.target_rads * FF_GAIN;   // open-loop seed
    float error = w.target_rads - w.actual_rads;
    w.integral += error * dt;
    if (w.integral >  INTEGRAL_MAX) w.integral =  INTEGRAL_MAX;
    if (w.integral < -INTEGRAL_MAX) w.integral = -INTEGRAL_MAX;
    float derivative = (error - w.prev_error) / dt;
    w.prev_error = error;
    return ff + KP * error + KI * w.integral + KD * derivative;
}

// ── Parse "C,left_mps,right_mps" command ────────────────────────────────────
static void parse_command(char *line) {
    if (line[0] == 'R') {
        odom_x = odom_y = odom_th = odom_vx = odom_vth = 0.0f;
        left_wheel.cumulative_rad = right_wheel.cumulative_rad = 0.0f;
        return;
    }
    if (line[0] != 'C') return;

    char *p = line + 2;
    float lmps = atof(p);
    while (*p && *p != ',') p++;
    if (*p == ',') p++;
    float rmps = atof(p);

    left_wheel.target_rads  = lmps / WHEEL_RADIUS_M;
    right_wheel.target_rads = rmps / WHEEL_RADIUS_M;
    last_cmd_ms = millis();
}

// ── Setup ─────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    Wire.begin();
    Wire.setClock(400000);

    pinMode(L_DIR, OUTPUT);
    pinMode(R_DIR, OUTPUT);
    analogWrite(L_PWM, 0);
    analogWrite(R_PWM, 0);

    // Read initial encoder positions
    mux_select(MUX_LEFT);
    left_wheel.prev_angle = as5600_read_angle();
    mux_select(MUX_RIGHT);
    right_wheel.prev_angle = as5600_read_angle();
    mux_select(0x00);

    uint32_t now = millis();
    last_pid_ms = now;
    last_odom_ms = now;
    last_cmd_ms  = now;

    mux_select(MUX_LEFT);
    Wire.beginTransmission(AS5600_ADDR);
    Wire.write(0x0B);
    Wire.endTransmission(false);
    Wire.requestFrom(AS5600_ADDR, (uint8_t)1);
    uint8_t status_l = Wire.read();
    Wire.beginTransmission(AS5600_ADDR);
    Wire.write(0x1A);
    Wire.endTransmission(false);
    Wire.requestFrom(AS5600_ADDR, (uint8_t)1);
    uint8_t agc_l = Wire.read();
    Serial.print("LEFT  status: 0x"); Serial.print(status_l, HEX);
    Serial.print("  AGC: "); Serial.println(agc_l);

    mux_select(MUX_RIGHT);
    Wire.beginTransmission(AS5600_ADDR);
    Wire.write(0x0B);
    Wire.endTransmission(false);
    Wire.requestFrom(AS5600_ADDR, (uint8_t)1);
    uint8_t status_r = Wire.read();
    Wire.beginTransmission(AS5600_ADDR);
    Wire.write(0x1A);
    Wire.endTransmission(false);
    Wire.requestFrom(AS5600_ADDR, (uint8_t)1);
    uint8_t agc_r = Wire.read();
    Serial.print("RIGHT status: 0x"); Serial.print(status_r, HEX);
    Serial.print("  AGC: "); Serial.println(agc_r);
    mux_select(0x00);
}

// ── Main loop ─────────────────────────────────────────────────────────────────
void loop() {
    uint32_t now = millis();

    // ── Serial receive ────────────────────────────────────────────────────────
    while (Serial.available()) {
        char c = Serial.read();
        if (c == '\n') {
            serial_buf[serial_pos] = '\0';
            parse_command(serial_buf);
            serial_pos = 0;
        } else if (serial_pos < sizeof(serial_buf) - 1) {
            serial_buf[serial_pos++] = c;
        }
    }

    // ── Command timeout — stop motors ─────────────────────────────────────────
    if (now - last_cmd_ms > CMD_TIMEOUT_MS) {
        left_wheel.target_rads  = 0.0f;
        right_wheel.target_rads = 0.0f;
    }

    // ── PID loop (100 Hz) ─────────────────────────────────────────────────────
    if (now - last_pid_ms >= PID_INTERVAL_MS) {
        float dt = (now - last_pid_ms) / 1000.0f;
        last_pid_ms = now;

        // Read encoders — skip update on I2C failure to avoid velocity spikes
        mux_select(MUX_LEFT);
        uint16_t la = as5600_read_angle();
        mux_select(MUX_RIGHT);
        uint16_t ra = as5600_read_angle();
        mux_select(0x00);

        if (la != 0xFFFF) {
            int16_t l_delta = angle_delta(la, left_wheel.prev_angle);
            left_wheel.prev_angle = la;
            float l_rad = L_ENC_DIR * l_delta * (2.0f * M_PI / 4096.0f);
            left_wheel.cumulative_rad += l_rad;
            float l_vel = l_rad / dt;
            // Reject implausible spikes (> 50 rad/s ≈ 2.5 m/s on these wheels)
            if (fabs(l_vel) < 50.0f) {
                left_wheel.actual_rads = (VEL_ALPHA * l_vel) + ((1.0f - VEL_ALPHA) * left_wheel.actual_rads);
            }
        }

        if (ra != 0xFFFF) {
            int16_t r_delta = angle_delta(ra, right_wheel.prev_angle);
            right_wheel.prev_angle = ra;
            float r_rad = R_ENC_DIR * r_delta * (2.0f * M_PI / 4096.0f);
            right_wheel.cumulative_rad += r_rad;
            float r_vel = r_rad / dt;
            if (fabs(r_vel) < 50.0f) {
                right_wheel.actual_rads = (VEL_ALPHA * r_vel) + ((1.0f - VEL_ALPHA) * right_wheel.actual_rads);
            }
        }

        float l_out = pid_update(left_wheel, dt);
        float r_out = pid_update(right_wheel, dt);

        drive_motor(L_PWM, L_DIR, l_out);
        drive_motor(R_PWM, R_DIR, r_out);
    }

    // ── Odometry output (20 Hz) ───────────────────────────────────────────────
    if (now - last_odom_ms >= ODOM_INTERVAL_MS) {
        float dt = (now - last_odom_ms) / 1000.0f;
        last_odom_ms = now;

        float current_left_rad  = left_wheel.cumulative_rad;
        float current_right_rad = right_wheel.cumulative_rad;
        float d_rad_l = current_left_rad  - prev_left_cumulative_rad;
        float d_rad_r = current_right_rad - prev_right_cumulative_rad;
        prev_left_cumulative_rad  = current_left_rad;
        prev_right_cumulative_rad = current_right_rad;
        float dl = d_rad_l * WHEEL_RADIUS_M;
        float dr = d_rad_r * WHEEL_RADIUS_M;

        float ds  = (dl + dr) / 2.0f;
        float dth = (dr - dl) / WHEEL_SEP_M;

        odom_x  += ds * cos(odom_th + dth / 2.0f);
        odom_y  += ds * sin(odom_th + dth / 2.0f);
        odom_th += dth;

        // Keep theta in [-π, π]
        while (odom_th >  M_PI) odom_th -= 2.0f * M_PI;
        while (odom_th < -M_PI) odom_th += 2.0f * M_PI;

        odom_vx  = ds / dt;
        odom_vth = dth / dt;

        // O,x,y,theta,vx,vtheta,left_wheel_mps,right_wheel_mps
        Serial.print("O,");
        Serial.print(odom_x,   4); Serial.print(",");
        Serial.print(odom_y,   4); Serial.print(",");
        Serial.print(odom_th,  4); Serial.print(",");
        Serial.print(odom_vx,  4); Serial.print(",");
        Serial.print(odom_vth, 4); Serial.print(",");
        Serial.print(left_wheel.actual_rads  * WHEEL_RADIUS_M, 4); Serial.print(",");
        Serial.println(right_wheel.actual_rads * WHEEL_RADIUS_M, 4);
    }
}
