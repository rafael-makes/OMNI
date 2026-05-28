/*
 * OMNI Motor Controller — Raspberry Pi Pico Port
 * Cytron MDD10A + AS5600 encoders via PCA9546 Multiplexer
 */

#include <Wire.h>
#include <math.h>

// ── Robot geometry ──────────────────────────────────────────────────────────
static const float WHEEL_RADIUS_M  = 0.053f;
static const float WHEEL_SEP_M     = 0.305f;

// ── Hardware pins (Pico Mapping) ─────────────────────────────────────────────
static const uint8_t L_PWM = 4;
static const uint8_t L_DIR = 3;
static const uint8_t R_PWM = 5;
static const uint8_t R_DIR = 6;

// ── I2C addresses ─────────────────────────────────────────────────────────────
static const uint8_t PCA9546_ADDR   = 0x70;
static const uint8_t AS5600_ADDR     = 0x36;
static const uint8_t AS5600_ANGLE_H  = 0x0C;

static const uint8_t MUX_LEFT  = 0x01;   // channel 0
static const uint8_t MUX_RIGHT = 0x08;   // channel 3

// ── Timing Intervals ──────────────────────────────────────────────────────────
static const uint32_t PID_INTERVAL_MS  = 10;   // 100 Hz PID
static const uint32_t ODOM_INTERVAL_MS = 53;   // ~20 Hz Odom Output
static const uint32_t CMD_TIMEOUT_MS   = 1000;  // Pi heartbeat at 50ms; 1s gives 20x headroom

// ── PID parameters ───────────────────────────────────────────────────────────
static const float KP = 1.0f;
static const float KI = 0.8f;
static const float KD = 0.0f;
static const float INTEGRAL_MAX = 40.0f;
static const float VEL_ALPHA    = 0.25f;

static const int   MIN_PWM    = 50;   // 10kg high-torque motors — min driveable ~0.16 m/s
static const float FF_GAIN_L  = 16.3f;   // new high-torque motors: left slightly faster, needs lower FF
static const float FF_GAIN_R  = 16.5f;   // new high-torque motors: trimmed down from 16.8 — right ran 1% fast

// ── Encoder sign configurations ──────────────────────────────────────────────
static const float L_ENC_DIR =  1.0f;   // flipped — encoder moved to wheel side
static const float R_ENC_DIR = -1.0f;   // flipped — encoder moved to wheel side

struct WheelState {
    float target_rads;
    float actual_rads;
    float integral;
    float prev_error;
    uint16_t prev_angle;
    float cumulative_rad;
};

WheelState left_wheel  = {};
WheelState right_wheel = {};

// Absolute Tracking Registers
float odom_x   = 0.0f;
float odom_y   = 0.0f;
float odom_th  = 0.0f;
float odom_vx  = 0.0f;
float odom_vth = 0.0f;

float prev_left_cumulative_rad  = 0.0f;
float prev_right_cumulative_rad = 0.0f;

uint32_t last_pid_ms   = 0;
uint32_t last_odom_ms  = 0;
uint32_t last_cmd_ms   = 0;

char serial_buf[64];
uint8_t serial_pos = 0;

// ── I2C Multiplexer Select ───────────────────────────────────────────────────
static void mux_select(uint8_t mask) {
    Wire.beginTransmission(PCA9546_ADDR);
    Wire.write(mask);
    Wire.endTransmission();
    delayMicroseconds(2);   // Pico's core speeds allow us to drop settling time safely
}

// ── AS5600 Atomic Word Read ──────────────────────────────────────────────────
static uint16_t as5600_read_angle() {
    Wire.beginTransmission(AS5600_ADDR);
    Wire.write(AS5600_ANGLE_H);
    if (Wire.endTransmission(false) != 0) return 0xFFFF;

    Wire.requestFrom(AS5600_ADDR, (uint8_t)2);
    if (Wire.available() < 2) return 0xFFFF;

    uint16_t angle = ((uint16_t)Wire.read() << 8) | Wire.read();
    return angle & 0x0FFF;
}

static int16_t angle_delta(uint16_t cur, uint16_t prev) {
    int16_t d = (int16_t)cur - (int16_t)prev;
    if (d >  2048) d -= 4096;
    if (d < -2048) d += 4096;
    return d;
}

static void drive_motor(uint8_t pwm_pin, uint8_t dir_pin, float command) {
    int pwm = (int)fabs(command);
    if (pwm > 255) pwm = 255;
    if (pwm < MIN_PWM) pwm = 0;
    digitalWrite(dir_pin, command >= 0.0f ? HIGH : LOW);
    analogWrite(pwm_pin, pwm);
}

static float pid_update(WheelState &w, float ff_gain, float dt) {
    if (w.target_rads == 0.0f) {
        w.integral   = 0.0f;
        w.prev_error = 0.0f;
        return 0.0f;
    }
    float ff    = w.target_rads * ff_gain;
    float error = w.target_rads - w.actual_rads;
    w.integral += error * dt;
    if (w.integral >  INTEGRAL_MAX) w.integral =  INTEGRAL_MAX;
    if (w.integral < -INTEGRAL_MAX) w.integral = -INTEGRAL_MAX;
    float derivative = (error - w.prev_error) / dt;
    w.prev_error = error;
    return ff + KP * error + KI * w.integral + KD * derivative;
}

static void parse_command(char *line) {
    if (line[0] == 'R') {
        odom_x = odom_y = odom_th = odom_vx = odom_vth = 0.0f;
        left_wheel.cumulative_rad = right_wheel.cumulative_rad = 0.0f;
        prev_left_cumulative_rad = prev_right_cumulative_rad = 0.0f;
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

void setup() {
    Serial.begin(115200); // Handled natively over high-speed USB on the Pico

    // Assign alternative hardware definitions for I2C0 mapping to GP0/GP1
    Wire.setSDA(0);
    Wire.setSCL(1);
    Wire.begin();
    Wire.setClock(400000);

    pinMode(L_DIR, OUTPUT);
    pinMode(R_DIR, OUTPUT);
    pinMode(L_PWM, OUTPUT);
    pinMode(R_PWM, OUTPUT);
    analogWrite(L_PWM, 0);
    analogWrite(R_PWM, 0);

    // Initial position capture
    mux_select(MUX_LEFT);
    left_wheel.prev_angle = as5600_read_angle();
    mux_select(MUX_RIGHT);
    right_wheel.prev_angle = as5600_read_angle();
    mux_select(0x00);

    uint32_t now = millis();
    last_pid_ms = now;
    last_odom_ms = now;
    last_cmd_ms  = now;
}

void loop() {
    uint32_t now = millis();

    // ── Serial processing over native USB link ──────────────────────────────────
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

    if (now - last_cmd_ms > CMD_TIMEOUT_MS) {
        left_wheel.target_rads  = 0.0f;
        right_wheel.target_rads = 0.0f;
    }

    // ── Stabilized 100 Hz Execution Engine ─────────────────────────────────────
    if (now - last_pid_ms >= PID_INTERVAL_MS) {
        float dt = (now - last_pid_ms) / 1000.0f;
        last_pid_ms = now;

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

        float l_out = pid_update(left_wheel,  FF_GAIN_L, dt);
        float r_out = pid_update(right_wheel, FF_GAIN_R, dt);

        drive_motor(L_PWM, L_DIR, -l_out);   // negated — new motors run opposite direction
        drive_motor(R_PWM, R_DIR, -r_out);   // negated — new motors run opposite direction
    }

    // ── Displacement Odometry Engine (20 Hz) ───────────────────────────────────
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

        while (odom_th >  M_PI) odom_th -= 2.0f * M_PI;
        while (odom_th < -M_PI) odom_th += 2.0f * M_PI;

        odom_vx  = ds / dt;
        odom_vth = dth / dt;

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
