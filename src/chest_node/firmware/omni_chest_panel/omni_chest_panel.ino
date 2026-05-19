/*
 * OMNI Robot — Chest Panel Firmware
 * ESP32 DevKit 30-pin
 *
 * Wiring:
 *   ST7920 128x64 LCD (U8g2 SW SPI)  SCK=13  MOSI=12  CS=14  RST=4
 *   WS2812B left  panel 64 LEDs      GPIO 27
 *   WS2812B right panel 64 LEDs      GPIO 26
 *   Encoder A  (BTN_EN1)             GPIO 18
 *   Encoder B  (BTN_EN2)             GPIO 19
 *   Encoder Btn (BTN_ENC)            GPIO 21
 *   Beeper                           GPIO 22
 *   Pi UART0 (Pi GPIO 14/15)         Serial @ 115200
 *
 * Pi → ESP32 protocol:
 *   STATE\n               IDLE / LISTENING / SPEAKING / NAVIGATING / EXPLORING / DOCKING / ERROR
 *   AUDIO:v0,...,v15\n    16 floats 0.0-1.0 (left panel = v0-v7, right = v8-v15)
 *   BAT:<pct>\n           integer 0-100
 *   SAFETY:OK\n           all clear
 *   SAFETY:<faults>\n     comma-separated active fault names e.g. "watchdog,estop"
 *
 * ESP32 → Pi:
 *   WIFI:<ssid>:<password>\n    sent when user confirms WiFi settings menu
 *
 * Libraries required (Arduino Library Manager):
 *   FastLED by Daniel Garcia
 *   U8g2 by oliver
 * Board: ESP32 Dev Module (esp32 by Espressif Systems)
 */

#include <FastLED.h>
#include <U8g2lib.h>

// ── Pins ──────────────────────────────────────────────────────────────────────
#define PIN_LED_LEFT   27
#define PIN_LED_RIGHT  26
#define NUM_LEDS       64
#define PIN_ENC_A      18
#define PIN_ENC_B      19
#define PIN_ENC_BTN    21
#define PIN_BEEPER     22

// ── Display ───────────────────────────────────────────────────────────────────
// Constructor: rotation, SCK, MOSI, CS, RST
U8G2_ST7920_128X64_F_SW_SPI u8g2(U8G2_R0, 13, 12, 14, 4);

// ── LEDs ──────────────────────────────────────────────────────────────────────
CRGB leftLEDs[NUM_LEDS];
CRGB rightLEDs[NUM_LEDS];

// ── Robot state ───────────────────────────────────────────────────────────────
enum RobotState {
    S_IDLE, S_LISTENING, S_SPEAKING,
    S_NAVIGATING, S_EXPLORING, S_DOCKING, S_ERROR
};

RobotState robotState  = S_IDLE;
int        batteryPct  = 0;
float      audio[16]   = {};  // 0.0-1.0, updated at 10 Hz from Pi
String     safetyStatus = "OK";  // content after "SAFETY:" from Pi
bool       safetyFault  = false; // true when status is not OK

// ── App mode ──────────────────────────────────────────────────────────────────
enum AppMode {
    MODE_NORMAL,       // display state + battery, LEDs animate
    MODE_MENU,         // settings menu
    MODE_WIFI_SSID,    // entering SSID
    MODE_WIFI_PASS,    // entering password
    MODE_WIFI_CONFIRM  // confirm before sending
};

AppMode appMode = MODE_NORMAL;

// ── Menu ──────────────────────────────────────────────────────────────────────
const char* MENU_ITEMS[] = { "WiFi Settings", "Exit" };
const int   MENU_COUNT   = 2;
int menuIdx = 0;

// ── WiFi character entry ──────────────────────────────────────────────────────
// Character set available for SSID/password entry
const String CS =
    " abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "!@#$%&*()_-+=;'\",./";

// Two virtual positions after the charset: [DEL] and [OK]
int csPos = 0;
#define CS_LEN   ((int)CS.length())
#define CS_DEL   (CS_LEN)
#define CS_OK    (CS_LEN + 1)
#define CS_TOTAL (CS_LEN + 2)

String wifiSSID = "";
String wifiPass = "";

// ── UART receive buffer ───────────────────────────────────────────────────────
String rxBuf = "";

// ── Timers / flags ────────────────────────────────────────────────────────────
bool     dispDirty = true;
uint32_t lastLedMs = 0;
uint32_t lastEncMs = 0;
uint32_t lastBtnMs = 0;
int      lastEncA  = HIGH;
bool     btnHeld   = false;

// ── Beeper ────────────────────────────────────────────────────────────────────
void beep(int hz, int ms) {
    tone(PIN_BEEPER, hz, ms);
}

// ── State helpers ─────────────────────────────────────────────────────────────
const char* stateStr(RobotState s) {
    switch (s) {
        case S_IDLE:       return "IDLE";
        case S_LISTENING:  return "LISTENING";
        case S_SPEAKING:   return "SPEAKING";
        case S_NAVIGATING: return "NAVIGATING";
        case S_EXPLORING:  return "EXPLORING";
        case S_DOCKING:    return "DOCKING";
        case S_ERROR:      return "ERROR";
        default:           return "---";
    }
}

CRGB stateColor(RobotState s) {
    switch (s) {
        case S_IDLE:       return CRGB(0,    80,  255);  // blue
        case S_LISTENING:  return CRGB(0,   255,  255);  // cyan
        case S_SPEAKING:   return CRGB(0,   220,    0);  // green
        case S_NAVIGATING: return CRGB(30,  150,  255);  // blue-white
        case S_EXPLORING:  return CRGB(160,   0,  255);  // purple
        case S_DOCKING:    return CRGB(255, 200,    0);  // yellow
        case S_ERROR:      return CRGB(255,   0,    0);  // red
        default:           return CRGB(0,    0,  255);
    }
}

// ── Safety fault label lookup ─────────────────────────────────────────────────
// Converts the raw comma-separated fault string from the Pi into short
// human-readable labels joined by spaces, e.g. "watchdog,estop" → "NO NAV E-STOP".
// Any unknown fault name is passed through unchanged as a fallback.
String faultLabel(const String& name) {
    if (name == "watchdog")  return "NO NAV";
    if (name == "estop")     return "E-STOP";
    if (name == "stall")     return "STALL";
    if (name == "tilt")      return "TILTING";
    if (name == "voltage")   return "LOW VOLT";
    if (name == "proximity") return "OBSTACLE";
    return name;  // unknown fault — show raw name so nothing is silently lost
}

// Split safetyStatus on commas, look up each token, return joined label string.
String buildFaultDisplay() {
    String result  = "";
    String token   = "";
    String src     = safetyStatus;  // e.g. "watchdog,estop" (spaces already stripped)

    for (int i = 0; i <= (int)src.length(); i++) {
        char c = (i < (int)src.length()) ? src[i] : ',';  // treat end-of-string as delimiter
        if (c == ',') {
            token.trim();
            if (token.length() > 0) {
                if (result.length() > 0) result += " ";
                result += faultLabel(token);
            }
            token = "";
        } else {
            token += c;
        }
    }
    return result;
}

// ── LED helpers ───────────────────────────────────────────────────────────────

// Serpentine 8×8 panel: row 0 = top, col 0 = left, data input at top-left.
// If bars appear inverted or sideways, swap rows or transpose col/row here.
int ledIdx(int col, int row) {
    return (row % 2 == 0) ? row * 8 + col : row * 8 + (7 - col);
}

// Compute bar level (0.0-1.0) for column col on a panel.
// side: 0 = left panel (audio[0..7]), 1 = right panel (audio[8..15])
float barLevel(int col, int side) {
    float t  = millis() / 1000.0f;
    float bi = (float)(col + side * 8);  // global bar index 0-15

    switch (robotState) {
        case S_IDLE:
            // Slow meandering sine wave
            return 0.18f + 0.20f * sinf(t * 0.75f + bi * 0.45f);

        case S_LISTENING: {
            // Pulses radiate outward from panel centre
            float dist = fabsf(col - 3.5f);
            float v    = sinf(t * 2.5f - dist * 0.7f);
            return 0.12f + 0.55f * max(0.0f, v);
        }

        case S_SPEAKING:
            // Bar heights driven directly by audio levels
            return constrain(audio[col + side * 8], 0.0f, 1.0f);

        case S_NAVIGATING: {
            // Single bright peak sweeps left→right across all 16 bars
            float pos  = fmodf(t * 5.5f, 20.0f) - 2.0f;
            float dist = fabsf(bi - pos);
            return constrain(0.95f - dist * 0.28f, 0.0f, 1.0f);
        }

        case S_EXPLORING:
            // Phase ripple across full panel width
            return 0.18f + 0.55f * (0.5f + 0.5f * sinf(t * 1.8f + bi * 0.55f));

        case S_DOCKING: {
            // Global fade × slow per-bar wave
            float fade = 0.40f + 0.35f * sinf(t * 1.0f);
            float wave = 0.50f + 0.30f * sinf(t * 0.65f + (float)col * 0.4f);
            return constrain(fade * wave, 0.0f, 1.0f);
        }

        case S_ERROR:
            // All bars flash in unison
            return (fmodf(t, 0.4f) < 0.2f) ? 1.0f : 0.0f;

        default:
            return 0.3f;
    }
}

void updateLEDs() {
    CRGB color = stateColor(robotState);

    for (int col = 0; col < 8; col++) {
        int hL = constrain((int)(barLevel(col, 0) * 8.0f + 0.5f), 0, 8);
        int hR = constrain((int)(barLevel(col, 1) * 8.0f + 0.5f), 0, 8);

        for (int row = 0; row < 8; row++) {
            // row 7 = bottom of panel; bars grow upward
            leftLEDs[ledIdx(col, row)]  = (row >= 8 - hL) ? color : CRGB::Black;
            rightLEDs[ledIdx(col, row)] = (row >= 8 - hR) ? color : CRGB::Black;
        }
    }
    FastLED.show();
}

// ── Display: normal mode ──────────────────────────────────────────────────────
void drawNormal() {
    char bat[14];
    snprintf(bat, sizeof(bat), "BAT: %d%%", batteryPct);

    u8g2.clearBuffer();

    // "OMNI" title — shifted up 3px to make room for safety row
    u8g2.setFont(u8g2_font_10x20_tr);
    u8g2.drawStr((128 - u8g2.getStrWidth("OMNI")) / 2, 17, "OMNI");

    u8g2.drawHLine(0, 20, 128);

    // State name — centred
    u8g2.setFont(u8g2_font_7x13B_tr);
    int sw = u8g2.getStrWidth(stateStr(robotState));
    u8g2.drawStr((128 - sw) / 2, 33, stateStr(robotState));

    // Safety status row
    // Shows [SAFE] when all clear, or !<labels> when faulted.
    // buildFaultDisplay() maps raw fault names to readable labels.
    u8g2.setFont(u8g2_font_6x10_tr);
    if (safetyFault) {
        String display = buildFaultDisplay();   // e.g. "NO NAV E-STOP"
        if (display.length() > 18)
            display = display.substring(0, 17) + "~";  // truncate if too many faults
        String safeLine = "!" + display;
        u8g2.drawStr(2, 46, safeLine.c_str());
    } else {
        u8g2.drawStr(2, 46, "[SAFE]");
    }

    // Battery text + bar — shifted down 2px
    u8g2.setFont(u8g2_font_6x10_tr);
    u8g2.drawStr(2, 60, bat);

    int barW = (batteryPct * 44) / 100;
    u8g2.drawFrame(82, 51, 44, 10);
    if (barW > 0) u8g2.drawBox(83, 52, barW, 8);

    u8g2.sendBuffer();
}

// ── Display: settings menu ────────────────────────────────────────────────────
void drawMenu() {
    u8g2.clearBuffer();
    u8g2.setFont(u8g2_font_6x10_tr);
    u8g2.drawStr(2, 10, "OMNI Settings");
    u8g2.drawHLine(0, 12, 128);

    for (int i = 0; i < MENU_COUNT; i++) {
        int y = 28 + i * 16;
        if (i == menuIdx) {
            u8g2.drawBox(0, y - 11, 128, 13);
            u8g2.setDrawColor(0);
            u8g2.drawStr(6, y, MENU_ITEMS[i]);
            u8g2.setDrawColor(1);
        } else {
            u8g2.drawStr(6, y, MENU_ITEMS[i]);
        }
    }
    u8g2.sendBuffer();
}

// ── Display: character wheel input ────────────────────────────────────────────
// title: "Enter SSID:" or "Enter PWD:"
void drawCharWheel(const char* title, const String& current) {
    // Map wheel position to display string
    auto charAt = [](int pos) -> String {
        if (pos == CS_DEL) return "[DEL]";
        if (pos == CS_OK)  return "[ OK]";
        return String(CS[pos]);
    };

    String curS  = charAt(csPos);
    String prevS = charAt((csPos - 1 + CS_TOTAL) % CS_TOTAL);
    String nextS = charAt((csPos + 1) % CS_TOTAL);

    // Truncate display string if too long
    String disp = (current.length() < 17)
        ? current + "_"
        : "..." + current.substring(current.length() - 13) + "_";

    u8g2.clearBuffer();

    u8g2.setFont(u8g2_font_6x10_tr);
    u8g2.drawStr(2, 10, title);
    u8g2.drawHLine(0, 12, 128);

    // String built so far
    u8g2.drawStr(2, 26, disp.c_str());
    u8g2.drawHLine(0, 28, 128);

    // Prev char (left, smaller)
    u8g2.setFont(u8g2_font_6x10_tr);
    u8g2.drawStr(4, 44, prevS.c_str());

    // Current char (centre, highlighted)
    u8g2.setFont(u8g2_font_7x13B_tr);
    int cw = u8g2.getStrWidth(curS.c_str());
    int cx = (128 - cw) / 2;
    u8g2.drawBox(cx - 3, 32, cw + 6, 16);
    u8g2.setDrawColor(0);
    u8g2.drawStr(cx, 44, curS.c_str());
    u8g2.setDrawColor(1);

    // Next char (right, smaller)
    u8g2.setFont(u8g2_font_6x10_tr);
    int nw = u8g2.getStrWidth(nextS.c_str());
    u8g2.drawStr(124 - nw, 44, nextS.c_str());

    // Hint
    u8g2.drawStr(2, 58, "turn=char  press=add");

    u8g2.sendBuffer();
}

// ── Display: WiFi confirm screen ──────────────────────────────────────────────
void drawConfirm() {
    String ssidLine = "SSID: " + wifiSSID;
    if (ssidLine.length() > 20) ssidLine = ssidLine.substring(0, 19) + "~";

    u8g2.clearBuffer();
    u8g2.setFont(u8g2_font_6x10_tr);
    u8g2.drawStr(2, 10, "Send WiFi config?");
    u8g2.drawHLine(0, 12, 128);
    u8g2.drawStr(2, 26, ssidLine.c_str());
    u8g2.drawStr(2, 40, "PWD:  ********");

    // YES / NO toggle
    if (menuIdx == 0) {
        u8g2.drawBox(8, 48, 38, 13);
        u8g2.setDrawColor(0); u8g2.drawStr(16, 58, "YES"); u8g2.setDrawColor(1);
        u8g2.drawStr(78, 58, "NO");
    } else {
        u8g2.drawStr(16, 58, "YES");
        u8g2.drawBox(72, 48, 28, 13);
        u8g2.setDrawColor(0); u8g2.drawStr(78, 58, "NO"); u8g2.setDrawColor(1);
    }
    u8g2.sendBuffer();
}

void updateDisplay() {
    switch (appMode) {
        case MODE_NORMAL:       drawNormal();                          break;
        case MODE_MENU:         drawMenu();                            break;
        case MODE_WIFI_SSID:    drawCharWheel("Enter SSID:", wifiSSID); break;
        case MODE_WIFI_PASS:    drawCharWheel("Enter PWD:",  wifiPass); break;
        case MODE_WIFI_CONFIRM: drawConfirm();                         break;
    }
    dispDirty = false;
}

// ── UART parsing ──────────────────────────────────────────────────────────────
void parseAudio(const String& s) {
    int idx = 0, start = 0;
    for (int i = 0; i <= (int)s.length() && idx < 16; i++) {
        if (i == (int)s.length() || s[i] == ',') {
            audio[idx++] = s.substring(start, i).toFloat();
            start = i + 1;
        }
    }
}

void parseLine(const String& line) {
    bool stateChange = false;

    if      (line == "IDLE")       { robotState = S_IDLE;       stateChange = true; }
    else if (line == "LISTENING")  { robotState = S_LISTENING;  stateChange = true; }
    else if (line == "SPEAKING")   { robotState = S_SPEAKING;   stateChange = true; }
    else if (line == "NAVIGATING") { robotState = S_NAVIGATING; stateChange = true; }
    else if (line == "EXPLORING")  { robotState = S_EXPLORING;  stateChange = true; }
    else if (line == "DOCKING")    { robotState = S_DOCKING;    stateChange = true; }
    else if (line == "ERROR")      { robotState = S_ERROR;      stateChange = true; }
    else if (line.startsWith("AUDIO:")) {
        parseAudio(line.substring(6));
    }
    else if (line.startsWith("BAT:")) {
        int pct = constrain(line.substring(4).toInt(), 0, 100);
        if (pct != batteryPct) {
            batteryPct = pct;
            if (appMode == MODE_NORMAL) dispDirty = true;
        }
    }
    else if (line.startsWith("SAFETY:")) {
        String s = line.substring(7);  // strip "SAFETY:" prefix
        s.trim();
        bool wasFault = safetyFault;
        safetyFault  = (s != "OK");
        safetyStatus = s;
        // Only redraw when the fault state actually flips — not on every message,
        // since safety_node sends this at 1 Hz even when nothing has changed.
        if (safetyFault != wasFault && appMode == MODE_NORMAL) dispDirty = true;
    }

    if (stateChange && appMode == MODE_NORMAL) dispDirty = true;
}

void readSerial() {
    while (Serial.available()) {
        char c = (char)Serial.read();
        if (c == '\n' || c == '\r') {
            if (rxBuf.length() > 0) {
                parseLine(rxBuf);
                rxBuf = "";
            }
        } else {
            rxBuf += c;
            if (rxBuf.length() > 160) rxBuf = "";  // overflow guard
        }
    }
}

// ── Encoder: rotation ─────────────────────────────────────────────────────────
void onRotate(int dir) {
    // dir: +1 = clockwise, -1 = counter-clockwise
    beep(dir > 0 ? 1600 : 1200, 15);

    switch (appMode) {
        case MODE_NORMAL:
            // Rotation in normal mode — enter settings menu
            appMode   = MODE_MENU;
            menuIdx   = 0;
            dispDirty = true;
            break;

        case MODE_MENU:
            menuIdx   = (menuIdx + dir + MENU_COUNT) % MENU_COUNT;
            dispDirty = true;
            break;

        case MODE_WIFI_SSID:
        case MODE_WIFI_PASS:
            csPos     = ((csPos + dir) % CS_TOTAL + CS_TOTAL) % CS_TOTAL;
            dispDirty = true;
            break;

        case MODE_WIFI_CONFIRM:
            menuIdx   = (menuIdx + 1) % 2;  // toggle YES / NO
            dispDirty = true;
            break;
    }
}

// ── Encoder: button press ────────────────────────────────────────────────────
void onButton() {
    beep(2000, 30);

    switch (appMode) {
        case MODE_NORMAL:
            appMode   = MODE_MENU;
            menuIdx   = 0;
            dispDirty = true;
            break;

        case MODE_MENU:
            if (menuIdx == 0) {
                // WiFi Settings — start SSID entry
                wifiSSID  = "";
                wifiPass  = "";
                csPos     = 0;
                appMode   = MODE_WIFI_SSID;
            } else {
                // Exit menu
                appMode   = MODE_NORMAL;
            }
            dispDirty = true;
            break;

        case MODE_WIFI_SSID:
            if (csPos == CS_OK) {
                if (wifiSSID.length() > 0) {
                    csPos   = 0;
                    appMode = MODE_WIFI_PASS;
                }
            } else if (csPos == CS_DEL) {
                if (wifiSSID.length() > 0)
                    wifiSSID.remove(wifiSSID.length() - 1);
            } else if (wifiSSID.length() < 32) {
                wifiSSID += CS[csPos];
            }
            dispDirty = true;
            break;

        case MODE_WIFI_PASS:
            if (csPos == CS_OK) {
                if (wifiPass.length() > 0) {
                    menuIdx = 0;
                    appMode = MODE_WIFI_CONFIRM;
                }
            } else if (csPos == CS_DEL) {
                if (wifiPass.length() > 0)
                    wifiPass.remove(wifiPass.length() - 1);
            } else if (wifiPass.length() < 64) {
                wifiPass += CS[csPos];
            }
            dispDirty = true;
            break;

        case MODE_WIFI_CONFIRM:
            if (menuIdx == 0) {
                // YES — send credentials to Pi
                Serial.print("WIFI:");
                Serial.print(wifiSSID);
                Serial.print(":");
                Serial.println(wifiPass);
                // Confirmation chirp
                beep(2500, 150);
                delay(160);
                beep(3200, 200);
            }
            appMode   = MODE_NORMAL;
            dispDirty = true;
            break;
    }
}

// ── setup ─────────────────────────────────────────────────────────────────────
void setup() {
    pinMode(PIN_ENC_A,   INPUT_PULLUP);
    pinMode(PIN_ENC_B,   INPUT_PULLUP);
    pinMode(PIN_ENC_BTN, INPUT_PULLUP);

    Serial.begin(115200);

    FastLED.addLeds<WS2812B, PIN_LED_LEFT,  GRB>(leftLEDs,  NUM_LEDS);
    FastLED.addLeds<WS2812B, PIN_LED_RIGHT, GRB>(rightLEDs, NUM_LEDS);
    FastLED.setBrightness(80);
    FastLED.clear(true);

    u8g2.begin();
    drawNormal();

    // Boot chirp — three ascending tones
    beep(1000, 80); delay(100);
    beep(1500, 80); delay(100);
    beep(2000, 120);

    lastEncA = digitalRead(PIN_ENC_A);
}

// ── loop ──────────────────────────────────────────────────────────────────────
void loop() {
    uint32_t now = millis();

    // Always drain the serial buffer first
    readSerial();

    // Encoder rotation — edge detect with 5ms debounce
    int encA = digitalRead(PIN_ENC_A);
    if (encA != lastEncA && (now - lastEncMs >= 5)) {
        onRotate(digitalRead(PIN_ENC_B) != encA ? 1 : -1);
        lastEncA  = encA;
        lastEncMs = now;
    }

    // Encoder button — 200ms debounce
    int btn = digitalRead(PIN_ENC_BTN);
    if (btn == LOW && !btnHeld && (now - lastBtnMs >= 200)) {
        btnHeld   = true;
        lastBtnMs = now;
        onButton();
    }
    if (btn == HIGH) btnHeld = false;

    // LEDs at ~20 FPS (every 50ms)
    if (now - lastLedMs >= 50) {
        updateLEDs();
        lastLedMs = now;
    }

    // Display — only on dirty flag (state change, battery change, menu nav)
    if (dispDirty) {
        updateDisplay();
    }
}
