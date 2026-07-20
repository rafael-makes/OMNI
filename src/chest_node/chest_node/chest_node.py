"""
OMNI chest_node
Bridges ESP32 chest panel via /dev/ttyAMA0 (UART0, GPIO 14/15, 115200 baud).

Pi → ESP32 protocol (ASCII, newline-terminated):
  <STATE>\n          — one of IDLE / LISTENING / SPEAKING / NAVIGATING / EXPLORING / DOCKING / ERROR
  AUDIO:v1,...,v16\n — 16 floats 0.0-1.0, EQ visualization, sent at 10 Hz
  BAT:<pct>\n        — integer 0-100, sent on >0.5% change
  PULSE:<camera>\n   — transient attention flash on the LED panels, from
                       /chest/pulse. Overlays the current state animation for
                       ~700 ms; does NOT change the displayed state.

ESP32 → Pi protocol:
  WIFI:<ssid>:<password>\n — published to /wifi_config as String("ssid:password")
"""

import threading
import time

import rclpy
from rclpy.node import Node

import serial

from sensor_msgs.msg import BatteryState
from std_msgs.msg import Float32MultiArray, String


_VALID_STATES = frozenset({
    'IDLE', 'LISTENING', 'SPEAKING',
    'NAVIGATING', 'EXPLORING', 'DOCKING', 'ERROR',
})

_AUDIO_CHANNELS = 16

# Camera names allowed through to the panel. The wire protocol is newline- and
# colon-delimited ASCII, so an unsanitised payload could inject a whole fake
# command (e.g. "x\nERROR") — an allowlist is the cheap, total fix, and the set
# of cameras is small and known. Unknown names are dropped with a warning rather
# than passed through.
_PULSE_CAMERAS = frozenset({'head', 'rear'})


class ChestNode(Node):

    def __init__(self):
        super().__init__('chest_node')

        self.declare_parameter('serial_port', '/dev/ttyAMA0')
        self.declare_parameter('baud_rate',   115200)
        self.declare_parameter('audio_hz',    10.0)
        self.declare_parameter('pulse_topic', '/chest/pulse')
        # Floor on the gap between pulses actually sent to the panel. The UART
        # already carries AUDIO at 10 Hz; a burst of presence events must not be
        # able to crowd it out. Retriggering inside this window is also visually
        # pointless — the firmware would just restart the same 700 ms animation.
        self.declare_parameter('pulse_min_interval', 0.25)

        self._port     = self.get_parameter('serial_port').value
        self._baud     = self.get_parameter('baud_rate').value
        self._audio_hz = self.get_parameter('audio_hz').value
        self._pulse_min_interval = float(
            self.get_parameter('pulse_min_interval').value)

        self._ser: serial.Serial | None = None
        self._serial_lock = threading.Lock()
        self._connect_serial()

        self._current_state: str | None = None
        self._audio_levels: list[float] = [0.0] * _AUDIO_CHANNELS
        self._audio_lock = threading.Lock()
        self._last_battery_pct: float | None = None
        self._last_safety_str: str | None = None
        self._last_pulse_time: float = 0.0

        self.create_subscription(String,            '/robot_state',    self._state_cb,   10)
        self.create_subscription(
            String, self.get_parameter('pulse_topic').value, self._pulse_cb, 10)
        self.create_subscription(Float32MultiArray, '/audio/levels',   self._audio_cb,   10)
        self.create_subscription(BatteryState, '/battery/status', self._battery_cb, 10)
        self.create_subscription(String,      '/safety/status',  self._safety_cb,  10)

        self._wifi_pub = self.create_publisher(String, '/wifi_config', 10)

        self.create_timer(1.0 / self._audio_hz, self._audio_timer_cb)

        self._running = True
        self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._read_thread.start()

        self.get_logger().info(
            f'chest_node started — {self._port} @ {self._baud}, '
            f'audio {self._audio_hz:.0f} Hz, '
            f'pulse on {self.get_parameter("pulse_topic").value} '
            f'(min interval {self._pulse_min_interval:.2f}s, '
            f'cameras {sorted(_PULSE_CAMERAS)})'
        )

    # ── Serial connection ─────────────────────────────────────────────────────

    def _connect_serial(self):
        try:
            self._ser = serial.Serial(
                self._port, self._baud,
                timeout=1.0,
                write_timeout=1.0,
            )
            time.sleep(0.5)
            self._ser.reset_input_buffer()
            self.get_logger().info(f'Serial connected: {self._port}')
            # Send IDLE immediately so the panel shows a defined state
            # before any /robot_state message arrives.
            self._send_state('IDLE')
        except serial.SerialException as e:
            self.get_logger().error(f'Serial open failed: {e}')
            self._ser = None

    def _serial_write(self, data: str) -> bool:
        if self._ser is None:
            return False
        try:
            with self._serial_lock:
                self._ser.write(data.encode('ascii'))
            return True
        except serial.SerialException as e:
            self.get_logger().warn(f'Serial write error: {e}')
            return False

    # ── /robot_state callback ─────────────────────────────────────────────────

    def _state_cb(self, msg: String):
        state = msg.data.strip().upper()
        if state not in _VALID_STATES:
            self.get_logger().warn(f'Unknown robot_state ignored: "{state}"')
            return
        if state == self._current_state:
            return
        self._send_state(state)

    def _send_state(self, state: str):
        if self._serial_write(f'{state}\n'):
            self._current_state = state
            self.get_logger().debug(f'State → ESP32: {state}')

    # ── /chest/pulse callback ─────────────────────────────────────────────────

    def _pulse_cb(self, msg: String):
        """Attention flash request from head_tracking_node.

        Payload is the source camera name ("head" / "rear"). This is a
        fire-and-forget visual cue that runs alongside whatever state the panel
        is showing — it deliberately does not touch _current_state, so a pulse
        can never leave the chest displaying the wrong thing.
        """
        camera = msg.data.strip().lower()
        if camera not in _PULSE_CAMERAS:
            self.get_logger().warn(f'Unknown pulse camera ignored: "{msg.data}"')
            return

        now = time.monotonic()
        if now - self._last_pulse_time < self._pulse_min_interval:
            return
        if self._serial_write(f'PULSE:{camera}\n'):
            self._last_pulse_time = now
            self.get_logger().debug(f'Pulse → ESP32: PULSE:{camera}')

    # ── /audio/levels callback ────────────────────────────────────────────────

    def _audio_cb(self, msg: Float32MultiArray):
        raw = list(msg.data)[:_AUDIO_CHANNELS]
        while len(raw) < _AUDIO_CHANNELS:
            raw.append(0.0)
        clamped = [max(0.0, min(1.0, v)) for v in raw]
        with self._audio_lock:
            self._audio_levels = clamped

    # ── 10 Hz audio send timer ────────────────────────────────────────────────

    def _audio_timer_cb(self):
        with self._audio_lock:
            levels = list(self._audio_levels)
        values_str = ','.join(f'{v:.4f}' for v in levels)
        self._serial_write(f'AUDIO:{values_str}\n')

    # ── /battery/status callback ──────────────────────────────────────────────

    def _battery_cb(self, msg: BatteryState):
        pct_float = max(0.0, min(100.0, msg.percentage * 100.0))
        if self._last_battery_pct is not None:
            if abs(pct_float - self._last_battery_pct) <= 0.5:
                return
        pct_int = round(pct_float)
        if self._serial_write(f'BAT:{pct_int}\n'):
            self._last_battery_pct = pct_float
            self.get_logger().debug(f'Battery → ESP32: BAT:{pct_int}')

    # ── /safety/status callback ───────────────────────────────────────────────

    def _safety_cb(self, msg: String):
        status = msg.data.strip()

        # Build the short string to send to the ESP32.
        # /safety/status format:
        #   "OK | 12.3V | tilt 1.2°"
        #   "FAULTED [watchdog, estop] | 12.3V | tilt 1.2°"
        if status.startswith('OK'):
            out = 'SAFETY:OK'
        elif status.startswith('FAULTED'):
            # Extract the fault names from inside the square brackets
            try:
                faults = status[status.index('[') + 1 : status.index(']')]
                out = f'SAFETY:{faults}'
            except ValueError:
                # Brackets missing — send the raw FAULTED word as a fallback
                out = 'SAFETY:FAULTED'
        else:
            return  # Unrecognised format — ignore

        # Only send when the string actually changes to avoid flooding the UART
        if out == self._last_safety_str:
            return
        if self._serial_write(f'{out}\n'):
            self._last_safety_str = out
            self.get_logger().debug(f'Safety → ESP32: {out}')

    # ── UART read loop (background thread) ───────────────────────────────────

    def _read_loop(self):
        while self._running:
            if self._ser is None:
                time.sleep(1.0)
                self._connect_serial()
                continue
            try:
                line = self._ser.readline().decode('ascii', errors='ignore').strip()
            except serial.SerialException as e:
                self.get_logger().warn(f'Serial read error: {e}')
                self._ser = None
                continue
            if not line:
                continue
            self._parse_esp32_line(line)

    # ── ESP32 → Pi message parser ─────────────────────────────────────────────

    def _parse_esp32_line(self, line: str):
        if line.startswith('WIFI:'):
            payload = line[len('WIFI:'):]
            if ':' not in payload:
                self.get_logger().warn(f'Malformed WIFI message: "{line}"')
                return
            self.get_logger().info('WiFi config received from ESP32')
            self._wifi_pub.publish(String(data=payload))
        else:
            self.get_logger().debug(f'Unrecognised ESP32 message: "{line}"')

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def destroy_node(self):
        self._running = False
        if self._ser:
            try:
                self._ser.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ChestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
