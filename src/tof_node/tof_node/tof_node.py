import json
import math
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import Range
from std_msgs.msg import String

try:
    import smbus2
    _SMBUS_AVAILABLE = True
    _SMBUS_ERROR = ''
except ImportError as e:
    _SMBUS_AVAILABLE = False
    _SMBUS_ERROR = str(e)

try:
    import board
    import busio
    from adafruit_vl53l0x import VL53L0X
    _ADAFRUIT_AVAILABLE = True
    _ADAFRUIT_ERROR = ''
except ImportError as e:
    _ADAFRUIT_AVAILABLE = False
    _ADAFRUIT_ERROR = str(e)

_LIBS_AVAILABLE = _SMBUS_AVAILABLE and _ADAFRUIT_AVAILABLE

# PCA9548A mux address, VL53L0X sensor address, I2C bus index
_MUX_ADDR    = 0x70
_SENSOR_ADDR = 0x29
_I2C_BUS     = 1

# VL53L0X measurement timing budget in microseconds (20ms = good speed/accuracy balance)
_TIMING_BUDGET_US = 20_000


class TofNode(Node):

    # (channel, position_name, topic, frame_id)
    _CHANNELS = [
        (0, 'left',        '/tof/left',        'tof_left'),
        (1, 'front_left',  '/tof/front_left',  'tof_front_left'),
        (2, 'front_right', '/tof/front_right', 'tof_front_right'),
        (3, 'right',       '/tof/right',       'tof_right'),
        (4, 'left_rear',   '/tof/left_rear',   'tof_left_rear'),
        (5, 'right_rear',  '/tof/right_rear',  'tof_right_rear'),
    ]

    _FOV_RAD   = math.radians(25.0)
    _MIN_RANGE = 0.03   # metres
    _MAX_RANGE = 1.20   # metres

    def __init__(self):
        super().__init__('tof_node')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('frame_prefix',    'tof')

        self._rate = self.get_parameter('publish_rate_hz').value

        # ── State ────────────────────────────────────────────────────────────
        self._sensors: dict  = {}          # ch → VL53L0X
        self._i2c           = None
        self._smbus         = None
        self._connected     = False
        self._last_connect_attempt = 0.0

        self._readings: dict      = {}     # ch → range_mm (int)
        self._per_ch_errors: dict = {ch: 0 for ch, *_ in self._CHANNELS}
        self._lock       = threading.Lock()
        self._stop_event = threading.Event()

        # ── Publishers ───────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._pubs = {}
        for ch, name, topic, _ in self._CHANNELS:
            self._pubs[ch] = self.create_publisher(Range, topic, sensor_qos)

        self._status_pub = self.create_publisher(String, '/tof/status', 10)

        # ── Library check ────────────────────────────────────────────────────
        if not _LIBS_AVAILABLE:
            missing = []
            if not _SMBUS_AVAILABLE:
                missing.append(f'smbus2: {_SMBUS_ERROR}')
            if not _ADAFRUIT_AVAILABLE:
                missing.append(f'adafruit_vl53l0x: {_ADAFRUIT_ERROR}')
            self.get_logger().error('Required libraries missing — ' + '; '.join(missing))

        # ── Initial connection ────────────────────────────────────────────────
        self._connect_sensors()

        # ── Background poll thread ────────────────────────────────────────────
        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True, name='tof_poll'
        )
        self._poll_thread.start()

        # ── Timers ───────────────────────────────────────────────────────────
        self.create_timer(1.0 / self._rate, self._publish_cb)
        self.create_timer(1.0,              self._reconnect_cb)

        self.get_logger().info(
            f'tof_node started — 6x VL53L0X via PCA9548A, '
            f'publish {self._rate:.0f} Hz, timing budget {_TIMING_BUDGET_US // 1000} ms'
        )

    # ── Mux control ──────────────────────────────────────────────────────────

    def _mux_select(self, channel: int) -> None:
        self._smbus.write_byte(_MUX_ADDR, 1 << channel)
        time.sleep(0.001)   # 1ms settling required after PCA9548A channel switch

    def _mux_close(self) -> None:
        try:
            self._smbus.write_byte(_MUX_ADDR, 0x00)
        except Exception:
            pass

    # ── Connection ───────────────────────────────────────────────────────────

    def _connect_sensors(self) -> None:
        if not _LIBS_AVAILABLE:
            return
        self._last_connect_attempt = time.monotonic()
        try:
            # Open smbus for mux control
            if self._smbus is None:
                self._smbus = smbus2.SMBus(_I2C_BUS)

            # Open busio I2C for sensor reads
            if self._i2c is None:
                self._i2c = busio.I2C(board.SCL, board.SDA)

            # Close all mux channels first to start clean
            self._smbus.write_byte(_MUX_ADDR, 0x00)
            time.sleep(0.005)

            sensors = {}
            failed = []
            for ch, name, _, _ in self._CHANNELS:
                self._mux_select(ch)
                try:
                    s = VL53L0X(self._i2c, address=_SENSOR_ADDR)
                    s.measurement_timing_budget = _TIMING_BUDGET_US
                    sensors[ch] = s
                    self.get_logger().info(f'  CH{ch} ({name}) — OK')
                except Exception as e:
                    failed.append((ch, name))
                    self.get_logger().warn(f'  CH{ch} ({name}) — NOT FOUND: {e}')
                finally:
                    self._mux_close()
                    time.sleep(0.005)

            if not sensors:
                raise RuntimeError('No VL53L0X sensors found on any channel')

            self._sensors = sensors
            with self._lock:
                self._per_ch_errors = {ch: 0 for ch, *_ in self._CHANNELS}
                self._readings = {}
            self._connected = True
            ok_count = len(sensors)
            if failed:
                missing = ', '.join(f'CH{c}({n})' for c, n in failed)
                self.get_logger().warn(
                    f'{ok_count}/6 sensors connected. Missing: {missing} — check wiring.'
                )
            else:
                self.get_logger().info('All 6 VL53L0X sensors connected.')

        except Exception as e:
            self._connected = False
            self._mux_close()
            # Force full re-init on next attempt
            try:
                if self._i2c is not None:
                    self._i2c.deinit()
            except Exception:
                pass
            self._i2c = None
            try:
                if self._smbus is not None:
                    self._smbus.close()
            except Exception:
                pass
            self._smbus = None
            self.get_logger().error(f'TOF sensor init failed: {e}')

    def _reconnect_cb(self) -> None:
        if not self._connected:
            if time.monotonic() - self._last_connect_attempt < 3.0:
                return
            self.get_logger().info('Attempting TOF sensor reconnect...')
            self._connect_sensors()

    # ── Background poll loop ──────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            if not self._connected:
                time.sleep(0.1)
                continue

            for ch, name, _, _ in self._CHANNELS:
                if self._stop_event.is_set():
                    break
                if ch not in self._sensors:
                    continue   # sensor absent on this channel — skip silently
                try:
                    self._mux_select(ch)
                    mm = self._sensors[ch].range   # blocking ~20ms
                    self._mux_close()
                    with self._lock:
                        self._readings[ch] = mm
                        self._per_ch_errors[ch] = 0
                except Exception as e:
                    self._mux_close()
                    with self._lock:
                        self._per_ch_errors[ch] += 1
                        errors = self._per_ch_errors[ch]
                    if errors <= 3:
                        self.get_logger().warn(f'CH{ch} ({name}) read error ({errors}): {e}')
                    if errors >= 5:
                        self.get_logger().error(
                            f'CH{ch} ({name}) failed 5 times — marking disconnected'
                        )
                        self._connected = False
                        break

    # ── Publish ──────────────────────────────────────────────────────────────

    def _publish_cb(self) -> None:
        now = self.get_clock().now().to_msg()

        with self._lock:
            readings = dict(self._readings)
            connected = self._connected

        ranges_m = {}
        for ch, name, topic, frame_id in self._CHANNELS:
            msg = Range()
            msg.header.stamp    = now
            msg.header.frame_id = frame_id
            msg.radiation_type  = Range.INFRARED
            msg.field_of_view   = self._FOV_RAD
            msg.min_range       = self._MIN_RANGE
            msg.max_range       = self._MAX_RANGE

            mm = readings.get(ch)
            if mm is None or not connected:
                msg.range = self._MAX_RANGE
            else:
                msg.range = max(self._MIN_RANGE, min(self._MAX_RANGE, mm / 1000.0))

            self._pubs[ch].publish(msg)
            ranges_m[name] = round(msg.range, 3)

        # Status at ~1 Hz
        if now.nanosec < 10_000_000:
            status = json.dumps({'connected': connected, 'ranges_m': ranges_m})
            self._status_pub.publish(String(data=status))

    # ── Cleanup ──────────────────────────────────────────────────────────────

    def destroy_node(self) -> None:
        self._stop_event.set()
        self._poll_thread.join(timeout=2.0)
        self._mux_close()
        try:
            if self._smbus is not None:
                self._smbus.close()
        except Exception:
            pass
        try:
            if self._i2c is not None:
                self._i2c.deinit()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TofNode()
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
