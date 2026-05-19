import json
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import Imu
from std_msgs.msg import String

try:
    import board
    import busio
    from adafruit_bno08x.i2c import BNO08X_I2C
    from adafruit_bno08x import (
        BNO_REPORT_GAME_ROTATION_VECTOR,
        BNO_REPORT_GYROSCOPE,
        BNO_REPORT_ACCELEROMETER,
    )
    _ADAFRUIT_AVAILABLE = True
    _ADAFRUIT_IMPORT_ERROR = ''
except ImportError as e:
    _ADAFRUIT_AVAILABLE = False
    _ADAFRUIT_IMPORT_ERROR = str(e)


class ImuNode(Node):

    # Covariance diagonals — BNO085 datasheet-derived, stored row-major 3x3
    _ORI_COV   = [0.00122, 0, 0,  0, 0.00122, 0,  0, 0, 0.00762]  # 2° r/p, 5° yaw
    _GYRO_COV  = [6e-6,    0, 0,  0, 6e-6,    0,  0, 0, 6e-6   ]  # 0.014°/s/√Hz
    _ACCEL_COV = [0.01,    0, 0,  0, 0.01,    0,  0, 0, 0.01   ]  # 0.1 m/s²

    def __init__(self):
        super().__init__('imu_node')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('i2c_address',     0x4A)
        self.declare_parameter('publish_rate_hz', 25.0)
        self.declare_parameter('frame_id',        'imu_link')
        # mount_rotation: Z-axis clockwise degrees (0/90/180/270)
        # 0   = STEMMA QT connector faces rear  → X forward (default)
        # 180 = STEMMA QT connector faces front → flip X and Y
        self.declare_parameter('mount_rotation',  0)

        self._i2c_addr  = self.get_parameter('i2c_address').value
        self._rate      = self.get_parameter('publish_rate_hz').value
        self._frame_id  = self.get_parameter('frame_id').value
        self._mount_rot = self.get_parameter('mount_rotation').value

        self._mount_quat = self._z_rotation_quat(self._mount_rot)

        # ── State ────────────────────────────────────────────────────────────
        self._bno: 'BNO08X_I2C | None' = None
        self._i2c = None
        self._connected = False
        self._consecutive_errors = 0
        self._last_connect_attempt = 0.0  # monotonic time of last _connect_imu() call

        # ── Publishers ───────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._imu_pub    = self.create_publisher(Imu,    '/imu/data',   sensor_qos)
        self._status_pub = self.create_publisher(String, '/imu/status', 10)

        if not _ADAFRUIT_AVAILABLE:
            self.get_logger().error(
                f'Adafruit libraries unavailable: {_ADAFRUIT_IMPORT_ERROR}'
            )

        # ── Initial connection ────────────────────────────────────────────────
        self._connect_imu()

        # ── Timers ───────────────────────────────────────────────────────────
        self.create_timer(1.0 / self._rate, self._publish_cb)   # 100 Hz publish
        self.create_timer(1.0,              self._reconnect_cb)  # 1 Hz reconnect

        self.get_logger().info(
            f'imu_node started — I2C 0x{self._i2c_addr:02X}, '
            f'{self._rate:.0f} Hz, frame={self._frame_id}, '
            f'mount_rotation={self._mount_rot}°'
        )

    # ── Connection ───────────────────────────────────────────────────────────

    def _connect_imu(self):
        if not _ADAFRUIT_AVAILABLE:
            return
        self._last_connect_attempt = time.monotonic()
        try:
            if self._i2c is None:
                self._i2c = busio.I2C(board.SCL, board.SDA)
            self._bno = BNO08X_I2C(self._i2c, address=self._i2c_addr)
            interval_us = int(1_000_000 / self._rate)
            self._bno.enable_feature(BNO_REPORT_GAME_ROTATION_VECTOR, interval_us)
            self._bno.enable_feature(BNO_REPORT_GYROSCOPE,            interval_us)
            self._bno.enable_feature(BNO_REPORT_ACCELEROMETER,        interval_us)
            # Drain any leftover batch packets from a previous session before
            # opening up to the publish timer. The BNO085 survives soft resets
            # with its report queue intact; without this drain, the first publish
            # tick hits stale "Unprocessable Batch bytes" and triggers a reconnect loop.
            time.sleep(0.3)
            for _ in range(10):
                try:
                    self._bno._process_available_packets()
                except Exception:
                    pass
            self._connected = True
            self._consecutive_errors = 0
            self.get_logger().info(
                f'BNO085 connected at 0x{self._i2c_addr:02X}, '
                f'interval={interval_us} µs ({self._rate:.0f} Hz)'
            )
        except Exception as e:
            self._bno = None
            self._connected = False
            # Force full I2C re-init if the bus itself failed
            try:
                self._i2c.deinit()
            except Exception:
                pass
            self._i2c = None
            self.get_logger().error(f'BNO085 connection failed: {e}')

    def _reconnect_cb(self):
        if not self._connected:
            # Rate-limit reconnects: BNO08X_I2C() blocks for ~1s on timeout,
            # causing queued timer ticks to fire back-to-back and hammer the bus.
            if time.monotonic() - self._last_connect_attempt < 3.0:
                return
            self.get_logger().info('Attempting BNO085 reconnect...')
            self._connect_imu()

    # ── Publish ──────────────────────────────────────────────────────────────

    def _publish_cb(self):
        if not self._connected or self._bno is None:
            return

        now = self.get_clock().now().to_msg()

        try:
            # Single poll per tick: drain the SHTP buffer once, then read cached values.
            # At 100kHz I2C each packet read costs ~9ms; with 3 reports enabled,
            # one poll drains all 3 packets (~27ms). This limits throughput to ~25 Hz
            # at 100kHz. After rebooting with i2c_arm_baudrate=400000, raise
            # publish_rate_hz to 100.0 via --ros-args -p publish_rate_hz:=100.0
            self._bno._process_available_packets()
            readings = self._bno._readings
            quat  = readings.get(BNO_REPORT_GAME_ROTATION_VECTOR)
            gyro  = readings.get(BNO_REPORT_GYROSCOPE)
            accel = readings.get(BNO_REPORT_ACCELEROMETER)
        except (OSError, RuntimeError) as e:
            err_str = str(e)
            # "Unprocessable Batch bytes" is always transient (old report queue
            # flushing through after a reconnect). Log it but don't count toward
            # the disconnect threshold — it self-clears within 1-2 ticks.
            if 'Unprocessable Batch bytes' in err_str:
                self.get_logger().warn(f'IMU transient batch error (skipping): {e}')
                return
            self._consecutive_errors += 1
            if self._consecutive_errors <= 3:
                self.get_logger().warn(
                    f'IMU read error ({self._consecutive_errors}): {e}'
                )
            if self._consecutive_errors >= 5:
                self.get_logger().error(
                    'Too many IMU errors — marking disconnected'
                )
                self._bno = None
                self._connected = False
                self._consecutive_errors = 0
            return
        except Exception as e:
            self.get_logger().error(f'Unexpected IMU error: {e}')
            self._bno = None
            self._connected = False
            return

        self._consecutive_errors = 0

        # None is normal for up to ~0.5s while sensor warms up
        if quat is None or gyro is None or accel is None:
            return

        # Apply mount rotation to all three outputs
        qx, qy, qz, qw = self._apply_mount_rotation(quat)
        gx, gy, gz      = self._apply_mount_rotation_vec(gyro)
        ax, ay, az      = self._apply_mount_rotation_vec(accel)

        msg = Imu()
        msg.header.stamp    = now
        msg.header.frame_id = self._frame_id

        msg.orientation.x = qx
        msg.orientation.y = qy
        msg.orientation.z = qz
        msg.orientation.w = qw
        msg.orientation_covariance = self._ORI_COV

        msg.angular_velocity.x = gx
        msg.angular_velocity.y = gy
        msg.angular_velocity.z = gz
        msg.angular_velocity_covariance = self._GYRO_COV

        msg.linear_acceleration.x = ax
        msg.linear_acceleration.y = ay
        msg.linear_acceleration.z = az
        msg.linear_acceleration_covariance = self._ACCEL_COV

        self._imu_pub.publish(msg)

        # Publish status at ~1 Hz: fire during first 10ms of each second
        if now.nanosec < 10_000_000:
            status = json.dumps({
                'connected': True,
                'quat': [round(v, 4) for v in (qx, qy, qz, qw)],
                'gyro_z_rads': round(gz, 4),
                'accel_z_ms2': round(az, 4),
            })
            self._status_pub.publish(String(data=status))

    # ── Mount rotation utilities ──────────────────────────────────────────────

    @staticmethod
    def _z_rotation_quat(degrees: int) -> tuple:
        rad = math.radians(degrees)
        return (0.0, 0.0, math.sin(rad / 2.0), math.cos(rad / 2.0))

    def _apply_mount_rotation(self, sensor_quat: tuple) -> tuple:
        """Hamilton product: mount_quat * sensor_quat."""
        if self._mount_rot == 0:
            return sensor_quat
        ax, ay, az, aw = self._mount_quat
        bx, by, bz, bw = sensor_quat
        return (
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
            aw*bw - ax*bx - ay*by - az*bz,
        )

    def _apply_mount_rotation_vec(self, vec: tuple) -> tuple:
        """Rotate (x, y) by mount_rotation around Z; z is unchanged."""
        if self._mount_rot == 0:
            return vec
        rad = math.radians(self._mount_rot)
        c, s = math.cos(rad), math.sin(rad)
        x, y, z = vec
        return (c*x - s*y, s*x + c*y, z)

    # ── Cleanup ──────────────────────────────────────────────────────────────

    def destroy_node(self):
        try:
            if self._i2c is not None:
                self._i2c.deinit()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ImuNode()
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
