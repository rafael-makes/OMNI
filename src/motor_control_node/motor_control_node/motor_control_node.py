"""
OMNI motor_control_node
Bridges /cmd_vel ↔ Arduino serial and republishes odometry.

Serial protocol (set by Arduino firmware):
  Pi → Arduino : "C,<left_mps>,<right_mps>\\n"
  Pi → Arduino : "R\\n"  (reset odometry)
  Arduino → Pi : "O,x,y,theta,vx,vtheta,left_mps,right_mps\\n"  @ 20 Hz
"""

import math
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import serial

from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from tf2_ros import TransformBroadcaster


class MotorControlNode(Node):

    def __init__(self):
        super().__init__('motor_control_node')

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter('serial_port',     '/dev/arduino')
        self.declare_parameter('baud_rate',       115200)
        self.declare_parameter('wheel_separation', 0.305)
        self.declare_parameter('wheel_radius',     0.050)
        self.declare_parameter('cmd_vel_timeout',  0.5)
        self.declare_parameter('odom_frame',         'odom')
        self.declare_parameter('base_frame',         'base_link')
        # Speed limits — clamp cmd_vel before kinematics so a runaway nav stack
        # can't send commands the hardware can't track (max hardware ≈ 1.0 m/s).
        # Conservative defaults for first-drive; raise in the launch file once tuned.
        self.declare_parameter('max_linear_speed',   0.5)   # m/s
        self.declare_parameter('max_angular_speed',  2.0)   # rad/s — 10kg motors, MIN_PWM=35, floor ~0.75 rad/s
        # publish_tf: set false when robot_localization EKF takes over odom→base_link
        self.declare_parameter('publish_tf', True)

        self._port    = self.get_parameter('serial_port').value
        self._baud    = self.get_parameter('baud_rate').value
        self._sep     = self.get_parameter('wheel_separation').value
        self._radius  = self.get_parameter('wheel_radius').value
        self._timeout = self.get_parameter('cmd_vel_timeout').value
        self._odom_frame  = self.get_parameter('odom_frame').value
        self._base_frame  = self.get_parameter('base_frame').value
        self._publish_tf  = self.get_parameter('publish_tf').value
        self._max_vx  = self.get_parameter('max_linear_speed').value
        self._max_wz  = self.get_parameter('max_angular_speed').value

        # ── Serial ───────────────────────────────────────────────────────────
        self._ser: serial.Serial | None = None
        self._serial_lock = threading.Lock()
        self._connect_serial()

        # ── Publishers / subscribers ─────────────────────────────────────────
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self._odom_pub   = self.create_publisher(Odometry, '/odom',                 sensor_qos)
        self._status_pub = self.create_publisher(String,   '/motor_status',         10)
        self._stall_pub  = self.create_publisher(Bool,     '/motor/stall_detected', 10)
        self.create_service(Trigger, '/motor/reset_odometry', self._reset_odom_cb)
        self._tf_broadcaster = TransformBroadcaster(self)

        self.create_subscription(Twist, '/cmd_vel', self._cmd_vel_cb, 10)

        # ── State ─────────────────────────────────────────────────────────────
        self._last_cmd_time  = time.monotonic()
        self._last_left_mps  = 0.0
        self._last_right_mps = 0.0

        # ── Background serial reader ──────────────────────────────────────────
        self._running = True
        self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._read_thread.start()

        # 20 Hz serial heartbeat — always re-sends the last command so the Pico
        # never sees a gap even when /cmd_vel jitters under CPU load.
        # This replaces the old on-receipt-only write and prevents the 500ms
        # Pico watchdog from firing and causing stutters / hard-turn restarts.
        self.create_timer(0.05, self._serial_heartbeat_cb)

        self.get_logger().info(
            f'motor_control_node started — {self._port} @ {self._baud}'
        )

    # ── Serial connection ─────────────────────────────────────────────────────

    def _connect_serial(self):
        try:
            self._ser = serial.Serial(
                self._port, self._baud,
                timeout=1.0,
                write_timeout=1.0,
            )
            # Flush any leftover Arduino boot messages
            time.sleep(2.0)
            self._ser.reset_input_buffer()
            self.get_logger().info(f'Serial connected: {self._port}')
        except serial.SerialException as e:
            self.get_logger().error(f'Serial open failed: {e}')
            self._ser = None

    def _serial_write(self, data: str):
        if self._ser is None:
            return
        try:
            with self._serial_lock:
                self._ser.write(data.encode())
        except serial.SerialException as e:
            self.get_logger().warn(f'Serial write error: {e}')

    # ── /cmd_vel callback ─────────────────────────────────────────────────────

    def _cmd_vel_cb(self, msg: Twist):
        vx = max(-self._max_vx, min(self._max_vx, msg.linear.x))
        wz = max(-self._max_wz, min(self._max_wz, msg.angular.z))
        # Store — the 20 Hz heartbeat timer sends these to serial
        self._last_left_mps  = vx - wz * self._sep / 2.0
        self._last_right_mps = vx + wz * self._sep / 2.0
        self._last_cmd_time  = time.monotonic()

    # ── 20 Hz serial heartbeat ────────────────────────────────────────────────
    # Sends the last known command every 50 ms regardless of /cmd_vel timing.
    # Keeps the Pico's 500 ms watchdog fed under CPU load (SLAM, etc.) and
    # prevents the stutter + hard-turn-on-restart that occurred during mapping.

    def _serial_heartbeat_cb(self):
        if time.monotonic() - self._last_cmd_time > self._timeout:
            self._serial_write('C,0.0000,0.0000\n')
        else:
            self._serial_write(
                f'C,{self._last_left_mps:.4f},{self._last_right_mps:.4f}\n'
            )

    def _reset_odom_cb(self, _req, response):
        self._serial_write('R\n')
        response.success = True
        response.message = 'Odometry reset sent to Arduino'
        return response

    # ── Serial read loop (background thread) ──────────────────────────────────

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

            if line.startswith('O,'):
                self._parse_odom(line)
            elif line.startswith('E:'):
                self._parse_encoder(line)

    # ── Odometry parser ───────────────────────────────────────────────────────

    def _parse_odom(self, line: str):
        parts = line.split(',')
        # Accept 8 fields (legacy, no stall) or 9 fields (with stall flag)
        if len(parts) not in (8, 9):
            return

        try:
            x, y, th, vx, vth, lv, rv = (float(p) for p in parts[1:8])
            stall = int(parts[8]) if len(parts) == 9 else 0
        except ValueError:
            return

        # Publish stall if detected
        if stall:
            self._stall_pub.publish(Bool(data=True))

        now = self.get_clock().now().to_msg()

        # TF: odom → base_link (disabled when robot_localization EKF is running)
        if self._publish_tf:
            tf = TransformStamped()
            tf.header.stamp = now
            tf.header.frame_id = self._odom_frame
            tf.child_frame_id  = self._base_frame
            tf.transform.translation.x = x
            tf.transform.translation.y = y
            tf.transform.translation.z = 0.0
            tf.transform.rotation.z = math.sin(th / 2.0)
            tf.transform.rotation.w = math.cos(th / 2.0)
            self._tf_broadcaster.sendTransform(tf)

        # Odometry message
        odom = Odometry()
        odom.header.stamp    = now
        odom.header.frame_id = self._odom_frame
        odom.child_frame_id  = self._base_frame

        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.orientation.z = math.sin(th / 2.0)
        odom.pose.pose.orientation.w = math.cos(th / 2.0)

        odom.twist.twist.linear.x  = vx
        odom.twist.twist.angular.z = vth

        # Diagonal covariance — loosen these once you have real encoder data
        odom.pose.covariance[0]  = 0.01   # x
        odom.pose.covariance[7]  = 0.01   # y
        odom.pose.covariance[35] = 0.05   # yaw
        odom.twist.covariance[0]  = 0.01  # vx
        odom.twist.covariance[35] = 0.05  # vyaw

        self._odom_pub.publish(odom)

        # Motor status (JSON for easy debugging)
        status = (
            f'{{"x":{x:.3f},"y":{y:.3f},"th":{th:.3f},'
            f'"vx":{vx:.3f},"vth":{vth:.3f},'
            f'"left_mps":{lv:.3f},"right_mps":{rv:.3f}}}'
        )
        self._status_pub.publish(String(data=status))

    # ── Encoder / stall parser ────────────────────────────────────────────────

    def _parse_encoder(self, line: str):
        # Arduino sends "E:left_ticks,right_ticks,stall_flag\n"
        # stall_flag is 0 (normal) or 1 (motor stalled / overcurrent detected)
        try:
            # Strip the "E:" prefix, then split into the three comma-separated fields
            payload = line[2:]           # everything after "E:"
            parts   = payload.split(',')
            if len(parts) != 3:
                self.get_logger().warn(f'_parse_encoder: expected 3 fields, got: {line}')
                return
            # Third field is the stall flag; ticks are available here if needed later
            stall_flag = bool(int(parts[2]))
            self._stall_pub.publish(Bool(data=stall_flag))
        except (ValueError, IndexError) as e:
            self.get_logger().warn(f'_parse_encoder bad data "{line}": {e}')

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def destroy_node(self):
        self._running = False
        self._serial_write('C,0.0000,0.0000\n')
        if self._ser:
            self._ser.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MotorControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
