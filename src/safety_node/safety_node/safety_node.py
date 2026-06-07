"""
OMNI safety_node
================
Acts as a command gateway between nav_node and motor_control_node.

The idea: nav_node publishes /cmd_vel_raw (what the robot WANTS to do).
safety_node subscribes to that plus every sensor that can indicate danger.
If everything is safe, it passes the command through to /cmd_vel unchanged.
If ANY fault is active, it publishes Twist() zeros to /cmd_vel instead.

WHY actively publish zeros instead of just going silent?
  motor_control_node has its own watchdog, but any gap between a real command
  and the watchdog firing could leave the motors running. Actively commanding
  zero is the belt-and-suspenders approach — always explicit, never ambiguous.

Topic flow:
  nav_node → /cmd_vel_raw → [safety_node] → /cmd_vel → motor_control_node
"""

import math
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu, Range
from std_msgs.msg import Bool, Float32, String


class SafetyNode(Node):

    def __init__(self):
        super().__init__('safety_node')

        # ── Parameters ────────────────────────────────────────────────────────
        # All tunable thresholds live here — never hardcoded below.
        # Override at launch with: --ros-args -p max_tilt_degrees:=25.0
        self.declare_parameter('max_tilt_degrees',     30.0)
        self.declare_parameter('critical_voltage',     10.5)  # 3S LiPo absolute floor (3.5V/cell)
        self.declare_parameter('warning_voltage',      11.1)  # 3S LiPo low warning (3.7V/cell)
        self.declare_parameter('watchdog_timeout_sec',  0.5)  # silence on /cmd_vel_raw → fault
        self.declare_parameter('min_proximity_m',       0.15) # 15 cm proximity stop

        self._max_tilt      = self.get_parameter('max_tilt_degrees').value
        self._crit_voltage  = self.get_parameter('critical_voltage').value
        self._warn_voltage  = self.get_parameter('warning_voltage').value
        self._watchdog_sec  = self.get_parameter('watchdog_timeout_sec').value
        self._min_proximity = self.get_parameter('min_proximity_m').value

        # ── Fault state ───────────────────────────────────────────────────────
        # Each fault is its own named bool — easy to read, easy to log, no
        # dict gymnastics when you want to check "is anything faulted?"
        #
        # AUTO-CLEARING faults resolve as soon as the condition goes away:
        self.fault_tilt      = False  # IMU tilt exceeded max_tilt_degrees
        self.fault_voltage   = False  # Battery below critical_voltage
        self.fault_watchdog  = False  # No /cmd_vel_raw received recently
        self.fault_proximity = False  # TOF sensor too close to an obstacle
        #
        # LATCHING faults stay active until /safety/clear_fault is published:
        self.fault_stall     = False  # Motor stall detected (hardware issue possible)
        self.fault_estop     = False  # E-stop button pressed (human intervention)

        # ── Shared state updated by subscriber callbacks ───────────────────────
        self._latest_cmd_vel_raw = Twist()          # last velocity command from nav
        self._last_cmd_time      = time.monotonic() # timestamp of last /cmd_vel_raw msg
        self._current_voltage    = 0.0              # latest BMS pack voltage
        self._current_tilt_deg   = 0.0              # latest tilt in degrees

        # ── Publishers ────────────────────────────────────────────────────────
        self._cmd_vel_pub = self.create_publisher(Twist,  '/cmd_vel',       10)
        self._status_pub  = self.create_publisher(String, '/safety/status', 10)
        self._fault_pub   = self.create_publisher(String, '/safety/fault',  10)

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(Twist,   '/cmd_vel_raw',          self._cmd_vel_raw_cb,  10)
        self.create_subscription(Imu,     '/imu/data',             self._imu_cb,          10)
        self.create_subscription(Float32, '/bms/battery_voltage',  self._voltage_cb,      10)
        self.create_subscription(Bool,    '/motor/stall_detected', self._stall_cb,        10)
        self.create_subscription(Bool,    '/estop/pressed',        self._estop_cb,        10)
        self.create_subscription(Range,   '/tof/proximity',        self._proximity_cb,    10)
        self.create_subscription(String,  '/safety/clear_fault',   self._clear_fault_cb,  10)

        # ── Timers ────────────────────────────────────────────────────────────
        # Safety checks at 20 Hz — fast enough to catch transient faults
        self.create_timer(0.05, self._safety_check_cb)
        # Status summary at 1 Hz — readable in the terminal without flooding it
        self.create_timer(1.0,  self._status_cb)

        self.get_logger().info('safety_node started')

    # ── Subscriber callbacks ──────────────────────────────────────────────────

    def _cmd_vel_raw_cb(self, msg: Twist):
        # Store the latest command and record when we received it.
        # The watchdog timer checks this timestamp to detect nav_node silence.
        self._latest_cmd_vel_raw = msg
        self._last_cmd_time = time.monotonic()

    def _imu_cb(self, msg: Imu):
        # Extract tilt angle from the IMU orientation quaternion.
        # We use pure Python math — no external libraries needed.
        try:
            x = msg.orientation.x
            y = msg.orientation.y
            z = msg.orientation.z
            w = msg.orientation.w

            # Convert quaternion to roll and pitch using the ZYX Euler decomposition.
            #
            # A quaternion (x, y, z, w) encodes a 3D rotation. To get the tilt of the
            # robot's floor plane we need roll (lean left/right) and pitch (lean forward/back).
            #
            # These formulas come from the standard quaternion-to-Euler conversion:
            #   roll  = atan2( 2*(w*x + y*z),  1 - 2*(x² + y²) )
            #   pitch = asin(  2*(w*y - z*x) )
            #
            # atan2 gives a full ±180° result. asin gives ±90°.
            # We clamp the asin argument to [-1, 1] to avoid domain errors from
            # floating-point noise in the quaternion.

            sinr_cosp = 2.0 * (w * x + y * z)
            cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
            roll = math.atan2(sinr_cosp, cosr_cosp)

            sinp = 2.0 * (w * y - z * x)
            sinp = max(-1.0, min(1.0, sinp))   # clamp for numerical safety
            pitch = math.asin(sinp)

            # Use whichever axis has the larger tilt — this catches any direction of lean
            self._current_tilt_deg = math.degrees(max(abs(roll), abs(pitch)))

        except Exception as e:
            self.get_logger().warn(f'IMU callback error: {e}')

    def _voltage_cb(self, msg: Float32):
        try:
            self._current_voltage = msg.data
        except Exception as e:
            self.get_logger().warn(f'Voltage callback error: {e}')

    def _stall_cb(self, msg: Bool):
        # Stall is a LATCHING fault — once detected, a human must clear it.
        # Reason: a stall often means something is physically jammed or broken.
        # We want a human to inspect before the robot moves again.
        try:
            if msg.data and not self.fault_stall:
                self.fault_stall = True
                self._publish_fault('fault_stall: motor stall detected — send "stall" or "all" to /safety/clear_fault to reset')
        except Exception as e:
            self.get_logger().warn(f'Stall callback error: {e}')

    def _estop_cb(self, msg: Bool):
        # E-stop is also LATCHING — pressing the physical button means a human
        # wanted the robot to stop. It should stay stopped until explicitly cleared.
        try:
            if msg.data and not self.fault_estop:
                self.fault_estop = True
                self._publish_fault('fault_estop: e-stop pressed — send "estop" or "all" to /safety/clear_fault to reset')
        except Exception as e:
            self.get_logger().warn(f'E-stop callback error: {e}')

    def _proximity_cb(self, msg: Range):
        # Multiple TOF sensors publish on the same topic. Each callback fires
        # independently. If ANY sensor sees an obstacle within min_proximity_m
        # the fault activates. It auto-clears when all sensors report clear again.
        try:
            if msg.range < self._min_proximity:
                if not self.fault_proximity:
                    self.fault_proximity = True
                    self._publish_fault(f'fault_proximity: TOF range {msg.range:.3f}m < {self._min_proximity}m')
            else:
                # Auto-clear: obstacle no longer detected by this sensor.
                # NOTE: if multiple sensors are faulted, one clearing won't un-fault
                # if another is still below threshold — that's acceptable because
                # the next callback from the still-close sensor will re-set it.
                self.fault_proximity = False
        except Exception as e:
            self.get_logger().warn(f'Proximity callback error: {e}')

    def _clear_fault_cb(self, msg: String):
        # Manual fault reset for latching faults.
        # Accepted payloads: "stall", "estop", "all"
        try:
            cmd = msg.data.strip().lower()
            if cmd in ('stall', 'all'):
                self.fault_stall = False
                self.get_logger().info('fault_stall cleared by /safety/clear_fault')
            if cmd in ('estop', 'all'):
                self.fault_estop = False
                self.get_logger().info('fault_estop cleared by /safety/clear_fault')
            if cmd not in ('stall', 'estop', 'all'):
                self.get_logger().warn(f'clear_fault: unknown command "{cmd}" — use "stall", "estop", or "all"')
        except Exception as e:
            self.get_logger().warn(f'clear_fault callback error: {e}')

    # ── Safety check (20 Hz) ─────────────────────────────────────────────────

    def _safety_check_cb(self):
        # This is the core of safety_node. It runs 20 times per second and
        # decides whether the latest nav command is safe to pass to the motors.

        # 1. Update auto-clearing faults based on latest sensor readings.
        #    Latching faults (stall, estop) are only cleared by _clear_fault_cb.

        # Watchdog: only relevant when motion was actively commanded.
        # If the last received command was zero (or nothing ever arrived), the
        # robot is legitimately idle — silence is expected and is not a fault.
        elapsed = time.monotonic() - self._last_cmd_time
        last = self._latest_cmd_vel_raw
        last_was_moving = (
            last.linear.x != 0.0 or last.linear.y != 0.0 or last.angular.z != 0.0
        )
        if last_was_moving and elapsed > self._watchdog_sec:
            if not self.fault_watchdog:
                self.fault_watchdog = True
                self._publish_fault(f'fault_watchdog: no /cmd_vel_raw for {elapsed:.2f}s after non-zero command')
        else:
            self.fault_watchdog = False

        # Tilt: robot is leaning too far — risk of falling or tipping a payload
        if self._current_tilt_deg > self._max_tilt:
            if not self.fault_tilt:
                self.fault_tilt = True
                self._publish_fault(f'fault_tilt: tilt {self._current_tilt_deg:.1f}° > {self._max_tilt}°')
        else:
            self.fault_tilt = False

        # Voltage: battery critically low — keep running but stop moving
        if self._current_voltage > 0.0 and self._current_voltage < self._crit_voltage:
            if not self.fault_voltage:
                self.fault_voltage = True
                self._publish_fault(f'fault_voltage: {self._current_voltage:.2f}V < {self._crit_voltage}V critical floor')
        else:
            self.fault_voltage = False

        # Warn (but don't fault) when voltage is low but not yet critical
        if 0.0 < self._current_voltage < self._warn_voltage and not self.fault_voltage:
            self.get_logger().warn(f'Battery low: {self._current_voltage:.2f}V (warning threshold {self._warn_voltage}V)')

        # 2. Gate the velocity command.
        any_fault = (
            self.fault_tilt      or
            self.fault_voltage   or
            self.fault_watchdog  or
            self.fault_proximity or
            self.fault_stall     or
            self.fault_estop
        )

        if any_fault:
            # CRITICAL: actively publish zeros — do NOT just stop publishing.
            # If we went silent, motor_control_node would keep its last command
            # until its own watchdog fired (up to 0.5s later). Always be explicit.
            self._cmd_vel_pub.publish(Twist())
        else:
            # All clear — pass the latest nav command straight through
            self._cmd_vel_pub.publish(self._latest_cmd_vel_raw)

    # ── Status publisher (1 Hz) ───────────────────────────────────────────────

    def _status_cb(self):
        # Human-readable summary of the safety state. Useful for monitoring
        # with: ros2 topic echo /safety/status
        active = [
            name for name, active in [
                ('tilt',      self.fault_tilt),
                ('voltage',   self.fault_voltage),
                ('watchdog',  self.fault_watchdog),
                ('proximity', self.fault_proximity),
                ('stall',     self.fault_stall),
                ('estop',     self.fault_estop),
            ] if active
        ]

        if active:
            summary = f'FAULTED [{", ".join(active)}] | {self._current_voltage:.2f}V | tilt {self._current_tilt_deg:.1f}°'
        else:
            summary = f'OK | {self._current_voltage:.2f}V | tilt {self._current_tilt_deg:.1f}°'

        self._status_pub.publish(String(data=summary))
        self.get_logger().info(f'safety: {summary}')

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _publish_fault(self, message: str):
        # Log at error level AND publish to /safety/fault so behavior_node
        # (or any other subscriber) can react to fault events immediately.
        self.get_logger().error(message)
        self._fault_pub.publish(String(data=message))


def main(args=None):
    rclpy.init(args=args)
    node = SafetyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
