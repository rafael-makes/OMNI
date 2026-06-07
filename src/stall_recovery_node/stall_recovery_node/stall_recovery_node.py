"""
stall_recovery_node — detects motor stalls during Nav2 navigation and recovers.

Recovery logic:
  1. Stop — flood /cmd_vel with zero at 10Hz for 0.3s
  2. Check rear clearance via /tof/left_rear and /tof/right_rear
     - If rear is clear (> min_rear_clearance), back up as far as safe
       (up to max_backup_dist, keeping 0.15m safety margin from rear wall)
     - If rear is blocked (< min_rear_clearance), do a short backup only
  3. Rotate opposite to stuck direction (alternates each attempt)
  4. Re-send original goal to Nav2

Publishing /cmd_vel at 10Hz during recovery overrides Nav2 controller_server
(which runs at 5Hz) without needing to cancel the active goal.
"""

import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import Range
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist, PoseStamped
from nav2_msgs.action import NavigateToPose


class StallRecoveryNode(Node):

    def __init__(self):
        super().__init__('stall_recovery_node')

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter('backup_speed',       0.28)  # m/s — 0.18 gave ~55 PWM, too close to MIN_PWM=50 causing flicker
        self.declare_parameter('min_backup_dist',    0.20)  # m — always back up at least this far
        self.declare_parameter('max_backup_dist',    0.60)  # m — never back up more than this
        self.declare_parameter('rear_safety_margin', 0.15)  # m — keep this gap behind when backing
        self.declare_parameter('min_rear_clearance', 0.25)  # m — below this = rear blocked, short backup only
        self.declare_parameter('rotate_speed',       1.6)   # rad/s — 1.1 gave ~52 PWM per wheel, too close to MIN_PWM=50
        self.declare_parameter('rotate_duration',    0.9)   # seconds (~82° at 1.6 rad/s)
        self.declare_parameter('cooldown',           2.5)   # seconds between recoveries
        self.declare_parameter('publish_hz',         10.0)  # rate to flood /cmd_vel during recovery

        self._backup_speed       = self.get_parameter('backup_speed').value
        self._min_backup_dist    = self.get_parameter('min_backup_dist').value
        self._max_backup_dist    = self.get_parameter('max_backup_dist').value
        self._rear_safety_margin = self.get_parameter('rear_safety_margin').value
        self._min_rear_clearance = self.get_parameter('min_rear_clearance').value
        self._rotate_speed       = self.get_parameter('rotate_speed').value
        self._rotate_duration    = self.get_parameter('rotate_duration').value
        self._cooldown           = self.get_parameter('cooldown').value
        self._publish_hz         = self.get_parameter('publish_hz').value

        # ── State ─────────────────────────────────────────────────────────────
        self._current_goal: PoseStamped | None = None
        self._recovering         = False
        self._last_recovery_time = 0.0
        self._last_angular_z     = 0.0
        self._recovery_count     = 0

        # Latest rear ToF readings (meters). None = not yet received.
        self._rear_left_range:  float | None = None
        self._rear_right_range: float | None = None

        # ── Nav2 action client ────────────────────────────────────────────────
        self._nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # ── Publishers / subscribers ──────────────────────────────────────────
        self._cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # tof_node publishes with BEST_EFFORT reliability — match it here
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(Bool,        '/motor/stall_detected', self._stall_cb,      10)
        self.create_subscription(PoseStamped, '/goal_pose',            self._goal_cb,       10)
        self.create_subscription(Twist,       '/cmd_vel',              self._cmdvel_cb,     10)
        self.create_subscription(Range,       '/tof/left_rear',        self._rear_left_cb,  sensor_qos)
        self.create_subscription(Range,       '/tof/right_rear',       self._rear_right_cb, sensor_qos)

        self.get_logger().info('stall_recovery_node ready')

    # ── Sensor callbacks ──────────────────────────────────────────────────────

    def _rear_left_cb(self, msg: Range):
        # Clamp to sensor valid range; treat out-of-range as max
        if msg.range < msg.min_range or msg.range > msg.max_range:
            self._rear_left_range = msg.max_range
        else:
            self._rear_left_range = msg.range

    def _rear_right_cb(self, msg: Range):
        if msg.range < msg.min_range or msg.range > msg.max_range:
            self._rear_right_range = msg.max_range
        else:
            self._rear_right_range = msg.range

    def _goal_cb(self, msg: PoseStamped):
        self._current_goal = msg

    def _cmdvel_cb(self, msg: Twist):
        if not self._recovering:
            self._last_angular_z = msg.angular.z

    # ── Stall trigger ─────────────────────────────────────────────────────────

    def _stall_cb(self, msg: Bool):
        if not msg.data:
            return
        if self._recovering:
            return

        now = time.monotonic()
        elapsed = now - self._last_recovery_time
        if elapsed < self._cooldown:
            self.get_logger().warn(
                f'Stall — cooldown ({elapsed:.1f}/{self._cooldown:.1f}s)')
            return

        self.get_logger().warn('Stall detected! Starting recovery.')
        self._recovering = True
        self._last_recovery_time = now

        t = threading.Thread(target=self._run_recovery, daemon=True)
        t.start()

    # ── Recovery helpers ──────────────────────────────────────────────────────

    def _flood_vel(self, linear_x: float, angular_z: float, duration: float):
        """Publish at publish_hz for `duration` seconds — overrides Nav2's 5Hz."""
        interval = 1.0 / self._publish_hz
        end_time = time.monotonic() + duration
        msg = Twist()
        msg.linear.x  = linear_x
        msg.angular.z = angular_z
        while time.monotonic() < end_time:
            self._cmd_vel_pub.publish(msg)
            time.sleep(interval)

    def _compute_backup_dist(self) -> float:
        """
        Use rear ToF sensors to decide how far to back up.
        Takes the smaller of the two rear readings (most conservative),
        subtracts the safety margin, then clamps to [min, max] backup range.
        If sensors haven't published yet, fall back to min_backup_dist.
        """
        readings = [r for r in (self._rear_left_range, self._rear_right_range)
                    if r is not None]

        if not readings:
            self.get_logger().warn(
                'No rear ToF readings yet — using minimum backup distance.')
            return self._min_backup_dist

        # Most conservative: smallest reading = least clearance
        rear_clearance = min(readings)

        if rear_clearance < self._min_rear_clearance:
            self.get_logger().info(
                f'Rear blocked ({rear_clearance:.2f}m < {self._min_rear_clearance:.2f}m) '
                f'— short backup only ({self._min_backup_dist:.2f}m)')
            return self._min_backup_dist

        # Back up to (clearance - safety_margin), clamped to [min, max]
        safe_dist = rear_clearance - self._rear_safety_margin
        backup_dist = max(self._min_backup_dist, min(safe_dist, self._max_backup_dist))
        self.get_logger().info(
            f'Rear clear ({rear_clearance:.2f}m) — backing up {backup_dist:.2f}m '
            f'(safety margin {self._rear_safety_margin:.2f}m)')
        return backup_dist

    def _send_nav_goal(self, pose: PoseStamped):
        if not self._nav_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error('Nav2 action server not available.')
            return
        goal = NavigateToPose.Goal()
        goal.pose = pose
        future = self._nav_client.send_goal_async(goal)
        deadline = time.monotonic() + 3.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        self.get_logger().info('Goal re-sent to Nav2.')

    # ── Main recovery thread ──────────────────────────────────────────────────

    def _run_recovery(self):
        try:
            self._recovery_count += 1
            attempt = self._recovery_count
            self.get_logger().info(
                f'Recovery attempt #{attempt} — flooding /cmd_vel at {self._publish_hz:.0f}Hz')

            # Step 1: stop
            self._flood_vel(0.0, 0.0, 0.3)

            # Step 2: compute backup distance from rear ToF sensors
            backup_dist = self._compute_backup_dist()
            backup_duration = backup_dist / self._backup_speed
            self.get_logger().info(
                f'Recovery: backing up {backup_dist:.2f}m ({backup_duration:.1f}s)...')
            self._flood_vel(-self._backup_speed, 0.0, backup_duration)

            # Step 3: stop briefly
            self._flood_vel(0.0, 0.0, 0.2)

            # Step 4: rotate — alternate direction each attempt
            if attempt % 2 == 1:
                rotate_dir = -1.0 if self._last_angular_z >= 0 else 1.0
                label = 'opposite to stuck dir'
            else:
                rotate_dir = 1.0 if self._last_angular_z >= 0 else -1.0
                label = 'same as stuck dir (retry)'

            self.get_logger().info(
                f'Recovery: rotating ({label}) for {self._rotate_duration:.1f}s...')
            self._flood_vel(0.0, rotate_dir * self._rotate_speed, self._rotate_duration)

            # Step 5: stop
            self._flood_vel(0.0, 0.0, 0.3)

            # Step 6: re-send goal
            if self._current_goal is not None:
                self.get_logger().info('Recovery: re-sending goal...')
                self._send_nav_goal(self._current_goal)
            else:
                self.get_logger().warn('Recovery: no stored goal — awaiting next goal.')

        except Exception as e:
            self.get_logger().error(f'Recovery exception: {e}')
        finally:
            self._recovering = False
            self.get_logger().info(f'Recovery #{self._recovery_count} complete.')


def main(args=None):
    rclpy.init(args=args)
    node = StallRecoveryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
