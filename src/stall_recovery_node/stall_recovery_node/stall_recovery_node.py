"""
stall_recovery_node — detects motor stalls during Nav2 navigation and recovers.

Recovery logic:
  1. Announce recovery on /stall_recovery/active so safety_node yields /cmd_vel
     (otherwise the stall fault's 20Hz zero-flood overrides our backup commands
     and the only motion that survives is the rotation after the fault clears)
  2. Stop — flood /cmd_vel with zero at 10Hz for 0.3s
  3. Check rear clearance via /tof/left_rear and /tof/right_rear
     - If rear is clear (> min_rear_clearance), back up as far as safe
       (up to max_backup_dist, keeping 0.15m safety margin from rear wall)
     - If rear is blocked (< min_rear_clearance), do a short backup only
  4. Check front clearance: if the backup opened space ahead, SKIP the rotation
     and let the planner replan from the new position. Rotating after a clean
     backup tends to swing the body back into whatever caused the stall.
  5. Only if front is still blocked: rotate opposite to stuck direction
  6. Re-send original goal to Nav2

Publishing /cmd_vel at 10Hz during recovery overrides Nav2 controller_server
(which runs at 5Hz) without needing to cancel the active goal.
"""

import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from rcl_interfaces.msg import SetParametersResult
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
        self.declare_parameter('min_front_clearance', 0.40) # m — front clear beyond this after backup = skip rotation

        self._backup_speed       = self.get_parameter('backup_speed').value
        self._min_backup_dist    = self.get_parameter('min_backup_dist').value
        self._max_backup_dist    = self.get_parameter('max_backup_dist').value
        self._rear_safety_margin = self.get_parameter('rear_safety_margin').value
        self._min_rear_clearance = self.get_parameter('min_rear_clearance').value
        self._rotate_speed       = self.get_parameter('rotate_speed').value
        self._rotate_duration    = self.get_parameter('rotate_duration').value
        self._cooldown           = self.get_parameter('cooldown').value
        self._publish_hz         = self.get_parameter('publish_hz').value
        self._min_front_clearance = self.get_parameter('min_front_clearance').value

        # ── Live parameter tuning ─────────────────────────────────────────────
        # All recovery thresholds are read at use-time, so updating the cached
        # value takes effect on the next recovery without a relaunch, e.g.:
        #   ros2 param set /stall_recovery_node backup_speed 0.30
        # Every value must be > 0. Updating a float is atomic (GIL), so this is
        # safe alongside the recovery sequence thread.
        self._param_attr = {
            'backup_speed':        '_backup_speed',
            'min_backup_dist':     '_min_backup_dist',
            'max_backup_dist':     '_max_backup_dist',
            'rear_safety_margin':  '_rear_safety_margin',
            'min_rear_clearance':  '_min_rear_clearance',
            'rotate_speed':        '_rotate_speed',
            'rotate_duration':     '_rotate_duration',
            'cooldown':            '_cooldown',
            'publish_hz':          '_publish_hz',
            'min_front_clearance': '_min_front_clearance',
        }
        self.add_on_set_parameters_callback(self._on_set_parameters)

        # ── State ─────────────────────────────────────────────────────────────
        self._current_goal: PoseStamped | None = None
        self._recovering         = False
        self._last_recovery_time = 0.0
        self._last_angular_z     = 0.0
        self._recovery_count     = 0

        # Latest ToF readings (meters). None = not yet received.
        self._rear_left_range:   float | None = None
        self._rear_right_range:  float | None = None
        self._front_left_range:  float | None = None
        self._front_right_range: float | None = None

        # ── Nav2 action client ────────────────────────────────────────────────
        self._nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # ── Publishers / subscribers ──────────────────────────────────────────
        self._cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        # safety_node yields /cmd_vel to us while this is True — otherwise its
        # 20Hz stall-fault zero-flood overrides our 10Hz recovery commands and
        # the backup never physically happens.
        self._active_pub = self.create_publisher(Bool, '/stall_recovery/active', 10)

        # tof_node publishes with BEST_EFFORT reliability — match it here
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(Bool,        '/motor/stall_detected', self._stall_cb,      10)
        self.create_subscription(PoseStamped, '/goal_pose',            self._goal_cb,       10)
        self.create_subscription(Twist,       '/cmd_vel',              self._cmdvel_cb,     10)
        self.create_subscription(Range,       '/tof/left_rear',        self._rear_left_cb,   sensor_qos)
        self.create_subscription(Range,       '/tof/right_rear',       self._rear_right_cb,  sensor_qos)
        self.create_subscription(Range,       '/tof/front_left',       self._front_left_cb,  sensor_qos)
        self.create_subscription(Range,       '/tof/front_right',      self._front_right_cb, sensor_qos)

        self.get_logger().info('stall_recovery_node ready')

    # ── Live parameter callback ───────────────────────────────────────────────

    def _on_set_parameters(self, params):
        """
        Validate and apply runtime parameter changes (ros2 param set). Returning
        successful=False rejects the whole batch and leaves the stored value
        untouched, so a bad value never reaches the recovery logic. Validate the
        full batch before mutating anything, since a set is applied atomically.
        """
        pending = []  # (attr, value, name) to apply once all checks pass
        for p in params:
            attr = self._param_attr.get(p.name)
            if attr is None:
                continue
            try:
                value = float(p.value)
            except (TypeError, ValueError):
                return SetParametersResult(
                    successful=False,
                    reason=f'{p.name} must be a number, got {p.value!r}')
            if value <= 0.0:
                return SetParametersResult(
                    successful=False,
                    reason=f'{p.name} must be > 0, got {value}')
            pending.append((attr, value, p.name))

        for attr, value, name in pending:
            setattr(self, attr, value)
            self.get_logger().info(f'Parameter updated: {name} = {value}')
        return SetParametersResult(successful=True)

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

    def _front_left_cb(self, msg: Range):
        if msg.range < msg.min_range or msg.range > msg.max_range:
            self._front_left_range = msg.max_range
        else:
            self._front_left_range = msg.range

    def _front_right_cb(self, msg: Range):
        if msg.range < msg.min_range or msg.range > msg.max_range:
            self._front_right_range = msg.max_range
        else:
            self._front_right_range = msg.range

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

    def _front_is_clear(self) -> bool:
        """True if both front ToF sensors see at least min_front_clearance."""
        readings = [r for r in (self._front_left_range, self._front_right_range)
                    if r is not None]
        if not readings:
            return False   # no data — assume blocked, fall through to rotation
        return min(readings) > self._min_front_clearance

    def _run_recovery(self):
        try:
            self._recovery_count += 1
            attempt = self._recovery_count
            self.get_logger().info(
                f'Recovery attempt #{attempt} — flooding /cmd_vel at {self._publish_hz:.0f}Hz')

            # Step 0: tell safety_node to yield /cmd_vel to us. Without this its
            # stall-fault zero-flood (20Hz) overrides our backup (10Hz) and the
            # robot never actually moves backward — it just spins later.
            self._active_pub.publish(Bool(data=True))
            time.sleep(0.1)   # let safety_node process the message before we drive

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

            # Step 4: rotate ONLY if the front is still blocked after backing up.
            # If the backup opened space, rotating just swings the body back into
            # whatever caused the stall — let the planner replan from here instead.
            if self._front_is_clear():
                self.get_logger().info(
                    'Recovery: front clear after backup — skipping rotation, replanning.')
            else:
                if attempt % 2 == 1:
                    rotate_dir = -1.0 if self._last_angular_z >= 0 else 1.0
                    label = 'opposite to stuck dir'
                else:
                    rotate_dir = 1.0 if self._last_angular_z >= 0 else -1.0
                    label = 'same as stuck dir (retry)'

                self.get_logger().info(
                    f'Recovery: front blocked — rotating ({label}) for {self._rotate_duration:.1f}s...')
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
            self._active_pub.publish(Bool(data=False))
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
