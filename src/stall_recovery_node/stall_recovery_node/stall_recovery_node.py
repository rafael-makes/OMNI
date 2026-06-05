"""
stall_recovery_node — detects motor stalls during Nav2 navigation and recovers.

Root cause of stall recovery not working:
  Nav2 controller_server publishes /cmd_vel at 5Hz continuously.
  If recovery just publishes once and stops, Nav2 immediately overwrites it.
  Fix: during recovery, publish cmd_vel at 10Hz (faster than Nav2's 5Hz) so
  recovery commands always win, even without cancelling the Nav2 goal.

Recovery sequence:
  1. Stop — flood /cmd_vel with zero velocity at 10Hz for 0.3s
  2. Back up — flood /cmd_vel with -x velocity at 10Hz for backup_duration
  3. Rotate — flood /cmd_vel with angular velocity at 10Hz for rotate_duration
  4. Stop — zero for 0.3s
  5. Re-send original goal to Nav2

Direction alternates each attempt so if attempt 1 fails, attempt 2 tries
rotating the opposite way (avoids spinning into the same wall repeatedly).
"""

import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from std_msgs.msg import Bool
from geometry_msgs.msg import Twist, PoseStamped
from nav2_msgs.action import NavigateToPose


class StallRecoveryNode(Node):

    def __init__(self):
        super().__init__('stall_recovery_node')

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter('backup_speed',    0.18)   # m/s backward
        self.declare_parameter('backup_duration', 1.2)    # seconds (~22cm)
        self.declare_parameter('rotate_speed',    1.1)    # rad/s
        self.declare_parameter('rotate_duration', 0.9)    # seconds (~57°)
        self.declare_parameter('cooldown',        2.5)    # seconds between recoveries
        self.declare_parameter('publish_hz',      10.0)   # rate to flood cmd_vel during recovery

        self._backup_speed    = self.get_parameter('backup_speed').value
        self._backup_duration = self.get_parameter('backup_duration').value
        self._rotate_speed    = self.get_parameter('rotate_speed').value
        self._rotate_duration = self.get_parameter('rotate_duration').value
        self._cooldown        = self.get_parameter('cooldown').value
        self._publish_hz      = self.get_parameter('publish_hz').value

        # ── State ─────────────────────────────────────────────────────────────
        self._current_goal: PoseStamped | None = None
        self._recovering   = False
        self._last_recovery_time = 0.0
        self._last_angular_z    = 0.0
        self._recovery_count    = 0

        # ── Nav2 action client ────────────────────────────────────────────────
        self._nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # ── Publishers / subscribers ──────────────────────────────────────────
        self._cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.create_subscription(Bool,        '/motor/stall_detected', self._stall_cb,  10)
        self.create_subscription(PoseStamped, '/goal_pose',            self._goal_cb,   10)
        self.create_subscription(Twist,       '/cmd_vel',              self._cmdvel_cb, 10)

        self.get_logger().info('stall_recovery_node ready')

    def _goal_cb(self, msg: PoseStamped):
        self._current_goal = msg

    def _cmdvel_cb(self, msg: Twist):
        if not self._recovering:
            self._last_angular_z = msg.angular.z

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
        """
        Publish (linear_x, angular_z) at publish_hz for `duration` seconds.
        Publishing faster than Nav2's 5Hz means recovery commands always win,
        even without cancelling the Nav2 goal — last-writer-wins on /cmd_vel.
        """
        interval = 1.0 / self._publish_hz
        end_time = time.monotonic() + duration
        msg = Twist()
        msg.linear.x  = linear_x
        msg.angular.z = angular_z
        while time.monotonic() < end_time:
            self._cmd_vel_pub.publish(msg)
            time.sleep(interval)

    def _send_nav_goal(self, pose: PoseStamped):
        """Re-send goal to Nav2 action server."""
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
            self.get_logger().info(f'Recovery attempt #{attempt} — flooding /cmd_vel at {self._publish_hz:.0f}Hz')

            # Step 1: stop (override Nav2 at 10Hz for 0.3s)
            self.get_logger().info('Recovery: stopping...')
            self._flood_vel(0.0, 0.0, 0.3)

            # Step 2: back up
            self.get_logger().info(f'Recovery: backing up {self._backup_duration}s...')
            self._flood_vel(-self._backup_speed, 0.0, self._backup_duration)

            # Step 3: stop briefly
            self._flood_vel(0.0, 0.0, 0.2)

            # Step 4: rotate — alternate direction each attempt
            # Odd  attempts: opposite to last commanded turn (escape the stuck wall)
            # Even attempts: same as last commanded turn (try the other side)
            if attempt % 2 == 1:
                rotate_dir = -1.0 if self._last_angular_z >= 0 else 1.0
                label = 'opposite to stuck dir'
            else:
                rotate_dir = 1.0 if self._last_angular_z >= 0 else -1.0
                label = 'same as stuck dir (retry)'

            self.get_logger().info(
                f'Recovery: rotating ({label}) for {self._rotate_duration}s...')
            self._flood_vel(0.0, rotate_dir * self._rotate_speed, self._rotate_duration)

            # Step 5: stop
            self._flood_vel(0.0, 0.0, 0.3)

            # Step 6: re-send goal
            if self._current_goal is not None:
                self.get_logger().info('Recovery: re-sending goal...')
                self._send_nav_goal(self._current_goal)
            else:
                self.get_logger().warn('Recovery: no stored goal — robot is free, awaiting next goal.')

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
