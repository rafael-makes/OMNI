#!/usr/bin/env python3
"""
ps4_teleop.py — PS4 controller teleoperation node for OMNI robot.

Controls:
  Left  stick vertical   → linear.x  (forward / back)
  Right stick horizontal → angular.z (turn left / right)
  L2 trigger             → deadman switch (must be held to move)

Publishes geometry_msgs/Twist to /cmd_vel at 10 Hz.
"""

import sys
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

try:
    import pygame
except ImportError:
    print("ERROR: pygame is not installed.  Run:  pip3 install pygame")
    sys.exit(1)

# ── Axis indices (pygame / ds4 via jstest) ────────────────────────────────────
AXIS_LINEAR  = 1   # left  stick up/down  — forward/back
AXIS_ANGULAR = 0   # left  stick left/right — turn
AXIS_L2      = 2   # L2 trigger: rests at -32767, fully pressed = +32767

# ── Speed limits (conservative for first drive) ───────────────────────────────
MAX_LINEAR_MPS  = 0.3   # m/s
MAX_ANGULAR_RPS = 0.8   # rad/s

PUBLISH_HZ = 10


def apply_deadband(value, deadband=0.08):
    if abs(value) < deadband:
        return 0.0
    return (value - deadband * (1 if value > 0 else -1)) / (1.0 - deadband)


class PS4Teleop(Node):

    def __init__(self):
        super().__init__('ps4_teleop')

        self.declare_parameter('deadband', 0.08)

        self._pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self._timer = self.create_timer(1.0 / PUBLISH_HZ, self._timer_cb)

        self._joy = None
        self._deadman_was_held = False
        self._init_pygame()

        db = self.get_parameter('deadband').value
        self.get_logger().info(
            "\n"
            "╔══════════════════════════════════════════╗\n"
            "║        OMNI PS4 Teleop — ready           ║\n"
            "╠══════════════════════════════════════════╣\n"
            "║  L2 (hold)       → deadman switch        ║\n"
            "║  Left  stick ↕   → forward / back        ║\n"
            "║  Right stick ↔   → turn left / right     ║\n"
            f"║  Max linear  : {MAX_LINEAR_MPS:.1f} m/s               ║\n"
            f"║  Max angular : {MAX_ANGULAR_RPS:.1f} rad/s             ║\n"
            f"║  Stick deadband: {db:.2f}                    ║\n"
            "║  Ctrl-C          → exit                  ║\n"
            "╚══════════════════════════════════════════╝"
        )

    # ── pygame / joystick initialisation ─────────────────────────────────────

    def _init_pygame(self):
        pygame.init()
        pygame.joystick.init()

        count = pygame.joystick.get_count()
        if count == 0:
            self.get_logger().error(
                "No joystick found at /dev/input/js0.  "
                "Is the controller paired and /dev/input/js0 present?"
            )
            sys.exit(1)

        self._joy = pygame.joystick.Joystick(0)
        self._joy.init()
        self.get_logger().info(
            f"Joystick detected: '{self._joy.get_name()}' "
            f"({self._joy.get_numaxes()} axes, "
            f"{self._joy.get_numbuttons()} buttons)"
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _get_axis(self, index: int) -> float:
        """Return axis value clamped to [-1, 1], 0.0 if axis missing."""
        if index < self._joy.get_numaxes():
            return max(-1.0, min(1.0, self._joy.get_axis(index)))
        return 0.0

    def _deadman_held(self) -> bool:
        """L2 axis: rests at -1.0 (raw -32767), active when > 0."""
        return self._get_axis(AXIS_L2) > 0.0

    def _publish_stop(self):
        self._pub.publish(Twist())

    # ── 10 Hz publish callback ────────────────────────────────────────────────

    def _timer_cb(self):
        # Pump pygame events so axis/button state stays current
        pygame.event.pump()

        deadman = self._deadman_held()

        if deadman:
            db = self.get_parameter('deadband').value
            # Left stick vertical: pygame up = negative → negate for ROS forward+
            raw_linear  = apply_deadband(-self._get_axis(AXIS_LINEAR),  db)
            # Left stick horizontal: pygame right = positive → negate for ROS left+
            raw_angular = apply_deadband(-self._get_axis(AXIS_ANGULAR), db)

            msg = Twist()
            msg.linear.x  = raw_linear  * MAX_LINEAR_MPS
            msg.angular.z = raw_angular * MAX_ANGULAR_RPS
            self._pub.publish(msg)
            self._deadman_was_held = True

        else:
            if self._deadman_was_held:
                # Deadman just released — send one clean stop
                self._publish_stop()
                self.get_logger().info("Deadman released — motors stopped.")
                self._deadman_was_held = False

    def destroy_node(self):
        self._publish_stop()
        pygame.quit()
        super().destroy_node()


# ── entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = PS4Teleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
