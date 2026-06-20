#!/usr/bin/env python3
"""
ps4_teleop.py — PS4 controller teleoperation node for OMNI robot.

Controls:
  D-pad up / down        → linear.x  (forward / back)
  D-pad left / right     → angular.z (turn left / right)
  Diagonal d-pad         → move + turn simultaneously
  L2 trigger             → deadman switch (must be held to move)
  R1                     → boost (2× speed while held)

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

# ── Axis / button indices (pygame DS4 Linux driver) ───────────────────────────
AXIS_L2      = 2   # L2 trigger: rests at -1.0, fully pressed = +1.0
BTN_R1       = 5   # R1 shoulder button — boost

# ── Speed limits ──────────────────────────────────────────────────────────────
# Defaults are conservative for mapping. Override at launch:
#   ros2 run motor_control_node ps4_teleop --ros-args -p max_linear_mps:=0.35
MAX_LINEAR_MPS_DEFAULT  = 0.35  # m/s
MAX_ANGULAR_RPS_DEFAULT = 2.2   # rad/s — must be >1.97 so pure turns exceed Pico MIN_PWM floor
BOOST_FACTOR            = 1.5   # multiplier when R1 held

# ── Wheel floor boost ─────────────────────────────────────────────────────────
# Pico firmware clamps PWM below MIN_PWM=85 to zero. With FF_GAIN≈14 and
# wheel_radius=0.053m this means wheel speeds below ~0.30 m/s produce no motion.
# When a commanded vx/wz would drive either wheel below this floor, we scale the
# whole command up so the slower wheel just meets the floor.
WHEEL_SEP_M   = 0.305   # metres — must match motor_control_node / Pico firmware
MIN_WHEEL_MPS = 0.32    # slightly above the ~0.30 m/s stall floor

# ── Velocity ramp ─────────────────────────────────────────────────────────────
ACCEL_LINEAR_MPS2  = 1.5   # m/s²
ACCEL_ANGULAR_RPS2 = 8.0   # rad/s² — faster ramp to reach 2.2 rad/s quickly

PUBLISH_HZ = 10


class PS4Teleop(Node):

    def __init__(self):
        super().__init__('ps4_teleop')

        self.declare_parameter('max_linear_mps',  MAX_LINEAR_MPS_DEFAULT)
        self.declare_parameter('max_angular_rps', MAX_ANGULAR_RPS_DEFAULT)
        self._max_linear  = self.get_parameter('max_linear_mps').value
        self._max_angular = self.get_parameter('max_angular_rps').value

        self._pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self._timer = self.create_timer(1.0 / PUBLISH_HZ, self._timer_cb)

        self._joy = None
        self._deadman_was_held = False
        self._vx = 0.0   # ramped linear velocity currently being commanded
        self._wz = 0.0   # ramped angular velocity currently being commanded
        self._init_pygame()

        self.get_logger().info(
            "\n"
            "╔══════════════════════════════════════════╗\n"
            "║        OMNI PS4 Teleop — ready           ║\n"
            "╠══════════════════════════════════════════╣\n"
            "║  L2 (hold)       → deadman switch        ║\n"
            "║  D-pad ↕         → forward / back        ║\n"
            "║  D-pad ↔         → turn left / right     ║\n"
            "║  D-pad diagonal  → arc (move + turn)     ║\n"
            "║  R1 (hold)       → 2× speed boost        ║\n"
            f"║  Max linear  : {self._max_linear:.2f} m/s              ║\n"
            f"║  Max angular : {self._max_angular:.2f} rad/s          ║\n"
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

    def _get_button(self, index: int) -> bool:
        """Return button state, False if button index missing."""
        if index < self._joy.get_numbuttons():
            return bool(self._joy.get_button(index))
        return False

    def _get_hat(self) -> tuple:
        """Return d-pad hat (x, y). x: -1=left/+1=right. y: +1=up/-1=down."""
        if self._joy.get_numhats() > 0:
            return self._joy.get_hat(0)
        return (0, 0)

    def _deadman_held(self) -> bool:
        """L2 axis: rests at -1.0, active when pressed past centre (> 0)."""
        return self._get_axis(AXIS_L2) > 0.0

    def _publish_stop(self):
        self._vx = 0.0
        self._wz = 0.0
        self._pub.publish(Twist())

    def _ramp(self, current: float, target: float, max_step: float) -> float:
        delta = target - current
        if abs(delta) <= max_step:
            return target
        return current + max_step * (1.0 if delta > 0.0 else -1.0)

    def _apply_wheel_floor(self, vx: float, wz: float) -> tuple[float, float]:
        """Scale vx/wz up so the slower wheel meets the Pico MIN_PWM floor.

        Pure turns at low wz would otherwise have both wheels below the stall
        floor. Scaling preserves the vx/wz ratio (same arc radius).
        """
        if abs(vx) < 0.001 and abs(wz) < 0.001:
            return 0.0, 0.0
        left  = vx - wz * WHEEL_SEP_M / 2.0
        right = vx + wz * WHEEL_SEP_M / 2.0
        driven = [abs(v) for v in (left, right) if abs(v) > 0.001]
        if not driven:
            return 0.0, 0.0
        min_speed = min(driven)
        if min_speed < MIN_WHEEL_MPS:
            scale = MIN_WHEEL_MPS / min_speed
            vx *= scale
            wz *= scale
        return vx, wz

    # ── 10 Hz publish callback ────────────────────────────────────────────────

    def _timer_cb(self):
        # Pump pygame events so hat/button state stays current
        pygame.event.pump()

        deadman = self._deadman_held()

        if deadman:
            hat_x, hat_y = self._get_hat()
            boost = BOOST_FACTOR if self._get_button(BTN_R1) else 1.0

            # hat_y: +1=up → forward (+linear.x)
            # hat_x: +1=right → right turn (-angular.z, ROS CCW-positive)
            target_vx = float(hat_y)  * self._max_linear  * boost
            target_wz = float(-hat_x) * self._max_angular * boost

            dt = 1.0 / PUBLISH_HZ
            self._vx = self._ramp(self._vx, target_vx, ACCEL_LINEAR_MPS2  * dt)
            self._wz = self._ramp(self._wz, target_wz, ACCEL_ANGULAR_RPS2 * dt)

            vx, wz = self._apply_wheel_floor(self._vx, self._wz)
            msg = Twist()
            msg.linear.x  = vx
            msg.angular.z = wz
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
