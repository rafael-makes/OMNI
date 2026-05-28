#!/usr/bin/env python3
"""
motor_direction_test.py — Interactive test to verify motor and encoder direction
after hardware changes (motor swap, encoder relocation, etc.)

Tests each wheel independently at a safe speed, reports what the encoder sees,
and tells you exactly which firmware constants to change if anything is wrong.

Run with:
    ros2 run motor_control_node motor_direction_test
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import sys
import threading
import time


# Safe test speed — above MIN_PWM=85 deadzone but slow enough to catch safely
TEST_SPEED_MPS = 0.45   # m/s
TEST_DURATION  = 2.0    # seconds per test


class MotorDirectionTest(Node):

    def __init__(self):
        super().__init__('motor_direction_test')
        self._pub  = self.create_publisher(Twist, '/cmd_vel', 10)
        self._left_vel  = 0.0
        self._right_vel = 0.0
        # Match motor_control_node's BEST_EFFORT odom QoS
        odom_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._odom_sub  = self.create_subscription(
            Odometry, '/odom', self._odom_cb, odom_qos)
        self._lock = threading.Lock()

    def _odom_cb(self, msg):
        with self._lock:
            self._left_vel  = msg.twist.twist.linear.x   # vx ≈ avg wheel speed
            # We can't get per-wheel from odom directly — use vx and vth
            self._vx  = msg.twist.twist.linear.x
            self._vth = msg.twist.twist.angular.z

    def _send(self, vx, wz):
        msg = Twist()
        msg.linear.x  = vx
        msg.angular.z = wz
        self._pub.publish(msg)

    def _stop(self):
        self._send(0.0, 0.0)

    def _wait_input(self, prompt):
        print(prompt, end='', flush=True)
        return input()

    def run(self):
        print("\n" + "═"*58)
        print("  OMNI Motor & Encoder Direction Test")
        print("═"*58)
        print("  This test drives each wheel independently.")
        print("  Keep OMNI on the floor with room to move slightly.")
        print("  Each test runs for 2 seconds then stops.")
        print("═"*58)

        results = {}

        # ── TEST 1: LEFT WHEEL FORWARD ────────────────────────────────────────
        self._wait_input("\n[TEST 1] Left wheel FORWARD — press Enter to start...")
        print(f"  Commanding left wheel forward at {TEST_SPEED_MPS} m/s...")
        self._vx = 0.0; self._vth = 0.0
        start_vx_samples = []

        for _ in range(int(TEST_DURATION * 10)):
            # Left forward only: vx=speed, wz=+speed/sep → left wheel cancels, right doubles
            # Actually use: vx + wz trick to isolate left: vx=v/2, wz=v/sep → left=0, right=v
            # Simpler: just use positive vx to drive both, observe left only physically
            # Best isolation: use wz only — left goes backward, right goes forward
            # For left forward isolation: vx=v/2, wz=-v/(wheel_sep) → left=v, right=0
            # wheel_sep=0.305, so wz = -TEST_SPEED_MPS/0.305 = -1.475
            self._send(TEST_SPEED_MPS / 2.0, -TEST_SPEED_MPS / 0.305)
            rclpy.spin_once(self, timeout_sec=0.1)
            with self._lock:
                start_vx_samples.append(self._vx)

        self._stop()
        rclpy.spin_once(self, timeout_sec=0.2)

        avg_vx_l = sum(start_vx_samples[-10:]) / max(len(start_vx_samples[-10:]), 1)
        print(f"  Observed odom vx (last 1s avg): {avg_vx_l:+.3f} m/s")

        lfw_dir  = self._wait_input("  Did the LEFT wheel spin FORWARD (robot moves fwd on that side)? [y/n]: ").strip().lower()
        lfw_enc  = self._wait_input("  Was odom vx POSITIVE (roughly > 0)? [y/n]: ").strip().lower()
        results['left_motor_fwd']  = lfw_dir == 'y'
        results['left_enc_fwd']    = lfw_enc == 'y'

        time.sleep(1.0)

        # ── TEST 2: RIGHT WHEEL FORWARD ───────────────────────────────────────
        self._wait_input("\n[TEST 2] Right wheel FORWARD — press Enter to start...")
        print(f"  Commanding right wheel forward at {TEST_SPEED_MPS} m/s...")
        self._vx = 0.0; self._vth = 0.0
        start_vx_samples = []

        for _ in range(int(TEST_DURATION * 10)):
            # Right forward isolation: vx=v/2, wz=+v/sep → right=v, left=0
            self._send(TEST_SPEED_MPS / 2.0, TEST_SPEED_MPS / 0.305)
            rclpy.spin_once(self, timeout_sec=0.1)
            with self._lock:
                start_vx_samples.append(self._vx)

        self._stop()
        rclpy.spin_once(self, timeout_sec=0.2)

        avg_vx_r = sum(start_vx_samples[-10:]) / max(len(start_vx_samples[-10:]), 1)
        print(f"  Observed odom vx (last 1s avg): {avg_vx_r:+.3f} m/s")

        rfw_dir = self._wait_input("  Did the RIGHT wheel spin FORWARD (robot moves fwd on that side)? [y/n]: ").strip().lower()
        rfw_enc = self._wait_input("  Was odom vx POSITIVE (roughly > 0)? [y/n]: ").strip().lower()
        results['right_motor_fwd'] = rfw_dir == 'y'
        results['right_enc_fwd']   = rfw_enc == 'y'

        # ── RESULTS ───────────────────────────────────────────────────────────
        print("\n" + "═"*58)
        print("  RESULTS — required firmware changes:")
        print("═"*58)

        changes_needed = False

        if not results['left_motor_fwd']:
            print("  ✗ LEFT motor direction WRONG")
            print("    → In firmware: swap L_DIR pin wiring, OR")
            print("      negate the drive_motor call for left:")
            print("      drive_motor(L_PWM, L_DIR, -l_out)")
            changes_needed = True
        else:
            print("  ✓ Left motor direction correct")

        if not results['left_enc_fwd']:
            print("  ✗ LEFT encoder direction WRONG")
            print("    → In firmware: change  L_ENC_DIR = -1.0f  to  L_ENC_DIR = 1.0f")
            print("      (or vice-versa — flip the sign)")
            changes_needed = True
        else:
            print("  ✓ Left encoder direction correct")

        if not results['right_motor_fwd']:
            print("  ✗ RIGHT motor direction WRONG")
            print("    → In firmware: swap R_DIR pin wiring, OR")
            print("      negate the drive_motor call for right:")
            print("      drive_motor(R_PWM, R_DIR, -r_out)")
            changes_needed = True
        else:
            print("  ✓ Right motor direction correct")

        if not results['right_enc_fwd']:
            print("  ✗ RIGHT encoder direction WRONG")
            print("    → In firmware: change  R_ENC_DIR = 1.0f  to  R_ENC_DIR = -1.0f")
            print("      (or vice-versa — flip the sign)")
            changes_needed = True
        else:
            print("  ✓ Right encoder direction correct")

        if not changes_needed:
            print("  ✓ All directions correct — no firmware changes needed!")
            print("  Note: FF_GAIN values may still need re-tuning after motor swap.")

        print("═"*58)
        print("  After any firmware fix: re-run this test to confirm,")
        print("  then re-run the straight-line calibration.")
        print("═"*58 + "\n")


def main(args=None):
    rclpy.init(args=args)
    node = MotorDirectionTest()

    # Give motor_control_node a moment to connect
    print("Waiting 2s for motor_control_node to be ready...")
    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.1)

    try:
        node.run()
    except KeyboardInterrupt:
        node._stop()
    finally:
        node._stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
