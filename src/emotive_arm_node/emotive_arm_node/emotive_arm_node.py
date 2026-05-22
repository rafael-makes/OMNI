"""
emotive_arm_node.py — State-driven expressive arm gestures for OMNI.

Subscribes to /robot_state (String) and publishes smooth interpolated
servo positions to /servo_commands (Float32MultiArray) based on the
current robot state.

SERVO COMMAND FORMAT:
  Float32MultiArray with triplets: [board_id, channel, angle_degrees, ...]
  Matches servo_node.py's _on_command() handler exactly.

YIELD LOGIC:
  behavior_node publishes /servo_commands for programmatic gestures (e.g.
  wave, point). If another publisher is active on /servo_commands, this
  node backs off for YIELD_TIMEOUT seconds before resuming emotive motion.
  Detection: compare the /servo_commands topic's publisher count — if > 1,
  another node is publishing. We track our own publishes separately to
  distinguish self from others.

INTERPOLATION:
  Each tick advances all servos toward their target by at most MAX_DEG_PER_SEC
  degrees. This produces smooth ramp motion — no snapping.

IDLE DRIFT:
  While IDLE, shoulders and wrists gently oscillate on independent sine
  waves to give OMNI a relaxed, breathing-like appearance.
"""

import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

# ── Servo layout ────────────────────────────────────────────────────────────
# (board, channel, neutral_angle°)  — angle 90 = neutral in servo_node terms
# These names and board/channel values match servo_config.yaml exactly.

SERVOS = {
    'left_shoulder':  (0, 0, 90),
    'left_elbow':     (0, 1, 90),
    'left_wrist':     (0, 2, 90),
    'left_gripper':   (0, 3, 90),
    'right_shoulder': (1, 0, 90),
    'right_elbow':    (1, 1, 90),
    'right_wrist':    (1, 2, 90),
    'right_gripper':  (1, 3, 90),
}

# Servo names in a stable order so we can index them consistently
SERVO_NAMES = list(SERVOS.keys())

# ── Gesture definitions ──────────────────────────────────────────────────────
# Each gesture is a dict {servo_name: target_angle_degrees}.
# Omitted servos hold their current position.
# 90° = neutral for all servos.
# Shoulder: >90 raises arm, <90 lowers.
# Elbow: >90 bends in, <90 extends.
# Wrist: >90 rotates outward, <90 inward.
# Gripper: 90 = open neutral.

GESTURES = {
    'IDLE': {
        # Relaxed, arms hanging loosely at sides
        'left_shoulder':  85,
        'left_elbow':     80,
        'left_wrist':     90,
        'left_gripper':   90,
        'right_shoulder': 95,
        'right_elbow':    100,
        'right_wrist':    90,
        'right_gripper':  90,
    },
    'LISTENING': {
        # Arms raised slightly, open and attentive
        'left_shoulder':  110,
        'left_elbow':     70,
        'left_wrist':     85,
        'left_gripper':   80,
        'right_shoulder': 70,
        'right_elbow':    110,
        'right_wrist':    95,
        'right_gripper':  80,
    },
    'SPEAKING': {
        # Slightly animated, mid-height — movement handled by oscillation
        'left_shoulder':  105,
        'left_elbow':     75,
        'left_wrist':     88,
        'left_gripper':   85,
        'right_shoulder': 75,
        'right_elbow':    105,
        'right_wrist':    92,
        'right_gripper':  85,
    },
    'NAVIGATING': {
        # Arms tucked in tight for balance while driving
        'left_shoulder':  80,
        'left_elbow':     60,
        'left_wrist':     90,
        'left_gripper':   90,
        'right_shoulder': 100,
        'right_elbow':    120,
        'right_wrist':    90,
        'right_gripper':  90,
    },
    'EXPLORING': {
        # Same as navigating — arms in for balance
        'left_shoulder':  80,
        'left_elbow':     60,
        'left_wrist':     90,
        'left_gripper':   90,
        'right_shoulder': 100,
        'right_elbow':    120,
        'right_wrist':    90,
        'right_gripper':  90,
    },
    'ERROR': {
        # Arms raised in alarm
        'left_shoulder':  130,
        'left_elbow':     60,
        'left_wrist':     80,
        'left_gripper':   70,
        'right_shoulder': 50,
        'right_elbow':    120,
        'right_wrist':    100,
        'right_gripper':  70,
    },
    'DOCKING': {
        # Same as navigating
        'left_shoulder':  80,
        'left_elbow':     60,
        'left_wrist':     90,
        'left_gripper':   90,
        'right_shoulder': 100,
        'right_elbow':    120,
        'right_wrist':    90,
        'right_gripper':  90,
    },
}

# Fall back to IDLE gesture for any unknown state
_DEFAULT_GESTURE = GESTURES['IDLE']

# ── Motion parameters ────────────────────────────────────────────────────────

TICK_HZ          = 20       # update rate for interpolation
MAX_DEG_PER_SEC  = 30.0     # max servo speed (degrees/second)
MAX_DEG_PER_TICK = MAX_DEG_PER_SEC / TICK_HZ

# Yield: back off for this many seconds when another publisher is detected
YIELD_TIMEOUT    = 2.0

# Oscillation parameters for IDLE drift and SPEAKING gesture
# Each servo gets its own phase so motion looks natural, not synchronized
_IDLE_AMP   = 3.0    # degrees peak-to-peak / 2
_IDLE_FREQ  = 0.15   # Hz — slow breathing-like drift
_SPEAK_AMP  = 5.0    # degrees — slightly more animated while speaking
_SPEAK_FREQ = 0.4    # Hz


class EmotiveArmNode(Node):

    def __init__(self):
        super().__init__('emotive_arm_node')

        # Current interpolated angles (start at neutral 90° everywhere)
        self._current = {name: 90.0 for name in SERVO_NAMES}

        # Target angles from the active gesture
        self._target = dict(GESTURES['IDLE'])

        # Active state
        self._state = 'IDLE'

        # Monotonic time of last external /servo_commands message
        # Used to implement yield logic
        self._last_external_cmd = 0.0

        # Monotonic start time — used for oscillation phase
        self._start_time = time.monotonic()

        # ── ROS interfaces ─────────────────────────────────────────────────
        self._cmd_pub = self.create_publisher(
            Float32MultiArray, '/servo_commands', 10)

        self.create_subscription(
            String, '/robot_state', self._on_state, 10)

        # Monitor /servo_commands to detect external publishers (yield logic).
        # We subscribe to our own output topic — any message we didn't publish
        # ourselves arrived from another node.
        self._publishing = False   # True for the duration of our own publish call
        self.create_subscription(
            Float32MultiArray, '/servo_commands', self._on_servo_cmd, 10)

        # Main motion tick
        self.create_timer(1.0 / TICK_HZ, self._tick)

        self.get_logger().info('emotive_arm_node ready')

    # ── State callback ─────────────────────────────────────────────────────

    def _on_state(self, msg: String):
        new_state = msg.data.strip()
        if new_state == self._state:
            return
        self._state = new_state
        gesture = GESTURES.get(new_state, _DEFAULT_GESTURE)
        self._target = dict(gesture)
        self.get_logger().info(f'Gesture → {new_state}')

    # ── Yield detection ────────────────────────────────────────────────────

    def _on_servo_cmd(self, _msg: Float32MultiArray):
        # If we are mid-publish, this echo is our own — ignore it.
        if self._publishing:
            return
        # Otherwise it came from another node — record the time.
        self._last_external_cmd = time.monotonic()

    def _yielding(self) -> bool:
        return time.monotonic() - self._last_external_cmd < YIELD_TIMEOUT

    # ── Motion tick ────────────────────────────────────────────────────────

    def _tick(self):
        if self._yielding():
            return

        elapsed = time.monotonic() - self._start_time
        gesture  = GESTURES.get(self._state, _DEFAULT_GESTURE)

        # Per-servo update
        data = []
        for i, name in enumerate(SERVO_NAMES):
            board, channel, _ = SERVOS[name]

            base_target = float(self._target.get(name, 90.0))

            # Oscillation overlay
            phase = i * (math.pi / 4)   # spread phases evenly
            if self._state == 'IDLE':
                overlay = _IDLE_AMP * math.sin(
                    2 * math.pi * _IDLE_FREQ * elapsed + phase)
            elif self._state == 'SPEAKING':
                overlay = _SPEAK_AMP * math.sin(
                    2 * math.pi * _SPEAK_FREQ * elapsed + phase)
            else:
                overlay = 0.0

            target = base_target + overlay

            # Clamp target to [0, 180]
            target = max(0.0, min(180.0, target))

            # Interpolate toward target
            diff = target - self._current[name]
            step = max(-MAX_DEG_PER_TICK, min(MAX_DEG_PER_TICK, diff))
            self._current[name] += step

            data.extend([float(board), float(channel), self._current[name]])

        msg = Float32MultiArray()
        msg.data = data

        self._publishing = True
        self._cmd_pub.publish(msg)
        self._publishing = False

    # ── Shutdown ───────────────────────────────────────────────────────────

    def destroy_node(self):
        # Return arms to neutral before shutting down
        data = []
        for name in SERVO_NAMES:
            board, channel, _ = SERVOS[name]
            data.extend([float(board), float(channel), 90.0])
        msg = Float32MultiArray()
        msg.data = data
        self._publishing = True
        self._cmd_pub.publish(msg)
        self._publishing = False
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = EmotiveArmNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
