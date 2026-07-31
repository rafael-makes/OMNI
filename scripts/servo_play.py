#!/usr/bin/env python3
"""
servo_play.py — drive OMNI's servos by name, interactively, from the keyboard.

It publishes Float32MultiArray messages on /servo_commands, which is the topic
servo_node listens on. The message is a flat list of triples:
    [board, channel, angle,  board, channel, angle,  ...]
and servo_node maps (board, channel) -> servo name and moves it. For every
servo, 90 degrees = the calibrated neutral/centre position.

PREREQUISITE: servo_node must already be running in another terminal:
    ros2 run servo_node servo_node

RUN THIS SCRIPT (from an interactive shell that has ROS2 sourced):
    python3 ~/omni_ws/scripts/servo_play.py

THINGS YOU CAN TYPE AT THE > PROMPT:
    head_pan 70                 move one servo to 70 degrees
    head_pan 60 head_tilt 84    move several at once (space separated pairs)
    home                        send every servo back to 90 (neutral), gently
    list                        show all servos and their safe angle ranges
    q  /  quit  /  Ctrl-D       exit

Angles are CLAMPED to the safe ranges below so you can't strain a servo by
fat-fingering a number. Edit SAFE_LIMITS if you've confirmed a wider mechanical
range by hand.
"""

import os
import sys
import time

import yaml
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

CONFIG_PATH = os.path.expanduser('~/omni_ws/config/servo_config.yaml')

# ---------------------------------------------------------------------------
# Safe angle limits, in degrees. 90 = neutral/centre for every servo.
#
# HEAD limits are MECHANICAL — the head physically cannot go past these, and
# pushing into a hard stop will stall and cook the servo. Do NOT widen these
# without watching the robot move.  (pan: 90=centre, higher=left, lower=right;
# tilt: 84=level, higher=down, lower=up.)
#
# ARM/GRIPPER bands are conservative defaults so play is safe out of the box.
# Once you've checked an arm's real range of motion by hand, widen its entry.
# ---------------------------------------------------------------------------
SAFE_LIMITS = {
    'head_pan':       (50, 130),
    'head_tilt':      (74,  94),
    'left_shoulder':  (45, 135),
    'left_elbow':     (45, 135),
    'left_wrist':     (45, 135),
    'left_gripper':   (45, 135),
    'right_shoulder': (45, 135),
    'right_elbow':    (45, 135),
    'right_wrist':    (45, 135),
    'right_gripper':  (45, 135),
}
DEFAULT_LIMIT = (45, 135)   # used if a servo isn't listed above


def load_servo_map(path):
    """Read servo_config.yaml -> {name: (board, channel)}. Keeps us in sync
    with the calibration tool instead of hardcoding channels here."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    servos = {}
    for name, c in cfg['servos'].items():
        servos[name] = (int(c['board']), int(c['channel']))
    return servos


class ServoPlay(Node):
    def __init__(self, servo_map):
        super().__init__('servo_play')
        self.servo_map = servo_map
        self.pub = self.create_publisher(Float32MultiArray, '/servo_commands', 10)

    def wait_for_servo_node(self, timeout=3.0):
        """Return True once servo_node has subscribed to /servo_commands."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.pub.get_subscription_count() > 0:
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.pub.get_subscription_count() > 0

    def clamp(self, name, angle):
        lo, hi = SAFE_LIMITS.get(name, DEFAULT_LIMIT)
        clamped = max(lo, min(hi, angle))
        if clamped != angle:
            print(f'  ! {name} {angle:g} is outside [{lo}, {hi}] '
                  f'-> clamped to {clamped:g}')
        return clamped

    def move(self, moves):
        """moves: list of (name, angle). Publishes one combined command."""
        data = []
        for name, angle in moves:
            board, channel = self.servo_map[name]
            angle = self.clamp(name, float(angle))
            data.extend([float(board), float(channel), angle])
            print(f'  -> {name} = {angle:g} deg')
        msg = Float32MultiArray()
        msg.data = data
        self.pub.publish(msg)

    def home(self):
        """Send everything to 90, one servo at a time with a short delay.
        Moving all servos at once can cause a current surge / BMS trip, so we
        space them out the same way servo_node does on startup."""
        print('  homing all servos to 90 deg...')
        for name in self.servo_map:
            self.move([(name, 90)])
            time.sleep(0.15)


def print_servo_list(servo_map):
    print('\nServos (90 = neutral):')
    for name in servo_map:
        lo, hi = SAFE_LIMITS.get(name, DEFAULT_LIMIT)
        print(f'  {name:<15} safe range {lo}-{hi}')
    print()


def parse_pairs(tokens):
    """Turn ['head_pan','70','head_tilt','84'] into [('head_pan',70.0),...].
    Raises ValueError on bad input."""
    if len(tokens) % 2 != 0:
        raise ValueError('expected pairs of <servo> <angle>')
    moves = []
    for i in range(0, len(tokens), 2):
        name, angle = tokens[i], tokens[i + 1]
        moves.append((name, float(angle)))
    return moves


def main():
    servo_map = load_servo_map(CONFIG_PATH)

    rclpy.init()
    node = ServoPlay(servo_map)

    if not node.wait_for_servo_node():
        print('WARNING: no subscriber on /servo_commands — is servo_node running?')
        print('         start it with:  ros2 run servo_node servo_node\n')

    print('OMNI servo player. Type "list" to see servos, "q" to quit.')
    print_servo_list(servo_map)

    try:
        while True:
            try:
                line = input('> ').strip()
            except EOFError:       # Ctrl-D
                print()
                break
            if not line:
                continue

            tokens = line.split()
            cmd = tokens[0].lower()

            if cmd in ('q', 'quit', 'exit'):
                break
            if cmd == 'list':
                print_servo_list(servo_map)
                continue
            if cmd == 'home':
                node.home()
                continue

            # Otherwise: one or more <servo> <angle> pairs.
            try:
                moves = parse_pairs(tokens)
            except ValueError as e:
                print(f'  ? {e}.  Example: head_pan 70')
                continue

            unknown = [n for n, _ in moves if n not in servo_map]
            if unknown:
                print(f'  ? unknown servo(s): {", ".join(unknown)}  (try "list")')
                continue

            node.move(moves)
    except KeyboardInterrupt:
        print()
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print('bye')


if __name__ == '__main__':
    main()
