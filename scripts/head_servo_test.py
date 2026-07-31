#!/usr/bin/env python3
"""
OMNI Head Pan/Tilt Direction Test
---------------------------------
Interactive jog tool for the head_pan and head_tilt servos. Publishes to
/servo_commands, so it drives the head through the SAME path the
head_tracking_node will use (servo_node owns the I2C bus and the
calibrated angle->PWM mapping).

  >>> Start servo_node FIRST, then run this in a second terminal. <<<
      ros2 run servo_node servo_node          # terminal 1
      python3 ~/omni_ws/scripts/head_servo_test.py   # terminal 2

Goal of this session: jog each axis and write down the sign convention --
  e.g. "pan angle 36 = head looks LEFT, 144 = head looks RIGHT"
       "tilt angle 36 = head looks DOWN, 144 = head looks UP"
so head_tracking_node's pan_gain / tilt_gain signs can be set correctly.

Controls
--------
  Arrow Left  / Right   Jog PAN  down / up in angle
  Arrow Down  / Up      Jog TILT down / up in angle
  f                     Toggle fine (1 deg) / coarse (5 deg) step
  n                     Both axes back to neutral (90 deg)
  s                     Sweep current understanding: print angle->PWM table
  q                     Return head to neutral and quit

Angles are clamped to 36-144 deg -- the same 20%-80% travel band that
head_tracking_node enforces, so this never drives into the end-stops.
"""

import os
import sys
import termios
import time
import tty

import rclpy
import yaml
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

CONFIG_PATH = os.path.expanduser('~/omni_ws/config/servo_config.yaml')

# Match head_tracking_node's travel limits: 20%-80% of the 0-180 range.
TRAVEL_MIN = 36.0
TRAVEL_MAX = 144.0
NEUTRAL    = 90.0

STEP_COARSE = 5.0
STEP_FINE   = 1.0

# Fallbacks if config can't be read; overwritten from servo_config.yaml below.
DEFAULT_PAN  = {'board': 0, 'channel': 4}
DEFAULT_TILT = {'board': 0, 'channel': 5}


# ─────────────────────────────── config ──────────────────────────────────────

def load_head_servos() -> tuple[dict, dict]:
    """Return (pan_cfg, tilt_cfg) board/channel from servo_config.yaml."""
    pan, tilt = dict(DEFAULT_PAN), dict(DEFAULT_TILT)
    try:
        with open(CONFIG_PATH) as f:
            servos = (yaml.safe_load(f) or {}).get('servos', {})
        if 'head_pan' in servos:
            pan = {'board': servos['head_pan']['board'],
                   'channel': servos['head_pan']['channel']}
        if 'head_tilt' in servos:
            tilt = {'board': servos['head_tilt']['board'],
                    'channel': servos['head_tilt']['channel']}
    except Exception as e:
        print(f'WARNING: could not read {CONFIG_PATH} ({e}); using defaults')
    return pan, tilt


# ─────────────────────────────── keyboard ────────────────────────────────────

def getch() -> str:
    """One keypress. Arrows -> 'UP'/'DOWN'/'LEFT'/'RIGHT'."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == '\x1b':                       # escape sequence
            seq = sys.stdin.read(2)
            return {'[A': 'UP', '[B': 'DOWN',
                    '[C': 'RIGHT', '[D': 'LEFT'}.get(seq, 'ESC')
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ─────────────────────────────── display ─────────────────────────────────────

BOLD  = '\033[1m'
CYAN  = '\033[96m'
GREEN = '\033[92m'
YELL  = '\033[93m'
RESET = '\033[0m'
CLEAR = '\033c'


def bar(angle: float) -> str:
    span   = TRAVEL_MAX - TRAVEL_MIN
    frac   = (angle - TRAVEL_MIN) / span
    length = 30
    filled = int(round(frac * length))
    return '█' * filled + '░' * (length - filled)


def draw(pan: float, tilt: float, fine: bool, pan_cfg: dict, tilt_cfg: dict,
         last_action: str):
    step = STEP_FINE if fine else STEP_COARSE
    print(CLEAR, end='')
    print(f'{BOLD}{CYAN}═══ OMNI Head Pan/Tilt Direction Test ═══{RESET}')
    print()
    print(f'  Driving via /servo_commands  (servo_node must be running)')
    print(f'  pan  -> board {pan_cfg["board"]}, ch {pan_cfg["channel"]}')
    print(f'  tilt -> board {tilt_cfg["board"]}, ch {tilt_cfg["channel"]}')
    print()
    print(f'  {BOLD}PAN{RESET}   {YELL}{pan:6.1f}°{RESET}  [{bar(pan)}]')
    print(f'  {BOLD}TILT{RESET}  {YELL}{tilt:6.1f}°{RESET}  [{bar(tilt)}]')
    print()
    print(f'  Range clamp : {TRAVEL_MIN:.0f}°–{TRAVEL_MAX:.0f}°   '
          f'neutral = {NEUTRAL:.0f}°')
    step_label = (f'{CYAN}FINE{RESET} ({step:.0f}°)' if fine
                  else f'{YELL}COARSE{RESET} ({step:.0f}°)')
    print(f'  Step size   : {step_label}  (press f to toggle)')
    print()
    print(f'  {GREEN}← / →{RESET}  Pan down / up      '
          f'{GREEN}↓ / ↑{RESET}  Tilt down / up')
    print(f'  {GREEN}n{RESET}      Neutral (90°)      '
          f'{GREEN}f{RESET}      Toggle step')
    print(f'  {GREEN}s{RESET}      Print angle table  '
          f'{GREEN}q{RESET}      Neutral & quit')
    print()
    if last_action:
        print(f'  {GREEN}>>{RESET} {last_action}')
    print()
    print('  Watch the head and note which physical direction each move is.')


def print_table(pan: float, tilt: float):
    print(CLEAR, end='')
    print(f'{BOLD}{CYAN}Document these — current commanded angles{RESET}')
    print()
    print(f'  PAN  = {pan:6.1f}°   ->  head physically looks: ____________')
    print(f'  TILT = {tilt:6.1f}°   ->  head physically looks: ____________')
    print()
    print('  Reference extremes to record for head_tracking_node signs:')
    print(f'    pan  {TRAVEL_MIN:.0f}° = ?    pan  {TRAVEL_MAX:.0f}° = ?'
          '   (left vs right)')
    print(f'    tilt {TRAVEL_MIN:.0f}° = ?    tilt {TRAVEL_MAX:.0f}° = ?'
          '   (up vs down)')
    print()
    print('  Press any key to return...')
    getch()


# ─────────────────────────────── main ────────────────────────────────────────

def clamp(a: float) -> float:
    return max(TRAVEL_MIN, min(TRAVEL_MAX, a))


def main():
    pan_cfg, tilt_cfg = load_head_servos()

    rclpy.init()
    node = Node('head_servo_test')
    pub = node.create_publisher(Float32MultiArray, '/servo_commands', 10)
    # Give DDS discovery a moment so the first command isn't dropped.
    time.sleep(0.5)

    pan  = NEUTRAL
    tilt = NEUTRAL
    fine = False
    last_action = 'started at neutral'

    def publish():
        msg = Float32MultiArray()
        msg.data = [
            float(pan_cfg['board']),  float(pan_cfg['channel']),  float(pan),
            float(tilt_cfg['board']), float(tilt_cfg['channel']), float(tilt),
        ]
        pub.publish(msg)

    try:
        publish()
        while True:
            draw(pan, tilt, fine, pan_cfg, tilt_cfg, last_action)
            key = getch()
            step = STEP_FINE if fine else STEP_COARSE

            if key == 'RIGHT':
                pan = clamp(pan + step)
                last_action = f'PAN  +{step:.0f}° -> {pan:.1f}°'
            elif key == 'LEFT':
                pan = clamp(pan - step)
                last_action = f'PAN  -{step:.0f}° -> {pan:.1f}°'
            elif key == 'UP':
                tilt = clamp(tilt + step)
                last_action = f'TILT +{step:.0f}° -> {tilt:.1f}°'
            elif key == 'DOWN':
                tilt = clamp(tilt - step)
                last_action = f'TILT -{step:.0f}° -> {tilt:.1f}°'
            elif key in ('f', 'F'):
                fine = not fine
                last_action = f'step -> {"FINE" if fine else "COARSE"}'
            elif key in ('n', 'N'):
                pan = tilt = NEUTRAL
                last_action = 'both -> neutral (90°)'
            elif key in ('s', 'S'):
                print_table(pan, tilt)
                continue
            elif key in ('q', 'Q', '\x03'):
                break
            else:
                continue

            publish()

    finally:
        # Return head to neutral on exit.
        pan = tilt = NEUTRAL
        publish()
        time.sleep(0.3)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print(CLEAR, end='')
        print(f'{BOLD}{GREEN}Head returned to neutral. Test ended.{RESET}')
        print()
        print('Next: record your pan/tilt direction notes, then we wire the')
        print('signs into head_tracking_node (pan_gain / tilt_gain).')
        print()


if __name__ == '__main__':
    main()
