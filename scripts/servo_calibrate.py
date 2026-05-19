#!/usr/bin/env python3
"""
OMNI Servo Neutral-Point Calibration Tool
------------------------------------------
Presents each servo one at a time. Move the joint visually to its
physical center (neutral), then press 'n' to advance to the next servo.
All neutral points are saved to ~/omni_ws/config/servo_config.yaml.

Controls
--------
  Arrow Up  /  +  /  e   Increase pulse width (coarse: 50 µs, fine: 10 µs)
  Arrow Down / -  /  d   Decrease pulse width (coarse: 50 µs, fine: 10 µs)
  f                       Toggle fine / coarse step size
  r                       Reset to 1500 µs center
  n                       Accept & next servo
  p                       Go back to previous servo
  s                       Save current state to file (mid-session checkpoint)
  q                       Save everything and quit
"""

import os
import sys
import termios
import time
import tty

import board
import busio
import yaml
from adafruit_pca9685 import PCA9685

CONFIG_PATH = os.path.expanduser('~/omni_ws/config/servo_config.yaml')
FREQ_HZ     = 50
PERIOD_US   = 1_000_000 / FREQ_HZ  # 20 000 µs

SERVO_ORDER = [
    'left_shoulder',
    'left_elbow',
    'left_wrist',
    'left_gripper',
    'head_pan',
    'head_tilt',
    'right_shoulder',
    'right_elbow',
    'right_wrist',
    'right_gripper',
]

SERVO_LABELS = {
    'left_shoulder':  'Left Shoulder  (board 0, ch 0) — 40kg HV',
    'left_elbow':     'Left Elbow     (board 0, ch 1) — 40kg HV',
    'left_wrist':     'Left Wrist     (board 0, ch 2) — MG995',
    'left_gripper':   'Left Gripper   (board 0, ch 3) — 6221MG',
    'head_pan':       'Head Pan       (board 0, ch 4) — MG995',
    'head_tilt':      'Head Tilt      (board 0, ch 5) — 6221MG',
    'right_shoulder': 'Right Shoulder (board 1, ch 0) — 40kg HV',
    'right_elbow':    'Right Elbow    (board 1, ch 1) — 40kg HV',
    'right_wrist':    'Right Wrist    (board 1, ch 2) — MG995',
    'right_gripper':  'Right Gripper  (board 1, ch 3) — 6221MG',
}


# ─────────────────────────────── hardware ────────────────────────────────────

def init_boards():
    i2c  = busio.I2C(board.SCL, board.SDA)
    pca0 = PCA9685(i2c, address=0x40)
    pca0.frequency = FREQ_HZ
    pca1 = PCA9685(i2c, address=0x41)
    pca1.frequency = FREQ_HZ
    return {0: pca0, 1: pca1}


def pw_to_duty(pw_us: float) -> int:
    return int(pw_us / PERIOD_US * 65_535)


def set_pw(boards: dict, board_id: int, channel: int, pw_us: float):
    boards[board_id].channels[channel].duty_cycle = pw_to_duty(pw_us)


def release_all(boards: dict):
    """Set all channels to 0 (no PWM signal) before exit."""
    for pca in boards.values():
        for ch in pca.channels:
            ch.duty_cycle = 0


# ─────────────────────────────── config I/O ──────────────────────────────────

def load_config() -> dict:
    try:
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def save_config(data: dict):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# ─────────────────────────────── keyboard ────────────────────────────────────

def getch() -> str:
    """Read one keypress, return string. Arrow keys returned as 'UP'/'DOWN'."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == '\x1b':                        # escape sequence
            seq = sys.stdin.read(2)
            if seq == '[A':
                return 'UP'
            if seq == '[B':
                return 'DOWN'
            return 'ESC'
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


def draw(servo_name: str, idx: int, total: int, pw: float, fine: bool, cfg: dict):
    min_pw  = cfg.get('min_pw',  500)
    max_pw  = cfg.get('max_pw', 2500)
    step    = 10 if fine else 50
    bar_len = 40
    frac    = (pw - min_pw) / max(max_pw - min_pw, 1)
    filled  = int(frac * bar_len)
    bar     = '█' * filled + '░' * (bar_len - filled)

    pct_of_range = frac * 100.0

    print(CLEAR, end='')
    print(f'{BOLD}{CYAN}═══ OMNI Servo Calibration ═══{RESET}')
    print()
    print(f'  Servo {idx + 1} / {total}')
    print(f'  {BOLD}{SERVO_LABELS.get(servo_name, servo_name)}{RESET}')
    print()
    print(f'  Pulse width : {BOLD}{YELL}{pw:.0f} µs{RESET}')
    print(f'  Range       : {min_pw} – {max_pw} µs')
    print(f'  Position    : {pct_of_range:.1f}% of range')
    print()
    print(f'  [{bar}]')
    print()
    step_label = f'{CYAN}FINE{RESET} ({step} µs)' if fine else f'{YELL}COARSE{RESET} ({step} µs)'
    print(f'  Step size   : {step_label}  (press f to toggle)')
    print()
    print(f'  {GREEN}↑ / + / e{RESET}  Increase    {GREEN}↓ / - / d{RESET}  Decrease')
    print(f'  {GREEN}r{RESET}          Reset 1500   {GREEN}f{RESET}          Toggle step')
    print(f'  {GREEN}n{RESET}          Next servo   {GREEN}p{RESET}          Prev servo')
    print(f'  {GREEN}s{RESET}          Save now     {GREEN}q{RESET}          Save & quit')
    print()
    print('  Move the joint to its PHYSICAL CENTER, then press n.')


# ─────────────────────────────── main ────────────────────────────────────────

def main():
    print('Loading config...')
    config = load_config()
    servos = config.get('servos', {})

    # Ensure all expected servos exist in config with defaults
    defaults = {
        'left_shoulder':  {'board': 0, 'channel': 0, 'neutral_pw': 1500, 'min_pw':  500, 'max_pw': 2500},
        'left_elbow':     {'board': 0, 'channel': 1, 'neutral_pw': 1500, 'min_pw':  500, 'max_pw': 2500},
        'left_wrist':     {'board': 0, 'channel': 2, 'neutral_pw': 1500, 'min_pw':  500, 'max_pw': 2500},
        'left_gripper':   {'board': 0, 'channel': 3, 'neutral_pw': 1500, 'min_pw':  700, 'max_pw': 2300},
        'head_pan':       {'board': 0, 'channel': 4, 'neutral_pw': 1500, 'min_pw':  500, 'max_pw': 2500},
        'head_tilt':      {'board': 0, 'channel': 5, 'neutral_pw': 1500, 'min_pw':  700, 'max_pw': 2300},
        'right_shoulder': {'board': 1, 'channel': 0, 'neutral_pw': 1500, 'min_pw':  500, 'max_pw': 2500},
        'right_elbow':    {'board': 1, 'channel': 1, 'neutral_pw': 1500, 'min_pw':  500, 'max_pw': 2500},
        'right_wrist':    {'board': 1, 'channel': 2, 'neutral_pw': 1500, 'min_pw':  500, 'max_pw': 2500},
        'right_gripper':  {'board': 1, 'channel': 3, 'neutral_pw': 1500, 'min_pw':  700, 'max_pw': 2300},
    }
    for name, dflt in defaults.items():
        if name not in servos:
            servos[name] = dflt.copy()

    print('Connecting to PCA9685 boards...')
    try:
        boards = init_boards()
    except Exception as e:
        print(f'ERROR: Could not connect to PCA9685: {e}')
        sys.exit(1)

    print('Hardware ready.')
    time.sleep(0.5)

    # Working pulse widths — start from current neutral_pw values
    current_pw = {name: float(cfg['neutral_pw']) for name, cfg in servos.items()}
    fine = False
    idx  = 0

    try:
        while True:
            name = SERVO_ORDER[idx]
            cfg  = servos[name]
            pw   = current_pw[name]

            # Move servo to current pw
            try:
                set_pw(boards, cfg['board'], cfg['channel'], pw)
            except Exception as e:
                print(f'\nI2C error: {e}')
                time.sleep(0.5)

            draw(name, idx, len(SERVO_ORDER), pw, fine, cfg)

            key = getch()

            if key in ('UP', '+', 'e', 'E'):
                step = 10 if fine else 50
                pw   = min(cfg['max_pw'], pw + step)
            elif key in ('DOWN', '-', 'd', 'D'):
                step = 10 if fine else 50
                pw   = max(cfg['min_pw'], pw - step)
            elif key in ('f', 'F'):
                fine = not fine
            elif key in ('r', 'R'):
                pw = 1500.0
            elif key in ('n', 'N', '\r', '\n'):
                servos[name]['neutral_pw'] = int(pw)
                current_pw[name] = pw
                idx += 1
                if idx >= len(SERVO_ORDER):
                    # All done
                    break
                # Silence current servo before moving to next
                set_pw(boards, cfg['board'], cfg['channel'], pw)
                time.sleep(0.1)
            elif key in ('p', 'P'):
                servos[name]['neutral_pw'] = int(pw)
                current_pw[name] = pw
                idx = max(0, idx - 1)
            elif key in ('s', 'S'):
                servos[name]['neutral_pw'] = int(pw)
                config['servos'] = servos
                save_config(config)
                print('\n  Saved!', end='', flush=True)
                time.sleep(0.8)
            elif key in ('q', 'Q', '\x03'):
                servos[name]['neutral_pw'] = int(pw)
                break

            current_pw[name] = pw

    except Exception as e:
        print(f'\nUnexpected error: {e}')
    finally:
        release_all(boards)

    # Final save
    config['servos'] = servos
    save_config(config)

    print(CLEAR, end='')
    print(f'{BOLD}{GREEN}Calibration complete — saved to:{RESET}')
    print(f'  {CONFIG_PATH}')
    print()
    print('Neutral points:')
    for name in SERVO_ORDER:
        pw = servos[name]['neutral_pw']
        print(f'  {name:<18} {pw} µs')
    print()


if __name__ == '__main__':
    main()
