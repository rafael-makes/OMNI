"""Boot-time map selector.

Reads the BMP280, matches the current pressure to the nearest calibrated floor
(from floors.yaml, built with baro_calibrate), and resolves which SLAM map to
load. With --launch it hands straight off to the localization launch:

    ros2 run baro_node baro_select_map --launch

Without --launch it just prints the chosen map path (useful for scripting or a
dry run):

    MAP=$(ros2 run baro_node baro_select_map --quiet)

Note: this uses ABSOLUTE calibrated pressure, which drifts with weather. When the
current pressure sits ambiguously between two floors it says so — the AprilTag
dock is the authoritative floor source and should call /baro/set_floor to
re-anchor once OMNI is docked. See feedback_baro_node / docking plan.
"""
import argparse
import os
import statistics
import sys
import time

import smbus2

from baro_node.baro_node import BMP280
from baro_node.floors import load_floors


def read_pressure(bus_num, address, seconds):
    bus = smbus2.SMBus(bus_num)
    try:
        sensor = BMP280(bus, address)
        samples = []
        deadline = time.time() + seconds
        while time.time() < deadline:
            _, p = sensor.read()
            samples.append(p)
            time.sleep(0.1)
    finally:
        bus.close()
    return statistics.fmean(samples)


def choose(floors, pressure):
    """Return (best_floor_entry, margin_pa). margin = gap to 2nd-nearest floor."""
    ranked = sorted(floors, key=lambda f: abs(f['pressure'] - pressure))
    best = ranked[0]
    margin = (abs(ranked[1]['pressure'] - pressure) - abs(best['pressure'] - pressure)
              if len(ranked) > 1 else float('inf'))
    return best, margin


def main():
    ap = argparse.ArgumentParser(description='Boot-time barometric map selector')
    ap.add_argument('--seconds', type=float, default=3.0, help='sampling duration')
    ap.add_argument('--bus', type=int, default=1)
    ap.add_argument('--addr', type=lambda x: int(x, 0), default=0x77)
    ap.add_argument('--launch', action='store_true',
                    help='exec the localization launch with the chosen map')
    ap.add_argument('--quiet', action='store_true',
                    help='print only the resolved map path (for scripting)')
    args = ap.parse_args()

    floors = load_floors()
    if not floors:
        print('No calibrated floors — run baro_calibrate first.', file=sys.stderr)
        sys.exit(2)

    pressure = read_pressure(args.bus, args.addr, args.seconds)
    best, margin = choose(floors, pressure)

    if not args.quiet:
        print(f'Current pressure: {pressure:.1f} Pa')
        print(f'Selected floor {best["floor"]} ("{best["name"]}") '
              f'@ {best["pressure"]:.1f} Pa  (margin {margin:.1f} Pa to next floor)')
        if margin < 20.0:
            print('  WARNING: low margin — pressure is ambiguous between floors '
                  '(weather drift?). Confirm via the dock AprilTag.', file=sys.stderr)

    map_path = best.get('map')
    if not map_path:
        print(f'Floor {best["floor"]} has no map path in floors.yaml.',
              file=sys.stderr)
        sys.exit(3)

    if args.launch:
        cmd = ['ros2', 'launch', 'slam_node', 'localization_launch.py',
               f'map_file:={map_path}']
        if not args.quiet:
            print(f'Launching: {" ".join(cmd)}')
        os.execvp('ros2', cmd)  # replaces this process
    else:
        # machine-readable: the map path on stdout
        print(map_path)


if __name__ == '__main__':
    main()
