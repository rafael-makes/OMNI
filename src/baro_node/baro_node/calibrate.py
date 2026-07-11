"""Barometric floor calibration.

Run this ONCE PER FLOOR, parked where OMNI will boot/dock. It samples the BMP280
for a few seconds, records the mean pressure for that floor, and reports whether
the floors are far enough apart in pressure to be told apart reliably.

Examples:
    # record the current spot as floor 0 ("main"), mapped to the home map
    ros2 run baro_node baro_calibrate --floor 0 --name main \
        --map /home/pi/omni_ws/maps/omni_home_map

    # then carry OMNI to the basement and record floor -1
    ros2 run baro_node baro_calibrate --floor -1 --name basement \
        --map /home/pi/omni_ws/maps/omni_basement_map

    # just show what's on file without recording
    ros2 run baro_node baro_calibrate --show
"""
import argparse
import statistics
import time

import smbus2

from baro_node.baro_node import BMP280, pressure_to_altitude
from baro_node.floors import load_floors, save_floors

# Rough short-term (hours) weather drift budget in Pa. If two floors are closer
# than this in pressure, absolute boot-time selection between them is unreliable
# and you should lean on the dock AprilTag to disambiguate.
DRIFT_BUDGET_PA = 40.0
# ~12 Pa per metre near sea level.
PA_PER_M = 12.0


def sample(bus_num, address, seconds):
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
    return samples


def report(floors):
    if not floors:
        print('No floors calibrated yet.')
        return
    print(f'\nCalibrated floors ({len(floors)}):')
    print(f'  {"floor":>5}  {"name":<10} {"pressure(Pa)":>13} {"noise":>7}  map')
    for f in floors:
        print(f'  {f["floor"]:>5}  {f["name"]:<10} {f["pressure"]:>13.1f} '
              f'{f.get("stddev", 0.0):>6.2f}   {f.get("map", "-")}')

    if len(floors) < 2:
        print('\nNeed at least 2 floors to check separation.')
        return

    print('\nPairwise separation:')
    ordered = sorted(floors, key=lambda f: -f['pressure'])
    for a, b in zip(ordered, ordered[1:]):
        dp = abs(a['pressure'] - b['pressure'])
        dm = dp / PA_PER_M
        noise = max(a.get('stddev', 0.0), b.get('stddev', 0.0), 0.01)
        snr = dp / noise
        verdict = 'OK' if dp >= DRIFT_BUDGET_PA else 'TIGHT — rely on dock tag'
        print(f'  {a["name"]:<10} <-> {b["name"]:<10} '
              f'Δ{dp:6.1f} Pa (~{dm:4.1f} m, {snr:5.0f}× noise)  [{verdict}]')
    print(f'\n(Separation ≥ {DRIFT_BUDGET_PA:.0f} Pa clears the weather-drift budget; '
          'tighter pairs need the AprilTag dock to confirm the floor.)')


def main():
    ap = argparse.ArgumentParser(description='BMP280 per-floor pressure calibration')
    ap.add_argument('--floor', type=int, help='integer floor index (basement = -1)')
    ap.add_argument('--name', type=str, help='human name, e.g. main / basement')
    ap.add_argument('--map', type=str, help='map path prefix for this floor')
    ap.add_argument('--seconds', type=float, default=5.0, help='sampling duration')
    ap.add_argument('--bus', type=int, default=1)
    ap.add_argument('--addr', type=lambda x: int(x, 0), default=0x77)
    ap.add_argument('--show', action='store_true', help='print floors and exit')
    args = ap.parse_args()

    floors = load_floors()

    if args.show:
        report(floors)
        return

    if args.floor is None or args.name is None:
        ap.error('provide --floor and --name to record (or --show to just view)')

    print(f'Sampling floor {args.floor} ("{args.name}") for {args.seconds:.0f}s — '
          'keep OMNI still...')
    samples = sample(args.bus, args.addr, args.seconds)
    mean = statistics.fmean(samples)
    sd = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    print(f'  n={len(samples)}  mean={mean:.1f} Pa  stddev={sd:.2f} Pa  '
          f'(alt {pressure_to_altitude(mean, 101325.0):.1f} m)')

    entry = {'floor': int(args.floor), 'name': args.name,
             'pressure': round(mean, 1), 'stddev': round(sd, 2)}
    if args.map:
        entry['map'] = args.map
    else:
        # preserve an existing map path if we're re-calibrating
        prev = next((f for f in floors if f['floor'] == args.floor), None)
        if prev and 'map' in prev:
            entry['map'] = prev['map']

    floors = [f for f in floors if f['floor'] != args.floor]
    floors.append(entry)
    save_floors(floors)
    print(f'Saved floor {args.floor} to floors.yaml.')
    report(load_floors())


if __name__ == '__main__':
    main()
