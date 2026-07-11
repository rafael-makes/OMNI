"""Shared floor-map configuration for the barometer floor detector.

Stored at ~/omni_ws/maps/floors.yaml (next to the maps, NOT inside the package,
so it survives colcon rebuilds and is writable by the calibration tool).

Format:
    floors:
      - floor: 0            # integer floor index (basement = -1, etc.)
        name: main
        map: /home/pi/omni_ws/maps/omni_home_map
        pressure: 101004.0  # calibrated mean pressure in Pa
        stddev: 0.6         # measured noise from calibration
"""
import os
import yaml

FLOORS_PATH = os.path.expanduser('~/omni_ws/maps/floors.yaml')


def load_floors():
    if not os.path.exists(FLOORS_PATH):
        return []
    with open(FLOORS_PATH) as f:
        data = yaml.safe_load(f) or {}
    return data.get('floors', [])


def save_floors(floors):
    os.makedirs(os.path.dirname(FLOORS_PATH), exist_ok=True)
    # Keep the file sorted by pressure (highest = lowest floor) for readability.
    floors = sorted(floors, key=lambda f: -f['pressure'])
    with open(FLOORS_PATH, 'w') as f:
        yaml.safe_dump({'floors': floors}, f, sort_keys=False)
