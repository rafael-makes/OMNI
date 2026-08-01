"""Persisted-pose storage for boot-time self-localization.

Two things live here, both ROS-free so they can be unit-tested with the robot
off (test_pose_store.py):

  1. The LAST-KNOWN pose, written periodically while OMNI is localized and read
     back at the next boot as the /initialpose prior. slam_toolbox has no global
     localization — its localization mode scan-matches from a prior — but OMNI is
     wheeled and boots roughly where it was left, and the localization config's
     correlation_search_space_dimension (1.5 m => ±0.75 m window) then snaps the
     seeded pose to the true one. So "last pose + scan-match refine" recovers the
     pose without a manual Foxglove 2D Pose Estimate. Keyed by FLOOR, because the
     map (and therefore the frame) is per-floor.
       -> ~/omni_ws/maps/last_pose_floor<N>.yaml

  2. The DOCKED pose — where map->base_link sits when OMNI is physically on the
     dock (NOT the pre-dock standoff 'dock' location in omni_config.yaml). When a
     boot sees the dock tag AND the rear ToF reads docking distance, this is the
     exact, drift-free pose to publish. Stored per floor inside floors.yaml
     (alongside the map path) so all the per-floor facts stay in one file.

Pose format everywhere: [x, y, yaw_deg] — the SAME triple save_location uses in
omni_config.yaml, so it reads the same to a human.
"""
import os
import time

import yaml

from baro_node.floors import FLOORS_PATH, load_floors, save_floors

MAPS_DIR = os.path.dirname(FLOORS_PATH)


# ── floor <-> map ────────────────────────────────────────────────────────────
def floor_for_map(map_path, floors=None):
    """The floor index whose map matches map_path, or None.

    Matches on the map prefix so '/.../omni_home_map_v2' resolves whether or not
    a trailing extension is present (slam_toolbox is given the prefix).
    """
    if not map_path:
        return None
    if floors is None:
        floors = load_floors()
    target = os.path.splitext(map_path)[0]
    for f in floors:
        entry = os.path.splitext(str(f.get('map', '')))[0]
        if entry and (entry == target or target.startswith(entry)):
            return f['floor']
    return None


# ── last-known pose (per floor, own file) ────────────────────────────────────
def _last_pose_path(floor):
    return os.path.join(MAPS_DIR, f'last_pose_floor{floor}.yaml')


def write_last_pose(floor, pose):
    """Persist [x, y, yaw_deg] for a floor. Atomic (write-temp + rename)."""
    path = _last_pose_path(floor)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        'floor': int(floor),
        'pose': [round(float(pose[0]), 4),
                 round(float(pose[1]), 4),
                 round(float(pose[2]), 2)],
        'stamp': round(time.time(), 1),
    }
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    os.replace(tmp, path)


def read_last_pose(floor):
    """(pose, stamp) for a floor, or (None, None) if no file / malformed."""
    path = _last_pose_path(floor)
    if not os.path.exists(path):
        return None, None
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        pose = data.get('pose')
        if isinstance(pose, (list, tuple)) and len(pose) == 3:
            return [float(pose[0]), float(pose[1]), float(pose[2])], data.get('stamp')
    except Exception:  # noqa: BLE001 — a corrupt file must not break boot
        pass
    return None, None


# ── docked pose (per floor, inside floors.yaml) ──────────────────────────────
def read_dock_pose(floor, floors=None):
    """The saved docked pose [x, y, yaw_deg] for a floor, or None."""
    if floors is None:
        floors = load_floors()
    for f in floors:
        if f.get('floor') == floor:
            pose = f.get('dock_pose')
            if isinstance(pose, (list, tuple)) and len(pose) == 3:
                return [float(pose[0]), float(pose[1]), float(pose[2])]
            return None
    return None


def write_dock_pose(floor, pose):
    """Record the docked pose for a floor inside floors.yaml. Returns False if the
    floor is not in floors.yaml (nothing to attach it to)."""
    floors = load_floors()
    hit = next((f for f in floors if f.get('floor') == floor), None)
    if hit is None:
        return False
    hit['dock_pose'] = [round(float(pose[0]), 4),
                        round(float(pose[1]), 4),
                        round(float(pose[2]), 2)]
    save_floors(floors)
    return True
