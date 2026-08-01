"""Offline tests for pose_store — no ROS, no real maps dir."""
import os

import pytest

from baro_node import pose_store

FLOORS = [
    {'floor': 0, 'name': 'main', 'map': '/home/pi/omni_ws/maps/omni_home_map_v2',
     'pressure': 100989.0, 'stddev': 0.68, 'dock_pose': [0.1, 0.2, 3.0]},
    {'floor': -1, 'name': 'basement', 'map': '/home/pi/omni_ws/maps/basement'},
]


def test_floor_for_map_matches_prefix_and_extension():
    assert pose_store.floor_for_map(
        '/home/pi/omni_ws/maps/omni_home_map_v2', FLOORS) == 0
    # slam_toolbox is given the prefix; a trailing extension must still match
    assert pose_store.floor_for_map(
        '/home/pi/omni_ws/maps/omni_home_map_v2.posegraph', FLOORS) == 0
    assert pose_store.floor_for_map(
        '/home/pi/omni_ws/maps/basement', FLOORS) == -1


def test_floor_for_map_unknown_is_none():
    assert pose_store.floor_for_map('/nope/other_map', FLOORS) is None
    assert pose_store.floor_for_map('', FLOORS) is None
    # omni_home_map (old) must NOT match omni_home_map_v2 as a substring victim:
    # target startswith(entry) is only used the other way. The old map isn't in
    # FLOORS, so it resolves to None.
    assert pose_store.floor_for_map(
        '/home/pi/omni_ws/maps/omni_home_map', FLOORS) is None


def test_read_dock_pose():
    assert pose_store.read_dock_pose(0, FLOORS) == [0.1, 0.2, 3.0]
    assert pose_store.read_dock_pose(-1, FLOORS) is None   # floor exists, no pose
    assert pose_store.read_dock_pose(99, FLOORS) is None   # floor absent


def test_last_pose_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(pose_store, 'MAPS_DIR', str(tmp_path))
    assert pose_store.read_last_pose(0) == (None, None)   # nothing yet
    pose_store.write_last_pose(0, [1.234567, -2.0, 45.678])
    pose, stamp = pose_store.read_last_pose(0)
    assert pose == pytest.approx([1.2346, -2.0, 45.68], abs=1e-4)
    assert stamp is not None
    # per-floor keying: floor -1 is independent
    assert pose_store.read_last_pose(-1) == (None, None)


def test_last_pose_atomic_no_tmp_left(tmp_path, monkeypatch):
    monkeypatch.setattr(pose_store, 'MAPS_DIR', str(tmp_path))
    pose_store.write_last_pose(0, [0.0, 0.0, 0.0])
    files = os.listdir(tmp_path)
    assert 'last_pose_floor0.yaml' in files
    assert not any(f.endswith('.tmp') for f in files)


def test_read_last_pose_corrupt_is_safe(tmp_path, monkeypatch):
    monkeypatch.setattr(pose_store, 'MAPS_DIR', str(tmp_path))
    (tmp_path / 'last_pose_floor0.yaml').write_text('pose: [1, 2]\n')  # too short
    assert pose_store.read_last_pose(0) == (None, None)
    (tmp_path / 'last_pose_floor0.yaml').write_text(': not : valid : yaml :\n')
    assert pose_store.read_last_pose(0) == (None, None)
