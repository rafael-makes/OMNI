"""Fault-handling tests for SafetyNode — the two 2026-07-31 fixes:

  1. _proximity_cb ignores INVALID ToF readings (out of [min_range, max_range]),
     so a battery-sag glitch (~0.0) can't trip a false proximity fault.
  2. /safety/clear_fault accepts "proximity"/"all" to reset a wedged proximity
     fault (stale frame_id stuck in _tof_close).

Instantiates the real node and calls the callbacks directly — no spinning.
"""
import pytest
import rclpy
from sensor_msgs.msg import Range
from std_msgs.msg import String

from safety_node.safety_node import SafetyNode


@pytest.fixture(scope='module', autouse=True)
def _ros():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def node():
    n = SafetyNode()
    yield n
    n.destroy_node()


def _range(frame, r, min_r=0.03, max_r=1.2):
    m = Range()
    m.header.frame_id = frame
    m.min_range = min_r
    m.max_range = max_r
    m.range = r
    return m


def test_valid_close_reading_sets_proximity(node):
    node._proximity_cb(_range('tof_left', 0.05))   # in range, < 0.075
    assert node.fault_proximity is True
    assert 'tof_left' in node._tof_close


def test_valid_clear_reading_releases_proximity(node):
    node._proximity_cb(_range('tof_left', 0.05))
    assert node.fault_proximity is True
    node._proximity_cb(_range('tof_left', 0.50))   # in range, >= 0.075
    assert node.fault_proximity is False
    assert node._tof_close == set()


def test_glitch_below_min_range_is_ignored(node):
    # 0.0 with min_range 0.03 is the classic VL53L0X sag glitch — must NOT fault.
    node._proximity_cb(_range('tof_left', 0.0))
    assert node.fault_proximity is False
    assert node._tof_close == set()


def test_reading_above_max_range_is_ignored(node):
    node._proximity_cb(_range('tof_left', 8.0))
    assert node.fault_proximity is False
    assert node._tof_close == set()


def test_open_space_at_max_range_clears_normally(node):
    node._proximity_cb(_range('tof_left', 0.05))   # close
    node._proximity_cb(_range('tof_left', 1.2))    # open space = max_range (valid)
    assert node.fault_proximity is False
    assert node._tof_close == set()


def test_clear_fault_proximity_unwedges_stuck_set(node):
    # Simulate a wedged phantom entry (the bug we hit on hardware 2026-07-31).
    node._tof_close.add('phantom_frame')
    node.fault_proximity = True
    node._clear_fault_cb(String(data='proximity'))
    assert node.fault_proximity is False
    assert node._tof_close == set()


def test_clear_fault_all_resets_everything(node):
    node.fault_stall = True
    node.fault_estop = True
    node.fault_proximity = True
    node.fault_watchdog = True
    node._tof_close.add('x')
    node._clear_fault_cb(String(data='all'))
    assert not (node.fault_stall or node.fault_estop
                or node.fault_proximity or node.fault_watchdog)
    assert node._tof_close == set()
