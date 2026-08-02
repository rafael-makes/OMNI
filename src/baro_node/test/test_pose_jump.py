"""Sanity guard for pose_writer — a big pose jump vs. previously persisted must not
be persisted (would poison last_pose.yaml and every future boot). Pure math, no ROS."""
from baro_node.pose_store import is_pose_jump


def test_small_change_is_not_a_jump():
    assert not is_pose_jump((0.0, 0.0, 0.0), (0.1, 0.1, 3.0), 1.0, 45.0)


def test_large_distance_is_a_jump():
    assert is_pose_jump((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), 1.0, 45.0)


def test_large_yaw_is_a_jump():
    assert is_pose_jump((0.0, 0.0, 0.0), (0.0, 0.0, 90.0), 1.0, 45.0)


def test_yaw_wraps_at_pm180():
    # 179° -> -179° is a 2° change, not 358° — must NOT count as a jump.
    assert not is_pose_jump((0.0, 0.0, 179.0), (0.0, 0.0, -179.0), 1.0, 45.0)
    assert not is_pose_jump((0.0, 0.0, -179.0), (0.0, 0.0, 179.0), 1.0, 45.0)


def test_exact_scenario_from_2026_08_02():
    # The failure: pose_writer persisted (0.399, -0.14, -13°) while the truth
    # was ~(0.49, -1.96, 92°) — a 1.83 m / 105° jump from a good previous.
    # With defaults (1.0 m / 45°) this must trip the guard.
    assert is_pose_jump(
        (0.4912, -1.9565, 91.6), (0.399, -0.14, -13.0), 1.0, 45.0)


def test_both_within_thresholds_is_not_a_jump():
    assert not is_pose_jump((0.0, 0.0, 0.0), (0.5, 0.5, 30.0), 1.0, 45.0)


def test_zero_thresholds_reject_any_change():
    assert is_pose_jump((0.0, 0.0, 0.0), (0.001, 0.0, 0.0), 0.0, 0.0)
