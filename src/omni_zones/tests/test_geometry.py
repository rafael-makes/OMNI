"""Geometry tests: bearing, person-position estimate, standoff. Robot off."""

import math

import pytest

from omni_zones.geometry import (
    bearing_from_bbox,
    estimate_person_xy,
    standoff_pose,
)


class TestBearing:
    def test_center_is_zero(self):
        assert bearing_from_bbox(960.0, 1920.0, math.radians(66.0)) == pytest.approx(0.0)

    def test_left_of_center_is_positive(self):
        # Column left of centre -> person to the robot's left -> +bearing (CCW).
        b = bearing_from_bbox(480.0, 1920.0, math.radians(66.0))
        assert b > 0
        # Quarter-frame left = half of half-FOV.
        assert b == pytest.approx(math.radians(66.0) * 0.25)

    def test_right_of_center_is_negative(self):
        assert bearing_from_bbox(1440.0, 1920.0, math.radians(66.0)) < 0

    def test_unknown_width_returns_zero(self):
        assert bearing_from_bbox(500.0, 0.0, math.radians(66.0)) == 0.0


class TestEstimatePersonXY:
    def test_straight_ahead(self):
        # Robot at origin facing +x (yaw 0), head cam (offset 0), person centred,
        # 2 m away -> person at (2, 0).
        x, y = estimate_person_xy(0.0, 0.0, 0.0, 0.0, 0.0, 2.0)
        assert (x, y) == pytest.approx((2.0, 0.0))

    def test_rear_camera_projects_behind(self):
        # Rear cam offset = pi. Robot facing +x, rear cam sees straight back ->
        # person at -x.
        x, y = estimate_person_xy(0.0, 0.0, 0.0, math.pi, 0.0, 2.0)
        assert (x, y) == pytest.approx((-2.0, 0.0), abs=1e-9)

    def test_robot_yaw_rotates_estimate(self):
        # Robot at origin facing +y (yaw 90), person centred, 3 m -> (0, 3).
        x, y = estimate_person_xy(0.0, 0.0, math.radians(90.0), 0.0, 0.0, 3.0)
        assert (x, y) == pytest.approx((0.0, 3.0), abs=1e-9)

    def test_bearing_adds_to_heading(self):
        # Facing +x, +90 deg bearing (person to the left) -> +y.
        x, y = estimate_person_xy(0.0, 0.0, 0.0, 0.0, math.radians(90.0), 1.0)
        assert (x, y) == pytest.approx((0.0, 1.0), abs=1e-9)

    def test_translation_offset_applied(self):
        x, y = estimate_person_xy(5.0, -3.0, 0.0, 0.0, 0.0, 2.0)
        assert (x, y) == pytest.approx((7.0, -3.0))


class TestStandoffPose:
    def test_stops_short_and_faces_person(self):
        # Robot at origin, person 5 m ahead on +x, standoff 1 m -> goal at (4,0)
        # facing +x (yaw 0).
        gx, gy, yaw = standoff_pose((5.0, 0.0), (0.0, 0.0), standoff=1.0)
        assert (gx, gy) == pytest.approx((4.0, 0.0))
        assert yaw == pytest.approx(0.0)

    def test_faces_person_to_the_side(self):
        gx, gy, yaw = standoff_pose((0.0, 5.0), (0.0, 0.0), standoff=1.0)
        assert (gx, gy) == pytest.approx((0.0, 4.0))
        assert yaw == pytest.approx(90.0)

    def test_already_within_standoff_stays_put(self):
        # Person only 0.5 m away, standoff 1 m -> do not back up, just face them.
        gx, gy, yaw = standoff_pose((0.5, 0.0), (0.0, 0.0), standoff=1.0)
        assert (gx, gy) == pytest.approx((0.0, 0.0))
        assert yaw == pytest.approx(0.0)

    def test_coincident_points_do_not_divide_by_zero(self):
        gx, gy, yaw = standoff_pose((1.0, 1.0), (1.0, 1.0), standoff=1.0)
        assert (gx, gy) == pytest.approx((1.0, 1.0))
        assert yaw == 0.0

    def test_goal_is_standoff_distance_from_person(self):
        gx, gy, yaw = standoff_pose((3.0, 4.0), (0.0, 0.0), standoff=1.5)
        d = math.hypot(3.0 - gx, 4.0 - gy)
        assert d == pytest.approx(1.5)


class TestLateralStandoff:
    """The Session 9 side offset: arrive BESIDE someone working, not in front."""

    def test_zero_lateral_is_the_session_7_behaviour(self):
        """Every existing caller passes no lateral — it must be bit-identical."""
        assert (standoff_pose((5.0, 0.0), (0.0, 0.0), standoff=1.0, lateral=0.0)
                == pytest.approx(standoff_pose((5.0, 0.0), (0.0, 0.0), standoff=1.0)))

    def test_positive_lateral_goes_to_the_left_of_the_approach(self):
        # Approaching along +x, the left-hand normal is +y.
        gx, gy, _ = standoff_pose((5.0, 0.0), (0.0, 0.0), standoff=1.0, lateral=0.6)
        assert (gx, gy) == pytest.approx((4.0, 0.6))

    def test_negative_lateral_goes_to_the_right(self):
        gx, gy, _ = standoff_pose((5.0, 0.0), (0.0, 0.0), standoff=1.0, lateral=-0.6)
        assert (gx, gy) == pytest.approx((4.0, -0.6))

    def test_lateral_is_relative_to_the_approach_direction(self):
        # Approaching along +y instead: the left-hand normal is now -x.
        gx, gy, _ = standoff_pose((0.0, 5.0), (0.0, 0.0), standoff=1.0, lateral=0.6)
        assert (gx, gy) == pytest.approx((-0.6, 4.0))

    def test_still_faces_the_person_from_the_offset_goal(self):
        """The point of recomputing yaw at the goal: standing to one side and
        still looking at them, rather than staring straight past them."""
        px, py = 5.0, 0.0
        gx, gy, yaw = standoff_pose((px, py), (0.0, 0.0), standoff=1.0, lateral=1.0)
        expected = math.degrees(math.atan2(py - gy, px - gx))
        assert yaw == pytest.approx(expected)
        # Offset left of an eastward approach → must now look south of east.
        assert yaw < 0.0

    def test_offset_goal_is_further_than_standoff_but_bounded(self):
        """Sliding sideways necessarily increases range to the person; it must be
        the hypotenuse, not something unbounded."""
        gx, gy, _ = standoff_pose((5.0, 0.0), (0.0, 0.0), standoff=1.0, lateral=0.6)
        d = math.hypot(5.0 - gx, 0.0 - gy)
        assert d == pytest.approx(math.hypot(1.0, 0.6))

    def test_lateral_applies_even_when_already_too_close(self):
        """Already inside the standoff radius: do not drive at them, but do step
        aside — "stop looming over them" is exactly right here."""
        gx, gy, _ = standoff_pose((0.5, 0.0), (0.0, 0.0), standoff=1.0, lateral=0.6)
        assert (gx, gy) == pytest.approx((0.0, 0.6))

    def test_coincident_points_ignore_lateral_safely(self):
        """No approach direction exists, so there is no side to step to."""
        gx, gy, yaw = standoff_pose((1.0, 1.0), (1.0, 1.0), standoff=1.0, lateral=0.6)
        assert (gx, gy) == pytest.approx((1.0, 1.0))
        assert yaw == 0.0
