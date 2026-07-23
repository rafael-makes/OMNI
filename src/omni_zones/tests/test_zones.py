"""Zone / ZoneMap / config-loading tests. Pure Python, robot off."""

import math

import pytest

from omni_zones.zones import (
    Zone,
    ZoneMap,
    load_zone_map,
    point_in_polygon,
    polygon_centroid,
    zone_from_dict,
)

# A 4x4 square from (0,0) to (4,4), and an L-shaped room to prove concave works.
SQUARE = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
L_SHAPE = [(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (2.0, 2.0), (2.0, 4.0), (0.0, 4.0)]


class TestPointInPolygon:
    def test_center_is_inside(self):
        assert point_in_polygon(2.0, 2.0, SQUARE)

    def test_outside_is_outside(self):
        assert not point_in_polygon(5.0, 5.0, SQUARE)
        assert not point_in_polygon(-1.0, 2.0, SQUARE)

    def test_concave_notch_is_outside(self):
        # The notch of the L (top-right quadrant) is outside the room.
        assert not point_in_polygon(3.0, 3.0, L_SHAPE)
        # ...but the two legs are inside.
        assert point_in_polygon(3.0, 1.0, L_SHAPE)
        assert point_in_polygon(1.0, 3.0, L_SHAPE)

    def test_degenerate_polygon_contains_nothing(self):
        assert not point_in_polygon(0.0, 0.0, [(0.0, 0.0), (1.0, 1.0)])


class TestCentroid:
    def test_square_centroid(self):
        cx, cy = polygon_centroid(SQUARE)
        assert cx == pytest.approx(2.0)
        assert cy == pytest.approx(2.0)

    def test_centroid_of_two_points_is_mean(self):
        cx, cy = polygon_centroid([(0.0, 0.0), (2.0, 4.0)])
        assert (cx, cy) == pytest.approx((1.0, 2.0))


class TestZone:
    def test_requires_three_vertices(self):
        with pytest.raises(ValueError):
            Zone(name="bad", polygon=((0.0, 0.0), (1.0, 1.0)))

    def test_requires_name(self):
        with pytest.raises(ValueError):
            Zone(name="", polygon=tuple(SQUARE))

    def test_nav_pose_uses_anchor_when_set(self):
        z = Zone(name="kitchen", polygon=tuple(SQUARE), anchor=(1.0, 1.0, 90.0))
        assert z.nav_pose == (1.0, 1.0, 90.0)

    def test_nav_pose_falls_back_to_centroid(self):
        z = Zone(name="kitchen", polygon=tuple(SQUARE))
        x, y, yaw = z.nav_pose
        assert (x, y) == pytest.approx((2.0, 2.0))
        assert yaw == 0.0


class TestZoneMap:
    def _map(self):
        return ZoneMap([
            Zone(name="kitchen", polygon=tuple(SQUARE)),
            Zone(name="hall", polygon=((4.0, 0.0), (8.0, 0.0), (8.0, 4.0), (4.0, 4.0))),
        ])

    def test_zone_at_picks_the_right_room(self):
        m = self._map()
        assert m.zone_at(2.0, 2.0) == "kitchen"
        assert m.zone_at(6.0, 2.0) == "hall"

    def test_zone_at_returns_none_outside_all(self):
        assert self._map().zone_at(20.0, 20.0) is None

    def test_names_and_get(self):
        m = self._map()
        assert m.names == ["kitchen", "hall"]
        assert m.get("kitchen").name == "kitchen"
        assert m.get("nope") is None

    def test_nav_pose_and_center_lookup(self):
        m = self._map()
        assert m.nav_pose("kitchen")[:2] == pytest.approx((2.0, 2.0))
        assert m.center("hall") == pytest.approx((6.0, 2.0))
        assert m.nav_pose("nope") is None

    def test_duplicate_names_rejected(self):
        with pytest.raises(ValueError):
            ZoneMap([
                Zone(name="kitchen", polygon=tuple(SQUARE)),
                Zone(name="kitchen", polygon=tuple(SQUARE)),
            ])

    def test_empty_map_is_falsy(self):
        m = ZoneMap([])
        assert not m
        assert m.zone_at(0.0, 0.0) is None
        assert len(m) == 0


class TestConfigLoading:
    def test_load_from_dict(self):
        cfg = {
            "kitchen": {
                "polygon": [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]],
                "anchor": [1.0, 1.0, 90.0],
            },
        }
        m = load_zone_map(cfg)
        assert m.names == ["kitchen"]
        assert m.nav_pose("kitchen") == (1.0, 1.0, 90.0)
        assert m.zone_at(2.0, 2.0) == "kitchen"

    def test_anchor_optional(self):
        z = zone_from_dict("den", {"polygon": SQUARE})
        assert z.anchor is None

    def test_anchor_without_yaw_defaults_zero(self):
        z = zone_from_dict("den", {"polygon": SQUARE, "anchor": [1.0, 2.0]})
        assert z.anchor == (1.0, 2.0, 0.0)

    def test_missing_polygon_raises(self):
        with pytest.raises(ValueError):
            zone_from_dict("den", {"anchor": [0.0, 0.0]})

    def test_empty_config_yields_empty_map(self):
        assert len(load_zone_map(None)) == 0
        assert len(load_zone_map({})) == 0


class TestCheckInSide:
    """Session 9: the fixed per-room approach side."""

    def test_absent_by_default(self):
        z = zone_from_dict("kitchen", {"polygon": [[0, 0], [1, 0], [1, 1]]})
        assert z.check_in_side is None

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_parsed_from_config(self, side):
        z = zone_from_dict(
            "workbench", {"polygon": [[0, 0], [1, 0], [1, 1]], "check_in_side": side})
        assert z.check_in_side == side

    def test_case_and_whitespace_tolerated(self):
        z = zone_from_dict(
            "workbench", {"polygon": [[0, 0], [1, 0], [1, 1]], "check_in_side": " LEFT "})
        assert z.check_in_side == "left"

    def test_nonsense_side_is_rejected_loudly(self):
        """A typo must fail at config load, not silently send OMNI to the wrong
        side of someone for months."""
        with pytest.raises(ValueError):
            zone_from_dict(
                "workbench",
                {"polygon": [[0, 0], [1, 0], [1, 1]], "check_in_side": "port"})

    def test_survives_a_full_map_load(self):
        zmap = load_zone_map({
            "workbench": {"polygon": [[0, 0], [1, 0], [1, 1]], "check_in_side": "right"},
            "kitchen": {"polygon": [[2, 2], [3, 2], [3, 3]]},
        })
        assert zmap.get("workbench").check_in_side == "right"
        assert zmap.get("kitchen").check_in_side is None
