"""Named zones (rooms) as polygons in the map frame. No ROS imports.

A *zone* is a named polygon in the SLAM map frame plus an optional navigation
*anchor* — the pose OMNI drives to when asked to go to that room. Two things read
this library:

* ``behavior_node`` — resolves ``navigate_to("kitchen")`` and ``go_to_person``
  to a goal pose, and needs the polygon of a person's zone.
* ``world_state`` — labels each person track with the zone their estimated map
  position falls in ("who is *where*").

Both must run on a desktop with no ROS installed, so this module is pure Python
(the same convention as ``world_state``'s core and ``omni_memory``).

Coordinates are metres in the ``map`` frame, the same frame Nav2 goals and
``save_location`` already use. A zone is "the bar, not the centimetre" — polygons
are drawn coarsely by hand from the map, and point-in-zone is a room-level
answer, not a survey.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

# A vertex / point in the map frame, metres.
Point = tuple[float, float]
# A navigation goal: (x, y, yaw_degrees) — the same shape omni_config locations
# already use, so anchors and legacy point-locations are interchangeable to a
# Nav2 goal builder.
Pose = tuple[float, float, float]


def point_in_polygon(x: float, y: float, polygon: Sequence[Point]) -> bool:
    """Ray-casting point-in-polygon. A point exactly on an edge is treated as
    inside on the "lower"/"left" sides only, which is fine at room scale — no
    two zones are expected to abut to the millimetre.

    A polygon of fewer than 3 vertices can contain nothing.
    """
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        # Does a horizontal ray from (x, y) cross edge (i, j)?
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def polygon_centroid(polygon: Sequence[Point]) -> Point:
    """Area-weighted centroid of a simple polygon. Falls back to the vertex mean
    for a degenerate (zero-area) polygon so a name always resolves to *some*
    point rather than dividing by zero."""
    n = len(polygon)
    if n == 0:
        raise ValueError("polygon has no vertices")
    if n < 3:
        cx = sum(p[0] for p in polygon) / n
        cy = sum(p[1] for p in polygon) / n
        return (cx, cy)

    area2 = 0.0   # twice the signed area
    cx = 0.0
    cy = 0.0
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        cross = xj * yi - xi * yj
        area2 += cross
        cx += (xi + xj) * cross
        cy += (yi + yj) * cross
        j = i

    if abs(area2) < 1e-9:
        # Collinear / zero-area: mean of vertices is the sensible fallback.
        return (sum(p[0] for p in polygon) / n, sum(p[1] for p in polygon) / n)

    cx /= (3.0 * area2)
    cy /= (3.0 * area2)
    return (cx, cy)


@dataclass(frozen=True)
class Zone:
    """A named room: a polygon in the map frame and an optional nav anchor.

    ``anchor`` is where OMNI actually parks when sent to this room — usually a
    clear spot with a sensible facing, not the geometric centre (which might be
    the middle of a table). When absent, the polygon centroid is used with yaw 0.

    ``check_in_side`` (Session 9) is which side of a person OMNI should stand on
    when it drives over to ask what they are working on: ``'left'``, ``'right'``,
    or None to use the caller's default.

    Why this is config and not perception: a side offset really wants the
    person's *facing* direction, and nothing in the system estimates that —
    `world_state` has a face box and no pose model. But a workbench does not
    move, and neither does the way you sit at it, so the useful approximation is
    a fixed convention per room. Facing estimation is a someday-item; this is the
    honest v1.
    """

    name: str
    polygon: tuple[Point, ...]
    anchor: Optional[Pose] = None
    check_in_side: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Zone.name is required")
        if self.check_in_side not in (None, "left", "right"):
            raise ValueError(
                f"zone {self.name!r}: check_in_side must be 'left', 'right' or "
                f"absent, got {self.check_in_side!r}")
        if len(self.polygon) < 3:
            raise ValueError(
                f"zone {self.name!r} needs a polygon of at least 3 vertices, "
                f"got {len(self.polygon)}"
            )

    def contains(self, x: float, y: float) -> bool:
        return point_in_polygon(x, y, self.polygon)

    @property
    def center(self) -> Point:
        """Geometric centre of the room — the best single-point estimate of
        'somewhere in this zone' when nothing finer is known."""
        return polygon_centroid(self.polygon)

    @property
    def nav_pose(self) -> Pose:
        """The goal pose to drive to for this zone: the explicit anchor if set,
        else the centroid facing yaw 0."""
        if self.anchor is not None:
            return self.anchor
        cx, cy = self.center
        return (cx, cy, 0.0)


class ZoneMap:
    """A collection of named zones with point-in-zone lookup.

    Zones may in principle overlap; ``zone_at`` returns the first match in
    definition order, so list the more specific room before a larger enclosing
    one if that ever happens. In practice rooms do not overlap.
    """

    def __init__(self, zones: Sequence[Zone] = ()) -> None:
        self._zones: list[Zone] = list(zones)
        names = [z.name for z in self._zones]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate zone names: {sorted(dupes)}")
        self._by_name = {z.name: z for z in self._zones}

    def __len__(self) -> int:
        return len(self._zones)

    def __bool__(self) -> bool:
        return bool(self._zones)

    @property
    def names(self) -> list[str]:
        return [z.name for z in self._zones]

    @property
    def zones(self) -> list[Zone]:
        return list(self._zones)

    def get(self, name: str) -> Optional[Zone]:
        return self._by_name.get(name)

    def zone_at(self, x: float, y: float) -> Optional[str]:
        """Name of the zone containing (x, y), or None if the point is in no
        defined zone (unmapped floor, a corridor with no polygon, etc.)."""
        for zone in self._zones:
            if zone.contains(x, y):
                return zone.name
        return None

    def center(self, name: str) -> Optional[Point]:
        zone = self._by_name.get(name)
        return zone.center if zone is not None else None

    def nav_pose(self, name: str) -> Optional[Pose]:
        """The goal pose for a named zone, or None if the name is unknown."""
        zone = self._by_name.get(name)
        return zone.nav_pose if zone is not None else None


# ── config loading ──────────────────────────────────────────────────────────

def _as_point(raw, ctx: str) -> Point:
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        raise ValueError(f"{ctx}: expected [x, y], got {raw!r}")
    return (float(raw[0]), float(raw[1]))


def zone_from_dict(name: str, spec: dict) -> Zone:
    """Build one Zone from its config mapping.

    Expected shape::

        kitchen:
          polygon: [[x0, y0], [x1, y1], [x2, y2], ...]   # >= 3 vertices, map frame, metres
          anchor:  [x, y, yaw_degrees]                    # optional nav goal
          check_in_side: left                             # optional: 'left' | 'right'
    """
    if not isinstance(spec, dict):
        raise ValueError(f"zone {name!r}: expected a mapping, got {type(spec).__name__}")
    raw_poly = spec.get("polygon")
    if not raw_poly:
        raise ValueError(f"zone {name!r}: missing 'polygon'")
    polygon = tuple(_as_point(p, f"zone {name!r} polygon vertex") for p in raw_poly)

    anchor: Optional[Pose] = None
    raw_anchor = spec.get("anchor")
    if raw_anchor is not None:
        if not isinstance(raw_anchor, (list, tuple)) or len(raw_anchor) < 2:
            raise ValueError(f"zone {name!r}: anchor must be [x, y] or [x, y, yaw]")
        ax, ay = float(raw_anchor[0]), float(raw_anchor[1])
        ayaw = float(raw_anchor[2]) if len(raw_anchor) > 2 else 0.0
        anchor = (ax, ay, ayaw)

    raw_side = spec.get("check_in_side")
    side = str(raw_side).strip().lower() if raw_side is not None else None

    return Zone(name=name, polygon=polygon, anchor=anchor, check_in_side=side)


def load_zone_map(config: Optional[dict]) -> ZoneMap:
    """Build a ZoneMap from a ``{name: {polygon, anchor}}`` mapping.

    A missing or empty mapping yields an empty ZoneMap — zone features then
    degrade cleanly (no polygons means every point is in no zone), rather than
    raising. This is the expected state until the space is actually traced into
    rooms.
    """
    if not config:
        return ZoneMap([])
    zones = [zone_from_dict(name, spec) for name, spec in config.items()]
    return ZoneMap(zones)
