# CLAUDE.md — omni_zones

Read this at the start of every session. Write the pass test before the
implementation.

## What this is
The **shared, ROS-free** definition of OMNI's rooms: named polygons in the SLAM
`map` frame, plus the coarse geometry for placing a person on the map. Two
packages import it so there is **one source of truth** for "where the rooms
are":

- `behavior_node` — `navigate_to("kitchen")` and `go_to_person()` resolve a goal
  pose from a zone anchor; `go_to_person` needs the polygon of a person's zone.
- `world_state` — labels each person track with the zone their estimated map
  position falls in.

**No ROS imports.** Like `world_state`'s core and `omni_memory`, this must
import and run on a desktop with no ROS installed. `package.xml` depends only on
`python3-yaml`. Don't add `rclpy`.

## Layout
- `omni_zones/zones.py` — `Zone` (name + polygon + optional anchor + optional
  `check_in_side`), `ZoneMap` (point-in-zone lookup, centres, anchors),
  `load_zone_map` (config → map), `point_in_polygon`, `polygon_centroid`.
- `omni_zones/geometry.py` — `bearing_from_bbox`, `estimate_person_xy`,
  `standoff_pose`. The coarse "where a person is" math.

## The side offset (Session 9)
`standoff_pose(..., lateral=)` slides the goal sideways off the approach line —
positive is **left of the robot as it faces the person** — so a proactive
check-in arrives *beside* someone working rather than blocking their bench. Yaw
is recomputed at the goal so OMNI still looks across at them. `lateral=0` is
bit-identical to the Session 7 behaviour, and `go_to_person` still passes
nothing, so that path is untouched.

The offset is relative to the **approach line, not the person's facing
direction** — nothing in the system estimates facing (`world_state` has a face
box, not a pose model). Which side is therefore a fixed per-room convention,
`check_in_side: left|right` in `zones.yaml`. A bench does not move and neither
does how you sit at it. Facing estimation is a someday-item.
- `config/zones.yaml` — the deployment's rooms. **Ships empty** until the space
  is traced into polygons; empty is valid and degrades cleanly.
- `tests/` — pytest, robot off (`test_zones.py`, `test_geometry.py`).

## The accuracy contract — this is deliberately coarse
`estimate_person_xy` is monocular and unranged: **room-level, not metric.** Read
the `geometry.py` module docstring before trusting a coordinate. Error sources,
largest first: **distance is assumed** (a face box gives no range), **head pan
is ignored** (only the camera *mounting* offset is modelled, not the live
pan/tilt angle `head_tracking` swings), and **bearing needs the image width**
(unknown width → bearing 0, i.e. "dead ahead"). Consumers should fall back to
the robot's own zone when an estimate lands in no zone — a person at
conversational range is almost always in the robot's room. "The bar, not the
centimetre."

## Coordinates
Metres in the `map` frame — the same frame Nav2 goals and `save_location`
already use. A zone anchor is `[x, y, yaw_deg]`, interchangeable with the point
locations in `omni_config.yaml`.

## Running
```
cd ~/omni_ws && colcon build --packages-select omni_zones && source install/setup.bash
cd src/omni_zones && python3 -m pytest tests/ -q         # robot off, no ROS needed
```

## Status
- [x] Library + tests (point-in-polygon incl. concave, centroids, config load,
      bearing, person estimate, standoff). **49 tests.**
- [x] Wired into `world_state` (per-person zone) and `behavior_node`
      (`navigate_to` anchors + `go_to_person`).
- [x] Session 9: `standoff_pose(lateral=)` side offset + per-zone
      `check_in_side`, with a test asserting `lateral=0` is unchanged from
      Session 7.
- [ ] `config/zones.yaml` is **empty** — no rooms traced yet. Until it is filled,
      `go_to_person` honestly reports it cannot place people and `navigate_to`
      uses point locations only.
- [ ] Not yet exercised against a live map + live world_state. The person
      estimate's real-world error is unmeasured; the assumed-distance and
      ignored-head-pan limits above are reasoned, not yet characterised on
      hardware.
