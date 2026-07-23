"""Coarse geometry for placing people on the map. No ROS imports.

Two jobs, both deliberately *coarse*:

1. ``estimate_person_xy`` — from where the robot is, which camera saw the
   person, and where in that camera's frame they appeared, guess a map-frame
   (x, y) for the person. This is what lets ``world_state`` label a track with a
   room.

2. ``standoff_pose`` — given a person's estimated position and where the robot
   is now, produce a Nav2 goal that stops a polite distance *short* of them and
   faces them, rather than driving onto the spot they occupy.

## Accuracy — read before trusting a coordinate

The person estimate is monocular and unranged. It is honest at the room level
and no finer. The error sources, roughly largest first:

* **Distance is assumed, not measured.** A single ``distance`` is used for every
  detection (a face box gives no reliable range). A person 3 m away estimated at
  1.5 m lands halfway to them — enough to fall in the wrong room near a doorway.
* **Head pan is ignored.** The head camera sits on a pan/tilt that
  ``head_tracking`` swings up to ~±40°. ``camera_offset`` captures only the
  *mounting* direction (0 for a forward head, π for the fixed rear camera); the
  live pan angle is not threaded through, so a person the head has turned to
  look at is estimated as if the head faced forward.
* **Bearing needs the image width.** ``bearing_from_bbox`` converts a pixel
  column to an angle using the horizontal FOV; give it a wrong width or FOV and
  the angle scales wrong. With width unknown it returns 0 (person dead ahead of
  the camera), which is a safe, boring default.

Consumers should treat the returned point as "roughly here" and fall back to the
robot's own zone when the estimate lands in no zone at all — a person being
spoken to at conversational range is almost always in the robot's room.
"""

from __future__ import annotations

import math
from typing import Optional

from .zones import Point, Pose


def bearing_from_bbox(cx: float, image_width: float, hfov_rad: float) -> float:
    """Horizontal bearing (radians) of a detection from the camera's optical
    axis, using REP-103 sign: **left of centre is positive** (counter-clockwise,
    +y to the robot's left).

    In an image, column 0 is the camera's left and increasing ``cx`` moves
    right, so a person left-of-centre (small ``cx``) yields a positive bearing.

    ``image_width <= 0`` (unknown) returns 0.0 — assume the detection is on the
    optical axis rather than inventing an angle from a bad width.
    """
    if image_width <= 0 or hfov_rad <= 0:
        return 0.0
    frac = cx / image_width - 0.5          # -0.5 (left edge) .. +0.5 (right edge)
    return -frac * hfov_rad                 # left edge -> +half-FOV


def estimate_person_xy(
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
    camera_offset: float,
    bearing: float,
    distance: float,
) -> Point:
    """Project a person into the map frame from the robot pose and a bearing.

    ``robot_yaw`` and ``camera_offset`` and ``bearing`` are all radians, added:
    the absolute map bearing to the person is
    ``robot_yaw + camera_offset + bearing``. ``camera_offset`` is the camera's
    mounting yaw relative to robot-forward (0 for the head, π for the rear cam).

    ``distance`` is metres along that bearing — an assumed conversational range,
    not a measurement (see module docstring).
    """
    theta = robot_yaw + camera_offset + bearing
    return (robot_x + distance * math.cos(theta),
            robot_y + distance * math.sin(theta))


def standoff_pose(
    person_xy: Point,
    from_xy: Point,
    standoff: float = 1.0,
    lateral: float = 0.0,
) -> Pose:
    """A Nav2 goal ``(x, y, yaw_deg)`` that stops ``standoff`` metres short of
    ``person_xy`` on the line back toward ``from_xy`` (where the robot is now),
    facing the person.

    So OMNI ends up a polite arm's length away, looking at them, instead of
    driving onto the exact spot the estimate put them (which is both rude and,
    given the estimate's error, possibly *through* them).

    ``lateral`` (Session 9) slides the goal sideways off that line, in metres:
    **positive is to the robot's left as it faces the person**, negative to the
    right. Zero reproduces the Session 7 behaviour exactly, so existing callers
    are unaffected.

    Why a side offset exists at all: a proactive check-in drives up to someone
    who is *working*. Parking squarely in front of a person at their bench is
    blocking them, and asking whether they need help while blocking them rather
    answers itself. Standing beside them reads as joining them.

    THE LIMIT WORTH KNOWING: this offsets relative to the **approach line**, not
    to the person's facing direction — `world_state` has no facing estimate at
    all (no pose model, only a face box). So "their left" is unknowable here, and
    the caller decides which side to pass as a fixed per-zone convention (you sit
    the same way at the bench every day). Facing estimation is a someday-item.

    Yaw always points at the person from wherever the goal lands, so a lateral
    offset turns OMNI to look across at them rather than driving past staring
    ahead.

    Degenerate case: if the robot's current position and the person estimate
    coincide (or are closer than ``standoff``), the goal is the robot's own
    position, facing the person's estimated bearing — moving would only add
    error. A lateral offset is still applied there, because "shuffle sideways to
    stop looming over them" is exactly right when already too close.
    """
    px, py = person_xy
    fx, fy = from_xy
    dx, dy = px - fx, py - fy
    dist = math.hypot(dx, dy)

    if dist <= 1e-6:
        # No approach direction to offset from, and no bearing to face.
        return (fx, fy, 0.0)

    ux, uy = dx / dist, dy / dist
    # Left-hand normal of the approach direction, in the map frame's usual
    # right-handed convention (x forward, y left, yaw counter-clockwise).
    nx, ny = -uy, ux

    if dist <= standoff:
        # Already at (or inside) the standoff radius — hold position, but still
        # honour the sideways shift.
        gx, gy = fx + nx * lateral, fy + ny * lateral
    else:
        gx = px - ux * standoff + nx * lateral
        gy = py - uy * standoff + ny * lateral

    # Face the person FROM THE GOAL, not along the original approach line — with
    # a lateral offset those differ, and using the old bearing would leave OMNI
    # looking past them.
    yaw = math.degrees(math.atan2(py - gy, px - gx))
    return (gx, gy, yaw)
