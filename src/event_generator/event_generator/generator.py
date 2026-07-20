"""EventGenerator — semantic events from a run of world-state snapshots.

Pure Python, no ROS. Feed it the dicts published on ``/omni/world_state`` (1 Hz)
in order; it returns the events that fall out of each one.

    gen = EventGenerator()
    for snapshot in snapshots:
        for event in gen.ingest(snapshot):
            ...

THE CONSTRAINT THAT SHAPES ALL OF THIS
--------------------------------------
World state is **face-anchored**. ``world_state`` runs on ``/camera/identities``
alone (see its CLAUDE.md, "Why person boxes are off by default"), so a track is
only visible while the Jetson can see and recognise a *face*. Someone at the
workbench who turns to pick up a screwdriver stops being visible. Someone who
tilts their head down stops being visible. This is normal, constant, and it is
NOT the person leaving the room.

So visibility is not presence. Every debounce here exists to put distance
between the two:

* ``person_left`` requires ``absence_grace`` seconds of *sustained* absence
  (default 90s). Below that the person stays PRESENT and no event fires at all.
* Because a flickering person never leaves PRESENT, they can never re-enter it,
  so ``person_appeared`` cannot re-fire on flicker. That is the whole mechanism —
  there is no separate "recently greeted" suppression in this library, and there
  should not be. (Greeting cooldown is a behaviour concern; it lives in
  behavior_node.)
* ``unknown_person_detected`` additionally requires the same stable ``unknown_N``
  id across ``unknown_min_snapshots`` snapshots.

WHAT IS DELIBERATELY IGNORED
----------------------------
Rows with **no identity at all** (``identity: null``, rendered as ``person_N``
in the snapshot's ``present`` list) generate no events of any kind. Upstream,
``/camera/identities`` is known to emit two rows for one face — one identified,
one empty a few pixels away (world_state CLAUDE.md, "Upstream note"). Treating
those as people would announce a phantom stranger standing next to every person
OMNI recognises. There is no debounce that fixes this, because the phantom is
perfectly stable; the only correct handling is to not look at it.

IDENTITY IS THE KEY, NOT track_id
---------------------------------
Tracks are cheap and get re-minted; identities are stable across cameras and
across re-minting. So presence is keyed on the identity string. The consequence
worth knowing: if the recognizer promotes ``unknown_3`` to ``rafael``, those are
two independent keys. ``rafael`` appears (no history, ``away_duration`` None) and
``unknown_3`` eventually goes absent silently — ``person_left`` only fires for
named people, so the promotion does not emit a spurious departure.
"""

from __future__ import annotations

from typing import Iterable, Optional

from .models import (
    PERSON_APPEARED,
    PERSON_LEFT,
    UNKNOWN_PERSON_DETECTED,
    Event,
    is_named,
    is_unknown,
)

# Seconds of sustained absence before a person counts as gone. Must comfortably
# exceed the longest normal face dropout, not the longest *detection* dropout:
# world_state's own visibility_timeout is 3s, and a person working with their
# hands drops out for far longer than that, repeatedly. 90s is chosen to sit well
# above "turned away for a while" and well below "went to make a coffee".
DEFAULT_ABSENCE_GRACE = 90.0

# Snapshots a stable unknown_N must survive before it is announced. At the 1 Hz
# publish rate this is ~3s of a consistent stranger. Guards against a recognizer
# id that appears for one frame and is never seen again.
DEFAULT_UNKNOWN_MIN_SNAPSHOTS = 3

# Snapshots a *named* person must survive before person_appeared fires. One, on
# purpose: an identity has already passed the Jetson recognizer's own hysteresis
# and frontality gate before it ever reaches world_state, so re-confirming it
# here buys nothing and costs a full second against the greeting latency budget.
DEFAULT_APPEAR_MIN_SNAPSHOTS = 1

# unknown_person_detected is scoped to the head camera. The rear camera publishes
# no identities topic at all today (world_state CLAUDE.md, "Rear camera publishes
# no identities topic yet"), so in practice every identity is already head-tagged
# — this is a guard for when the rear recognizer lands, not a filter that does
# anything right now.
DEFAULT_UNKNOWN_CAMERAS = ("head",)

# Pixels. An unknown_N whose centre lands within this of a named person we
# believe is present is treated as that person's own face misread, not a
# stranger. Matches world_state's own `match_radius`, which was tuned for exactly
# this question ("is this the same person on this camera") in the same 1280x720
# publish space.
#
# THE TRADE, stated plainly: a genuine stranger standing shoulder-to-shoulder
# with someone OMNI knows — within ~160px, so roughly touching at conversational
# range — will not be announced. That is the deliberate choice. Live measurement
# put the false-stranger rate at five in two minutes from one seated person, and
# a phantom stranger announced next to every person OMNI recognises is a much
# worse failure than a missed one. Lower this if strangers-beside-known-people
# ever matter more than quiet.
DEFAULT_NAMED_OVERLAP_RADIUS = 160.0

# Pixels. Smallest face box an `unknown_N` may have and still be announced as a
# stranger, measured on the shorter side.
#
# Measured live 2026-07-19 in the 1280x720 publish space: Rafael's face at
# conversational range is ~88x115px. The phantoms that survived overlap
# suppression were 34.7x34.8px at the far left edge and 23.0x25.1px at the far
# right — background clutter that YuNet scores above its 0.6 threshold (a pattern
# on a shelf, a photo, a reflection). There is clean separation between the two
# populations, and no debounce over time can tell them apart, because the clutter
# is perfectly stationary and perfectly persistent.
#
# A person far enough away to present a sub-50px face is also too far away to be
# worth announcing as having walked in.
DEFAULT_UNKNOWN_MIN_FACE_PX = 50.0

# Presence states.
_PRESENT = "present"
_ABSENT = "absent"


def _as_bbox(raw) -> Optional[tuple]:
    """Snapshot bbox -> (cx, cy, w, h), or None if it is not usable."""
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    try:
        return tuple(float(v) for v in raw[:4])
    except (TypeError, ValueError):
        return None


def _snapshot_named_boxes(snapshot: dict) -> list[tuple]:
    """Boxes of the visible, named people in one snapshot."""
    boxes = []
    for row in snapshot.get("people", []) or []:
        if not isinstance(row, dict) or not row.get("visible"):
            continue
        if not is_named(row.get("identity")):
            continue
        box = _as_bbox(row.get("bbox"))
        if box is not None:
            boxes.append(box)
    return boxes


def _overlaps_named(box: Optional[tuple], named_boxes: list[tuple],
                    radius: float) -> bool:
    """True if `box` is sitting on or beside a known person's face.

    Containment in either direction, matching world_state's own association idiom
    (`_track_containing` / `_contained_by`): the two boxes are rarely the same
    size, because the recognizer's fallback detection is often a tighter or looser
    crop of the same face.

    Plus a centroid radius, because containment alone is not enough. Measured
    live: Rafael leaned across to (699.7, 220.1) in the last frame before his
    face dropped out, so his remembered box froze at that outlier; the phantom
    `unknown_17` then appeared at (614.4, 168.0), his usual seated spot, ~85px
    away — too far to be contained, obviously the same person. The remembered box
    is a snapshot of wherever they happened to be at dropout, which is very often
    mid-movement (moving is *why* recognition failed).
    """
    if box is None or not named_boxes:
        return False
    cx, cy, w, h = box
    for nx, ny, nw, nh in named_boxes:
        if abs(nx - cx) <= w / 2 and abs(ny - cy) <= h / 2:
            return True
        if abs(cx - nx) <= nw / 2 and abs(cy - ny) <= nh / 2:
            return True
        if radius > 0 and ((nx - cx) ** 2 + (ny - cy) ** 2) ** 0.5 <= radius:
            return True
    return False


def _face_big_enough(box: Optional[tuple], min_px: float) -> bool:
    """True if this face box is large enough to be a person worth announcing.

    Judged on the SHORTER side. A wide-but-flat or tall-but-thin box is a
    detector artifact rather than a face, and using area would let a 20x200px
    sliver through on the same budget as a genuine 63x63 face.
    """
    if min_px <= 0:
        return True
    if box is None:
        return False   # no box to judge -> not credible as a new arrival
    return min(abs(box[2]), abs(box[3])) >= min_px


class _Presence:
    """Per-identity presence bookkeeping. Internal; not part of the API."""

    __slots__ = ("identity", "state", "last_seen", "camera", "snapshots_seen",
                 "announced", "left_at", "bbox")

    def __init__(self, identity: str, camera: str, last_seen: float) -> None:
        self.identity = identity
        self.state = _ABSENT          # every key starts absent and transitions in
        self.last_seen = last_seen
        self.camera = camera
        self.snapshots_seen = 0
        self.announced = False        # unknown_person_detected already emitted
        self.left_at: Optional[float] = None   # last_seen at the moment we gave up
        self.bbox: Optional[tuple] = None      # newest box, for overlap tests


class EventGenerator:
    """Turns world-state snapshots into semantic events.

    Clock-agnostic, like ``world_state``: time comes from each snapshot's
    ``stamp`` field, never from ``time.time()``. That is what makes the replay
    test able to run a ten-minute absence in a millisecond.
    """

    def __init__(
        self,
        absence_grace: float = DEFAULT_ABSENCE_GRACE,
        unknown_min_snapshots: int = DEFAULT_UNKNOWN_MIN_SNAPSHOTS,
        appear_min_snapshots: int = DEFAULT_APPEAR_MIN_SNAPSHOTS,
        unknown_cameras: Iterable[str] = DEFAULT_UNKNOWN_CAMERAS,
        named_overlap_radius: float = DEFAULT_NAMED_OVERLAP_RADIUS,
        unknown_min_face_px: float = DEFAULT_UNKNOWN_MIN_FACE_PX,
    ) -> None:
        if absence_grace <= 0:
            raise ValueError("absence_grace must be > 0")
        if unknown_min_snapshots < 1 or appear_min_snapshots < 1:
            raise ValueError("snapshot thresholds must be >= 1")
        self.absence_grace = absence_grace
        self.unknown_min_snapshots = unknown_min_snapshots
        self.appear_min_snapshots = appear_min_snapshots
        self.unknown_cameras = frozenset(unknown_cameras)
        self.named_overlap_radius = named_overlap_radius
        self.unknown_min_face_px = unknown_min_face_px
        self._people: dict[str, _Presence] = {}
        self._last_stamp: Optional[float] = None

    # ── ingest ────────────────────────────────────────────────────────────────

    def ingest(self, snapshot: dict) -> list[Event]:
        """Fold in one ``/omni/world_state`` snapshot; return its events.

        Snapshots arriving out of order (or a replayed fixture restarting) are
        dropped rather than rewinding the clock — a backwards stamp would make
        every absence calculation negative.
        """
        if not isinstance(snapshot, dict):
            return []
        stamp = snapshot.get("stamp")
        if not isinstance(stamp, (int, float)):
            return []
        stamp = float(stamp)
        if self._last_stamp is not None and stamp < self._last_stamp:
            return []
        self._last_stamp = stamp

        events: list[Event] = []
        seen_this_snapshot: set[str] = set()

        # Where the named people are. An unknown_N sitting on top of one of these
        # is that person's own face misread, not a stranger — see _overlaps_named.
        #
        # This deliberately includes the LAST KNOWN box of named people we still
        # believe are PRESENT but whose face has dropped out of this frame. That
        # case is not an edge case, it is the main one: the recognizer mints a
        # fresh unknown_N precisely *because* it stopped recognising the face, so
        # in the very frame the phantom appears, the real person is usually no
        # longer visible and contributes no box to compare against. Our own
        # debounced presence belief is the only thing that still knows they are
        # there. A remembered box goes slightly stale if they move, which is
        # acceptable: faces here are ~70x100px at conversational range and the
        # test is containment, not equality.
        named_boxes = _snapshot_named_boxes(snapshot)
        named_boxes.extend(
            p.bbox for p in self._people.values()
            if p.state == _PRESENT and is_named(p.identity) and p.bbox is not None
        )

        for row in snapshot.get("people", []) or []:
            if not isinstance(row, dict):
                continue
            identity = row.get("identity")
            # The empty-identity phantom (see module docstring) stops here, and
            # so does any row shaped unexpectedly.
            if not isinstance(identity, str) or not identity.strip():
                continue
            identity = identity.strip()
            camera = str(row.get("camera") or "")
            visible = bool(row.get("visible"))
            last_seen = row.get("last_seen")
            last_seen = float(last_seen) if isinstance(last_seen, (int, float)) else stamp

            seen_this_snapshot.add(identity)
            person = self._people.get(identity)
            if person is None:
                person = _Presence(identity, camera, last_seen)
                self._people[identity] = person

            if camera:
                person.camera = camera
            person.bbox = _as_bbox(row.get("bbox"))
            # last_seen only ever moves forward: an away row keeps reporting the
            # same last_seen every snapshot, and we must not let a stale row from
            # another camera pull it backwards.
            person.last_seen = max(person.last_seen, last_seen)

            if visible:
                person.snapshots_seen += 1
                events.extend(self._on_visible(person, stamp, named_boxes))
            elif person.state != _PRESENT:
                # CONSECUTIVE, not cumulative. Measured live 2026-07-19: over 129s
                # of one person at a desk, counting cumulative sightings announced
                # FIVE strangers, one of them on 4 visible snapshots scattered
                # across 98. Recognizer churn mints short-lived unknown_N ids
                # constantly, and any of them eventually accumulates N sightings
                # if the count never resets. A real stranger standing in front of
                # the robot clears a consecutive run easily; noise does not.
                person.snapshots_seen = 0

        # Absence is evaluated for everyone we know about, including keys that
        # have vanished from `people` entirely — world_state prunes anonymous
        # away-tracks (max_history), so "no longer in the list" is itself a
        # perfectly ordinary way for an unknown_N to disappear.
        for person in self._people.values():
            if person.identity in seen_this_snapshot:
                continue
            # Vanishing from the list breaks a consecutive run just as surely as
            # being listed but not visible.
            if person.state != _PRESENT:
                person.snapshots_seen = 0
            events.extend(self._check_absence(person, stamp))

        # A row can be present-but-not-visible; that path is checked too.
        for identity in seen_this_snapshot:
            events.extend(self._check_absence(self._people[identity], stamp))

        return events

    # ── transitions ───────────────────────────────────────────────────────────

    def _on_visible(self, person: _Presence, stamp: float,
                    named_boxes: list[tuple]) -> list[Event]:
        if person.state == _PRESENT:
            return []   # already here — flicker back into view is a non-event

        if is_named(person.identity):
            if person.snapshots_seen < self.appear_min_snapshots:
                return []
            person.state = _PRESENT
            away = None if person.left_at is None else max(0.0, stamp - person.left_at)
            person.left_at = None
            return [Event(
                kind=PERSON_APPEARED,
                identity=person.identity,
                camera=person.camera,
                timestamp=stamp,
                away_duration=away,
                detail=(
                    "first sighting" if away is None
                    else f"back after {away:.0f}s away"
                ),
            )]

        if is_unknown(person.identity):
            # A stranger standing exactly where a person we KNOW is standing is
            # not a stranger. Measured live 2026-07-19: `unknown_58` at bbox
            # (606.2, 187.9) while `rafael` sat at (606.3, 187.4) — same face,
            # same frame, both flagged visible. The Jetson recognizer drops to a
            # fresh unknown_N whenever confidence dips (a head turn is enough),
            # and world_state cannot merge them back: "no re-identification, that
            # is a new track". Suppressing on overlap is the only place this can
            # be caught. Without it, one person at a desk generated FIVE stranger
            # announcements in two minutes.
            if _overlaps_named(person.bbox, named_boxes,
                               self.named_overlap_radius):
                person.snapshots_seen = 0
                return []
            # Too small to be a person who has "arrived". See the constant.
            if not _face_big_enough(person.bbox, self.unknown_min_face_px):
                person.snapshots_seen = 0
                return []
            # Not credible yet — stay ABSENT so the snapshot counter keeps
            # accumulating instead of latching presence on a one-frame id.
            if person.snapshots_seen < self.unknown_min_snapshots:
                return []
            person.state = _PRESENT
            # Announce once per stranger. A re-appearing unknown_N is the *same*
            # stranger the recognizer already told us about, so a second
            # announcement would be noise.
            if person.announced or person.camera not in self.unknown_cameras:
                return []
            person.announced = True
            return [Event(
                kind=UNKNOWN_PERSON_DETECTED,
                identity=person.identity,
                camera=person.camera,
                timestamp=stamp,
                detail=(
                    f"stable across {person.snapshots_seen} snapshots"
                ),
            )]

        return []

    def _check_absence(self, person: _Presence, stamp: float) -> list[Event]:
        if person.state != _PRESENT:
            return []
        gap = stamp - person.last_seen
        if gap <= self.absence_grace:
            return []   # inside the grace window — turned away, not gone

        person.state = _ABSENT
        person.snapshots_seen = 0
        # Anchor the away clock to the last real sighting, not to now: the
        # grace period is part of how long they were gone, not a free pass.
        person.left_at = person.last_seen

        if not is_named(person.identity):
            return []   # strangers leaving is not worth an event
        return [Event(
            kind=PERSON_LEFT,
            identity=person.identity,
            camera=person.camera,
            timestamp=stamp,
            detail=f"not seen for {gap:.0f}s",
        )]

    # ── query ─────────────────────────────────────────────────────────────────

    def present(self) -> list[str]:
        """Identities currently believed present (debounced, not raw visibility)."""
        return sorted(p.identity for p in self._people.values() if p.state == _PRESENT)

    def is_present(self, identity: str) -> bool:
        person = self._people.get(identity)
        return person is not None and person.state == _PRESENT
