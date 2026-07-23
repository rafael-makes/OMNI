"""person_nav.py — the ROS-free decision behind go_to_person().

Kept out of function_handlers for the same reason scene_describer and
greeting_decider are: the *decision* ("do I know where this person is, and is it
recent enough to drive to?") is pure logic over a world_state snapshot, and it
should be testable on a desktop with no ROS. The handler in function_handlers
turns a :class:`PersonPlan` into the actual Nav2 goal and the spoken line.

The world_state snapshot is coarse — a person's ``zone`` is room-level and their
``map_xy`` is a monocular estimate (see world_state/CLAUDE.md). So this leans
toward honesty: unknown, unplaced, or stale all resolve to a "say so" outcome
rather than a confident drive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# Outcomes, in the order the handler checks them.
NO_NAME        = 'no_name'          # no name given and nobody is being talked to
NO_WORLD_STATE = 'no_world_state'   # world_state not publishing
UNKNOWN        = 'unknown'          # never seen this person
UNPLACED       = 'unplaced'         # seen, but no map position or zone
STALE          = 'stale'            # last seen longer ago than the threshold
GO             = 'go'               # good to navigate


@dataclass
class PersonPlan:
    outcome: str
    name: str = ''
    zone: Optional[str] = None
    map_xy: Optional[tuple] = None
    seconds_since_seen: Optional[float] = None

    @property
    def display(self) -> str:
        return self.name.capitalize() if self.name else ''


def clean_name(raw: str) -> str:
    """Reduce a spoken name to a single clean identity token (first name),
    matching how learn_person / the recognizer key identities."""
    return ''.join(ch for ch in (raw or '').lower()
                   if ch.isalnum() or ch in ('_', '-'))


def humanise_age(secs) -> str:
    """A coarse, speakable 'how long ago'."""
    if not isinstance(secs, (int, float)):
        return 'recently'
    s = int(secs)
    if s < 10:
        return 'just now'
    if s < 60:
        return f'about {s} seconds ago'
    m = s // 60
    if m < 60:
        return f"about {m} minute{'s' if m != 1 else ''} ago"
    h = m // 60
    return f"about {h} hour{'s' if h != 1 else ''} ago"


def plan_go_to_person(
    snapshot: Optional[dict],
    name: str,
    session_person: Optional[str],
    stale_after: float,
) -> PersonPlan:
    """Decide what go_to_person should do, without touching ROS.

    ``name``           — the requested person (may be empty → "come here").
    ``session_person`` — who OMNI is currently talking with, used for "come here".
    ``stale_after``    — seconds after which a last-known location is untrusted.
    """
    resolved = clean_name(name) or clean_name(session_person or '')
    if not resolved:
        return PersonPlan(NO_NAME)

    if not snapshot or not isinstance(snapshot, dict):
        return PersonPlan(NO_WORLD_STATE, name=resolved)

    people = snapshot.get('people', []) or []
    person = next(
        (p for p in people if (p.get('identity') or '').lower() == resolved), None)
    if person is None:
        return PersonPlan(UNKNOWN, name=resolved)

    secs = person.get('seconds_since_seen')
    zone = person.get('zone')
    map_xy = person.get('map_xy')
    map_xy = tuple(map_xy) if isinstance(map_xy, (list, tuple)) and len(map_xy) >= 2 else None

    if not map_xy and not zone:
        return PersonPlan(UNPLACED, name=resolved, seconds_since_seen=secs)

    if isinstance(secs, (int, float)) and secs > stale_after:
        return PersonPlan(STALE, name=resolved, zone=zone,
                          map_xy=map_xy, seconds_since_seen=secs)

    return PersonPlan(GO, name=resolved, zone=zone,
                      map_xy=map_xy, seconds_since_seen=secs)
