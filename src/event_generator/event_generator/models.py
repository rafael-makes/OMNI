"""Data types for the semantic event generator. No ROS imports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Event kinds. Consumers switch on these strings, so they are part of the wire
# contract of /omni/events — renaming one is a breaking change.
PERSON_APPEARED = "person_appeared"
PERSON_LEFT = "person_left"
UNKNOWN_PERSON_DETECTED = "unknown_person_detected"
# A named person has stayed in one zone continuously past a threshold. Unlike the
# three above, this is not a visibility *transition* — it is a duration crossing,
# and it re-fires on an interval while the dwell continues. The policy layer
# (check_in_policy in behavior_node) decides whether any given firing is worth
# acting on; the generator only reports that the opportunity exists. See
# generator.py "DWELL" and the package CLAUDE.md.
PERSON_DWELLING = "person_dwelling"


@dataclass(frozen=True)
class Event:
    """One semantic transition derived from a run of world-state snapshots.

    ``away_duration`` is seconds between the last time this person was actually
    seen and this reappearance — *including* the debounce grace period, so it is
    the real wall-clock gap, not the gap since we admitted they had left. It is
    None on a first-ever sighting, which is the signal "no history, this is not
    a return".

    ``zone`` and ``dwell_duration`` are populated only on ``person_dwelling``:
    the named room the person has been sitting in, and how many seconds they have
    now been there continuously. Both are None on the presence-transition events,
    where they carry no meaning.
    """

    kind: str
    identity: str
    camera: str
    timestamp: float
    away_duration: Optional[float] = None
    detail: str = ""
    zone: Optional[str] = None
    dwell_duration: Optional[float] = None

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "identity": self.identity,
            "camera": self.camera,
            "timestamp": round(self.timestamp, 3),
            "away_duration": (
                None if self.away_duration is None else round(self.away_duration, 3)
            ),
            "detail": self.detail,
            "zone": self.zone,
            "dwell_duration": (
                None if self.dwell_duration is None else round(self.dwell_duration, 3)
            ),
        }


def is_unknown(identity: Optional[str]) -> bool:
    """True for the recognizer's stable anonymous ids ("unknown_3").

    These are *ids*, not names: they persist across reboots and handoff like a
    name, but there is nobody to greet by name behind them.
    """
    return bool(identity) and identity.startswith("unknown_")


def is_named(identity: Optional[str]) -> bool:
    """True for a real person name — the only thing worth greeting by name."""
    return bool(identity) and not is_unknown(identity)
