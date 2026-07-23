"""Data types for the world-state people tracker. No ROS imports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

# A bbox is (cx, cy, w, h) in whatever units the source publishes. The tracker
# never interprets it — it is carried through to the snapshot for consumers.
BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    """One observation of one person from one camera at one instant.

    ``identity`` is the recognizer's label: a known name ("rafael"), a stable
    anonymous id ("unknown_3"), or None/'' when the source has no identity at
    all (a bare YOLO person box). Empty string is normalised to None.

    ``source_track`` is the *source-side* track id, unique only within a camera.
    Supplying it is what lets an anonymous detection later be promoted when the
    recognizer puts a name to the same track.

    ``map_xy`` and ``zone`` are the coarse *where*: an estimated (x, y) in the
    map frame and the named zone it falls in. Both are computed by the ROS
    wrapper (which knows the robot pose) and carried through untouched — the
    tracker never does geometry. Both are None when localisation or a zone map
    is unavailable, and the tracker degrades cleanly to identity-only "who".
    """

    camera: str
    timestamp: float
    identity: Optional[str] = None
    confidence: float = 0.0
    bbox: Optional[BBox] = None
    source_track: Optional[int] = None
    map_xy: Optional[tuple[float, float]] = None
    zone: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.camera:
            raise ValueError("Detection.camera is required")
        # Normalise '' -> None so callers can pass the recognizer's empty label
        # straight through without special-casing it.
        if self.identity is not None and not self.identity.strip():
            object.__setattr__(self, "identity", None)
        elif self.identity is not None:
            object.__setattr__(self, "identity", self.identity.strip())


@dataclass
class PersonTrack:
    """A person the robot believes exists, present or not."""

    track_id: int
    identity: Optional[str]
    confidence: float
    camera: str                       # camera of the most recent detection
    first_seen: float
    last_seen: float
    visible: bool = True
    bbox: Optional[BBox] = None
    cameras_seen: list[str] = field(default_factory=list)
    source_track: Optional[int] = None   # binding on `camera`, for promotion
    # Coarse location of the most recent detection that carried one. Kept even
    # after the person goes away, so "where did I last see Rafael" is answerable.
    map_xy: Optional[tuple[float, float]] = None
    zone: Optional[str] = None

    @property
    def is_identified(self) -> bool:
        """True for a real name. A stable ``unknown_N`` is an id, not a name."""
        return bool(self.identity) and not self.identity.startswith("unknown_")

    def as_dict(self, now: float) -> dict:
        return {
            "track_id": self.track_id,
            "identity": self.identity,
            "identified": self.is_identified,
            "confidence": round(self.confidence, 3),
            "camera": self.camera,
            "cameras_seen": list(self.cameras_seen),
            "first_seen": round(self.first_seen, 3),
            "last_seen": round(self.last_seen, 3),
            "seconds_since_seen": round(max(0.0, now - self.last_seen), 3),
            "visible": self.visible,
            "bbox": list(self.bbox) if self.bbox else None,
            "zone": self.zone,
            "map_xy": [round(self.map_xy[0], 3), round(self.map_xy[1], 3)] if self.map_xy else None,
        }


@dataclass(frozen=True)
class StateEvent:
    """A transition worth logging: a person appeared, was named, or went away."""

    kind: str          # "appeared" | "identified" | "away" | "returned"
    track_id: int
    identity: Optional[str]
    camera: str
    timestamp: float
    detail: str = ""


def cameras_of(tracks: Sequence[PersonTrack]) -> list[str]:
    seen: list[str] = []
    for t in tracks:
        if t.camera not in seen:
            seen.append(t.camera)
    return seen
