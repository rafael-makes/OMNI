"""WorldState — one shared, live answer to "who is where right now".

Pure Python, no ROS. Feed it :class:`Detection` events from any number of
cameras; ask it for a snapshot. It maintains one :class:`PersonTrack` per
believed person, flips them to *away* when they stop being seen, and keeps
them in history so "when did I last see Rafael" stays answerable.

Matching (deliberately simple for v1 — see CLAUDE.md "Known limits"):

1. A detection carrying an **identity** matches the track with that identity on
   *any* camera. This is what makes camera handoff work: the same label seen by
   the rear camera continues the track the head camera started, rather than
   minting a second person.
2. A detection with **no identity** matches by ``(camera, source_track)``, so a
   bare person box stays one track across frames within its own camera.
3. A detection with neither an identity nor a source track — which is what the
   Jetson's ``/camera/detections`` actually publishes, it never sets
   ``Detection2D.id`` — matches the **nearest anonymous track on the same
   camera** within ``match_radius`` pixels. Without this, every YOLO box mints
   a fresh track on every frame.
4. When a previously anonymous track turns up carrying an identity, it is
   **absorbed** into the identified one, and the earlier ``first_seen`` wins.
   That is how "a person walked in, then the recognizer worked out who they
   were" stays a single person.
"""

from __future__ import annotations

import time
from typing import Iterable, Optional

from .models import BBox, Detection, PersonTrack, StateEvent

DEFAULT_VISIBILITY_TIMEOUT = 3.0   # s without a detection -> not visible
DEFAULT_MAX_HISTORY = 50           # away anonymous tracks retained
DEFAULT_MATCH_RADIUS = 160.0       # px; centroid association for untracked boxes
DEFAULT_REJOIN_WINDOW = 20.0       # s an away track can still absorb a detection


class WorldState:
    def __init__(
        self,
        visibility_timeout: float = DEFAULT_VISIBILITY_TIMEOUT,
        max_history: int = DEFAULT_MAX_HISTORY,
        min_confidence: float = 0.0,
        match_radius: float = DEFAULT_MATCH_RADIUS,
        rejoin_window: float = DEFAULT_REJOIN_WINDOW,
    ) -> None:
        if visibility_timeout <= 0:
            raise ValueError("visibility_timeout must be > 0")
        self.visibility_timeout = visibility_timeout
        self.max_history = max_history
        self.min_confidence = min_confidence
        self.match_radius = match_radius
        self.rejoin_window = rejoin_window
        self._tracks: dict[int, PersonTrack] = {}
        self._next_id = 1

    # ── ingest ────────────────────────────────────────────────────────────────

    def observe(self, detection: Detection) -> Optional[PersonTrack]:
        """Fold one detection in. Returns the track it landed on, or None if
        the detection was dropped (below the confidence floor)."""
        if detection.confidence < self.min_confidence:
            return None

        track = self._match(detection)
        if track is None:
            track = self._create(detection)
        else:
            self._update(track, detection)
        return track

    def observe_many(self, detections: Iterable[Detection]) -> list[PersonTrack]:
        return [t for t in (self.observe(d) for d in detections) if t is not None]

    # ── time ──────────────────────────────────────────────────────────────────

    def tick(self, now: Optional[float] = None) -> list[StateEvent]:
        """Age tracks out of visibility. Call this on a timer, not only on
        detections — otherwise nobody ever goes away once detections stop."""
        now = self._now(now)
        events: list[StateEvent] = []
        for track in self._tracks.values():
            if track.visible and (now - track.last_seen) > self.visibility_timeout:
                track.visible = False
                events.append(
                    StateEvent(
                        kind="away",
                        track_id=track.track_id,
                        identity=track.identity,
                        camera=track.camera,
                        timestamp=now,
                        detail=f"not seen for {now - track.last_seen:.1f}s",
                    )
                )
        self._prune()
        return events

    # ── query ─────────────────────────────────────────────────────────────────

    def snapshot(self, now: Optional[float] = None) -> dict:
        """The full world state, JSON-serialisable."""
        now = self._now(now)
        people = sorted(
            self._tracks.values(),
            key=lambda t: (not t.visible, -t.last_seen),
        )
        present = [t for t in people if t.visible]
        return {
            "stamp": round(now, 3),
            "visibility_timeout": self.visibility_timeout,
            "present_count": len(present),
            "present": [t.identity or f"person_{t.track_id}" for t in present],
            "known_present": sorted(t.identity for t in present if t.is_identified),
            "cameras": sorted({t.camera for t in present}),
            "people": [t.as_dict(now) for t in people],
        }

    def present(self, now: Optional[float] = None) -> list[PersonTrack]:
        self.tick(now)
        return [t for t in self._tracks.values() if t.visible]

    def find(self, identity: str) -> Optional[PersonTrack]:
        """Most recently seen track for an identity, present or away."""
        matches = [t for t in self._tracks.values() if t.identity == identity]
        return max(matches, key=lambda t: t.last_seen) if matches else None

    @property
    def tracks(self) -> list[PersonTrack]:
        return list(self._tracks.values())

    # ── internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _now(now: Optional[float]) -> float:
        return time.time() if now is None else now

    def _match(self, det: Detection) -> Optional[PersonTrack]:
        if det.identity:
            by_identity = self.find(det.identity)
            if by_identity is not None:
                # The same person, possibly on a different camera. Fold in any
                # anonymous track standing in for them on this camera — by
                # source track, or by the body box their face sits inside —
                # rather than leaving a phantom behind.
                for anon in (self._by_source(det.camera, det.source_track),
                             self._contained_by(det)):
                    if anon is not None and anon is not by_identity and not anon.identity:
                        self._absorb(by_identity, anon)
                return by_identity
            # First sighting of this identity — but it may already be tracked
            # anonymously on this camera, in which case name that track.
            anon = self._by_source(det.camera, det.source_track)
            if anon is not None and not anon.identity:
                return anon
            contained = self._contained_by(det)
            if contained is not None and not contained.identity:
                return contained
            return None

        by_source = self._by_source(det.camera, det.source_track)
        if by_source is not None:
            return by_source
        # A body box that *contains* an existing track's face is that person —
        # including an already-named one. Without this, the person box and the
        # recognised face count as two people standing in the same spot.
        containing = self._track_containing(det)
        if containing is not None:
            return containing
        return self._nearest_anonymous(det)

    def _candidates(self, det: Detection):
        """Tracks on this camera eligible to absorb this detection: not already
        updated by this same frame, and either still visible or only recently
        away.

        The rejoin window matters more than it looks. Detections flicker — a
        face drops out for a few seconds and comes back. Matching only
        *visible* tracks mints a brand new person on every flicker, so a single
        stationary false detection became four "people" in 60 seconds on the
        first live run."""
        for track in self._tracks.values():
            if (track.camera != det.camera or track.bbox is None
                    or track.last_seen == det.timestamp):
                continue
            age = det.timestamp - track.last_seen
            if track.visible or age <= self.rejoin_window:
                yield track

    def _track_containing(self, det: Detection) -> Optional[PersonTrack]:
        """The track whose stored box centre falls inside this detection's box
        (a person box swallowing a known face). Nearest centre wins when boxes
        overlap."""
        if det.bbox is None:
            return None
        cx, cy, bw, bh = det.bbox
        best: Optional[PersonTrack] = None
        best_dist = float("inf")
        for track in self._candidates(det):
            tx, ty = track.bbox[0], track.bbox[1]
            if abs(tx - cx) <= bw / 2 and abs(ty - cy) <= bh / 2:
                dist = ((tx - cx) ** 2 + (ty - cy) ** 2) ** 0.5
                if dist < best_dist:
                    best, best_dist = track, dist
        return best

    def _contained_by(self, det: Detection) -> Optional[PersonTrack]:
        """The track whose stored box *contains* this detection's centre (a
        face landing inside an already-tracked body)."""
        if det.bbox is None:
            return None
        cx, cy = det.bbox[0], det.bbox[1]
        for track in self._candidates(det):
            tx, ty, tw, th = track.bbox
            if abs(tx - cx) <= tw / 2 and abs(ty - cy) <= th / 2:
                return track
        return None

    def _nearest_anonymous(self, det: Detection) -> Optional[PersonTrack]:
        """Associate an untracked box with the closest anonymous track on the
        same camera. Tracks already updated at this exact timestamp are skipped
        so two boxes in one frame cannot collapse into a single person."""
        if det.bbox is None or self.match_radius <= 0:
            return None
        cx, cy = det.bbox[0], det.bbox[1]
        best: Optional[PersonTrack] = None
        best_dist = self.match_radius
        for track in self._candidates(det):
            if track.identity:
                continue
            dist = ((track.bbox[0] - cx) ** 2 + (track.bbox[1] - cy) ** 2) ** 0.5
            if dist <= best_dist:
                best, best_dist = track, dist
        return best

    def _by_source(self, camera: str, source_track: Optional[int]) -> Optional[PersonTrack]:
        if source_track is None:
            return None
        for track in self._tracks.values():
            if track.camera == camera and track.source_track == source_track:
                return track
        return None

    def _create(self, det: Detection) -> PersonTrack:
        track = PersonTrack(
            track_id=self._next_id,
            identity=det.identity,
            confidence=det.confidence,
            camera=det.camera,
            first_seen=det.timestamp,
            last_seen=det.timestamp,
            visible=True,
            bbox=det.bbox,
            cameras_seen=[det.camera],
            source_track=det.source_track,
        )
        self._tracks[track.track_id] = track
        self._next_id += 1
        return track

    def _update(self, track: PersonTrack, det: Detection) -> None:
        if det.identity and not track.identity:
            track.identity = det.identity          # anonymous track just got named
        track.confidence = det.confidence
        track.camera = det.camera
        track.source_track = det.source_track
        track.bbox = det.bbox
        track.last_seen = max(track.last_seen, det.timestamp)
        track.first_seen = min(track.first_seen, det.timestamp)
        track.visible = True
        if det.camera not in track.cameras_seen:
            track.cameras_seen.append(det.camera)

    def _absorb(self, keep: PersonTrack, drop: PersonTrack) -> None:
        keep.first_seen = min(keep.first_seen, drop.first_seen)
        keep.last_seen = max(keep.last_seen, drop.last_seen)
        for cam in drop.cameras_seen:
            if cam not in keep.cameras_seen:
                keep.cameras_seen.append(cam)
        self._tracks.pop(drop.track_id, None)

    def _prune(self) -> None:
        """Cap retained history. Identified people are never dropped — knowing
        Rafael left an hour ago is the point. Anonymous strangers are."""
        anon_away = sorted(
            (t for t in self._tracks.values() if not t.visible and not t.is_identified),
            key=lambda t: t.last_seen,
        )
        excess = len(anon_away) - self.max_history
        for track in anon_away[:max(0, excess)]:
            self._tracks.pop(track.track_id, None)
