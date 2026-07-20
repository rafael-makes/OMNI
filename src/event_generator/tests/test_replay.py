"""Replay gate against a captured live world_state sequence.

The fixture is real: 130 snapshots, 129 seconds, recorded 2026-07-19 from the
Jetson vision stack with Rafael sitting at the workbench. Capture it again with

    ros2 launch event_generator event_generator.launch.py \
        record_path:=/tmp/ws_capture.jsonl

This is the test that earns its keep. The synthetic tests in test_generator.py
describe how the debounce is *supposed* to behave; this one describes what the
hardware actually emits, which turned out to be considerably nastier:

  * 471 rows with a null identity across 130 snapshots — the known upstream
    duplicate-face bug, in volume.
  * Rafael visible in only 104 of the 127 snapshots he appears in. 23 snapshots
    of face dropout from one person sitting still.
  * Five different `unknown_N` ids minted over two minutes, most of them sitting
    on top of Rafael's own face because the recognizer drops to a fresh anonymous
    id whenever confidence dips.

The first cut of this library announced FIVE strangers on this data. Everything
the overlap suppression does exists because of this file.
"""

from __future__ import annotations

import json
import os

import pytest

from event_generator import (
    PERSON_APPEARED,
    UNKNOWN_PERSON_DETECTED,
    EventGenerator,
)

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures",
                       "world_state_live.jsonl")


@pytest.fixture(scope="module")
def snapshots():
    with open(FIXTURE, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def replay(snapshots, **kwargs):
    gen = EventGenerator(**kwargs)
    events = []
    for snapshot in snapshots:
        events.extend(gen.ingest(snapshot))
    return gen, events


# ── the fixture itself ────────────────────────────────────────────────────────

def test_fixture_is_the_real_thing(snapshots):
    """Guards against someone regenerating this from synthetic data."""
    assert len(snapshots) > 100
    span = snapshots[-1]["stamp"] - snapshots[0]["stamp"]
    assert span > 100.0

    null_rows = sum(
        1 for s in snapshots for p in s["people"] if not p.get("identity"))
    assert null_rows > 100, "fixture should contain the null-identity phantoms"

    unknowns = {p["identity"] for s in snapshots for p in s["people"]
                if (p.get("identity") or "").startswith("unknown_")}
    assert len(unknowns) >= 4, "fixture should contain recognizer id churn"

    assert any(p.get("identity") == "rafael"
               for s in snapshots for p in s["people"])


def test_rafael_has_real_face_dropout(snapshots):
    """The property that makes this fixture worth having: a stationary person
    whose face repeatedly stops being visible."""
    total = sum(1 for s in snapshots for p in s["people"]
                if p.get("identity") == "rafael")
    visible = sum(1 for s in snapshots for p in s["people"]
                  if p.get("identity") == "rafael" and p.get("visible"))
    assert visible < total, "no dropout in fixture — it proves nothing"
    assert total - visible >= 10


# ── what the generator makes of it ────────────────────────────────────────────

def test_rafael_is_greeted_exactly_once(snapshots):
    """THE headline assertion. One person, present throughout, 23 snapshots of
    face dropout — and exactly one person_appeared. If this ever becomes two,
    OMNI has started greeting someone every time they look down."""
    _, events = replay(snapshots)
    appeared = [e for e in events if e.kind == PERSON_APPEARED]
    assert len(appeared) == 1
    assert appeared[0].identity == "rafael"
    assert appeared[0].away_duration is None   # first sighting of the run


def test_rafael_never_leaves(snapshots):
    """He was there the whole time. Any person_left here is the face-anchored
    trap firing — a dropout mistaken for a departure."""
    gen, events = replay(snapshots)
    assert [e for e in events if e.kind == "person_left"] == []
    assert gen.is_present("rafael")


def test_phantom_strangers_are_suppressed(snapshots):
    """Five unknown_N ids churn through this recording; four of them are
    Rafael's own face. Only unknown_46 — which lives at (321, 530), nowhere near
    him, and stays put — survives as a genuine distinct detection."""
    _, events = replay(snapshots)
    announced = [e.identity for e in events
                 if e.kind == UNKNOWN_PERSON_DETECTED]
    assert announced == ["unknown_46"]


def test_overlap_suppression_is_what_does_it(snapshots):
    """Pin the mechanism, not just the outcome: with the radius disabled the
    phantoms come back. Without this, someone could 'simplify' the overlap check
    away and every other assertion here would still pass."""
    _, events = replay(snapshots, named_overlap_radius=0.0)
    announced = [e.identity for e in events
                 if e.kind == UNKNOWN_PERSON_DETECTED]
    assert len(announced) > 1


def test_no_event_storm(snapshots):
    """130 snapshots of ordinary desk activity must not produce a stream of
    events. Anything that lands on /omni/events wakes behavior_node up."""
    _, events = replay(snapshots)
    assert len(events) <= 3


def test_replay_is_deterministic(snapshots):
    _, first = replay(snapshots)
    _, second = replay(snapshots)
    assert [e.as_dict() for e in first] == [e.as_dict() for e in second]


def test_every_event_serialises(snapshots):
    """These go onto a String topic as JSON; a non-serialisable field would
    only show up at runtime."""
    _, events = replay(snapshots)
    for event in events:
        assert json.loads(json.dumps(event.as_dict()))["kind"] == event.kind


def test_a_stricter_threshold_stays_quiet(snapshots):
    """Raising unknown_min_snapshots must never make it noisier."""
    _, base = replay(snapshots)
    _, strict = replay(snapshots, unknown_min_snapshots=8)
    assert len(strict) <= len(base)
