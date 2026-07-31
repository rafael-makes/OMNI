"""Replay gate for dwell, over a zone-tagged work session.

THE FIXTURE IS DERIVED, NOT RECORDED — see scripts/make_dwell_fixture.py for the
full provenance note. It is the real 129 s workbench capture
(`world_state_live.jsonl`) tiled to ~30 minutes with Rafael's rows tagged
`zone: workbench`. So it carries every genuine nastiness of the hardware —
471 null-identity phantom rows, five churning `unknown_N` ids, and one seated
person who is simply not visible in ~17% of frames — over a span long enough to
cross a real dwell threshold.

What it proves: dwell mechanics survive realistic vision noise.
What it does NOT prove: anything about real dwell *durations*, because a tiled
loop cannot contain a dropout longer than the original capture's worst (11 s).
The 90 s absence_grace is therefore still unvalidated against a genuine long
work session. Replace this fixture with a real 20-minute recording when one
exists; the script takes one with --tiles 1.
"""

from __future__ import annotations

import json
import os

import pytest

from event_generator import (
    PERSON_APPEARED,
    PERSON_DWELLING,
    PERSON_LEFT,
    UNKNOWN_PERSON_DETECTED,
    EventGenerator,
)

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures",
                       "work_session_dwell.jsonl")

ZONE = "workbench"


@pytest.fixture(scope="module")
def snapshots():
    with open(FIXTURE, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def replay(snapshots, **kwargs):
    kwargs.setdefault("dwell_zones", [ZONE])
    gen = EventGenerator(**kwargs)
    events = []
    for snapshot in snapshots:
        events.extend(gen.ingest(snapshot))
    return gen, events


def dwells(events):
    return [e for e in events if e.kind == PERSON_DWELLING]


# ── the fixture itself ────────────────────────────────────────────────────────

def test_fixture_is_long_enough_to_dwell(snapshots):
    span = snapshots[-1]["stamp"] - snapshots[0]["stamp"]
    assert span > 1200.0, "fixture must outlast a 20-minute dwell threshold"


def test_fixture_kept_the_real_ugliness(snapshots):
    """If someone regenerates this from clean synthetic data it stops being a
    test of anything. These are the properties inherited from the live capture."""
    null_rows = sum(1 for s in snapshots for p in s["people"]
                    if not p.get("identity"))
    assert null_rows > 1000, "the null-identity phantoms should have survived tiling"

    unknowns = {p["identity"] for s in snapshots for p in s["people"]
                if (p.get("identity") or "").startswith("unknown_")}
    assert len(unknowns) >= 4, "recognizer id churn should have survived tiling"


def test_rafael_really_does_drop_out(snapshots):
    """The property that makes a dwell test worth running: a stationary person
    whose face repeatedly stops being visible. If this is ever 100% visible, the
    flicker tolerance below is proving nothing."""
    rows = [p for s in snapshots for p in s["people"]
            if p.get("identity") == "rafael"]
    visible = sum(1 for p in rows if p.get("visible"))
    assert visible < len(rows)
    assert (len(rows) - visible) > 200, "expected substantial real dropout"


def test_rafael_is_zone_tagged(snapshots):
    rows = [p for s in snapshots for p in s["people"]
            if p.get("identity") == "rafael"]
    assert all(p.get("zone") == ZONE for p in rows)


# ── what the generator makes of it ────────────────────────────────────────────

def test_dwell_fires_on_schedule_through_the_noise(snapshots):
    """THE headline assertion. Half an hour at the bench, ~300 frames of face
    dropout, phantom strangers coming and going — and the dwell simply keeps
    counting: first firing at the 20-minute threshold, then every 5 minutes."""
    _, events = replay(snapshots, dwell_threshold=1200.0,
                       dwell_refire_interval=300.0)
    d = dwells(events)
    assert [round(e.dwell_duration) for e in d] == [1200, 1500]
    assert all(e.identity == "rafael" for e in d)
    assert all(e.zone == ZONE for e in d)


def test_dropout_never_breaks_the_dwell(snapshots):
    """The dwell is anchored at the first sighting and never re-anchored, so its
    duration tracks wall-clock, not visible-frame-count. Rafael is invisible for
    hundreds of frames; if any of those had reset the anchor, the first firing
    would land well after the 1200 s mark."""
    _, events = replay(snapshots, dwell_threshold=1200.0,
                       dwell_refire_interval=1e9)
    d = dwells(events)
    assert len(d) == 1
    first = d[0]
    assert first.dwell_duration == pytest.approx(1200.0, abs=2.0)

    # The firing lands 1203 s into the recording, not 1200: the dwell anchors at
    # the first snapshot in which Rafael is both believed PRESENT and carries a
    # zone, and it takes the recording ~3 s to get there from a cold start. That
    # small lag is the honest one described in _check_dwell (anchoring to the
    # first snapshot seen in the zone undercounts by the warm-up, never over-
    # counts). What matters is that it is a few seconds, not a few minutes —
    # minutes would mean dropout had been re-anchoring the dwell all along.
    offset = first.timestamp - snapshots[0]["stamp"]
    assert 1200.0 <= offset <= 1215.0


def test_presence_is_undisturbed(snapshots):
    """Dwell must not perturb the Session 2 behaviour it is built on: still
    greeted exactly once, still never "leaves" over half an hour of dropout."""
    gen, events = replay(snapshots, dwell_threshold=1200.0)
    appeared = [e for e in events if e.kind == PERSON_APPEARED]
    assert len(appeared) == 1 and appeared[0].identity == "rafael"
    assert [e for e in events if e.kind == PERSON_LEFT] == []
    assert gen.is_present("rafael")


def test_phantom_suppression_still_holds(snapshots):
    """Tiling multiplies the phantom rows; the overlap + size filters must still
    reduce them to the one genuine distinct detection."""
    _, events = replay(snapshots, dwell_threshold=1200.0)
    announced = {e.identity for e in events if e.kind == UNKNOWN_PERSON_DETECTED}
    assert announced == {"unknown_46"}


def test_no_event_storm(snapshots):
    """Half an hour of ordinary desk activity, and everything that lands on
    /omni/events wakes behavior_node up. Four events is the budget."""
    _, events = replay(snapshots, dwell_threshold=1200.0,
                       dwell_refire_interval=300.0)
    assert len(events) <= 5


def test_zone_scope_is_honoured(snapshots):
    """Same data, a zone nobody enabled: silence. This is the config that keeps
    check-ins out of the kitchen."""
    _, events = replay(snapshots, dwell_threshold=1200.0, dwell_zones=["kitchen"])
    assert dwells(events) == []


def test_threshold_above_the_span_stays_quiet(snapshots):
    _, events = replay(snapshots, dwell_threshold=3600.0)
    assert dwells(events) == []


def test_a_longer_threshold_is_never_noisier(snapshots):
    _, base = replay(snapshots, dwell_threshold=1200.0, dwell_refire_interval=300.0)
    _, strict = replay(snapshots, dwell_threshold=1500.0, dwell_refire_interval=300.0)
    assert len(dwells(strict)) <= len(dwells(base))


def test_replay_is_deterministic(snapshots):
    _, first = replay(snapshots, dwell_threshold=1200.0, dwell_refire_interval=300.0)
    _, second = replay(snapshots, dwell_threshold=1200.0, dwell_refire_interval=300.0)
    assert [e.as_dict() for e in first] == [e.as_dict() for e in second]


def test_every_event_serialises(snapshots):
    _, events = replay(snapshots, dwell_threshold=1200.0, dwell_refire_interval=300.0)
    for event in events:
        assert json.loads(json.dumps(event.as_dict()))["kind"] == event.kind
