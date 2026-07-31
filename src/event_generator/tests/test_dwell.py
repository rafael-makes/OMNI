"""Dwell events (Session 9): person_dwelling from continuous same-zone presence.

The three properties the spec calls out — flicker does not break a dwell, a zone
change resets it, and it re-fires on a cadence — plus the guards around them
(threshold, enabled-zone scope, sustained absence, unnamed people, no-localisation).

These are synthetic: they describe how the debounce is *meant* to behave, in the
same spirit as test_generator.py. The nastier "what the hardware actually emits"
coverage is in test_dwell_replay.py, over a zone-tagged capture.

Every stamp is an integer second so the threshold arithmetic is exact: a dwell
anchored at t0 crosses a 60 s threshold at exactly t0 + 60.
"""

from __future__ import annotations

import json

import pytest

from event_generator import (
    PERSON_APPEARED,
    PERSON_DWELLING,
    PERSON_LEFT,
    EventGenerator,
)

WORKBENCH = "workbench"
BBOX = [600.0, 180.0, 80.0, 110.0]


def _snapshot(stamp, identity, *, visible, zone, last_seen, camera="head"):
    row = {
        "identity": identity,
        "camera": camera,
        "visible": visible,
        "bbox": list(BBOX),
        "last_seen": float(last_seen),
    }
    # A real "no pose this frame" row omits zone (or sends null); model that by
    # simply not adding the key when zone is None.
    if zone is not None:
        row["zone"] = zone
    return {"stamp": float(stamp), "people": [row]}


def build(segments, *, t0=1000.0, identity="rafael"):
    """Per-second snapshots from (duration_s, visible, zone) segments.

    last_seen tracks the most recent *visible* stamp, exactly as world_state does:
    a person who has turned away keeps reporting the same last_seen while their
    seconds_since_seen climbs, which is what a face-anchored dropout looks like.
    """
    snaps = []
    t = t0
    last_seen = t0
    for secs, visible, zone in segments:
        for _ in range(int(secs)):
            if visible:
                last_seen = t
            snaps.append(
                _snapshot(t, identity, visible=visible, zone=zone, last_seen=last_seen))
            t += 1.0
    return snaps


def run(snaps, **kwargs):
    kwargs.setdefault("dwell_zones", [WORKBENCH])
    gen = EventGenerator(**kwargs)
    events = []
    for s in snaps:
        events.extend(gen.ingest(s))
    return gen, events


def dwells(events):
    return [e for e in events if e.kind == PERSON_DWELLING]


# ── the happy path ────────────────────────────────────────────────────────────

def test_fires_once_after_threshold():
    snaps = build([(70, True, WORKBENCH)])
    _, events = run(snaps, dwell_threshold=60.0, dwell_refire_interval=1e9)
    d = dwells(events)
    assert len(d) == 1
    assert d[0].identity == "rafael"
    assert d[0].zone == WORKBENCH
    # Anchored at t0=1000, crosses 60 at 1060 → duration is exactly 60.
    assert d[0].dwell_duration == pytest.approx(60.0)
    assert d[0].timestamp == pytest.approx(1060.0)


def test_does_not_fire_before_threshold():
    snaps = build([(50, True, WORKBENCH)])   # 50s < 60s threshold
    _, events = run(snaps, dwell_threshold=60.0)
    assert dwells(events) == []


def test_default_zones_are_empty_so_nothing_fires():
    """Opt-in by default: a full session at the bench emits no dwell until the
    operator names the zone. Mirrors omni_zones shipping an empty config."""
    snaps = build([(300, True, WORKBENCH)])
    gen = EventGenerator(dwell_threshold=60.0)   # no dwell_zones passed
    events = [e for s in snaps for e in gen.ingest(s)]
    assert dwells(events) == []


# ── property 1: flicker does not break a dwell ────────────────────────────────

def test_flicker_does_not_break_dwell():
    """Rafael turns away every other second — visible/invisible/visible — for a
    span that dwarfs the threshold. He never leaves PRESENT, so the dwell never
    resets, and it fires exactly once, anchored at the very first sighting."""
    segs = []
    for _ in range(40):
        segs.append((1, True, WORKBENCH))
        segs.append((1, False, WORKBENCH))   # head-down, still in the zone
    _, events = run(build(segs), dwell_threshold=60.0, dwell_refire_interval=1e9,
                    absence_grace=90.0)
    assert [e for e in events if e.kind == PERSON_LEFT] == []
    assert len([e for e in events if e.kind == PERSON_APPEARED]) == 1
    d = dwells(events)
    assert len(d) == 1
    # Anchored at t0 despite ~40 dropouts along the way — the duration reflects
    # the full span, not just the visible frames.
    assert d[0].dwell_duration >= 60.0


def test_flicker_gap_just_under_grace_is_survived():
    """One long turn-away, 80 s, below the 90 s absence_grace: still one dwell,
    still no departure."""
    segs = [(30, True, WORKBENCH), (80, False, WORKBENCH), (30, True, WORKBENCH)]
    _, events = run(build(segs), dwell_threshold=60.0, dwell_refire_interval=1e9,
                    absence_grace=90.0)
    assert [e for e in events if e.kind == PERSON_LEFT] == []
    assert len(dwells(events)) == 1


# ── property 2: a zone change resets the dwell ────────────────────────────────

def test_zone_change_resets_dwell():
    """40 s at the bench, a 10 s trip to the (non-enabled) kitchen, back to the
    bench. Credit does not carry across the trip: the dwell must accrue a fresh
    full threshold after the return, so the one firing lands after return+60, not
    at continuous-60."""
    segs = [(40, True, WORKBENCH), (10, True, "kitchen"), (80, True, WORKBENCH)]
    _, events = run(build(segs), dwell_threshold=60.0, dwell_refire_interval=1e9)
    d = dwells(events)
    assert len(d) == 1
    # Return to the bench happens at t0+50 = 1050; fires at 1050+60 = 1110.
    assert d[0].timestamp == pytest.approx(1110.0)
    assert d[0].dwell_duration == pytest.approx(60.0)   # not ~110


def test_move_to_enabled_zone_starts_fresh_dwell():
    """A second enabled zone is tracked independently."""
    segs = [(40, True, WORKBENCH), (70, True, "desk")]
    _, events = run(build(segs), dwell_threshold=60.0, dwell_zones=[WORKBENCH, "desk"])
    d = dwells(events)
    assert len(d) == 1
    assert d[0].zone == "desk"


# ── property 3: re-fire cadence ───────────────────────────────────────────────

def test_refire_cadence():
    """Threshold 60, re-fire every 30: fires at durations 60, 90, 120, 150, 180."""
    snaps = build([(201, True, WORKBENCH)])   # t0..t0+200
    _, events = run(snaps, dwell_threshold=60.0, dwell_refire_interval=30.0)
    d = dwells(events)
    assert [round(e.dwell_duration) for e in d] == [60, 90, 120, 150, 180]
    # Strictly increasing, and each carries the grown duration for the policy.
    stamps = [e.timestamp for e in d]
    assert stamps == sorted(stamps)


def test_refire_interval_not_yet_elapsed_stays_quiet():
    snaps = build([(75, True, WORKBENCH)])
    _, events = run(snaps, dwell_threshold=60.0, dwell_refire_interval=30.0)
    # Only the first crossing (60) — the next is due at 90, past the end.
    assert len(dwells(events)) == 1


# ── guards ────────────────────────────────────────────────────────────────────

def test_non_enabled_zone_never_fires():
    snaps = build([(300, True, "kitchen")])
    _, events = run(snaps, dwell_threshold=60.0)   # only workbench enabled
    assert dwells(events) == []


def test_unplaced_person_never_fires_or_anchors():
    """Present the whole time but never localised (zone always None)."""
    snaps = build([(300, True, None)])
    _, events = run(snaps, dwell_threshold=60.0)
    assert dwells(events) == []


def test_null_zone_frames_do_not_reset_a_running_dwell():
    """A few no-pose frames mid-dwell must not wipe the last-known zone. The
    dwell counts straight through them and still fires at continuous-60."""
    segs = [(40, True, WORKBENCH), (5, True, None), (25, True, WORKBENCH)]
    _, events = run(build(segs), dwell_threshold=60.0, dwell_refire_interval=1e9)
    d = dwells(events)
    assert len(d) == 1
    assert d[0].timestamp == pytest.approx(1060.0)   # uninterrupted


def test_dwell_accrues_through_a_grace_covered_dropout():
    """Deliberate, and a consequence of the face-anchored design: a dwell can
    cross its threshold while the person is *not currently visible*, as long as
    they are still inside absence_grace and therefore still believed PRESENT.

    Last seen at t0+39, threshold crossed at t0+60, grace not exhausted until
    t0+129 — so it fires mid-dropout. That is correct: someone bent over the
    bench has not stopped being at the bench. Pinned because the obvious
    "only fire while visible" tightening would break the whole feature on real
    hardware, where a stationary person is invisible ~20% of the time."""
    segs = [(40, True, WORKBENCH), (60, False, WORKBENCH)]
    _, events = run(build(segs), dwell_threshold=60.0, dwell_refire_interval=1e9,
                    absence_grace=90.0)
    assert [e for e in events if e.kind == PERSON_LEFT] == []
    d = dwells(events)
    assert len(d) == 1
    assert d[0].timestamp == pytest.approx(1060.0)


def test_sustained_absence_resets_dwell():
    """Gone past the grace window → person_left, and the dwell resets so the
    return has to earn a fresh threshold.

    Threshold is 200 s here specifically so the first stint cannot cross it even
    with the full 90 s grace added on (40 + 90 = 130 < 200). That isolates the
    reset: the single firing must be 200 s after the RETURN, not 200 s after the
    original arrival — which would have been t0+200 = 1200, before it lands."""
    segs = [
        (40, True, WORKBENCH),
        (120, False, WORKBENCH),   # 120s > 90s grace → left at 1130
        (250, True, WORKBENCH),    # back at 1160
    ]
    _, events = run(build(segs), dwell_threshold=200.0, dwell_refire_interval=1e9,
                    absence_grace=90.0)
    assert any(e.kind == PERSON_LEFT for e in events)
    d = dwells(events)
    assert len(d) == 1
    # Fresh anchor at the return (1160) + 200 = 1360. Not 1200.
    assert d[0].timestamp == pytest.approx(1360.0)
    assert d[0].dwell_duration == pytest.approx(200.0)


def test_unnamed_people_never_dwell():
    """A stable stranger sitting still is not a check-in candidate."""
    snaps = build([(300, True, WORKBENCH)], identity="unknown_5")
    _, events = run(snaps, dwell_threshold=60.0)
    assert dwells(events) == []


# ── serialisation + validation ────────────────────────────────────────────────

def test_dwell_event_serialises_with_its_fields():
    snaps = build([(70, True, WORKBENCH)])
    _, events = run(snaps, dwell_threshold=60.0)
    d = dwells(events)[0].as_dict()
    round_tripped = json.loads(json.dumps(d))
    assert round_tripped["kind"] == PERSON_DWELLING
    assert round_tripped["zone"] == WORKBENCH
    assert round_tripped["dwell_duration"] == pytest.approx(60.0)
    assert round_tripped["away_duration"] is None


def test_presence_events_carry_null_dwell_fields():
    """Adding the fields must not change what person_appeared serialises to
    beyond two new explicit nulls."""
    snaps = build([(2, True, WORKBENCH)])
    _, events = run(snaps, dwell_threshold=60.0)
    appeared = next(e for e in events if e.kind == PERSON_APPEARED).as_dict()
    assert appeared["zone"] is None
    assert appeared["dwell_duration"] is None


@pytest.mark.parametrize("kwargs", [
    {"dwell_threshold": 0.0},
    {"dwell_threshold": -1.0},
    {"dwell_refire_interval": 0.0},
])
def test_invalid_dwell_config_rejected(kwargs):
    with pytest.raises(ValueError):
        EventGenerator(**kwargs)
