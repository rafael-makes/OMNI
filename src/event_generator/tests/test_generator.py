"""Debounce tests for EventGenerator. Pure Python — passes with the robot off.

Every test here is a statement about the face-anchored constraint: visibility is
not presence. If one of these starts failing, the question to ask is not "what
changed in the test" but "did someone make a dropped face mean the person left".
"""

from __future__ import annotations

import pytest

from event_generator import (
    PERSON_APPEARED,
    PERSON_LEFT,
    UNKNOWN_PERSON_DETECTED,
    EventGenerator,
)


# ── snapshot builders ─────────────────────────────────────────────────────────

def row(identity, *, visible=True, last_seen=0.0, camera="head", track_id=1):
    """One entry of the snapshot's `people` list, shaped like world_state's."""
    return {
        "track_id": track_id,
        "identity": identity,
        "identified": bool(identity) and not str(identity).startswith("unknown_"),
        "confidence": 1.0,
        "camera": camera,
        "cameras_seen": [camera],
        "first_seen": 0.0,
        "last_seen": last_seen,
        "seconds_since_seen": 0.0,
        "visible": visible,
        "bbox": [320.0, 240.0, 80.0, 80.0],
    }


def snap(stamp, people):
    return {"stamp": stamp, "people": people}


def kinds(events):
    return [e.kind for e in events]


def run(gen, snapshots):
    """Feed a list of snapshots, return every event in order."""
    out = []
    for s in snapshots:
        out.extend(gen.ingest(s))
    return out


# ── appearing ─────────────────────────────────────────────────────────────────

def test_first_sighting_fires_appeared_with_no_away_duration():
    gen = EventGenerator()
    events = gen.ingest(snap(100.0, [row("rafael", last_seen=100.0)]))
    assert kinds(events) == [PERSON_APPEARED]
    assert events[0].identity == "rafael"
    # None, not 0.0 — "no history" and "returned instantly" are different facts,
    # and the greeting logic keys on that difference.
    assert events[0].away_duration is None


def test_appeared_fires_once_not_every_snapshot():
    gen = EventGenerator()
    events = run(gen, [
        snap(100.0 + i, [row("rafael", last_seen=100.0 + i)]) for i in range(30)
    ])
    assert kinds(events) == [PERSON_APPEARED]


def test_unnamed_row_never_generates_events():
    """The known upstream duplicate-face bug: a null-identity row beside a real
    person. It is perfectly stable, so no debounce can filter it — it must
    simply never be looked at."""
    gen = EventGenerator()
    events = run(gen, [
        snap(100.0 + i, [
            row("rafael", last_seen=100.0 + i),
            row(None, last_seen=100.0 + i, track_id=2),
            row("", last_seen=100.0 + i, track_id=3),
        ])
        for i in range(10)
    ])
    assert kinds(events) == [PERSON_APPEARED]
    assert events[0].identity == "rafael"


# ── the turn-away case ────────────────────────────────────────────────────────

def test_turning_away_does_not_fire_left():
    """The workbench case. Face drops out for 30s (well inside the 90s grace),
    person never stops being PRESENT."""
    gen = EventGenerator()
    events = run(gen, [snap(100.0, [row("rafael", last_seen=100.0)])])
    assert kinds(events) == [PERSON_APPEARED]

    # 30 snapshots where the row is present but not visible and last_seen frozen.
    events = run(gen, [
        snap(101.0 + i, [row("rafael", visible=False, last_seen=100.0)])
        for i in range(30)
    ])
    assert events == []
    assert gen.is_present("rafael")


def test_turn_away_and_back_does_not_re_greet():
    """Explicitly the live verification case: turn away at the workbench, turn
    back, no second greeting."""
    gen = EventGenerator()
    run(gen, [snap(100.0, [row("rafael", last_seen=100.0)])])

    events = run(gen, (
        # away 40s
        [snap(101.0 + i, [row("rafael", visible=False, last_seen=100.0)])
         for i in range(40)]
        # then visible again
        + [snap(141.0 + i, [row("rafael", last_seen=141.0 + i)]) for i in range(5)]
    ))
    assert events == []


def test_repeated_flicker_never_re_fires_appeared():
    """Ten turn-away/turn-back cycles. Each gap is short; the total elapsed time
    is far past the grace period, which is exactly the trap — grace is measured
    from the last SIGHTING, not from the last event."""
    gen = EventGenerator()
    run(gen, [snap(100.0, [row("rafael", last_seen=100.0)])])

    t = 101.0
    events = []
    for _ in range(10):
        for _ in range(20):        # 20s out of view
            events.extend(gen.ingest(
                snap(t, [row("rafael", visible=False, last_seen=t - 20.0)])))
            t += 1.0
        for _ in range(5):         # 5s back in view
            events.extend(gen.ingest(snap(t, [row("rafael", last_seen=t)])))
            t += 1.0
    assert events == []


# ── leaving ───────────────────────────────────────────────────────────────────

def test_sustained_absence_fires_left():
    gen = EventGenerator()
    run(gen, [snap(100.0, [row("rafael", last_seen=100.0)])])

    events = run(gen, [
        snap(101.0 + i, [row("rafael", visible=False, last_seen=100.0)])
        for i in range(120)
    ])
    assert kinds(events) == [PERSON_LEFT]
    assert not gen.is_present("rafael")


def test_left_fires_exactly_once():
    gen = EventGenerator()
    run(gen, [snap(100.0, [row("rafael", last_seen=100.0)])])
    events = run(gen, [
        snap(101.0 + i, [row("rafael", visible=False, last_seen=100.0)])
        for i in range(600)
    ])
    assert kinds(events) == [PERSON_LEFT]


def test_left_fires_when_row_vanishes_entirely():
    """world_state prunes away-tracks, so a key can disappear from `people`
    rather than sitting there with visible=false. Both must mean the same."""
    gen = EventGenerator()
    run(gen, [snap(100.0, [row("rafael", last_seen=100.0)])])
    events = run(gen, [snap(101.0 + i, []) for i in range(120)])
    assert kinds(events) == [PERSON_LEFT]


def test_absence_grace_boundary_is_not_crossed_early():
    gen = EventGenerator()
    run(gen, [snap(100.0, [row("rafael", last_seen=100.0)])])
    # Exactly at the grace boundary — still inside, no event.
    assert gen.ingest(snap(190.0, [row("rafael", visible=False, last_seen=100.0)])) == []
    # One second past it — gone.
    assert kinds(gen.ingest(
        snap(191.0, [row("rafael", visible=False, last_seen=100.0)]))) == [PERSON_LEFT]


# ── returning ─────────────────────────────────────────────────────────────────

def test_return_after_long_absence_reports_true_gap():
    """away_duration is measured from the last real sighting, so it includes the
    90s grace. A 10-minute absence must read as ~600s, not ~510s — the greeting
    prompt reasons about how long they were actually gone."""
    gen = EventGenerator()
    run(gen, [snap(100.0, [row("rafael", last_seen=100.0)])])
    run(gen, [snap(101.0 + i, []) for i in range(700)])

    events = gen.ingest(snap(800.0, [row("rafael", last_seen=800.0)]))
    assert kinds(events) == [PERSON_APPEARED]
    assert events[0].away_duration == pytest.approx(700.0)


def test_sixty_second_absence_produces_no_events_at_all():
    """The 'brief absence → silence' verification case, at the library level:
    60s is inside the grace window, so there is no person_left and therefore no
    person_appeared to greet on."""
    gen = EventGenerator()
    run(gen, [snap(100.0, [row("rafael", last_seen=100.0)])])
    events = run(gen, (
        [snap(101.0 + i, []) for i in range(60)]
        + [snap(161.0, [row("rafael", last_seen=161.0)])]
    ))
    assert events == []


# ── unknown people ────────────────────────────────────────────────────────────

def test_unknown_needs_minimum_snapshots():
    gen = EventGenerator()
    assert gen.ingest(snap(100.0, [row("unknown_3", last_seen=100.0)])) == []
    assert gen.ingest(snap(101.0, [row("unknown_3", last_seen=101.0)])) == []
    events = gen.ingest(snap(102.0, [row("unknown_3", last_seen=102.0)]))
    assert kinds(events) == [UNKNOWN_PERSON_DETECTED]
    assert events[0].identity == "unknown_3"


def test_one_frame_unknown_is_never_announced():
    gen = EventGenerator()
    events = run(gen, [
        snap(100.0, [row("unknown_9", last_seen=100.0)]),
        snap(101.0, []),
        snap(102.0, []),
        snap(103.0, []),
    ])
    assert events == []


def test_unknown_announced_once_even_across_absence():
    gen = EventGenerator()
    run(gen, [snap(100.0 + i, [row("unknown_3", last_seen=100.0 + i)])
              for i in range(5)])
    # Long gone, then back — same stranger, already announced.
    events = run(gen, (
        [snap(105.0 + i, []) for i in range(200)]
        + [snap(305.0 + i, [row("unknown_3", last_seen=305.0 + i)])
           for i in range(5)]
    ))
    assert events == []


def test_unknown_leaving_is_silent():
    gen = EventGenerator()
    run(gen, [snap(100.0 + i, [row("unknown_3", last_seen=100.0 + i)])
              for i in range(5)])
    events = run(gen, [snap(105.0 + i, []) for i in range(200)])
    assert events == []   # person_left is for named people only


def test_unknown_on_rear_camera_is_not_announced():
    """Scoped to the head camera until the rear recognizer lands."""
    gen = EventGenerator()
    events = run(gen, [
        snap(100.0 + i, [row("unknown_3", last_seen=100.0 + i, camera="rear")])
        for i in range(10)
    ])
    assert events == []


def test_unknown_sitting_on_a_named_face_is_suppressed():
    """The recognizer drops to a fresh unknown_N whenever confidence dips, so a
    stranger appearing at exactly a known person's coordinates is that person's
    own face misread. Observed live: unknown_58 at (606.2, 187.9) while rafael
    sat at (606.3, 187.4), same frame, both visible."""
    gen = EventGenerator()
    events = run(gen, [
        snap(100.0 + i, [
            row("rafael", last_seen=100.0 + i),
            row("unknown_58", last_seen=100.0 + i, track_id=2),
        ])
        for i in range(20)
    ])
    assert kinds(events) == [PERSON_APPEARED]


def test_unknown_suppressed_while_the_named_face_has_dropped_out():
    """The case that actually matters: the phantom appears BECAUSE recognition
    failed, so the real person is usually not visible in that frame. Only our own
    debounced presence belief still knows they are there."""
    gen = EventGenerator()
    run(gen, [snap(100.0, [row("rafael", last_seen=100.0)])])
    events = run(gen, [
        snap(101.0 + i, [
            row("rafael", visible=False, last_seen=100.0),
            row("unknown_58", last_seen=101.0 + i, track_id=2),
        ])
        for i in range(20)
    ])
    assert events == []


def test_unknown_well_away_from_a_named_face_is_announced():
    """Suppression must not swallow a real stranger across the room."""
    gen = EventGenerator()
    events = run(gen, [
        snap(100.0 + i, [
            row("rafael", last_seen=100.0 + i),
            {**row("unknown_58", last_seen=100.0 + i, track_id=2),
             "bbox": [320.0, 530.0, 70.0, 80.0]},
        ])
        for i in range(10)
    ])
    assert kinds(events) == [PERSON_APPEARED, UNKNOWN_PERSON_DETECTED]


def test_overlap_radius_is_configurable_and_disablable():
    def scenario(gen):
        return run(gen, [
            snap(100.0 + i, [
                row("rafael", last_seen=100.0 + i),
                {**row("unknown_58", last_seen=100.0 + i, track_id=2),
                 "bbox": [420.0, 240.0, 80.0, 80.0]},   # 100px away
            ])
            for i in range(10)
        ])

    # Inside the default radius — suppressed.
    assert kinds(scenario(EventGenerator())) == [PERSON_APPEARED]
    # Radius off entirely — the boxes do not contain each other, so it fires.
    assert UNKNOWN_PERSON_DETECTED in kinds(
        scenario(EventGenerator(named_overlap_radius=0.0)))


def test_tiny_face_boxes_are_not_strangers():
    """Measured live: background clutter that YuNet scores above threshold shows
    up as 23x25px and 35x35px 'faces' at the frame edges, versus ~88x115px for a
    real face at conversational range. No amount of debouncing separates these —
    the clutter is perfectly stationary and perfectly persistent."""
    gen = EventGenerator()
    events = run(gen, [
        snap(100.0 + i, [
            {**row("unknown_18", last_seen=100.0 + i),
             "bbox": [46.1, 332.9, 34.7, 34.8]},
            {**row("unknown_27", last_seen=100.0 + i, track_id=2),
             "bbox": [1046.4, 223.6, 23.0, 25.1]},
        ])
        for i in range(20)
    ])
    assert events == []


def test_a_real_sized_face_at_the_same_spot_is_announced():
    """The size filter must not be a blanket mute on that region of the frame."""
    gen = EventGenerator()
    events = run(gen, [
        snap(100.0 + i, [
            {**row("unknown_18", last_seen=100.0 + i),
             "bbox": [46.1, 332.9, 80.0, 100.0]},
        ])
        for i in range(10)
    ])
    assert kinds(events) == [UNKNOWN_PERSON_DETECTED]


def test_face_size_is_judged_on_the_shorter_side():
    """A 20x200px sliver is a detector artifact, not a face — area would pass it."""
    gen = EventGenerator()
    events = run(gen, [
        snap(100.0 + i, [
            {**row("unknown_5", last_seen=100.0 + i),
             "bbox": [400.0, 300.0, 20.0, 200.0]},
        ])
        for i in range(10)
    ])
    assert events == []


def test_size_filter_is_disablable():
    gen = EventGenerator(unknown_min_face_px=0.0)
    events = run(gen, [
        snap(100.0 + i, [
            {**row("unknown_27", last_seen=100.0 + i),
             "bbox": [1046.4, 223.6, 23.0, 25.1]},
        ])
        for i in range(10)
    ])
    assert kinds(events) == [UNKNOWN_PERSON_DETECTED]


def test_unknown_with_no_bbox_is_not_announced():
    """No box means nothing to judge — not credible as a new arrival."""
    gen = EventGenerator()
    events = run(gen, [
        snap(100.0 + i, [{**row("unknown_5", last_seen=100.0 + i), "bbox": None}])
        for i in range(10)
    ])
    assert events == []


def test_unknown_must_be_consecutive_not_cumulative():
    """A flickering id that is visible in 1 snapshot out of every 5 never
    persists, however long it churns. Counting cumulatively announced a stranger
    on 4 sightings scattered across 98 snapshots."""
    gen = EventGenerator(unknown_min_snapshots=3)
    events = []
    for cycle in range(10):
        t = 100.0 + cycle * 5
        events.extend(gen.ingest(snap(t, [row("unknown_7", last_seen=t)])))
        for i in range(1, 5):
            events.extend(gen.ingest(snap(t + i, [])))
    assert events == []


def test_named_person_on_rear_camera_still_appears():
    """The camera scope applies to strangers only — a recognised name is worth
    greeting wherever it was seen."""
    gen = EventGenerator()
    events = gen.ingest(
        snap(100.0, [row("rafael", last_seen=100.0, camera="rear")]))
    assert kinds(events) == [PERSON_APPEARED]


def test_unknown_promoted_to_name_emits_no_departure():
    """The recognizer putting a name to a stranger is one person, not a
    departure plus an arrival."""
    gen = EventGenerator()
    run(gen, [snap(100.0 + i, [row("unknown_3", last_seen=100.0 + i)])
              for i in range(5)])
    events = run(gen, [
        snap(105.0 + i, [row("rafael", last_seen=105.0 + i, track_id=2)])
        for i in range(200)
    ])
    assert kinds(events) == [PERSON_APPEARED]
    assert events[0].away_duration is None


# ── multiple people ───────────────────────────────────────────────────────────

def test_two_people_tracked_independently():
    gen = EventGenerator()
    events = run(gen, [
        snap(100.0, [row("rafael", last_seen=100.0)]),
        snap(101.0, [row("rafael", last_seen=101.0),
                     row("zoe", last_seen=101.0, track_id=2)]),
    ])
    assert kinds(events) == [PERSON_APPEARED, PERSON_APPEARED]
    assert [e.identity for e in events] == ["rafael", "zoe"]

    # Rafael leaves, Zoe stays.
    events = run(gen, [
        snap(102.0 + i, [row("rafael", visible=False, last_seen=101.0),
                         row("zoe", last_seen=102.0 + i, track_id=2)])
        for i in range(120)
    ])
    assert kinds(events) == [PERSON_LEFT]
    assert events[0].identity == "rafael"
    assert gen.present() == ["zoe"]


# ── configuration + robustness ────────────────────────────────────────────────

def test_absence_grace_is_configurable():
    gen = EventGenerator(absence_grace=10.0)
    run(gen, [snap(100.0, [row("rafael", last_seen=100.0)])])
    events = run(gen, [snap(101.0 + i, []) for i in range(20)])
    assert kinds(events) == [PERSON_LEFT]


def test_rejects_nonsense_config():
    with pytest.raises(ValueError):
        EventGenerator(absence_grace=0.0)
    with pytest.raises(ValueError):
        EventGenerator(unknown_min_snapshots=0)


def test_malformed_snapshots_are_survived_not_raised():
    """The node hands us whatever was on the topic. A parse-level surprise must
    not take down a node that is meant to run for days."""
    gen = EventGenerator()
    assert gen.ingest({}) == []
    assert gen.ingest({"stamp": "not-a-number", "people": []}) == []
    assert gen.ingest({"stamp": 1.0, "people": None}) == []
    assert gen.ingest({"stamp": 2.0, "people": ["not-a-dict", 42]}) == []
    assert gen.ingest({"stamp": 3.0}) == []
    assert gen.ingest(None) == []
    # Still functional afterwards.
    assert kinds(gen.ingest(snap(4.0, [row("rafael", last_seen=4.0)]))) == [
        PERSON_APPEARED]


def test_out_of_order_snapshots_are_dropped():
    """A backwards stamp would make every absence gap negative and silently
    freeze all departures."""
    gen = EventGenerator()
    run(gen, [snap(500.0, [row("rafael", last_seen=500.0)])])
    assert gen.ingest(snap(100.0, [])) == []
    assert gen.is_present("rafael")


def test_row_last_seen_never_moves_backwards():
    gen = EventGenerator()
    run(gen, [snap(100.0, [row("rafael", last_seen=100.0)])])
    # A stale row reporting an older last_seen must not reset the absence clock.
    gen.ingest(snap(101.0, [row("rafael", visible=False, last_seen=50.0)]))
    assert gen.ingest(
        snap(180.0, [row("rafael", visible=False, last_seen=50.0)])) == []
