"""Unit tests for the ROS-free WorldState core. Runs with the robot off."""

import pytest

from world_state import Detection, WorldState

HEAD = "head"
REAR = "rear"


def det(t, identity=None, camera=HEAD, conf=0.9, source_track=None, bbox=None):
    return Detection(
        camera=camera,
        timestamp=t,
        identity=identity,
        confidence=conf,
        bbox=bbox,
        source_track=source_track,
    )


# ── appearance ────────────────────────────────────────────────────────────────


def test_first_detection_creates_a_visible_track():
    ws = WorldState(visibility_timeout=3.0)
    track = ws.observe(det(100.0, "rafael"))

    assert track.identity == "rafael"
    assert track.visible
    assert track.first_seen == track.last_seen == 100.0
    assert track.cameras_seen == [HEAD]


def test_repeated_detections_extend_one_track():
    ws = WorldState(visibility_timeout=3.0)
    ws.observe(det(100.0, "rafael"))
    ws.observe(det(101.0, "rafael"))
    ws.observe(det(102.0, "rafael"))

    assert len(ws.tracks) == 1
    track = ws.find("rafael")
    assert track.first_seen == 100.0
    assert track.last_seen == 102.0


def test_two_people_are_two_tracks():
    ws = WorldState()
    ws.observe(det(100.0, "rafael"))
    ws.observe(det(100.0, "zoe"))

    assert len(ws.tracks) == 2
    assert ws.snapshot(100.0)["known_present"] == ["rafael", "zoe"]


def test_anonymous_detections_on_one_camera_stay_one_track():
    ws = WorldState()
    ws.observe(det(100.0, source_track=7))
    ws.observe(det(100.5, source_track=7))

    assert len(ws.tracks) == 1
    assert ws.tracks[0].identity is None
    assert ws.tracks[0].is_identified is False


# ── disappearance / timeout ───────────────────────────────────────────────────


def test_track_goes_away_after_timeout():
    ws = WorldState(visibility_timeout=3.0)
    ws.observe(det(100.0, "rafael"))

    assert ws.tick(102.0) == []                    # still within the window
    assert ws.find("rafael").visible

    events = ws.tick(104.0)
    assert [e.kind for e in events] == ["away"]
    assert events[0].identity == "rafael"
    assert ws.find("rafael").visible is False


def test_away_person_is_kept_in_history():
    ws = WorldState(visibility_timeout=3.0)
    ws.observe(det(100.0, "rafael"))
    ws.tick(110.0)

    snap = ws.snapshot(110.0)
    assert snap["present_count"] == 0
    assert snap["known_present"] == []
    person = snap["people"][0]
    assert person["identity"] == "rafael"
    assert person["visible"] is False
    assert person["seconds_since_seen"] == pytest.approx(10.0)


def test_away_then_seen_again_resumes_the_same_track():
    ws = WorldState(visibility_timeout=3.0)
    ws.observe(det(100.0, "rafael"))
    ws.tick(200.0)
    ws.observe(det(300.0, "rafael"))

    assert len(ws.tracks) == 1
    track = ws.find("rafael")
    assert track.visible
    assert track.first_seen == 100.0               # original arrival preserved
    assert track.last_seen == 300.0


def test_tick_only_ages_the_person_who_left():
    ws = WorldState(visibility_timeout=3.0)
    ws.observe(det(100.0, "rafael"))
    ws.observe(det(100.0, "zoe"))
    ws.observe(det(105.0, "zoe"))

    events = ws.tick(106.0)
    assert [e.identity for e in events] == ["rafael"]
    assert ws.find("zoe").visible


# ── camera handoff ────────────────────────────────────────────────────────────


def test_same_identity_on_a_second_camera_is_not_a_new_person():
    ws = WorldState(visibility_timeout=3.0)
    ws.observe(det(100.0, "rafael", camera=HEAD, source_track=1))
    ws.observe(det(101.0, "rafael", camera=REAR, source_track=4))

    assert len(ws.tracks) == 1
    track = ws.find("rafael")
    assert track.camera == REAR                    # follows the latest sighting
    assert track.cameras_seen == [HEAD, REAR]
    assert track.first_seen == 100.0


def test_handoff_across_a_gap_still_resumes_one_track():
    """Walking out of the head cam and into the rear cam leaves a gap longer
    than the visibility timeout — it must still be the same person."""
    ws = WorldState(visibility_timeout=2.0)
    ws.observe(det(100.0, "rafael", camera=HEAD, source_track=1))
    ws.tick(103.0)
    assert ws.find("rafael").visible is False

    ws.observe(det(104.0, "rafael", camera=REAR, source_track=9))

    assert len(ws.tracks) == 1
    assert ws.find("rafael").visible
    assert ws.snapshot(104.0)["cameras"] == [REAR]


def test_anonymous_track_is_absorbed_when_the_recognizer_names_it():
    """A person box appears first, the face recognizer catches up a second
    later — one person, with the earlier arrival time."""
    ws = WorldState()
    ws.observe(det(100.0, camera=HEAD, source_track=3))          # unnamed
    ws.observe(det(101.0, "rafael", camera=HEAD, source_track=3))  # named

    assert len(ws.tracks) == 1
    track = ws.find("rafael")
    assert track.first_seen == 100.0
    assert track.identity == "rafael"


def test_known_person_absorbs_a_separate_anonymous_track_on_arrival():
    ws = WorldState()
    ws.observe(det(90.0, "rafael", camera=HEAD, source_track=1))   # known, head
    ws.observe(det(100.0, camera=REAR, source_track=5))            # anon, rear
    assert len(ws.tracks) == 2

    # The rear recognizer works out that rear track 5 is Rafael.
    ws.observe(det(101.0, "rafael", camera=REAR, source_track=5))

    assert len(ws.tracks) == 1
    assert ws.find("rafael").first_seen == 90.0


# ── untracked person boxes (what /camera/detections actually publishes) ───────


def test_untracked_boxes_at_the_same_place_stay_one_track():
    """head_detector never sets Detection2D.id, so anonymous person boxes
    arrive with no source track. Without centroid association each frame would
    mint a new person — at 30fps that is 30 phantom people per second."""
    ws = WorldState(match_radius=160.0)
    for i in range(30):
        ws.observe(det(100.0 + i * 0.03, bbox=(320.0 + i, 240.0, 80.0, 200.0)))

    assert len(ws.tracks) == 1


def test_untracked_boxes_far_apart_are_two_people():
    ws = WorldState(match_radius=160.0)
    ws.observe(det(100.0, bbox=(100.0, 240.0, 80.0, 200.0)))
    ws.observe(det(100.0, bbox=(600.0, 240.0, 80.0, 200.0)))

    assert len(ws.tracks) == 2


def test_two_boxes_in_one_frame_cannot_collapse_into_one_track():
    """Same timestamp = same frame. Even if the second box is within the match
    radius, it must not be folded into the track the first just updated."""
    ws = WorldState(match_radius=1000.0)
    ws.observe(det(100.0, bbox=(300.0, 240.0, 80.0, 200.0)))
    ws.observe(det(100.0, bbox=(340.0, 240.0, 80.0, 200.0)))

    assert len(ws.tracks) == 2


def test_a_walking_person_is_followed_across_frames():
    ws = WorldState(match_radius=160.0)
    for i in range(20):
        ws.observe(det(100.0 + i * 0.1, bbox=(100.0 + i * 25, 240.0, 80.0, 200.0)))

    assert len(ws.tracks) == 1
    assert ws.tracks[0].bbox[0] == pytest.approx(575.0)


def test_body_box_and_recognised_face_are_one_person():
    """/camera/detections and /camera/identities describe the same human from
    the same camera. The face box sits inside the body box, so this must read
    as one person — not a body-person plus a face-person."""
    ws = WorldState(match_radius=160.0)
    ws.observe(det(100.0, bbox=(320.0, 240.0, 80.0, 200.0)))            # YOLO body
    ws.observe(det(100.5, "rafael", bbox=(320.0, 180.0, 60.0, 60.0)))   # face inside

    assert len(ws.tracks) == 1
    assert ws.snapshot(100.5)["present_count"] == 1
    assert ws.find("rafael").first_seen == 100.0


def test_face_first_then_body_box_is_still_one_person():
    """The topics arrive in either order; the body box must not re-mint a
    second track around a face that is already named."""
    ws = WorldState(match_radius=160.0)
    ws.observe(det(100.0, "rafael", bbox=(320.0, 180.0, 60.0, 60.0)))   # face
    ws.observe(det(100.5, bbox=(320.0, 240.0, 80.0, 200.0)))            # body around it

    assert len(ws.tracks) == 1
    assert ws.snapshot(100.5)["known_present"] == ["rafael"]


def test_alternating_body_and_face_frames_do_not_churn_tracks():
    """The real failure mode: two topics at ~10Hz each. If association only
    worked one way, tracks would be created and absorbed forever."""
    ws = WorldState(match_radius=160.0)
    t = 100.0
    for _ in range(15):
        ws.observe(det(t, bbox=(320.0, 240.0, 80.0, 200.0)))            # body
        t += 0.05
        ws.observe(det(t, "rafael", bbox=(320.0, 180.0, 60.0, 60.0)))   # face
        t += 0.05

    assert len(ws.tracks) == 1
    assert ws.snapshot(t)["present_count"] == 1


def test_two_people_keep_their_own_faces():
    """Containment must be spatial, not first-come — Zoe's face may not be
    swallowed by Rafael's body box."""
    ws = WorldState(match_radius=100.0)
    ws.observe(det(100.0, bbox=(200.0, 240.0, 80.0, 200.0)))            # body A
    ws.observe(det(100.0, bbox=(600.0, 240.0, 80.0, 200.0)))            # body B
    ws.observe(det(100.5, "rafael", bbox=(200.0, 180.0, 60.0, 60.0)))
    ws.observe(det(100.5, "zoe", bbox=(600.0, 180.0, 60.0, 60.0)))

    assert len(ws.tracks) == 2
    assert ws.snapshot(100.5)["known_present"] == ["rafael", "zoe"]


def test_flickering_static_detection_stays_one_track():
    """Regression, from the first live run: a stationary false face detection
    at ~[1051, 283] flickered in and out and minted FOUR separate people
    (tracks 4, 5, 21, 30) over 60s, because association only considered
    currently-visible tracks. Real timings from that capture."""
    ws = WorldState(visibility_timeout=3.0, rejoin_window=20.0)
    for t in (505371.6, 505388.0, 505404.0, 505416.1):
        ws.observe(det(t, bbox=(1051.0, 283.0, 34.0, 39.0)))
        ws.tick(t + 10.0)          # long enough to go away between sightings

    assert len(ws.tracks) == 1


def test_rejoin_window_is_bounded():
    """A detection in the same spot much later is a new person, not a revival —
    otherwise a doorway would accumulate everyone who ever walked through it
    into one immortal track."""
    ws = WorldState(visibility_timeout=3.0, rejoin_window=20.0)
    ws.observe(det(100.0, bbox=(500.0, 300.0, 60.0, 80.0)))
    ws.tick(150.0)
    ws.observe(det(200.0, bbox=(500.0, 300.0, 60.0, 80.0)))

    assert len(ws.tracks) == 2


def test_rejoin_does_not_resurrect_across_cameras():
    ws = WorldState(rejoin_window=20.0)
    ws.observe(det(100.0, camera=HEAD, bbox=(500.0, 300.0, 60.0, 80.0)))
    ws.tick(110.0)
    ws.observe(det(112.0, camera=REAR, bbox=(500.0, 300.0, 60.0, 80.0)))

    assert len(ws.tracks) == 2


def test_match_radius_zero_disables_centroid_association():
    """Boxes near each other but neither containing the other's centre: only
    the centroid rule can join them, so radius 0 must leave them apart.
    (Containment still applies independently — an identical box is the same
    person however small the radius.)"""
    near = [(320.0, 240.0, 20.0, 20.0), (350.0, 240.0, 20.0, 20.0)]

    joined = WorldState(match_radius=160.0)
    joined.observe(det(100.0, bbox=near[0]))
    joined.observe(det(100.1, bbox=near[1]))
    assert len(joined.tracks) == 1

    split = WorldState(match_radius=0.0)
    split.observe(det(100.0, bbox=near[0]))
    split.observe(det(100.1, bbox=near[1]))
    assert len(split.tracks) == 2


def test_different_cameras_do_not_merge_unnamed_people():
    """Documented v1 limitation: anonymous detections are never deduplicated
    across cameras. Two cameras seeing one stranger reads as two people."""
    ws = WorldState()
    ws.observe(det(100.0, camera=HEAD, source_track=1))
    ws.observe(det(100.0, camera=REAR, source_track=1))

    assert len(ws.tracks) == 2


# ── snapshot / misc ───────────────────────────────────────────────────────────


def test_stable_unknown_ids_track_but_do_not_count_as_identified():
    ws = WorldState()
    ws.observe(det(100.0, "unknown_3", camera=HEAD))
    ws.observe(det(101.0, "unknown_3", camera=REAR))

    assert len(ws.tracks) == 1                     # stable id => handoff works
    track = ws.find("unknown_3")
    assert track.is_identified is False
    assert ws.snapshot(101.0)["known_present"] == []
    assert ws.snapshot(101.0)["present"] == ["unknown_3"]


def test_empty_identity_is_treated_as_anonymous():
    assert Detection(camera=HEAD, timestamp=1.0, identity="").identity is None
    assert Detection(camera=HEAD, timestamp=1.0, identity="  ").identity is None


def test_detections_below_the_confidence_floor_are_dropped():
    ws = WorldState(min_confidence=0.5)
    assert ws.observe(det(100.0, "rafael", conf=0.2)) is None
    assert ws.tracks == []
    assert ws.observe(det(100.0, "rafael", conf=0.8)) is not None


def test_snapshot_is_json_serialisable_and_orders_present_first():
    import json

    ws = WorldState(visibility_timeout=3.0)
    ws.observe(det(100.0, "rafael", bbox=(10.0, 20.0, 30.0, 40.0)))
    ws.observe(det(100.0, "zoe"))
    ws.observe(det(110.0, "zoe"))
    ws.tick(110.0)                                 # rafael ages out

    snap = ws.snapshot(110.0)
    assert json.loads(json.dumps(snap)) == snap
    assert snap["people"][0]["identity"] == "zoe"  # visible first
    assert snap["people"][0]["visible"] is True
    assert snap["people"][1]["identity"] == "rafael"
    assert snap["people"][1]["bbox"] == [10.0, 20.0, 30.0, 40.0]


def test_anonymous_history_is_capped_but_named_people_are_kept():
    ws = WorldState(visibility_timeout=1.0, max_history=2)
    for i in range(6):
        ws.observe(det(100.0 + i, camera=HEAD, source_track=i))
    ws.observe(det(100.0, "rafael", camera=REAR))
    ws.tick(500.0)

    identities = [t.identity for t in ws.tracks]
    assert "rafael" in identities
    assert len([t for t in ws.tracks if t.identity is None]) == 2


def test_camera_is_required():
    with pytest.raises(ValueError):
        Detection(camera="", timestamp=1.0)
