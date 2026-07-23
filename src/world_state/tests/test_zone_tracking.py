"""Person tracks carry a coarse zone through the tracker. Robot off.

These exercise the *core* library only — the ROS wrapper computes map_xy/zone
and hands them in via Detection, exactly as these tests do by hand. The
geometry that produces those coordinates is tested in omni_zones.
"""

from world_state.models import Detection
from world_state.tracker import WorldState


def det(identity=None, t=0.0, camera="head", map_xy=None, zone=None, track=None):
    return Detection(
        camera=camera,
        timestamp=t,
        identity=identity,
        confidence=1.0,
        bbox=(100.0, 100.0, 40.0, 40.0),
        source_track=track,
        map_xy=map_xy,
        zone=zone,
    )


def test_track_carries_zone_from_detection():
    ws = WorldState()
    track = ws.observe(det(identity="rafael", map_xy=(2.0, 3.0), zone="kitchen"))
    assert track.zone == "kitchen"
    assert track.map_xy == (2.0, 3.0)
    snap = ws.snapshot(now=0.0)
    person = snap["people"][0]
    assert person["zone"] == "kitchen"
    assert person["map_xy"] == [2.0, 3.0]


def test_zone_updates_as_person_moves():
    ws = WorldState()
    ws.observe(det(identity="rafael", t=0.0, map_xy=(1.0, 1.0), zone="kitchen"))
    ws.observe(det(identity="rafael", t=1.0, map_xy=(6.0, 1.0), zone="hall"))
    track = ws.find("rafael")
    assert track.zone == "hall"
    assert track.map_xy == (6.0, 1.0)


def test_missing_location_does_not_wipe_last_known():
    # A frame with no pose (map_xy None) must not erase a good last-known zone.
    ws = WorldState()
    ws.observe(det(identity="rafael", t=0.0, map_xy=(1.0, 1.0), zone="kitchen"))
    ws.observe(det(identity="rafael", t=1.0, map_xy=None, zone=None))
    track = ws.find("rafael")
    assert track.zone == "kitchen"
    assert track.map_xy == (1.0, 1.0)


def test_zone_survives_going_away():
    # "Where did I last see Rafael" must remain answerable after he leaves.
    ws = WorldState(visibility_timeout=1.0)
    ws.observe(det(identity="rafael", t=0.0, map_xy=(2.0, 2.0), zone="kitchen"))
    ws.tick(now=5.0)  # ages him out of visibility
    track = ws.find("rafael")
    assert track is not None
    assert not track.visible
    assert track.zone == "kitchen"


def test_zone_none_when_never_localised():
    ws = WorldState()
    ws.observe(det(identity="rafael"))
    track = ws.find("rafael")
    assert track.zone is None
    assert track.map_xy is None
    assert ws.snapshot(now=0.0)["people"][0]["zone"] is None
