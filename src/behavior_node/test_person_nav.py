"""Unit tests for the ROS-free go_to_person decision (person_nav).

Runs with the robot off — no ROS, no Gemini. Mirrors the convention of
test_greeting_decider.py / test_scene_describer.py.
"""

from behavior_node.person_nav import (
    GO,
    NO_NAME,
    NO_WORLD_STATE,
    STALE,
    UNKNOWN,
    UNPLACED,
    clean_name,
    humanise_age,
    plan_go_to_person,
)


def snap(people):
    return {"present_count": len(people), "people": people}


def person(identity, secs=1.0, zone="kitchen", map_xy=(2.0, 3.0), visible=True):
    return {
        "identity": identity,
        "seconds_since_seen": secs,
        "zone": zone,
        "map_xy": list(map_xy) if map_xy else None,
        "visible": visible,
    }


class TestCleanName:
    def test_strips_and_lowercases(self):
        assert clean_name("  Rafael! ") == "rafael"

    def test_keeps_underscore_hyphen(self):
        assert clean_name("unknown_3") == "unknown_3"

    def test_empty(self):
        assert clean_name("") == ""
        assert clean_name(None) == ""


class TestHumaniseAge:
    def test_just_now(self):
        assert humanise_age(3) == "just now"

    def test_seconds(self):
        assert "seconds" in humanise_age(30)

    def test_minutes_singular_plural(self):
        assert humanise_age(60) == "about 1 minute ago"
        assert humanise_age(180) == "about 3 minutes ago"

    def test_hours(self):
        assert humanise_age(7200) == "about 2 hours ago"

    def test_non_numeric(self):
        assert humanise_age(None) == "recently"


class TestPlan:
    def test_come_here_uses_session_person(self):
        # No name given -> resolves to who we're talking with.
        plan = plan_go_to_person(
            snap([person("rafael")]), name="", session_person="Rafael",
            stale_after=120.0)
        assert plan.outcome == GO
        assert plan.name == "rafael"

    def test_no_name_and_no_session(self):
        plan = plan_go_to_person(snap([]), name="", session_person=None,
                                 stale_after=120.0)
        assert plan.outcome == NO_NAME

    def test_no_world_state(self):
        plan = plan_go_to_person(None, name="rafael", session_person=None,
                                 stale_after=120.0)
        assert plan.outcome == NO_WORLD_STATE
        assert plan.name == "rafael"

    def test_unknown_person(self):
        plan = plan_go_to_person(
            snap([person("sofia")]), name="rafael", session_person=None,
            stale_after=120.0)
        assert plan.outcome == UNKNOWN

    def test_absent_person_is_unknown(self):
        # The Session-7 success criterion: "go to [absent person]" -> honest.
        plan = plan_go_to_person(
            snap([]), name="rafael", session_person=None, stale_after=120.0)
        assert plan.outcome == UNKNOWN

    def test_seen_but_unplaced(self):
        plan = plan_go_to_person(
            snap([person("rafael", zone=None, map_xy=None)]),
            name="rafael", session_person=None, stale_after=120.0)
        assert plan.outcome == UNPLACED

    def test_stale_last_seen(self):
        plan = plan_go_to_person(
            snap([person("rafael", secs=300.0)]),
            name="rafael", session_person=None, stale_after=120.0)
        assert plan.outcome == STALE
        assert plan.zone == "kitchen"

    def test_go_with_recent_fix(self):
        plan = plan_go_to_person(
            snap([person("rafael", secs=2.0)]),
            name="rafael", session_person=None, stale_after=120.0)
        assert plan.outcome == GO
        assert plan.map_xy == (2.0, 3.0)
        assert plan.zone == "kitchen"

    def test_zone_only_is_navigable(self):
        # No point estimate but a known zone -> still GO (uses the zone anchor).
        plan = plan_go_to_person(
            snap([person("rafael", secs=2.0, map_xy=None)]),
            name="rafael", session_person=None, stale_after=120.0)
        assert plan.outcome == GO
        assert plan.zone == "kitchen"
        assert plan.map_xy is None

    def test_map_xy_only_is_navigable(self):
        plan = plan_go_to_person(
            snap([person("rafael", secs=2.0, zone=None)]),
            name="rafael", session_person=None, stale_after=120.0)
        assert plan.outcome == GO
        assert plan.map_xy == (2.0, 3.0)

    def test_explicit_name_wins_over_session(self):
        plan = plan_go_to_person(
            snap([person("sofia", secs=2.0)]),
            name="Sofia", session_person="rafael", stale_after=120.0)
        assert plan.outcome == GO
        assert plan.name == "sofia"

    def test_display_capitalises(self):
        plan = plan_go_to_person(
            snap([person("rafael")]), name="rafael", session_person=None,
            stale_after=120.0)
        assert plan.display == "Rafael"
