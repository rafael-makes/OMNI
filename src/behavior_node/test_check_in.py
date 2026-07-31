"""State-machine tests for the proactive check-in (Session 9).

Runs with the robot off, against a fake node. `check_in.py` is ROS-coupled by
nature — it drives Nav2 and opens Gemini sessions — so it cannot be ROS-free like
the policy. But every hook it uses is a plain method call, which means the whole
mission can be flown on a stub and the branches that matter (silence, refusal,
person walks off, wake word, failed return) can be asserted without hardware.

`_start()` runs on a daemon thread in production, so the tests join on the
mission reaching a settled state rather than assuming it completed inline.
"""

from __future__ import annotations

import time
from datetime import datetime, time as dt_time, timedelta

import pytest

from behavior_node.check_in import APPROACH, ASK, INACTIVE, LISTEN, RETURN, CheckInBehavior
from behavior_node.check_in_policy import (
    OUTCOME_NO,
    OUTCOME_NOT_NOW,
    OUTCOME_NO_RESPONSE,
    OUTCOME_YES,
    CheckInConfig,
    CheckInPolicy,
)
from omni_zones import load_zone_map

SUCCEEDED, CANCELED, ABORTED = 4, 5, 6

PARAMS = {
    "check_in_standoff_distance": 1.0,
    "check_in_lateral_offset": 0.6,
    "check_in_default_side": "left",
    "check_in_silence_timeout": 15.0,
    "check_in_max_duration": 300.0,
}


class FakeParam:
    def __init__(self, value):
        self.value = value


class FakeLogger:
    def __init__(self):
        self.lines = []

    def _log(self, level, msg):
        self.lines.append(f"{level}: {msg}")

    def info(self, msg):
        self._log("INFO", msg)

    def warn(self, msg):
        self._log("WARN", msg)

    def error(self, msg):
        self._log("ERROR", msg)

    def debug(self, msg):
        self._log("DEBUG", msg)

    def text(self):
        return "\n".join(self.lines)


class FakeBridge:
    def __init__(self):
        self.speech = ""
        self.closed = False

    def user_speech(self):
        return self.speech

    def close_session(self):
        self.closed = True

    def is_session_active(self):
        return False


class FakeMemory:
    def __init__(self):
        self.stored = []

    def store_transcript(self, text, person=None, source=""):
        self.stored.append({"text": text, "person": person, "source": source})


class FakePub:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg.data)


class FakeFn:
    def __init__(self, pose):
        self.pose = pose

    def robot_pose(self):
        return self.pose


class FakeNode:
    """The slice of behavior_node that CheckInBehavior actually touches."""

    def __init__(self, *, pose=(0.0, 0.0, 0.0), person_xy=(5.0, 0.0),
                 zone="workbench", nav_ready=True, docked=False, zones=None):
        self._logger = FakeLogger()
        self._fn = FakeFn(pose)
        self._bridge = FakeBridge()
        self._memory = FakeMemory()
        self._events_pub = FakePub()
        self._memory_enabled = True
        self._docked = docked
        self._zones = load_zone_map(zones if zones is not None else {
            "workbench": {
                "polygon": [[4, -2], [8, -2], [8, 2], [4, 2]],
                "anchor": [6.0, 0.0, 0.0],
            },
        })
        self._nav_ready = nav_ready
        self._person_xy = person_xy
        self._person_zone = zone
        self.goals = []
        self.spoken = []
        self.state = "IDLE"
        self.nav_cancelled = 0
        self.goal_in_flight = False

    def cancel_navigation(self):
        if not self.goal_in_flight:
            return False
        self.goal_in_flight = False
        self.nav_cancelled += 1
        return True

    # ── the hooks CheckInBehavior calls ──
    def get_parameter(self, name):
        return FakeParam(PARAMS[name])

    def get_logger(self):
        return self._logger

    def robot_status(self):
        from behavior_node.suppression import RobotStatus
        return RobotStatus(state=self.state, battery_pct=90.0, docked=self._docked)

    def latest_world_state(self):
        row = {"identity": "rafael", "visible": True, "zone": self._person_zone}
        if self._person_xy is not None:
            row["map_xy"] = list(self._person_xy)
        return {"people": [row]}

    def nav_is_ready(self):
        return self._nav_ready

    def start_navigation(self, x, y, yaw):
        self.goals.append((x, y, yaw))
        self.state = "NAVIGATING"
        self.goal_in_flight = True

    def _speak_unprompted(self, prompt, person=None, memory_context=""):
        self.spoken.append({"prompt": prompt, "person": person})
        self.state = "LISTENING"


def make(node=None, policy=None, **cfg):
    node = node or FakeNode()
    cfg.setdefault("min_dwell", 3600.0)
    policy = policy or CheckInPolicy(CheckInConfig(**cfg), zones=("workbench",))
    return node, policy, CheckInBehavior(node, policy)


def dwell_event(zone="workbench", dwell=4000.0, identity="rafael"):
    return {
        "kind": "person_dwelling",
        "identity": identity,
        "zone": zone,
        "dwell_duration": dwell,
        "camera": "head",
        "timestamp": 1000.0,
    }


def wait_for(predicate, timeout=2.0):
    """_start() runs on a daemon thread; give it a moment to land."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def fly_to_approach(node, behavior):
    behavior.on_dwell_event(dwell_event())
    assert wait_for(lambda: node.goals), "never dispatched an approach goal"
    return node.goals[0]


# ── suppression: nothing happens, loudly ──────────────────────────────────────

def test_a_suppressed_dwell_does_not_move_the_robot():
    node, _, behavior = make()
    node.state = "NAVIGATING"          # busy
    behavior.on_dwell_event(dwell_event())
    assert node.goals == []
    assert not behavior.is_active()


def test_suppression_reason_is_logged():
    """The live verification step reads exactly this line."""
    node, _, behavior = make()
    node.state = "ERROR"
    behavior.on_dwell_event(dwell_event())
    assert "suppressed" in node._logger.text()
    assert "ERROR" in node._logger.text()


def test_short_dwell_is_suppressed():
    node, _, behavior = make()
    behavior.on_dwell_event(dwell_event(dwell=60.0))
    assert node.goals == []


def test_a_second_dwell_event_mid_mission_is_ignored():
    node, _, behavior = make()
    fly_to_approach(node, behavior)
    behavior.on_dwell_event(dwell_event())
    assert len(node.goals) == 1


# ── SNAPSHOT + APPROACH ───────────────────────────────────────────────────────

def test_approach_goal_is_offset_to_the_side():
    """Beside them, not in front. Robot at origin, person 5 m along +x, standoff
    1 m, lateral 0.6 m to the left → (4.0, 0.6), turned to look back at them."""
    node, _, behavior = make()
    x, y, yaw = fly_to_approach(node, behavior)
    assert (x, y) == pytest.approx((4.0, 0.6))
    assert yaw < 0.0            # looking back across at them, not straight ahead
    assert behavior.state == APPROACH


def test_zone_config_chooses_the_side():
    node = FakeNode(zones={
        "workbench": {"polygon": [[4, -2], [8, -2], [8, 2], [4, 2]],
                      "check_in_side": "right"},
    })
    node, _, behavior = make(node=node)
    _, y, _ = fly_to_approach(node, behavior)
    assert y == pytest.approx(-0.6)     # right of the approach, not left


def test_return_pose_is_snapshotted_before_moving():
    node, _, behavior = make(node=FakeNode(pose=(1.5, -2.0, 90.0)))
    fly_to_approach(node, behavior)
    behavior.on_nav_result(SUCCEEDED)        # arrive
    behavior.on_conversation_end()           # nothing said → leave
    assert wait_for(lambda: len(node.goals) >= 2)
    assert node.goals[-1] == pytest.approx((1.5, -2.0, 90.0))


def test_no_robot_pose_means_no_mission():
    """Without a pose there is no way home, so it must not set off at all."""
    node, _, behavior = make(node=FakeNode(pose=None))
    behavior.on_dwell_event(dwell_event())
    assert wait_for(lambda: not behavior.is_active())
    assert node.goals == []
    assert "no way back" in node._logger.text()


def test_docked_after_approval_stands_down():
    node = FakeNode()
    node, _, behavior = make(node=node)
    node._docked = True     # became docked between decision and start
    behavior.on_dwell_event(dwell_event())
    assert wait_for(lambda: not behavior.is_active())
    assert node.goals == []


def test_nav_unavailable_stands_down():
    node, _, behavior = make(node=FakeNode(nav_ready=False))
    behavior.on_dwell_event(dwell_event())
    assert wait_for(lambda: not behavior.is_active())
    assert node.goals == []


def test_unplaceable_person_falls_back_to_the_zone_anchor():
    node, _, behavior = make(node=FakeNode(person_xy=None))
    x, y, _ = fly_to_approach(node, behavior)
    # Anchor is (6, 0); standoff 1 m back toward the robot, 0.6 m to the left.
    assert (x, y) == pytest.approx((5.0, 0.6))


def test_failed_approach_returns_without_speaking():
    node, _, behavior = make()
    fly_to_approach(node, behavior)
    behavior.on_nav_result(ABORTED)
    assert node.spoken == []
    assert wait_for(lambda: len(node.goals) >= 2)   # went home


# ── ASK ───────────────────────────────────────────────────────────────────────

def test_arrival_asks_exactly_one_question():
    node, _, behavior = make()
    fly_to_approach(node, behavior)
    behavior.on_nav_result(SUCCEEDED)
    assert len(node.spoken) == 1
    assert behavior.state == LISTEN
    assert node.spoken[0]["person"] == "rafael"


def test_the_opener_is_instructed_to_be_restrained():
    """The whole charm of the feature is restraint, and the prompt is the only
    place that could turn it into an interrogation."""
    node, _, behavior = make()
    fly_to_approach(node, behavior)
    behavior.on_nav_result(SUCCEEDED)
    prompt = node.spoken[0]["prompt"].lower()
    assert "one short" in prompt
    assert "workbench" in prompt          # references the zone naturally
    assert "do not ask a second question" in prompt


# ── LISTEN / BRANCH ───────────────────────────────────────────────────────────

def test_silence_times_out_to_not_now_and_leaves():
    node, _, behavior = make()
    fly_to_approach(node, behavior)
    behavior.on_nav_result(SUCCEEDED)
    assert behavior.state == LISTEN

    behavior._asked_at = time.monotonic() - 16.0    # 15s silence elapsed
    behavior.tick()

    assert node._bridge.closed, "must not loiter with an open mic"
    assert wait_for(lambda: len(node.goals) >= 2)
    assert behavior._policy.outcome_records()[0].outcome == OUTCOME_NO_RESPONSE


def test_silence_does_not_re_ask():
    node, _, behavior = make()
    fly_to_approach(node, behavior)
    behavior.on_nav_result(SUCCEEDED)
    behavior._asked_at = time.monotonic() - 16.0
    behavior.tick()
    behavior.tick()
    assert len(node.spoken) == 1


def test_a_reply_stops_the_silence_timer():
    node, _, behavior = make()
    fly_to_approach(node, behavior)
    behavior.on_nav_result(SUCCEEDED)
    node._bridge.speech = "I'm fitting the servo bracket"
    behavior._asked_at = time.monotonic() - 60.0    # long past the timeout
    behavior.tick()
    assert behavior.state == LISTEN                 # still conversing
    assert len(node.goals) == 1                     # has not left


@pytest.mark.parametrize("reply,expected", [
    ("I'm rebuilding the gearbox", OUTCOME_YES),
    ("no thanks, I'm good", OUTCOME_NO),
    ("not right now", OUTCOME_NOT_NOW),
    ("no, not right now", OUTCOME_NOT_NOW),
])
def test_conversation_end_records_the_right_outcome(reply, expected):
    node, policy, behavior = make()
    fly_to_approach(node, behavior)
    behavior.on_nav_result(SUCCEEDED)
    node._bridge.speech = reply
    behavior.on_conversation_end()
    assert policy.outcome_records()[0].outcome == expected


def test_conversation_end_with_no_speech_is_no_response():
    node, policy, behavior = make()
    fly_to_approach(node, behavior)
    behavior.on_nav_result(SUCCEEDED)
    behavior.on_conversation_end()
    assert policy.outcome_records()[0].outcome == OUTCOME_NO_RESPONSE


def test_a_no_blocks_that_zone_for_the_cooldown():
    """The live check: say no, keep working, and OMNI must not come back.

    Quiet hours are disabled for this one (start == end) so the assertion can
    only be satisfied by the zone cooldown. Without that, the test passes or
    fails depending on what time of day it is run — the outcome is recorded
    against the real clock, and now+3h drifts into the quiet window each evening.
    """
    node, policy, behavior = make(
        quiet_start=dt_time(0, 0), quiet_end=dt_time(0, 0))
    fly_to_approach(node, behavior)
    behavior.on_nav_result(SUCCEEDED)
    node._bridge.speech = "no thanks"
    behavior.on_conversation_end()
    assert wait_for(lambda: not behavior.is_active() or behavior.state == RETURN)
    behavior.on_nav_result(SUCCEEDED)               # got home

    node.state = "IDLE"
    later = datetime.now() + timedelta(hours=3)     # past the 2h global cooldown
    decision = policy.decide(dwell_event(), node.robot_status(), later)
    assert not decision.approved
    assert "zone" in decision.reason


def test_outcome_is_logged_to_memory():
    node, _, behavior = make()
    fly_to_approach(node, behavior)
    behavior.on_nav_result(SUCCEEDED)
    node._bridge.speech = "no thanks"
    behavior.on_conversation_end()
    assert node._memory.stored
    assert node._memory.stored[0]["source"] == "check_in"
    assert node._memory.stored[0]["person"] == "rafael"


def test_outcome_is_recorded_only_once():
    node, policy, behavior = make()
    fly_to_approach(node, behavior)
    behavior.on_nav_result(SUCCEEDED)
    node._bridge.speech = "no"
    behavior.on_conversation_end()
    behavior.on_conversation_end()
    assert len(policy.outcome_records()) == 1


# ── interrupts ────────────────────────────────────────────────────────────────

def test_person_leaving_mid_approach_aborts_to_return():
    node, _, behavior = make()
    fly_to_approach(node, behavior)
    node._person_zone = "kitchen"        # they got up and walked off
    behavior.tick()
    assert behavior.state == RETURN
    assert node.spoken == []             # never asked an empty bench anything
    assert len(node.goals) == 2


def test_a_missing_world_state_does_not_abort_the_approach():
    """world_state is a soft dependency; losing it mid-drive must not be read as
    the person having left."""
    node, _, behavior = make()
    fly_to_approach(node, behavior)
    node.latest_world_state = lambda: None
    behavior.tick()
    assert behavior.state == APPROACH


def test_wake_word_aborts_cleanly_without_driving_home():
    """They started talking to OMNI — driving off mid-sentence to restore a pose
    would be worse than staying put."""
    node, policy, behavior = make()
    fly_to_approach(node, behavior)
    behavior.on_nav_result(SUCCEEDED)
    behavior.abort("wake word")
    assert not behavior.is_active()
    assert len(node.goals) == 1                       # no return trip
    assert policy.outcome_records()[0].outcome == OUTCOME_YES   # they engaged


def test_abort_during_approach_records_no_outcome():
    """Nothing was asked, so nothing was answered — do not burn a zone cooldown
    on a mission that never got there."""
    node, policy, behavior = make()
    fly_to_approach(node, behavior)
    behavior.abort("safety fault")
    assert not behavior.is_active()
    assert policy.outcome_records() == ()


def test_abort_mid_approach_cancels_the_nav_goal():
    """Without this, a wake word mid-drive leaves OMNI still motoring toward
    someone who is already talking to it — and the goal's eventual SUCCEEDED
    result reaches the generic arrival handler, which announces the arrival over
    the top of the conversation."""
    node, _, behavior = make()
    fly_to_approach(node, behavior)
    assert node.goal_in_flight
    behavior.abort("wake word")
    assert node.nav_cancelled == 1
    assert not node.goal_in_flight


def test_abort_after_arrival_cancels_nothing():
    """Already stopped — there is no goal to cancel once it has arrived."""
    node, _, behavior = make()
    fly_to_approach(node, behavior)
    behavior.on_nav_result(SUCCEEDED)
    node.goal_in_flight = False
    behavior.abort("wake word")
    assert node.nav_cancelled == 0


def test_abort_is_idempotent():
    node, _, behavior = make()
    fly_to_approach(node, behavior)
    behavior.abort("first")
    behavior.abort("second")
    assert not behavior.is_active()


def test_the_hard_ceiling_ends_a_wedged_mission():
    node, _, behavior = make()
    fly_to_approach(node, behavior)
    behavior._started = time.monotonic() - 400.0     # past check_in_max_duration
    behavior.tick()
    assert behavior.state == RETURN


# ── RETURN ────────────────────────────────────────────────────────────────────

def test_successful_return_finishes_the_mission():
    node, _, behavior = make()
    fly_to_approach(node, behavior)
    behavior.on_nav_result(SUCCEEDED)
    behavior.on_conversation_end()
    assert wait_for(lambda: behavior.state == RETURN)
    behavior.on_nav_result(SUCCEEDED)
    assert not behavior.is_active()


def test_failed_return_is_reported_loudly_not_silently():
    """There is no docking routine to fall back to yet, so the requirement
    "never end stranded" is currently met by saying so, not by pretending."""
    node, _, behavior = make()
    fly_to_approach(node, behavior)
    behavior.on_nav_result(SUCCEEDED)
    behavior.on_conversation_end()
    assert wait_for(lambda: behavior.state == RETURN)
    behavior.on_nav_result(ABORTED)
    assert not behavior.is_active()
    log = node._logger.text()
    assert "could not return" in log
    assert "_dock_fallback" in log      # points at the exact spot to wire it


def test_nav_results_are_ignored_when_no_mission_is_running():
    node, _, behavior = make()
    assert behavior.on_nav_result(SUCCEEDED) is False


# ── /omni/events ──────────────────────────────────────────────────────────────

def test_every_phase_is_published():
    import json
    node, _, behavior = make()
    fly_to_approach(node, behavior)
    behavior.on_nav_result(SUCCEEDED)
    node._bridge.speech = "no thanks"
    behavior.on_conversation_end()
    assert wait_for(lambda: behavior.state == RETURN)
    behavior.on_nav_result(SUCCEEDED)

    phases = [json.loads(m)["phase"] for m in node._events_pub.messages]
    assert phases[0] == "approach"
    for expected in ("ask", "outcome", "return", "done"):
        assert expected in phases


def test_published_events_are_valid_json_with_the_shared_shape():
    import json
    node, _, behavior = make()
    fly_to_approach(node, behavior)
    for message in node._events_pub.messages:
        payload = json.loads(message)
        assert payload["kind"] == "check_in"
        assert payload["identity"] == "rafael"
        assert payload["zone"] == "workbench"
        assert isinstance(payload["timestamp"], float)
