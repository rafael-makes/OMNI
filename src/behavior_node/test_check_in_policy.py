"""Exhaustive tests for the check-in manners layer (Session 9).

Runs with the robot off — no ROS, no Gemini, no clock. Mirrors the convention of
test_greeting_decider.py / test_person_nav.py.

This is pure decision logic and it is the part of the feature most likely to be
"tidied" by a future session, so the tests pin *behaviour differences* (a "no"
costing more than a "not now") rather than just return values.
"""

from __future__ import annotations

import json
from datetime import datetime, time, timedelta

import pytest

from behavior_node.check_in_policy import (
    ALL_OUTCOMES,
    OUTCOME_NO,
    OUTCOME_NOT_NOW,
    OUTCOME_NO_RESPONSE,
    OUTCOME_YES,
    REASON_DISABLED,
    REASON_DWELL_TOO_SHORT,
    REASON_GLOBAL_COOLDOWN,
    REASON_NOT_DWELLING,
    REASON_NO_ZONE,
    REASON_OK,
    REASON_QUIET_HOURS,
    REASON_UNNAMED,
    REASON_ZONE_COOLDOWN,
    REASON_ZONE_NOT_ENABLED,
    CheckInConfig,
    CheckInPolicy,
    OutcomeRecord,
    in_quiet_hours,
)
from behavior_node.suppression import RobotStatus

# 10:00 on a Tuesday — comfortably outside the default 21:00–08:00 quiet window.
NOON = datetime(2026, 7, 21, 10, 0)
HOUR = timedelta(hours=1)


def event(identity="rafael", zone="workbench", dwell=3600.0,
          kind="person_dwelling"):
    return {
        "kind": kind,
        "identity": identity,
        "camera": "head",
        "timestamp": 1000.0,
        "away_duration": None,
        "detail": "settled in workbench for 60 min",
        "zone": zone,
        "dwell_duration": dwell,
    }


def idle(**kwargs):
    kwargs.setdefault("battery_pct", 90.0)
    return RobotStatus(state="IDLE", **kwargs)


def policy(**cfg_kwargs):
    zones = cfg_kwargs.pop("zones", ("workbench", "desk"))
    return CheckInPolicy(CheckInConfig(**cfg_kwargs), zones=zones)


# ── dwell threshold ───────────────────────────────────────────────────────────

def test_approves_a_long_dwell():
    d = policy().decide(event(dwell=3600.0), idle(), NOON)
    assert d.approved
    assert d.reason == REASON_OK
    assert d.identity == "rafael"
    assert d.zone == "workbench"
    assert bool(d) is True


def test_rejects_a_short_dwell():
    d = policy().decide(event(dwell=1800.0), idle(), NOON)
    assert not d.approved
    assert d.reason == REASON_DWELL_TOO_SHORT
    assert bool(d) is False


def test_threshold_is_inclusive():
    """Exactly at the threshold counts — the alternative is an off-by-one that
    only ever shows up as a check-in that mysteriously never happens."""
    assert policy(min_dwell=3600.0).decide(event(dwell=3600.0), idle(), NOON).approved


def test_default_threshold_is_at_least_an_hour():
    """Pinned because lowering this is the single easiest way to make the whole
    feature obnoxious."""
    assert CheckInConfig().min_dwell >= 3600.0


def test_missing_dwell_duration_reads_as_zero():
    ev = event()
    del ev["dwell_duration"]
    assert policy().decide(ev, idle(), NOON).reason == REASON_DWELL_TOO_SHORT


# ── event validity ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kind", ["person_appeared", "person_left",
                                  "unknown_person_detected", "", None])
def test_only_dwell_events_are_considered(kind):
    assert policy().decide(event(kind=kind), idle(), NOON).reason == REASON_NOT_DWELLING


@pytest.mark.parametrize("bad", [None, [], "person_dwelling", 42])
def test_malformed_events_are_rejected(bad):
    assert policy().decide(bad, idle(), NOON).reason == REASON_NOT_DWELLING


@pytest.mark.parametrize("identity", ["unknown_5", "unknown_46", "", "   "])
def test_strangers_are_never_checked_in_on(identity):
    assert policy().decide(event(identity=identity), idle(), NOON).reason == REASON_UNNAMED


@pytest.mark.parametrize("zone", ["", None, "   "])
def test_an_unplaced_dwell_is_rejected(zone):
    assert policy().decide(event(zone=zone), idle(), NOON).reason == REASON_NO_ZONE


def test_zone_must_be_enabled():
    d = policy(zones=("workbench",)).decide(event(zone="kitchen"), idle(), NOON)
    assert d.reason == REASON_ZONE_NOT_ENABLED


def test_empty_zone_list_trusts_the_generator():
    """No second belt configured → any zone the generator emitted is accepted."""
    assert CheckInPolicy(CheckInConfig()).decide(
        event(zone="anywhere"), idle(), NOON).approved


def test_identity_is_normalised():
    d = policy().decide(event(identity="  RAFAEL  "), idle(), NOON)
    assert d.approved and d.identity == "rafael"


def test_disabled_policy_says_so_first():
    assert policy(enabled=False).decide(event(), idle(), NOON).reason == REASON_DISABLED


# ── robot-state suppression (the shared helper) ───────────────────────────────

@pytest.mark.parametrize("state", ["NAVIGATING", "EXPLORING", "DOCKING", "ERROR"])
def test_busy_states_suppress(state):
    d = policy().decide(event(), RobotStatus(state=state, battery_pct=90.0), NOON)
    assert not d.approved
    assert state in d.reason


@pytest.mark.parametrize("state", ["LISTENING", "SPEAKING"])
def test_never_interrupts_a_conversation(state):
    d = policy().decide(event(), RobotStatus(state=state, battery_pct=90.0), NOON)
    assert not d.approved
    assert "conversation in progress" in d.reason


def test_open_session_suppresses():
    d = policy().decide(event(), idle(session_active=True), NOON)
    assert not d.approved and "session" in d.reason


def test_docked_suppresses():
    d = policy().decide(event(), idle(docked=True), NOON)
    assert not d.approved and "docked" in d.reason


def test_low_battery_flag_suppresses():
    d = policy().decide(event(), idle(low_battery=True), NOON)
    assert not d.approved and "battery is low" in d.reason


def test_battery_below_floor_suppresses():
    d = policy(battery_floor=40.0).decide(event(), idle(battery_pct=35.0), NOON)
    assert not d.approved and "below" in d.reason


def test_battery_above_floor_is_fine():
    assert policy(battery_floor=40.0).decide(
        event(), idle(battery_pct=41.0), NOON).approved


def test_unknown_battery_does_not_block():
    """Same posture as greetings: an absent reading is not evidence of a flat
    battery, and blocking on it would disable the feature whenever bms_node is
    slow to start."""
    assert policy().decide(event(), idle(battery_pct=None), NOON).approved


def test_check_in_battery_floor_is_stricter_than_a_greeting():
    """A greeting costs a sentence; a check-in costs a trip across the room and
    back. 20% is the greeting floor — this must be well above it."""
    assert CheckInConfig().battery_floor > 20.0


# ── quiet hours ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("hour", [21, 22, 23, 0, 3, 7])
def test_quiet_hours_suppress(hour):
    when = NOON.replace(hour=hour)
    assert policy().decide(event(), idle(), when).reason == REASON_QUIET_HOURS


@pytest.mark.parametrize("hour", [8, 12, 17, 20])
def test_outside_quiet_hours_is_fine(hour):
    assert policy().decide(event(), idle(), NOON.replace(hour=hour)).approved


def test_quiet_window_wraps_midnight():
    assert in_quiet_hours(NOON.replace(hour=23), time(21, 0), time(8, 0))
    assert in_quiet_hours(NOON.replace(hour=2), time(21, 0), time(8, 0))
    assert not in_quiet_hours(NOON.replace(hour=12), time(21, 0), time(8, 0))


def test_quiet_window_within_one_day():
    assert in_quiet_hours(NOON.replace(hour=14), time(13, 0), time(15, 0))
    assert not in_quiet_hours(NOON.replace(hour=16), time(13, 0), time(15, 0))


def test_equal_bounds_means_no_quiet_hours():
    assert not in_quiet_hours(NOON, time(9, 0), time(9, 0))


def test_quiet_boundaries_are_half_open():
    """Start is quiet, end is not — so a 21:00–08:00 window is silent at 21:00
    sharp and awake at 08:00 sharp."""
    assert in_quiet_hours(NOON.replace(hour=21, minute=0), time(21, 0), time(8, 0))
    assert not in_quiet_hours(NOON.replace(hour=8, minute=0), time(21, 0), time(8, 0))


# ── cooldowns ─────────────────────────────────────────────────────────────────

def test_recent_interaction_blocks_a_check_in():
    p = policy()
    p.record_interaction("rafael", NOON)
    assert p.decide(event(), idle(), NOON + HOUR).reason == REASON_GLOBAL_COOLDOWN


def test_global_cooldown_expires():
    p = policy(global_cooldown=7200.0)
    p.record_interaction("rafael", NOON)
    assert p.decide(event(), idle(), NOON + 3 * HOUR).approved


def test_global_cooldown_spans_every_zone():
    """A greeting in the kitchen still suppresses a workbench check-in — it is
    the person's attention that was spent, not the room's."""
    p = policy()
    p.record_interaction("rafael", NOON)
    assert p.decide(event(zone="desk"), idle(), NOON + HOUR).reason == REASON_GLOBAL_COOLDOWN


def test_global_cooldown_is_per_person():
    p = policy()
    p.record_interaction("someone_else", NOON)
    assert p.decide(event(identity="rafael"), idle(), NOON + HOUR).approved


def test_default_global_cooldown_is_at_least_two_hours():
    assert CheckInConfig().global_cooldown >= 7200.0


def test_no_and_not_now_have_different_consequences():
    """THE headline manners assertion. Three hours later — past the 2 h global
    cooldown either way — the zone that was told "not now" is available again and
    the zone that was told "no" is still closed. If these ever collapse into one
    cooldown, OMNI has stopped listening to the difference."""
    said_no = policy()
    said_no.record_outcome("rafael", "workbench", OUTCOME_NO, NOON)

    said_not_now = policy()
    said_not_now.record_outcome("rafael", "workbench", OUTCOME_NOT_NOW, NOON)

    later = NOON + 3 * HOUR
    assert said_no.decide(event(), idle(), later).reason == REASON_ZONE_COOLDOWN
    assert said_not_now.decide(event(), idle(), later).approved


def test_no_cooldown_eventually_expires():
    p = policy(no_cooldown=14400.0)
    p.record_outcome("rafael", "workbench", OUTCOME_NO, NOON)
    assert p.decide(event(), idle(), NOON + 5 * HOUR).approved


def test_no_blocks_only_the_zone_it_was_said_in():
    """"No" is about this bench, not about the whole house. Four and a half hours
    on, the workbench is still closed but the desk is open — and the global
    cooldown has expired for both."""
    p = policy()
    p.record_outcome("rafael", "workbench", OUTCOME_NO, NOON)
    later = NOON + 3 * HOUR
    assert p.decide(event(zone="workbench"), idle(), later).reason == REASON_ZONE_COOLDOWN
    assert p.decide(event(zone="desk"), idle(), later).approved


def test_yes_only_costs_the_global_cooldown():
    p = policy()
    p.record_outcome("rafael", "workbench", OUTCOME_YES, NOON)
    # Inside 2 h: blocked globally. Past it: the zone was never blocked at all.
    assert p.decide(event(), idle(), NOON + HOUR).reason == REASON_GLOBAL_COOLDOWN
    assert p.decide(event(), idle(), NOON + 3 * HOUR).approved


def test_silence_is_treated_as_not_now():
    p_silent = policy()
    p_silent.record_outcome("rafael", "workbench", OUTCOME_NO_RESPONSE, NOON)
    p_not_now = policy()
    p_not_now.record_outcome("rafael", "workbench", OUTCOME_NOT_NOW, NOON)
    assert (p_silent.zone_blocked_until("rafael", "workbench")
            == p_not_now.zone_blocked_until("rafael", "workbench"))


def test_silence_is_still_logged_distinctly():
    """Same cooldown as "not now", different record — the two must stay
    distinguishable for the future learning pass."""
    p = policy()
    p.record_outcome("rafael", "workbench", OUTCOME_NO_RESPONSE, NOON)
    assert p.outcome_records()[0].outcome == OUTCOME_NO_RESPONSE


def test_every_outcome_including_refusal_starts_the_global_cooldown():
    """Being told "no" is still OMNI having spent your attention."""
    for outcome in ALL_OUTCOMES:
        p = policy()
        p.record_outcome("rafael", "workbench", outcome, NOON)
        assert p.last_interaction("rafael") == NOON, outcome


def test_default_no_cooldown_is_at_least_four_hours():
    assert CheckInConfig().no_cooldown >= 14400.0


def test_not_now_is_shorter_than_no():
    cfg = CheckInConfig()
    assert cfg.not_now_cooldown < cfg.no_cooldown


def test_a_longer_cooldown_is_never_shortened_by_a_later_softer_one():
    """A "no" then a "not now" ten minutes later must not unlock the zone early."""
    p = policy()
    p.record_outcome("rafael", "workbench", OUTCOME_NO, NOON)
    p.record_outcome("rafael", "workbench", OUTCOME_NOT_NOW, NOON + timedelta(minutes=10))
    assert p.decide(event(), idle(), NOON + 3 * HOUR).reason == REASON_ZONE_COOLDOWN


def test_out_of_order_interaction_does_not_rewind_the_cooldown():
    p = policy()
    p.record_interaction("rafael", NOON)
    p.record_interaction("rafael", NOON - 5 * HOUR)   # a late-arriving stale call
    assert p.last_interaction("rafael") == NOON


def test_record_interaction_ignores_empty_identity():
    p = policy()
    p.record_interaction("", NOON)
    assert p.last_interaction("") is None


def test_unknown_outcome_is_rejected():
    with pytest.raises(ValueError):
        policy().record_outcome("rafael", "workbench", "maybe", NOON)


# ── reason precedence ─────────────────────────────────────────────────────────

def test_dwell_is_checked_before_robot_state():
    """The cheap, most common reason wins the log line."""
    d = policy().decide(event(dwell=60.0),
                        RobotStatus(state="NAVIGATING", battery_pct=90.0), NOON)
    assert d.reason == REASON_DWELL_TOO_SHORT


def test_state_is_checked_before_quiet_hours():
    d = policy().decide(event(),
                        RobotStatus(state="ERROR", battery_pct=90.0),
                        NOON.replace(hour=23))
    assert "ERROR" in d.reason


def test_quiet_hours_checked_before_cooldowns():
    p = policy()
    p.record_interaction("rafael", NOON)
    assert p.decide(event(), idle(), NOON.replace(hour=23)).reason == REASON_QUIET_HOURS


def test_decision_carries_context_for_logging():
    d = policy().decide(event(dwell=4000.0), idle(), NOON)
    assert d.dwell_duration == 4000.0
    assert d.threshold == 3600.0
    assert d.zone == "workbench"


# ── the v1.5 learning bias ────────────────────────────────────────────────────

def test_bias_needs_evidence_before_it_moves():
    p = policy(bias_min_samples=3)
    p.record_outcome("rafael", "workbench", OUTCOME_NO, NOON)
    p.record_outcome("rafael", "workbench", OUTCOME_NO, NOON + HOUR)
    assert p.effective_threshold("rafael", "workbench") == 3600.0


def test_repeated_no_stretches_the_threshold():
    p = policy(bias_min_samples=3, bias_max_multiplier=2.0)
    for i in range(3):
        p.record_outcome("rafael", "workbench", OUTCOME_NO, NOON + i * HOUR)
    assert p.effective_threshold("rafael", "workbench") == pytest.approx(7200.0)


def test_repeated_yes_keeps_the_floor():
    p = policy(bias_min_samples=3)
    for i in range(3):
        p.record_outcome("rafael", "workbench", OUTCOME_YES, NOON + i * HOUR)
    assert p.effective_threshold("rafael", "workbench") == 3600.0


def test_enthusiasm_never_lowers_the_floor():
    """One-directional on purpose: a run of "yes" is not a licence to interrupt
    more often than the configured minimum."""
    p = policy(bias_min_samples=1)
    for i in range(10):
        p.record_outcome("rafael", "workbench", OUTCOME_YES, NOON + i * HOUR)
    assert p.effective_threshold("rafael", "workbench") >= 3600.0


def test_soft_declines_stretch_less_than_hard_ones():
    hard = policy(bias_min_samples=3)
    soft = policy(bias_min_samples=3)
    for i in range(3):
        hard.record_outcome("rafael", "workbench", OUTCOME_NO, NOON + i * HOUR)
        soft.record_outcome("rafael", "workbench", OUTCOME_NOT_NOW, NOON + i * HOUR)
    assert (soft.effective_threshold("rafael", "workbench")
            < hard.effective_threshold("rafael", "workbench"))


def test_bias_is_clamped():
    p = policy(bias_min_samples=1, bias_max_multiplier=1.5)
    for i in range(20):
        p.record_outcome("rafael", "workbench", OUTCOME_NO, NOON + i * HOUR)
    assert p.effective_threshold("rafael", "workbench") == pytest.approx(5400.0)


def test_bias_is_per_zone():
    """Being rebuffed at the bench says nothing about the desk."""
    p = policy(bias_min_samples=3)
    for i in range(3):
        p.record_outcome("rafael", "workbench", OUTCOME_NO, NOON + i * HOUR)
    assert p.effective_threshold("rafael", "desk") == 3600.0


def test_bias_is_per_person():
    p = policy(bias_min_samples=3)
    for i in range(3):
        p.record_outcome("someone_else", "workbench", OUTCOME_NO, NOON + i * HOUR)
    assert p.effective_threshold("rafael", "workbench") == 3600.0


def test_bias_can_be_disabled():
    p = policy(bias_enabled=False, bias_min_samples=1)
    for i in range(5):
        p.record_outcome("rafael", "workbench", OUTCOME_NO, NOON + i * HOUR)
    assert p.effective_threshold("rafael", "workbench") == 3600.0


def test_the_bias_actually_gates_a_decision():
    """End to end: after three refusals a 70-minute dwell is no longer enough at
    that bench, though it would have been on day one."""
    p = policy(bias_min_samples=3, bias_max_multiplier=2.0)
    base = NOON
    for i in range(3):
        p.record_outcome("rafael", "workbench", OUTCOME_NO, base + i * HOUR)
    much_later = base + timedelta(days=2)   # every cooldown long expired
    assert p.decide(event(dwell=4200.0), idle(), much_later).reason == REASON_DWELL_TOO_SHORT
    assert p.decide(event(dwell=7300.0), idle(), much_later).approved


# ── outcome records ───────────────────────────────────────────────────────────

def test_outcome_record_is_returned_and_kept():
    p = policy()
    rec = p.record_outcome("rafael", "workbench", OUTCOME_NO, NOON, dwell_duration=3900.0)
    assert isinstance(rec, OutcomeRecord)
    assert p.outcome_records() == (rec,)


def test_outcome_record_serialises():
    rec = policy().record_outcome("rafael", "workbench", OUTCOME_YES, NOON,
                                  dwell_duration=3900.0)
    d = json.loads(json.dumps(rec.as_dict()))
    assert d["identity"] == "rafael"
    assert d["zone"] == "workbench"
    assert d["outcome"] == OUTCOME_YES
    assert d["dwell_duration"] == pytest.approx(3900.0)
    assert d["when"].startswith("2026-07-21T10:00")


@pytest.mark.parametrize("outcome", ALL_OUTCOMES)
def test_memory_text_is_a_readable_sentence(outcome):
    rec = policy().record_outcome("rafael", "workbench", outcome, NOON,
                                  dwell_duration=3900.0)
    text = rec.as_memory_text()
    assert text.startswith("Rafael ")
    assert "workbench" in text
    assert text.endswith(".")


def test_history_is_ordered_and_immutable_to_callers():
    p = policy()
    p.record_outcome("rafael", "workbench", OUTCOME_NO, NOON)
    p.record_outcome("rafael", "desk", OUTCOME_YES, NOON + HOUR)
    records = p.outcome_records()
    assert [r.zone for r in records] == ["workbench", "desk"]
    assert isinstance(records, tuple)   # a copy — callers cannot mutate history


# ── config validation ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("kwargs", [
    {"min_dwell": 0.0},
    {"min_dwell": -1.0},
    {"bias_max_multiplier": 0.5},
    {"bias_min_samples": 0},
])
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        CheckInConfig(**kwargs)


def test_policy_defaults_construct():
    assert CheckInPolicy().decide(event(), idle(), NOON).approved
