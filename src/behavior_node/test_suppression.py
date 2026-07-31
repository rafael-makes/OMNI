"""Tests for the shared unprompted-interaction suppression rules.

Runs with the robot off. This module is small but load-bearing: both the Session
2 greeting and the Session 9 check-in gate on it, so a change here changes when
OMNI is willing to speak unprompted at all.
"""

from __future__ import annotations

import pytest

from behavior_node.suppression import (
    BUSY_STATES,
    CONVERSATION_STATES,
    RobotStatus,
    interaction_blocked,
)


def test_idle_and_charged_is_permitted():
    assert interaction_blocked(RobotStatus(state="IDLE", battery_pct=90.0)) is None


@pytest.mark.parametrize("state", BUSY_STATES)
def test_busy_states_are_blocked(state):
    reason = interaction_blocked(RobotStatus(state=state))
    assert reason == f"robot is {state}"


@pytest.mark.parametrize("state", CONVERSATION_STATES)
def test_conversation_states_are_blocked(state):
    reason = interaction_blocked(RobotStatus(state=state))
    assert reason == f"conversation in progress ({state})"


def test_the_two_state_lists_do_not_overlap():
    assert not set(BUSY_STATES) & set(CONVERSATION_STATES)


def test_an_open_session_is_blocked_even_in_idle():
    """Belt and braces: a session can be open in a state the checks above wave
    through — an /audio/say announcement still playing, for instance."""
    reason = interaction_blocked(RobotStatus(state="IDLE", session_active=True))
    assert "session" in reason


def test_docked_is_blocked():
    assert interaction_blocked(RobotStatus(state="IDLE", docked=True)) == "robot is docked"


def test_low_battery_flag_is_blocked():
    assert interaction_blocked(RobotStatus(state="IDLE", low_battery=True)) == "battery is low"


def test_battery_below_floor_is_blocked():
    reason = interaction_blocked(
        RobotStatus(state="IDLE", battery_pct=19.0), min_battery=20.0)
    assert reason == "battery 19% below 20%"


def test_battery_at_floor_is_permitted():
    assert interaction_blocked(
        RobotStatus(state="IDLE", battery_pct=20.0), min_battery=20.0) is None


def test_unknown_battery_is_permitted():
    """An absent reading is not evidence of a flat battery. Blocking on it would
    disable unprompted speech entirely whenever bms_node is slow to start."""
    assert interaction_blocked(
        RobotStatus(state="IDLE", battery_pct=None), min_battery=20.0) is None


def test_zero_floor_disables_the_percentage_check():
    assert interaction_blocked(
        RobotStatus(state="IDLE", battery_pct=1.0), min_battery=0.0) is None


def test_state_beats_battery_in_the_reported_reason():
    """Most-specific-first: "robot is NAVIGATING" is more actionable than a
    battery percentage when both are true."""
    reason = interaction_blocked(
        RobotStatus(state="NAVIGATING", battery_pct=5.0, low_battery=True),
        min_battery=20.0)
    assert reason == "robot is NAVIGATING"


def test_docked_beats_battery():
    reason = interaction_blocked(
        RobotStatus(state="IDLE", docked=True, low_battery=True))
    assert reason == "robot is docked"


def test_an_unknown_state_is_not_blocked_by_itself():
    """Only the listed states suppress. A state nobody enumerated here (a future
    one) must not silently start blocking speech — it should be added
    deliberately, with a test."""
    assert interaction_blocked(RobotStatus(state="SOMETHING_NEW")) is None
