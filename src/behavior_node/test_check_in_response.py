"""Tests for the check-in reply classifier. Robot off.

The stakes are a cooldown, not a catastrophe — but the "no" / "not now"
distinction is the one the session brief asks to verify from the logs, so it is
pinned hard here.
"""

from __future__ import annotations

import pytest

from behavior_node.check_in_policy import OUTCOME_NO, OUTCOME_NOT_NOW, OUTCOME_YES
from behavior_node.check_in_response import classify_reply


# ── engaged ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("reply", [
    "I'm fitting the servo bracket",
    "just soldering the new power board",
    "trying to work out why the encoder drifts",
    "yeah, come and have a look at this",
    "actually yes, can you hold this",
    "I'm rewiring the chest panel, it's a nightmare",
])
def test_engagement(reply):
    assert classify_reply(reply) == OUTCOME_YES


def test_unrecognised_replies_default_to_engagement():
    """The safe default. One mumbled sentence must not lock the zone out for
    four hours — a wrong YES costs a conversation the person can simply end."""
    assert classify_reply("mmm the flange bracket thingy") == OUTCOME_YES


def test_empty_is_not_classified_here():
    """The caller owns silence — only it knows whether the pause was long
    enough to count. Classifying emptiness here would double-count it."""
    assert classify_reply("") == OUTCOME_YES
    assert classify_reply("   ") == OUTCOME_YES


# ── refusal ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("reply", [
    "no",
    "No.",
    "no thanks",
    "no thank you",
    "nope",
    "nah",
    "I'm good",
    "I'm fine thanks",
    "I am okay",
    "no need",
    "not interested",
    "leave me to it",
    "leave me alone",
    "go away",
])
def test_refusal(reply):
    assert classify_reply(reply) == OUTCOME_NO


# ── deferral ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("reply", [
    "not right now",
    "not now",
    "not at the moment",
    "maybe later",
    "ask me later",
    "give me a minute",
    "in a bit",
    "hold on",
    "I'm in the middle of something",
    "busy right now",
])
def test_deferral(reply):
    assert classify_reply(reply) == OUTCOME_NOT_NOW


# ── the distinction that matters ──────────────────────────────────────────────

@pytest.mark.parametrize("reply", [
    "no, not right now",
    "no, maybe later",
    "nope, ask me later",
    "no — give me a minute",
])
def test_a_qualified_no_is_a_deferral_not_a_refusal(reply):
    """THE assertion. These all contain a bare "no", and reading them as a flat
    refusal would turn every "in a minute" into a four-hour lockout. The softer,
    more specific reading wins — which is why NOT_NOW is matched first."""
    assert classify_reply(reply) == OUTCOME_NOT_NOW


def test_plain_no_stays_a_refusal():
    """The other half of the same rule: without a time qualifier it really is
    a refusal, and must earn the longer cooldown."""
    assert classify_reply("no, I'm good thanks") == OUTCOME_NO


# ── robustness ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("reply", ["NO THANKS", "No Thanks", "nO tHaNkS"])
def test_case_insensitive(reply):
    assert classify_reply(reply) == OUTCOME_NO


def test_leading_whitespace_and_punctuation():
    assert classify_reply("   no.  ") == OUTCOME_NO


def test_no_inside_another_word_is_not_a_refusal():
    """Word boundaries matter: "nothing" is a refusal but "notice" is not, and
    "know" must never read as "no"."""
    assert classify_reply("I know what I'm doing here") == OUTCOME_YES
    assert classify_reply("I noticed the bracket is bent") == OUTCOME_YES


@pytest.mark.parametrize("reply", [None, 0, [], {}])
def test_non_string_input_does_not_explode(reply):
    assert classify_reply(reply) in (OUTCOME_YES, OUTCOME_NO, OUTCOME_NOT_NOW)


def test_only_the_three_outcomes_are_ever_returned():
    for reply in ["no", "not now", "building a robot", "", "?!"]:
        assert classify_reply(reply) in (OUTCOME_YES, OUTCOME_NO, OUTCOME_NOT_NOW)
