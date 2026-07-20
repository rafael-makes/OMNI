"""Offline tests for greeting_decider. No ROS, no network, no API key.

Run from the package root:  python3 -m pytest test_greeting_decider.py -q

Same convention as test_scene_describer.py — a fake client is injected so the
parsing and silence paths are exercised without spending a call.
"""

from __future__ import annotations

import pytest

from behavior_node.greeting_decider import (
    GreetingDecider,
    GreetingDecisionError,
    describe_absence,
    load_prompt,
)


class _FakeResponse:
    def __init__(self, text):
        self.text = text


class _FakeClient:
    """Stands in for genai.Client. Records the last request for inspection."""

    def __init__(self, reply="Good evening, Rafael."):
        self._reply = reply
        self.last_contents = None
        self.last_config = None
        self.calls = 0

        outer = self

        class _Models:
            def generate_content(self, *, model, contents, config):
                outer.calls += 1
                outer.last_contents = contents
                outer.last_config = config
                if isinstance(outer._reply, Exception):
                    raise outer._reply
                return _FakeResponse(outer._reply)

        self.models = _Models()


def decider(reply="Good evening, Rafael."):
    client = _FakeClient(reply)
    return GreetingDecider(client=client, prompt="test-prompt"), client


# ── the silence path ──────────────────────────────────────────────────────────

def test_silence_token_yields_silence():
    d, _ = decider("SILENCE")
    result = d.decide(identity="rafael", away_duration=120.0)
    assert result.speak is False
    assert result.line == ""


def test_silence_token_tolerates_decoration():
    """Models add punctuation and casing of their own accord."""
    for reply in ('silence', 'SILENCE.', '"SILENCE"', ' Silence ', 'SILENCE!'):
        d, _ = decider(reply)
        assert d.decide(identity="rafael", away_duration=60.0).speak is False, reply


def test_empty_reply_is_treated_as_silence():
    """Ambiguity resolves to silence: a missed greeting is invisible, a spurious
    one is not."""
    for reply in ("", "   ", None):
        d, _ = decider(reply)
        assert d.decide(identity="rafael", away_duration=900.0).speak is False


def test_no_identity_is_silence_without_calling_the_api():
    d, client = decider()
    assert d.decide(identity="", away_duration=900.0).speak is False
    assert client.calls == 0


# ── the speaking path ─────────────────────────────────────────────────────────

def test_greeting_line_is_returned():
    d, _ = decider("Good evening, Rafael.")
    result = d.decide(identity="rafael", away_duration=3600.0)
    assert result.speak is True
    assert result.line == "Good evening, Rafael."


def test_wrapping_quotes_are_stripped():
    """Spoken aloud, a stray quote mark is an audible stumble."""
    d, _ = decider('"Welcome back, Rafael."')
    assert d.decide(identity="rafael", away_duration=3600.0).line == \
        "Welcome back, Rafael."


def test_only_one_sentence_survives():
    d, _ = decider("Welcome back, Rafael. I trust the soldering went well. "
                   "Shall I fetch the multimeter?")
    result = d.decide(identity="rafael", away_duration=3600.0)
    assert result.line == "Welcome back, Rafael."


def test_multiline_reply_is_flattened():
    d, _ = decider("Welcome back,\n  Rafael.\n")
    assert d.decide(identity="rafael", away_duration=3600.0).line == \
        "Welcome back, Rafael."


# ── what the model is told ────────────────────────────────────────────────────

def test_absence_is_described_not_quoted_as_seconds():
    """Raw seconds invite the model to read the number back out loud."""
    d, client = decider()
    d.decide(identity="rafael", away_duration=613.0)
    assert "613" not in client.last_contents
    assert "10 minutes" in client.last_contents


def test_first_sighting_is_flagged_distinctly():
    d, client = decider()
    d.decide(identity="rafael", away_duration=None)
    assert "first sighting" in client.last_contents


def test_memory_context_is_included_when_present():
    d, client = decider()
    d.decide(identity="rafael", away_duration=3600.0,
             memory_context="- Rafael is building a robot called OMNI.")
    assert "building a robot" in client.last_contents


def test_absent_memory_is_stated_explicitly():
    """Saying 'you have no memories' beats leaving a gap the model fills in."""
    d, client = decider()
    d.decide(identity="rafael", away_duration=3600.0, memory_context="")
    assert "no specific memories" in client.last_contents


def test_thinking_is_disabled():
    """Reasoning tokens come out of max_output_tokens — with thinking on, a low
    cap yields a truncated fragment instead of a short answer."""
    d, client = decider()
    d.decide(identity="rafael", away_duration=3600.0)
    assert client.last_config.thinking_config.thinking_budget == 0


# ── failure handling ──────────────────────────────────────────────────────────

def test_api_failure_raises_our_own_error_type():
    d, _ = decider(RuntimeError("503 backend unavailable"))
    with pytest.raises(GreetingDecisionError):
        d.decide(identity="rafael", away_duration=3600.0)


def test_warmup_swallows_failure():
    d, _ = decider(RuntimeError("no network"))
    assert d.warmup() is False


def test_missing_api_key_is_a_clear_error():
    d = GreetingDecider(client=None, api_key="", prompt="p")
    with pytest.raises(GreetingDecisionError, match="GEMINI_API_KEY"):
        d.decide(identity="rafael", away_duration=1.0)


# ── prompt loading ────────────────────────────────────────────────────────────

def test_missing_prompt_file_falls_back_to_default():
    assert load_prompt("/nonexistent/greeting_prompt.txt").startswith("You are OMNI")
    assert load_prompt("").startswith("You are OMNI")
    assert load_prompt(None).startswith("You are OMNI")


def test_prompt_file_is_read_when_present(tmp_path):
    p = tmp_path / "greeting_prompt.txt"
    p.write_text("custom greeting rules")
    assert load_prompt(str(p)) == "custom greeting rules"


def test_shipped_prompt_states_the_silence_contract():
    """The prompt and the parser must agree on the exact declining token."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "config", "greeting_prompt.txt")) as fh:
        text = fh.read()
    assert "SILENCE" in text
    assert "ten minutes" in text   # the short-absence guidance


# ── absence phrasing ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("seconds,expected", [
    (None, "first sighting"),
    (30.0, "under a minute"),
    (300.0, "short absence"),
    (1800.0, "30 minutes"),
    (7200.0, "long absence"),
    (200000.0, "more than a day"),
])
def test_absence_phrasing(seconds, expected):
    assert expected in describe_absence(seconds)


def test_ten_minute_boundary_is_stated_not_implied():
    """Measured 2026-07-19: left to infer the rule from the number alone, the
    model read 'about 10 minutes' as inside the under-ten quiet band and stayed
    silent on a genuine return — the exact case the spec requires a greeting for.
    Both sides of the line now say which side they are on, in words."""
    assert "less than ten" in describe_absence(540.0)          # 9 min
    assert "ten minutes or more" in describe_absence(600.0)    # 10 min
    assert "ten minutes or more" in describe_absence(620.0)


def test_prompt_and_phrasing_agree_on_the_boundary():
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "config", "greeting_prompt.txt")) as fh:
        text = fh.read().lower()
    assert "ten minutes or more" in text
    assert "less than about ten minutes" in text
