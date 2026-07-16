"""Unit tests for the Step 5 memory-integration formatting helpers.

Pure logic — no ROS, no network. Run: python -m pytest test_memory_format.py
(The live end-to-end pass test is manual; see scripts/step5_manual_gate.md.)
"""
from behavior_node.memory_format import coalesce_transcript, wrap_memory_context


def test_wrap_empty_returns_blank():
    assert wrap_memory_context("") == ""
    assert wrap_memory_context("   \n ") == ""
    assert wrap_memory_context(None) == ""


def test_wrap_nonempty_prepends_header():
    block = "Relevant things you remember:\n- Rafael prefers coffee black at 6 AM."
    out = wrap_memory_context(block)
    assert out.startswith("[MEMORY]")
    assert block in out


def test_coalesce_merges_consecutive_same_speaker():
    segments = [
        ("User", "what "), ("User", "time is my "), ("User", "coffee?"),
        ("OMNI", "You take it "), ("OMNI", "at 6 AM."),
    ]
    out = coalesce_transcript(segments)
    assert out == "User: what time is my coffee?\nOMNI: You take it at 6 AM."


def test_coalesce_alternating_speakers_keeps_order():
    segments = [
        ("User", "hi"), ("OMNI", "hello"), ("User", "bye"), ("OMNI", "goodbye"),
    ]
    out = coalesce_transcript(segments)
    assert out.splitlines() == ["User: hi", "OMNI: hello", "User: bye", "OMNI: goodbye"]


def test_coalesce_ignores_empty_fragments():
    segments = [("User", ""), ("User", None), ("OMNI", "  "), ("User", "real text")]
    out = coalesce_transcript(segments)
    assert out == "User: real text"


def test_coalesce_empty_input():
    assert coalesce_transcript([]) == ""
    assert coalesce_transcript([("User", ""), ("OMNI", None)]) == ""
