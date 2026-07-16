"""Step 3 pass test — transcript summarizer.

Gate (per SPEC): run summarize_transcript() against three fixtures —
- fact_rich: yields several sane, self-contained memory records
- small_talk: yields nothing ([])
- correction: yields a record reflecting the corrected fact

Requires GEMINI_API_KEY (Gemini text API). Skips (not fails) without it. No
Supabase needed — the summarizer is storage-independent.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from omni_memory.models import MemoryRecord
from omni_memory.store import load_env
from omni_memory.summarizer import summarize_transcript

load_env()

pytestmark = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set; skipping Gemini summarizer test",
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> str:
    return (FIXTURES / name).read_text()


# Module-scoped so each fixture transcript is summarized exactly once — keeps us
# under the Gemini free-tier 5 req/min cap (3 API calls total for this module).
@pytest.fixture(scope="module")
def fact_rich_records():
    return summarize_transcript(_load("fact_rich.txt"))


@pytest.fixture(scope="module")
def correction_records():
    return summarize_transcript(
        _load("correction.txt"), session_id="sess-xyz", default_person="rafael"
    )


def _assert_record_shape(records):
    for r in records:
        assert isinstance(r, MemoryRecord)
        assert r.content and r.content.strip()
        assert r.source == "conversation"
        assert 1 <= r.importance <= 5
        # No meta / transcript-excerpt phrasing.
        low = r.content.lower()
        assert "the user" not in low
        assert "user said" not in low
        assert not low.startswith("omni:")


def test_fact_rich_extracts_multiple_facts(fact_rich_records):
    records = fact_rich_records
    assert len(records) >= 3, f"expected several facts, got {len(records)}"
    _assert_record_shape(records)

    joined = " || ".join(r.content.lower() for r in records)
    # Durable facts we planted should surface (allow phrasing latitude).
    assert "coffee" in joined
    assert "peanut" in joined
    # Per-person attribution works: at least one record is attributed to someone.
    assert any(r.person for r in records), "expected at least one person-attributed record"
    assert {r.person for r in records} & {"rafael", "sofia"}


def test_small_talk_yields_nothing():
    records = summarize_transcript(_load("small_talk.txt"))
    assert records == [], f"small talk should yield no memories, got {records}"


def test_correction_reflects_new_value(correction_records):
    records = correction_records
    assert records, "correction transcript should yield at least one record"
    _assert_record_shape(records)
    joined = " || ".join(r.content for r in records)
    # The corrected value (6:30) must appear.
    assert "6:30" in joined, f"corrected time not captured: {joined!r}"


def test_empty_transcript_no_api_call():
    # Empty/whitespace short-circuits without touching the API.
    assert summarize_transcript("") == []
    assert summarize_transcript("   \n  ") == []


def test_session_and_person_stamping(correction_records):
    records = correction_records
    assert records
    assert all(r.session_id == "sess-xyz" for r in records)
    # default_person fills in where the model didn't attribute a person.
    assert all(r.person in ("rafael", "sofia") for r in records)
