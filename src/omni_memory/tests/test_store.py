"""Step 1 pass test.

Gate: from a desktop (or the Pi over WireGuard), insert 3 records into the live
omni-core `memories` table, read them back, and verify every field round-trips.

Requires SUPABASE_URL + SUPABASE_SERVICE_KEY in the environment (or in a `.env`
at the project root). If they are absent the test is skipped, not failed, so the
suite stays runnable on a machine with no VPS access.
"""
from __future__ import annotations

import os
import uuid

import pytest

from omni_memory.models import MemoryRecord
from omni_memory.store import MemoryStore, load_env

# Pick up .env if present so `pytest` works without exporting vars by hand.
load_env()

pytestmark = pytest.mark.skipif(
    not (os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY")),
    reason="SUPABASE_URL / SUPABASE_SERVICE_KEY not set; skipping live Supabase test",
)


@pytest.fixture()
def store():
    return MemoryStore()


@pytest.fixture()
def session_tag() -> str:
    # Unique marker so this run's rows are isolated and cleanable.
    return f"pytest-{uuid.uuid4().hex[:12]}"


@pytest.fixture()
def sample_records(session_tag: str) -> list[MemoryRecord]:
    return [
        MemoryRecord(
            content="Rafael prefers coffee around 6 AM.",
            person="rafael",
            source="conversation",
            location="kitchen",
            session_id=session_tag,
            importance=4,
        ),
        MemoryRecord(
            content="The workshop belt tension is checked every Sunday.",
            person=None,  # general / household
            source="observation",
            location="workshop",
            session_id=session_tag,
            importance=2,
        ),
        MemoryRecord(
            content="OMNI was made taller in July 2026; the old map is stale.",
            person=None,
            source="system",
            location=None,
            session_id=session_tag,
            importance=5,
        ),
    ]


def test_insert_read_roundtrip(store: MemoryStore, sample_records, session_tag):
    stored: list[MemoryRecord] = []
    try:
        # --- insert ---
        for rec in sample_records:
            saved = store.store(rec)
            assert saved.id is not None, "store() must return the DB-assigned id"
            assert saved.created_at is not None, "store() must return created_at"
            stored.append(saved)

        # --- read back by id and verify field-by-field round-trip ---
        by_content = {r.content: r for r in sample_records}
        for saved in stored:
            fetched = store.get_by_id(saved.id)
            assert fetched is not None, f"row {saved.id} not found on read-back"
            original = by_content[fetched.content]
            assert fetched.content == original.content
            assert fetched.person == original.person
            assert fetched.source == original.source
            assert fetched.location == original.location
            assert fetched.session_id == original.session_id
            assert fetched.importance == original.importance
            assert fetched.id == saved.id
            # Step 1: embeddings not populated yet.
            assert fetched.embedding is None

        # --- recent() returns our rows, newest first, filterable by person ---
        recent = store.recent(50)
        recent_ids = {r.id for r in recent}
        assert {s.id for s in stored} <= recent_ids

        rafael = store.recent(50, person="rafael")
        assert all(r.person == "rafael" for r in rafael)
        assert any(r.session_id == session_tag for r in rafael)
    finally:
        # Clean up this run's rows so the live table isn't polluted.
        for saved in stored:
            if saved.id:
                store.client.table(store.table).delete().eq("id", saved.id).execute()


def test_recent_ordering(store: MemoryStore, session_tag):
    ids: list[str] = []
    try:
        for i in range(3):
            saved = store.store(
                MemoryRecord(
                    content=f"ordering probe {i}",
                    session_id=session_tag,
                    source="system",
                )
            )
            ids.append(saved.id)

        recent = store.recent(100)
        ours = [r for r in recent if r.session_id == session_tag]
        # Newest insert should appear before the oldest in recent() output.
        positions = {r.id: idx for idx, r in enumerate(ours)}
        assert positions[ids[-1]] < positions[ids[0]], "recent() must be newest-first"
    finally:
        for _id in ids:
            store.client.table(store.table).delete().eq("id", _id).execute()
