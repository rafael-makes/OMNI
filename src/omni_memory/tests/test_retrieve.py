"""Step 2 pass test — embeddings + similarity search.

Gate (per SPEC): insert ~10 varied memories, query "belt tension" and get the
workshop-related memory ranked first; verify the person filter.

Requires SUPABASE_URL + SUPABASE_SERVICE_KEY *and* GEMINI_API_KEY. Skips (not
fails) when any is absent so the suite stays runnable offline.
"""
from __future__ import annotations

import os
import uuid

import pytest

from omni_memory.embedder import GeminiEmbedder
from omni_memory.models import MemoryRecord
from omni_memory.store import MemoryStore, load_env

load_env()

pytestmark = pytest.mark.skipif(
    not (
        os.environ.get("SUPABASE_URL")
        and os.environ.get("SUPABASE_SERVICE_KEY")
        and os.environ.get("GEMINI_API_KEY")
    ),
    reason="SUPABASE_URL / SUPABASE_SERVICE_KEY / GEMINI_API_KEY not all set",
)

# (content, person, source, location, importance)
FIXTURES = [
    ("The workshop drive belt should be re-tensioned whenever it starts to squeal.",
     None, "observation", "workshop", 3),
    ("Rafael prefers coffee around 6 AM.", "rafael", "conversation", "kitchen", 4),
    ("OMNI's lidar sits at 1210 mm after the July 2026 height change.",
     None, "system", None, 3),
    ("The basement docking station uses AprilTag id 1.", None, "system", "basement", 3),
    ("Rafael is learning ROS2 and likes to be taught step by step.",
     "rafael", "conversation", None, 4),
    ("The living-room curtains are usually closed after sunset.",
     None, "observation", "living_room", 2),
    ("Maria waters the balcony plants on Saturday mornings.",
     "maria", "conversation", "balcony", 3),
    ("The garage door remote battery was replaced in June 2026.",
     None, "observation", "garage", 2),
    ("OMNI should greet household members by name once it recognizes them.",
     None, "conversation", None, 4),
    ("Rafael's favourite debugging tool is py-spy.", "rafael", "conversation", None, 3),
]


@pytest.fixture(scope="module")
def session_tag() -> str:
    return f"pytest-s2-{uuid.uuid4().hex[:10]}"


@pytest.fixture(scope="module")
def store(session_tag):
    s = MemoryStore(embedder=GeminiEmbedder())
    inserted = []
    for content, person, source, location, importance in FIXTURES:
        saved = s.store(
            MemoryRecord(
                content=content,
                person=person,
                source=source,
                location=location,
                importance=importance,
                session_id=session_tag,
            )
        )
        inserted.append(saved)
    yield s
    # Teardown: remove this run's rows.
    s.client.table(s.table).delete().eq("session_id", session_tag).execute()


def test_store_populates_embedding(store):
    """store() with an embedder writes a 768-dim vector."""
    rows = store.recent(1)
    assert rows, "expected at least one stored row"
    fetched = store.get_by_id(rows[0].id)
    assert fetched.embedding is not None
    assert len(fetched.embedding) == 768


def test_belt_tension_ranks_workshop_first(store):
    results = store.retrieve("belt tension", k=3)
    assert results, "retrieve returned nothing"
    top = results[0]
    assert "belt" in top.content.lower(), f"unexpected top result: {top.content!r}"
    assert top.location == "workshop"
    # Similarity should be populated and be the highest in the returned set.
    assert top.similarity is not None
    assert top.similarity == max(r.similarity for r in results)


def test_person_filter_excludes_others(store):
    # Querying as 'rafael' must never surface Maria's memories, but may return
    # rafael-specific and general (person=None) records.
    results = store.retrieve("what does he like to drink", k=10, person="rafael")
    assert results
    persons = {r.person for r in results}
    assert "maria" not in persons
    assert persons <= {"rafael", None}
    # Rafael's coffee preference should surface for this query.
    assert any("coffee" in r.content.lower() for r in results)


def test_no_person_filter_sees_all(store):
    results = store.retrieve("who waters the plants", k=5)
    assert any("maria" == r.person for r in results), (
        "unfiltered retrieve should be able to return Maria's memory"
    )
