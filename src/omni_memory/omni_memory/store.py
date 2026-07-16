"""MemoryStore — thin storage abstraction over the omni-core Supabase project.

All reads/writes for the memory layer go through this class so the backend
(Supabase/pgvector today, possibly local SQLite + sqlite-vec later) stays a
single swap point. No ROS imports — Steps 1-3 run on a desktop.

Step 1 surface: store(), get_by_id(), recent(). Embeddings arrive in Step 2;
the schema/column already exist but store() sends an embedding only if the
record carries one.
"""
from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .embedder import Embedder
from .models import MemoryRecord

DEFAULT_TABLE = "memories"

# Light re-rank weights. Kept small so semantic similarity (~0..1) dominates and
# the boost only breaks near-ties. Max combined boost ~0.05.
_RECENCY_WEIGHT = 0.03
_IMPORTANCE_WEIGHT = 0.02
_RECENCY_HALFLIFE_DAYS = 30.0


def load_env(path: Optional[str] = None) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ (without overriding
    already-set vars). Walks up from CWD to find `.env` if no path is given.

    Kept dependency-free (no python-dotenv required) so the core stays lean.
    """
    env_path = Path(path) if path else _find_dotenv()
    if not env_path or not env_path.is_file():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _find_dotenv() -> Optional[Path]:
    for base in (Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parent.parent):
        candidate = base / ".env"
        if candidate.is_file():
            return candidate
    return None


class MemoryStore:
    """Storage abstraction backed by Supabase (PostgREST + service key)."""

    def __init__(
        self,
        url: Optional[str] = None,
        service_key: Optional[str] = None,
        client=None,
        table: str = DEFAULT_TABLE,
        embedder: Optional[Embedder] = None,
    ) -> None:
        self.table = table
        # Optional: when absent, store() inserts no embedding (Step 1 behavior)
        # and retrieve() is unavailable.
        self.embedder = embedder
        if client is not None:
            # Dependency injection — used by tests and future backends.
            self._client = client
            return

        url = url or os.environ.get("SUPABASE_URL")
        service_key = service_key or os.environ.get("SUPABASE_SERVICE_KEY")
        if not url or not service_key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set "
                "(pass explicitly, export them, or add a .env file)."
            )
        # Imported lazily so importing this module never requires supabase-py.
        from supabase import create_client

        self._client = create_client(url, service_key)

    @property
    def client(self):
        """Underlying Supabase client (escape hatch for tests / admin ops)."""
        return self._client

    # --- writes -----------------------------------------------------------

    def store(self, record: MemoryRecord) -> MemoryRecord:
        """Insert a record; return it with DB-assigned id and created_at.

        If the store has an embedder and the record has no embedding yet, the
        content is embedded (as a document) before insert.
        """
        row = record.to_row()
        if self.embedder is not None and "embedding" not in row:
            row["embedding"] = self.embedder.embed_document(record.content)
        resp = self._client.table(self.table).insert(row).execute()
        rows = _rows(resp)
        if not rows:
            raise RuntimeError("insert returned no row")
        return MemoryRecord.from_row(rows[0])

    def rekey(self, from_person: str, to_person: str) -> int:
        """Re-label all memories from one person id to another; return row count.

        Used when a previously unknown person is given a name — their existing
        memories carry over to the name.
        """
        if not from_person or not to_person or from_person == to_person:
            return 0
        resp = (
            self._client.table(self.table)
            .update({"person": to_person})
            .eq("person", from_person)
            .execute()
        )
        return len(_rows(resp))

    # --- reads ------------------------------------------------------------

    def get_by_id(self, memory_id: str) -> Optional[MemoryRecord]:
        resp = (
            self._client.table(self.table)
            .select("*")
            .eq("id", memory_id)
            .limit(1)
            .execute()
        )
        rows = _rows(resp)
        return MemoryRecord.from_row(rows[0]) if rows else None

    def recent(self, n: int = 10, person: Optional[str] = None) -> list[MemoryRecord]:
        """Most-recent-first records, optionally filtered to a person."""
        query = self._client.table(self.table).select("*")
        if person is not None:
            query = query.eq("person", person)
        resp = query.order("created_at", desc=True).limit(n).execute()
        return [MemoryRecord.from_row(r) for r in _rows(resp)]

    def retrieve(
        self,
        query: str,
        k: int = 5,
        person: Optional[str] = None,
        candidate_pool: Optional[int] = None,
    ) -> list[MemoryRecord]:
        """Top-k memories by cosine similarity to `query`.

        When `person` is given, results are restricted to that person's records
        plus general (person=None) records (SPEC: person-specific + general).
        A candidate pool is pulled by ANN similarity (via the match_memories
        RPC / HNSW index), then lightly re-ranked by recency and importance.
        Each returned record carries its cosine `similarity`.
        """
        if self.embedder is None:
            raise RuntimeError("retrieve() requires a MemoryStore built with an embedder")

        query_embedding = self.embedder.embed_query(query)
        pool = candidate_pool if candidate_pool is not None else max(k * 4, 20)
        resp = self._client.rpc(
            "match_memories",
            {
                "query_embedding": query_embedding,
                "match_count": pool,
                "filter_person": person,
            },
        ).execute()
        records = [MemoryRecord.from_row(r) for r in _rows(resp)]
        records.sort(key=self._rank_score, reverse=True)
        return records[:k]

    @staticmethod
    def _rank_score(record: MemoryRecord) -> float:
        """Similarity + light recency + light importance boost."""
        score = record.similarity or 0.0
        # importance 1..5 -> [-1, 1]
        score += _IMPORTANCE_WEIGHT * ((record.importance - 3) / 2.0)
        if record.created_at:
            try:
                created = datetime.fromisoformat(record.created_at)
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                age_days = (datetime.now(timezone.utc) - created).total_seconds() / 86400.0
                score += _RECENCY_WEIGHT * math.exp(-age_days / _RECENCY_HALFLIFE_DAYS)
            except ValueError:
                pass
        return score


def _rows(resp) -> list[dict]:
    """Normalize a supabase-py response to a list of row dicts."""
    data = getattr(resp, "data", resp)
    if data is None:
        return []
    return data if isinstance(data, list) else [data]
