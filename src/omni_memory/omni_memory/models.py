"""Data model for OMNI memory records.

Pure Python — no ROS, no Supabase imports. Safe to run on a desktop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

VALID_SOURCES = ("conversation", "observation", "system")

# Columns whose values the DB assigns; never sent on insert.
_DB_ASSIGNED = ("id", "created_at")


@dataclass
class MemoryRecord:
    """One self-contained memory fact.

    Mirrors the `memories` table. `id` and `created_at` are populated by the DB
    and only present on records that have been stored / read back.
    """

    content: str
    person: Optional[str] = None
    source: str = "conversation"
    location: Optional[str] = None
    session_id: Optional[str] = None
    importance: int = 3
    embedding: Optional[Sequence[float]] = None
    id: Optional[str] = None
    created_at: Optional[str] = None  # ISO-8601 string as returned by Supabase
    # Transient: cosine similarity from a retrieve() call. Not a column; never
    # sent on insert. Only populated on records returned by similarity search.
    similarity: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.content or not self.content.strip():
            raise ValueError("MemoryRecord.content must be a non-empty string")
        if self.source not in VALID_SOURCES:
            raise ValueError(
                f"MemoryRecord.source must be one of {VALID_SOURCES}, got {self.source!r}"
            )
        if not isinstance(self.importance, int) or not (1 <= self.importance <= 5):
            raise ValueError(
                f"MemoryRecord.importance must be an int in 1..5, got {self.importance!r}"
            )

    def to_row(self) -> dict:
        """Column dict for insert. Omits DB-assigned fields and null embedding."""
        row = {
            "content": self.content,
            "person": self.person,
            "source": self.source,
            "location": self.location,
            "session_id": self.session_id,
            "importance": self.importance,
        }
        if self.embedding is not None:
            # PostgREST accepts a JSON array for a pgvector column.
            row["embedding"] = list(self.embedding)
        return row

    @classmethod
    def from_row(cls, row: dict) -> "MemoryRecord":
        """Build a record from a Supabase row dict."""
        embedding = row.get("embedding")
        if isinstance(embedding, str):
            # pgvector may serialize as "[0.1,0.2,...]"; parse defensively.
            embedding = _parse_vector_literal(embedding)
        return cls(
            content=row["content"],
            person=row.get("person"),
            source=row.get("source", "conversation"),
            location=row.get("location"),
            session_id=row.get("session_id"),
            importance=int(row.get("importance", 3)),
            embedding=embedding,
            id=row.get("id"),
            created_at=row.get("created_at"),
            similarity=row.get("similarity"),
        )


def _parse_vector_literal(text: str) -> Optional[list[float]]:
    text = text.strip()
    if not text or text in ("[]", "null"):
        return None
    return [float(x) for x in text.strip("[]").split(",") if x.strip()]
