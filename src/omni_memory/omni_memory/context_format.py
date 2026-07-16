"""Format retrieved memories into a plain-text block for context injection.

Kept separate from the ROS node so Step 5 (Gemini Live integration) can reuse it.
Pure Python, no ROS.
"""
from __future__ import annotations

from typing import Iterable

from .models import MemoryRecord

DEFAULT_HEADER = "Relevant things you remember:"


def format_context(
    records: Iterable[MemoryRecord],
    header: str = DEFAULT_HEADER,
) -> str:
    """Render records as a bulleted block. Returns '' when there are none.

    Each record's `content` is already a self-contained third-person statement,
    so we just bullet it and append a location tag when present.
    """
    records = list(records)
    if not records:
        return ""
    lines = [header]
    for r in records:
        location = f" (in the {r.location})" if r.location else ""
        lines.append(f"- {r.content}{location}")
    return "\n".join(lines)
