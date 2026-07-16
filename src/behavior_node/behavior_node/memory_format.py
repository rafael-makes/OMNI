"""memory_format.py — pure formatting helpers for the memory integration.

No ROS, no google.genai imports, so these are unit-testable on their own.
Used by gemini_bridge (transcript coalescing) and memory_client (context wrap).
"""
from __future__ import annotations

from typing import Iterable

_MEMORY_HEADER = (
    "[MEMORY] Things you remember about the people here. Use them naturally when "
    "relevant to the conversation; never recite or list them unprompted.\n"
)


def wrap_memory_context(context_block: str) -> str:
    """Wrap a retrieved context block with an instruction for the model.
    Returns '' for an empty block (nothing to inject)."""
    context_block = (context_block or "").strip()
    if not context_block:
        return ""
    return _MEMORY_HEADER + context_block


def coalesce_transcript(segments: Iterable[tuple[str, str]]) -> str:
    """Turn streamed (speaker, text_fragment) pairs into a labeled transcript.

    Gemini streams transcription in small fragments; consecutive fragments from
    the same speaker are concatenated into one line. Fragments already carry
    their own spacing, so they are joined without inserting extra spaces.
    Returns '' if there is nothing with real text.
    """
    lines: list[list[str]] = []  # [speaker, accumulated_text]
    for speaker, text in segments:
        if not text:
            continue
        if lines and lines[-1][0] == speaker:
            lines[-1][1] += text
        else:
            lines.append([speaker, text])
    out = [f"{spk}: {txt.strip()}" for spk, txt in lines if txt.strip()]
    return "\n".join(out)
