"""Transcript -> discrete memory records, via the Gemini text API.

Turns a conversation transcript into a list of self-contained `MemoryRecord`s
following the SPEC memory-record rules. Small-talk-only / empty transcripts
yield []. Pure Python, no ROS; google-genai imported lazily.
"""
from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Optional

from .models import MemoryRecord

DEFAULT_TEXT_MODEL = "gemini-2.5-flash"

# Generic speaker tokens a transcript's own turns get labeled with; these map to
# the provided default_person (the recognized speaker's identity).
_GENERIC_SPEAKERS = {"user", "the user", "speaker", "me", "i", "omni's user"}

# Bounded retry for TRANSIENT failures: 429 rate limits and 5xx server blips (Gemini
# returns 503 "experiencing high demand" under load). Both are temporary, and a
# dropped summarize would silently lose a whole conversation's memories. Not meant to
# mask sustained quota exhaustion or a real outage — just to ride out short bursts.
_TRANSIENT_RETRIES = 3
_RETRY_CAP_SECONDS = 30.0

SYSTEM_PROMPT = """You extract durable, long-term memories from a conversation \
transcript for a home robot named OMNI.

Return a JSON array. Each element is ONE self-contained memory with fields:
- "content": string. One durable fact, preference, event, commitment, or \
correction, written in the THIRD PERSON and fully self-contained — no "the user \
said", no dangling pronouns, no transcript quoting. Example: "Rafael prefers \
coffee around 6 AM."
- "person": string or null. Lowercase first name of the person the memory is \
about or from (e.g. "rafael", "sofia"), or null if it is general/household or \
the person is unknown.
- "importance": integer 1-5. 5 = safety/health or major commitments; 3 = an \
ordinary useful fact or stable preference; 1 = minor but worth keeping.
- "location": string or null. A lowercase room/area tag if clearly implied \
(e.g. "kitchen", "workshop"), else null.

Rules:
- ONE fact per element. Split compound statements into separate records.
- Extract ONLY durable information: facts, preferences, events, commitments, \
corrections.
- SKIP small talk, greetings, pleasantries, and ephemeral chatter. If there is \
nothing durable to remember, return an empty array [].
- If the conversation CORRECTS an earlier statement, record the corrected fact \
as a new record with the updated value (do not mention the old value).
- No filler, no meta-commentary. Third person only.

Return ONLY the JSON array, nothing else."""


class Summarizer(ABC):
    @abstractmethod
    def summarize(
        self,
        text: str,
        *,
        session_id: Optional[str] = None,
        default_person: Optional[str] = None,
    ) -> list[MemoryRecord]:
        ...


class GeminiSummarizer(Summarizer):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_TEXT_MODEL,
        client=None,
    ) -> None:
        self.model = model
        if client is not None:
            self._client = client
            return
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set (export it or add it to .env).")
        from google import genai  # lazy

        self._client = genai.Client(api_key=api_key)

    def summarize(
        self,
        text: str,
        *,
        session_id: Optional[str] = None,
        default_person: Optional[str] = None,
    ) -> list[MemoryRecord]:
        if not text or not text.strip():
            return []
        from google.genai import types  # lazy

        # A live transcript labels the speaker generically ("User:"), so without this
        # the model writes "The user prefers ..." — not self-contained, and OMNI ends
        # up thinking the person is literally called "user". When we know who is
        # speaking, name them so records read "Rafael prefers ..." per the SPEC rules.
        if default_person and not default_person.startswith("unknown"):
            who = default_person.capitalize()
            text = (
                f'[Context: the speaker labelled "User" is named {who}. Write their '
                f'facts using that name — e.g. "{who} prefers ..." — never "the user".]'
                f"\n\n{text}"
            )

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            # Extraction task: keep it deterministic. Higher temps occasionally
            # drop obvious facts (observed an empty result at 0.2).
            temperature=0.0,
        )
        attempt = 0
        while True:
            try:
                resp = self._client.models.generate_content(
                    model=self.model, contents=text, config=config
                )
                break
            except Exception as exc:  # noqa: BLE001 - narrowed via _is_transient
                if not (_is_transient(exc) and attempt < _TRANSIENT_RETRIES):
                    raise
                time.sleep(_retry_delay(exc, default=6.0 * (attempt + 1)))
                attempt += 1
        return _parse(resp.text, session_id=session_id, default_person=default_person)


# Reused across calls of the module-level convenience function.
_default_summarizer: Optional[Summarizer] = None


def summarize_transcript(
    text: str,
    *,
    session_id: Optional[str] = None,
    default_person: Optional[str] = None,
    summarizer: Optional[Summarizer] = None,
) -> list[MemoryRecord]:
    """Extract durable memory records from a transcript (SPEC Step 3 API).

    Empty/whitespace input returns [] without any API call. A `summarizer` may
    be injected (tests / alternate backends); otherwise a default Gemini one is
    lazily created and reused.
    """
    if not text or not text.strip():
        return []
    global _default_summarizer
    engine = summarizer or _default_summarizer
    if engine is None:
        engine = _default_summarizer = GeminiSummarizer()
    return engine.summarize(text, session_id=session_id, default_person=default_person)


def _is_transient(exc: Exception) -> bool:
    """True for retryable blips: 429 (rate limit) or any 5xx (e.g. Gemini 503
    'high demand'). Everything else — bad key, bad request — should fail fast."""
    code = getattr(exc, "code", None)
    if isinstance(code, int) and (code == 429 or 500 <= code <= 599):
        return True
    text = str(exc)
    return any(m in text for m in ("RESOURCE_EXHAUSTED", "UNAVAILABLE", "429", "503"))


def _retry_delay(exc: Exception, default: float) -> float:
    """Seconds to wait before retrying, honoring the server's hint if present."""
    text = str(exc)
    match = re.search(r"retry in ([0-9.]+)s", text) or re.search(
        r"'retryDelay': '([0-9.]+)s'", text
    )
    delay = float(match.group(1)) + 0.5 if match else default
    return min(delay, _RETRY_CAP_SECONDS)


def _parse(
    raw: Optional[str],
    *,
    session_id: Optional[str],
    default_person: Optional[str],
) -> list[MemoryRecord]:
    data = _loads_json_array(raw)
    records: list[MemoryRecord] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue

        importance = item.get("importance", 3)
        try:
            importance = int(importance)
        except (TypeError, ValueError):
            importance = 3
        importance = max(1, min(5, importance))

        person = item.get("person")
        person = person.strip().lower() or None if isinstance(person, str) else None
        # default_person identifies the speaker. A transcript labels their turns
        # generically ("User:"), so the model attributes their facts to a generic
        # speaker token — map those to default_person. Named third parties (e.g.
        # "sofia") keep their own name. Null (general/household) also falls back.
        if person in _GENERIC_SPEAKERS or person is None:
            person = default_person

        location = item.get("location")
        location = location.strip().lower() or None if isinstance(location, str) else None

        records.append(
            MemoryRecord(
                content=content,
                person=person,
                source="conversation",
                location=location,
                importance=importance,
                session_id=session_id,
            )
        )
    return records


def _loads_json_array(raw: Optional[str]) -> list:
    """Parse a JSON array, tolerating stray prose / code fences defensively."""
    if not raw:
        return []
    raw = raw.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("["), raw.rfind("]")
        if start == -1 or end == -1 or end < start:
            return []
        try:
            data = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return []
    return data if isinstance(data, list) else []
