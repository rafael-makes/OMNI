"""scene_describer.py — turn a JPEG frame into a spoken-length scene description.

DELIBERATELY ROS-FREE. Same rule as omni_memory's core library (SPEC "Constraints
that shaped the design"): this module must import and run on a desktop with no ROS
installed, so it can be exercised by scripts/describe_image.py against a saved
image with the robot powered off. Do not add rclpy/sensor_msgs imports here — put
ROS glue in frame_client.py instead.

Two independent length caps, both necessary:
  1. max_output_tokens on the request — stops the model generating a paragraph we
     then throw away (wasted latency, and we are on a ~3s budget end to end).
  2. _cap_sentences() on the reply — models overshoot a prompt's "one sentence"
     instruction often enough that a hard trim is the only reliable guard. Spoken
     output that rambles is the failure mode called out in the build request.

Errors surface as SceneDescriptionError so the caller can say something in
character rather than leaking a stack trace into OMNI's mouth.
"""
from __future__ import annotations

import base64
import os
import re

# JPEG SOI marker. frame_server always sends JPEG; anything else means a wire
# or config mistake, and it is cheaper to fail here than to pay for an API call.
_JPEG_MAGIC = b"\xff\xd8\xff"

# Flash-lite is the right tier here: describing a visible scene needs no reasoning,
# and we are on a ~3s spoken-response budget. Verified available on this API key —
# note there is no plain "gemini-3.1-flash" endpoint, only the -lite variant.
DEFAULT_MODEL = "gemini-3.1-flash-lite"

DEFAULT_PROMPT = (
    "You are the eyes of OMNI, a small indoor robot. Describe what is in front of "
    "you in at most two short sentences, as if speaking aloud. Mention only what "
    "you can actually see. Do not narrate the image as a photograph."
)

# ~2 short spoken sentences. Generous enough that the model finishes its thought
# (a mid-sentence cutoff reads as a glitch when spoken) but not a paragraph.
DEFAULT_MAX_TOKENS = 120
DEFAULT_MAX_SENTENCES = 2


# An 8x8 grey JPEG, for warmup() only. Embedded as bytes rather than generated so
# this module keeps no image-library dependency (it must import on a bare desktop).
_WARMUP_JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDACgcHiMeGSgjISMtKygwPGRBPDc3PHtYXUlkkYCZlo+A"
    "jIqgtObDoKrarYqMyP/L2u71////m8H////6/+b9//j/2wBDASstLTw1PHZBQXb4pYyl+Pj4+Pj4"
    "+Pj4+Pj4+Pj4+Pj4+Pj4+Pj4+Pj4+Pj4+Pj4+Pj4+Pj4+Pj4+Pj4+Pj4+Pj/wAARCAAIAAgDASIA"
    "AhEBAxEB/8QAFAABAAAAAAAAAAAAAAAAAAAAAP/EABQQAQAAAAAAAAAAAAAAAAAAAAD/xAAUAQEA"
    "AAAAAAAAAAAAAAAAAAAA/8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAwDAQACEQMRAD8AAA//2Q=="
)


class SceneDescriptionError(RuntimeError):
    """The description could not be produced (API error, empty reply, ...)."""


def load_prompt(path: str | None) -> str:
    """Read the scene prompt from disk, falling back to DEFAULT_PROMPT.

    Never raises — a missing or unreadable prompt file degrades to the built-in
    default rather than taking the tool offline.
    """
    if not path:
        return DEFAULT_PROMPT
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read().strip()
        return text or DEFAULT_PROMPT
    except OSError:
        return DEFAULT_PROMPT


def _cap_sentences(text: str, max_sentences: int) -> str:
    """Collapse whitespace and keep at most max_sentences whole sentences.

    Whole sentences only: truncating mid-clause is obvious and jarring when the
    result is read aloud.
    """
    flat = re.sub(r"\s+", " ", text).strip()
    if not flat or max_sentences <= 0:
        return flat
    # Split after ., ! or ? followed by whitespace. Keeps the terminator attached.
    parts = re.findall(r".*?[.!?](?:\s|$)|.+$", flat)
    if len(parts) <= max_sentences:
        return flat
    return "".join(parts[:max_sentences]).strip()


class SceneDescriber:
    """Wraps one Gemini vision call. Cheap to construct, safe to keep around."""

    def __init__(
        self,
        *,
        client=None,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        prompt: str | None = None,
        prompt_path: str | None = None,
        max_sentences: int = DEFAULT_MAX_SENTENCES,
        max_output_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        """
        client        — pre-built genai.Client; injected by tests. If None, one is
                        built lazily on first describe() so constructing this
                        object never needs an API key or network.
        prompt        — literal system prompt; wins over prompt_path.
        prompt_path   — path to scene_prompt.txt.
        max_sentences — hard cap applied to the reply.
        """
        self._client = client
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._model = model
        self._prompt = prompt if prompt is not None else load_prompt(prompt_path)
        self._max_sentences = max_sentences
        self._max_output_tokens = max_output_tokens

    @property
    def prompt(self) -> str:
        return self._prompt

    def _ensure_client(self):
        if self._client is None:
            if not self._api_key:
                raise SceneDescriptionError(
                    "GEMINI_API_KEY is not set — cannot reach the vision endpoint. "
                    "It lives in ~/.bashrc, so this needs an interactive shell."
                )
            try:
                from google import genai
            except ImportError as exc:  # pragma: no cover - env problem, not logic
                raise SceneDescriptionError(f"google-genai is not installed: {exc}") from exc
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def warmup(self) -> bool:
        """Make one throwaway request so the first REAL one is fast.

        The very first vision call in a process costs ~3.4s versus ~0.7s for every
        call after it (measured 2026-07-18). The cost is per-process connection
        setup — TLS handshake and client init inside google-genai — not per-request
        and not per-SceneDescriber: a brand-new describer in an already-warm process
        is immediately fast. Left unwarmed, that penalty lands squarely on the first
        "what do you see?" of a session, which is the one a person actually notices.

        Best-effort and silent: returns False on any failure. Call it from a daemon
        thread at startup — it must never block node init or raise.
        """
        try:
            self.describe(_WARMUP_JPEG)
            return True
        except Exception:  # noqa: BLE001 - warmup is best-effort by design
            return False

    def describe(self, jpeg: bytes) -> str:
        """JPEG bytes -> a short description string. Raises on failure.

        Validation happens before the client is built so bad input costs nothing.
        """
        if not jpeg:
            raise ValueError("image is empty — no frame to describe")
        if not jpeg.startswith(_JPEG_MAGIC):
            raise ValueError(
                "image is not JPEG (missing SOI marker) — frame_server should "
                "always send JPEG"
            )

        client = self._ensure_client()
        from google.genai import types as genai_types

        try:
            response = client.models.generate_content(
                model=self._model,
                contents=[
                    genai_types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"),
                ],
                config=genai_types.GenerateContentConfig(
                    system_instruction=self._prompt,
                    max_output_tokens=self._max_output_tokens,
                    # Deterministic-ish: we want a plain report of what is there,
                    # not creative variation on every call.
                    temperature=0.2,
                    # Thinking OFF. Two reasons, both load-bearing:
                    #  - Reasoning tokens are drawn from max_output_tokens, so with
                    #    thinking on the models spend the whole budget deliberating
                    #    and emit a truncated fragment ("There is a red square").
                    #    Observed on gemini-3-flash-preview and gemini-2.5-flash.
                    #  - It costs seconds we do not have on a spoken-response path.
                    # Describing what is plainly in view needs no deliberation.
                    thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
                ),
            )
        except Exception as exc:  # noqa: BLE001 - any transport/API error is fatal here
            raise SceneDescriptionError(f"vision request failed: {exc}") from exc

        text = (getattr(response, "text", None) or "").strip()
        if not text:
            raise SceneDescriptionError("the vision model returned an empty description")

        return _cap_sentences(text, self._max_sentences)
