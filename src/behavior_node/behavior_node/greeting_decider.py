"""greeting_decider.py — decide whether an arriving person is worth greeting.

DELIBERATELY ROS-FREE, same rule as scene_describer and omni_memory's core: this
must import and run on a desktop with no ROS installed so the decision can be
exercised offline. ROS glue belongs in behavior_node.py.

WHY THIS IS A SEPARATE ONE-SHOT CALL AND NOT THE LIVE SESSION
-------------------------------------------------------------
Deciding "greet or stay quiet, and if greeting, what words" is a one-shot text
task. It needs no microphone, no audio duplex, and no tool loop. Running it as a
`generateContent` call (the same path scene_describer uses) means the SILENCE
outcome — which is the common one — costs nothing at all: no Gemini Live session,
no ALSA handoff, no wake-word teardown, no 30s of the robot sitting in LISTENING
with an open mic because it decided not to say anything.

Only when the answer is "greet" does behavior_node open a Live session, using the
composed line as `initial_prompt` — the mechanism `_on_say` and `_on_safety_fault`
already use. That also leaves the person in a live conversation they can simply
reply to, which is the point of being greeted.

WHAT THIS MODULE DOES NOT DECIDE
--------------------------------
Suppression is not the model's job. Robot state, the per-person cooldown, and
the low-power check are all enforced in behavior_node BEFORE this is called, so
the model is only ever asked about greetings that are already permissible. A
prompt cannot be talked out of a rule; an `if` can't be. Keep it that way.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

# Same tier and reasoning as scene_describer: no reasoning required, and we are
# on a spoken-latency budget. There is no plain "gemini-3.1-flash" endpoint.
DEFAULT_MODEL = "gemini-3.1-flash-lite"

# One short spoken sentence. Small on purpose — an unprompted greeting that runs
# long is worse than no greeting, because nobody asked for it.
DEFAULT_MAX_TOKENS = 80

# The literal token the model must emit to decline. Checked case-insensitively
# and tolerant of surrounding punctuation, because models like to decorate.
_SILENCE_TOKEN = "SILENCE"

DEFAULT_PROMPT = (
    "You are OMNI, a small indoor robot with a dry, courteous, slightly fussy "
    "manner. Someone has just walked into view. Decide whether to greet them "
    "out loud.\n"
    "\n"
    "Staying silent is a completely normal, correct outcome — most of the time "
    "it is the right one. A person who has been gone only a short while "
    "(roughly ten minutes or less) has not 'arrived'; they stepped out and came "
    "back, and greeting them again is intrusive and makes you seem broken. "
    "Greet someone when they genuinely have just arrived: a long absence, or "
    "the first time you have seen them today.\n"
    "\n"
    "If you decide to stay silent, reply with exactly one word: SILENCE\n"
    "\n"
    "If you decide to greet them, reply with the single short sentence you "
    "would say out loud — nothing else. No quotation marks, no stage "
    "directions, no explanation of your reasoning. Use their name. Keep it to "
    "one sentence. If you remember something specific and recent about them, "
    "you may refer to it lightly, but do not interrogate them and do not recite "
    "a list of facts back at them."
)


class GreetingDecisionError(RuntimeError):
    """The decision could not be produced (API error, empty reply, ...)."""


@dataclass(frozen=True)
class GreetingDecision:
    """Either a line to say, or silence. ``line`` is empty when speak is False."""

    speak: bool
    line: str = ""
    reason: str = ""

    @classmethod
    def silent(cls, reason: str) -> "GreetingDecision":
        return cls(speak=False, line="", reason=reason)


def load_prompt(path: Optional[str]) -> str:
    """Read the greeting prompt from disk, falling back to the built-in default.

    Never raises: a missing prompt file degrades to the default rather than
    taking greetings offline entirely.
    """
    if not path:
        return DEFAULT_PROMPT
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read().strip()
        return text or DEFAULT_PROMPT
    except OSError:
        return DEFAULT_PROMPT


def _first_sentence(text: str) -> str:
    """Collapse whitespace, keep one whole sentence, strip wrapping quotes.

    Models hand back `"Hello, Rafael!"` with the quotes included often enough
    that stripping them is not optional — spoken aloud, a stray quote mark
    becomes an audible stumble.
    """
    flat = re.sub(r"\s+", " ", text).strip()
    flat = flat.strip('"“”‘’\'')
    if not flat:
        return ""
    parts = re.findall(r".*?[.!?](?:\s|$)|.+$", flat)
    return (parts[0] if parts else flat).strip()


def describe_absence(away_duration: Optional[float]) -> str:
    """Render away_duration as something a language model reasons well about.

    Raw seconds invite arithmetic errors and, worse, invite the model to quote
    the number back ("Welcome back after 613 seconds"). A phrase does not.
    """
    if away_duration is None:
        return "You have not seen this person before today — this is a first sighting."
    minutes = away_duration / 60.0
    if minutes < 1:
        return "They were out of sight for well under a minute — they barely moved."
    if minutes < 10:
        return (f"They were away for about {minutes:.0f} minutes — "
                f"less than ten, so a short absence.")
    # At and above the ten-minute line, say so explicitly. Left to infer it from
    # the number alone, the model read "about 10 minutes" as being inside the
    # quiet band and stayed silent on a genuine return (measured 2026-07-19).
    if minutes < 60:
        return (f"They were away for about {minutes:.0f} minutes — "
                f"ten minutes or more, so they genuinely left and came back.")
    hours = minutes / 60.0
    if hours < 24:
        return f"They were away for about {hours:.1f} hours — a long absence."
    return "They were away for more than a day."


class GreetingDecider:
    """Wraps one Gemini text call. Cheap to construct, safe to keep around."""

    def __init__(
        self,
        *,
        client=None,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        prompt: Optional[str] = None,
        prompt_path: Optional[str] = None,
        max_output_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        """
        client      — pre-built genai.Client; injected by tests. If None, one is
                      built lazily on first decide(), so constructing this never
                      needs an API key.
        prompt      — literal prompt text; wins over prompt_path.
        prompt_path — path to greeting_prompt.txt.
        """
        self._client = client
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._model = model
        self._prompt = prompt if prompt is not None else load_prompt(prompt_path)
        self._max_output_tokens = max_output_tokens

    @property
    def prompt(self) -> str:
        return self._prompt

    def _ensure_client(self):
        if self._client is None:
            if not self._api_key:
                raise GreetingDecisionError(
                    "GEMINI_API_KEY is not set — cannot reach the text endpoint. "
                    "It lives in ~/.bashrc, so this needs an interactive shell."
                )
            try:
                from google import genai
            except ImportError as exc:  # pragma: no cover - env problem, not logic
                raise GreetingDecisionError(
                    f"google-genai is not installed: {exc}") from exc
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def warmup(self) -> bool:
        """One throwaway request so the first real decision is fast.

        Same per-process connection cost scene_describer documents (~3.4s cold vs
        ~0.7s warm). Here it matters more, not less: a greeting is on a ~3s budget
        from face identification, and unlike describe_scene nobody asked for it,
        so a late one lands after the person has already walked past.

        Best-effort and silent — returns False on any failure.
        """
        try:
            self.decide(identity="warmup", away_duration=None, memory_context="")
            return True
        except Exception:  # noqa: BLE001 - warmup is best-effort by design
            return False

    def decide(
        self,
        *,
        identity: str,
        away_duration: Optional[float],
        memory_context: str = "",
        period: str = "",
    ) -> GreetingDecision:
        """Compose the greeting, or decline. Raises GreetingDecisionError on failure.

        The caller has already established that greeting is permitted; this only
        answers "is it *warranted*, and what would you say".
        """
        name = (identity or "").strip()
        if not name:
            return GreetingDecision.silent("no identity to greet")

        client = self._ensure_client()
        from google.genai import types as genai_types

        lines = [
            f"The person who just came into view is {name.capitalize()}.",
            describe_absence(away_duration),
        ]
        if period:
            lines.append(f"It is currently {period}.")
        if memory_context and memory_context.strip():
            lines.append(
                "Here is what you remember about them:\n" + memory_context.strip())
        else:
            lines.append("You have no specific memories of them to draw on.")
        lines.append("Greet them, or reply SILENCE.")

        try:
            response = client.models.generate_content(
                model=self._model,
                contents="\n\n".join(lines),
                config=genai_types.GenerateContentConfig(
                    system_instruction=self._prompt,
                    max_output_tokens=self._max_output_tokens,
                    # Some variation is wanted here — the same person arriving
                    # every morning should not hear a byte-identical sentence.
                    temperature=0.7,
                    # Thinking OFF, for both of scene_describer's reasons: the
                    # reasoning tokens come out of max_output_tokens (a low cap
                    # then yields a truncated fragment rather than a short
                    # answer), and it costs seconds this path does not have.
                    thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
                ),
            )
        except Exception as exc:  # noqa: BLE001 - surface as our own error type
            raise GreetingDecisionError(f"greeting call failed: {exc}") from exc

        raw = (getattr(response, "text", "") or "").strip()
        if not raw:
            # An empty reply is ambiguous, and the safe reading of ambiguous is
            # silence: a missed greeting is invisible, a spurious one is not.
            return GreetingDecision.silent("model returned an empty reply")

        stripped = raw.strip().strip('".!').upper()
        if stripped.startswith(_SILENCE_TOKEN):
            return GreetingDecision.silent("model chose silence")

        line = _first_sentence(raw)
        if not line:
            return GreetingDecision.silent("model reply had no usable sentence")
        return GreetingDecision(speak=True, line=line, reason="model chose to greet")
