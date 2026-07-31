"""check_in_response.py — what did they actually say to the check-in?

DELIBERATELY ROS-FREE (scene_describer / greeting_decider / person_nav rule) so
the classification is testable with the robot off.

Three answers matter, because each earns a different cooldown:

    "yeah, I'm fitting the servo bracket"   -> YES      (engaged; talk normally)
    "no, I'm good thanks"                   -> NO       (4 h, this zone)
    "not right now"                         -> NOT_NOW  (1 h, this zone)

and saying nothing at all is handled by the caller as NO_RESPONSE.

WHY KEYWORDS AND NOT A MODEL
----------------------------
The honest reason is cost and testability, not superiority. A second Gemini call
to classify one short reply would add a round trip to the end of every check-in,
and — more to the point — the thing it decides is a *cooldown*, where the penalty
for being wrong is bounded and self-correcting: misread a "no" as engagement and
OMNI simply stays for a conversation the person ends; misread engagement as a
"no" and it leaves politely and waits four hours. Neither is a disaster, and both
are cheaper than an API dependency on the exit path of a behaviour whose entire
charm is that it gets out of the way.

WHERE THIS IS FRAGILE — READ BEFORE TRUSTING IT
-----------------------------------------------
It is a keyword matcher. It will misread sarcasm, unusual phrasing, and anything
in a language other than English. Two deliberate choices bound the damage:

1. **NOT_NOW is checked before NO**, because "no, not right now" contains both
   and the softer, more specific reading is the correct one. Getting this
   backwards would turn every deferral into a four-hour lockout.
2. **The default is YES.** An unrecognised reply means the person said *something*
   — they engaged. Defaulting to a decline would let one mumbled sentence lock
   the zone out for four hours, which is the failure that actually annoys.

The upgrade path is a one-shot `generateContent` classifier behind the same
`classify_reply()` signature, with these rules kept as the offline fallback.
"""

from __future__ import annotations

import re

from .check_in_policy import OUTCOME_NO, OUTCOME_NOT_NOW, OUTCOME_YES

# "Later" — a deferral. Checked FIRST: these phrases frequently contain a bare
# "no" ("no, not right now"), and the deferral is the more specific reading.
_NOT_NOW_PATTERNS = (
    r"\bnot (right )?now\b",
    r"\bnot (at )?the moment\b",
    r"\bnot just (now|yet)\b",
    r"\bmaybe later\b",
    r"\blater\b",
    r"\bin a (bit|minute|sec|second|moment)\b",
    r"\bgive me a (minute|sec|second|moment)\b",
    r"\bone (sec|second|minute|moment)\b",
    r"\bhold on\b",
    r"\bcome back\b",
    r"\bask me (again|later)\b",
    r"\bbusy (right )?now\b",
    r"\bin the middle of\b",
)

# A refusal. No time qualifier — they do not want the help, full stop.
_NO_PATTERNS = (
    r"^\s*no\b",
    r"\bno,? thank(s| you)\b",
    r"\bno thanks\b",
    r"\bnope\b",
    r"\bnah\b",
    r"\bi'?m (all )?(good|fine|ok|okay|alright|all right)\b",
    r"\bi am (all )?(good|fine|ok|okay|alright|all right)\b",
    r"\bleave me (alone|to it)\b",
    r"\bgo away\b",
    r"\bnot interested\b",
    r"\bdon'?t need\b",
    r"\bno need\b",
    r"\bnothing\b",
)

_NOT_NOW_RE = tuple(re.compile(p, re.IGNORECASE) for p in _NOT_NOW_PATTERNS)
_NO_RE = tuple(re.compile(p, re.IGNORECASE) for p in _NO_PATTERNS)


def classify_reply(text: str) -> str:
    """Classify a spoken reply as OUTCOME_YES / OUTCOME_NO / OUTCOME_NOT_NOW.

    Empty or whitespace-only text returns OUTCOME_YES rather than inventing a
    refusal — the *caller* owns the "they said nothing at all" case, because only
    it knows whether the silence lasted long enough to count (see
    `check_in_silence_timeout`). Classifying emptiness here would double-count it.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return OUTCOME_YES

    for pattern in _NOT_NOW_RE:
        if pattern.search(cleaned):
            return OUTCOME_NOT_NOW

    for pattern in _NO_RE:
        if pattern.search(cleaned):
            return OUTCOME_NO

    # They said something we do not recognise as a refusal — that is engagement.
    return OUTCOME_YES
