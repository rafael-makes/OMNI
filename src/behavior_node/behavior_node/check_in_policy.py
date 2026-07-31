"""check_in_policy.py — should OMNI walk over and ask what you're working on?

DELIBERATELY ROS-FREE (scene_describer / greeting_decider / person_nav rule):
this must import and run on a desktop with no ROS. It is pure decision logic over
a `person_dwelling` event plus a snapshot of the robot's condition, and it is the
single most heavily tested thing in the package — because this is the manners
layer, and manners are the whole feature.

THE ONE QUESTION
----------------
`decide()` consumes a `person_dwelling` event from `event_generator` and answers
exactly one thing: **check in now, or not?** It never composes speech, never
touches navigation, never talks to Gemini. `behavior_node` runs the state machine
that acts on an approval.

WHY THE BAR IS SET DELIBERATELY HIGH
------------------------------------
Interrupting someone's focus is how this feature gets turned off — permanently,
by the person it was built for. Every default here is chosen to fail toward
silence:

* **≥60 minutes of dwell.** Not "they sat down", but "they have been at this a
  long while".
* **A long global cooldown after ANY interaction** (≥2 h). A greeting this
  morning is a reason not to interrupt this afternoon. OMNI having *just talked
  to you* is the most reliable signal that it does not need to talk to you again.
* **"No" costs more than "not now"** (4 h vs 1 h, per zone). These are different
  sentences and they mean different things; collapsing them would teach the robot
  nothing and annoy the human twice.
* **Quiet hours.** The child sleeps.

The generator re-fires `person_dwelling` on an interval precisely so that a
declined opportunity is not the last one — the policy can approve a later firing
once a cooldown has expired, without the generator ever deciding on its behalf.

WHAT COUNTS AS AN INTERACTION
-----------------------------
Everything: a greeting, a wake-word conversation, and any completed check-in —
including one that was refused. Being told "no" is still OMNI having spent your
attention, so it starts the global cooldown exactly like a chat would.

STATE IS IN MEMORY AND DOES NOT SURVIVE A RESTART
-------------------------------------------------
Cooldowns live in this object. Restart `behavior_node` and a "no" from ten
minutes ago is forgotten, so a fresh check-in becomes possible immediately. That
is a real gap, deliberately not solved here (persisting it needs a store this
library must not depend on). The outcome log written through `memory_client` is
the raw material for fixing it later; see `outcome_records()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Iterable, Optional, Sequence

from .suppression import RobotStatus, interaction_blocked

# ── Outcomes ──────────────────────────────────────────────────────────────────
# What actually happened when OMNI asked. These strings are logged and will be
# read back by the future learning pass, so they are a stored data contract —
# renaming one orphans the history.
OUTCOME_YES = "yes"                 # engaged; a real conversation happened
OUTCOME_NO = "no"                   # declined outright ("no, I'm good")
OUTCOME_NOT_NOW = "not_now"         # declined for the moment ("not right now")
OUTCOME_NO_RESPONSE = "no_response"  # ignored; the ask timed out

ALL_OUTCOMES = (OUTCOME_YES, OUTCOME_NO, OUTCOME_NOT_NOW, OUTCOME_NO_RESPONSE)

# How much each outcome counts as a "decline" when biasing the dwell threshold.
# A hard "no" is a full decline. "Not now" and silence are softer — they say
# "wrong moment", not "wrong idea" — so they push the threshold up half as hard.
_DECLINE_WEIGHT = {
    OUTCOME_YES: 0.0,
    OUTCOME_NO: 1.0,
    OUTCOME_NOT_NOW: 0.5,
    OUTCOME_NO_RESPONSE: 0.5,
}

# Reasons `decide()` can return. Stable strings — the live verification step
# ("check policy logs show the suppression reason") reads these.
REASON_OK = "ok"
REASON_DISABLED = "check-ins disabled"
REASON_NOT_DWELLING = "not a person_dwelling event"
REASON_UNNAMED = "no named person"
REASON_NO_ZONE = "event carries no zone"
REASON_ZONE_NOT_ENABLED = "zone not enabled for check-ins"
REASON_DWELL_TOO_SHORT = "dwell too short"
REASON_QUIET_HOURS = "quiet hours"
REASON_GLOBAL_COOLDOWN = "recent interaction"
REASON_ZONE_COOLDOWN = "zone cooldown"


@dataclass(frozen=True)
class CheckInConfig:
    """Every threshold in one place. All durations in seconds."""

    enabled: bool = True

    # Dwell before a check-in is even considered. 60 minutes.
    min_dwell: float = 3600.0

    # Below this the answer is always no. Set well above the greeting floor
    # (20%): a greeting costs a sentence, a check-in costs a round trip across
    # the room and back.
    battery_floor: float = 40.0

    # No check-ins between quiet_start and quiet_end (wraps midnight).
    quiet_start: time = time(21, 0)
    quiet_end: time = time(8, 0)

    # After ANY interaction with this person — greeting, conversation, or a
    # previous check-in of any outcome.
    global_cooldown: float = 7200.0        # 2 h

    # Per (person, zone), by how the last check-in there ended.
    no_cooldown: float = 14400.0           # 4 h
    not_now_cooldown: float = 3600.0       # 1 h

    # v1.5 learning: stretch the dwell threshold in zones where declines
    # dominate. Needs at least `bias_min_samples` outcomes before it will move at
    # all, and never stretches beyond `bias_max_multiplier` x min_dwell.
    bias_enabled: bool = True
    bias_min_samples: int = 3
    bias_max_multiplier: float = 2.0

    def __post_init__(self) -> None:
        if self.min_dwell <= 0:
            raise ValueError("min_dwell must be > 0")
        if self.bias_max_multiplier < 1.0:
            raise ValueError("bias_max_multiplier must be >= 1.0")
        if self.bias_min_samples < 1:
            raise ValueError("bias_min_samples must be >= 1")


@dataclass(frozen=True)
class CheckInDecision:
    """Approved or not, and why. The reason is logged verbatim."""

    approved: bool
    reason: str
    identity: str = ""
    zone: str = ""
    dwell_duration: float = 0.0
    threshold: float = 0.0      # the effective threshold applied, after bias

    def __bool__(self) -> bool:
        return self.approved


@dataclass(frozen=True)
class OutcomeRecord:
    """One completed check-in, for the log and the future learning pass."""

    identity: str
    zone: str
    outcome: str
    when: datetime
    dwell_duration: float = 0.0

    def as_dict(self) -> dict:
        return {
            "identity": self.identity,
            "zone": self.zone,
            "outcome": self.outcome,
            "when": self.when.isoformat(timespec="seconds"),
            "dwell_duration": round(self.dwell_duration, 1),
        }

    def as_memory_text(self) -> str:
        """A sentence for the memory store.

        Written as plain prose rather than a JSON blob because omni_memory
        summarises and embeds what it is given: "Rafael declined" is retrievable
        by a later question about Rafael, a serialised dict is not.
        """
        phrasing = {
            OUTCOME_YES: "accepted the check-in and talked",
            OUTCOME_NO: "declined the check-in",
            OUTCOME_NOT_NOW: "asked to be checked in with later",
            OUTCOME_NO_RESPONSE: "did not respond to the check-in",
        }.get(self.outcome, f"check-in ended as {self.outcome}")
        minutes = self.dwell_duration / 60.0
        return (
            f"{self.identity.capitalize()} {phrasing} at the {self.zone} "
            f"after working there for about {minutes:.0f} minutes."
        )


def in_quiet_hours(now: datetime, start: time, end: time) -> bool:
    """True if `now` falls in the quiet window, which normally wraps midnight.

    start == end is read as "no quiet hours at all" rather than "quiet always" —
    the harmless reading of an ambiguous config.
    """
    if start == end:
        return False
    current = now.time()
    if start < end:
        return start <= current < end
    return current >= start or current < end


class CheckInPolicy:
    """Decides whether a `person_dwelling` event earns a check-in.

    Holds the cooldown state, so `behavior_node` keeps exactly one instance.
    Clock-agnostic in the same spirit as the event generator: every method takes
    the current time, so the tests run years of cooldowns in a millisecond.
    """

    def __init__(
        self,
        config: Optional[CheckInConfig] = None,
        *,
        zones: Iterable[str] = (),
    ) -> None:
        self.config = config or CheckInConfig()
        # Zones where a check-in is welcome. Empty means "trust the generator's
        # own dwell_zones list", since that is already configured to the same
        # rooms; supplying it here is a second, independent belt.
        self.zones = frozenset(zones)
        self._last_interaction: dict[str, datetime] = {}
        self._zone_blocked_until: dict[tuple[str, str], datetime] = {}
        self._history: list[OutcomeRecord] = []

    # ── the decision ─────────────────────────────────────────────────────────

    def decide(
        self,
        event: dict,
        status: RobotStatus,
        now: datetime,
    ) -> CheckInDecision:
        """Check in now, or not? `event` is a `person_dwelling` event as JSON."""
        cfg = self.config

        if not cfg.enabled:
            return CheckInDecision(False, REASON_DISABLED)

        if not isinstance(event, dict) or event.get("kind") != "person_dwelling":
            return CheckInDecision(False, REASON_NOT_DWELLING)

        identity = str(event.get("identity") or "").strip().lower()
        # A stranger has nobody to check in with, and event_generator does not
        # emit dwell for them anyway — this is the defensive half of that.
        if not identity or identity.startswith("unknown"):
            return CheckInDecision(False, REASON_UNNAMED)

        zone = str(event.get("zone") or "").strip()
        if not zone:
            return CheckInDecision(False, REASON_NO_ZONE, identity=identity)

        if self.zones and zone not in self.zones:
            return CheckInDecision(
                False, REASON_ZONE_NOT_ENABLED, identity=identity, zone=zone)

        raw_dwell = event.get("dwell_duration")
        dwell = float(raw_dwell) if isinstance(raw_dwell, (int, float)) else 0.0

        # The threshold is per-zone and may have been stretched by past declines.
        threshold = self.effective_threshold(identity, zone)

        def no(reason: str) -> CheckInDecision:
            return CheckInDecision(
                False, reason, identity=identity, zone=zone,
                dwell_duration=dwell, threshold=threshold)

        if dwell < threshold:
            return no(REASON_DWELL_TOO_SHORT)

        # Busy / mid-conversation / docked / flat. The shared helper — the same
        # rules that gate an unprompted greeting.
        blocked = interaction_blocked(status, min_battery=cfg.battery_floor)
        if blocked:
            return no(blocked)

        if in_quiet_hours(now, cfg.quiet_start, cfg.quiet_end):
            return no(REASON_QUIET_HOURS)

        last = self._last_interaction.get(identity)
        if last is not None and now - last < timedelta(seconds=cfg.global_cooldown):
            return no(REASON_GLOBAL_COOLDOWN)

        blocked_until = self._zone_blocked_until.get((identity, zone))
        if blocked_until is not None and now < blocked_until:
            return no(REASON_ZONE_COOLDOWN)

        return CheckInDecision(
            True, REASON_OK, identity=identity, zone=zone,
            dwell_duration=dwell, threshold=threshold)

    # ── recording what happened ──────────────────────────────────────────────

    def record_interaction(self, identity: str, when: datetime) -> None:
        """Note ANY interaction with this person — greeting, conversation, or a
        check-in. Starts the global cooldown.

        behavior_node calls this from the greeting path and at conversation end,
        which is what makes "OMNI said hello this morning" suppress an afternoon
        check-in.
        """
        identity = (identity or "").strip().lower()
        if not identity:
            return
        previous = self._last_interaction.get(identity)
        # Never let an out-of-order call rewind a cooldown.
        if previous is None or when > previous:
            self._last_interaction[identity] = when

    def record_outcome(
        self,
        identity: str,
        zone: str,
        outcome: str,
        when: datetime,
        *,
        dwell_duration: float = 0.0,
    ) -> OutcomeRecord:
        """Record how a check-in ended, applying the per-outcome cooldowns.

        Returns the record so the caller can log it through memory_client — the
        policy deliberately does not reach for a store itself (ROS-free rule),
        and that log is what a future learning pass reads back.
        """
        identity = (identity or "").strip().lower()
        zone = (zone or "").strip()
        if outcome not in ALL_OUTCOMES:
            raise ValueError(
                f"unknown outcome {outcome!r}; expected one of {ALL_OUTCOMES}")

        record = OutcomeRecord(
            identity=identity, zone=zone, outcome=outcome, when=when,
            dwell_duration=dwell_duration)
        self._history.append(record)

        # Every outcome is an interaction, including a refusal: OMNI spent this
        # person's attention either way.
        self.record_interaction(identity, when)

        cooldown = {
            OUTCOME_NO: self.config.no_cooldown,
            OUTCOME_NOT_NOW: self.config.not_now_cooldown,
            # Silence is read as "not now" — the softer cooldown — because being
            # ignored usually means bad timing, not a standing refusal. It is
            # still logged distinctly, so the two can be told apart later.
            OUTCOME_NO_RESPONSE: self.config.not_now_cooldown,
        }.get(outcome)

        if cooldown:
            until = when + timedelta(seconds=cooldown)
            key = (identity, zone)
            current = self._zone_blocked_until.get(key)
            if current is None or until > current:
                self._zone_blocked_until[key] = until

        return record

    # ── the v1.5 learning bias ───────────────────────────────────────────────

    def effective_threshold(self, identity: str, zone: str) -> float:
        """The dwell threshold to apply for this person in this zone.

        Learning manners rather than only following them: in a zone where the
        answer has usually been "no", OMNI waits longer before asking again. The
        multiplier is linear in the weighted decline rate and clamped to
        `bias_max_multiplier`, so the worst case is a threshold twice as long —
        never an unbounded retreat that silently disables the feature.

        Deliberately one-directional: a history of "yes" returns the multiplier
        to 1.0 but never below it. Enthusiasm is not a licence to interrupt more
        often than the configured floor.
        """
        cfg = self.config
        if not cfg.bias_enabled:
            return cfg.min_dwell

        relevant = [r for r in self._history
                    if r.identity == (identity or "").strip().lower()
                    and r.zone == (zone or "").strip()]
        if len(relevant) < cfg.bias_min_samples:
            return cfg.min_dwell

        decline_rate = (
            sum(_DECLINE_WEIGHT.get(r.outcome, 0.0) for r in relevant)
            / len(relevant)
        )
        multiplier = 1.0 + (cfg.bias_max_multiplier - 1.0) * decline_rate
        return cfg.min_dwell * multiplier

    # ── introspection (logging, tests, future persistence) ───────────────────

    def outcome_records(self) -> Sequence[OutcomeRecord]:
        """Every outcome recorded this process, oldest first."""
        return tuple(self._history)

    def zone_blocked_until(self, identity: str, zone: str) -> Optional[datetime]:
        return self._zone_blocked_until.get(
            ((identity or "").strip().lower(), (zone or "").strip()))

    def last_interaction(self, identity: str) -> Optional[datetime]:
        return self._last_interaction.get((identity or "").strip().lower())
