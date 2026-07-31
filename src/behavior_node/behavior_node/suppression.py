"""suppression.py — when OMNI must not start an unprompted interaction.

DELIBERATELY ROS-FREE, same rule as scene_describer / greeting_decider /
person_nav: this must import and run on a desktop with no ROS so the rules are
exercised offline. ROS glue belongs in behavior_node.py.

WHY THIS EXISTS AS A SHARED MODULE
----------------------------------
There are now two features that speak without being spoken to — the Session 2
unprompted greeting and the Session 9 proactive check-in — and they must agree on
what "busy" means. Two copies of that list would drift, and the drift would be
invisible: the failure mode is not a crash, it is OMNI cheerfully interrupting a
conversation six months from now because only one of the copies learned about a
new state.

So the rule set lives here once. `behavior_node._greeting_blocked()` and
`CheckInPolicy` both call `interaction_blocked()`; each adds only the gates that
are genuinely its own (greetings: the per-person cooldown; check-ins: dwell,
quiet hours, per-zone cooldowns).

THE PRINCIPLE, UNCHANGED FROM SESSION 2
---------------------------------------
**Suppression is enforced in code and is never mentioned to the model.** A
suppression rule expressed as prompt text is a suggestion; expressed as a
`return`, it is a rule. Everything here returns a reason string, and the caller
that gets a reason simply does not act.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Driving, exploring, docking, faulted. Interrupting any of these with an
# unprompted remark is at best noise and at worst masks something urgent.
BUSY_STATES = ("NAVIGATING", "EXPLORING", "DOCKING", "ERROR")

# A wake-word conversation is live. Someone walking in — or a dwell timer
# expiring — must not talk over the person already talking.
CONVERSATION_STATES = ("LISTENING", "SPEAKING")


@dataclass(frozen=True)
class RobotStatus:
    """Everything the suppression rules need to know about the robot right now.

    A plain snapshot so the rules stay pure and testable. behavior_node builds
    one of these from its own state under `_state_lock`; nothing here reaches
    back into ROS.

    ``docked`` is separate from ``state == 'DOCKING'``: DOCKING is the transient
    act of driving onto the charger, whereas docked is the persistent "sitting on
    the charger" condition. As of Session 9 the latter is only partially
    knowable — `bms_node` builds BatteryState field by field and never sets
    `power_supply_status` (verified in its source), so there is no charging
    signal to read. behavior_node passes what it can infer; when a real docked
    signal lands, it feeds this field and every consumer gets it at once.
    """

    state: str
    session_active: bool = False
    low_battery: bool = False
    battery_pct: Optional[float] = None
    docked: bool = False


def interaction_blocked(
    status: RobotStatus,
    *,
    min_battery: float = 0.0,
) -> Optional[str]:
    """Why an unprompted interaction must not start now, or None if it may.

    Ordered most-specific-first so the logged reason is the useful one: being
    told "robot is NAVIGATING" is more actionable than "battery 34%" when both
    happen to be true.
    """
    if status.state in BUSY_STATES:
        return f"robot is {status.state}"

    if status.state in CONVERSATION_STATES:
        return f"conversation in progress ({status.state})"

    # Belt and braces: a session can be open in a state the checks above would
    # wave through (a fault session that already recovered state, an /audio/say
    # announcement still playing). open_session() would reject us anyway — better
    # to not start the work at all.
    if status.session_active:
        return "a Gemini session is already active"

    # Sitting on the charger. A robot that undocks to offer help and then has to
    # announce it needs to charge is a joke that lands exactly once.
    if status.docked:
        return "robot is docked"

    if status.low_battery:
        return "battery is low"

    if (
        min_battery > 0.0
        and status.battery_pct is not None
        and status.battery_pct < min_battery
    ):
        return f"battery {status.battery_pct:.0f}% below {min_battery:.0f}%"

    return None
