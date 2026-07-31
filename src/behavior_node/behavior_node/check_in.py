"""check_in.py — the proactive check-in state machine (Session 9).

"Rafael has been at the workbench for an hour" -> drive over, stand beside him,
ask ONE question, and leave again if he does not want company.

    person_dwelling
          |
    CheckInPolicy.decide()        <- code-enforced manners. Nothing past here
          |  (approved)              was not already permitted.
      SNAPSHOT   remember where we are standing (TF map->base_link)
          |
      APPROACH   Nav2 goal: side-offset standoff beside them
          |
        ASK      one soft opener, via the Session 2 proactive-speech path
          |
      LISTEN     engaged? "no"? "not now"? silence?
          |
      RETURN     Nav2 goal back to the remembered pose
          |
       (done)    outcome recorded -> cooldowns applied, logged to memory

WHY A CODE STATE MACHINE AND NOT A PROMPT
-----------------------------------------
Gemini triggers nothing here. The policy decides whether to go, this class
decides where to stand and when to leave, and the model is handed exactly one
job: say the opening line and then converse normally. Everything with a
consequence — driving, leaving, the cooldown that follows a refusal — is a
`if`, not a suggestion. Same principle as greeting suppression.

THE SEAM WITH behavior_node
---------------------------
This class never builds a Nav2 goal or writes robot state itself. It calls
`node.start_navigation()` (the single goal builder, per behavior_node/CLAUDE.md)
and `node._speak_unprompted()` (the single unprompted-speech path), and it is
driven by four hooks the node calls into:

    on_dwell_event()      a person_dwelling arrived      (ROS executor thread)
    on_nav_result()       a Nav2 goal finished           (ROS executor thread)
    on_conversation_end() the Live session wrapped up    (any thread)
    tick()                periodic; timeouts and aborts  (ROS timer)
    abort()               wake word / e-stop / shutdown  (any thread)

THREADING
---------
The ROS executor is single-threaded, so anything that blocks runs on a daemon
thread — here that is `_start()`, because `nav_is_ready()` waits up to a second.
`_lock` guards the state field and the mission data; it is never held across a
call into the node, because `_set_state()` and `start_navigation()` take their
own locks and holding both invites a deadlock.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from typing import Optional

from std_msgs.msg import String

from omni_zones import standoff_pose

from .check_in_policy import (
    OUTCOME_NOT_NOW,
    OUTCOME_NO_RESPONSE,
    OUTCOME_YES,
    CheckInPolicy,
)
from .check_in_response import classify_reply

# Mission states. Distinct from the robot's own state machine — OMNI is in
# NAVIGATING during APPROACH and LISTENING during ASK/LISTEN; these track what
# the *check-in* is doing, not what the robot is doing.
INACTIVE = "inactive"
APPROACH = "approach"
ASK = "ask"
LISTEN = "listen"
RETURN = "return"

# Published on /omni/events for anything watching (chest LEDs, logs, dashboards).
EVENT_KIND = "check_in"


class CheckInBehavior:
    """One check-in mission at a time. Owned by behavior_node."""

    def __init__(self, node, policy: CheckInPolicy) -> None:
        self._node = node
        self._policy = policy
        self._lock = threading.Lock()

        self._state = INACTIVE
        self._identity = ""
        self._zone = ""
        self._dwell = 0.0
        self._return_pose: Optional[tuple] = None
        self._started = 0.0
        self._asked_at = 0.0
        self._outcome_done = False

        p = node.get_parameter
        self._standoff = float(p("check_in_standoff_distance").value)
        self._lateral = float(p("check_in_lateral_offset").value)
        self._default_side = str(p("check_in_default_side").value or "left").lower()
        self._silence_timeout = float(p("check_in_silence_timeout").value)
        self._max_duration = float(p("check_in_max_duration").value)

    # ── introspection ────────────────────────────────────────────────────────

    def is_active(self) -> bool:
        with self._lock:
            return self._state != INACTIVE

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def _log(self):
        return self._node.get_logger()

    # ── entry point ──────────────────────────────────────────────────────────

    def on_dwell_event(self, event: dict) -> None:
        """A person_dwelling event arrived. Decide, and go if permitted.

        Runs on the ROS executor thread, so the only work done inline is the
        (cheap, pure) policy decision. Everything that can block moves to a
        daemon thread.
        """
        decision = self._policy.decide(
            event, self._node.robot_status(), datetime.now())

        # ALWAYS log the reason, approved or not. The live verification step
        # ("trigger while docked / low battery / quiet hours -> check the policy
        # logs show the suppression reason") reads exactly this line.
        self._log().info(
            f"check-in: {decision.identity or '?'} dwelling in "
            f"{decision.zone or '?'} "
            f"({decision.dwell_duration / 60.0:.0f} min, threshold "
            f"{decision.threshold / 60.0:.0f} min) — "
            f"{'GO' if decision.approved else f'suppressed: {decision.reason}'}")

        if not decision.approved:
            return

        with self._lock:
            if self._state != INACTIVE:
                self._log().info("check-in: already in progress, ignoring")
                return
            self._state = APPROACH
            self._identity = decision.identity
            self._zone = decision.zone
            self._dwell = decision.dwell_duration
            self._started = time.monotonic()
            self._outcome_done = False
            self._return_pose = None

        threading.Thread(
            target=self._start, name="check-in", daemon=True).start()

    # ── SNAPSHOT + APPROACH ──────────────────────────────────────────────────

    def _start(self) -> None:
        """SNAPSHOT the current pose, then drive to a spot beside the person."""
        try:
            node = self._node

            # The policy already refuses to approve a check-in while docked, but
            # assert it here too: undocking to go and ask a question, then
            # needing to charge, is the one outcome that makes the whole feature
            # look broken.
            if getattr(node, "_docked", False):
                self._finish("docked after approval — standing down")
                return

            # SNAPSHOT. Nav2 cannot take us home without a pose to go home to, and
            # a robot with no pose cannot navigate at all — so this is a hard gate,
            # not a degradation. Slight staleness is fine: "roughly where it was"
            # is the whole requirement (session brief, "don't chase centimetres").
            pose = self._robot_pose()
            if pose is None:
                self._finish(
                    "no robot pose (is AMCL localised?) — not moving, because "
                    "there would be no way back")
                return

            person_xy = self._person_xy()
            if person_xy is None:
                self._finish(
                    f"world_state cannot place {self._identity} on the map")
                return

            if not node.nav_is_ready():
                self._finish("Nav2 is not available")
                return

            with self._lock:
                self._return_pose = pose
                zone = self._zone
                identity = self._identity

            goal = self._approach_goal(person_xy, pose, zone)
            self._publish("approach", detail=(
                f"driving to {goal[0]:.2f},{goal[1]:.2f} beside {identity} "
                f"in {zone}"))
            self._log().info(
                f"check-in: APPROACH {identity} in {zone} — "
                f"return_pose=({pose[0]:.2f},{pose[1]:.2f},{pose[2]:.0f}°) "
                f"goal=({goal[0]:.2f},{goal[1]:.2f},{goal[2]:.0f}°)")
            node.start_navigation(*goal)
        except Exception as exc:  # noqa: BLE001 - a check-in must never kill the node
            self._log().warn(f"check-in: start failed: {exc}")
            self._finish("start failed")

    def _approach_goal(self, person_xy, robot_pose, zone: str):
        """Where to stand: `standoff` short of them, offset to one side.

        The side is a fixed per-zone convention from zones.yaml, because nothing
        in the system estimates which way a person is facing (world_state has a
        face box, not a pose model). You sit the same way at a given bench every
        day, so a constant is the honest approximation — see omni_zones.Zone.
        """
        side = self._default_side
        zone_obj = self._node._zones.get(zone) if zone else None
        if zone_obj is not None and getattr(zone_obj, "check_in_side", None):
            side = zone_obj.check_in_side

        lateral = self._lateral if side == "left" else -self._lateral
        return standoff_pose(
            person_xy, (robot_pose[0], robot_pose[1]),
            standoff=self._standoff, lateral=lateral)

    # ── ASK ──────────────────────────────────────────────────────────────────

    def _ask(self) -> None:
        """One line, one question, then listen. Restraint is the whole charm."""
        with self._lock:
            self._state = ASK
            identity = self._identity
            zone = self._zone
            dwell_min = self._dwell / 60.0

        self._publish("ask", detail=f"asking {identity} about {zone}")

        # Deliberately prescriptive: this is the one place the model could turn a
        # polite enquiry into an interrogation, and an unprompted robot that
        # arrives and then MONOLOGUES is worse than one that never came.
        prompt = (
            f"You have just walked over to {identity.capitalize()}, who has been "
            f"working at the {zone} for about {dwell_min:.0f} minutes. You came "
            f"over on your own initiative to see what they are up to and whether "
            f"they want any help.\n\n"
            f"Say ONE short, warm, unhurried sentence that mentions they have "
            f"been at the {zone} a while and asks what they are working on. Then "
            f"STOP and wait for them to reply. Do not ask a second question, do "
            f"not list what you can do, do not explain why you came over. If they "
            f"say they are busy or would rather not talk, accept it gracefully in "
            f"a few words and do not press."
        )

        try:
            self._node._speak_unprompted(prompt, person=identity)
            with self._lock:
                self._state = LISTEN
                self._asked_at = time.monotonic()
            self._log().info(f"check-in: ASK -> LISTEN ({identity}, {zone})")
        except Exception as exc:  # noqa: BLE001
            self._log().warn(f"check-in: could not speak: {exc}")
            self._begin_return()

    # ── LISTEN / BRANCH ──────────────────────────────────────────────────────

    def tick(self) -> None:
        """Periodic: silence timeout, person-left-the-zone, and the hard ceiling.

        Called from a ROS timer on the executor thread. Cheap and non-blocking.
        """
        with self._lock:
            state = self._state
            started = self._started
            asked_at = self._asked_at

        if state == INACTIVE:
            return

        # A wedged mission must never leave OMNI parked in the middle of the room
        # believing it is still checking in.
        if self._max_duration > 0 and time.monotonic() - started > self._max_duration:
            self._log().warn(
                f"check-in: exceeded {self._max_duration:.0f}s — returning")
            self._record(OUTCOME_NO_RESPONSE)
            self._begin_return()
            return

        if state == APPROACH and not self._person_still_in_zone():
            # They got up mid-drive. The mission is moot: do not arrive at an
            # empty bench and ask it what it is working on.
            self._log().info(
                "check-in: person left the zone mid-approach — returning")
            self._publish("aborted", detail="person left the zone")
            self._begin_return()
            return

        if state == LISTEN and self._silence_timeout > 0:
            if self._node._bridge.user_speech():
                return   # they answered; the conversation flow owns it now
            if time.monotonic() - asked_at > self._silence_timeout:
                self._log().info(
                    f"check-in: no reply in {self._silence_timeout:.0f}s — "
                    f"treating as 'not now' and leaving quietly")
                self._record(OUTCOME_NO_RESPONSE)
                # Close the session ourselves rather than waiting out the full
                # conversation timeout: standing there with an open mic having
                # been ignored is exactly the loitering this must not do.
                try:
                    self._node._bridge.close_session()
                except Exception:  # noqa: BLE001
                    pass
                self._begin_return()

    def on_conversation_end(self) -> None:
        """The Live session ended. Classify what they said, then go home.

        MUST be called before behavior_node flushes the transcript to memory —
        `pop_transcript()` empties the buffer this reads.
        """
        with self._lock:
            if self._state not in (ASK, LISTEN):
                return

        reply = ""
        try:
            reply = self._node._bridge.user_speech()
        except Exception:  # noqa: BLE001
            pass

        if not reply.strip():
            outcome = OUTCOME_NO_RESPONSE
        else:
            outcome = classify_reply(reply)

        self._log().info(
            f"check-in: conversation ended — outcome={outcome} "
            f"(reply: {reply[:80]!r})")
        self._record(outcome)
        self._begin_return()

    # ── RETURN ───────────────────────────────────────────────────────────────

    def _begin_return(self) -> None:
        """Head back to where we were standing when this started."""
        with self._lock:
            if self._state == RETURN:
                return
            self._state = RETURN
            pose = self._return_pose

        if pose is None:
            self._finish("nowhere to return to")
            return

        self._publish("return", detail=f"returning to {pose[0]:.2f},{pose[1]:.2f}")
        self._log().info(
            f"check-in: RETURN to ({pose[0]:.2f},{pose[1]:.2f},{pose[2]:.0f}°)")
        try:
            if not self._node.nav_is_ready():
                self._finish("Nav2 unavailable for the return trip")
                return
            self._node.start_navigation(*pose)
        except Exception as exc:  # noqa: BLE001
            self._log().warn(f"check-in: return failed to dispatch: {exc}")
            self._dock_fallback()

    def _dock_fallback(self) -> None:
        """Returning failed. Never end a check-in stranded mid-room.

        NOTE: there is no docking routine to call yet — the DOCKING state is a
        placeholder in behavior_node ("no docking logic implemented yet") and
        `baro_floor_resolver`/AprilTag docking is not wired to a behaviour. So
        this currently logs loudly and stands down rather than pretending to
        dock. When the docking work lands, call it HERE — this is the spot, and
        the session brief's "if return_pose was the dock, re-dock properly"
        belongs here too.
        """
        self._log().warn(
            "check-in: could not return to the start pose, and there is no "
            "docking routine to fall back to yet — standing down where I am. "
            "Wire the dock call into CheckInBehavior._dock_fallback().")
        self._publish("stranded", detail="return failed, no dock routine")
        self._finish("return failed")

    # ── nav plumbing ─────────────────────────────────────────────────────────

    def on_nav_result(self, status: int) -> bool:
        """A Nav2 goal finished. Returns True if this check-in consumed it.

        behavior_node's own `_nav_result_callback` announces arrivals and opens a
        session; that is exactly wrong for a check-in, whose arrival is followed
        by a specific opener and whose return is followed by silence. So the node
        offers us the result first.
        """
        with self._lock:
            state = self._state
        if state == INACTIVE:
            return False

        succeeded = status == 4     # 4=SUCCEEDED, 5=CANCELED, 6=ABORTED

        if state == APPROACH:
            if succeeded:
                self._ask()
            else:
                # Could not get there. Say nothing — nobody asked for this visit,
                # so an apology would be the first they hear of it.
                self._log().info(
                    f"check-in: approach did not complete (status {status}) — "
                    f"returning without speaking")
                self._publish("aborted", detail=f"approach status {status}")
                self._begin_return()
            return True

        if state == RETURN:
            if succeeded:
                self._finish("returned")
            else:
                self._dock_fallback()
            return True

        # ASK / LISTEN: a stray nav result we did not initiate. Leave it alone.
        return False

    # ── abort + finish ───────────────────────────────────────────────────────

    def abort(self, reason: str) -> None:
        """Stop cleanly from any state — wake word, e-stop, or shutdown.

        Deliberately does NOT drive home: a wake word means the person is now
        talking to OMNI, and driving off mid-sentence to restore a pose is worse
        than simply staying put. The mission is over; where OMNI ends up is no
        longer the mission's business.
        """
        with self._lock:
            if self._state == INACTIVE:
                return
            state = self._state
            identity, zone = self._identity, self._zone

        self._log().info(f"check-in: aborted in {state} — {reason}")
        self._publish("aborted", detail=reason)

        # Stop driving. Aborting mid-APPROACH without this leaves OMNI still
        # motoring toward someone who is now already talking to it, and the
        # goal's eventual SUCCEEDED result reaches the generic arrival handler,
        # which announces the arrival over the top of the conversation.
        if state in (APPROACH, RETURN):
            try:
                if self._node.cancel_navigation():
                    self._log().info("check-in: cancelled the in-flight nav goal")
            except Exception as exc:  # noqa: BLE001
                self._log().warn(f"check-in: nav cancel failed: {exc}")

        # A wake word during the ask IS an answer — they engaged. Record it so
        # the global cooldown applies and OMNI does not come back in ten minutes.
        if state in (ASK, LISTEN):
            self._record(OUTCOME_YES)
        self._finish(reason)

    def _record(self, outcome: str) -> None:
        """Apply the cooldowns and log the outcome. Idempotent per mission."""
        with self._lock:
            if self._outcome_done:
                return
            self._outcome_done = True
            identity, zone, dwell = self._identity, self._zone, self._dwell

        try:
            record = self._policy.record_outcome(
                identity, zone, outcome, datetime.now(), dwell_duration=dwell)
        except ValueError as exc:
            self._log().warn(f"check-in: bad outcome {outcome!r}: {exc}")
            return

        blocked = self._policy.zone_blocked_until(identity, zone)
        self._log().info(
            f"check-in: outcome {outcome} for {identity} at {zone} — "
            f"zone clear again at {blocked.strftime('%H:%M') if blocked else 'now'}, "
            f"next check-in for {identity} no earlier than "
            f"{(self._policy.last_interaction(identity)).strftime('%H:%M')} + "
            f"{self._policy.config.global_cooldown / 3600.0:.1f}h")
        self._publish("outcome", detail=outcome)

        # Log it to memory so the v1.5 learning has data to read back later, and
        # so "did OMNI already ask me today" is answerable by the model.
        try:
            if getattr(self._node, "_memory_enabled", False):
                self._node._memory.store_transcript(
                    record.as_memory_text(),
                    person=identity,
                    source="check_in",
                )
        except Exception as exc:  # noqa: BLE001 - memory is a soft dependency
            self._log().warn(f"check-in: could not log outcome to memory: {exc}")

    def _finish(self, reason: str) -> None:
        with self._lock:
            was = self._state
            self._state = INACTIVE
            self._return_pose = None
        if was != INACTIVE:
            self._log().info(f"check-in: done ({reason})")
            self._publish("done", detail=reason)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _robot_pose(self):
        """(x, y, yaw_deg) in the map frame, or None if TF cannot answer."""
        try:
            return self._node._fn.robot_pose()
        except Exception:  # noqa: BLE001
            return None

    def _person_row(self):
        snapshot = self._node.latest_world_state()
        if not isinstance(snapshot, dict):
            return None
        with self._lock:
            identity = self._identity
        for row in snapshot.get("people", []) or []:
            if (row.get("identity") or "").lower() == identity:
                return row
        return None

    def _person_xy(self):
        row = self._person_row()
        if not row:
            return None
        xy = row.get("map_xy")
        if isinstance(xy, (list, tuple)) and len(xy) >= 2:
            return (float(xy[0]), float(xy[1]))
        # No point estimate — fall back to the zone's anchor, which still lands
        # in the right room (the success criterion for go_to_person too).
        with self._lock:
            zone = self._zone
        anchor = self._node._zones.nav_pose(zone) if zone else None
        return (anchor[0], anchor[1]) if anchor else None

    def _person_still_in_zone(self) -> bool:
        """True unless world_state clearly says they have left.

        Biased toward continuing: world_state is a SOFT dependency, and a missing
        snapshot must not abort a drive that is otherwise fine. Only an explicit,
        different zone counts as "they got up".
        """
        row = self._person_row()
        if not row:
            return True
        zone = row.get("zone")
        if not zone:
            return True
        with self._lock:
            return zone == self._zone

    def _publish(self, phase: str, *, detail: str = "") -> None:
        """Announce a transition on /omni/events, alongside the presence events.

        Best-effort: a publish failure must never break the mission.
        """
        try:
            with self._lock:
                identity, zone = self._identity, self._zone
            payload = {
                "kind": EVENT_KIND,
                "phase": phase,
                "identity": identity,
                "zone": zone,
                "detail": detail,
                "timestamp": round(time.time(), 3),
            }
            self._node._events_pub.publish(String(data=json.dumps(payload)))
        except Exception:  # noqa: BLE001
            pass
