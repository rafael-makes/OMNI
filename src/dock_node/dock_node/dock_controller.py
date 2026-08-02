"""
dock_controller.py — ROS-free back-in docking control law for OMNI.

Pure logic so it can be unit-tested with the robot OFF (test_dock_controller.py),
like person_nav / check_in_policy in behavior_node. The ROS wrapper (dock_node.py)
feeds it normalized observations each tick and applies the returned velocity command.

OMNI backs onto a wall-mounted AprilTag (id 0) seen by the REAR camera:
  SEARCH   no tag in view — PULSE-rotate to sweep for it
  ALIGN    tag in view — PULSE-rotate until it's centred (back pointed at dock)
  REVERSE  back up straight; if heading drifts, stop and PULSE-correct; rear ToF stops it
  DONE     rear ToF within stop_range — docked
  FAILED   a timeout, or the tag lost with nothing near behind

WHY PULSED ROTATION (measured 2026-07-26, see feedback_drivetrain_speed_floor):
the differential drivetrain physically STALLS below ~1.1 rad/s in-place rotation
(Pico MIN_PWM=50 = the motors' true stall point; firmware can't lower it). So a fine
heading can't be servoed continuously — it would overshoot and hunt. Instead we rotate
in short BURSTS at pulse_speed (above the floor, so the motors actually move), then STOP
and look, then decide again. ~pulse_dur of motion per pulse ≈ a few degrees. REVERSE is
the one continuous move: reverse_speed is above the ~0.16 m/s translation floor, and the
rear-ToF stop overshoot stays above safety_node's 0.075 m fault.

Metric distance is used ONLY for the rear ToF (a real sensor). The tag is used in PIXEL
space (horizontal error, skew) — no camera calibration needed.

Sign convention: the rear camera may be mirrored, so which way to yaw for a given pixel
error isn't knowable a priori. steer_sign (+1/-1) captures it, calibrated live at low
speed with a hand on the e-stop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Phase(str, Enum):
    IDLE = "IDLE"
    ORIENT = "ORIENT"   # pulse-rotate to a known heading so the tag is in view
    SEARCH = "SEARCH"
    ALIGN = "ALIGN"
    REVERSE = "REVERSE"
    DONE = "DONE"
    FAILED = "FAILED"


def _wrap(a: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


@dataclass
class TagObs:
    """One rear-camera observation of the dock tag, in normalized pixel space."""
    seen: bool = False
    ex: float = 0.0         # horizontal error (cx - w/2)/(w/2); + = tag right of centre
    ey: float = 0.0         # vertical error (telemetry only)
    size_frac: float = 0.0  # tag width / image width — a rough closeness cue
    skew: float = 0.0       # (right_edge_h - left_edge_h)/mean — perspective squareness


@dataclass
class RearRange:
    """Latest rear ToF ranges in metres, or None if stale / out of range."""
    left: Optional[float] = None
    right: Optional[float] = None

    def valid_min(self) -> Optional[float]:
        vals = [v for v in (self.left, self.right) if v is not None]
        return min(vals) if vals else None

    def delta(self) -> Optional[float]:
        """left - right; + means the left side is further (angled left-back)."""
        if self.left is None or self.right is None:
            return None
        return self.left - self.right


@dataclass
class DockConfig:
    # tolerances (ex is normalized pixel error in [-1, 1])
    align_tol: float = 0.15          # |ex| within this = aligned enough to reverse
    reverse_correct_tol: float = 0.30  # |ex| beyond this during REVERSE → stop & pulse-correct
    # pulsed rotation (bang-bang — the drivetrain can't rotate slowly, see module docstring)
    pulse_speed: float = 1.2         # rad/s per pulse; MUST exceed the ~1.1 rad/s rotation floor
    pulse_dur: float = 0.12          # s of rotation per pulse (a few degrees)
    settle_dur: float = 0.25         # s stopped between pulses — lets motion settle + a fresh frame
    # translation
    reverse_speed: float = 0.18      # m/s; above the ~0.16 m/s translation floor
    # ranges (m)
    stop_range: float = 0.13         # rear ToF stop — ~2.5cm margin above safety's 0.075 after overshoot
    # final square-up: near the wall, pulse-rotate to equalise the two rear ToF (parallel to wall)
    square_engage_range: float = 0.25  # begin squaring when the nearest rear ToF is within this
    square_tol: float = 0.02         # |left-right| below this (m) = square enough
    square_sign: float = 1.0         # +1/-1; ToF-delta→yaw sign, calibrated live (may differ from steer_sign)
    # ORIENT: pulse-rotate to a caller-supplied heading BEFORE searching, so the rear
    # camera is pointed at the tag on arrival (Nav2 delivers OMNI to the standoff
    # position at an arbitrary heading — see project_dock_node). Ends early the moment
    # the tag comes into view, so orient_tol is only the fallback endpoint.
    orient_tol: float = 0.12         # rad (~7°) — |heading - target| within this = done
    t_orient_max: float = 20.0
    # timing (s) — generous because pulsing is slow
    tag_lost_grace: float = 0.6      # tolerate this long a tag dropout before reacting
    t_search_max: float = 20.0
    t_align_max: float = 30.0
    t_reverse_max: float = 45.0
    t_overall_max: float = 120.0
    # geometry
    steer_sign: float = 1.0          # +1/-1, calibrated live for the rear camera


@dataclass
class DockCommand:
    linear_x: float = 0.0
    angular_z: float = 0.0
    phase: Phase = Phase.IDLE
    done: bool = False
    failed: bool = False
    message: str = ""


def _sign(x: float) -> float:
    return 1.0 if x >= 0.0 else -1.0


class DockController:
    def __init__(self, cfg: DockConfig):
        self.cfg = cfg
        self.phase = Phase.IDLE
        self._t_start = 0.0
        self._t_phase = 0.0
        self._t_last_seen = 0.0
        # pulsed-rotation sub-state
        self._pulse_mode = 'idle'   # 'idle' | 'pulsing' | 'settling'
        self._pulse_t = 0.0
        self._pulse_dir = 1.0
        # SEARCH sweep state — see _search()
        self._search_dir = 1.0
        self._search_in_leg = 0
        self._search_leg_len = 2
        # ORIENT target heading (rad, in the same frame the caller feeds as `heading`)
        self._orient_target: Optional[float] = None

    def start(self, now: float, tag: Optional[TagObs] = None,
              orient_target: Optional[float] = None,
              heading: Optional[float] = None) -> None:
        """Begin a docking attempt.

        orient_target/heading (both required to orient): pulse-rotate to the
        orient_target heading FIRST, so the rear camera faces the tag before
        searching. Without them, behaves exactly as before (ALIGN if the tag is
        already seen, else SEARCH).
        """
        self._t_start = now
        self._pulse_reset()
        self._orient_target = orient_target
        # If we have a target heading, ORIENT — even if `heading` is None at kickoff
        # (transient TF gap). The phase's own tick tolerates a brief gap and only falls
        # to SEARCH after t_orient_max. Falling straight to blind SEARCH when the caller
        # said which way to turn is exactly the failure mode observed 2026-08-02.
        if orient_target is not None:
            self._enter(Phase.ORIENT, now)
        elif tag and tag.seen:
            self._t_last_seen = now
            self._enter(Phase.ALIGN, now)
        else:
            self._enter(Phase.SEARCH, now)

    def cancel(self) -> None:
        self.phase = Phase.IDLE
        self._pulse_reset()

    def _enter(self, phase: Phase, now: float) -> None:
        self.phase = phase
        self._t_phase = now
        if phase is Phase.SEARCH:
            self._search_reset()

    def _search_reset(self) -> None:
        """Restart the sweep centred on wherever we currently point."""
        self._search_dir = 1.0
        self._search_in_leg = 0
        self._search_leg_len = 2

    # ── pulsed rotation ─────────────────────────────────────────────────────────
    # A pulse: rotate at pulse_speed for pulse_dur, then hold still for settle_dur,
    # then go idle so the phase can read a fresh (settled) ex and decide again.
    def _pulse_reset(self) -> None:
        self._pulse_mode = 'idle'

    def _pulse_active(self) -> bool:
        return self._pulse_mode != 'idle'

    def _pulse_start(self, direction: float, now: float) -> float:
        self._pulse_mode = 'pulsing'
        self._pulse_t = now
        self._pulse_dir = _sign(direction)
        return self._pulse_dir * self.cfg.pulse_speed

    def _pulse_continue(self, now: float) -> float:
        cfg = self.cfg
        if self._pulse_mode == 'pulsing':
            if now - self._pulse_t < cfg.pulse_dur:
                return self._pulse_dir * cfg.pulse_speed
            self._pulse_mode = 'settling'
            self._pulse_t = now
            return 0.0
        # settling
        if now - self._pulse_t < cfg.settle_dur:
            return 0.0
        self._pulse_mode = 'idle'
        return 0.0

    # ── main update ───────────────────────────────────────────────────────────��─
    def update(self, tag: TagObs, rear: RearRange, now: float,
               heading: Optional[float] = None) -> DockCommand:
        cfg = self.cfg
        if self.phase in (Phase.IDLE, Phase.DONE, Phase.FAILED):
            return DockCommand(phase=self.phase,
                               done=self.phase is Phase.DONE,
                               failed=self.phase is Phase.FAILED)

        if tag.seen:
            self._t_last_seen = now
        seen_recent = (now - self._t_last_seen) <= cfg.tag_lost_grace

        if now - self._t_start > cfg.t_overall_max:
            return self._fail(now, "overall docking timeout")

        if self.phase is Phase.ORIENT:
            return self._orient(tag, heading, now)
        if self.phase is Phase.SEARCH:
            return self._search(tag, now)
        if self.phase is Phase.ALIGN:
            return self._align(tag, seen_recent, now)
        if self.phase is Phase.REVERSE:
            return self._reverse(tag, seen_recent, rear, now)
        return self._fail(now, f"unexpected phase {self.phase}")

    # ── phases ────────────────────────────────────────────────────────────────
    def _orient(self, tag: TagObs, heading: Optional[float], now: float) -> DockCommand:
        cfg = self.cfg
        # Tag came into view mid-turn — hand straight to ALIGN, don't overshoot.
        if tag.seen:
            self._pulse_reset()
            self._enter(Phase.ALIGN, now)
            return DockCommand(phase=Phase.ALIGN, message="tag acquired during orient")
        if now - self._t_phase > cfg.t_orient_max:
            self._pulse_reset()
            self._enter(Phase.SEARCH, now)
            return DockCommand(phase=Phase.SEARCH, message="orient timeout — searching")
        if self._orient_target is None:
            # No target — can't orient. (Shouldn't happen: start() only enters ORIENT
            # when orient_target is set.)
            self._pulse_reset()
            self._enter(Phase.SEARCH, now)
            return DockCommand(phase=Phase.SEARCH, message="no orient target — searching")
        if heading is None:
            # TF gap — hold still (no pulse) and try again next tick. t_orient_max
            # above catches a permanent gap and falls through to SEARCH honestly.
            self._pulse_reset()
            return DockCommand(linear_x=0.0, angular_z=0.0, phase=Phase.ORIENT,
                               message="orient waiting for TF")
        err = _wrap(self._orient_target - heading)
        # finish an in-progress pulse before re-deciding (decisions only when settled)
        if self._pulse_active():
            return DockCommand(angular_z=self._pulse_continue(now), phase=Phase.ORIENT,
                               message=f"orient err={math.degrees(err):+.0f}deg")
        if abs(err) <= cfg.orient_tol:
            # Reached the target heading but the tag still isn't visible — search from
            # here (rare: the standoff heading should show the tag).
            self._pulse_reset()
            self._enter(Phase.SEARCH, now)
            return DockCommand(phase=Phase.SEARCH, message="oriented, no tag — searching")
        # +err (target CCW of current heading) → +yaw (CCW). This is a robot/odom-frame
        # rotation, so there is NO steer_sign here — that sign is for camera-PIXEL error.
        yaw = self._pulse_start(_sign(err), now)
        return DockCommand(angular_z=yaw, phase=Phase.ORIENT,
                           message=f"orient err={math.degrees(err):+.0f}deg")

    def _search(self, tag: TagObs, now: float) -> DockCommand:
        cfg = self.cfg
        if tag.seen:
            self._pulse_reset()
            self._enter(Phase.ALIGN, now)
            return DockCommand(phase=Phase.ALIGN, message="tag acquired")
        if now - self._t_phase > cfg.t_search_max:
            return self._fail(now, "tag not found during search")
        if self._pulse_active():
            return DockCommand(angular_z=self._pulse_continue(now),
                               phase=Phase.SEARCH, message="searching (pulse)")
        # Settled, still no tag -> pulse again as part of a BACK-AND-FORTH sweep that
        # widens each leg: 2 pulses one way, 4 back, 6 the first way, ... so the arc
        # stays centred on where the search began and grows outwards.
        #
        # This used to pulse `steer_sign` EVERY time, i.e. sweep one direction only.
        # If the tag happened to be the other way, OMNI rotated further and further
        # from it until t_search_max expired — observed live 2026-07-30, it turned
        # left away from the tag and failed. A one-way sweep can only ever find a tag
        # it is already turning toward.
        self._search_in_leg += 1
        if self._search_in_leg >= self._search_leg_len:
            self._search_in_leg = 0
            self._search_leg_len += 2
            self._search_dir = -self._search_dir
        yaw = self._pulse_start(self._search_dir * cfg.steer_sign, now)
        return DockCommand(angular_z=yaw, phase=Phase.SEARCH,
                           message=f"searching (sweep {'+' if self._search_dir > 0 else '-'})")

    def _align(self, tag: TagObs, seen_recent: bool, now: float) -> DockCommand:
        cfg = self.cfg
        if not seen_recent:
            self._pulse_reset()
            self._enter(Phase.SEARCH, now)
            return DockCommand(phase=Phase.SEARCH, message="tag lost — re-searching")
        if now - self._t_phase > cfg.t_align_max:
            return self._fail(now, "align timeout")
        # finish an in-progress pulse before re-deciding (decisions only when settled)
        if self._pulse_active():
            return DockCommand(angular_z=self._pulse_continue(now),
                               phase=Phase.ALIGN, message=f"align pulse ex={tag.ex:+.2f}")
        if abs(tag.ex) <= cfg.align_tol:
            self._pulse_reset()
            self._enter(Phase.REVERSE, now)
            return DockCommand(phase=Phase.REVERSE, message="aligned — reversing")
        direction = -_sign(tag.ex) * cfg.steer_sign
        yaw = self._pulse_start(direction, now)
        return DockCommand(angular_z=yaw, phase=Phase.ALIGN, message=f"align pulse ex={tag.ex:+.2f}")

    def _reverse(self, tag: TagObs, seen_recent: bool,
                 rear: RearRange, now: float) -> DockCommand:
        cfg = self.cfg
        rmin = rear.valid_min()
        # physical stop — the whole point of the rear ToF (checked every tick, even mid-pulse)
        if rmin is not None and rmin <= cfg.stop_range:
            self._pulse_reset()
            return self._done(now, f"docked (rear {rmin:.3f} m)")
        if now - self._t_phase > cfg.t_reverse_max:
            return self._fail(now, "reverse timeout")

        # a correction/square pulse in progress: keep rotating in place (no translation)
        if self._pulse_active():
            yaw = self._pulse_continue(now)
            return DockCommand(linear_x=0.0, angular_z=yaw, phase=Phase.REVERSE,
                               message=f"correcting rear={_fmt(rmin)}")

        delta = rear.delta()
        near_wall = rmin is not None and rmin <= cfg.square_engage_range

        # FINAL APPROACH — square to the wall on the rear ToF delta (heading now comes from
        # the wall geometry, not the tag; robust even if the tag drops out this close in).
        if near_wall:
            if delta is not None and abs(delta) > cfg.square_tol:
                direction = _sign(delta) * cfg.square_sign
                yaw = self._pulse_start(direction, now)
                return DockCommand(linear_x=0.0, angular_z=yaw, phase=Phase.REVERSE,
                                   message=f"square-up d={delta:+.3f} rear={_fmt(rmin)}")
            return DockCommand(linear_x=-cfg.reverse_speed, angular_z=0.0, phase=Phase.REVERSE,
                               message=f"reversing square rear={_fmt(rmin)}")

        # FAR FROM WALL — hold heading on the tag, else back straight
        if seen_recent and tag.seen:
            if abs(tag.ex) > cfg.reverse_correct_tol:
                direction = -_sign(tag.ex) * cfg.steer_sign
                yaw = self._pulse_start(direction, now)
                return DockCommand(linear_x=0.0, angular_z=yaw, phase=Phase.REVERSE,
                                   message=f"reverse-correct ex={tag.ex:+.2f}")
            return DockCommand(linear_x=-cfg.reverse_speed, angular_z=0.0, phase=Phase.REVERSE,
                               message=f"reversing ex={tag.ex:+.2f} rear={_fmt(rmin)}")

        # tag lost while still far from the wall — nothing to steer by
        return self._fail(now, "lost tag mid-approach with nothing near behind")

    # ── terminal ──────────────────────────────────────────────────────────────
    def _done(self, now: float, msg: str) -> DockCommand:
        self._enter(Phase.DONE, now)
        return DockCommand(phase=Phase.DONE, done=True, message=msg)

    def _fail(self, now: float, msg: str) -> DockCommand:
        self._pulse_reset()
        self._enter(Phase.FAILED, now)
        return DockCommand(phase=Phase.FAILED, failed=True, message=msg)


def _fmt(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.3f}"
