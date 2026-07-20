"""
head_tracking_node.py — Pan/tilt head tracking for OMNI.

Drives head_pan + head_tilt servos via /servo_commands to keep the tracked
subject centered in the camera frame. Two detection sources (both from the
Jetson head_detector_node), with faces preferred:

  * /camera/faces      (class_id='face')   — YuNet face boxes. When a face is
    fresh the head centers DIRECTLY on the face box center (accurate gaze).
  * /camera/detections (class_id='person') — YOLO person boxes. Fallback when
    no face is visible (person far/turned away); aims at the head region of
    the person box (head_target_frac) as before.

This face-preferred/person-fallback design keeps close-range gaze locked on the
real face while still tracking a person across the room. Faces are the near-term
target for the persistent per-person memory work; /camera/detections stays the
general world feed the semantic-mapping layer will consume.

P controller (independent axes, normalized servo position):
  pan_error  = detection_cx / image_width  - 0.5   (+ve = person right of center)
  tilt_error = detection_cy / image_height - 0.5   (+ve = person below center)
  servo_norm += gain * error                         servo position in [0.0, 1.0]

Internal position is normalized [0.0, 1.0] where angle = norm * 180°.
Each axis has its own measured travel envelope and resting/neutral angle
(pan 52–128°@90°, tilt 76–92°@84°) — see project memory head-servo-directions.
These are kept inside the mechanical end-stops (pan 50–130, tilt 74–94) so the
P-controller can never drive a servo into a hard stop.

Subject tracking is active in IDLE, LISTENING, and SPEAKING (tracking_states).
After return_to_neutral_timeout seconds without a person the head returns to
neutral via proportional decay (smooth, not instant).

NAVIGATING and EXPLORING (gaze_states) used to snap the head to neutral — clean,
but exactly the robotic tell this node exists to avoid. They now run GAZE
CHOREOGRAPHY instead (see the big block below). DOCKING and ERROR keep the
neutral-snap unchanged: docking depends on the rear camera being unobstructed and
on predictable, still kinematics for AprilTag geometry, and ERROR wants a
visibly halted head. Those two are deliberately sacred — do not add gaze to them.

Two liveliness behaviours sit UNDERNEATH tracking — they only ever run on ticks
where the tracker has nothing to track, so the face/person control path above is
untouched:

  * IDLE DRIFT — once the head has settled at neutral with nobody visible, it
    makes slow randomized looks around the envelope instead of sitting frozen.
    Dwell times are drawn from a lognormal, not a fixed interval: a constant
    period reads as mechanical within about half a minute, which is the whole
    thing this is trying to avoid. Restricted to drift_states (IDLE/LISTENING)
    — a drifting head during SPEAKING would look inattentive.
  * INSTANT ORIENT — a person_appeared event on /omni/events snaps the head
    toward that camera's direction immediately, without waiting for the agent
    layer to respond. Handled in the subscription callback (not the 10 Hz tick)
    so motion begins on the next 50 Hz slew, ~20 ms after the event.

Orient is a fallback, not an override: the moment a real face or person box is
fresh, tracking wins and the orient pose is abandoned mid-slew. That is correct
— orienting exists only to cover the window before the subject is visible.

Rear-camera caveat: pan tops out at the mechanical envelope (~128°), which does
not fully face rear. A partial turn toward the direction is the intended result.

Upstream caveat, true as of 2026-07-20: world_state's identity source is
head-only ('/camera/identities=head') and the rear camera publishes no
identities topic yet, so every /omni/events message today carries camera="head".
The rear branch here is live code on a dormant path — it starts working the day
that topic exists, with no change to this node.

GAZE CHOREOGRAPHY (NAVIGATING / EXPLORING)
==========================================
Replaces the old neutral-snap on those two states. Four movements:

  1. GLANCE-BEFORE-MOVE (NAVIGATING only). On entering NAVIGATING the head
     claims itself, waits for the first Nav2 global plan (/plan, map frame),
     transforms its endpoint into base_link via TF to get the goal BEARING, and
     slews there — clamped to the pan envelope, so a goal behind the robot gets a
     partial turn (same rear caveat as orient). It HOLDS glance_hold seconds
     (sampled in [glance_hold_min, glance_hold_max], not fixed — a constant reads
     as mechanical) before travel gaze begins.

     THE GATING DECISION — why the hold is not enforced on the wheels.
     The prompt framed a choice: gate /cmd_vel until the glance completes, or
     accept near-simultaneity and rely on controller startup. This node takes
     NEAR-SIMULTANEITY, deliberately:
       * A cmd_vel gate would put head choreography in the safety-critical
         velocity path — the path safety_node and motor_control_node own. Coupling
         a cosmetic head movement to whether the wheels are allowed to turn is an
         unacceptable failure surface for zero locomotion benefit. head_tracking
         must never be able to stop the robot from moving.
       * The lead is real without a gate. The head begins slewing ~20 ms after
         the goal bearing is known (next 50 Hz slew tick), while Nav2 still owes
         its global-plan compute, its first 5 Hz controller cycle, and the base's
         own acceleration ramp off zero. The head is both computationally and
         mechanically faster to start, so it visibly leads even though nothing
         blocks the wheels. The hold overlaps that startup latency rather than
         extending it.
       * If the base does start rolling mid-hold, nothing breaks: travel gaze
         (2) takes over smoothly from wherever the head is. The hold is an
         intent-reading dwell, not a hard barrier.
     Trigger note: the authoritative "new goal" edge is the /robot_state →
     NAVIGATING transition (behavior_node sets it the instant it dispatches the
     NavigateToPose goal, before any motion). We do NOT subscribe to a "goal
     topic": nav_node/send_goal and behavior_node both send goals via the
     navigate_to_pose ACTION, not a topic, so there is no goal topic to subscribe
     to. Reading the goal from /plan instead needs no change to the sender (this
     project has shipped bugs from one-sided wire-format changes) and also covers
     goals sent by RViz or the debug script, not just the one dispatcher we edit.

  2. TRAVEL GAZE (NAVIGATING after the hold, and EXPLORING throughout). Each tick
     takes a look-ahead point travel_lookahead_m along DWB's local plan
     (/local_plan, odom frame), transforms it to base_link, and aims the head at
     its bearing. A look-ahead point does the right thing for free: on a straight
     stretch it sits dead ahead (bearing ~0 -> head centers), into a curve it
     swings to the inside (head leads the turn). A Schmitt trigger
     (travel_engage_deg / travel_release_deg) stops it dithering around center.
     Tilt is held level throughout — the tilt envelope is only ~11 deg and a
     travelling robot looks where it is going, not up or down.

  3. FACE OVERRIDE. If a person is in view (a fresh /camera/faces or
     /camera/detections sample — the same signal world_state is built from, at
     10 Hz rather than the 1 Hz /omni/world_state JSON, and it doubles as the aim
     point) AND the robot is moving slower than face_override_speed, face
     tracking overrides travel gaze: a robot creeping up to someone should look
     at them, not at the corridor. Speed hysteresis (face_override_speed +
     face_override_speed_hyst to release) plus the existing face_timeout keep it
     from fighting travel gaze at the threshold. Not applied during the initial
     glance — that movement is about the destination.

  4. ARRIVAL SETTLE. Leaving a gaze state for a tracking state (arrival:
     NAVIGATING -> LISTENING/IDLE) runs a short scripted settle — glance one way,
     the other, rest at neutral — then hands back to normal IDLE behaviour
     (tracking / drift). Like drift and orient it sits UNDER tracking: a real
     subject appearing mid-settle cancels it and tracking wins. Leaving a gaze
     state for DOCKING or ERROR skips the settle and snaps to neutral (sacred).

All four transform plan geometry through TF with a ZERO lookup timeout: a
non-zero timeout would block the single-threaded executor (see project memory
ekf-transform-timeout). A missing transform just means "hold" for that tick, not
a stall. Every threshold, rate and topic is a parameter; the defaults are
starting points for filming A/B against Session 5, not measured values.
"""

import json
import math
import random
import time

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.qos import QoSPresetProfiles
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import Float32MultiArray, String
from vision_msgs.msg import Detection2DArray
from nav_msgs.msg import Path, Odometry
from tf2_ros import Buffer, TransformListener, TransformException

# Don't apply a correction if |error| is below this normalized threshold.
# Prevents servo chatter when the person is already near center.
_DEAD_ZONE = 0.12

# Proportional gain for neutral return (per 10 Hz tick).
# 0.1 → closes 10% of remaining distance each tick; smooth exponential decay.
_NEUTRAL_RETURN_RATE = 0.1

# Once within this distance of the axis neutral, snap exactly to it to stop creeping.
_NEUTRAL_SNAP = 0.01

# Output smoothing (anti-notch): the 10 Hz control law only sets a TARGET; a faster
# slew timer moves the actually-published position toward that target so the servos
# glide instead of snapping in 10 Hz steps. slew_alpha (fraction of the remaining gap
# closed per slew tick) is a live-tunable param; _SLEW_HZ is the slew rate.
_SLEW_HZ = 50.0
_SLEW_SNAP = 0.0015

# Idle drift: how close (normalized) counts as "arrived" at a drift target.
_DRIFT_ARRIVE = 0.002

# Tries allowed when rejecting a drift target for being too close to the current
# pose. Bounded so a narrow envelope (tilt is only ~11° wide) can't spin here —
# after the last try we take whatever was sampled.
_DRIFT_TARGET_TRIES = 8

# Event kinds on /omni/events. Mirrors event_generator.models; not imported from
# there because that package is a soft dependency this node must not require.
_EVENT_PERSON_APPEARED = 'person_appeared'

# Gaze choreography phases, held in self._gaze_phase while in a gaze_state.
_GAZE_GLANCE = 'GLANCE'   # NAVIGATING: holding the look toward the goal bearing
_GAZE_TRAVEL = 'TRAVEL'   # travel gaze from the local plan (+ face override)

# A stored plan (global or local) older than this is treated as absent — the
# head holds rather than aiming at a stale path the robot has already left.
_PLAN_STALE_S = 1.0

# An odometry sample older than this makes the speed unknown; face override then
# falls back to "assume moving" (travel gaze), never silently sticks on a face.
_ODOM_STALE_S = 1.0


def _deg_to_norm(deg: float) -> float:
    """servo_node maps 0–180° onto a servo's pulse range; we track [0,1] internally."""
    return deg / 180.0


def _approach(current: float, goal: float, max_move: float) -> float:
    """Move current toward goal by at most max_move. Never overshoots."""
    delta = goal - current
    if abs(delta) <= max_move:
        return goal
    return current + math.copysign(max_move, delta)


class HeadTrackingNode(Node):

    def __init__(self):
        super().__init__('head_tracking_node')

        # ── Parameters ──────────────────────────────────────────────────────
        # Gain signs from measured direction. Pan: person right -> pan_gain -ve.
        # TILT REVERSED 2026-07-09 — single servo on ch6 with the servo reverser
        # removed: higher servo angle = head UP, so tilt_gain is NEGATIVE. Speeds
        # pan_gain was -0.18, tuned live BEFORE head tracking ran in the full stack.
        # At that gain a half-frame error (0.5) gives 0.18*0.5 = 0.09, which saturates
        # max_step — the head slews ~16°/tick, 160°/s, on feedback already stale by the
        # Jetson's inference + network latency. It overshot. Halved 2026-07-20; this is
        # a STARTING POINT for live tuning, not a measured value:
        #   ros2 param set /head_tracking_node pan_gain -0.09
        self.declare_parameter('pan_gain',                  -0.09)
        self.declare_parameter('tilt_gain',                 -0.05)
        self.declare_parameter('max_step',                  0.09)
        # Aim point: fraction of the person-box HEIGHT measured DOWN from the TOP of
        # the box. 0.15 ~ forehead/face, so the head looks at the face (not the torso
        # center) and tilt stops saturating downward. 0.5 = old bbox-center behavior.
        self.declare_parameter('head_target_frac',          0.2)
        # Output smoothing: fraction of the remaining gap the head glides each 50 Hz
        # slew tick. Lower = smoother but a touch laggier; higher = snappier/notchier.
        self.declare_parameter('slew_alpha',                0.4)
        self.declare_parameter('detection_confidence_min',  0.7)
        # Face tracking (preferred source). prefer_faces gates whether a fresh
        # face overrides the person box. face_confidence_min filters YuNet faces.
        # face_timeout: how long a face stays "fresh" before we fall back to the
        # person box — short, so losing the face for a moment doesn't stall.
        self.declare_parameter('prefer_faces',              True)
        self.declare_parameter('face_confidence_min',       0.6)
        self.declare_parameter('face_timeout',              0.6)
        self.declare_parameter('tracking_states',           'IDLE,LISTENING,SPEAKING')
        self.declare_parameter('return_to_neutral_timeout', 2.0)
        self.declare_parameter('image_width',               1280)
        self.declare_parameter('image_height',              720)
        self.declare_parameter('pan_board',                 0)
        self.declare_parameter('pan_channel',               4)
        self.declare_parameter('tilt_board',                0)
        self.declare_parameter('tilt_channel',              6)  # ch5 died (overheat 2026-07-09)
        # Per-axis safe travel envelope and resting gaze, in degrees. NARROWER than
        # the mechanical end-stops on purpose — driving into a stop stalls the servo.
        # TILT RECALIBRATED 2026-07-09 for the single servo on ch6 (reverser removed):
        # measured full-down=90°, full-up=105°, level=97° (higher angle = up). Limits
        # keep ~2° margin from each stop.
        # RE-CENTRED 2026-07-20: head_tilt neutral_pw moved 1500→1562 µs so servo_node's
        # startup neutral is level instead of full-down. servo_node pins 90° to
        # neutral_pw, so every tilt angle here shifted -7° to keep the same pulse widths
        # (level 97→90, down 92→85, up 103→96). Do not move one side without the other.
        self.declare_parameter('pan_min_deg',      52.0)
        self.declare_parameter('pan_max_deg',     128.0)
        self.declare_parameter('pan_neutral_deg',  90.0)
        self.declare_parameter('tilt_min_deg',     85.0)
        self.declare_parameter('tilt_max_deg',     96.0)
        self.declare_parameter('tilt_neutral_deg', 90.0)

        # ── Idle drift ───────────────────────────────────────────────────────
        # Runs only after the head has settled at neutral with nobody visible.
        # Targets are sampled inside the SAME travel envelope the tracker uses
        # (pan_min/max, tilt_min/max above) minus a margin, so drift can never
        # park on a limit and re-tuning the envelope moves drift with it.
        #
        # ALL OF THESE ARE TUNE-BY-FEEL. The defaults are a starting point, not
        # measured values — watch two minutes of it and adjust live.
        self.declare_parameter('drift_enabled',        True)
        # Deliberately NOT the full tracking_states set: drifting while SPEAKING
        # reads as not paying attention to the person being spoken to.
        self.declare_parameter('drift_states',         'IDLE,LISTENING')
        # Degrees/second the drift target advances. This is the "calm" dial —
        # the single knob that most decides whether motion reads alive or twitchy.
        self.declare_parameter('drift_speed_deg_s',    4.0)
        # Dwell = how long the head rests after arriving, before the next look.
        # Lognormal, NOT a fixed interval: a constant period is recognisable as
        # machine timing within ~30s. Lognormal is right-skewed, so most rests
        # are near the median with an occasional long "lost in thought" pause,
        # which is what the tail buys. Sigma 0.0 collapses it to a constant.
        self.declare_parameter('drift_dwell_median',   3.5)
        self.declare_parameter('drift_dwell_sigma',    0.6)
        self.declare_parameter('drift_dwell_min',      1.5)
        self.declare_parameter('drift_dwell_max',     12.0)
        # Reject a sampled pan target closer than this to where the head already
        # is. Without it the sampler regularly picks a neighbouring angle and the
        # head jiggles in place, which looks broken rather than idle.
        self.declare_parameter('drift_min_move_deg',   8.0)
        self.declare_parameter('drift_pan_margin_deg', 4.0)
        self.declare_parameter('drift_tilt_margin_deg', 1.0)

        # ── Instant orient ───────────────────────────────────────────────────
        # person_appeared on /omni/events → turn toward that camera immediately.
        self.declare_parameter('orient_enabled',   True)
        self.declare_parameter('orient_events_topic', '/omni/events')
        # Allowed in SPEAKING too (unlike drift) — turning toward someone who
        # just walked in is attentive even mid-sentence. Never includes
        # NAVIGATING/EXPLORING/DOCKING/ERROR.
        self.declare_parameter('orient_states',    'IDLE,LISTENING,SPEAKING')
        # camera=pan_degrees. 'head' orients to neutral (a snap to attention out
        # of a drift pose); 'rear' goes to the pan limit — a partial turn, since
        # the mechanical envelope cannot reach behind. Cameras absent from this
        # map produce no orient rather than a guessed direction.
        self.declare_parameter('orient_camera_pan_deg', 'head=90.0,rear=128.0')
        self.declare_parameter('orient_tilt_deg',  90.0)
        # Slew fraction per 50 Hz tick while orienting — higher than slew_alpha
        # so the turn is urgent, low enough that it glides rather than slams the
        # servo. 0.15 starts moving within one slew tick (~20 ms) and covers most
        # of the travel in ~0.5 s.
        self.declare_parameter('orient_slew_alpha', 0.15)
        # Hold the orient pose this long before drift/neutral may resume. Long
        # enough that a person walking in has time to enter frame and be picked
        # up by real tracking, which then takes over.
        self.declare_parameter('orient_hold',      2.5)
        # Chest LED pulse. Payload is the bare camera name; chest_node forwards
        # it as "PULSE:<camera>" to the panel firmware.
        # Publishing to /robot_state instead was rejected: behavior_node owns
        # that topic and four nodes key their state machines off it, so an LED
        # effect sent there would fight the system state machine.
        self.declare_parameter('chest_pulse_topic', '/chest/pulse')

        # ── Gaze choreography (NAVIGATING / EXPLORING) ────────────────────────
        # States that get gaze choreography instead of the neutral-snap. DOCKING
        # and ERROR are deliberately NOT here — they keep the snap (docking needs
        # predictable geometry, ERROR wants a halted head). Adding either would
        # break the AprilTag dock; do not.
        self.declare_parameter('gaze_states',        'NAVIGATING,EXPLORING')
        # Global plan (goal bearing) and local plan (travel curvature). Frames are
        # taken from each message header, not assumed, so a remap in nav2_params
        # needs no change here; base_frame is where the head lives.
        self.declare_parameter('plan_topic',         '/plan')
        self.declare_parameter('local_plan_topic',   '/local_plan')
        self.declare_parameter('odom_topic',         '/odometry/fused')
        self.declare_parameter('base_frame',         'base_link')
        # Sign mapping a body-frame bearing (rad, +ve = goal to the LEFT) onto pan
        # angle. Pan is hi=left/lo=right (see head-servo-directions), so +1.0 is
        # correct: a leftward goal raises the pan angle. Flip only if the servo is
        # recalibrated the other way.
        self.declare_parameter('gaze_pan_sign',      1.0)

        # Glance-before-move. Hold is SAMPLED in [min, max] each goal so it never
        # reads as a fixed beat; the prompt's 0.4–0.7 s window is the default.
        self.declare_parameter('glance_hold_min',    0.4)
        self.declare_parameter('glance_hold_max',    0.7)
        # Slew fraction/tick while turning toward the goal — deliberate and fairly
        # quick, like orient. Travel gaze is calmer (below).
        self.declare_parameter('glance_slew_alpha',  0.15)

        # Travel gaze. Look-ahead distance along the local plan sets how far the
        # head reads into the turn: too short and it never anticipates, too long
        # and it stares off into a curve the robot has not committed to.
        self.declare_parameter('travel_lookahead_m', 0.8)
        # Schmitt trigger against dithering: start LEADING once the look-ahead
        # bearing exceeds engage, only recenter once it drops back under release.
        # engage > release, always, or the hysteresis inverts and it chatters.
        self.declare_parameter('travel_engage_deg',  10.0)
        self.declare_parameter('travel_release_deg',  4.0)
        # Calm smoothing for the lead — lower than glance so travelling reads
        # relaxed rather than darting.
        self.declare_parameter('travel_slew_alpha',  0.08)

        # Face override during travel. Below face_override_speed (m/s) a visible
        # person wins over travel gaze; it only releases back above
        # speed + hyst, so creeping across the threshold does not flip the head.
        self.declare_parameter('face_override_speed',      0.12)
        self.declare_parameter('face_override_speed_hyst', 0.06)

        # Arrival settle: scripted glance offsets (deg from neutral, applied in
        # order) and the rest at each before the next. Ends at neutral, then hands
        # back to IDLE behaviour. A real subject appearing cancels it.
        self.declare_parameter('settle_glance_deg',  16.0)
        self.declare_parameter('settle_step_hold',   0.45)
        self.declare_parameter('settle_slew_alpha',  0.12)

        self._pan_gain        = self.get_parameter('pan_gain').value
        self._tilt_gain       = self.get_parameter('tilt_gain').value
        self._max_step        = self.get_parameter('max_step').value
        self._head_target_frac = self.get_parameter('head_target_frac').value
        self._slew_alpha       = self.get_parameter('slew_alpha').value
        self._conf_min        = self.get_parameter('detection_confidence_min').value
        self._prefer_faces    = self.get_parameter('prefer_faces').value
        self._face_conf_min   = self.get_parameter('face_confidence_min').value
        self._face_timeout    = self.get_parameter('face_timeout').value
        self._tracking_states = {
            s.strip()
            for s in self.get_parameter('tracking_states').value.split(',')
            if s.strip()
        }
        self._return_timeout  = self.get_parameter('return_to_neutral_timeout').value
        self._img_w           = float(self.get_parameter('image_width').value)
        self._img_h           = float(self.get_parameter('image_height').value)
        self._pan_board       = int(self.get_parameter('pan_board').value)
        self._pan_channel     = int(self.get_parameter('pan_channel').value)
        self._tilt_board      = int(self.get_parameter('tilt_board').value)
        self._tilt_channel    = int(self.get_parameter('tilt_channel').value)

        # Per-axis travel limits and neutral, converted to internal [0,1] norm.
        self._pan_min      = _deg_to_norm(self.get_parameter('pan_min_deg').value)
        self._pan_max      = _deg_to_norm(self.get_parameter('pan_max_deg').value)
        self._pan_neutral  = _deg_to_norm(self.get_parameter('pan_neutral_deg').value)
        self._tilt_min     = _deg_to_norm(self.get_parameter('tilt_min_deg').value)
        self._tilt_max     = _deg_to_norm(self.get_parameter('tilt_max_deg').value)
        self._tilt_neutral = _deg_to_norm(self.get_parameter('tilt_neutral_deg').value)

        # Drift / orient cached values. Speeds and margins are kept in degrees
        # where they are per-second rates (converted at use) and normalized where
        # they are positions, matching how the tracker stores each kind.
        self._drift_enabled     = bool(self.get_parameter('drift_enabled').value)
        self._drift_states      = {
            s.strip()
            for s in self.get_parameter('drift_states').value.split(',')
            if s.strip()
        }
        self._drift_speed       = self.get_parameter('drift_speed_deg_s').value
        self._drift_dwell_med   = self.get_parameter('drift_dwell_median').value
        self._drift_dwell_sigma = self.get_parameter('drift_dwell_sigma').value
        self._drift_dwell_min   = self.get_parameter('drift_dwell_min').value
        self._drift_dwell_max   = self.get_parameter('drift_dwell_max').value
        self._drift_min_move    = _deg_to_norm(
            self.get_parameter('drift_min_move_deg').value)
        self._drift_pan_margin  = _deg_to_norm(
            self.get_parameter('drift_pan_margin_deg').value)
        self._drift_tilt_margin = _deg_to_norm(
            self.get_parameter('drift_tilt_margin_deg').value)

        self._orient_enabled    = bool(self.get_parameter('orient_enabled').value)
        self._orient_states     = {
            s.strip()
            for s in self.get_parameter('orient_states').value.split(',')
            if s.strip()
        }
        self._orient_tilt       = _deg_to_norm(
            self.get_parameter('orient_tilt_deg').value)
        self._orient_slew_alpha = self.get_parameter('orient_slew_alpha').value
        self._orient_hold       = self.get_parameter('orient_hold').value
        self._orient_pan_by_cam = self._parse_camera_map(
            self.get_parameter('orient_camera_pan_deg').value)

        # Gaze choreography cached values. Angles that are POSITIONS become
        # normalized [0,1]; angles that are THRESHOLDS/OFFSETS stay in degrees
        # (compared against a bearing computed in degrees), matching how each is
        # used downstream.
        self._gaze_states = {
            s.strip()
            for s in self.get_parameter('gaze_states').value.split(',')
            if s.strip()
        }
        self._base_frame        = self.get_parameter('base_frame').value
        self._gaze_pan_sign     = self.get_parameter('gaze_pan_sign').value
        self._glance_hold_min   = self.get_parameter('glance_hold_min').value
        self._glance_hold_max   = self.get_parameter('glance_hold_max').value
        self._glance_slew_alpha = self.get_parameter('glance_slew_alpha').value
        self._travel_lookahead  = self.get_parameter('travel_lookahead_m').value
        self._travel_engage     = self.get_parameter('travel_engage_deg').value
        self._travel_release    = self.get_parameter('travel_release_deg').value
        self._travel_slew_alpha = self.get_parameter('travel_slew_alpha').value
        self._face_ovr_speed    = self.get_parameter('face_override_speed').value
        self._face_ovr_hyst     = self.get_parameter('face_override_speed_hyst').value
        self._settle_glance     = self.get_parameter('settle_glance_deg').value
        self._settle_step_hold  = self.get_parameter('settle_step_hold').value
        self._settle_slew_alpha = self.get_parameter('settle_slew_alpha').value

        # ── Live parameter tuning ─────────────────────────────────────────────
        # The tracking-loop knobs are read every 10 Hz tick, so changes take
        # effect immediately without a relaunch — this is the calibration loop,
        # e.g.:
        #   ros2 param set /head_tracking_node tilt_gain 0.08
        #   ros2 param set /head_tracking_node pan_max_deg 120.0
        # _PLAIN_PARAMS map straight to a cached float; _DEG_PARAMS are stored
        # internally as normalized [0,1] via _deg_to_norm(). Servo board/channel,
        # image size and tracking_states are init-time and not tunable here.
        self._plain_params = {
            'pan_gain':                 '_pan_gain',
            'tilt_gain':                '_tilt_gain',
            'max_step':                 '_max_step',
            'head_target_frac':         '_head_target_frac',
            'slew_alpha':               '_slew_alpha',
            'detection_confidence_min': '_conf_min',
            'face_confidence_min':      '_face_conf_min',
            'face_timeout':             '_face_timeout',
            'return_to_neutral_timeout': '_return_timeout',
            # Drift/orient tuning — all live, because every one of these is a
            # judged-by-eye value that needs adjusting while watching the robot.
            'drift_speed_deg_s':   '_drift_speed',
            'drift_dwell_median':  '_drift_dwell_med',
            'drift_dwell_sigma':   '_drift_dwell_sigma',
            'drift_dwell_min':     '_drift_dwell_min',
            'drift_dwell_max':     '_drift_dwell_max',
            'orient_slew_alpha':   '_orient_slew_alpha',
            'orient_hold':         '_orient_hold',
            # Gaze choreography — every knob is judged by eye against film, so
            # every one is live-tunable while the robot drives.
            'gaze_pan_sign':       '_gaze_pan_sign',
            'glance_hold_min':     '_glance_hold_min',
            'glance_hold_max':     '_glance_hold_max',
            'glance_slew_alpha':   '_glance_slew_alpha',
            'travel_lookahead_m':  '_travel_lookahead',
            'travel_engage_deg':   '_travel_engage',
            'travel_release_deg':  '_travel_release',
            'travel_slew_alpha':   '_travel_slew_alpha',
            'face_override_speed':      '_face_ovr_speed',
            'face_override_speed_hyst': '_face_ovr_hyst',
            'settle_glance_deg':   '_settle_glance',
            'settle_step_hold':    '_settle_step_hold',
            'settle_slew_alpha':   '_settle_slew_alpha',
        }
        self._deg_params = {
            'pan_min_deg':      '_pan_min',
            'pan_max_deg':      '_pan_max',
            'pan_neutral_deg':  '_pan_neutral',
            'tilt_min_deg':     '_tilt_min',
            'tilt_max_deg':     '_tilt_max',
            'tilt_neutral_deg': '_tilt_neutral',
            'drift_min_move_deg':    '_drift_min_move',
            'drift_pan_margin_deg':  '_drift_pan_margin',
            'drift_tilt_margin_deg': '_drift_tilt_margin',
            'orient_tilt_deg':       '_orient_tilt',
        }
        self.add_on_set_parameters_callback(self._on_set_parameters)

        # ── State ────────────────────────────────────────────────────────────
        # Servo positions as normalized [0.0, 1.0]; axis neutral is per-axis
        # (pan 90° → 0.5, tilt 84° → 0.467) because tilt's level gaze is offset.
        self._pan_norm         = self._pan_neutral   # TARGET (10 Hz control updates this)
        self._tilt_norm        = self._tilt_neutral
        self._pan_out          = self._pan_neutral   # smoothed OUTPUT (50 Hz slew publishes)
        self._tilt_out         = self._tilt_neutral
        self._robot_state      = 'IDLE'
        # Initialise to boot time so the 2-second timeout doesn't fire
        # immediately before any detection arrives.
        self._last_person_time = time.monotonic()
        self._latest_detection = None   # (cx_norm, cy_norm) when person seen
        self._last_face_time   = time.monotonic()
        self._latest_face      = None   # (cx_norm, cy_norm) when face seen

        # Idle drift: _drift_goal is where the head is walking to; while it is
        # None the head is resting and _drift_resume_at says until when.
        self._drift_goal      = None    # (pan_norm, tilt_norm)
        self._drift_resume_at = 0.0
        # Latch: True once drift has taken over the head. Cleared whenever
        # something else claims it (tracking, orient, state change) so that the
        # next idle period starts from a neutral return again.
        self._drifting        = False

        # Instant orient: _orient_until is the monotonic deadline the pose is
        # held to. Past it, drift/neutral may resume. Zero means not orienting.
        self._orient_until = 0.0

        # ── Gaze choreography state ───────────────────────────────────────────
        # _gaze_phase is None outside gaze_states, else GLANCE or TRAVEL.
        self._gaze_phase        = None
        # Glance: bearing is None until the first /plan lets us compute it; the
        # hold deadline is armed at that same moment, so "hold before move" is
        # measured from when the head actually knows where to look.
        self._glance_pan_goal   = None    # normalized pan target toward the goal
        self._glance_hold_until = 0.0
        # Travel gaze Schmitt-trigger latch: True while actively leading a turn,
        # False while centered. Flips on engage/release thresholds only.
        self._travel_leading    = False
        # Face-override latch, with its own speed hysteresis (separate from the
        # face_timeout freshness test) so it does not flip at the speed threshold.
        self._face_override     = False
        # Latest plans and odometry speed, each with a receipt time for staleness.
        self._global_plan       = None    # (frame_id, [(x, y), ...])
        self._global_plan_time  = 0.0
        self._local_plan        = None    # (frame_id, [(x, y), ...])
        self._local_plan_time   = 0.0
        self._speed             = 0.0
        self._speed_time        = 0.0

        # Arrival settle sequence: a list of (pan_norm, hold_s) steps walked in
        # order, then cleared. _settling gates it; _settle_arrive_at times the
        # dwell at each step. Empty/False means not settling.
        self._settling          = False
        self._settle_steps      = []
        self._settle_idx        = 0
        self._settle_arrive_at  = 0.0

        # TF: goal/curvature bearings are computed by transforming plan poses into
        # base_frame. Zero-timeout lookups only (single-threaded executor — see
        # ekf-transform-timeout); a missing transform means "hold this tick".
        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # ── Publishers ───────────────────────────────────────────────────────
        self._servo_pub = self.create_publisher(Float32MultiArray, '/servo_commands', 10)
        self._chest_pub = self.create_publisher(
            String, self.get_parameter('chest_pulse_topic').value, 10)

        # ── Subscriptions ────────────────────────────────────────────────────
        sensor_qos = QoSPresetProfiles.SENSOR_DATA.value
        self.create_subscription(
            Detection2DArray, '/camera/detections', self._on_detections, sensor_qos
        )
        self.create_subscription(
            Detection2DArray, '/camera/faces', self._on_faces, sensor_qos
        )
        self.create_subscription(String, '/robot_state', self._on_robot_state, 10)
        # Reliable depth-10, matching event_generator's publisher QoS exactly —
        # a BEST_EFFORT sub here would silently match nothing.
        self.create_subscription(
            String, self.get_parameter('orient_events_topic').value,
            self._on_event, 10
        )
        # Gaze choreography inputs. Nav2 publishes /plan and /local_plan on the
        # default (reliable) QoS; odometry/fused is SENSOR_DATA (best effort) like
        # any high-rate odom source — a reliable sub would silently match nothing.
        self.create_subscription(
            Path, self.get_parameter('plan_topic').value, self._on_global_plan, 10
        )
        self.create_subscription(
            Path, self.get_parameter('local_plan_topic').value,
            self._on_local_plan, 10
        )
        self.create_subscription(
            Odometry, self.get_parameter('odom_topic').value, self._on_odom,
            QoSPresetProfiles.SENSOR_DATA.value
        )

        # ── Timers: 10 Hz control sets the TARGET; 50 Hz slew smooths the OUTPUT ──
        self.create_timer(0.1, self._control_cb)
        self.create_timer(1.0 / _SLEW_HZ, self._slew_cb)

        self.get_logger().info(
            f'head_tracking_node started — '
            f'pan_gain={self._pan_gain}, tilt_gain={self._tilt_gain}, '
            f'max_step={self._max_step}, conf_min={self._conf_min}, '
            f'prefer_faces={self._prefer_faces}, '
            f'face_conf_min={self._face_conf_min}, face_timeout={self._face_timeout}s, '
            f'dead_zone={_DEAD_ZONE}, '
            f'pan_travel=[{self._pan_min * 180:.0f}°, {self._pan_max * 180:.0f}°]@'
            f'{self._pan_neutral * 180:.0f}°, '
            f'tilt_travel=[{self._tilt_min * 180:.0f}°, {self._tilt_max * 180:.0f}°]@'
            f'{self._tilt_neutral * 180:.0f}°, '
            f'tracking_states={sorted(self._tracking_states)}, '
            f'return_timeout={self._return_timeout}s, '
            f'frame={int(self._img_w)}x{int(self._img_h)}, '
            f'pan=board{self._pan_board}/ch{self._pan_channel}, '
            f'tilt=board{self._tilt_board}/ch{self._tilt_channel}'
        )
        self.get_logger().info(
            f'idle drift {"on" if self._drift_enabled else "OFF"} — '
            f'states={sorted(self._drift_states)}, '
            f'speed={self._drift_speed}°/s, '
            f'dwell=lognormal(median={self._drift_dwell_med}s, '
            f'sigma={self._drift_dwell_sigma}) '
            f'clamped [{self._drift_dwell_min}, {self._drift_dwell_max}]s, '
            f'min_move={self._drift_min_move * 180:.0f}° | '
            f'orient {"on" if self._orient_enabled else "OFF"} — '
            f'states={sorted(self._orient_states)}, '
            f'cameras={ {c: f"{p * 180:.0f}°" for c, p in self._orient_pan_by_cam.items()} }, '
            f'slew_alpha={self._orient_slew_alpha}, hold={self._orient_hold}s'
        )
        self.get_logger().info(
            f'gaze choreography — states={sorted(self._gaze_states)}, '
            f'glance hold=[{self._glance_hold_min}, {self._glance_hold_max}]s '
            f'(slew_alpha={self._glance_slew_alpha}), '
            f'travel lookahead={self._travel_lookahead}m '
            f'engage/release={self._travel_engage}/{self._travel_release}° '
            f'(slew_alpha={self._travel_slew_alpha}), '
            f'face_override<{self._face_ovr_speed}m/s '
            f'(+{self._face_ovr_hyst} to release), '
            f'settle=±{self._settle_glance}°@{self._settle_step_hold}s | '
            f'plan={self.get_parameter("plan_topic").value}, '
            f'local_plan={self.get_parameter("local_plan_topic").value}, '
            f'odom={self.get_parameter("odom_topic").value}, '
            f'base_frame={self._base_frame}'
        )

    # ── Init helpers ──────────────────────────────────────────────────────────

    def _parse_camera_map(self, spec: str) -> dict:
        """Parse 'head=90.0,rear=128.0' into {camera: pan_norm}.

        Malformed entries are warned about and skipped rather than raising: a
        typo in a launch argument should cost one orient direction, not the head
        servos. The resulting angles are clamped into the pan envelope, so a map
        asking for 180° gets the mechanical limit instead of a stalled servo.
        """
        out = {}
        for entry in str(spec).split(','):
            entry = entry.strip()
            if not entry:
                continue
            if '=' not in entry:
                self.get_logger().warn(
                    f'orient_camera_pan_deg entry {entry!r} is not camera=degrees, skipping')
                continue
            cam, deg = entry.split('=', 1)
            cam = cam.strip()
            try:
                norm = _deg_to_norm(float(deg))
            except ValueError:
                self.get_logger().warn(
                    f'orient_camera_pan_deg entry {entry!r} has a non-numeric angle, skipping')
                continue
            if not cam:
                continue
            out[cam] = max(self._pan_min, min(self._pan_max, norm))
        return out

    # ── Live parameter callback ───────────────────────────────────────────────

    def _on_set_parameters(self, params):
        """
        Validate and apply runtime tuning changes (ros2 param set). Rejecting
        (successful=False) leaves the stored value untouched. Validate the whole
        batch before mutating, since a set is applied atomically. Degree params
        are stored normalized via _deg_to_norm(). Gains may be negative (the sign
        sets tracking direction), so only their type is checked.
        """
        pending = []  # (attr, value, name) to apply once all checks pass
        for p in params:
            if p.name in self._plain_params:
                try:
                    value = float(p.value)
                except (TypeError, ValueError):
                    return SetParametersResult(
                        successful=False,
                        reason=f'{p.name} must be a number, got {p.value!r}')
                if (p.name in ('detection_confidence_min', 'face_confidence_min')
                        and not (0.0 <= value <= 1.0)):
                    return SetParametersResult(
                        successful=False,
                        reason=f'{p.name} must be in 0.0–1.0, got {value}')
                if (p.name in ('max_step', 'return_to_neutral_timeout', 'face_timeout')
                        and value <= 0.0):
                    return SetParametersResult(
                        successful=False,
                        reason=f'{p.name} must be > 0, got {value}')
                pending.append((self._plain_params[p.name], value, p.name))

            elif p.name == 'prefer_faces':
                pending.append(('_prefer_faces', bool(p.value), p.name))

            elif p.name in self._deg_params:
                try:
                    deg = float(p.value)
                except (TypeError, ValueError):
                    return SetParametersResult(
                        successful=False,
                        reason=f'{p.name} must be a number, got {p.value!r}')
                if not (0.0 <= deg <= 180.0):
                    return SetParametersResult(
                        successful=False,
                        reason=f'{p.name} must be in 0–180°, got {deg}')
                pending.append((self._deg_params[p.name], _deg_to_norm(deg), p.name))
            # all other params are init-time only — accept without caching

        for attr, value, name in pending:
            setattr(self, attr, value)
            self.get_logger().info(f'Parameter updated: {name} = {value}')
        return SetParametersResult(successful=True)

    # ── Subscription callbacks ────────────────────────────────────────────────

    def _on_robot_state(self, msg: String):
        new_state = msg.data
        old_state = self._robot_state
        if new_state == old_state:
            return
        self._robot_state = new_state

        old_gaze = old_state in self._gaze_states
        new_tracking = new_state in self._tracking_states
        new_gaze     = new_state in self._gaze_states

        if new_gaze:
            # NAVIGATING / EXPLORING → begin choreography instead of snapping.
            self._begin_gaze(new_state)
        elif new_tracking:
            # Returning to a tracking state. If we were navigating/exploring this
            # is an arrival — run the settle sequence, then hand back to tracking.
            # Coming from DOCKING/ERROR (already snapped) it is a plain resume.
            self._end_gaze()
            if old_gaze:
                self.get_logger().info(
                    f'State {old_state} → {new_state} — arrival, running settle')
                self._begin_settle()
        else:
            # DOCKING / ERROR / anything else → the sacred neutral-snap. Docking
            # depends on predictable geometry; do not choreograph these.
            self.get_logger().info(
                f'State → {new_state} — snapping head to neutral')
            self._end_gaze()
            self._snap_to_neutral()

    def _on_detections(self, msg: Detection2DArray):
        # Record during gaze states too: face override (during travel) needs a
        # fresh person aim point, and _gaze_control reads _latest_detection.
        if (self._robot_state not in self._tracking_states
                and self._robot_state not in self._gaze_states):
            return

        best_score = -1.0
        best_cx = best_cy = None
        for det in msg.detections:
            for result in det.results:
                if (result.hypothesis.class_id.lower() == 'person'
                        and result.hypothesis.score >= self._conf_min
                        and result.hypothesis.score > best_score):
                    best_score = result.hypothesis.score
                    best_cx    = det.bbox.center.position.x / self._img_w
                    # Aim at the head region (near the top of the person box) rather
                    # than the torso center, so the robot looks at the face.
                    head_y     = (det.bbox.center.position.y
                                  - det.bbox.size_y * (0.5 - self._head_target_frac))
                    best_cy    = head_y / self._img_h

        if best_cx is not None:
            # Clamp to [0,1] — belt-and-suspenders against out-of-frame coords
            self._latest_detection = (
                max(0.0, min(1.0, best_cx)),
                max(0.0, min(1.0, best_cy)),
            )
            self._last_person_time = time.monotonic()

    def _on_faces(self, msg: Detection2DArray):
        """Highest-confidence face → aim directly at the face box CENTER.

        Unlike the person path, no head_target_frac offset is applied: the face
        box already IS the face, so its center is the gaze target.
        """
        # Record during gaze states too — face override during travel aims here.
        if (self._robot_state not in self._tracking_states
                and self._robot_state not in self._gaze_states):
            return

        best_score = -1.0
        best_cx = best_cy = None
        for det in msg.detections:
            for result in det.results:
                if (result.hypothesis.class_id.lower() == 'face'
                        and result.hypothesis.score >= self._face_conf_min
                        and result.hypothesis.score > best_score):
                    best_score = result.hypothesis.score
                    best_cx    = det.bbox.center.position.x / self._img_w
                    best_cy    = det.bbox.center.position.y / self._img_h

        if best_cx is not None:
            self._latest_face = (
                max(0.0, min(1.0, best_cx)),
                max(0.0, min(1.0, best_cy)),
            )
            self._last_face_time = time.monotonic()

    def _on_event(self, msg: String):
        """person_appeared on /omni/events → turn toward that camera NOW.

        The turn is started here rather than deferred to the 10 Hz control tick:
        setting the target immediately means the 50 Hz slew begins moving on its
        next tick (~20 ms), instead of waiting up to 100 ms for the control loop
        to come round. That is the difference between "it reacted" and "it
        thought about it first".

        Note this deliberately does NOT check whether a person is currently
        visible. If they are, the very next control tick's face/person branch
        overrides the orient target anyway — tracking always wins — so there is
        no need to guess here.
        """
        if not self._orient_enabled or self._robot_state not in self._orient_states:
            return
        try:
            event = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            self.get_logger().warn('unparseable /omni/events payload, ignoring', once=True)
            return
        if not isinstance(event, dict) or event.get('kind') != _EVENT_PERSON_APPEARED:
            return

        camera = event.get('camera', '')
        pan_goal = self._orient_pan_by_cam.get(camera)
        if pan_goal is None:
            self.get_logger().debug(
                f'person_appeared on unmapped camera {camera!r} — no orient')
            return

        now = time.monotonic()
        self._orient_until = now + self._orient_hold
        self._drift_goal   = None          # abandon any drift look in progress
        self._drifting     = False
        self._pan_norm     = pan_goal
        self._tilt_norm    = max(self._tilt_min, min(self._tilt_max, self._orient_tilt))
        self._pulse_chest(camera)
        self.get_logger().info(
            f'orienting to {camera} — pan {pan_goal * 180:.0f}° '
            f'(identity={event.get("identity", "?")})'
        )

    def _pulse_chest(self, camera: str):
        """Ask the chest panel for an attention pulse.

        Payload is the bare camera name — the topic already says "pulse".
        chest_node allowlists it and forwards "PULSE:<camera>" over the UART;
        the firmware overlays a ~700 ms flash on the LED bars without touching
        the displayed robot state.
        """
        self._chest_pub.publish(String(data=camera))

    # ── Gaze inputs (plans + odometry) ─────────────────────────────────────────

    @staticmethod
    def _path_points(msg: Path):
        """Flatten a nav_msgs/Path into (frame_id, [(x, y), ...])."""
        return (
            msg.header.frame_id,
            [(p.pose.position.x, p.pose.position.y) for p in msg.poses],
        )

    def _on_global_plan(self, msg: Path):
        """Nav2 global plan. Its endpoint is the goal — used for the glance."""
        if not msg.poses:
            return
        self._global_plan      = self._path_points(msg)
        self._global_plan_time = time.monotonic()

    def _on_local_plan(self, msg: Path):
        """DWB local plan. Its near-future curvature drives travel gaze."""
        if not msg.poses:
            return
        self._local_plan      = self._path_points(msg)
        self._local_plan_time = time.monotonic()

    def _on_odom(self, msg: Odometry):
        """Fused odometry → planar speed, for the face-override threshold."""
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self._speed      = math.hypot(vx, vy)
        self._speed_time = time.monotonic()

    # ── Gaze lifecycle ─────────────────────────────────────────────────────────

    def _begin_gaze(self, state: str):
        """Enter NAVIGATING/EXPLORING choreography (replaces the neutral-snap).

        NAVIGATING starts in GLANCE and waits for the first /plan to learn the
        goal bearing. EXPLORING has no single goal, so it goes straight to travel
        gaze off the local plan. Either way the head is levelled and any idle /
        drift / orient / settle activity is dropped.
        """
        self._latest_detection = None
        self._latest_face      = None
        self._drift_goal       = None
        self._drifting         = False
        self._orient_until     = 0.0
        self._clear_settle()
        self._travel_leading   = False
        self._face_override    = False
        # Level the head immediately; pan holds where it is until we know the goal.
        self._tilt_norm = self._tilt_neutral
        if state == 'NAVIGATING':
            self._gaze_phase      = _GAZE_GLANCE
            self._glance_pan_goal = None          # armed by the first /plan
            self._glance_hold_until = 0.0
            self.get_logger().info('NAVIGATING — gaze glance, awaiting goal plan')
        else:
            self._gaze_phase      = _GAZE_TRAVEL
            self.get_logger().info(f'{state} — travel gaze')

    def _end_gaze(self):
        """Leave gaze states: drop the phase and travel/glance latches."""
        self._gaze_phase      = None
        self._glance_pan_goal = None
        self._travel_leading  = False
        self._face_override   = False

    def _begin_settle(self):
        """Arm the arrival settle: glance one way, the other, rest at neutral.

        Runs UNDER tracking (like drift): the next real subject cancels it. The
        offsets are clamped into the pan envelope so a wide settle on a narrow
        rig still lands on legal poses.
        """
        off = _deg_to_norm(self._settle_glance)
        left  = max(self._pan_min, min(self._pan_max, self._pan_neutral + off))
        right = max(self._pan_min, min(self._pan_max, self._pan_neutral - off))
        self._settle_steps     = [left, right, self._pan_neutral]
        self._settle_idx       = 0
        self._settle_arrive_at = 0.0
        self._settling         = True
        self._tilt_norm        = self._tilt_neutral

    def _clear_settle(self):
        self._settling         = False
        self._settle_steps     = []
        self._settle_idx       = 0
        self._settle_arrive_at = 0.0

    # ── TF bearing helpers ──────────────────────────────────────────────────────

    def _bearing_in_base(self, frame: str, x: float, y: float):
        """Bearing (rad) from the robot to map/odom point (x, y), +ve = LEFT.

        Transforms the point into base_frame via TF, then atan2(y, x). Returns
        None if the transform is unavailable — a zero-timeout lookup, so this
        never blocks the executor; the caller treats None as "hold this tick".
        """
        try:
            tf = self._tf_buffer.lookup_transform(
                self._base_frame, frame, Time(), timeout=Duration(seconds=0.0))
        except TransformException:
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        # Planar yaw of base_frame←frame from the quaternion (z, w suffice in 2D).
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        x_bl = cos_y * x - sin_y * y + t.x
        y_bl = sin_y * x + cos_y * y + t.y
        return math.atan2(y_bl, x_bl)

    def _bearing_to_pan(self, bearing_rad: float) -> float:
        """Map a body-frame bearing onto a clamped normalized pan target.

        1:1 in degrees so the head physically points along the bearing, signed by
        gaze_pan_sign and clamped to the pan envelope (a goal behind the robot
        becomes a partial turn to the limit, same as the orient rear caveat).
        """
        pan_deg = (self._pan_neutral * 180.0
                   + self._gaze_pan_sign * math.degrees(bearing_rad))
        return max(self._pan_min, min(self._pan_max, _deg_to_norm(pan_deg)))

    def _lookahead_bearing(self, plan, plan_time: float, now: float):
        """Bearing to a point travel_lookahead_m along a stored plan, or None.

        Walks the plan accumulating arc length until travel_lookahead_m, then
        takes that point's bearing in base_frame. None if the plan is stale,
        empty, or its transform is unavailable — all "hold this tick".
        """
        if plan is None or (now - plan_time) > _PLAN_STALE_S:
            return None
        frame, pts = plan
        if not pts:
            return None
        target = pts[-1]
        acc = 0.0
        prev = pts[0]
        for pt in pts[1:]:
            acc += math.hypot(pt[0] - prev[0], pt[1] - prev[1])
            prev = pt
            if acc >= self._travel_lookahead:
                target = pt
                break
        return self._bearing_in_base(frame, target[0], target[1])

    # ── Control loop (10 Hz) ──────────────────────────────────────────────────

    def _control_cb(self):
        now = time.monotonic()
        if self._robot_state in self._tracking_states:
            self._tracking_control(now)
        elif self._robot_state in self._gaze_states:
            self._gaze_control(now)
        # DOCKING / ERROR / unknown: the head was snapped to neutral on entry and
        # is left still. Nothing to do here.

    def _tracking_control(self, now: float):
        # Was a face seen recently? This is deliberately a TIMESTAMP test, independent
        # of whether an unconsumed sample is sitting in _latest_face. /camera/faces
        # publishes at 10 Hz and this loop runs at 10 Hz — two unsynchronised timers,
        # so they drift in and out of phase and there are regularly ticks with no new
        # face message. Gating the source choice on `_latest_face is not None` (as this
        # did) made those ticks fall through to the person box, and since the person
        # aim point (head_target_frac down the box) is a DIFFERENT point than the face
        # box center, the head visibly wandered between two targets. That also made
        # face_timeout dead: the None check always fired first.
        face_recent = (self._prefer_faces
                       and (now - self._last_face_time) < self._face_timeout)
        person_fresh = (self._latest_detection is not None
                        and (now - self._last_person_time) < self._return_timeout)

        if face_recent and self._latest_face is not None:
            # Preferred: a face is visible — aim straight at the face box center.
            cx_norm, cy_norm  = self._latest_face
            self._latest_face = None   # consume; next update comes from next msg
            self._apply_tracking(cx_norm, cy_norm)
        elif face_recent:
            # A face was seen within face_timeout but no new sample this tick. HOLD —
            # do not fall back to the person box. Switching source here is what caused
            # the wander. Holding is also correct on its own terms: the last face
            # measurement was already applied, and re-applying it would integrate a
            # stale error. Once face_timeout lapses the person fallback resumes.
            pass
        elif person_fresh:
            # Fallback: no fresh face, but a person box is available.
            cx_norm, cy_norm       = self._latest_detection
            self._latest_detection = None
            self._apply_tracking(cx_norm, cy_norm)
        elif (now - max(self._last_face_time, self._last_person_time)
              ) >= self._return_timeout:
            # Nothing seen for a while. The arrival settle, if armed, runs first
            # (it sits above idle drift / neutral, but still under real tracking,
            # which is why it lives on THIS no-subject branch and is cancelled in
            # _apply_tracking). Once the settle finishes we fall through to the
            # ordinary idle behaviour, handing the head back to IDLE.
            if self._settling:
                if self._settle_control(now):
                    return
            self._idle_cb(now)

    # ── Idle behaviour (runs only when the tracker has nothing) ───────────────

    def _idle_cb(self, now: float):
        """Neutral return, then idle drift. Only reached with no fresh subject.

        Order matters: the existing neutral decay runs to completion first, so
        the familiar "person leaves, head settles back to centre" motion is
        unchanged. Drift begins only once the head is actually resting at
        neutral, which also gives it a well-defined starting pose.
        """
        if now < self._orient_until:
            # Holding an orient pose — the slew is still carrying the head
            # there. Decaying to neutral now would drag it straight back.
            return

        drift_ok = self._drift_enabled and self._robot_state in self._drift_states
        if self._drifting and not drift_ok:
            # Drift was running and is no longer allowed (e.g. IDLE → SPEAKING).
            # Drop the latch and fall through to the neutral return below, so the
            # head walks back to centre instead of freezing wherever it was.
            self._drifting   = False
            self._drift_goal = None

        if not self._drifting:
            # Neutral return runs whether or not drift is enabled — this is the
            # pre-existing behaviour for every tracking state and must not become
            # conditional on drift.
            if not self._at_neutral():
                self._step_toward_neutral()
                return
            if not drift_ok:
                return
            # At neutral and allowed to drift: take the latch. It is load-bearing
            # — without it drift moves the head off neutral, the not-at-neutral
            # test passes again next tick, and drift and the neutral decay fight
            # each other to a standstill near centre.
            self._drifting = True

        self._drift_step(now)

    def _at_neutral(self) -> bool:
        return (abs(self._pan_norm  - self._pan_neutral)  <= _NEUTRAL_SNAP
                and abs(self._tilt_norm - self._tilt_neutral) <= _NEUTRAL_SNAP)

    def _drift_step(self, now: float):
        """One 10 Hz tick of idle drift: rest, pick a look, walk to it, repeat."""
        if self._drift_goal is None:
            # Resting between looks.
            if now < self._drift_resume_at:
                return
            self._drift_goal = self._pick_drift_target()
            return

        goal_pan, goal_tilt = self._drift_goal
        # Advance the TARGET at a fixed angular rate; the 50 Hz slew smooths the
        # published output on top. Rate-limiting the target (rather than leaning
        # on slew_alpha) is what makes the speed independent of the tracking
        # smoothing — turning drift down must not make face tracking laggy.
        max_move = _deg_to_norm(self._drift_speed) * 0.1   # per 10 Hz tick

        self._pan_norm  = _approach(self._pan_norm,  goal_pan,  max_move)
        self._tilt_norm = _approach(self._tilt_norm, goal_tilt, max_move)

        if (abs(self._pan_norm - goal_pan) <= _DRIFT_ARRIVE
                and abs(self._tilt_norm - goal_tilt) <= _DRIFT_ARRIVE):
            self._drift_goal      = None
            self._drift_resume_at = now + self._sample_dwell()

    def _pick_drift_target(self):
        """Sample a look target inside the envelope, avoiding tiny moves.

        Rejection-sampling on the pan distance is what stops the head jiggling
        in place: a uniform sample lands near the current angle often enough to
        be noticeable, and a 2° twitch reads as a fault rather than as idling.
        Tilt is not distance-checked — its envelope is only ~11° wide, so any
        minimum move there would reject nearly everything.
        """
        pan_lo  = min(self._pan_min + self._drift_pan_margin, self._pan_max)
        pan_hi  = max(self._pan_max - self._drift_pan_margin, pan_lo)
        tilt_lo = min(self._tilt_min + self._drift_tilt_margin, self._tilt_max)
        tilt_hi = max(self._tilt_max - self._drift_tilt_margin, tilt_lo)

        pan = self._pan_norm
        for _ in range(_DRIFT_TARGET_TRIES):
            pan = random.uniform(pan_lo, pan_hi)
            if abs(pan - self._pan_norm) >= self._drift_min_move:
                break
        return (pan, random.uniform(tilt_lo, tilt_hi))

    def _sample_dwell(self) -> float:
        """Lognormal rest time, clamped to [min, max].

        Lognormal because the failure mode being avoided is *rhythm*: a fixed
        or uniform-narrow dwell makes the head a metronome. The median is the
        typical rest; the right tail supplies the occasional long pause that
        breaks the pattern. sigma=0 degenerates to a constant, which is a
        legitimate setting for A/B-ing the effect while tuning.
        """
        median = max(1e-3, self._drift_dwell_med)
        if self._drift_dwell_sigma <= 0.0:
            dwell = median
        else:
            dwell = random.lognormvariate(math.log(median), self._drift_dwell_sigma)
        lo, hi = self._drift_dwell_min, max(self._drift_dwell_min, self._drift_dwell_max)
        return max(lo, min(hi, dwell))

    # ── Gaze choreography (10 Hz, only while in a gaze_state) ──────────────────

    def _gaze_control(self, now: float):
        """One tick of NAVIGATING/EXPLORING gaze. Sets the TARGET; slew smooths."""
        if self._gaze_phase == _GAZE_GLANCE:
            self._tilt_norm = self._tilt_neutral
            if self._glance_pan_goal is None:
                # Still waiting on the first plan + a usable transform. Hold pose.
                self._arm_glance(now)
                return
            # Aim at the goal and hold before travel gaze takes over.
            self._pan_norm = self._glance_pan_goal
            if now >= self._glance_hold_until:
                self._gaze_phase = _GAZE_TRAVEL
                self.get_logger().info('glance hold complete — travel gaze')
            return

        # TRAVEL (and EXPLORING throughout): face override, else lead the plan.
        if self._face_override_active(now):
            self._face_override = True
            aim = (self._latest_face if self._latest_face is not None
                   else self._latest_detection)
            if aim is not None:
                self._latest_face = None   # consume; next comes from next msg
                self._apply_tracking(aim[0], aim[1])
            return
        self._face_override = False
        self._travel_step(now)

    def _arm_glance(self, now: float):
        """Compute the goal bearing from the first fresh global plan we can
        transform, and arm the hold. Silently retries next tick until both the
        plan and the TF are available — that wait IS the head 'thinking', and it
        overlaps Nav2's own plan-compute latency."""
        if self._global_plan is None or (now - self._global_plan_time) > _PLAN_STALE_S:
            return
        frame, pts = self._global_plan
        gx, gy = pts[-1]                        # plan endpoint = the goal
        bearing = self._bearing_in_base(frame, gx, gy)
        if bearing is None:
            return
        self._glance_pan_goal   = self._bearing_to_pan(bearing)
        hold = random.uniform(self._glance_hold_min,
                              max(self._glance_hold_min, self._glance_hold_max))
        self._glance_hold_until = now + hold
        self.get_logger().info(
            f'glance armed — goal bearing {math.degrees(bearing):+.0f}°, '
            f'pan {self._glance_pan_goal * 180:.0f}°, hold {hold:.2f}s')

    def _current_speed(self, now: float) -> float:
        """Planar speed, or +inf when odom is stale — an unknown speed must read
        as 'moving' so face override never sticks the head on a face blind."""
        if (now - self._speed_time) > _ODOM_STALE_S:
            return math.inf
        return self._speed

    def _face_override_active(self, now: float) -> bool:
        """Should a visible person override travel gaze this tick?

        Visible AND slow. Speed uses hysteresis around face_override_speed (the
        latch self._face_override carries the state) so creeping across the
        threshold does not flip the head between the person and the corridor.
        """
        face_recent  = (self._prefer_faces
                        and (now - self._last_face_time) < self._face_timeout)
        person_fresh = (self._latest_detection is not None
                        and (now - self._last_person_time) < self._return_timeout)
        if not (face_recent or person_fresh):
            return False
        speed = self._current_speed(now)
        if self._face_override:
            return speed <= (self._face_ovr_speed + self._face_ovr_hyst)
        return speed < self._face_ovr_speed

    def _travel_step(self, now: float):
        """Lead the head into the local plan's upcoming curvature, center on
        straights, with a Schmitt trigger so it never dithers at the boundary."""
        self._tilt_norm = self._tilt_neutral
        bearing = self._lookahead_bearing(
            self._local_plan, self._local_plan_time, now)
        if bearing is None:
            # No usable local plan (stopped, planning, TF gap) — rest at center.
            self._pan_norm       = self._pan_neutral
            self._travel_leading = False
            return
        bearing_deg = abs(math.degrees(bearing))
        if self._travel_leading:
            if bearing_deg < self._travel_release:
                self._travel_leading = False
        elif bearing_deg > self._travel_engage:
            self._travel_leading = True
        self._pan_norm = (self._bearing_to_pan(bearing)
                          if self._travel_leading else self._pan_neutral)

    def _settle_control(self, now: float) -> bool:
        """Walk the arrival settle sequence. Returns True while still settling
        (so the idle path yields to it), False once done or nothing armed."""
        if self._settle_idx >= len(self._settle_steps):
            self._clear_settle()
            return False
        target = self._settle_steps[self._settle_idx]
        self._pan_norm  = target
        self._tilt_norm = self._tilt_neutral
        if abs(self._pan_out - target) <= _NEUTRAL_SNAP:
            if self._settle_arrive_at == 0.0:
                self._settle_arrive_at = now + self._settle_step_hold
            elif now >= self._settle_arrive_at:
                self._settle_idx      += 1
                self._settle_arrive_at = 0.0
        return True

    def _apply_tracking(self, cx_norm, cy_norm):
        """P-control step toward a target (already the intended gaze point).

        Updates the TARGET only; the 50 Hz slew timer publishes smoothed motion.
        """
        # A real subject has claimed the head — release the drift and settle
        # latches so that when they leave, the head returns to neutral before
        # idling again. (Also cancels an in-progress arrival settle: a person is
        # more important than finishing a scripted glance.)
        self._drifting   = False
        self._drift_goal = None
        self._clear_settle()

        pan_error  = cx_norm - 0.5
        tilt_error = cy_norm - 0.5

        if abs(pan_error) > _DEAD_ZONE:
            delta = max(-self._max_step, min(self._max_step,
                self._pan_gain * pan_error))
            self._pan_norm = max(self._pan_min, min(self._pan_max,
                self._pan_norm + delta))
        if abs(tilt_error) > _DEAD_ZONE:
            delta = max(-self._max_step, min(self._max_step,
                self._tilt_gain * tilt_error))
            self._tilt_norm = max(self._tilt_min, min(self._tilt_max,
                self._tilt_norm + delta))

    def _slew_cb(self):
        """50 Hz: glide the published output toward the target — kills the 10 Hz notch.

        While orienting, a larger alpha is used so the turn is urgent. Drift uses
        the normal alpha: its slowness comes from rate-limiting the target, not
        from the smoothing, so the two dials stay independent. The gaze phases
        pick their own: glance is deliberate-quick, travel is calm, face override
        during travel borrows the responsive tracking alpha, settle is gentle.
        """
        now = time.monotonic()
        if now < self._orient_until:
            alpha = self._orient_slew_alpha
        elif self._gaze_phase == _GAZE_GLANCE:
            alpha = self._glance_slew_alpha
        elif self._gaze_phase == _GAZE_TRAVEL:
            alpha = self._slew_alpha if self._face_override else self._travel_slew_alpha
        elif self._settling:
            alpha = self._settle_slew_alpha
        else:
            alpha = self._slew_alpha
        new_pan  = self._pan_out  + (self._pan_norm  - self._pan_out)  * alpha
        new_tilt = self._tilt_out + (self._tilt_norm - self._tilt_out) * alpha
        # settle exactly on target so we stop republishing once caught up
        if abs(self._pan_norm  - new_pan)  <= _SLEW_SNAP:
            new_pan  = self._pan_norm
        if abs(self._tilt_norm - new_tilt) <= _SLEW_SNAP:
            new_tilt = self._tilt_norm
        if new_pan != self._pan_out or new_tilt != self._tilt_out:
            self._pan_out  = new_pan
            self._tilt_out = new_tilt
            self._publish_servos()

    # ── Servo helpers ─────────────────────────────────────────────────────────

    def _step_toward_neutral(self):
        """Proportional decay toward each axis's neutral. Snaps once close enough."""
        pan_close  = abs(self._pan_norm  - self._pan_neutral)  <= _NEUTRAL_SNAP
        tilt_close = abs(self._tilt_norm - self._tilt_neutral) <= _NEUTRAL_SNAP
        if pan_close and tilt_close:
            return

        if not pan_close:
            step = (self._pan_neutral - self._pan_norm) * _NEUTRAL_RETURN_RATE
            step = max(-self._max_step, min(self._max_step, step))
            self._pan_norm = max(self._pan_min, min(self._pan_max, self._pan_norm + step))
            if abs(self._pan_norm - self._pan_neutral) <= _NEUTRAL_SNAP:
                self._pan_norm = self._pan_neutral
        if not tilt_close:
            step = (self._tilt_neutral - self._tilt_norm) * _NEUTRAL_RETURN_RATE
            step = max(-self._max_step, min(self._max_step, step))
            self._tilt_norm = max(self._tilt_min, min(self._tilt_max, self._tilt_norm + step))
            if abs(self._tilt_norm - self._tilt_neutral) <= _NEUTRAL_SNAP:
                self._tilt_norm = self._tilt_neutral
        # target decayed toward neutral; the 50 Hz slew publishes the smoothed motion

    def _snap_to_neutral(self):
        """Immediately publish neutral and reset internal state.

        Also cancels drift and orient. This is the path taken on entry to
        NAVIGATING/EXPLORING/DOCKING/ERROR, so an in-flight idle look must not
        survive into a state that expects a still head.
        """
        self._latest_detection = None
        self._latest_face      = None
        self._drift_goal       = None
        self._drift_resume_at  = 0.0
        self._drifting         = False
        self._orient_until     = 0.0
        # Also drop any gaze choreography — a snap is taken on entry to
        # DOCKING/ERROR, which must not inherit an in-flight glance or settle.
        self._gaze_phase       = None
        self._glance_pan_goal  = None
        self._travel_leading   = False
        self._face_override    = False
        self._clear_settle()
        self._pan_norm  = self._pan_neutral
        self._tilt_norm = self._tilt_neutral
        self._pan_out   = self._pan_neutral
        self._tilt_out  = self._tilt_neutral
        self._publish_servos()

    def _publish_servos(self):
        """Publish pan and tilt angles as a single Float32MultiArray command."""
        pan_deg  = self._pan_out  * 180.0
        tilt_deg = self._tilt_out * 180.0

        msg = Float32MultiArray()
        # servo_node expects groups of three: [board_id, channel, angle_degrees, ...]
        msg.data = [
            float(self._pan_board),  float(self._pan_channel),  pan_deg,
            float(self._tilt_board), float(self._tilt_channel), tilt_deg,
        ]
        self._servo_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = HeadTrackingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
