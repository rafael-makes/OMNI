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

Active only in IDLE, LISTENING, and SPEAKING states. Transitions to other
states (NAVIGATING, EXPLORING, DOCKING, ERROR) snap the head to neutral
immediately. After return_to_neutral_timeout seconds without a person
the head returns to neutral via proportional decay (smooth, not instant).

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
"""

import json
import math
import random
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import Float32MultiArray, String
from vision_msgs.msg import Detection2DArray

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
        if new_state == self._robot_state:
            return
        was_tracking = self._robot_state in self._tracking_states
        self._robot_state = new_state
        if was_tracking and new_state not in self._tracking_states:
            self.get_logger().info(
                f'State → {new_state} — suspending head tracking, snapping to neutral'
            )
            self._snap_to_neutral()

    def _on_detections(self, msg: Detection2DArray):
        if self._robot_state not in self._tracking_states:
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
        if self._robot_state not in self._tracking_states:
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

    # ── Control loop (10 Hz) ──────────────────────────────────────────────────

    def _control_cb(self):
        if self._robot_state not in self._tracking_states:
            return

        now = time.monotonic()
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
            # Nothing seen for a while.
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

    def _apply_tracking(self, cx_norm, cy_norm):
        """P-control step toward a target (already the intended gaze point).

        Updates the TARGET only; the 50 Hz slew timer publishes smoothed motion.
        """
        # A real subject has claimed the head — release the drift latch so that
        # when they leave, the head returns to neutral before idling again.
        self._drifting   = False
        self._drift_goal = None

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
        from the smoothing, so the two dials stay independent.
        """
        alpha = (self._orient_slew_alpha
                 if time.monotonic() < self._orient_until
                 else self._slew_alpha)
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
