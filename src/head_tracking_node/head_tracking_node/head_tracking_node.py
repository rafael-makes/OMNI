"""
head_tracking_node.py — Pan/tilt head tracking for OMNI.

Subscribes to /camera/detections and drives head_pan + head_tilt servos
via /servo_commands to keep the highest-confidence person detection
centered in the camera frame.

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
"""

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


def _deg_to_norm(deg: float) -> float:
    """servo_node maps 0–180° onto a servo's pulse range; we track [0,1] internally."""
    return deg / 180.0


class HeadTrackingNode(Node):

    def __init__(self):
        super().__init__('head_tracking_node')

        # ── Parameters ──────────────────────────────────────────────────────
        # Gain signs come from measured direction convention (see project memory
        # head-servo-directions): pan hi=left/lo=right, tilt hi=down/lo=up.
        #   person right (pan_error +ve) -> turn right -> angle down -> pan_gain -ve
        #   person below (tilt_error +ve) -> tilt down -> angle up   -> tilt_gain +ve
        self.declare_parameter('pan_gain',                  -0.05)
        self.declare_parameter('tilt_gain',                  0.05)
        self.declare_parameter('max_step',                  0.02)
        self.declare_parameter('detection_confidence_min',  0.7)
        self.declare_parameter('tracking_states',           'IDLE,LISTENING,SPEAKING')
        self.declare_parameter('return_to_neutral_timeout', 2.0)
        self.declare_parameter('image_width',               1280)
        self.declare_parameter('image_height',              720)
        self.declare_parameter('pan_board',                 0)
        self.declare_parameter('pan_channel',               4)
        self.declare_parameter('tilt_board',                0)
        self.declare_parameter('tilt_channel',              5)
        # Per-axis safe travel envelope and resting gaze, in degrees. These are
        # NARROWER than the mechanical end-stops (pan 50–130, tilt 74–94) on
        # purpose — driving into a stop stalls the servo (BMS-trip risk). Tilt
        # level gaze is 84°, not 90°, so its neutral is offset.
        self.declare_parameter('pan_min_deg',      52.0)
        self.declare_parameter('pan_max_deg',     128.0)
        self.declare_parameter('pan_neutral_deg',  90.0)
        self.declare_parameter('tilt_min_deg',     76.0)
        self.declare_parameter('tilt_max_deg',     92.0)
        self.declare_parameter('tilt_neutral_deg', 84.0)

        self._pan_gain        = self.get_parameter('pan_gain').value
        self._tilt_gain       = self.get_parameter('tilt_gain').value
        self._max_step        = self.get_parameter('max_step').value
        self._conf_min        = self.get_parameter('detection_confidence_min').value
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
            'detection_confidence_min': '_conf_min',
            'return_to_neutral_timeout': '_return_timeout',
        }
        self._deg_params = {
            'pan_min_deg':      '_pan_min',
            'pan_max_deg':      '_pan_max',
            'pan_neutral_deg':  '_pan_neutral',
            'tilt_min_deg':     '_tilt_min',
            'tilt_max_deg':     '_tilt_max',
            'tilt_neutral_deg': '_tilt_neutral',
        }
        self.add_on_set_parameters_callback(self._on_set_parameters)

        # ── State ────────────────────────────────────────────────────────────
        # Servo positions as normalized [0.0, 1.0]; axis neutral is per-axis
        # (pan 90° → 0.5, tilt 84° → 0.467) because tilt's level gaze is offset.
        self._pan_norm         = self._pan_neutral
        self._tilt_norm        = self._tilt_neutral
        self._robot_state      = 'IDLE'
        # Initialise to boot time so the 2-second timeout doesn't fire
        # immediately before any detection arrives.
        self._last_person_time = time.monotonic()
        self._latest_detection = None   # (cx_norm, cy_norm) when person seen

        # ── Publishers ───────────────────────────────────────────────────────
        self._servo_pub = self.create_publisher(Float32MultiArray, '/servo_commands', 10)

        # ── Subscriptions ────────────────────────────────────────────────────
        sensor_qos = QoSPresetProfiles.SENSOR_DATA.value
        self.create_subscription(
            Detection2DArray, '/camera/detections', self._on_detections, sensor_qos
        )
        self.create_subscription(String, '/robot_state', self._on_robot_state, 10)

        # ── Control timer (10 Hz — matches camera detection rate) ────────────
        self.create_timer(0.1, self._control_cb)

        self.get_logger().info(
            f'head_tracking_node started — '
            f'pan_gain={self._pan_gain}, tilt_gain={self._tilt_gain}, '
            f'max_step={self._max_step}, conf_min={self._conf_min}, '
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
                if p.name == 'detection_confidence_min' and not (0.0 <= value <= 1.0):
                    return SetParametersResult(
                        successful=False,
                        reason=f'{p.name} must be in 0.0–1.0, got {value}')
                if p.name in ('max_step', 'return_to_neutral_timeout') and value <= 0.0:
                    return SetParametersResult(
                        successful=False,
                        reason=f'{p.name} must be > 0, got {value}')
                pending.append((self._plain_params[p.name], value, p.name))

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
                    best_cy    = det.bbox.center.position.y / self._img_h

        if best_cx is not None:
            # Clamp to [0,1] — belt-and-suspenders against out-of-frame coords
            self._latest_detection = (
                max(0.0, min(1.0, best_cx)),
                max(0.0, min(1.0, best_cy)),
            )
            self._last_person_time = time.monotonic()

    # ── Control loop (10 Hz) ──────────────────────────────────────────────────

    def _control_cb(self):
        if self._robot_state not in self._tracking_states:
            return

        elapsed = time.monotonic() - self._last_person_time

        if elapsed < self._return_timeout and self._latest_detection is not None:
            cx_norm, cy_norm      = self._latest_detection
            self._latest_detection = None   # consume; next update comes from next msg

            pan_error  = cx_norm - 0.5
            tilt_error = cy_norm - 0.5

            changed = False
            if abs(pan_error) > _DEAD_ZONE:
                delta = max(-self._max_step, min(self._max_step,
                    self._pan_gain * pan_error))
                self._pan_norm = max(self._pan_min, min(self._pan_max,
                    self._pan_norm + delta))
                changed = True
            if abs(tilt_error) > _DEAD_ZONE:
                delta = max(-self._max_step, min(self._max_step,
                    self._tilt_gain * tilt_error))
                self._tilt_norm = max(self._tilt_min, min(self._tilt_max,
                    self._tilt_norm + delta))
                changed = True
            if changed:
                self._publish_servos()

        elif elapsed >= self._return_timeout:
            self._step_toward_neutral()

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

        self._publish_servos()

    def _snap_to_neutral(self):
        """Immediately publish neutral and reset internal state."""
        self._latest_detection = None
        self._pan_norm  = self._pan_neutral
        self._tilt_norm = self._tilt_neutral
        self._publish_servos()

    def _publish_servos(self):
        """Publish pan and tilt angles as a single Float32MultiArray command."""
        pan_deg  = self._pan_norm  * 180.0
        tilt_deg = self._tilt_norm * 180.0

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
