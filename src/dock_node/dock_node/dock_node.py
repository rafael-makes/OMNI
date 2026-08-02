#!/usr/bin/env python3
"""
dock_node.py — ROS2 wrapper around DockController (the pure control law).

Subscribes:
  /detections      apriltag_msgs/AprilTagDetectionArray  (rear cam, id 0)
  /tof/left_rear   sensor_msgs/Range
  /tof/right_rear  sensor_msgs/Range
Publishes:
  /cmd_vel_raw     geometry_msgs/Twist  (safety-gated; NEVER /cmd_vel direct)
  /dock/status     std_msgs/String      (phase + telemetry, 1/tick)
  /dock/docked     std_msgs/Bool        (latched result)
Services:
  /dock/start      std_srvs/Trigger     begin a docking attempt
  /dock/cancel     std_srvs/Trigger     abort and stop

Design notes:
  * Commands go to /cmd_vel_raw so safety_node keeps e-stop / tilt / watchdog and
    its 75 mm proximity backstop. stop_range (0.10 m) is ABOVE that 75 mm, so a
    normal dock stops before safety would ever trip. If a future dock needs
    electrical contact (<75 mm), add a /dock/active handshake in safety_node like
    /stall_recovery/active — do NOT lower the proximity fault.
  * The 15 Hz loop also satisfies safety's 0.5 s /cmd_vel_raw watchdog while docking.
  * The rear camera may be mirrored — steer_sign is calibrated live. First run at
    crawl speed, hand on e-stop.
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, QoSProfile, qos_profile_sensor_data)
import tf2_ros

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Range
from std_msgs.msg import Bool, Float64, String
from std_srvs.srv import Trigger
from apriltag_msgs.msg import AprilTagDetectionArray

from dock_node.dock_controller import (
    DockCommand, DockConfig, DockController, RearRange, TagObs)


class DockNode(Node):
    def __init__(self):
        super().__init__('dock_node')

        p = self.declare_parameter
        self._target_id = p('target_tag_id', 0).value
        self._img_w = float(p('image_width', 800).value)
        self._img_h = float(p('image_height', 600).value)

        cfg = DockConfig(
            align_tol           = p('align_tol', 0.15).value,
            reverse_correct_tol = p('reverse_correct_tol', 0.30).value,
            pulse_speed         = p('pulse_speed', 1.2).value,
            pulse_dur           = p('pulse_dur', 0.12).value,
            settle_dur          = p('settle_dur', 0.25).value,
            reverse_speed       = p('reverse_speed', 0.18).value,
            stop_range          = p('stop_range', 0.13).value,
            square_engage_range = p('square_engage_range', 0.25).value,
            square_tol          = p('square_tol', 0.02).value,
            square_sign         = float(p('square_sign', 1.0).value),
            orient_tol          = p('orient_tol', 0.12).value,
            t_orient_max        = p('t_orient_max', 20.0).value,
            tag_lost_grace      = p('tag_lost_grace', 0.6).value,
            t_search_max        = p('t_search_max', 20.0).value,
            t_align_max         = p('t_align_max', 30.0).value,
            t_reverse_max       = p('t_reverse_max', 45.0).value,
            t_overall_max       = p('t_overall_max', 120.0).value,
            steer_sign          = float(p('steer_sign', 1.0).value),
        )
        self._cfg = cfg
        self._tag_timeout = p('tag_timeout', 0.3).value
        self._rear_fresh = p('rear_fresh', 0.3).value
        # Undock: drive straight FORWARD off the dock. Nav2 can't plan out (the docked
        # start pose is inside robot_radius of the wall), and this is also the maneuver
        # to pull forward off charger contacts. Closed-loop on map->base_link distance.
        self._undock_distance = p('undock_distance', 0.40).value
        self._undock_speed = p('undock_speed', 0.18).value   # forward, above the 0.16 floor
        self._t_undock_max = p('t_undock_max', 12.0).value
        self._docked_threshold = p('docked_threshold', 0.28).value  # rear ToF ≤ this = "docked"
        self._ctrl = DockController(cfg)
        self._active = False
        self._undocking = False
        self._undock_start = None
        self._undock_t0 = 0.0

        self._last_tag = TagObs()
        self._last_tag_t = 0.0
        self._rear_left = (None, 0.0)    # (range_or_None, monotonic_stamp)
        self._rear_right = (None, 0.0)

        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel_raw', 10)
        self._status_pub = self.create_publisher(String, '/dock/status', 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._docked_pub = self.create_publisher(Bool, '/dock/docked', latched)
        self._result_pub = self.create_publisher(String, '/dock/result', latched)
        self._undock_result_pub = self.create_publisher(String, '/dock/undock_result', latched)

        self.create_subscription(AprilTagDetectionArray, '/detections',
                                 self._on_detections, 10)
        self.create_subscription(Range, '/tof/left_rear',
                                 lambda m: self._on_rear('left', m),
                                 qos_profile_sensor_data)
        self.create_subscription(Range, '/tof/right_rear',
                                 lambda m: self._on_rear('right', m),
                                 qos_profile_sensor_data)

        self.create_service(Trigger, '/dock/start', self._srv_start)
        self.create_service(Trigger, '/dock/cancel', self._srv_cancel)
        self.create_service(Trigger, '/dock/undock', self._srv_undock)

        # Heading source for the ORIENT phase (map->base_link yaw). Nav2 delivers
        # OMNI to the standoff at an arbitrary heading, so before searching we turn
        # to the standoff heading the mission publishes on /dock/orient_target.
        self._tf_buf = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buf, self)
        self._orient_target_deg = None
        self._orient_consumed = True   # one-shot; only a fresh publish enables orient
        latched_in = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Float64, '/dock/orient_target',
                                 self._on_orient_target, latched_in)

        self._timer = self.create_timer(1.0 / 15.0, self._tick)
        self._publish_docked(False)
        self.get_logger().info('dock_node ready — call /dock/start to dock')

    # ── subscriptions ──────────────────────────────────────────────────────────
    def _on_detections(self, msg: AprilTagDetectionArray):
        det = next((d for d in msg.detections if d.id == self._target_id), None)
        if det is None:
            return   # keep last obs; staleness is handled by _tick's timeout
        xs = [c.x for c in det.corners]
        width_px = (max(xs) - min(xs)) if xs else 0.0
        # apriltag corner order: 0 bottom-left, 1 bottom-right, 2 top-right, 3 top-left
        try:
            left_h = abs(det.corners[3].y - det.corners[0].y)
            right_h = abs(det.corners[2].y - det.corners[1].y)
            mean_h = (left_h + right_h) / 2.0 or 1.0
            skew = (right_h - left_h) / mean_h
        except (IndexError, ZeroDivisionError):
            skew = 0.0
        self._last_tag = TagObs(
            seen=True,
            ex=(det.centre.x - self._img_w / 2.0) / (self._img_w / 2.0),
            ey=(det.centre.y - self._img_h / 2.0) / (self._img_h / 2.0),
            size_frac=width_px / self._img_w,
            skew=skew,
        )
        self._last_tag_t = time.monotonic()

    def _on_rear(self, side: str, msg: Range):
        val = msg.range if (msg.min_range <= msg.range <= msg.max_range) else None
        stamp = (val, time.monotonic())
        if side == 'left':
            self._rear_left = stamp
        else:
            self._rear_right = stamp

    def _on_orient_target(self, msg: Float64):
        # The mission publishes the standoff heading (deg, map frame) before docking.
        self._orient_target_deg = float(msg.data)
        self._orient_consumed = False

    def _current_heading(self):
        """map->base_link yaw (rad), or None if TF cannot answer."""
        try:
            tf = self._tf_buf.lookup_transform('map', 'base_link', rclpy.time.Time())
        except Exception:  # noqa: BLE001 — no TF yet is normal
            return None
        q = tf.transform.rotation
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                          1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    # ── services ───────────────────────────────────────────────────────────────
    def _srv_start(self, req, resp):
        if self._active:
            resp.success = False
            resp.message = 'already docking'
            return resp
        now = time.monotonic()
        # TF can be transiently unavailable right when /dock/start fires (rare, but
        # observed 2026-08-02 — OMNI arrived heading-wrong and fell to a blind SEARCH
        # instead of ORIENTing, because a single lookup happened to return None). Try
        # a few times over ~0.5s before giving up — the tick loop already tolerates
        # None heading mid-run, this is just to catch a settling lookup at kickoff.
        heading = None
        for _ in range(5):
            heading = self._current_heading()
            if heading is not None:
                break
            time.sleep(0.1)
        orient_target = None
        want_orient = (not self._orient_consumed
                       and self._orient_target_deg is not None)
        if want_orient and heading is not None:
            orient_target = math.radians(self._orient_target_deg)
            self.get_logger().info(
                f'orient: turning to {self._orient_target_deg:.1f}° before searching '
                f'(current heading {math.degrees(heading):.1f}°)')
        elif want_orient and heading is None:
            # Enter ORIENT anyway — the phase's own tick tolerates a brief TF gap
            # and only falls to SEARCH after t_orient_max. Much better than starting
            # a blind SEARCH when the caller explicitly said which way to turn.
            orient_target = math.radians(self._orient_target_deg)
            self.get_logger().warn(
                f'orient: TF gap at kickoff, entering ORIENT on '
                f'/dock/orient_target={self._orient_target_deg:.1f}° and will pick '
                f'up heading as it comes back')
        elif self._orient_target_deg is None:
            self.get_logger().info(
                'no /dock/orient_target published — searching without pre-orient')
        else:
            self.get_logger().info(
                'orient_target already consumed (no fresh publish this attempt) — '
                'searching without pre-orient')
        self._orient_consumed = True   # one-shot per publish; manual /dock/start won't orient
        self._ctrl.start(now, self._current_tag(now),
                         orient_target=orient_target, heading=heading)
        self._active = True
        self._publish_docked(False)
        self.get_logger().info('docking started')
        resp.success = True
        resp.message = 'docking started'
        return resp

    def _srv_cancel(self, req, resp):
        if self._undocking:
            self._finish_undock(False, 'cancelled')
        self._stop_and_idle('cancelled by service')
        resp.success = True
        resp.message = 'cancelled'
        return resp

    def _srv_undock(self, req, resp):
        """Drive straight forward off the dock so Nav2 can plan from a clear pose
        (and, later, to pull off charger contacts). Outcome on /dock/undock_result."""
        if self._active or self._undocking:
            resp.success = False
            resp.message = 'busy'
            return resp
        pos = self._current_position()
        if pos is None:
            resp.success = False
            resp.message = 'no map->base_link TF — cannot undock'
            return resp
        self._undock_start = pos
        self._undock_t0 = time.monotonic()
        self._undocking = True
        self.get_logger().info(
            f'undocking — driving forward {self._undock_distance:.2f} m off the dock')
        resp.success = True
        resp.message = 'undocking started'
        return resp

    # ── control loop ─────────────────────────────────────────────────────────
    def _tick(self):
        # Publish the LIVE docked state (rear ToF) every tick, so the brain knows
        # whether OMNI is physically on the dock — for greeting/check-in suppression
        # AND to trigger an undock before navigating away.
        self._publish_docked(self._rear_is_docked())
        if self._undocking:
            self._tick_undock()
            return
        if not self._active:
            return
        now = time.monotonic()
        tag = self._current_tag(now)
        rear = self._current_rear(now)
        cmd = self._ctrl.update(tag, rear, now, heading=self._current_heading())
        self._publish_status(cmd, tag, rear)
        if cmd.done or cmd.failed:
            self._finish(cmd)
            return
        tw = Twist()
        tw.linear.x = float(cmd.linear_x)
        tw.angular.z = float(cmd.angular_z)
        self._cmd_pub.publish(tw)

    def _current_tag(self, now: float) -> TagObs:
        if (now - self._last_tag_t) > self._tag_timeout:
            return TagObs(seen=False)
        return self._last_tag

    def _current_rear(self, now: float) -> RearRange:
        lv, lt = self._rear_left
        rv, rt = self._rear_right
        return RearRange(
            left=lv if (now - lt) <= self._rear_fresh else None,
            right=rv if (now - rt) <= self._rear_fresh else None,
        )

    # ── undock ─────────────────────────────────────────────────────────────────
    def _tick_undock(self):
        now = time.monotonic()
        pos = self._current_position()
        if pos is None:
            self._finish_undock(False, 'lost map->base_link TF during undock')
            return
        dist = math.hypot(pos[0] - self._undock_start[0], pos[1] - self._undock_start[1])
        if dist >= self._undock_distance:
            self._finish_undock(True, f'undocked {dist:.2f} m')
            return
        if now - self._undock_t0 > self._t_undock_max:
            self._finish_undock(False, f'undock timeout at {dist:.2f} m')
            return
        tw = Twist()
        tw.linear.x = float(self._undock_speed)   # forward, away from the wall/dock
        self._cmd_pub.publish(tw)

    def _finish_undock(self, ok: bool, msg: str):
        self._undocking = False
        self._cmd_pub.publish(Twist())   # explicit stop
        self._undock_result_pub.publish(
            String(data='undocked' if ok else f'failed: {msg}'))
        if ok:
            self.get_logger().info(f'UNDOCKED — {msg}')
        else:
            self.get_logger().warn(f'undock FAILED — {msg}')

    def _rear_is_docked(self) -> bool:
        rmin = self._current_rear(time.monotonic()).valid_min()
        return rmin is not None and rmin <= self._docked_threshold

    def _current_position(self):
        """(x, y) of base_link in the map frame, or None if TF cannot answer."""
        try:
            tf = self._tf_buf.lookup_transform('map', 'base_link', rclpy.time.Time())
        except Exception:  # noqa: BLE001
            return None
        return (tf.transform.translation.x, tf.transform.translation.y)

    def _finish(self, cmd: DockCommand):
        self._cmd_pub.publish(Twist())   # explicit stop
        self._active = False
        if cmd.done:
            self._publish_docked(True)
            self._publish_result('docked')
            self.get_logger().info(f'DOCKED — {cmd.message}')
        else:
            self._publish_result(f'failed: {cmd.message}')
            self.get_logger().warn(f'docking FAILED — {cmd.message}')

    def _stop_and_idle(self, why: str):
        was_active = self._active
        self._ctrl.cancel()
        self._active = False
        self._cmd_pub.publish(Twist())
        if was_active:
            self._publish_result('cancelled')
        self.get_logger().info(f'docking stopped — {why}')

    # ── publishers ───────────────────────────────────────────────────────────
    def _publish_status(self, cmd: DockCommand, tag: TagObs, rear: RearRange):
        m = String()
        m.data = (f'{cmd.phase.value} | {cmd.message} | '
                  f'ex={tag.ex:+.2f} skew={tag.skew:+.2f} seen={int(tag.seen)} | '
                  f'rearL={_f(rear.left)} rearR={_f(rear.right)} | '
                  f'cmd v={cmd.linear_x:+.3f} w={cmd.angular_z:+.3f}')
        self._status_pub.publish(m)

    def _publish_docked(self, val: bool):
        b = Bool()
        b.data = val
        self._docked_pub.publish(b)

    def _publish_result(self, text: str):
        m = String()
        m.data = text
        self._result_pub.publish(m)


def _f(v):
    return 'n/a' if v is None else f'{v:.3f}'


def main():
    rclpy.init()
    node = DockNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._cmd_pub.publish(Twist())
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
