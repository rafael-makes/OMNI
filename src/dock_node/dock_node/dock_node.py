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

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, QoSProfile, qos_profile_sensor_data)

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Range
from std_msgs.msg import Bool, String
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
        self._ctrl = DockController(cfg)
        self._active = False

        self._last_tag = TagObs()
        self._last_tag_t = 0.0
        self._rear_left = (None, 0.0)    # (range_or_None, monotonic_stamp)
        self._rear_right = (None, 0.0)

        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel_raw', 10)
        self._status_pub = self.create_publisher(String, '/dock/status', 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._docked_pub = self.create_publisher(Bool, '/dock/docked', latched)
        self._result_pub = self.create_publisher(String, '/dock/result', latched)

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

    # ── services ───────────────────────────────────────────────────────────────
    def _srv_start(self, req, resp):
        if self._active:
            resp.success = False
            resp.message = 'already docking'
            return resp
        now = time.monotonic()
        self._ctrl.start(now, self._current_tag(now))
        self._active = True
        self._publish_docked(False)
        self.get_logger().info('docking started')
        resp.success = True
        resp.message = 'docking started'
        return resp

    def _srv_cancel(self, req, resp):
        self._stop_and_idle('cancelled by service')
        resp.success = True
        resp.message = 'cancelled'
        return resp

    # ── control loop ─────────────────────────────────────────────────────────
    def _tick(self):
        if not self._active:
            return
        now = time.monotonic()
        tag = self._current_tag(now)
        rear = self._current_rear(now)
        cmd = self._ctrl.update(tag, rear, now)
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
