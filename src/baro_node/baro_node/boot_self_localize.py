#!/usr/bin/env python3
"""boot_self_localize — set /initialpose at boot so no manual Foxglove pose is needed.

Runs once, a few seconds after slam_toolbox activates (localization mode waits for
an /initialpose; it will not self-locate). It decides where OMNI is and seeds it:

  DOCKED   dock tag (id 0 on the main floor) in the rear cam AND the rear ToF at
           docking distance  -> publish the SAVED DOCKED pose. Exact, drift-free.

  NOT DOCKED  -> publish the persisted last-known pose for this floor (pose_writer
           saved it periodically last run). slam_toolbox scan-matches from there;
           its ±0.75 m correlation window snaps a slightly-stale prior to truth.

  NEITHER  -> publish nothing and say so. A manual 2D Pose Estimate is still the
           fallback, but only in the genuinely-unknown case (first boot on a map,
           or moved while powered off), not every boot.

The floor is derived from the resolved --map path (the launch's floor resolver
already picked the map by AprilTag/barometer). The docked/not-docked poses come
from pose_store (floors.yaml dock_pose, and last_pose_floor<N>.yaml).

Capture the docked pose once, on the dock, with the robot localized:

    ros2 run baro_node boot_self_localize --save-dock --map <map_prefix>
"""
import argparse
import math
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
import tf2_ros

from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import Range

from baro_node.pose_store import (
    floor_for_map, read_dock_pose, read_last_pose, write_dock_pose)

try:
    from apriltag_msgs.msg import AprilTagDetectionArray
    HAVE_APRILTAG = True
except ImportError:
    HAVE_APRILTAG = False


def _yaw_to_quat(yaw_deg):
    y = math.radians(yaw_deg)
    return math.sin(y / 2.0), math.cos(y / 2.0)


def _yaw_deg_from_quat(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.degrees(math.atan2(siny, cosy))


class BootLocalizer(Node):
    def __init__(self, args):
        super().__init__('boot_self_localize')
        self._args = args
        self._dock_tag_seen = False
        self._rear_left = None
        self._rear_right = None

        # Latched so slam_toolbox (which may still be finishing activation) still
        # receives the pose even if it subscribes a moment after we publish.
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, args.initialpose_topic, latched)

        if HAVE_APRILTAG:
            self.create_subscription(
                AprilTagDetectionArray, args.tag_topic, self._on_tags, 10)
        self.create_subscription(
            Range, '/tof/left_rear',
            lambda m: self._on_rear('left', m), qos_profile_sensor_data)
        self.create_subscription(
            Range, '/tof/right_rear',
            lambda m: self._on_rear('right', m), qos_profile_sensor_data)

    def _on_tags(self, msg):
        for det in msg.detections:
            if det.id == self._args.dock_tag_id:
                self._dock_tag_seen = True
                return

    def _on_rear(self, side, msg):
        val = msg.range if (msg.min_range <= msg.range <= msg.max_range) else None
        if side == 'left':
            self._rear_left = val
        else:
            self._rear_right = val

    def _rear_min(self):
        vals = [v for v in (self._rear_left, self._rear_right) if v is not None]
        return min(vals) if vals else None

    def is_docked(self):
        rmin = self._rear_min()
        return (self._dock_tag_seen and rmin is not None
                and rmin <= self._args.dock_range)

    def publish_pose(self, pose, tight):
        x, y, yaw_deg = pose
        qz, qw = _yaw_to_quat(yaw_deg)
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        # Docked = confident (tight xy). Persisted prior = let scan-match search xy.
        # But YAW is seeded tight in BOTH cases: a parked robot boots at the heading
        # it was left, so the heading prior is trustworthy — and a LOOSE yaw let
        # scan-match flip the heading ~180° in the dock's symmetric geometry
        # (mislocalized on 2026-08-01). A wrong heading (robot rotated while off) is
        # the rare "moved" case that correctly falls through to a manual estimate.
        var_xy = 0.03 if tight else 0.25
        var_yaw = 0.02
        cov = [0.0] * 36
        cov[0] = var_xy
        cov[7] = var_xy
        cov[35] = var_yaw
        msg.pose.covariance = cov
        self._pose_pub.publish(msg)

    def lookup_map_base(self):
        buf = tf2_ros.Buffer()
        tf2_ros.TransformListener(buf, self)
        deadline = time.time() + 5.0
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            try:
                tf = buf.lookup_transform('map', 'base_link', rclpy.time.Time())
            except Exception:  # noqa: BLE001
                continue
            t = tf.transform.translation
            return [t.x, t.y, _yaw_deg_from_quat(tf.transform.rotation)]
        return None


def main():
    ap = argparse.ArgumentParser(description='Boot-time /initialpose seeder')
    ap.add_argument('--map', default='', help='resolved map prefix (to derive floor)')
    ap.add_argument('--tag-topic', default='/detections')
    ap.add_argument('--dock-tag-id', type=int, default=0,
                    help='dock AprilTag id for this floor (0 = main, 1 = basement)')
    ap.add_argument('--dock-range', type=float, default=0.30,
                    help='rear ToF (m) at/under which OMNI counts as docked. Must be '
                    'ABOVE the docked rest distance WITH MARGIN — measured 0.18-0.23 m '
                    'across docks (varies with lateral offset/angle), so 0.20 was too '
                    'tight and a docked boot fell back to the persisted pose (2026-08-01). '
                    '0.30 catches it; still well below the ~0.6 m standoff so a standoff '
                    'boot is NOT read as docked.')
    ap.add_argument('--sense-time', type=float, default=3.0,
                    help='seconds to gather tag + rear ToF before deciding')
    ap.add_argument('--republish', type=int, default=5,
                    help='times to republish /initialpose (~0.3s apart)')
    ap.add_argument('--initialpose-topic', default='/initialpose')
    ap.add_argument('--save-dock', action='store_true',
                    help='capture map->base_link now as this floor\'s docked pose, then exit')
    args = ap.parse_args()

    floor = floor_for_map(args.map)
    if floor is None:
        print(f'boot_self_localize: map "{args.map}" not in floors.yaml — '
              f'cannot determine floor; leaving pose to manual.', file=sys.stderr)
        # Not a hard failure for the launch; just do nothing.
        return

    rclpy.init()
    node = BootLocalizer(args)

    if args.save_dock:
        pose = node.lookup_map_base()
        if pose is None:
            print('boot_self_localize: no map->base_link TF (is localization up '
                  'and the pose set?) — nothing saved.', file=sys.stderr)
            node.destroy_node(); rclpy.shutdown(); sys.exit(2)
        if write_dock_pose(floor, pose):
            print(f'Saved docked pose for floor {floor}: '
                  f'[{pose[0]:.4f}, {pose[1]:.4f}, {pose[2]:.2f}]')
        else:
            print(f'Floor {floor} not in floors.yaml — cannot save dock pose.',
                  file=sys.stderr)
        node.destroy_node(); rclpy.shutdown()
        return

    # Gather sensor evidence for the docked check.
    deadline = time.time() + args.sense_time
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)

    docked = node.is_docked()
    if docked:
        pose = read_dock_pose(floor)
        source = 'docked (tag + rear ToF)'
        if pose is None:
            print('boot_self_localize: looks docked but no dock_pose saved for '
                  f'floor {floor} — falling back to last-known pose.', file=sys.stderr)
            pose, _ = read_last_pose(floor)
            source = 'last-known (no dock_pose saved)'
    else:
        pose, stamp = read_last_pose(floor)
        age = f', {time.time() - stamp:.0f}s old' if stamp else ''
        source = f'last-known pose{age}'

    if pose is None:
        print('boot_self_localize: no pose to publish (not docked, no persisted '
              'pose for this floor). Set an initial pose manually in Foxglove.',
              file=sys.stderr)
        node.destroy_node(); rclpy.shutdown()
        return

    print(f'boot_self_localize: publishing /initialpose from {source}: '
          f'[{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.1f}°]')
    for _ in range(max(1, args.republish)):
        node.publish_pose(pose, tight=docked)
        end = time.time() + 0.3
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.05)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
