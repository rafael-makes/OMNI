#!/usr/bin/env python3
"""pose_writer — persist map->base_link periodically for boot self-localization.

Long-running node. Every `write_period` seconds it looks up the map->base_link
transform and writes [x, y, yaw_deg] to ~/omni_ws/maps/last_pose_floor<N>.yaml
(see pose_store). At the next boot, boot_self_localize reads it back as the
/initialpose prior and slam_toolbox scan-matches from there — no manual Foxglove
2D Pose Estimate needed.

WHY PERIODIC, NOT JUST ON SHUTDOWN
  OMNI browns out and hard-reboots (see feedback_pi_network_dropouts) — there is
  often no clean SIGTERM, so a shutdown-only write would routinely lose the pose.
  Periodic writing means the worst case is `write_period` of staleness, which the
  ±0.75 m scan-match window on the next boot absorbs easily.

WHY start_delay
  Right after boot, localization has not converged (it is waiting for the boot
  /initialpose and the first scan matches). Persisting during that window would
  save a wrong pose. So writing does not begin until start_delay has elapsed AND
  a transform is actually available.

The floor is derived once from the resolved `map_file` param (the map frame is
per-floor; OMNI does not change floors at runtime in this system). If the map is
not in floors.yaml the node logs and idles — better than keying a bad file.
"""
import math

import rclpy
from rclpy.node import Node
import tf2_ros

from baro_node.pose_store import floor_for_map, write_last_pose


def _yaw_deg(q):
    """Yaw in degrees from a geometry_msgs quaternion (planar-safe full formula)."""
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.degrees(math.atan2(siny, cosy))


class PoseWriter(Node):
    def __init__(self):
        super().__init__('pose_writer')
        p = self.declare_parameter
        self._map_file    = str(p('map_file', '').value)
        self._write_period = float(p('write_period', 5.0).value)
        self._start_delay  = float(p('start_delay', 20.0).value)
        self._map_frame    = str(p('map_frame', 'map').value)
        self._base_frame   = str(p('base_frame', 'base_link').value)

        self._floor = floor_for_map(self._map_file)
        if self._floor is None:
            self.get_logger().warn(
                f'pose_writer: map "{self._map_file}" not found in floors.yaml — '
                f'not persisting pose (nothing to key the file by).')
            return

        self._tf_buf = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buf, self)
        self._started = self.get_clock().now()
        self._last_pose = None
        self._timer = self.create_timer(self._write_period, self._tick)
        self.get_logger().info(
            f'pose_writer: persisting floor {self._floor} pose every '
            f'{self._write_period:.0f}s after a {self._start_delay:.0f}s settle.')

    def _current_pose(self):
        try:
            tf = self._tf_buf.lookup_transform(
                self._map_frame, self._base_frame, rclpy.time.Time())
        except Exception:  # noqa: BLE001 — TF not ready yet is normal early on
            return None
        t = tf.transform.translation
        return [t.x, t.y, _yaw_deg(tf.transform.rotation)]

    def _tick(self):
        elapsed = (self.get_clock().now() - self._started).nanoseconds / 1e9
        if elapsed < self._start_delay:
            return
        pose = self._current_pose()
        if pose is None:
            return
        self._last_pose = pose
        try:
            write_last_pose(self._floor, pose)
        except Exception as exc:  # noqa: BLE001 — persistence must never kill the node
            self.get_logger().warn(f'pose_writer: write failed: {exc}')

    def final_write(self):
        """Best-effort last write on a clean shutdown."""
        if self._floor is None:
            return
        pose = self._current_pose() or self._last_pose
        if pose is not None:
            try:
                write_last_pose(self._floor, pose)
                self.get_logger().info(
                    f'pose_writer: final pose saved '
                    f'({pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.1f}°).')
            except Exception:  # noqa: BLE001
                pass


def main():
    rclpy.init()
    node = PoseWriter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.final_write()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
