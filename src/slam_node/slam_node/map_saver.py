#!/usr/bin/env python3
"""
Calls the slam_toolbox /slam_toolbox/save_map service to serialise the
current pose graph to disk.

Usage:
  ros2 run slam_node map_saver               # saves to default path
  ros2 run slam_node map_saver my_map_name   # saves to ~/omni_ws/maps/<name>
"""

import os
import sys
import rclpy
from rclpy.node import Node
from slam_toolbox.srv import SaveMap
from std_msgs.msg import String


MAPS_DIR = '/home/pi/omni_ws/maps'
DEFAULT_MAP = os.path.join(MAPS_DIR, 'omni_home_map')


def _resolve(name: str) -> str:
    """A bare name saves under MAPS_DIR. slam_toolbox's SaveMap throws (result
    255) when handed a filename with no parent directory, so never pass one."""
    return name if os.path.isabs(name) else os.path.join(MAPS_DIR, name)


class MapSaverNode(Node):
    def __init__(self, map_name: str):
        super().__init__('map_saver_client')
        self._map_name = map_name
        self._cli = self.create_client(SaveMap, '/slam_toolbox/save_map')

    def save(self) -> bool:
        if not self._cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('slam_toolbox save_map service not available')
            return False
        req = SaveMap.Request()
        req.name = String(data=self._map_name)
        future = self._cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
        if future.result() is None:
            self.get_logger().error('Map save call failed')
            return False
        result = future.result()
        if result.result == 0:
            self.get_logger().info(f'Map saved: {self._map_name}')
            return True
        else:
            self.get_logger().error(f'Map save returned error code {result.result}')
            return False


def main():
    map_name = _resolve(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MAP
    rclpy.init()
    node = MapSaverNode(map_name)
    ok = node.save()
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
