"""Boot-time floor resolver: AprilTag first, barometer fallback.

Priority:
  1. Look for a dock AprilTag for a few seconds (apriltag_ros ->
     apriltag_msgs/AprilTagDetectionArray on /detections). Tag id -> floor is
     configurable; default: id 0 = floor 0 (main), id 1 = floor -1 (basement).
     A tag is AUTHORITATIVE (absolute, no weather drift) and also re-anchors the
     barometer by publishing the floor to /baro/set_floor.
  2. If no known tag is seen before the timeout, fall back to the barometer:
     read the BMP280 and pick the nearest calibrated floor (floors.yaml).

Then resolve that floor's map and, with --launch, hand off to the localization
launch:

    ros2 run baro_node baro_floor_resolver --launch

Dry run (prints chosen map path only):

    ros2 run baro_node baro_floor_resolver --quiet

Until apriltag_msgs is built and apriltag_ros is publishing, step 1 is skipped
automatically and this behaves exactly like baro_select_map.
"""
import argparse
import os
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import Int32

from baro_node.floors import load_floors
from baro_node.select_map import read_pressure, choose

try:
    from apriltag_msgs.msg import AprilTagDetectionArray
    HAVE_APRILTAG = True
except ImportError:
    HAVE_APRILTAG = False


def parse_tag_floors(spec):
    """'0:0,1:-1' -> {0: 0, 1: -1}  (tag id -> floor index)."""
    out = {}
    for pair in spec.split(','):
        pair = pair.strip()
        if not pair:
            continue
        tag, floor = pair.split(':')
        out[int(tag)] = int(floor)
    return out


class FloorResolver(Node):
    def __init__(self, tag_topic, tag_floors):
        super().__init__('floor_resolver')
        self.tag_floors = tag_floors
        self.detected_floor = None
        self.detected_tag = None
        # Latched: a late-matching baro_node subscription still gets the anchor.
        anchor_qos = QoSProfile(depth=1)
        anchor_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.set_floor_pub = self.create_publisher(Int32, '/baro/set_floor', anchor_qos)
        if HAVE_APRILTAG:
            self.create_subscription(
                AprilTagDetectionArray, tag_topic, self.on_tags, 10)

    def on_tags(self, msg):
        for det in msg.detections:
            if det.id in self.tag_floors:
                self.detected_tag = det.id
                self.detected_floor = self.tag_floors[det.id]
                return


def map_for_floor(floors, floor_index):
    return next((f for f in floors if f['floor'] == floor_index), None)


def main():
    ap = argparse.ArgumentParser(description='AprilTag-first / baro-fallback floor resolver')
    ap.add_argument('--tag-topic', default='/detections')
    ap.add_argument('--tag-floors', default='0:0,1:-1',
                    help='tag_id:floor pairs, e.g. "0:0,1:-1"')
    ap.add_argument('--tag-timeout', type=float, default=4.0,
                    help='seconds to wait for a dock tag before using the baro')
    ap.add_argument('--seconds', type=float, default=3.0, help='baro sampling duration')
    ap.add_argument('--bus', type=int, default=1)
    ap.add_argument('--addr', type=lambda x: int(x, 0), default=0x77)
    ap.add_argument('--launch', action='store_true',
                    help='exec the localization launch with the chosen map')
    ap.add_argument('--quiet', action='store_true', help='print only the map path')
    args = ap.parse_args()

    tag_floors = parse_tag_floors(args.tag_floors)
    floors = load_floors()
    if not floors:
        print('No calibrated floors — run baro_calibrate first.', file=sys.stderr)
        sys.exit(2)

    rclpy.init()
    node = FloorResolver(args.tag_topic, tag_floors)

    def log(m):
        if not args.quiet:
            print(m)

    if HAVE_APRILTAG:
        log(f'Looking for dock tag on {args.tag_topic} '
            f'({args.tag_floors}) for up to {args.tag_timeout:.0f}s...')
        deadline = time.time() + args.tag_timeout
        while time.time() < deadline and node.detected_floor is None:
            rclpy.spin_once(node, timeout_sec=0.1)
    else:
        log('apriltag_msgs not installed — skipping tag check, using baro.')

    if node.detected_floor is not None:
        floor = node.detected_floor
        source = f'AprilTag id {node.detected_tag}'
        # Authoritative: re-anchor the barometer to this known floor. Wait briefly
        # for baro_node's subscription to match (pub/sub discovery), else the
        # message is published into the void and lost. Skip if baro_node isn't up.
        deadline = time.time() + 2.0
        while (time.time() < deadline
               and node.set_floor_pub.get_subscription_count() == 0):
            rclpy.spin_once(node, timeout_sec=0.05)
        if node.set_floor_pub.get_subscription_count() > 0:
            node.set_floor_pub.publish(Int32(data=int(floor)))
            for _ in range(10):
                rclpy.spin_once(node, timeout_sec=0.02)
            log(f'  anchored barometer to floor {floor} via /baro/set_floor')
        else:
            log('  baro_node not subscribed — skipped baro anchoring')
        entry = map_for_floor(floors, floor)
    else:
        if HAVE_APRILTAG:
            log('No dock tag seen — falling back to barometer.')
        pressure = read_pressure(args.bus, args.addr, args.seconds)
        entry, margin = choose(floors, pressure)
        floor = entry['floor']
        source = f'baro {pressure:.0f} Pa (margin {margin:.0f} Pa)'
        if margin < 20.0:
            print('  WARNING: low baro margin — floor is ambiguous (weather drift?).',
                  file=sys.stderr)

    node.destroy_node()
    rclpy.shutdown()

    if entry is None:
        print(f'Resolved floor {floor} ({source}) but it is not in floors.yaml.',
              file=sys.stderr)
        sys.exit(3)

    log(f'Resolved floor {floor} ("{entry["name"]}") via {source}')
    map_path = entry.get('map')
    if not map_path:
        print(f'Floor {floor} has no map path in floors.yaml.', file=sys.stderr)
        sys.exit(3)

    if args.launch:
        cmd = ['ros2', 'launch', 'slam_node', 'localization_launch.py',
               f'map_file:={map_path}']
        log(f'Launching: {" ".join(cmd)}')
        os.execvp('ros2', cmd)
    else:
        print(map_path)


if __name__ == '__main__':
    main()
