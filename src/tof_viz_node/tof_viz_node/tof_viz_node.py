"""
tof_viz_node — publishes ToF sensor readings as a MarkerArray for Foxglove.

Subscribes to the 6 individual /tof/* Range topics and emits a single
/tof/markers MarkerArray. Each sensor is rendered as an arrow that:
  - Originates at the sensor's TF frame
  - Points forward (range direction)
  - Scales in length with the measured distance
  - Changes colour: green (far) → yellow → red (close)

Add /tof/markers as a single topic in the Foxglove 3D panel — no labels,
no clutter, just 6 coloured distance arrows on the robot.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import Range
from visualization_msgs.msg import Marker, MarkerArray


# Maps topic → (marker id, frame_id)
TOF_SENSORS = {
    '/tof/front_left':  (0, 'tof_front_left'),
    '/tof/front_right': (1, 'tof_front_right'),
    '/tof/left':        (2, 'tof_left'),
    '/tof/right':       (3, 'tof_right'),
    '/tof/left_rear':   (4, 'tof_left_rear'),
    '/tof/right_rear':  (5, 'tof_right_rear'),
}

# Colour thresholds (metres)
CLOSE_M  = 0.20   # red
MEDIUM_M = 0.50   # yellow
# beyond MEDIUM_M → green

ARROW_SHAFT_DIAMETER = 0.03   # metres
ARROW_HEAD_DIAMETER  = 0.06
ARROW_HEAD_LENGTH    = 0.05


def _colour(distance: float) -> tuple[float, float, float]:
    """Return (r, g, b) interpolated from green→yellow→red."""
    if distance <= CLOSE_M:
        return (1.0, 0.0, 0.0)
    if distance >= MEDIUM_M:
        return (0.0, 1.0, 0.0)
    t = (distance - CLOSE_M) / (MEDIUM_M - CLOSE_M)  # 0=close, 1=far
    return (1.0 - t, t, 0.0)


class TofVizNode(Node):

    def __init__(self):
        super().__init__('tof_viz_node')

        self._ranges: dict[int, tuple[float, str]] = {}  # id → (range, frame_id)

        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        for topic, (marker_id, frame_id) in TOF_SENSORS.items():
            self.create_subscription(
                Range, topic,
                lambda msg, mid=marker_id, fid=frame_id: self._range_cb(msg, mid, fid),
                sensor_qos,
            )

        self._pub = self.create_publisher(MarkerArray, '/tof/markers', 10)
        self.create_timer(0.1, self._publish_markers)  # 10 Hz

        self.get_logger().info('tof_viz_node started — publishing /tof/markers')

    def _range_cb(self, msg: Range, marker_id: int, frame_id: str):
        self._ranges[marker_id] = (msg.range, frame_id, msg.header.stamp)

    def _publish_markers(self):
        if not self._ranges:
            return

        array = MarkerArray()

        for marker_id, data in self._ranges.items():
            distance, frame_id, stamp = data

            r, g, b = _colour(distance)

            m = Marker()
            m.header.frame_id = frame_id
            m.header.stamp = stamp
            m.ns = 'tof'
            m.id = marker_id
            m.type = Marker.ARROW
            m.action = Marker.ADD

            # Arrow starts at sensor origin, points along +X (sensor forward axis)
            m.pose.orientation.w = 1.0

            # Scale: x = total length, y/z = shaft/head diameters
            m.scale.x = max(0.01, float(distance))
            m.scale.y = ARROW_SHAFT_DIAMETER
            m.scale.z = ARROW_HEAD_DIAMETER

            m.color.r = r
            m.color.g = g
            m.color.b = b
            m.color.a = 0.85

            m.lifetime.sec = 1  # auto-hide if sensor stops publishing

            array.markers.append(m)

        self._pub.publish(array)


def main(args=None):
    rclpy.init(args=args)
    node = TofVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
