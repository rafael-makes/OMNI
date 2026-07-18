"""frame_server_node.py — serve the newest camera frame as JPEG, on demand.

Runs on the Jetson. head_detector owns the camera exclusively (cv2.VideoCapture),
so this node never touches the device: it subscribes to the clean JPEG feed
head_detector publishes, keeps only the most recent message, and hands it back
over the GetCameraFrame service.

  /camera/image_clean/compressed  (CompressedImage) ──► [cache] ──► GetCameraFrame

The subscribe side is Jetson-local, so it costs nothing on the wire; only the
service reply crosses the 192.168.50.0/24 link to the Pi.

WHY A CACHE RATHER THAN A GRAB
  The Pi asks for a frame in the middle of a spoken conversation, on a ~3 second
  budget. Waiting for a fresh capture would add a round trip to a camera that is
  already busy doing inference. The newest cached frame is at most
  1/clean_image_fps old, which is plenty fresh for "what do you see?".

STALENESS
  A cached frame outlives the publisher — if head_detector dies, this node would
  happily serve a minute-old picture and OMNI would describe a room it is no
  longer in. Frames older than max_frame_age are refused instead.
"""
from __future__ import annotations

import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import CompressedImage

from omni_vision_msgs.srv import GetCameraFrame

DEFAULT_CAMERA = 'head'


class FrameServerNode(Node):

    def __init__(self):
        super().__init__('frame_server')

        # camera_id -> topic. One entry today; the rear IMX219 used for dock
        # approach is the obvious second, hence the id in the request rather
        # than a single hardcoded camera.
        self.declare_parameter('head_topic', '/camera/image_clean/compressed')
        # Refuse anything older than this. Slightly generous relative to the
        # default 2 Hz publish rate so a single dropped frame is not an error.
        self.declare_parameter('max_frame_age', 5.0)

        head_topic = str(self.get_parameter('head_topic').value)
        self._max_age = float(self.get_parameter('max_frame_age').value)

        # Newest frame per camera, under a lock: the subscription callback and the
        # service callback are both on the executor, but a MultiThreadedExecutor
        # (or a future one) would let them overlap.
        self._lock = threading.Lock()
        self._frames: dict[str, CompressedImage] = {}

        # Must match head_detector's publisher QoS (BEST_EFFORT / VOLATILE) or the
        # subscription silently never matches and every request returns "no frame".
        sensor_qos = QoSProfile(depth=1,
                                reliability=ReliabilityPolicy.BEST_EFFORT,
                                durability=DurabilityPolicy.VOLATILE)

        self._topics = {DEFAULT_CAMERA: head_topic}
        for camera_id, topic in self._topics.items():
            self.create_subscription(
                CompressedImage, topic,
                lambda msg, cid=camera_id: self._on_frame(cid, msg),
                sensor_qos)
            self.get_logger().info(f"frame_server: camera '{camera_id}' <- {topic}")

        self._srv = self.create_service(
            GetCameraFrame, '/vision/get_camera_frame', self._on_request)

        self._served = 0
        self.get_logger().info(
            f'frame_server ready on /vision/get_camera_frame '
            f'(cameras: {", ".join(sorted(self._topics))}, max_frame_age={self._max_age}s)')

    def _on_frame(self, camera_id: str, msg: CompressedImage) -> None:
        with self._lock:
            self._frames[camera_id] = msg

    def _age_seconds(self, msg: CompressedImage) -> float:
        stamp = msg.header.stamp
        msg_ns = stamp.sec * 10**9 + stamp.nanosec
        return (self.get_clock().now().nanoseconds - msg_ns) / 1e9

    def _on_request(self, request, response):
        camera_id = (request.camera_id or '').strip().lower() or DEFAULT_CAMERA

        if camera_id not in self._topics:
            known = ', '.join(sorted(self._topics))
            response.success = False
            response.message = f"unknown camera_id '{camera_id}' (known: {known})"
            self.get_logger().warn(f'frame_server: {response.message}')
            return response

        with self._lock:
            msg = self._frames.get(camera_id)

        if msg is None:
            response.success = False
            response.message = (
                f"no frame received yet on {self._topics[camera_id]} — is head_detector "
                f"running with publish_clean_image:=true?")
            self.get_logger().warn(f'frame_server: {response.message}')
            return response

        age = self._age_seconds(msg)
        if age > self._max_age:
            response.success = False
            response.age_seconds = float(age)
            response.message = (
                f'newest frame is {age:.1f}s old (limit {self._max_age}s) — the camera '
                f'feed has stopped')
            self.get_logger().warn(f'frame_server: {response.message}')
            return response

        response.success = True
        response.jpeg = msg.data
        response.stamp = msg.header.stamp
        response.age_seconds = float(age)
        # Dimensions would need a JPEG decode to read; not worth the CPU on this
        # path, and the caller does not need them. 0 documented as "unknown".
        response.width = 0
        response.height = 0
        response.message = f'ok ({len(msg.data)} bytes, {age:.2f}s old)'

        self._served += 1
        self.get_logger().info(
            f"frame_server: served '{camera_id}' "
            f"({len(msg.data) / 1024:.0f} KB, age {age:.2f}s, total {self._served})")
        return response


def main(args=None):
    rclpy.init(args=args)
    node = FrameServerNode()
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
