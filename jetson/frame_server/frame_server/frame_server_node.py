"""frame_server_node.py — serve the newest camera frame(s) as JPEG, on demand.

Runs on the Jetson. Each camera has exactly ONE owner process that holds the
V4L2 device (head_detector for the head cam, rear_camera for the rear cam), so
this node never touches a device: it subscribes to the clean JPEG feeds those
nodes publish, keeps only the most recent message per camera, and hands them
back over the GetCameraFrame service.

  /camera/image_clean/compressed       ──┐
                                         ├─► [cache] ──► GetCameraFrame
  /camera/rear/image_clean/compressed  ──┘

The subscribe side is Jetson-local, so it costs nothing on the wire; only the
service reply crosses the 192.168.50.0/24 link to the Pi.

MULTI-CAMERA
  camera_id is "head", "rear", or "all". "all" returns every camera that has a
  live frame in ONE reply, which is the point — behavior_node's fusion prompt
  describes the room as a single scene, and two sequential fetches would not be
  simultaneous views. Order is stable (head first) so the fusion prompt can name
  the views, though each frame also carries its own camera_id.

  A partial "all" is a SUCCESS, not a failure: if the rear feed is down, OMNI
  describing only what is in front of it beats OMNI saying it cannot see. The
  reply's message names which cameras were missing so the caller can say so.

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

from omni_vision_msgs.msg import CameraFrame
from omni_vision_msgs.srv import GetCameraFrame

DEFAULT_CAMERA = 'head'
ALL_CAMERAS = 'all'

# Stable, meaningful order: head first. behavior_node's fusion prompt introduces
# the views in this order ("head camera ... rear camera"), and a caller that
# ignores camera_id and just takes frames[0] gets the forward view, which is the
# sane default. Cameras absent from the cache are skipped, not reordered.
CAMERA_ORDER = ('head', 'rear')


class FrameServerNode(Node):

    def __init__(self):
        super().__init__('frame_server')

        # camera_id -> topic. Both feeds come from whichever node owns that
        # device; this one only ever subscribes.
        self.declare_parameter('head_topic', '/camera/image_clean/compressed')
        self.declare_parameter('rear_topic', '/camera/rear/image_clean/compressed')
        # Set false to run head-only (e.g. rear_camera is not launched). The
        # subscription is harmless either way, but this keeps 'rear' and 'all'
        # honestly reporting "not configured" rather than "no frame yet".
        self.declare_parameter('rear_enabled', True)
        # Refuse anything older than this. Slightly generous relative to the
        # default 2 Hz publish rate so a single dropped frame is not an error.
        self.declare_parameter('max_frame_age', 5.0)

        head_topic = str(self.get_parameter('head_topic').value)
        rear_topic = str(self.get_parameter('rear_topic').value)
        rear_enabled = bool(self.get_parameter('rear_enabled').value)
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
        if rear_enabled:
            self._topics['rear'] = rear_topic
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

    def _owner_hint(self, camera_id: str) -> str:
        """Which node to go look at when a feed is missing.

        Worth spelling out in the error: the two feeds have different owners and
        different enabling flags, and 'no frame' otherwise sends you to the wrong
        node entirely.
        """
        if camera_id == 'rear':
            return 'is rear_camera running with publish_clean_image:=true?'
        return 'is head_detector running with publish_clean_image:=true?'

    def _take(self, camera_id: str):
        """Newest usable frame for one camera -> (CameraFrame, age) or (None, why)."""
        with self._lock:
            msg = self._frames.get(camera_id)

        if msg is None:
            return None, f'no frame received yet on {self._topics[camera_id]} — ' \
                         f'{self._owner_hint(camera_id)}'

        age = self._age_seconds(msg)
        if age > self._max_age:
            # A cache outlives its publisher. Serving a minute-old picture would
            # have OMNI confidently describing a room it has already left.
            return None, (f'newest {camera_id} frame is {age:.1f}s old '
                          f'(limit {self._max_age}s) — that feed has stopped')

        frame = CameraFrame()
        frame.camera_id = camera_id
        frame.jpeg = msg.data
        frame.stamp = msg.header.stamp
        frame.age_seconds = float(age)
        # Dimensions would need a JPEG decode to read; not worth the CPU on this
        # path, and the caller does not need them. 0 documented as "unknown".
        frame.width = 0
        frame.height = 0
        return frame, age

    def _on_request(self, request, response):
        camera_id = (request.camera_id or '').strip().lower() or DEFAULT_CAMERA

        if camera_id == ALL_CAMERAS:
            wanted = [c for c in CAMERA_ORDER if c in self._topics]
        elif camera_id in self._topics:
            wanted = [camera_id]
        else:
            known = ', '.join(sorted(self._topics) + [ALL_CAMERAS])
            response.success = False
            response.message = f"unknown camera_id '{camera_id}' (known: {known})"
            self.get_logger().warn(f'frame_server: {response.message}')
            return response

        frames = []
        problems = []
        for cid in wanted:
            frame, detail = self._take(cid)
            if frame is None:
                problems.append(str(detail))
            else:
                frames.append(frame)

        if not frames:
            response.success = False
            response.message = '; '.join(problems) or 'no cameras configured'
            self.get_logger().warn(f'frame_server: {response.message}')
            return response

        response.success = True
        response.frames = frames

        # Legacy mirror. A Pi built against the single-frame version of this
        # service reads these and nothing else, so they must always be populated
        # — and always from frames[0], the forward view.
        first = frames[0]
        response.jpeg = first.jpeg
        response.stamp = first.stamp
        response.age_seconds = first.age_seconds
        response.width = first.width
        response.height = first.height

        served = ', '.join(f'{f.camera_id} {len(f.jpeg) / 1024:.0f}KB '
                           f'{f.age_seconds:.2f}s' for f in frames)
        # A partial 'all' still succeeds; say so plainly so the caller can decide
        # whether to mention the missing view out loud.
        response.message = f'ok ({served})'
        if problems:
            response.message += f' [missing: {"; ".join(problems)}]'

        self._served += 1
        self.get_logger().info(
            f"frame_server: served '{camera_id}' -> {served} (total {self._served})"
            + (f'  MISSING: {"; ".join(problems)}' if problems else ''))
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
