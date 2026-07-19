"""rear_camera_node.py — sole owner of OMNI's rear (2K HD) USB camera.

Runs on the Jetson. Mirrors the head camera's architecture exactly: ONE node
opens the V4L2 device and everything else consumes its topics.

  rear cam ──► [this node] ──┬─► /camera/rear/image_clean/compressed  (frame_server)
                             └─► /camera/rear/image_raw               (apriltag_ros)

WHY THIS NODE EXISTS AT ALL
  The rear cam had two would-be owners: dock_apriltag's v4l2_camera (YUYV 800x600,
  for AprilTag floor detection) and, now, rear-view scene description (MJPG, for
  "what's behind you?"). A V4L2 device serves ONE process in ONE format — the
  second opener gets -EBUSY. Rather than making the two features mutually
  exclusive, this node owns the device and publishes for both. Same pattern as
  head_detector, which owns the head cam and feeds frame_server.

CAMERA QUIRKS, MEASURED 2026-07-18 (do not "fix" these without re-measuring)
  - This cam advertises ONLY 30fps intervals for MJPG. Requesting framerate=5/1
    does not give you 5fps, it fails caps negotiation and the device never opens
    ("Internal data stream error", VideoCapture returns False). This is NOT a
    bandwidth problem and dropping the resolution does not help — 320x240 @ 5fps
    fails identically. Ask for 30 on the wire and throttle downstream. That is
    what decode_fps does.
  - It delivers ~25fps, not the 30 it advertises. Measured 24.99 fps alone and
    25.00 fps with the head cam and the OAK-D both streaming, so that ceiling is
    the camera, not the USB bus.
  - All three USB devices share one 480 Mbps bus (see lsusb -t). At MJPG this is
    a non-issue: 640x480 MJPG is roughly 6 Mbps. Keep MJPG. Raw YUYV at this size
    would be ~147 Mbps and would put the bus genuinely at risk.

DECODE THROTTLING
  The wire rate is fixed at ~25fps, so decoding every frame is pure waste when the
  only consumer is a 2fps cache. videorate drops buffers BEFORE jpegdec, so
  decode_fps controls CPU, not bus traffic (the bus cost is already paid). The
  original "rear at 5fps" budget lives here — it is the only place this camera
  can honour it.
"""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import CameraInfo, CompressedImage, Image

# Stable by-id path. NEVER use /dev/videoN here: the head cam and this one swap
# node numbers across reboots, and pointing the dock detector at the head camera
# is a genuinely confusing failure to debug.
DEFAULT_DEVICE = ('/dev/v4l/by-id/'
                  'usb-GENERAL_2K_HD_Camera-video-index0')


class RearCameraNode(Node):

    def __init__(self):
        super().__init__('rear_camera')

        self.declare_parameter('device', DEFAULT_DEVICE)
        self.declare_parameter('capture_width', 640)
        self.declare_parameter('capture_height', 480)
        # 30 is the ONLY MJPG rate this camera will negotiate. See the module
        # docstring — lowering this does not lower it, it breaks the open.
        self.declare_parameter('capture_fps', 30)
        # Post-capture decode rate. This is the real "rear cam runs at 5fps" knob.
        self.declare_parameter('decode_fps', 5.0)

        # Clean JPEG feed for frame_server -> describe_scene(direction='behind').
        self.declare_parameter('publish_clean_image', True)
        self.declare_parameter('clean_image_fps', 2.0)
        self.declare_parameter('clean_jpeg_quality', 75)
        self.declare_parameter('clean_max_width', 640)

        # Raw feed for apriltag_ros. OFF by default: docking is an on-demand
        # behaviour, and publishing 5fps of bgr8 around the clock for nobody is
        # wasted CPU and DDS traffic. dock_apriltag_shared.launch.py turns it on.
        self.declare_parameter('publish_raw_image', False)
        self.declare_parameter('raw_image_fps', 5.0)

        self.declare_parameter('frame_id', 'dock_cam')
        # How long a dead capture may go unnoticed before we tear down and reopen.
        self.declare_parameter('reopen_after', 3.0)

        self._device = str(self.get_parameter('device').value)
        self._cap_w = int(self.get_parameter('capture_width').value)
        self._cap_h = int(self.get_parameter('capture_height').value)
        self._cap_fps = int(self.get_parameter('capture_fps').value)
        self._decode_fps = float(self.get_parameter('decode_fps').value)

        self._pub_clean = bool(self.get_parameter('publish_clean_image').value)
        self._clean_fps = float(self.get_parameter('clean_image_fps').value)
        self._clean_q = int(self.get_parameter('clean_jpeg_quality').value)
        self._clean_max_w = int(self.get_parameter('clean_max_width').value)

        self._pub_raw = bool(self.get_parameter('publish_raw_image').value)
        self._raw_fps = float(self.get_parameter('raw_image_fps').value)

        self._frame_id = str(self.get_parameter('frame_id').value)
        self._reopen_after = float(self.get_parameter('reopen_after').value)

        # THE TWO FEEDS NEED DIFFERENT QoS. This is not a style choice — get it
        # wrong and the subscriber silently never matches, with no error anywhere.
        #
        # clean: BEST_EFFORT/VOLATILE, matching head_detector's clean-image
        #   publisher, because frame_server subscribes BEST_EFFORT. A RELIABLE
        #   publisher here makes every 'rear' request return "no frame yet".
        clean_qos = QoSProfile(depth=1,
                               reliability=ReliabilityPolicy.BEST_EFFORT,
                               durability=DurabilityPolicy.VOLATILE)
        # raw: RELIABLE, because apriltag_ros's image_transport subscriber uses
        #   the default RELIABLE profile. Measured 2026-07-18 with
        #   `ros2 topic info -v`: BEST_EFFORT pub + RELIABLE sub is an INCOMPATIBLE
        #   pair, so apriltag received zero frames and docking broke silently —
        #   the topic published happily at 5Hz the whole time. Do not "simplify"
        #   these back into one profile.
        raw_qos = QoSProfile(depth=5,
                             reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.VOLATILE)

        self._clean_pub = None
        if self._pub_clean:
            self._clean_pub = self.create_publisher(
                CompressedImage, '/camera/rear/image_clean/compressed', clean_qos)

        self._raw_pub = None
        self._info_pub = None
        if self._pub_raw:
            self._raw_pub = self.create_publisher(
                Image, '/camera/rear/image_raw', raw_qos)
            self._info_pub = self.create_publisher(
                CameraInfo, '/camera/rear/camera_info', raw_qos)

        self._cap = None
        self._frames = 0
        self._last_frame_at = 0.0
        self._last_clean_at = 0.0
        self._last_raw_at = 0.0
        self._stop = threading.Event()

        # Capture runs on its own thread, not a ROS timer. cap.read() blocks until
        # the next frame arrives; on a single-threaded executor that would stall
        # every other callback in this node for the frame interval.
        self._thread = threading.Thread(
            target=self._capture_loop, name='rear-capture', daemon=True)
        self._thread.start()

        self.get_logger().info(
            f'rear_camera: {self._device} {self._cap_w}x{self._cap_h} MJPG '
            f'@{self._cap_fps} wire / {self._decode_fps} decoded  '
            f'(clean={self._pub_clean}@{self._clean_fps}Hz, '
            f'raw={self._pub_raw}@{self._raw_fps}Hz)')

    # ── Frame source ──────────────────────────────────────────────────────────
    def _pipeline(self) -> str:
        """MJPG off the wire, decimated, then decoded once.

        videorate sits BEFORE jpegdec on purpose: dropping compressed buffers is
        nearly free, whereas decoding then discarding is the cost we are avoiding.
        drop=1/max-buffers=1 keeps the cache newest-frame rather than a backlog.
        """
        rate = ''
        if self._decode_fps > 0:
            # Fraction, not float: GStreamer caps framerate is a fraction type.
            num = max(1, int(round(self._decode_fps)))
            rate = f'videorate ! image/jpeg,framerate={num}/1 ! '
        return (f'v4l2src device={self._device} ! '
                f'image/jpeg,width={self._cap_w},height={self._cap_h},'
                f'framerate={self._cap_fps}/1 ! '
                f'{rate}'
                f'jpegdec ! videoconvert ! video/x-raw,format=BGR ! '
                f'appsink drop=1 max-buffers=1 sync=false')

    def _open(self):
        cap = cv2.VideoCapture(self._pipeline(), cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            cap.release()
            return None
        return cap

    def _capture_loop(self) -> None:
        while not self._stop.is_set():
            if self._cap is None:
                self._cap = self._open()
                if self._cap is None:
                    self.get_logger().warn(
                        f'rear_camera: cannot open {self._device} — retrying in 2s. '
                        f'If this persists, check nothing else holds the device '
                        f'(fuser -v {self._device}); dock_apriltag.launch.py opens '
                        f'it directly and must not run alongside this node.',
                        throttle_duration_sec=30.0)
                    self._stop.wait(2.0)
                    continue
                self.get_logger().info(f'rear_camera: opened {self._device}')
                self._last_frame_at = time.monotonic()

            ok, frame = self._cap.read()
            now = time.monotonic()

            if not ok or frame is None:
                # A camera yanked mid-stream returns False forever. Reopening is
                # the only recovery; without this the node stays alive and silent.
                if now - self._last_frame_at > self._reopen_after:
                    self.get_logger().warn(
                        'rear_camera: capture stalled — reopening device')
                    self._cap.release()
                    self._cap = None
                else:
                    self._stop.wait(0.01)
                continue

            self._frames += 1
            self._last_frame_at = now
            self._publish(frame, now)

        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ── Publishing ────────────────────────────────────────────────────────────
    def _due(self, deadline: float, now: float, interval: float) -> tuple[bool, float]:
        """Deadline-based throttle: is this frame due, and when is the next one?

        The naive form (`if now - last >= interval: last = now`) QUANTIZES to the
        input rate: with frames arriving every 0.2s and a 0.5s interval, the first
        frame at-or-after 0.5s lands at 0.6s, and 2Hz silently becomes 1.67Hz.
        Advancing the deadline by exactly one interval instead lets the emission
        alternate 0.4/0.6 and average the rate actually asked for.

        The catch-up guard matters when the source stalls: without it a 10s gap
        would leave the deadline far in the past and dump a burst of frames.
        """
        if now < deadline:
            return False, deadline
        nxt = deadline + interval
        if nxt < now:
            nxt = now + interval
        return True, nxt

    def _publish(self, frame: np.ndarray, now: float) -> None:
        stamp = self.get_clock().now().to_msg()

        if self._clean_pub is not None and self._clean_fps > 0:
            due, self._last_clean_at = self._due(
                self._last_clean_at, now, 1.0 / self._clean_fps)
            if due:
                self._publish_clean(frame, stamp)

        if self._raw_pub is not None and self._raw_fps > 0:
            due, self._last_raw_at = self._due(
                self._last_raw_at, now, 1.0 / self._raw_fps)
            if due:
                self._publish_raw(frame, stamp)

    def _publish_clean(self, frame: np.ndarray, stamp) -> None:
        img = frame
        h, w = img.shape[:2]
        if 0 < self._clean_max_w < w:
            scale = self._clean_max_w / w
            img = cv2.resize(img, (self._clean_max_w, int(round(h * scale))),
                             interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode('.jpg', img,
                               [int(cv2.IMWRITE_JPEG_QUALITY), self._clean_q])
        if not ok:
            self.get_logger().warn('rear_camera: JPEG encode failed',
                                   throttle_duration_sec=10.0)
            return
        msg = CompressedImage()
        msg.header.stamp = stamp
        msg.header.frame_id = self._frame_id
        msg.format = 'jpeg'
        msg.data = buf.tobytes()
        self._clean_pub.publish(msg)

    def _publish_raw(self, frame: np.ndarray, stamp) -> None:
        h, w = frame.shape[:2]
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = self._frame_id
        msg.height = h
        msg.width = w
        msg.encoding = 'bgr8'
        msg.is_bigendian = 0
        msg.step = w * 3
        # Hand-rolled rather than cv_bridge: cv_bridge is a heavy dependency for
        # one array copy, and v4l2_camera 0.7.1's cv_bridge path is exactly what
        # forced YUYV on the old dock pipeline in the first place.
        msg.data = np.ascontiguousarray(frame).tobytes()
        self._raw_pub.publish(msg)

        # apriltag_ros subscribes to camera_info even when pose estimation is off
        # (dock_apriltag.yaml sets pose_estimation_method: "") — it just never
        # reads K. Publishing an uncalibrated info keeps the subscription matched
        # without pretending we have a calibration we do not.
        info = CameraInfo()
        info.header = msg.header
        info.height = h
        info.width = w
        info.distortion_model = 'plumb_bob'
        self._info_pub.publish(info)


def main(args=None):
    rclpy.init(args=args)
    node = RearCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop.set()
        node._thread.join(timeout=5.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
