import json
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

try:
    from picamera2 import Picamera2
    from picamera2.devices.imx500 import IMX500
    _PICAMERA2_AVAILABLE = True
    _PICAMERA2_ERROR = ''
except ImportError as e:
    _PICAMERA2_AVAILABLE = False
    _PICAMERA2_ERROR = str(e)

_DEFAULT_MODEL = (
    '/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk'
)

# COCO-91 labels as embedded in the SSD MobileNetV2 FPN Lite _pp model.
# Index 0 = "person" — highest priority class for OMNI person-following.
# "-" entries are unused COCO-91 slots; detections on these are silently dropped.
_FALLBACK_LABELS = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', '-', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', '-', 'backpack', 'umbrella',
    '-', '-', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', '-', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', '-', 'dining table', '-', '-',
    'toilet', '-', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    '-', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush',
]


class CameraNode(Node):

    def __init__(self):
        super().__init__('camera_node')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('fps',                  30.0)
        self.declare_parameter('detection_fps',        10.0)
        self.declare_parameter('frame_id',             'camera_link')
        self.declare_parameter('image_width',          1280)
        self.declare_parameter('image_height',         720)
        self.declare_parameter('model_path',           _DEFAULT_MODEL)
        self.declare_parameter('confidence_threshold', 0.5)

        self._fps         = self.get_parameter('fps').value
        self._det_fps     = self.get_parameter('detection_fps').value
        self._frame_id    = self.get_parameter('frame_id').value
        self._img_w       = self.get_parameter('image_width').value
        self._img_h       = self.get_parameter('image_height').value
        self._model_path  = self.get_parameter('model_path').value
        self._conf_thresh = self.get_parameter('confidence_threshold').value

        # ── State ────────────────────────────────────────────────────────────
        self._latest_frame      = None   # numpy [H, W, 3] RGB888
        self._latest_detections = []     # list of dicts
        self._lock              = threading.Lock()
        self._stop_event        = threading.Event()
        self._connected         = False
        self._inference_ok      = False
        self._last_connect_attempt = 0.0
        self._frame_count       = 0

        self._labels            = _FALLBACK_LABELS
        self._bbox_normalization = None  # read from intrinsics after connect
        self._bbox_order        = None   # read from intrinsics after connect

        self._picam2        = None
        self._imx500        = None
        self._det_cb_count  = 0   # counter for 1 Hz status publish

        # ── Publishers ───────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._image_pub     = self.create_publisher(Image,            '/camera/image',      sensor_qos)
        self._detection_pub = self.create_publisher(Detection2DArray, '/camera/detections', sensor_qos)
        self._status_pub    = self.create_publisher(String,           '/camera/status',     10)

        if not _PICAMERA2_AVAILABLE:
            self.get_logger().error(f'picamera2 unavailable: {_PICAMERA2_ERROR}')

        # ── Initial connection ────────────────────────────────────────────────
        self._connect_camera()

        # ── Background capture thread ─────────────────────────────────────────
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name='cam_capture'
        )
        self._capture_thread.start()

        # ── Timers ───────────────────────────────────────────────────────────
        self.create_timer(1.0 / self._fps,     self._publish_image_cb)
        self.create_timer(1.0 / self._det_fps, self._publish_detections_cb)
        self.create_timer(1.0,                 self._reconnect_cb)

        self.get_logger().info(
            f'camera_node started — {self._img_w}x{self._img_h} @ {self._fps:.0f} fps, '
            f'detections @ {self._det_fps:.0f} Hz, '
            f'confidence ≥ {self._conf_thresh}'
        )

    # ── Connection ───────────────────────────────────────────────────────────

    def _connect_camera(self):
        if not _PICAMERA2_AVAILABLE:
            return
        self._last_connect_attempt = time.monotonic()
        try:
            self.get_logger().info(
                'Loading IMX500 model firmware — this may take ~30 seconds on first run...'
            )
            self._imx500 = IMX500(self._model_path)

            # Read intrinsics embedded in the model firmware
            ni = self._imx500.network_intrinsics
            if ni is not None:
                if ni.labels:
                    self._labels = ni.labels
                    self.get_logger().info(f'  Loaded {len(ni.labels)} labels from model')
                self._bbox_normalization = ni.bbox_normalization
                self._bbox_order        = ni.bbox_order

            self._picam2 = Picamera2(self._imx500.camera_num)
            config = self._picam2.create_preview_configuration(
                main={'size': (self._img_w, self._img_h), 'format': 'RGB888'},
                controls={'FrameRate': self._fps},
                buffer_count=6,
            )
            self._picam2.configure(config)
            self._picam2.start()

            self._connected    = True
            self._inference_ok = True
            self.get_logger().info(
                f'Camera connected — model: {self._model_path.split("/")[-1]}, '
                f'bbox_normalization={self._bbox_normalization}, '
                f'bbox_order={self._bbox_order}'
            )
        except Exception as e:
            self._connected    = False
            self._inference_ok = False
            if self._picam2 is not None:
                try:
                    self._picam2.stop()
                    self._picam2.close()
                except Exception:
                    pass
            self._picam2 = None
            self._imx500 = None
            self.get_logger().error(f'Camera init failed: {e}')

    def _reconnect_cb(self):
        if not self._connected:
            if time.monotonic() - self._last_connect_attempt < 3.0:
                return
            self.get_logger().info('Attempting camera reconnect...')
            self._connect_camera()

    # ── Capture loop (background thread) ─────────────────────────────────────

    def _capture_loop(self):
        while not self._stop_event.is_set():
            if not self._connected or self._picam2 is None:
                time.sleep(0.1)
                continue
            try:
                request  = self._picam2.capture_request()
                array    = request.make_array('main')   # [H, W, 3] RGB888
                metadata = request.get_metadata()
                request.release()

                detections = []
                if self._inference_ok:
                    try:
                        outputs = self._imx500.get_outputs(metadata, add_batch=True)
                        if outputs is not None:
                            detections = self._parse_detections(outputs, metadata)
                    except Exception as e:
                        self.get_logger().warn(
                            f'Inference parse error: {e}',
                            throttle_duration_sec=5.0,
                        )

                with self._lock:
                    self._latest_frame      = array
                    self._latest_detections = detections
                    self._frame_count      += 1

            except Exception as e:
                self.get_logger().warn(
                    f'Capture error: {e}',
                    throttle_duration_sec=5.0,
                )
                time.sleep(0.05)

    # ── Detection parsing ─────────────────────────────────────────────────────

    def _parse_detections(self, outputs, metadata):
        """Parse on-chip SSD _pp outputs using the official picamera2 convention."""
        if len(outputs) < 3:
            return []

        # Official output order from imx500_object_detection_demo.py:
        #   outputs[0][0] → boxes  [N, 4]  in [y_min, x_min, y_max, x_max]
        #   outputs[1][0] → scores [N]
        #   outputs[2][0] → classes [N]
        boxes   = np.asarray(outputs[0][0])  # [N, 4]
        scores  = np.asarray(outputs[1][0])  # [N]
        classes = np.asarray(outputs[2][0])  # [N]

        # Normalize boxes if the model emits pixel-space coordinates
        if self._bbox_normalization:
            input_h = self._imx500.get_input_size()[1]
            boxes = boxes / input_h

        # Reorder from xy to yx if needed (SSD standard is already yx)
        if self._bbox_order == 'xy':
            boxes = boxes[:, [1, 0, 3, 2]]

        detections = []
        for i in range(len(scores)):
            score = float(scores[i])
            if score < self._conf_thresh:
                continue

            cls_id = int(classes[i])
            label  = self._labels[cls_id] if 0 <= cls_id < len(self._labels) else str(cls_id)
            if label == '-':
                continue   # undefined COCO-91 slot

            # convert_inference_coords maps normalized yx inference coords → image
            # pixel (x, y, w, h), correctly handling aspect-ratio letterboxing
            try:
                px, py, pw, ph = self._imx500.convert_inference_coords(
                    boxes[i], metadata, self._picam2
                )
            except Exception:
                continue
            if pw <= 0 or ph <= 0:
                continue

            detections.append({
                'label': label,
                'score': score,
                'cx':    float(px + pw / 2),
                'cy':    float(py + ph / 2),
                'bw':    float(pw),
                'bh':    float(ph),
            })

        # Person detections float to the top — feeds behavior_node person-following
        detections.sort(key=lambda d: (0 if d['label'] == 'person' else 1, -d['score']))
        return detections

    # ── Image publish ─────────────────────────────────────────────────────────

    def _publish_image_cb(self):
        with self._lock:
            frame = self._latest_frame

        if frame is None:
            return

        h, w, _ = frame.shape
        msg             = Image()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.height          = h
        msg.width           = w
        msg.encoding        = 'rgb8'
        msg.is_bigendian    = False
        msg.step            = w * 3
        msg.data            = frame.tobytes()
        self._image_pub.publish(msg)

    # ── Detection publish ─────────────────────────────────────────────────────

    def _publish_detections_cb(self):
        now = self.get_clock().now().to_msg()

        with self._lock:
            detections = list(self._latest_detections)
            connected  = self._connected

        arr             = Detection2DArray()
        arr.header.stamp    = now
        arr.header.frame_id = self._frame_id

        for d in detections:
            det                 = Detection2D()
            det.header.stamp    = now
            det.header.frame_id = self._frame_id

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = d['label']
            hyp.hypothesis.score    = d['score']
            det.results.append(hyp)

            det.bbox.center.position.x = d['cx']
            det.bbox.center.position.y = d['cy']
            det.bbox.size_x             = d['bw']
            det.bbox.size_y             = d['bh']

            arr.detections.append(det)

        self._detection_pub.publish(arr)

        # Status at ~1 Hz: every det_fps callbacks (10 Hz ÷ 10 = 1 Hz)
        self._det_cb_count += 1
        if self._det_cb_count % max(1, round(self._det_fps)) == 0:
            person_count = sum(1 for d in detections if d['label'] == 'person')
            status = json.dumps({
                'connected':  connected,
                'inference':  self._inference_ok,
                'detections': len(detections),
                'persons':    person_count,
                'frames':     self._frame_count,
            })
            self._status_pub.publish(String(data=status))

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def destroy_node(self):
        self._stop_event.set()
        self._capture_thread.join(timeout=2.0)
        if self._picam2 is not None:
            try:
                self._picam2.stop()
                self._picam2.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
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
