"""
head_detector_node — Jetson IMX219 + YOLO26n (native TensorRT) person detector
plus YuNet face detection for head tracking.

Publishes TWO vision_msgs/Detection2DArray topics (both QoS BEST_EFFORT /
VOLATILE / depth 10, frame_id 'camera_link', bbox in IMAGE PIXELS at
image_width x image_height — default 1280x720 — matching head_tracking_node):

  * /camera/detections — YOLO26n person boxes, class_id='person'. Unchanged
    world/object feed that behavior_node consumes and that the planned
    semantic-mapping layer will build on (YOLO already detects all COCO
    classes; only 'person' is published today).
  * /camera/faces      — YuNet face boxes, class_id='face'. The head-tracking
    target feed: head_tracking_node centres the head directly on the face box
    when a face is visible, and falls back to the person box otherwise.

  * score : float in [0, 1]  <-- ONE convention, end-to-end (both topics)

YOLO inference: YOLO26n exported to ONNX and built into an FP16 TensorRT engine
with trtexec against the on-board CUDA 13.2 / TRT 10.16 toolchain. The engine is
END-TO-END (NMS-free): output is [1, 300, 6] rows of [x1,y1,x2,y2,score,cls] in
letterboxed 640x640 pixels — so no NMS is done here.

Face inference: OpenCV's YuNet (cv2.FaceDetectorYN, the 2022mar ONNX — the
2023mar model needs OpenCV >= 4.8 and the Jetson has 4.6). Runs on the full BGR
frame each capture iteration; cheap CPU DNN. Faces are independent of the YOLO
boxes here (association to persons is left to the semantic-mapping layer). If the
YuNet model fails to load, face publishing is disabled and person detection
continues unaffected.

Phantom-person suppression (the old IMX500 node reported ~5 false persons on
clutter): min-confidence + min-box-area + a short temporal-persistence filter
(a box must recur in K of the last M frames), then largest-box-first ordering.
All thresholds are runtime-tunable via add_on_set_parameters_callback.
"""

import json
import threading
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

try:
    from head_detector.face_recognizer import FaceRecognizer, IdentitySmoother
except Exception:  # noqa: BLE001 - identity is optional; never block detection
    FaceRecognizer = None
    IdentitySmoother = None

import tensorrt as trt
from cuda.bindings import runtime as cudart


# ── COCO 'person' class index for YOLO ────────────────────────────────────────
_PERSON_CLASS = 0


def _cuda_check(ret, msg=''):
    """cuda-python calls return (err, *results); raise on non-zero, unwrap result."""
    if isinstance(ret, tuple):
        err, *rest = ret
        if int(err) != 0:
            raise RuntimeError(f'CUDA error {int(err)} ({msg})')
        return rest[0] if len(rest) == 1 else rest
    if int(ret) != 0:
        raise RuntimeError(f'CUDA error {int(ret)} ({msg})')
    return None


class TrtYolo:
    """Minimal TensorRT 10 runtime wrapper for a static-shape YOLO26n engine.

    All CUDA calls (malloc/memcpy/execute) must happen on ONE thread — this is
    constructed and invoked only from the capture thread.
    """

    def __init__(self, engine_path, logger):
        self._log = logger
        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(trt_logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f'failed to deserialize engine {engine_path}')
        self.ctx = self.engine.create_execution_context()

        self._buffers = {}
        self.in_name = self.out_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            dptr = _cuda_check(cudart.cudaMalloc(nbytes), f'malloc {name}')
            self.ctx.set_tensor_address(name, int(dptr))
            self._buffers[name] = dict(shape=shape, dtype=dtype, nbytes=nbytes, dptr=dptr)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.in_name = name
            else:
                self.out_name = name
        self.in_shape = self._buffers[self.in_name]['shape']      # (1,3,640,640)
        self.out_shape = self._buffers[self.out_name]['shape']    # (1,300,6)
        self.input_size = (self.in_shape[3], self.in_shape[2])    # (w, h)
        self._host_out = np.empty(int(np.prod(self.out_shape)),
                                  dtype=self._buffers[self.out_name]['dtype'])
        self.stream = _cuda_check(cudart.cudaStreamCreate(), 'stream')

    def infer(self, chw_f32):
        """Run one inference. chw_f32: contiguous float32 [1,3,H,W]. Returns [N,6]."""
        ib = self._buffers[self.in_name]
        ob = self._buffers[self.out_name]
        host_in = np.ascontiguousarray(chw_f32, dtype=np.float32)
        _cuda_check(cudart.cudaMemcpyAsync(
            int(ib['dptr']), host_in.ctypes.data, ib['nbytes'],
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream), 'H2D')
        if not self.ctx.execute_async_v3(self.stream):
            raise RuntimeError('execute_async_v3 failed')
        _cuda_check(cudart.cudaMemcpyAsync(
            self._host_out.ctypes.data, int(ob['dptr']), ob['nbytes'],
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream), 'D2H')
        _cuda_check(cudart.cudaStreamSynchronize(self.stream), 'sync')
        return self._host_out.reshape(self.out_shape)[0]          # [300, 6]


class YuNetFace:
    """Thin wrapper around cv2.FaceDetectorYN (YuNet).

    Constructed and invoked only from the capture thread. detect() takes a BGR
    frame at capture resolution and returns a list of face dicts in the SAME
    dict shape the YOLO postprocess uses ({score,x1,y1,x2,y2,cx,cy,bw,bh}) but
    already scaled into publish space (image_width x image_height), so the
    publish path is identical for persons and faces.

    YuNet raw output is [N, 15]: cols 0-3 = box (x, y, w, h) in the input-size
    pixel space, cols 4-13 = 5 landmarks (x, y), col 14 = score.
    """

    def __init__(self, model_path, cap_wh, pub_wh, score_thr, nms_thr, top_k):
        self._cap_w, self._cap_h = cap_wh
        self._sx = pub_wh[0] / float(cap_wh[0])   # capture -> publish scale
        self._sy = pub_wh[1] / float(cap_wh[1])
        self._pub_w, self._pub_h = pub_wh
        # config="" for ONNX; input_size set to the frame we actually pass in.
        self._fd = cv2.FaceDetectorYN.create(
            model_path, '', (self._cap_w, self._cap_h), score_thr, nms_thr, top_k)
        self._score_thr = score_thr

    def set_score_threshold(self, thr):
        """Re-applied from the capture thread when the live param changes."""
        if thr != self._score_thr:
            self._fd.setScoreThreshold(thr)
            self._score_thr = thr

    def detect(self, bgr):
        h, w = bgr.shape[:2]
        if (w, h) != (self._cap_w, self._cap_h):
            # capture size drifted (e.g. source reopened at a different mode)
            self._cap_w, self._cap_h = w, h
            self._sx = self._pub_w / float(w)
            self._sy = self._pub_h / float(h)
            self._fd.setInputSize((w, h))
        _, faces = self._fd.detect(bgr)
        out = []
        if faces is None:
            return out
        for f in faces:
            x, y, bw, bh, score = f[0], f[1], f[2], f[3], f[14]
            x1 = float(np.clip(x * self._sx, 0, self._pub_w))
            y1 = float(np.clip(y * self._sy, 0, self._pub_h))
            x2 = float(np.clip((x + bw) * self._sx, 0, self._pub_w))
            y2 = float(np.clip((y + bh) * self._sy, 0, self._pub_h))
            w2, h2 = x2 - x1, y2 - y1
            if w2 <= 1 or h2 <= 1:
                continue
            out.append({'score': float(score), 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'cx': x1 + w2 / 2, 'cy': y1 + h2 / 2, 'bw': w2, 'bh': h2,
                        # raw YuNet row (box + 5 landmarks, capture-space) for SFace
                        # alignment; ignored by the publish path.
                        '_raw': f})
        return out


class HeadDetectorNode(Node):

    def __init__(self):
        super().__init__('head_detector_node')

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter('engine_path',
                               '/home/Omni/head_detector/models/yolo26n_fp16.engine')
        # 'csi' -> nvarguscamerasrc (IMX219 on the Jetson CSI port);
        # 'usb' -> a UVC/USB webcam via v4l2src (opens `usb_device`);
        # any other value is treated as an image file path looped for offline validation.
        self.declare_parameter('source',               'csi')
        self.declare_parameter('sensor_id',            0)
        # For source=='usb': the V4L2 device to open. Default is the HHWei head cam by
        # its STABLE /dev/v4l/by-id path so it never swaps /dev/videoN with the rear cam.
        self.declare_parameter('usb_device',
            '/dev/v4l/by-id/usb-HHWei_Technology_Co.__Ltd._USB_Camera_HHW001-video-index0')
        self.declare_parameter('capture_width',        1280)
        self.declare_parameter('capture_height',       720)
        # IMX219 has NO native 1280x720@30 mode — the only 720p mode is @60 (mode 4).
        # Requesting 30 makes Argus negotiate an ambiguous mode -> INVALID_SETTINGS
        # halts, so 60 is the correct default for the 1280x720 capture above.
        self.declare_parameter('capture_fps',          60)
        self.declare_parameter('image_width',          1280)   # published coord space
        self.declare_parameter('image_height',         720)
        self.declare_parameter('detection_fps',        10.0)
        self.declare_parameter('frame_id',             'camera_link')
        self.declare_parameter('confidence_threshold', 0.4)    # publish gate (0-1)
        self.declare_parameter('max_detections',       5)
        # phantom-person suppression
        self.declare_parameter('min_box_area_frac',    0.004)  # >=0.4% of frame
        self.declare_parameter('persist_window',       5)      # last M frames
        self.declare_parameter('persist_min_hits',     3)      # need K of M
        self.declare_parameter('persist_iou',          0.3)    # box match IoU
        # ── Face detection (YuNet) ────────────────────────────────────────────
        # publish_faces gates the whole face path (model load + /camera/faces).
        # yunet_model_path defaults next to the TRT engine. face_score_threshold
        # and face_nms_threshold are YuNet's own gates; face_top_k caps candidates
        # before NMS. face_score_threshold is runtime-tunable.
        self.declare_parameter('publish_faces',         True)
        self.declare_parameter('yunet_model_path',      '')     # '' -> engine dir/yunet_face.onnx
        self.declare_parameter('face_score_threshold',  0.6)
        self.declare_parameter('face_nms_threshold',    0.3)
        self.declare_parameter('face_top_k',            50)
        # ── Face recognition (Step 6) — publishes /camera/identity for the Pi's
        # per-person memory. OFF by default: turn on with publish_identity:=true.
        # Requires publish_faces (needs YuNet landmarks) + an SFace model + a
        # gallery of enrolled photos. Fail-safe: any error disables identity only.
        self.declare_parameter('publish_identity',      False)
        self.declare_parameter('sface_model_path',      '')     # '' -> engine dir/sface.onnx
        self.declare_parameter('face_gallery_dir',      '/home/Omni/head_detector/faces')
        self.declare_parameter('face_unknown_dir',      '/home/Omni/head_detector/unknown_faces')
        self.declare_parameter('recognition_threshold', 0.363)  # SFace cosine
        # Quality gate — a turned/tiny/low-confidence face embeds poorly, so it must
        # not mint a new unknown_N (that produced endless junk ids). Frontality is
        # judged from the YuNet landmarks (nose offset vs eye separation).
        self.declare_parameter('recognition_min_score',   0.80)
        self.declare_parameter('recognition_min_face_px', 80)
        self.declare_parameter('recognition_max_nose_off', 0.40)
        # Hysteresis — majority vote over a sliding window so a single bad frame
        # doesn't flip the published identity.
        self.declare_parameter('identity_smooth_window', 15)
        self.declare_parameter('identity_switch_ratio',  0.6)
        # Multi-crop enrolment: keep capturing good frames briefly after an enrol
        # request so a person matches across poses, not just one dead-on shot.
        self.declare_parameter('enroll_samples',  5)
        self.declare_parameter('enroll_seconds',  2.0)
        # Debug feed: publish the annotated frame as CompressedImage (JPEG) on
        # /camera/image_annotated/compressed so you can watch it in Foxglove from a
        # PC. Off by default (saves USB/net + CPU); turn on with -p publish_debug_image:=true.
        self.declare_parameter('publish_debug_image',   False)
        self.declare_parameter('debug_jpeg_quality',    60)

        gp = lambda n: self.get_parameter(n).value
        self._engine_path = gp('engine_path')
        self._source      = str(gp('source'))
        self._sensor_id   = int(gp('sensor_id'))
        self._usb_device  = str(gp('usb_device'))
        self._cap_w       = int(gp('capture_width'))
        self._cap_h       = int(gp('capture_height'))
        self._cap_fps     = int(gp('capture_fps'))
        self._img_w       = int(gp('image_width'))
        self._img_h       = int(gp('image_height'))
        self._det_fps     = float(gp('detection_fps'))
        self._frame_id    = str(gp('frame_id'))
        self._conf_thresh = float(gp('confidence_threshold'))
        self._max_det     = int(gp('max_detections'))
        self._min_area_frac = float(gp('min_box_area_frac'))
        self._persist_win = int(gp('persist_window'))
        self._persist_hits = int(gp('persist_min_hits'))
        self._persist_iou = float(gp('persist_iou'))
        self._debug_image = bool(gp('publish_debug_image'))
        self._jpeg_quality = int(gp('debug_jpeg_quality'))
        self._publish_faces = bool(gp('publish_faces'))
        self._yunet_path    = str(gp('yunet_model_path'))
        self._face_score    = float(gp('face_score_threshold'))
        self._face_nms      = float(gp('face_nms_threshold'))
        self._face_top_k    = int(gp('face_top_k'))
        self._publish_identity = bool(gp('publish_identity')) and self._publish_faces
        self._sface_path    = str(gp('sface_model_path'))
        self._gallery_dir   = str(gp('face_gallery_dir'))
        self._unknown_dir   = str(gp('face_unknown_dir'))
        self._recog_thr     = float(gp('recognition_threshold'))
        self._recog_min_score   = float(gp('recognition_min_score'))
        self._recog_min_face_px = float(gp('recognition_min_face_px'))
        self._recog_max_nose_off = float(gp('recognition_max_nose_off'))
        self._smooth_window = int(gp('identity_smooth_window'))
        self._switch_ratio  = float(gp('identity_switch_ratio'))
        self._enroll_samples = int(gp('enroll_samples'))
        self._enroll_seconds = float(gp('enroll_seconds'))

        self.add_on_set_parameters_callback(self._on_set_parameters)

        # ── State ────────────────────────────────────────────────────────────
        self._lock = threading.Lock()
        self._latest_detections = []          # list of published person dicts
        self._latest_faces = []               # list of published face dicts
        self._stop = threading.Event()
        self._connected = False
        self._frame_count = 0
        self._infer_ms = 0.0
        self._face_ms = 0.0
        self._history = deque(maxlen=64)       # recent raw person boxes per frame
        self._trt = None
        self._yunet = None                     # YuNetFace, built in the capture thread
        self._latest_frame = None              # newest BGR frame, for the debug feed
        self._recognizer = None                # FaceRecognizer, built in the capture thread
        self._latest_identity = ''             # SMOOTHED primary-face person id ('' = none)
        self._enroll_frame = None              # newest frame + primary face for on-the-fly
        self._enroll_face = None               # enrollment (Step 6 remember_person)
        # Hysteresis over raw per-frame recognition (built with the recognizer).
        self._smoother = None
        # Multi-crop enrolment in progress: {'name', 'deadline', 'left'} or None.
        # Written by the enrol callback (ROS thread), consumed by the capture thread.
        self._enroll_pending = None

        # ── Publishers (match old camera_node QoS) ───────────────────────────
        sensor_qos = QoSProfile(depth=10,
                                reliability=ReliabilityPolicy.BEST_EFFORT,
                                durability=DurabilityPolicy.VOLATILE)
        self._det_pub = self.create_publisher(Detection2DArray, '/camera/detections', sensor_qos)
        self._face_pub = (
            self.create_publisher(Detection2DArray, '/camera/faces', sensor_qos)
            if self._publish_faces else None)
        self._status_pub = self.create_publisher(String, '/camera/status', 10)
        self._identity_pub = (
            self.create_publisher(String, '/camera/identity', 10)
            if self._publish_identity else None)
        # On-the-fly enrollment (Step 6): Pi publishes a name -> we save the face
        # currently in view under it. Result echoed on /camera/enroll_result.
        self._enroll_result_pub = None
        if self._publish_identity:
            self._enroll_result_pub = self.create_publisher(String, '/camera/enroll_result', 10)
            self.create_subscription(String, '/camera/enroll_request', self._on_enroll_request, 10)
        self._debug_pub = (
            self.create_publisher(CompressedImage, '/camera/image_annotated/compressed', 1)
            if self._debug_image else None)

        # ── Capture+inference thread + publish timer ─────────────────────────
        self._thread = threading.Thread(target=self._capture_loop, daemon=True,
                                        name='head_det_capture')
        self._thread.start()
        self.create_timer(1.0 / self._det_fps, self._publish_cb)
        self._status_ticks = 0

        self.get_logger().info(
            f'head_detector_node started — source={self._source}, '
            f'publish space {self._img_w}x{self._img_h}, conf>={self._conf_thresh}, '
            f'persist {self._persist_hits}/{self._persist_win}, max_det={self._max_det}, '
            f'faces={"on" if self._publish_faces else "off"} '
            f'(score>={self._face_score})')

    # ── Live parameter callback ───────────────────────────────────────────────
    def _on_set_parameters(self, params):
        pending = []
        for p in params:
            if p.name == 'confidence_threshold':
                try:
                    v = float(p.value)
                except (TypeError, ValueError):
                    return SetParametersResult(successful=False,
                                               reason='confidence_threshold must be a number')
                if not (0.0 <= v <= 1.0):
                    return SetParametersResult(successful=False,
                                               reason='confidence_threshold must be in 0.0-1.0')
                pending.append(('_conf_thresh', v, p.name))
            elif p.name == 'max_detections':
                try:
                    v = int(p.value)
                except (TypeError, ValueError):
                    return SetParametersResult(successful=False,
                                               reason='max_detections must be an integer')
                if v < 1:
                    return SetParametersResult(successful=False,
                                               reason='max_detections must be >= 1')
                pending.append(('_max_det', v, p.name))
            elif p.name == 'min_box_area_frac':
                v = float(p.value)
                if not (0.0 <= v <= 1.0):
                    return SetParametersResult(successful=False,
                                               reason='min_box_area_frac must be in 0.0-1.0')
                pending.append(('_min_area_frac', v, p.name))
            elif p.name == 'persist_window':
                pending.append(('_persist_win', max(1, int(p.value)), p.name))
            elif p.name == 'persist_min_hits':
                pending.append(('_persist_hits', max(1, int(p.value)), p.name))
            elif p.name == 'persist_iou':
                pending.append(('_persist_iou', float(p.value), p.name))
            elif p.name == 'face_score_threshold':
                try:
                    v = float(p.value)
                except (TypeError, ValueError):
                    return SetParametersResult(successful=False,
                                               reason='face_score_threshold must be a number')
                if not (0.0 <= v <= 1.0):
                    return SetParametersResult(successful=False,
                                               reason='face_score_threshold must be in 0.0-1.0')
                # Applied to the live YuNet detector from the capture thread.
                pending.append(('_face_score', v, p.name))
        for attr, val, name in pending:
            setattr(self, attr, val)
            self.get_logger().info(f'Parameter updated: {name} = {val}')
        return SetParametersResult(successful=True)

    # ── Frame source ──────────────────────────────────────────────────────────
    def _gst_pipeline(self):
        return (
            f'nvarguscamerasrc sensor-id={self._sensor_id} ! '
            f'video/x-raw(memory:NVMM),width={self._cap_w},height={self._cap_h},'
            f'framerate={self._cap_fps}/1 ! '
            f'nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! '
            f'video/x-raw,format=BGR ! appsink drop=1 max-buffers=1 sync=false'
        )

    def _usb_pipeline(self):
        # UVC webcam: request MJPG (keeps USB bandwidth low), decode to BGR for OpenCV.
        # drop=1/max-buffers=1 keeps latency low — we always want the freshest frame.
        return (
            f'v4l2src device={self._usb_device} ! '
            f'image/jpeg,width={self._cap_w},height={self._cap_h},'
            f'framerate={self._cap_fps}/1 ! '
            f'jpegdec ! videoconvert ! video/x-raw,format=BGR ! '
            f'appsink drop=1 max-buffers=1 sync=false'
        )

    def _open_source(self):
        if self._source == 'csi':
            cap = cv2.VideoCapture(self._gst_pipeline(), cv2.CAP_GSTREAMER)
            return cap if cap.isOpened() else None
        if self._source == 'usb':
            cap = cv2.VideoCapture(self._usb_pipeline(), cv2.CAP_GSTREAMER)
            return cap if cap.isOpened() else None
        # offline image-file mode: return a sentinel; frames read from disk
        img = cv2.imread(self._source)
        if img is None:
            self.get_logger().error(f'source image not readable: {self._source}',
                                    throttle_duration_sec=10.0)
            return None
        self._still = cv2.resize(img, (self._cap_w, self._cap_h))
        return 'still'

    # ── Preprocess / postprocess ──────────────────────────────────────────────
    def _letterbox(self, bgr):
        """Resize+pad BGR frame to the engine input square; return (chw, r, padx, pady)."""
        iw, ih = self._trt.input_size            # (640, 640)
        h, w = bgr.shape[:2]
        r = min(iw / w, ih / h)
        nw, nh = int(round(w * r)), int(round(h * r))
        resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((ih, iw, 3), 114, dtype=np.uint8)
        padx, pady = (iw - nw) // 2, (ih - nh) // 2
        canvas[pady:pady + nh, padx:padx + nw] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])   # [1,3,H,W]
        return chw, r, padx, pady

    def _postprocess(self, out, r, padx, pady):
        """out: [300,6] letterbox-space rows -> list of person boxes in image space."""
        if out.size == 0:
            return []
        cls = out[:, 5].astype(np.int32)
        score = out[:, 4]
        keep = (cls == _PERSON_CLASS) & (score >= self._conf_thresh)
        rows = out[keep]
        boxes = []
        min_area = self._min_area_frac * (self._img_w * self._img_h)
        sx = self._img_w / self._cap_w   # capture->publish scale (usually 1.0)
        sy = self._img_h / self._cap_h
        for x1, y1, x2, y2, sc, _ in rows:
            # undo letterbox to capture pixels, then scale to publish space
            bx1 = (x1 - padx) / r * sx
            by1 = (y1 - pady) / r * sy
            bx2 = (x2 - padx) / r * sx
            by2 = (y2 - pady) / r * sy
            bx1 = float(np.clip(bx1, 0, self._img_w)); bx2 = float(np.clip(bx2, 0, self._img_w))
            by1 = float(np.clip(by1, 0, self._img_h)); by2 = float(np.clip(by2, 0, self._img_h))
            bw, bh = bx2 - bx1, by2 - by1
            if bw <= 1 or bh <= 1 or (bw * bh) < min_area:
                continue
            boxes.append({'score': float(sc), 'x1': bx1, 'y1': by1, 'x2': bx2, 'y2': by2,
                          'cx': bx1 + bw / 2, 'cy': by1 + bh / 2, 'bw': bw, 'bh': bh})
        return boxes

    @staticmethod
    def _iou(a, b):
        ix1, iy1 = max(a['x1'], b['x1']), max(a['y1'], b['y1'])
        ix2, iy2 = min(a['x2'], b['x2']), min(a['y2'], b['y2'])
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        ua = a['bw'] * a['bh'] + b['bw'] * b['bh'] - inter
        return inter / ua if ua > 0 else 0.0

    def _persistence_filter(self, current):
        """Keep only boxes that recur in >= persist_min_hits of the last window frames."""
        window = list(self._history)[-self._persist_win:]
        confirmed = []
        for det in current:
            hits = 1  # counts current frame
            for past in window:
                if any(self._iou(det, p) >= self._persist_iou for p in past):
                    hits += 1
            if hits >= self._persist_hits:
                confirmed.append(det)
        return confirmed

    # ── Capture loop (background thread) ──────────────────────────────────────
    def _capture_loop(self):
        try:
            self._trt = TrtYolo(self._engine_path, self.get_logger())
            self.get_logger().info(
                f'TRT engine loaded: in={self._trt.in_shape} out={self._trt.out_shape}')
        except Exception as e:
            self.get_logger().error(f'TRT init failed: {e}')
            return

        # YuNet is optional: a load failure disables face publishing but leaves
        # person detection fully working.
        if self._publish_faces:
            try:
                import os
                model = self._yunet_path or os.path.join(
                    os.path.dirname(self._engine_path), 'yunet_face.onnx')
                self._yunet = YuNetFace(
                    model, (self._cap_w, self._cap_h), (self._img_w, self._img_h),
                    self._face_score, self._face_nms, self._face_top_k)
                self.get_logger().info(f'YuNet face detector loaded: {model}')
            except Exception as e:
                self._yunet = None
                self.get_logger().error(
                    f'YuNet load failed ({e}) — face detection disabled, '
                    f'person detection continues')

        # Face recognition (Step 6). Built here (capture thread) alongside YuNet.
        # Fail-safe: on any problem, identity stays '' and detection is unaffected.
        if self._publish_identity and self._yunet is not None and FaceRecognizer is not None:
            try:
                import os
                sface = self._sface_path or os.path.join(
                    os.path.dirname(self._engine_path), 'sface.onnx')
                yunet_model = self._yunet_path or os.path.join(
                    os.path.dirname(self._engine_path), 'yunet_face.onnx')
                self._recognizer = FaceRecognizer(
                    sface, yunet_model, self._gallery_dir, self.get_logger(),
                    unknown_dir=self._unknown_dir,
                    match_threshold=self._recog_thr, unknown_threshold=self._recog_thr,
                    min_score=self._recog_min_score,
                    min_face_px=self._recog_min_face_px,
                    max_nose_off=self._recog_max_nose_off)
                if not self._recognizer.ok:
                    self._recognizer = None
                else:
                    self._smoother = IdentitySmoother(
                        window=self._smooth_window, switch_ratio=self._switch_ratio)
            except Exception as e:  # noqa: BLE001
                self._recognizer = None
                self.get_logger().error(
                    f'FaceRecognizer load failed ({e}) — identity disabled, '
                    f'detection continues')

        cap = None
        while not self._stop.is_set():
            if cap is None:
                cap = self._open_source()
                if cap is None:
                    self._connected = False
                    time.sleep(1.0)
                    continue
                self._connected = True
                self.get_logger().info(f'frame source open: {self._source}')

            if cap == 'still':
                frame = self._still.copy()
                time.sleep(1.0 / max(1.0, self._cap_fps))
            else:
                ok, frame = cap.read()
                if not ok or frame is None:
                    self.get_logger().warn('frame read failed; reopening source',
                                           throttle_duration_sec=5.0)
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = None
                    self._connected = False
                    continue

            t0 = time.monotonic()
            chw, r, padx, pady = self._letterbox(frame)
            out = self._trt.infer(chw)
            self._infer_ms = (time.monotonic() - t0) * 1000.0

            person_boxes = self._postprocess(out, r, padx, pady)
            self._history.append(person_boxes)
            confirmed = self._persistence_filter(person_boxes)
            # largest-box first (area), then score
            confirmed.sort(key=lambda d: (-(d['bw'] * d['bh']), -d['score']))

            # ── Face detection (YuNet) — independent of the YOLO boxes ─────────
            faces = []
            if self._yunet is not None:
                try:
                    self._yunet.set_score_threshold(self._face_score)
                    tf = time.monotonic()
                    faces = self._yunet.detect(frame)
                    self._face_ms = (time.monotonic() - tf) * 1000.0
                    # largest face first — the closest person's face wins tracking
                    faces.sort(key=lambda d: -(d['bw'] * d['bh']))
                    faces = faces[:self._max_det]
                except Exception as e:
                    self.get_logger().warn(f'YuNet detect error: {e}',
                                           throttle_duration_sec=5.0)

            # ── Face recognition (Step 6) — resolve the PRIMARY (largest) face ─
            # to a person id for /camera/identity. Fail-safe; '' when no face.
            identity = ''
            if self._recognizer is not None:
                raw_id = ''
                if faces:
                    try:
                        raw_id = self._recognizer.identify(frame, faces[0]['_raw'])
                    except Exception as e:  # noqa: BLE001
                        self.get_logger().warn(f'recognition error: {e}',
                                               throttle_duration_sec=5.0)
                    self._service_pending_enroll(frame, faces[0]['_raw'])
                # Hysteresis: publish the smoothed id, so one off-angle or blank
                # frame can't flip identity (raw_id flickers; the vote doesn't).
                identity = (self._smoother.update(raw_id)
                            if self._smoother is not None else raw_id)

            with self._lock:
                self._latest_detections = confirmed[:self._max_det]
                self._latest_faces = faces
                self._latest_identity = identity
                # Keep the current primary face + frame so a concurrent enroll
                # request can learn whoever is in view right now.
                if self._recognizer is not None and faces:
                    self._enroll_frame = frame
                    self._enroll_face = faces[0]['_raw']
                else:
                    self._enroll_frame = None
                    self._enroll_face = None
                self._frame_count += 1
                if self._debug_image:
                    self._latest_frame = frame

    # ── Detection publish ─────────────────────────────────────────────────────
    def _publish_cb(self):
        now = self.get_clock().now().to_msg()
        with self._lock:
            dets = list(self._latest_detections)
            faces = list(self._latest_faces)
            connected = self._connected
            frames = self._frame_count
            infer_ms = self._infer_ms
            face_ms = self._face_ms
            identity = self._latest_identity
            frame = self._latest_frame if self._debug_pub is not None else None

        self._det_pub.publish(self._build_array(now, dets, 'person'))
        if self._face_pub is not None:
            self._face_pub.publish(self._build_array(now, faces, 'face'))
        if self._identity_pub is not None:
            self._identity_pub.publish(String(data=identity))

        self._status_ticks += 1
        if self._status_ticks % max(1, round(self._det_fps)) == 0:
            self._status_pub.publish(String(data=json.dumps({
                'connected': connected, 'persons': len(dets), 'faces': len(faces),
                'frames': frames, 'infer_ms': round(infer_ms, 2),
                'face_ms': round(face_ms, 2)})))

        if self._debug_pub is not None and frame is not None:
            self._publish_annotated(now, frame, dets, faces)

    def _build_array(self, stamp, items, class_id):
        """Pack a box-dict list into a Detection2DArray with the given class_id."""
        arr = Detection2DArray()
        arr.header.stamp = stamp
        arr.header.frame_id = self._frame_id
        for d in items:
            det = Detection2D()
            det.header.stamp = stamp
            det.header.frame_id = self._frame_id
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = class_id
            hyp.hypothesis.score = d['score']          # 0-1, single convention
            det.results.append(hyp)
            det.bbox.center.position.x = d['cx']
            det.bbox.center.position.y = d['cy']
            det.bbox.size_x = d['bw']
            det.bbox.size_y = d['bh']
            arr.detections.append(det)
        return arr

    def _service_pending_enroll(self, frame, raw_face):
        """Multi-crop enrolment (capture thread): after an enrol request, keep adding
        good frames for a short window so the person matches across poses rather than
        one dead-on shot. Only frontal, confident faces are sampled."""
        pending = self._enroll_pending
        if not pending or self._recognizer is None:
            return
        now = time.monotonic()
        if now > pending['deadline'] or pending['left'] <= 0:
            self._enroll_pending = None
            self.get_logger().info(
                f"enrol '{pending['name']}': {pending['taken']} sample(s) captured")
            return
        if now < pending['next_at'] or not self._recognizer.quality_ok(raw_face):
            return
        try:
            if self._recognizer.enroll(pending['name'], frame, raw_face):
                pending['left'] -= 1
                pending['taken'] += 1
                pending['next_at'] = now + pending['interval']
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'enrol sample error: {e}', throttle_duration_sec=5.0)

    def _start_pending_enroll(self, name, taken):
        now = time.monotonic()
        interval = self._enroll_seconds / max(1, self._enroll_samples)
        self._enroll_pending = {
            'name': name, 'taken': taken, 'left': max(0, self._enroll_samples - taken),
            'deadline': now + self._enroll_seconds,
            'next_at': now + interval, 'interval': interval,
        }

    def _on_enroll_request(self, msg):
        """Learn the face currently in view as msg.data (a name). Fail-safe.
        Takes one sample now (if the face is good) and collects a few more over the
        next couple of seconds for pose robustness."""
        name = (msg.data or '').strip().lower()
        with self._lock:
            frame = self._enroll_frame
            face = self._enroll_face
        ok = False
        detail = ''
        if self._recognizer is None:
            detail = 'recognizer unavailable'
        elif not name:
            detail = 'empty name'
        elif frame is None or face is None:
            detail = 'no face in view'
        else:
            try:
                if self._recognizer.quality_ok(face):
                    ok = self._recognizer.enroll(name, frame, face)
                    detail = 'enrolled' if ok else 'enroll failed'
                else:
                    # Face present but turned/small — let the collection window catch
                    # a good frame instead of enrolling a poor reference.
                    ok = True
                    detail = 'face not frontal enough — collecting'
                if ok and self._enroll_samples > 1:
                    self._start_pending_enroll(name, 1 if detail == 'enrolled' else 0)
            except Exception as e:  # noqa: BLE001
                detail = f'error: {e}'
        self.get_logger().info(f"enroll_request '{name}': {detail}")
        if self._enroll_result_pub is not None:
            self._enroll_result_pub.publish(String(data=json.dumps(
                {'name': name, 'ok': bool(ok), 'detail': detail})))

    def _publish_annotated(self, stamp, frame, dets, faces):
        """Draw person boxes (green) + face boxes (cyan) on the frame, JPEG it."""
        img = frame.copy()
        # boxes are in publish space (image_width x image_height); the frame is capture
        # space (capture_width x capture_height) — scale the boxes back onto the frame.
        fx = self._cap_w / float(self._img_w)
        fy = self._cap_h / float(self._img_h)
        for d in dets:
            x1 = int((d['cx'] - d['bw'] / 2) * fx); y1 = int((d['cy'] - d['bh'] / 2) * fy)
            x2 = int((d['cx'] + d['bw'] / 2) * fx); y2 = int((d['cy'] + d['bh'] / 2) * fy)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{d['score']:.2f}", (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        # face boxes (cyan): head_tracking centres directly on the face box centre.
        for d in faces:
            x1 = int((d['cx'] - d['bw'] / 2) * fx); y1 = int((d['cy'] - d['bh'] / 2) * fy)
            x2 = int((d['cx'] + d['bw'] / 2) * fx); y2 = int((d['cy'] + d['bh'] / 2) * fy)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.circle(img, (int(d['cx'] * fx), int(d['cy'] * fy)), 4, (0, 0, 255), -1)
        ok, buf = cv2.imencode('.jpg', img,
                               [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality])
        if not ok:
            return
        m = CompressedImage()
        m.header.stamp = stamp
        m.header.frame_id = self._frame_id
        m.format = 'jpeg'
        m.data = buf.tobytes()
        self._debug_pub.publish(m)

    def destroy_node(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HeadDetectorNode()
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
