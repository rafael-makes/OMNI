# OMNI — Jetson-side vision packages

These ROS 2 (Jazzy) packages run on the **Jetson Orin Nano**, not the Pi. The Pi's
`colcon` ignores this folder (`COLCON_IGNORE`). This directory is the version-controlled
source; the Jetson runs copies under `~/omni_jetson_ws/src/`.

## Packages
- **`head_detector`** — person detector for head tracking. Captures from a USB webcam
  (`source:=usb`, default) or the IMX219 CSI camera (`source:=csi`), runs YOLO26n as a
  native TensorRT engine, and publishes `vision_msgs/Detection2DArray` on
  `/camera/detections` (+ `/camera/status`, and an optional annotated
  `/camera/image_annotated/compressed` when `publish_debug_image:=true`).
- **`frame_server`** — serves the newest head-camera frame as JPEG on the
  `/vision/get_camera_frame` service (`omni_vision_msgs/srv/GetCameraFrame`). The Pi's
  behavior_node calls it across the direct link when Gemini invokes `describe_scene`
  ("what do you see?"). Subscribes to head_detector's clean feed Jetson-locally, so only
  the service reply crosses the wire. Needs `head_detector` running with
  `publish_clean_image:=true` (the bringup launch does this).
- **`omni_jetson_bringup`** — launch + config:
  - `head_detector.launch.py` — head USB cam detector (1280x720 MJPG, HHWei by-id) +
    `frame_server`. Capture resolution is DECOUPLED from detection: `detect_max_width`
    (848) is what YOLO/YuNet see, while the scene-description feed gets the full capture
    frame. Raising capture without that split doubled YuNet's cost and pushed the
    pipeline under its 10 Hz target.
    Also launches `rear_camera` (`rear_vision:=true` by default), so
    `describe_scene(direction='behind'|'all')` has a cached rear frame.
  - `oak_scan.launch.py` — OAK-D Lite depth → `/oak/scan` (via `depthimage_to_laserscan`)
    for the Pi's Nav2 obstacle layer; publishes `base_link -> oak` TF.
  - `dock_apriltag_shared.launch.py` — **preferred** dock detector. `rear_camera`
    owns the device and feeds apriltag_ros, so docking and rear-view scene
    description work at the same time.
  - `dock_apriltag.launch.py` — legacy dock detector; opens the rear device
    directly in YUYV. Kept as a fallback. **Mutually exclusive** with
    `rear_camera` — run it only with `rear_vision:=false`.
  - `config/oak_lite_depth.yaml` — depth-only OAK pipeline + mount TF.

### One camera, one owner
Each USB camera is opened by exactly ONE process; everything else subscribes.
head cam → `head_detector`, rear cam → `rear_camera`, both → `frame_server`.
A V4L2 device serves one process in one format, so a second opener gets `-EBUSY`.
If a camera node logs "cannot open", check `fuser -v /dev/v4l/by-id/<cam>` first.

### Rear camera gotchas (measured 2026-07-18)
- The 2K rear cam advertises **only 30fps** MJPG intervals. Asking for 5fps does
  not give you 5fps — it fails caps negotiation and the device never opens.
  Dropping the resolution does not help. Throttle with `decode_fps` instead.
- It delivers ~25fps, not 30. That ceiling is the camera, not the USB bus.
- All three USB devices (head cam, rear cam, OAK-D) share ONE 480 Mbps bus.
  Verified stable streaming simultaneously: rear ran 24.99 fps alone vs 25.00 fps
  under full load, zero USB errors. Keep MJPG — raw YUYV at 640x480 would cost
  ~147 Mbps against a ~35 MB/s practical budget.
- `rear_camera` publishes its two feeds with **different QoS on purpose**: the
  clean JPEG feed is BEST_EFFORT (to match `frame_server`), the raw feed is
  RELIABLE (to match apriltag_ros). Unifying them silently breaks one consumer.

## Deploy to the Jetson
```bash
rsync -a --exclude __pycache__ jetson/head_detector        Omni@<jetson>:~/omni_jetson_ws/src/
rsync -a --exclude __pycache__ jetson/omni_jetson_bringup  Omni@<jetson>:~/omni_jetson_ws/src/
rsync -a --exclude __pycache__ jetson/frame_server         Omni@<jetson>:~/omni_jetson_ws/src/
rsync -a --exclude __pycache__ jetson/rear_camera          Omni@<jetson>:~/omni_jetson_ws/src/
# Shared srv package — lives in src/ (the Pi builds it too) and MUST exist on both
# machines, or the Pi's behavior_node cannot talk to frame_server.
rsync -a --exclude __pycache__ src/omni_vision_msgs        Omni@<jetson>:~/omni_jetson_ws/src/
ssh Omni@<jetson> 'source /opt/ros/jazzy/setup.bash && cd ~/omni_jetson_ws && colcon build --symlink-install'
```

## Models  (`models/`)
- `yolo26n.onnx`, `yolo26n.pt` — portable model source (versioned here).
- The TensorRT engine (`yolo26n_fp16.engine`) is **device-specific and NOT in git** —
  rebuild it on the Jetson from the ONNX (also do this if the JetPack/TensorRT version
  changes):
  ```bash
  /usr/src/tensorrt/bin/trtexec --onnx=yolo26n.onnx --fp16 --saveEngine=yolo26n_fp16.engine
  ```
  head_detector's default `engine_path` is `/home/Omni/head_detector/models/yolo26n_fp16.engine`.

## Runtime deps on the Jetson (JetPack 7.2 / L4T r39)
`ros-jazzy-ros-base`, `ros-jazzy-depthai-ros`, `ros-jazzy-depthimage-to-laserscan`,
`ros-jazzy-foxglove-bridge`, system `tensorrt` python bindings, `cuda-python`
(`pip install --break-system-packages`), `cv2` (with GStreamer), `v4l-utils`.
