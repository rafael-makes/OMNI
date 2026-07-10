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
- **`omni_jetson_bringup`** — launch + config:
  - `head_detector.launch.py` — head USB cam detector (848x480 MJPG, HHWei by-id).
  - `oak_scan.launch.py` — OAK-D Lite depth → `/oak/scan` (via `depthimage_to_laserscan`)
    for the Pi's Nav2 obstacle layer; publishes `base_link -> oak` TF.
  - `config/oak_lite_depth.yaml` — depth-only OAK pipeline + mount TF.

## Deploy to the Jetson
```bash
rsync -a --exclude __pycache__ jetson/head_detector       Omni@<jetson>:~/omni_jetson_ws/src/
rsync -a --exclude __pycache__ jetson/omni_jetson_bringup  Omni@<jetson>:~/omni_jetson_ws/src/
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
