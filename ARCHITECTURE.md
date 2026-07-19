# OMNI — Architecture

OMNI runs as a **distributed two-machine system**. A Raspberry Pi 5 owns the robot
body — navigation, motion, sensors, and voice — while a Jetson Orin Nano owns
everything vision. The two talk over a direct Ethernet link as a single ROS 2 Jazzy
graph (both machines run matched Jazzy builds).

```
┌─────────────────────────────┐         ┌─────────────────────────────┐
│      Raspberry Pi 5         │  ROS 2  │     Jetson Orin Nano        │
│  (192.168.50.1)             │◄───────►│     (192.168.50.2)          │
│                             │  eth0   │                             │
│  Nav2 / SLAM / EKF          │         │  YOLO26n person detection   │
│  Base + motor control       │         │  YuNet face detect/recog    │
│  IMU, ToF, LiDAR, baro, BMS │         │  Scene description frames   │
│  Audio bridge (Gemini Live) │         │  Identity monitoring        │
│  Behavior / safety / memory │         │  Rear camera + AprilTag     │
└─────────────────────────────┘         └─────────────────────────────┘
```

---

## Split of responsibilities

### Raspberry Pi 5 — navigation, base control, audio bridge

Everything that must be physically wired to the robot, plus anything latency-critical
to motion or conversation.

- **Navigation** — Nav2, `slam_toolbox`, `ekf_node`, `yaw_fusion_node`, `nav_node`
- **Base control** — `motor_control_node` (Arduino over serial), `servo_node`,
  `stall_recovery_node`, `safety_node`
- **Sensors** — `lidar_node`, `imu_node`, `tof_node`, `baro_node`, `bms_node`
- **Audio bridge** — `audio_node` (Gemini Live API, wake word, mic/speaker)
- **Displays + behavior** — `eye_node`, `chest_node`, `behavior_node`
- **Memory** — `omni_memory` (conversation memory, per-person recall)

### Jetson Orin Nano — computer vision, scene description, identity

Everything that needs GPU inference. The Jetson holds the camera devices and does the
heavy lifting locally, so only small messages cross the link.

- **Computer vision** — `head_detector`: YOLO26n as a native TensorRT engine, publishes
  `vision_msgs/Detection2DArray` on `/camera/detections`; YuNet face detection on
  `/camera/faces`
- **Scene description** — `frame_server`: serves the newest camera frame as JPEG over
  the `/vision/get_camera_frame` service. The Pi's `behavior_node` calls it when Gemini
  invokes `describe_scene`
- **Identity monitoring** — `head_detector/face_recognizer.py` and
  `head_detector/scripts/identity_monitor.py`; the Pi consumes the result and pairs it
  with `omni_memory`
- **Rear camera + docking** — `rear_camera` owns the rear device and feeds
  `apriltag_ros` for dock detection
- **Bringup** — `omni_jetson_bringup`: launch files and config for all of the above

---

## Directory structure

Where new code goes is determined by **which machine it runs on**.

```
omni_ws/
├── src/                      ← Pi 5 packages (built by colcon on the Pi)
│   ├── nav_node/
│   ├── motor_control_node/
│   ├── audio_node/
│   ├── behavior_node/
│   └── ...                   (~35 packages, incl. vendored ROS 2 sources)
│
└── jetson/                   ← Jetson Orin Nano packages
    ├── COLCON_IGNORE         ← keeps the Pi's build out of this tree
    ├── head_detector/
    ├── frame_server/
    ├── rear_camera/
    ├── omni_jetson_bringup/
    └── models/
```

### Rules for future code

1. **Pi-side node → new colcon package directly under `omni_ws/src/`.**
   Flat layout, one directory per package. No nesting.

2. **Jetson-side node → new package under `omni_ws/jetson/`.**
   Never under `src/` — the Pi would try to build it and fail on CUDA/TensorRT deps.

3. **`omni_ws/jetson/COLCON_IGNORE` must stay.** It is the only thing preventing the
   Pi's `colcon build` from walking into Jetson packages.

4. **`omni_ws/jetson/` is the version-controlled source of truth** for Jetson code, but
   it is *not* where the Jetson runs from. The Jetson builds copies under
   `~/omni_jetson_ws/src/`. Edits here must be synced across before they take effect.

5. **Shared message packages** (`omni_vision_msgs`, `omni_memory_msgs`, `vision_msgs`,
   `apriltag_msgs`) live in `omni_ws/src/` and are built on **both** machines. A wire
   change is not done until both sides are rebuilt and the sender actually emits the new
   format.

6. **Decide by hardware, not by topic.** If a node opens a camera or needs the GPU, it
   is Jetson-side. If it touches I2C, SPI, GPIO, the Arduino serial link, or audio
   devices, it is Pi-side.

---

## Cross-machine conventions

- **Prefer services over image topics.** Raw frames must not cross the link. Jetson
  nodes subscribe to camera feeds Jetson-locally and expose a service (as
  `frame_server` does), so only the reply travels.
- **One owner per camera device.** Two nodes opening the same `/dev/video*` will fail
  the OPEN. `rear_camera` owns the rear device and republishes for AprilTag and scene
  description alike.
- **Check QoS when a subscription is silent.** A `BEST_EFFORT` publisher and a
  `RELIABLE` subscriber match nothing, with no error. Verify with
  `ros2 topic info -v <topic>`.
- **Cold-start the first service call.** Fresh `rmw_fastrtps` clients drop the first
  reply. Warm up or retry on the cold path.

---

## Launch

Full system, from the Pi:

```bash
ros2 launch behavior_node omni_full_launch.py
```

Jetson vision stack, from the Jetson:

```bash
ros2 launch omni_jetson_bringup head_detector.launch.py
```
