# OMNI

OMNI is an autonomous indoor robot running on a **Raspberry Pi 5 paired with a Jetson Orin Nano**. It uses the Gemini Live API for real-time voice conversation, Nav2 for autonomous navigation, and SLAM for mapping. You can wake it up by saying "Omni", have a conversation, and ask it to navigate to rooms, report its battery, describe what it sees, or explore the house.

The Pi owns navigation, base control, sensors, and the audio bridge; the Jetson owns all
computer vision. See [ARCHITECTURE.md](ARCHITECTURE.md) for the full split and for where
new code belongs.

---

## Hardware

| Component | Details | Host |
|---|---|---|
| Compute | Raspberry Pi 5 (8GB) | — |
| Vision compute | Jetson Orin Nano (JetPack 7.2, ROS 2 Jazzy) | — |
| Microphone | USB microphone (device index 0) | Pi |
| Speaker | USB speaker (device index 0) | Pi |
| Head camera | USB webcam (1280×720 MJPG, 30fps max) | Jetson |
| Rear camera | IMX219 CSI (rear view + AprilTag docking) | Jetson |
| Depth camera | OAK-D Lite → `/oak/scan` for Nav2 obstacle layer | Jetson |
| LiDAR | LD19 (CP2102 USB adapter, `/dev/lidar`) | Pi |
| IMU | BNO085 (I2C, GAME_ROTATION_VECTOR mode) | Pi |
| ToF sensors | 6× VL53L0X (I2C via PCA9546 mux) | Pi |
| Barometer | BMP280 (I2C 0x77, floor detection) | Pi |
| Drive | Differential drive, Arduino motor controller (`/dev/arduino`) | Pi |
| Chest display | ESP32 + OLED + 16×LED matrix (UART `/dev/ttyAMA0`) | Pi |
| Eye display | GC9A01 round TFT (SPI via lgpio) | Pi |
| Servos | PCA9685 PWM driver (head pan/tilt) | Pi |
| Battery | 3-cell 11.4V LiPo with BMS (BLE) | Pi |

The two machines are joined by a direct Ethernet link (Pi `192.168.50.1`, Jetson
`192.168.50.2`) and share one ROS 2 Jazzy graph.

**TF tree** (heights updated 2026-07-07 when OMNI was made taller):
```
map → odom → base_link → lidar_link  (z=1.210m)   — slam_launch.py
                       → imu_link    (x=0.129m, z=0.625m)
                       → camera_link (x=0.089m, z=1.075m)
                       → oak         — published by the Jetson's depthai driver
```

---

## Software dependencies

On the **Pi**, all ROS2 packages are built from source (Debian Trixie does not have ROS2 apt packages). The **Jetson** uses apt ROS 2 Jazzy. Non-ROS Python dependencies on the Pi:

```
google-genai>=2.0.1
openwakeword
sounddevice
numpy
bleak
smbus2
adafruit-circuitpython-busdevice
adafruit-circuitpython-vl53l0x
lgpio
picamera2   # only needed by the decommissioned camera_node
```

### External ROS2 packages

Upstream sources (Nav2, slam_toolbox, BehaviorTree.CPP, foxglove_bridge, …) live in
`src/` but are **not tracked here** — they have their own git repos. `omni.repos` pins
what each was built from. From a fresh clone:

```bash
cd ~/omni_ws
vcs import src < omni.repos                                    # sudo apt install python3-vcstool
patch -d src/slam_toolbox -p1 < patches/slam_toolbox-headless.patch
```

The slam_toolbox patch is **not optional** — it makes the RViz2 plugin conditional so
the package configures without Qt5/rviz2, which the headless Pi doesn't have.

### Build

On the Pi (`jetson/` is skipped via `COLCON_IGNORE`):

```bash
cd ~/omni_ws
colcon build --symlink-install
source install/setup.bash
```

Jetson-side source lives in `jetson/`, but the Jetson builds from its own copies under
`~/omni_jetson_ws/src/` — edits here must be synced across before they take effect.

---

## Configuration

### System prompt and locations

Edit `src/behavior_node/config/omni_config.yaml` to change OMNI's personality, instructions, and named navigation locations.

---

## Launching

The Jetson vision stack must be up for head tracking and scene description to work.

**Jetson** (on the Jetson, `192.168.50.2`):
```bash
ros2 launch omni_jetson_bringup head_detector.launch.py
```

**Full stack** (on the Pi — all nodes, localization mode; use this for normal operation):
```bash
ros2 launch behavior_node omni_full_launch.py
```

**Behavior node only** (Gemini + wake word, no nav/SLAM):
```bash
ros2 launch behavior_node behavior_minimal_launch.py
```

**Common overrides:**
```bash
# Custom wake word sensitivity
ros2 launch behavior_node omni_full_launch.py wake_word_threshold:=0.4

# Different Gemini voice
ros2 launch behavior_node omni_full_launch.py gemini_voice:=Charon

# TCP mic from Pi Zero instead of local USB mic
ros2 launch behavior_node omni_full_launch.py tcp_mic_port:=9001
```

---

## Node overview

### Pi 5 — `src/`

| Package | Role |
|---|---|
| `behavior_node` | Central brain — Gemini Live session, wake word, state machine |
| `motor_control_node` | Arduino serial bridge — velocity commands, odometry |
| `yaw_fusion_node` | Fuses IMU yaw with wheel odometry; owns `odom→base_link` TF |
| `imu_node` | BNO085 orientation publisher |
| `lidar_node` | LD19 scan publisher |
| `tof_node` | 6× VL53L0X range publishers |
| `baro_node` | BMP280 pressure — relative floor detection for multi-map |
| `bms_node` | BLE battery state publisher |
| `safety_node` | Velocity gate — stops motors on fault (`/cmd_vel_raw` → `/cmd_vel`) |
| `stall_recovery_node` | Backs off / rotates out of a stall, handshakes with `safety_node` |
| `head_tracking_node` | Drives head servos from `/camera/faces` (Jetson), person-box fallback |
| `eye_node` | GC9A01 eye animation display |
| `servo_node` | PCA9685 head servo controller |
| `chest_node` | ESP32 OLED + LED matrix display |
| `omni_memory` | Persistent memory layer — per-person recall (Supabase over WireGuard) |
| `slam_node` / `slam_toolbox` | Async online SLAM — publishes `/map` and `map→odom` TF |
| `nav_node` / `navigation2` | Nav2 stack — bt_navigator, planner, controller, behaviors |
| `ekf_node` | robot_localization EKF |
| `camera_node` | **Decommissioned 2026-07-07** — IMX500 detection moved to the Jetson. Do not launch it; it double-publishes `/camera/detections` |

### Jetson Orin Nano — `jetson/`

| Package | Role |
|---|---|
| `head_detector` | YOLO26n (TensorRT) person detection → `/camera/detections`; YuNet faces → `/camera/faces`; face recognition + identity monitoring |
| `frame_server` | Serves newest camera frame as JPEG on `/vision/get_camera_frame` (backs `describe_scene`) |
| `rear_camera` | Owns the rear device — rear view + feeds `apriltag_ros` for docking |
| `omni_jetson_bringup` | Launch + config for the Jetson vision stack |

---

## State machine

OMNI's behavior is driven by a state machine in `behavior_node`:

| State | Meaning |
|---|---|
| `IDLE` | Wake word listening active, Gemini stream closed |
| `LISTENING` | Wake word heard, Gemini stream open, waiting for speech |
| `SPEAKING` | Gemini is playing an audio response |
| `NAVIGATING` | Nav2 driving to a goal |
| `EXPLORING` | Autonomous exploration mode |
| `ERROR` | Safety fault — Gemini stream open to react in character |

Current state is published on `/robot_state` (String, 10Hz).

---

## Audio topics

| Topic | Type | Description |
|---|---|---|
| `/audio/levels` | `Float32MultiArray` | 16 amplitude bands (0.0–1.0) at 10Hz during playback |
| `/audio/speech` | `String` | Transcript of recognized speech |
