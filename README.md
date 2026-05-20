# OMNI

OMNI is an autonomous indoor robot powered by a Raspberry Pi 5. It uses the Gemini Live API for real-time voice conversation, Nav2 for autonomous navigation, and SLAM for mapping. You can wake it up by saying "Hey Mycroft", have a conversation, and ask it to navigate to rooms, report its battery, or explore the house.

---

## Hardware

| Component | Details |
|---|---|
| Compute | Raspberry Pi 5 (8GB) |
| Microphone | USB microphone (device index 0) |
| Speaker | USB speaker (device index 0) |
| Camera | Raspberry Pi AI Camera (IMX500) |
| LiDAR | LD19 (CP2102 USB adapter, `/dev/lidar`) |
| IMU | BNO085 (I2C, GAME_ROTATION_VECTOR mode) |
| ToF sensors | 6× VL53L0X (I2C via PCA9546 mux) |
| Drive | Differential drive, Arduino motor controller (`/dev/arduino`) |
| Chest display | ESP32 + OLED + 16×LED matrix (UART `/dev/ttyAMA0`) |
| Eye display | GC9A01 round TFT (SPI via lgpio) |
| Servos | PCA9685 PWM driver |
| Battery | 3-cell 11.4V LiPo with BMS (BLE) |

**TF tree:**
```
map → odom → base_link → lidar_link  (z=0.38m)
                       → imu_link    (x=0.129m, z=0.275m)
                       → camera_link (x=0.089m, z=0.695m)
```

---

## Software dependencies

All ROS2 packages are built from source (Debian Trixie does not have ROS2 apt packages). Non-ROS Python dependencies:

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
picamera2
```

Build the workspace:

```bash
cd ~/omni_ws
colcon build --symlink-install
source install/setup.bash
```

---

## Configuration

### API key

```bash
echo 'export GEMINI_API_KEY=your_key_here' >> ~/.bashrc
source ~/.bashrc
```

### System prompt and locations

Edit `src/behavior_node/config/omni_config.yaml` to change OMNI's personality, instructions, and named navigation locations.

---

## Launching

**Full stack** (all nodes — use this for normal operation):
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

| Package | Role |
|---|---|
| `behavior_node` | Central brain — Gemini Live session, wake word, state machine |
| `motor_control_node` | Arduino serial bridge — velocity commands, odometry |
| `imu_node` | BNO085 orientation publisher |
| `lidar_node` | LD19 scan publisher |
| `tof_node` | 6× VL53L0X range publishers |
| `camera_node` | IMX500 image + inference publisher |
| `bms_node` | BLE battery state publisher |
| `safety_node` | Velocity gate — stops motors on fault (`/cmd_vel_raw` → `/cmd_vel`) |
| `eye_node` | GC9A01 eye animation display |
| `servo_node` | PCA9685 head servo controller |
| `chest_node` | ESP32 OLED + LED matrix display |
| `slam_node` / `slam_toolbox` | Async online SLAM — publishes `/map` and `map→odom` TF |
| `nav_node` / `navigation2` | Nav2 stack — bt_navigator, planner, controller, behaviors |

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
