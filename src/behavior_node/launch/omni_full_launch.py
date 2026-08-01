"""
omni_full_launch.py — Complete OMNI robot bring-up.

Starts the full stack in dependency order:

  t=0s   Sensor nodes      — motor_control, yaw_fusion, imu, lidar, tof
                             (person detection runs on the Jetson, not here)
  t=0s   Support nodes     — bms, safety, stall_recovery, eye, servo, chest
  t=0s   Static TFs        — base_link→imu_link, base_link→camera_link
  ~t=0s  Floor resolution  — AprilTag dock → BMP280 fallback picks map_file
                             (blocks ~6s), then baro_node starts
  t=0s   slam_toolbox      — spawned but self-managed via slam_launch.py TimerActions:
  t=3s     └─ configure
  t=8s     └─ activate  ← /map and map→odom TF live from here
  t=12s  Nav2 stack        — bt_navigator + planner + controller + behavior + smoother
  t=0s   behavior_node     — Gemini Live brain (wake word active immediately)
  t=0s   omni_memory       — persistent memory layer (Supabase over WireGuard);
                             behavior_node retrieves on wake / stores at chat end

TF tree (heights updated 2026-07-07 when OMNI was made taller):
  map → odom             slam_toolbox (async_slam_toolbox_node)
  odom → base_link       yaw_fusion_node (motor_control publish_tf=False)
  base_link → lidar_link static_transform_publisher inside the included slam pkg
                         launch — localization_launch.py on this path, NOT
                         slam_launch.py (mapping). See the z-mismatch note below.
  base_link → imu_link   static_transform_publisher (this file, z=0.625m)
  base_link → camera_link static_transform_publisher (this file, z=1.075m)
  base_link → oak        published by the Jetson's depthai driver

cmd_vel chain (safety-gated):
  Nav2/behavior_server → /cmd_vel_raw → safety_node → /cmd_vel → motor_control_node

audio_node is intentionally excluded — behavior_node owns the Gemini Live session
and the USB audio device. Running both creates duplicate WebSocket connections and
two processes fighting over the same ALSA device.

Usage:
    ros2 launch behavior_node omni_full_launch.py

Override examples:
    ros2 launch behavior_node omni_full_launch.py motor_serial_port:=/dev/ttyUSB0
    ros2 launch behavior_node omni_full_launch.py wake_word_threshold:=0.7
    ros2 launch behavior_node omni_full_launch.py nav_params_file:=/path/to/custom.yaml

GEMINI_API_KEY must be exported before launching:
    export GEMINI_API_KEY=your_key_here
"""

import os
import subprocess

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    SetLaunchConfiguration,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():

    # ── Package share directories ──────────────────────────────────────────────
    behavior_pkg = get_package_share_directory('behavior_node')
    slam_pkg     = get_package_share_directory('slam_node')
    nav_pkg      = get_package_share_directory('nav_node')

    # ── Launch arguments ───────────────────────────────────────────────────────

    # Global
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock — false for real hardware, true for Gazebo',
    )

    # SLAM
    slam_map_arg = DeclareLaunchArgument(
        'map_file',
        default_value='auto',
        description='Pose-graph path prefix, or "auto" to pick by floor '
                    '(AprilTag dock -> barometer fallback) at boot',
    )

    # Nav2
    nav_params_arg = DeclareLaunchArgument(
        'nav_params_file',
        default_value=os.path.join(nav_pkg, 'config', 'nav2_params.yaml'),
        description='Path to Nav2 nav2_params.yaml',
    )

    # Hardware serial ports (most likely to need per-machine override)
    motor_port_arg = DeclareLaunchArgument(
        'motor_serial_port',
        default_value='/dev/arduino',
        description='Serial device for Arduino motor controller + odometry',
    )
    lidar_port_arg = DeclareLaunchArgument(
        'lidar_serial_port',
        default_value='/dev/lidar',
        description='Serial device for LD19 LiDAR (publishes /scan)',
    )
    chest_port_arg = DeclareLaunchArgument(
        'chest_serial_port',
        default_value='/dev/ttyAMA0',
        description='UART device for chest_node (OLED display + button board)',
    )

    # behavior_node — identical to behavior_minimal_launch.py
    gemini_model_arg = DeclareLaunchArgument(
        'gemini_model',
        # Must be a tool-capable Live model: native-audio models reject `tools`
        # (function calling) with WS 1008 and drop mid-turn. See behavior_node.py.
        default_value='models/gemini-3.1-flash-live-preview',
        description='Gemini model name passed to the Live API',
    )
    gemini_voice_arg = DeclareLaunchArgument(
        'gemini_voice',
        default_value='Algieba',
        description='Gemini Live voice name (e.g. Algieba, Charon, Fenrir)',
    )
    config_file_arg = DeclareLaunchArgument(
        'config_file_path',
        default_value='/home/pi/omni_config.yaml',
        description='Absolute path to omni_config.yaml (system prompt + locations)',
    )
    wake_word_model_arg = DeclareLaunchArgument(
        'wake_word_model',
        default_value='omni',
        description='openwakeword model name, without .onnx extension',
    )
    wake_word_threshold_arg = DeclareLaunchArgument(
        'wake_word_threshold',
        default_value='0.5',
        description='Wake word confidence threshold (0.0–1.0); lower = more sensitive',
    )
    wake_word_suppress_arg = DeclareLaunchArgument(
        'wake_word_startup_suppress',
        default_value='1.5',
        description='Seconds to suppress wake word scoring after detector restart (drains speaker bleed)',
    )
    conv_timeout_arg = DeclareLaunchArgument(
        'conversation_timeout',
        default_value='30.0',
        description='Seconds of Gemini silence before closing stream and returning to IDLE',
    )
    idle_timeout_arg = DeclareLaunchArgument(
        'idle_return_timeout',
        default_value='30.0',
        description='Seconds before an unacknowledged LISTENING state auto-returns to IDLE',
    )
    mic_index_arg = DeclareLaunchArgument(
        'mic_device_index',
        default_value='0',
        description='sounddevice input device index for the local USB microphone',
    )
    speaker_index_arg = DeclareLaunchArgument(
        'speaker_device_index',
        default_value='0',
        description='sounddevice output device index for the USB speaker',
    )
    tcp_mic_arg = DeclareLaunchArgument(
        'tcp_mic_port',
        default_value='0',
        description='TCP port to receive stereo PCM from Pi Zero mic (0 = use local mic)',
    )
    memory_enabled_arg = DeclareLaunchArgument(
        'memory_enabled',
        default_value='true',
        description='Enable persistent memory (retrieve on wake / store at chat end). '
                    'false skips all memory calls (no omni_memory dependency).',
    )
    presence_timeout_arg = DeclareLaunchArgument(
        'presence_timeout',
        default_value='10.0',
        description='Seconds with no person seen before the wake word disarms. Needs the '
                    'Jetson head_detector to see anyone; set very high (e.g. 100000) to keep '
                    'the wake word always armed when running without person detection.',
    )
    memory_env_file_arg = DeclareLaunchArgument(
        'memory_env_file',
        default_value='/home/pi/omni_ws/src/omni_memory/.env',
        description='.env with SUPABASE_URL/SUPABASE_SERVICE_KEY for the omni_memory node',
    )

    # ── Proactive check-in (Session 9) ─────────────────────────────────────────
    # Exposed as launch arguments specifically so a live test does not require
    # editing and rebuilding this file. The DEFAULTS ARE THE PRODUCTION ONES —
    # an hour of dwell, hours of cooldown, quiet after 21:00 — which are correct
    # for daily use and unusable for a test you want to finish today. See the
    # test recipe in behavior_node/CLAUDE.md for the dialled-down set.
    check_in_enabled_arg = DeclareLaunchArgument(
        'check_in_enabled', default_value='true',
        description='Enable proactive check-ins ("what are you working on?")',
    )
    check_in_min_dwell_arg = DeclareLaunchArgument(
        'check_in_min_dwell', default_value='3600.0',
        description='Seconds at one spot before a check-in is considered. '
                    'Lower this for testing; 3600 is the production bar.',
    )
    check_in_min_battery_arg = DeclareLaunchArgument(
        'check_in_min_battery', default_value='40.0',
        description='Battery %% floor for a check-in (a round trip costs more '
                    'than a greeting, so this is well above the greeting floor)',
    )
    check_in_quiet_start_arg = DeclareLaunchArgument(
        'check_in_quiet_start', default_value='21:00',
        description='HH:MM local. Set equal to check_in_quiet_end to disable '
                    'quiet hours entirely (useful when testing in the evening).',
    )
    check_in_quiet_end_arg = DeclareLaunchArgument(
        'check_in_quiet_end', default_value='08:00',
        description='HH:MM local; the quiet window wraps midnight',
    )
    check_in_global_cooldown_arg = DeclareLaunchArgument(
        'check_in_global_cooldown', default_value='7200.0',
        description='Seconds after ANY interaction with that person',
    )
    check_in_no_cooldown_arg = DeclareLaunchArgument(
        'check_in_no_cooldown', default_value='14400.0',
        description='Seconds, per zone, after a "no"',
    )
    check_in_not_now_cooldown_arg = DeclareLaunchArgument(
        'check_in_not_now_cooldown', default_value='3600.0',
        description='Seconds, per zone, after a "not now" or silence',
    )
    check_in_silence_timeout_arg = DeclareLaunchArgument(
        'check_in_silence_timeout', default_value='15.0',
        description='Seconds of silence after the opener before OMNI reads it '
                    'as "not now" and quietly leaves. It never re-asks.',
    )
    check_in_zones_arg = DeclareLaunchArgument(
        'check_in_zones', default_value='',
        description='COMMA-SEPARATED zones a check-in may happen in. Empty = '
                    "trust event_generator's dwell_zones.",
    )

    # ── Static transforms ──────────────────────────────────────────────────────
    # base_link → lidar_link is handled inside the included slam pkg launch, which on
    # THIS path is localization_launch.py (see slam_include below), not slam_launch.py.
    # Both slam pkg launches now publish z=1.210m. They had diverged: the 2026-07-07
    # mast change updated slam_launch.py only, leaving localization_launch.py — the one
    # THIS file includes — on the old 0.825m for ~12 days. Keep the two in sync.
    #
    # Heights below are measured, not placeholders — updated 2026-07-07. Re-measure if
    # the mast changes again; the old map is stale.

    imu_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_imu',
        # BNO085: x=0.129m forward, y=0, z=0.625m above floor plane (raised from
        # 0.275m when OMNI was made taller, 2026-07-07). z barely affects the 2D
        # yaw fusion; x/y unchanged (only the mast height grew).
        arguments=['0.129', '0', '0.625', '0', '0', '0', 'base_link', 'imu_link'],
        output='screen',
    )

    camera_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_camera',
        # Defines camera_link so the Jetson's /camera/detections (frame_id=camera_link)
        # has a frame. NOTE: PLACEHOLDER — offsets are the old IMX500 head position and
        # this is STATIC, but the IMX219 now rides the pan/tilt head, so camera_link
        # actually moves. Fine while head_tracking works in pixel space; revisit if any
        # node uses camera_link geometrically (then publish it dynamically off servo state).
        # z=1.075m measured 2026-07-07 (IMX219 head height on the taller OMNI). x is
        # still the old IMX500 estimate — head is a placeholder anyway.
        arguments=['0.089', '0', '1.075', '0', '0', '0', 'base_link', 'camera_link'],
        output='screen',
    )

    # ── Sensor nodes ───────────────────────────────────────────────────────────

    motor_control_node = Node(
        package='motor_control_node',
        executable='motor_control_node',
        name='motor_control_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'serial_port': LaunchConfiguration('motor_serial_port'),
            'publish_tf': False,   # yaw_fusion_node owns odom→base_link TF
        }],
    )

    yaw_fusion_node = Node(
        package='yaw_fusion_node',
        executable='yaw_fusion_node',
        name='yaw_fusion_node',
        output='screen',
        emulate_tty=True,
    )

    imu_node = Node(
        package='imu_node',
        executable='imu_node',
        name='imu_node',
        output='screen',
        emulate_tty=True,
        parameters=[{'publish_rate_hz': 100.0}],
    )

    lidar_node = Node(
        package='lidar_node',
        executable='lidar_node',
        name='lidar_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'serial_port': LaunchConfiguration('lidar_serial_port'),
        }],
    )

    tof_node = Node(
        package='tof_node',
        executable='tof_node',
        name='tof_node',
        output='screen',
        emulate_tty=True,
    )

    # camera_node (Pi IMX500) DECOMMISSIONED 2026-07-07 — person detection moved to
    # the Jetson (IMX219 + YOLO26n TensorRT). The Jetson's head_detector node now
    # publishes /camera/detections and /camera/status over the network. Do NOT run a
    # Pi camera_node here or it double-publishes /camera/detections and conflicts.
    # Jetson launch: `ros2 run head_detector head_detector_node --ros-args -p source:=csi`

    # ── BMS node ───────────────────────────────────────────────────────────────
    # Publishes /battery/status — behavior_node reads this for report_status().

    bms_node = Node(
        package='bms_node',
        executable='bms_node',
        name='bms_node',
        output='screen',
        emulate_tty=True,
    )

    # ── Safety node ────────────────────────────────────────────────────────────
    # Sits between Nav2/behavior_server (/cmd_vel_raw) and the motors (/cmd_vel).
    # Must be running before Nav2 sends any velocity commands.

    safety_node = Node(
        package='safety_node',
        executable='safety_node',
        name='safety_node',
        output='screen',
        emulate_tty=True,
    )

    # ── Peripheral nodes ───────────────────────────────────────────────────────

    eye_node = Node(
        package='eye_node',
        executable='eye_node',
        name='eye_node',
        output='screen',
        emulate_tty=True,
    )

    servo_node = Node(
        package='servo_node',
        executable='servo_node',
        name='servo_node',
        output='screen',
        emulate_tty=True,
    )

    # Drives head pan/tilt off the Jetson's /camera/faces (person boxes as fallback).
    # Gains/travel limits keep the node defaults (calibrated live, live-tunable via
    # `ros2 param set /head_tracking_node ...`). Its tilt envelope is coupled to
    # head_tilt.neutral_pw in config/servo_config.yaml — see the note there.
    # Safe at t=0: servo_node creates its /servo_commands subscription only after its
    # one-at-a-time init sweep finishes, so nothing lands mid-sweep.
    #
    # WANDER DISABLED 2026-07-26 — the head was in constant motion because idle drift
    # (IDLE/LISTENING) and gaze choreography (NAVIGATING/EXPLORING) both roam the head
    # when no real subject is tracked. Both are off here:
    #   drift_enabled=False   → head returns to neutral and holds when nobody visible
    #   gaze_states=''        → NAVIGATING/EXPLORING snap to neutral and hold (no roam)
    # Real face/person tracking (tracking_states) and instant-orient on walk-ins are
    # untouched. To restore the "alive" motion, drop these two overrides.
    head_tracking_node = Node(
        package='head_tracking_node',
        executable='head_tracking_node',
        name='head_tracking_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'drift_enabled': False,
            'gaze_states':   '',
        }],
    )

    chest_node = Node(
        package='chest_node',
        executable='chest_node',
        name='chest_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'serial_port': LaunchConfiguration('chest_serial_port'),
        }],
    )

    # ── Boot floor resolver ────────────────────────────────────────────────────
    # When map_file == 'auto', pick the map by which floor OMNI booted on:
    # AprilTag dock tag first (Jetson /detections; id 0 = main, id 1 = basement),
    # barometer fallback if no tag is seen. Runs baro_floor_resolver in its own
    # process — which reads /detections and the BMP280 directly — and captures the
    # chosen map path from stdout. An explicit map_file:=<path> skips all of this.
    #
    # Placed before baro_node below so the resolver's direct BMP280 read at 0x77
    # doesn't overlap baro_node's polling of the same device.
    _DEFAULT_MAP = '/home/pi/omni_ws/maps/omni_home_map'

    def _resolve_map_file(context, *args, **kwargs):
        requested = LaunchConfiguration('map_file').perform(context)
        if requested != 'auto':
            return []  # explicit override — respect it, resolver not run
        map_path = _DEFAULT_MAP
        try:
            result = subprocess.run(
                ['ros2', 'run', 'baro_node', 'baro_floor_resolver',
                 '--quiet', '--tag-timeout', '4', '--seconds', '2'],
                capture_output=True, text=True, timeout=25)
            lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
            if lines and lines[-1].startswith('/'):
                map_path = lines[-1].strip()
                note = f'[omni_full] boot floor resolver selected map: {map_path}'
            else:
                note = (f'[omni_full] floor resolver gave no map; '
                        f'using default {map_path}')
        except Exception as e:  # noqa: BLE001 — never let boot fail on this
            note = f'[omni_full] floor resolver error ({e}); using default {map_path}'
        return [
            LogInfo(msg=note),
            SetLaunchConfiguration('map_file', map_path),
        ]

    resolve_map_action = OpaqueFunction(function=_resolve_map_file)

    # Barometer node — live floor tracking + accepts /baro/set_floor re-anchoring
    # from the dock behaviour. Starts after the resolver (see note above).
    baro_node = Node(
        package='baro_node',
        executable='baro_node',
        name='baro_node',
        output='screen',
    )

    # ── SLAM — localization mode ───────────────────────────────────────────────
    # Loads the saved pose graph and localises within the existing map.
    # Does not modify the map. Delegates to localization_launch.py which:
    #   - Starts localization_slam_toolbox_node as a LifecycleNode
    #   - Starts base_link → lidar_link + all ToF static TF publishers
    #   - Sends configure at t=3s, activate at t=8s via internal TimerActions

    slam_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_pkg, 'launch', 'localization_launch.py')
        ),
        launch_arguments=[
            ('use_sim_time',          LaunchConfiguration('use_sim_time')),
            ('map_file',              LaunchConfiguration('map_file')),
            ('autostart',             'true'),
            ('use_lifecycle_manager', 'false'),
        ],
    )

    # ── Foxglove bridge ────────────────────────────────────────────────────────
    foxglove_node = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        output='screen',
    )

    tof_viz_node = Node(
        package='tof_viz_node',
        executable='tof_viz_node',
        name='tof_viz_node',
        output='screen',
    )

    # ── Stall recovery ─────────────────────────────────────────────────────────
    stall_recovery_node = Node(
        package='stall_recovery_node',
        executable='stall_recovery_node',
        name='stall_recovery_node',
        output='screen',
        emulate_tty=True,
    )

    # ── Docking controller ─────────────────────────────────────────────────────
    # Idle until behavior_node's `dock` mission calls /dock/start. Needs the Jetson
    # dock detector (/detections) for the visual back-in — that runs on the Jetson,
    # not here. See dock_node/CLAUDE-less package: dock_test.launch.py for standalone.
    dock_node = Node(
        package='dock_node',
        executable='dock_node',
        name='dock_node',
        output='screen',
        emulate_tty=True,
        parameters=[os.path.join(
            get_package_share_directory('dock_node'), 'config', 'dock_params.yaml')],
    )

    # ── Boot self-localization ─────────────────────────────────────────────────
    # slam_toolbox localization mode waits for an /initialpose; it does not self-
    # locate. At t=10s (after SLAM activates at t=8s, before Nav2 at t=12s) run
    # boot_self_localize, which decides where OMNI is and seeds /initialpose:
    #   DOCKED (dock tag id 0 + rear ToF at docking distance) -> the saved,
    #     drift-free DOCKED pose (floors.yaml dock_pose).
    #   NOT DOCKED -> the persisted last-known pose (pose_writer saved it last run);
    #     slam_toolbox's ±0.75 m correlation window scan-matches it to truth.
    #   NEITHER -> publishes nothing; a manual Foxglove 2D Pose Estimate is the
    #     fallback ONLY in the genuinely-unknown case, not every boot.
    # The old hardcoded (-3.4144, 3.5072) pose is gone — it was captured on the
    # OLD map (omni_home_map, Jun 19) and is wrong on omni_home_map_v2.
    # The floor comes from the resolved map_file (the floor resolver already picked
    # the map by AprilTag/barometer above).
    self_localize_action = TimerAction(
        period=10.0,
        actions=[
            LogInfo(msg='[omni_full] t=10s — boot self-localization (docked pose / '
                        'persisted pose / manual fallback)'),
            ExecuteProcess(
                cmd=[
                    'ros2', 'run', 'baro_node', 'boot_self_localize',
                    '--map', LaunchConfiguration('map_file'),
                    '--dock-range', '0.30',   # docked ToF measured 0.18-0.23 m; 0.20 was too tight (2026-08-01)
                    '--sense-time', '3.0',
                    '--republish', '5',
                ],
                output='screen',
            ),
        ],
    )

    # Persists map->base_link periodically so the NEXT boot can self-localize
    # without a manual pose. Placed after the floor resolver so it reads the
    # resolved map_file (the frame is per-floor). Self-delays writing until
    # localization has converged (start_delay). See baro_node/pose_writer.py.
    pose_writer_node = Node(
        package='baro_node',
        executable='pose_writer',
        name='pose_writer',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'map_file':     LaunchConfiguration('map_file'),
            'write_period': 5.0,
            'start_delay':  25.0,   # > Nav2 start (t=12s) + a margin to converge
        }],
    )

    # ── Nav2 stack (delayed) ───────────────────────────────────────────────────
    # slam_toolbox activates at t=8s. Nav2 starts at t=12s (4s margin) so the
    # map→odom TF exists before bt_navigator begins bonding. Nav2's lifecycle_manager
    # retries bond checks automatically, so if SLAM is slow it self-heals.

    nav_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav_pkg, 'launch', 'nav_launch.py')
        ),
        launch_arguments=[
            ('params_file',  LaunchConfiguration('nav_params_file')),
            ('use_sim_time', LaunchConfiguration('use_sim_time')),
        ],
    )

    nav_delayed = TimerAction(
        period=12.0,
        actions=[
            LogInfo(msg='[omni_full] t=12s — launching Nav2 stack (SLAM should be active)'),
            nav_include,
        ],
    )

    # ── behavior_node ──────────────────────────────────────────────────────────

    behavior_node = Node(
        package='behavior_node',
        executable='behavior_node',
        name='behavior_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'gemini_model':               LaunchConfiguration('gemini_model'),
            'gemini_voice':               LaunchConfiguration('gemini_voice'),
            'config_file_path':           LaunchConfiguration('config_file_path'),
            'wake_word_model':            LaunchConfiguration('wake_word_model'),
            'wake_word_threshold':        LaunchConfiguration('wake_word_threshold'),
            'wake_word_startup_suppress': LaunchConfiguration('wake_word_startup_suppress'),
            'conversation_timeout':       LaunchConfiguration('conversation_timeout'),
            'idle_return_timeout':        LaunchConfiguration('idle_return_timeout'),
            'mic_device_index':           LaunchConfiguration('mic_device_index'),
            'speaker_device_index':       LaunchConfiguration('speaker_device_index'),
            'tcp_mic_port':               LaunchConfiguration('tcp_mic_port'),
            'memory_enabled':             LaunchConfiguration('memory_enabled'),
            'presence_timeout':           LaunchConfiguration('presence_timeout'),
            'check_in_enabled':           LaunchConfiguration('check_in_enabled'),
            'check_in_min_dwell':         LaunchConfiguration('check_in_min_dwell'),
            'check_in_min_battery':       LaunchConfiguration('check_in_min_battery'),
            # Force STRING type: 'HH:MM' like '21:00' is otherwise read by YAML as a
            # base-60 int (21*60=1260), and the node declares these as strings → crash.
            'check_in_quiet_start':       ParameterValue(LaunchConfiguration('check_in_quiet_start'), value_type=str),
            'check_in_quiet_end':         ParameterValue(LaunchConfiguration('check_in_quiet_end'), value_type=str),
            'check_in_global_cooldown':   LaunchConfiguration('check_in_global_cooldown'),
            'check_in_no_cooldown':       LaunchConfiguration('check_in_no_cooldown'),
            'check_in_not_now_cooldown':  LaunchConfiguration('check_in_not_now_cooldown'),
            'check_in_silence_timeout':   LaunchConfiguration('check_in_silence_timeout'),
            'check_in_zones':             LaunchConfiguration('check_in_zones'),
        }],
    )

    # ── omni_memory ────────────────────────────────────────────────────────────
    # Persistent memory layer (Supabase/pgvector over WireGuard). behavior_node is
    # a SOFT client — if this node is absent, OMNI still converses. Kept as a plain
    # t=0 node so it's discovered well before the first wake word.
    omni_memory_node = Node(
        package='omni_memory',
        executable='omni_memory_node',
        name='omni_memory',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'env_file': LaunchConfiguration('memory_env_file'),
        }],
    )

    # ── Pre-clean stale FastDDS SHM locks ────────────────────────────────────────
    # Every relaunch leaves /dev/shm/fastrtps_* segments behind; over a session they
    # pile up until fresh `ros2` CLI processes can no longer join DDS discovery (empty
    # topic echo / hz / tf2_echo, "topic not published yet") — the running stack is
    # fine, but introspection goes blind. Wipe them before any node starts. Runs via a
    # shell so the glob expands; harmless when there's nothing to remove. Safe here
    # because a prior stack must be down before this launch runs.
    clean_shm = ExecuteProcess(
        cmd=['bash', '-c',
             'rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* 2>/dev/null || true'],
        output='screen',
    )

    # ── Assembly ───────────────────────────────────────────────────────────────

    return LaunchDescription([
        # Arguments
        use_sim_time_arg,
        slam_map_arg,
        nav_params_arg,
        motor_port_arg,
        lidar_port_arg,
        chest_port_arg,
        gemini_model_arg,
        gemini_voice_arg,
        config_file_arg,
        wake_word_model_arg,
        wake_word_threshold_arg,
        wake_word_suppress_arg,
        conv_timeout_arg,
        idle_timeout_arg,
        mic_index_arg,
        speaker_index_arg,
        tcp_mic_arg,
        memory_enabled_arg,
        memory_env_file_arg,
        presence_timeout_arg,
        check_in_enabled_arg,
        check_in_min_dwell_arg,
        check_in_min_battery_arg,
        check_in_quiet_start_arg,
        check_in_quiet_end_arg,
        check_in_global_cooldown_arg,
        check_in_no_cooldown_arg,
        check_in_not_now_cooldown_arg,
        check_in_silence_timeout_arg,
        check_in_zones_arg,
        # Pre-clean stale FastDDS SHM locks before anything starts (keeps ros2 CLI
        # introspection working across relaunches). See note above.
        clean_shm,
        # Static TF
        LogInfo(msg='[omni_full] Starting OMNI full stack — localization mode'),
        imu_tf,
        camera_tf,
        # Sensors (t=0)
        motor_control_node,
        yaw_fusion_node,
        imu_node,
        lidar_node,
        tof_node,
        # camera_node removed — person detection now runs on the Jetson (see note above)
        # Support (t=0)
        bms_node,
        safety_node,
        # Peripherals (t=0)
        eye_node,
        servo_node,
        head_tracking_node,
        chest_node,
        # Boot floor resolution: AprilTag dock -> baro fallback sets map_file.
        # Blocks ~6s while it reads the tag topic + BMP280, then baro_node starts.
        resolve_map_action,
        baro_node,
        # SLAM localization (configure@3s, activate@8s — internal timers)
        slam_include,
        # Boot self-localization (t=10s — after SLAM active, before Nav2 starts)
        self_localize_action,
        # Persist pose for the next boot's self-localization (self-delayed writes)
        pose_writer_node,
        # Nav2 (t=12s — waits for SLAM to be active)
        nav_delayed,
        # Brain (t=0 — wake word active immediately)
        behavior_node,
        # Persistent memory (t=0 — soft dependency of behavior_node)
        omni_memory_node,
        # Tools
        foxglove_node,
        tof_viz_node,
        stall_recovery_node,
        dock_node,
    ])
