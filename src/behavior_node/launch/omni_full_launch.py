"""
omni_full_launch.py — Complete OMNI robot bring-up.

Starts the full stack in dependency order:

  t=0s   Sensor nodes      — motor_control, imu, lidar, tof, camera
  t=0s   Support nodes     — bms, safety, eye, servo, chest
  t=0s   Static TFs        — base_link→imu_link, base_link→camera_link
  t=0s   slam_toolbox      — spawned but self-managed via slam_launch.py TimerActions:
  t=3s     └─ configure
  t=8s     └─ activate  ← /map and map→odom TF live from here
  t=12s  Nav2 stack        — bt_navigator + planner + controller + behavior + smoother
  t=0s   behavior_node     — Gemini Live brain (wake word active immediately)

TF tree:
  map → odom             slam_toolbox (async_slam_toolbox_node)
  odom → base_link       motor_control_node (TransformBroadcaster)
  base_link → lidar_link static_transform_publisher inside slam_launch.py (z=0.38m)
  base_link → imu_link   static_transform_publisher (this file)
  base_link → camera_link static_transform_publisher (this file)

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

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


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
    slam_params_arg = DeclareLaunchArgument(
        'slam_params_file',
        default_value=os.path.join(slam_pkg, 'config', 'mapper_params_online_async.yaml'),
        description='Path to slam_toolbox mapper_params_online_async.yaml',
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
        default_value='models/gemini-2.5-flash-native-audio-latest',
        description='Gemini model name passed to the Live API',
    )
    gemini_voice_arg = DeclareLaunchArgument(
        'gemini_voice',
        default_value='Algieba',
        description='Gemini Live voice name (e.g. Algieba, Charon, Fenrir)',
    )
    config_file_arg = DeclareLaunchArgument(
        'config_file_path',
        default_value=os.path.join(behavior_pkg, 'config', 'omni_config.yaml'),
        description='Absolute path to omni_config.yaml (system prompt + locations)',
    )
    wake_word_model_arg = DeclareLaunchArgument(
        'wake_word_model',
        default_value='hey_mycroft',
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

    # ── Static transforms ──────────────────────────────────────────────────────
    # base_link → lidar_link is handled inside slam_launch.py (z=0.38m, centred).
    # Update x/y/z below once you measure the actual mounting positions on hardware.

    imu_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_imu',
        # BNO085: x=0.129m forward, y=0, z=0.275m above floor plane.
        arguments=['0.129', '0', '0.275', '0', '0', '0', 'base_link', 'imu_link'],
        output='screen',
    )

    camera_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_camera',
        # IMX500: x=0.089m forward, y=0, z=0.695m above floor plane.
        arguments=['0.089', '0', '0.695', '0', '0', '0', 'base_link', 'camera_link'],
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
        }],
    )

    imu_node = Node(
        package='imu_node',
        executable='imu_node',
        name='imu_node',
        output='screen',
        emulate_tty=True,
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

    camera_node = Node(
        package='camera_node',
        executable='camera_node',
        name='camera_node',
        output='screen',
        emulate_tty=True,
    )

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

    # ── SLAM ───────────────────────────────────────────────────────────────────
    # Delegates entirely to slam_launch.py, which:
    #   - Starts async_slam_toolbox_node as a LifecycleNode
    #   - Starts base_link → lidar_link static TF publisher
    #   - Sends configure at t=3s, activate at t=8s via internal TimerActions
    # use_lifecycle_manager=false keeps slam_toolbox self-managed, separate from
    # Nav2's lifecycle_manager (which does not include slam_toolbox in node_names).

    slam_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_pkg, 'launch', 'slam_launch.py')
        ),
        launch_arguments=[
            ('use_sim_time',          LaunchConfiguration('use_sim_time')),
            ('slam_params_file',      LaunchConfiguration('slam_params_file')),
            ('autostart',             'true'),
            ('use_lifecycle_manager', 'false'),
        ],
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
        }],
    )

    # ── Assembly ───────────────────────────────────────────────────────────────

    return LaunchDescription([
        # Arguments
        use_sim_time_arg,
        slam_params_arg,
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
        # Static TF
        LogInfo(msg='[omni_full] Starting OMNI full stack'),
        imu_tf,
        camera_tf,
        # Sensors (t=0)
        motor_control_node,
        imu_node,
        lidar_node,
        tof_node,
        camera_node,
        # Support (t=0)
        bms_node,
        safety_node,
        # Peripherals (t=0)
        eye_node,
        servo_node,
        chest_node,
        # SLAM (configure@3s, activate@8s — internal timers)
        slam_include,
        # Nav2 (t=12s — waits for SLAM to be active)
        nav_delayed,
        # Brain (t=0 — wake word active immediately)
        behavior_node,
    ])
