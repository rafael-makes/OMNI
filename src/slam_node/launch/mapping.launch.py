"""
mapping.launch.py — one-command SLAM mapping bring-up for OMNI.

Launch-and-drive re-mapping: brings up the minimal set needed to build a fresh
map in one shot, so re-mapping doesn't mean juggling four terminals.

  motor_control_node   wheel odometry (publish_tf=False — yaw_fusion owns the TF)
  yaw_fusion_node      odom → base_link TF (wheel odom fused with IMU yaw)
  imu_node             /imu at 100 Hz (feeds yaw_fusion)
  lidar_node           /scan from the LD19
  slam_launch.py       slam_toolbox in MAPPING mode + base_link→lidar_link (z=1.210)
                       + all ToF static TFs (configure@3s, activate@8s internally)
  foxglove_bridge      so you can watch the map grow while you drive (optional)

Deliberately NOT included (mapping needs none of them):
  camera / OAK / Nav2 / behavior / tof_node / safety.

Teleop is kept OUT so it doesn't fight autostart — drive from its own terminal:
  ros2 run motor_control_node ps4_teleop         # PS4 pad
  ros2 run teleop_twist_keyboard teleop_twist_keyboard   # keyboard

Usage:
    ros2 launch slam_node mapping.launch.py
    # drive to cover the whole floor, watch it build in Foxglove, then save.

SAVING — the pose graph is what matters. localization_launch.py loads a
'.posegraph' (Nav2 runs NO map_server; slam_toolbox publishes /map live from the
graph). That comes from the serialize_map service, NOT the map_saver console
script (which only writes PGM/YAML — handy as a backup/preview, not loaded by
nav):
    # the one that counts — writes omni_home_map_v2.posegraph + .data:
    ros2 service call /slam_toolbox/serialize_map slam_toolbox/srv/SerializePoseGraph \
        '{filename: "/home/pi/omni_ws/maps/omni_home_map_v2"}'
    # optional PGM/YAML preview + backup:
    ros2 run slam_node map_saver omni_home_map_v2

SAVE GOTCHA: the name 'omni_home_map' OVERWRITES the current office map. Save the
fresh one under a NEW name and only swap it in (point localization at it) once
you've confirmed it localises well.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            LogInfo)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    slam_pkg = get_package_share_directory('slam_node')

    # ── Launch arguments ───────────────────────────────────────────────────────
    # Serial-port defaults match omni_full_launch.py so both launches agree.
    motor_port_arg = DeclareLaunchArgument(
        'motor_serial_port', default_value='/dev/arduino',
        description='Serial device for Arduino motor controller + odometry')
    lidar_port_arg = DeclareLaunchArgument(
        'lidar_serial_port', default_value='/dev/lidar',
        description='Serial device for LD19 LiDAR (publishes /scan)')
    use_foxglove_arg = DeclareLaunchArgument(
        'use_foxglove', default_value='true',
        description='Run foxglove_bridge so you can watch the map build')

    # ── IMU static TF ──────────────────────────────────────────────────────────
    # base_link → imu_link. Copied from omni_full_launch.py (z=0.625m, 2026-07-07).
    # z barely affects the 2D yaw fusion, but keeping the TF tree complete makes
    # Foxglove's TF display sane while mapping.
    imu_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_imu',
        arguments=['0.129', '0', '0.625', '0', '0', '0', 'base_link', 'imu_link'],
        output='screen',
    )

    # ── Sensor + odom nodes (copied EXACTLY from omni_full_launch.py) ───────────
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

    # ── slam_toolbox — MAPPING mode ────────────────────────────────────────────
    # slam_launch.py defaults to mapper_params_online_async.yaml (mode: mapping)
    # and publishes base_link→lidar_link (z=1.210) + the ToF static TFs, and
    # self-drives the lifecycle (configure@3s, activate@8s).
    slam_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_pkg, 'launch', 'slam_launch.py')
        ),
    )

    # ── Foxglove bridge (optional) ─────────────────────────────────────────────
    foxglove_node = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_foxglove')),
    )

    return LaunchDescription([
        motor_port_arg,
        lidar_port_arg,
        use_foxglove_arg,
        LogInfo(msg='[mapping] OMNI SLAM mapping — drive to cover the floor, '
                    'then serialize the map under a NEW name (not omni_home_map)'),
        imu_tf,
        motor_control_node,
        yaw_fusion_node,
        imu_node,
        lidar_node,
        slam_include,
        foxglove_node,
    ])
