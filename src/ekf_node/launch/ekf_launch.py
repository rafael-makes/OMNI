"""
ekf_launch.py — robot_localization EKF for OMNI robot

Fuses /odom (wheel encoders) + /imu/data (BNO085) → /odometry/filtered
and publishes the odom→base_link TF.

Run AFTER motor_control_node and imu_node are up.
motor_control_node must be launched with publish_tf:=false.

Usage:
  ros2 launch ekf_node ekf_launch.py
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ekf_params = os.path.join(
        get_package_share_directory('ekf_node'),
        'config', 'ekf_params.yaml'
    )

    return LaunchDescription([
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[ekf_params],
            remappings=[
                ('odometry/filtered', '/odometry/filtered'),
            ]
        ),
    ])
