"""
dock_test.launch.py — minimal stack to test back-in docking in isolation.

Brings up ONLY the motion + sensing + control chain — NO nav, behavior, Gemini,
or SLAM — so docking can be calibrated and tuned safely:

  motor_control_node   /cmd_vel -> motors
  safety_node          /cmd_vel_raw -> /cmd_vel gate (e-stop, tilt, proximity)
  tof_node             /tof/*  incl. the two rear sensors used for the stop
  dock_node            the controller

Also needs the Jetson dock detector publishing /detections (tag id 0):
  ssh Omni '~/start_dock_detector.sh'

Then, hand on the e-stop and a clear floor behind OMNI:
  ros2 service call /dock/start  std_srvs/srv/Trigger '{}'
  ros2 topic echo /dock/status
  ros2 service call /dock/cancel std_srvs/srv/Trigger '{}'
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory('dock_node'),
                       'config', 'dock_params.yaml')
    motor_port = DeclareLaunchArgument(
        'motor_serial_port', default_value='/dev/arduino')

    return LaunchDescription([
        motor_port,
        Node(package='motor_control_node', executable='motor_control_node',
             name='motor_control_node', output='screen', emulate_tty=True,
             parameters=[{'serial_port': LaunchConfiguration('motor_serial_port'),
                          'publish_tf': False}]),
        Node(package='safety_node', executable='safety_node',
             name='safety_node', output='screen', emulate_tty=True),
        Node(package='tof_node', executable='tof_node',
             name='tof_node', output='screen', emulate_tty=True),
        Node(package='dock_node', executable='dock_node',
             name='dock_node', output='screen', emulate_tty=True,
             parameters=[cfg]),
    ])
