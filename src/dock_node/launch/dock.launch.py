"""
dock.launch.py — run the back-in docking controller.

Prereqs that must already be up (dock_node only adds the controller):
  * Sensor stack incl. tof_node (publishes /tof/left_rear, /tof/right_rear) and
    motor_control + safety_node (so /cmd_vel_raw -> /cmd_vel reaches the motors).
  * Jetson dock detector publishing /detections (tag id 0). ids-only is enough —
    no pose / calibration needed for the pixel-servo controller.

Then trigger:
  ros2 service call /dock/start std_srvs/srv/Trigger '{}'
  ros2 topic echo /dock/status        # watch SEARCH -> ALIGN -> REVERSE -> DONE
  ros2 service call /dock/cancel std_srvs/srv/Trigger '{}'   # abort
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory('dock_node'),
                       'config', 'dock_params.yaml')
    return LaunchDescription([
        Node(
            package='dock_node',
            executable='dock_node',
            name='dock_node',
            output='screen',
            emulate_tty=True,
            parameters=[cfg],
        ),
    ])
