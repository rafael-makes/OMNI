from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='bms_node',
            executable='bms_node',
            name='bms_node',
            output='screen',
        ),
    ])
