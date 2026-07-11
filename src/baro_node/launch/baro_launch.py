from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='baro_node',
            executable='baro_node',
            name='baro_node',
            output='screen',
            parameters=[{
                'i2c_bus': 1,
                'i2c_address': 0x77,
                'publish_rate': 10.0,
                'filter_alpha': 0.9,
                'floor_height_m': 3.0,
                'start_floor': 0,
            }],
        ),
    ])
