"""Launch frame_server alone.

head_detector must be running with publish_clean_image:=true, or every request
returns "no frame received yet". See omni_jetson_bringup for the combined launch.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('head_topic', default_value='/camera/image_clean/compressed'),
        DeclareLaunchArgument('max_frame_age', default_value='5.0'),
        Node(
            package='frame_server',
            executable='frame_server_node',
            name='frame_server',
            output='screen',
            parameters=[{
                'head_topic': LaunchConfiguration('head_topic'),
                'max_frame_age': LaunchConfiguration('max_frame_age'),
            }],
        ),
    ])
