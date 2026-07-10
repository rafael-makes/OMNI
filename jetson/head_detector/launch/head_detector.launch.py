"""Launch the Jetson head detector + the base_link -> camera_link static TF.

NOTE: the static-transform xyz/rpy below are PLACEHOLDERS for the head-mounted
IMX219 position — Rafael to measure on hardware and update. head_tracking_node
does not consume this TF, but it keeps the frame defined for other consumers.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    source = LaunchConfiguration('source')
    return LaunchDescription([
        DeclareLaunchArgument('source', default_value='csi',
                              description="'csi' for nvarguscamerasrc, or an image path for offline test"),
        Node(
            package='head_detector',
            executable='head_detector_node',
            name='head_detector_node',
            output='screen',
            emulate_tty=True,
            parameters=[{'source': source}],
        ),
        # base_link -> camera_link (head IMX219). PLACEHOLDER offsets.
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_camera_link',
            arguments=['0.10', '0.0', '0.95', '0', '0', '0', 'base_link', 'camera_link'],
            output='screen',
        ),
    ])
