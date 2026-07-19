"""Launch the rear camera owner alone (scene-description feed only).

This is what runs in normal operation: a 2fps clean JPEG feed for frame_server,
so "what's behind you?" is answered from cache with no device-open latency.

For docking, use omni_jetson_bringup's dock_apriltag_shared.launch.py instead —
it launches this node with publish_raw_image:=true and points apriltag_ros at the
published topic. Do NOT run the legacy dock_apriltag.launch.py at the same time
as this node: it opens the device directly and the second opener gets -EBUSY.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_REAR_CAM = ('/dev/v4l/by-id/'
             'usb-GENERAL_2K_HD_Camera-video-index0')


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'device', default_value=_REAR_CAM,
            description='V4L2 by-id path of the rear 2K camera'),
        DeclareLaunchArgument(
            'publish_raw_image', default_value='false',
            description='Also publish /camera/rear/image_raw for apriltag_ros'),
        DeclareLaunchArgument(
            'decode_fps', default_value='5.0',
            description='Frames actually decoded per second. The wire rate is '
                        'fixed at 30 (the only rate this cam negotiates); this '
                        'is the real throttle.'),
        DeclareLaunchArgument('clean_image_fps', default_value='2.0'),
        Node(
            package='rear_camera',
            executable='rear_camera_node',
            name='rear_camera',
            output='screen',
            parameters=[{
                'device': LaunchConfiguration('device'),
                'capture_width': 640,
                'capture_height': 480,
                # 30 is the ONLY MJPG rate this camera will negotiate; asking for
                # less fails the open outright. Throttle via decode_fps.
                'capture_fps': 30,
                'decode_fps': LaunchConfiguration('decode_fps'),
                'publish_clean_image': True,
                'clean_image_fps': LaunchConfiguration('clean_image_fps'),
                'publish_raw_image': LaunchConfiguration('publish_raw_image'),
            }],
        ),
    ])
