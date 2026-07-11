"""OMNI dock AprilTag detector (Jetson-side).

Rear 2K USB docking cam -> v4l2_camera -> apriltag_ros -> /detections
(apriltag_msgs/AprilTagDetectionArray). The Pi's baro_floor_resolver reads this
at boot: tag id 0 = main floor, id 1 = basement.

Camera notes (learned the hard way):
  - v4l2_camera 0.7.1 cannot decode this cam's MJPG to rgb8 (cv_bridge throws).
    Use YUYV. YUYV maxes at 1280x720; 800x600 @ ~25 Hz is plenty for tag ids and
    lighter on the USB bus.
  - Opened by stable /dev/v4l/by-id path so it never swaps nodes with the head cam.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

# Rear 2K docking cam (see: ls /dev/v4l/by-id/)
_DOCK_CAM = '/dev/v4l/by-id/usb-GENERAL_2K_HD_Camera-video-index0'


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory('omni_jetson_bringup'),
                       'config', 'dock_apriltag.yaml')
    usb_device = LaunchConfiguration('usb_device')

    return LaunchDescription([
        DeclareLaunchArgument(
            'usb_device', default_value=_DOCK_CAM,
            description='V4L2 by-id path of the rear docking cam'),
        Node(
            package='v4l2_camera',
            executable='v4l2_camera_node',
            name='v4l2_camera',
            namespace='dock_cam',
            output='screen',
            parameters=[{
                'video_device': usb_device,
                'pixel_format': 'YUYV',
                'image_size': [800, 600],
                'camera_frame_id': 'dock_cam',
            }],
        ),
        Node(
            package='apriltag_ros',
            executable='apriltag_node',
            name='dock_apriltag',
            output='screen',
            parameters=[cfg],
            remappings=[
                ('image_rect', '/dock_cam/image_raw'),
                ('camera_info', '/dock_cam/camera_info'),
                ('detections', '/detections'),
            ],
        ),
    ])
