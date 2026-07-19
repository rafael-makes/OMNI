"""OMNI dock AprilTag detector — SHARED-camera variant (Jetson-side).

Same job as dock_apriltag.launch.py: rear cam -> apriltag_ros -> /detections,
which the Pi's baro_floor_resolver reads at boot (tag id 0 = main floor,
id 1 = basement). The difference is who holds the camera.

  legacy: v4l2_camera opens /dev/video* directly  -> rear-view scene description
          cannot run at the same time (device is -EBUSY for the second opener)

  this:   rear_camera owns the device and publishes  -> docking AND
          "what's behind you?" both work concurrently

Use this one. dock_apriltag.launch.py is kept as an untouched fallback in case
this path ever misbehaves — but the two are mutually exclusive, and so is
running the legacy file alongside head_detector.launch.py's rear_vision:=true.

WHY THIS IS A SAFE SWAP
  Floor selection needs only tag IDs, so dock_apriltag.yaml sets
  pose_estimation_method: "" — apriltag_ros never reads camera_info's K and no
  calibration is involved. The old pipeline fed apriltag UNRECTIFIED frames on
  image_rect already, so nothing is lost by publishing plain bgr8 here.

  Format note: the legacy file used YUYV because v4l2_camera 0.7.1's cv_bridge
  path cannot decode this camera's MJPG to rgb8. rear_camera does its own
  jpegdec + bgr8 conversion and never touches cv_bridge, so MJPG is fine here —
  and MJPG is what keeps the shared USB 2.0 bus comfortable (~6 Mbps instead of
  the ~147 Mbps raw YUYV would cost at this size).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

_REAR_CAM = ('/dev/v4l/by-id/'
             'usb-GENERAL_2K_HD_Camera-video-index0')


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory('omni_jetson_bringup'),
                       'config', 'dock_apriltag.yaml')
    device = LaunchConfiguration('device')

    return LaunchDescription([
        DeclareLaunchArgument(
            'device', default_value=_REAR_CAM,
            description='V4L2 by-id path of the rear docking cam'),
        DeclareLaunchArgument(
            'decode_fps', default_value='5.0',
            description='Decode rate. Tag IDs at docking speed need very little; '
                        'the wire rate is fixed at 30 regardless (this camera '
                        'negotiates no other MJPG rate).'),
        # Owns the device; feeds BOTH apriltag (raw) and frame_server (clean JPEG).
        # If head_detector.launch.py is already running with rear_vision:=true,
        # do not launch this — that node already holds the camera. Launch this
        # file INSTEAD, with rear_vision:=false over there.
        Node(
            package='rear_camera',
            executable='rear_camera_node',
            name='rear_camera',
            output='screen',
            parameters=[{
                'device': device,
                'capture_width': 640,
                'capture_height': 480,
                'capture_fps': 30,
                'decode_fps': LaunchConfiguration('decode_fps'),
                'publish_clean_image': True,
                'clean_image_fps': 2.0,
                'publish_raw_image': True,      # the point of this launch file
                'raw_image_fps': 5.0,
                'frame_id': 'dock_cam',
            }],
        ),
        Node(
            package='apriltag_ros',
            executable='apriltag_node',
            name='dock_apriltag',
            output='screen',
            parameters=[cfg],
            remappings=[
                ('image_rect', '/camera/rear/image_raw'),
                ('camera_info', '/camera/rear/camera_info'),
                ('detections', '/detections'),
            ],
        ),
    ])
