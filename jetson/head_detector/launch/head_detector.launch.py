"""Launch the Jetson head detector + the base_link -> camera_link static TF.

CAMERA: OMNI now uses a USB webcam (the IMX219 CSI camera was removed), so `source`
defaults to 'usb'. capture_fps defaults to 30 because both attached USB cams top out
at 30fps MJPG @1280x720 — the node's own default of 60 fails to negotiate and the
capture loop just spins on "frame read failed".

Face recognition (publish_identity) is ON here: it feeds /camera/identity, which the
Pi's behavior_node uses to key OMNI's per-person memory. It is fail-safe — if the
SFace model or gallery is missing, identity is disabled and person/face detection
continue unaffected. Pass publish_identity:=false to skip it entirely.

NOTE: the static-transform xyz/rpy below are PLACEHOLDERS for the head-mounted
camera position — Rafael to measure on hardware and update. head_tracking_node
does not consume this TF, but it keeps the frame defined for other consumers.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    source = LaunchConfiguration('source')
    capture_fps = LaunchConfiguration('capture_fps')
    publish_identity = LaunchConfiguration('publish_identity')
    return LaunchDescription([
        DeclareLaunchArgument('source', default_value='usb',
                              description="'usb' for a UVC webcam, 'csi' for nvarguscamerasrc, "
                                          "or an image path for offline test"),
        DeclareLaunchArgument('capture_fps', default_value='30',
                              description='Capture framerate. Keep at 30 for the USB cams — '
                                          'they do not offer 60fps and the pipeline fails to open.'),
        DeclareLaunchArgument('publish_identity', default_value='true',
                              description='Publish /camera/identity (SFace face recognition) '
                                          'for per-person memory. false disables recognition.'),
        Node(
            package='head_detector',
            executable='head_detector_node',
            name='head_detector_node',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'source': source,
                'capture_fps': capture_fps,
                'publish_identity': publish_identity,
            }],
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
