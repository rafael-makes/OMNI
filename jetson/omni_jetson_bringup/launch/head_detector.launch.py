"""OMNI head camera — USB webcam person detector (YOLO26n / TensorRT on the Jetson).

Publishes /camera/detections + /camera/status for the Pi's head_tracking_node.
The head cam is the HHWei USB webcam, opened by its stable /dev/v4l/by-id path so it
never swaps device nodes with the rear (2K) docking cam.

USB gotcha baked in here: capture_fps=30 (the node default of 60 is CSI-only; USB
webcams max out at 30). Captures 848x480 (native MJPG mode, ~29fps), publishes in
1280x720 space so head_tracking's pixel->angle mapping is unchanged.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_HEAD_CAM = ('/dev/v4l/by-id/'
             'usb-HHWei_Technology_Co.__Ltd._USB_Camera_HHW001-video-index0')


def generate_launch_description():
    usb_device = LaunchConfiguration('usb_device')

    return LaunchDescription([
        DeclareLaunchArgument(
            'usb_device', default_value=_HEAD_CAM,
            description='V4L2 by-id path of the head USB camera'),
        Node(
            package='head_detector',
            executable='head_detector_node',
            name='head_detector_node',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'source': 'usb',
                'usb_device': usb_device,
                'capture_width': 848,
                'capture_height': 480,
                'capture_fps': 30,      # USB webcams cap at 30 (node default 60 is CSI-only)
            }],
        ),
    ])
