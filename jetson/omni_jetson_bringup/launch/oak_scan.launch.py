"""OAK-D Lite depth -> LaserScan for OMNI's Nav2 obstacle layer.

Runs the depthai driver (depth-only, tuned for USB2 throughput via
config/oak_lite_depth.yaml) + depthimage_to_laserscan, publishing /oak/scan —
the topic the Pi's nav2_params.yaml oak_depth observation_source expects.

The depthai driver also publishes base_link -> oak TF (see the camera.i_tf_*
params in oak_lite_depth.yaml), so /oak/scan (frame oak_rgb_camera_optical_frame)
is reachable from base_link on the Pi's TF tree.
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    bringup_share = get_package_share_directory('omni_jetson_bringup')
    oak_cfg = os.path.join(bringup_share, 'config', 'oak_lite_depth.yaml')

    depthai_launch = os.path.join(
        get_package_share_directory('depthai_ros_driver'),
        'launch', 'camera.launch.py')

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(depthai_launch),
            launch_arguments={'params_file': oak_cfg}.items(),
        ),
        Node(
            package='depthimage_to_laserscan',
            executable='depthimage_to_laserscan_node',
            name='oak_depth_to_scan',
            remappings=[
                ('depth', '/oak/stereo/image_raw'),
                ('depth_camera_info', '/oak/stereo/camera_info'),
                ('scan', '/oak/scan'),
            ],
            parameters=[{
                'scan_time': 0.033,
                'range_min': 0.3,
                'range_max': 8.0,
                # one row band around image center -> a single planar scan line
                'scan_height': 10,
                'output_frame': 'oak_rgb_camera_optical_frame',
            }],
        ),
    ])
