"""
OMNI localization launch — use when the home map already exists.

Loads the saved pose graph and localises within the existing map.
Does not modify the map.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent,
                             LogInfo, TimerAction)
from launch.conditions import IfCondition
from launch.events import matches_action
from launch.substitutions import (AndSubstitution, LaunchConfiguration,
                                  NotSubstitution)
from launch_ros.actions import LifecycleNode, Node
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition


DEFAULT_MAP = '/home/pi/omni_ws/maps/omni_home_map'


def generate_launch_description():

    pkg_dir     = get_package_share_directory('slam_node')
    params_file = os.path.join(pkg_dir, 'config', 'mapper_params_localization.yaml')

    autostart            = LaunchConfiguration('autostart', default='true')
    use_lifecycle_manager = LaunchConfiguration('use_lifecycle_manager', default='false')
    use_sim_time         = LaunchConfiguration('use_sim_time', default='false')
    map_file             = LaunchConfiguration('map_file', default=DEFAULT_MAP)

    lidar_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_lidar',
        arguments=['0', '0', '0.825', '0', '0', '0',
                   'base_link', 'lidar_link'],
        output='screen',
    )

    # ToF sensor static transforms (base_link → tof_* frames)
    # Robot base 400×400mm. Sensor height 65mm. x=forward, y=left.
    # Quaternions: forward=(0,0,0,1), left=(0,0,0.7071,0.7071),
    #              right=(0,0,-0.7071,0.7071), backward=(0,0,1,0)
    tof_tfs = [
        Node(package='tf2_ros', executable='static_transform_publisher',
             name='base_to_tof_left',
             arguments=['0.065', '0.200', '0.065',
                        '0', '0', '0.7071', '0.7071',
                        'base_link', 'tof_left']),
        Node(package='tf2_ros', executable='static_transform_publisher',
             name='base_to_tof_front_left',
             arguments=['0.200', '0.135', '0.065',
                        '0', '0', '0', '1',
                        'base_link', 'tof_front_left']),
        Node(package='tf2_ros', executable='static_transform_publisher',
             name='base_to_tof_front_right',
             arguments=['0.200', '-0.135', '0.065',
                        '0', '0', '0', '1',
                        'base_link', 'tof_front_right']),
        Node(package='tf2_ros', executable='static_transform_publisher',
             name='base_to_tof_right',
             arguments=['0.065', '-0.200', '0.065',
                        '0', '0', '-0.7071', '0.7071',
                        'base_link', 'tof_right']),
        Node(package='tf2_ros', executable='static_transform_publisher',
             name='base_to_tof_left_rear',
             arguments=['-0.200', '0.135', '0.065',
                        '0', '0', '1', '0',
                        'base_link', 'tof_left_rear']),
        Node(package='tf2_ros', executable='static_transform_publisher',
             name='base_to_tof_right_rear',
             arguments=['-0.200', '-0.135', '0.065',
                        '0', '0', '1', '0',
                        'base_link', 'tof_right_rear']),
    ]

    slam_node = LifecycleNode(
        package='slam_toolbox',
        executable='localization_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        namespace='',
        parameters=[
            params_file,
            {
                'use_lifecycle_manager': use_lifecycle_manager,
                'use_sim_time': use_sim_time,
                'map_file_name': map_file,
                'mode': 'localization',
            },
        ],
    )

    configure_event = TimerAction(
        period=3.0,
        actions=[
            LogInfo(msg='[slam_node] localization: sending configure...'),
            EmitEvent(
                event=ChangeState(
                    lifecycle_node_matcher=matches_action(slam_node),
                    transition_id=Transition.TRANSITION_CONFIGURE,
                ),
                condition=IfCondition(
                    AndSubstitution(autostart, NotSubstitution(use_lifecycle_manager))
                ),
            ),
        ],
    )

    activate_event = TimerAction(
        period=8.0,
        actions=[
            LogInfo(msg='[slam_node] localization: sending activate...'),
            EmitEvent(
                event=ChangeState(
                    lifecycle_node_matcher=matches_action(slam_node),
                    transition_id=Transition.TRANSITION_ACTIVATE,
                ),
                condition=IfCondition(
                    AndSubstitution(autostart, NotSubstitution(use_lifecycle_manager))
                ),
            ),
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument('autostart', default_value='true',
                              description='Auto configure+activate on launch'),
        DeclareLaunchArgument('use_lifecycle_manager', default_value='false'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('map_file', default_value=DEFAULT_MAP,
                              description='Path prefix of the saved pose graph'),
        LogInfo(msg=['OMNI localization starting — loading map: ', map_file]),
        lidar_tf,
        *tof_tfs,
        slam_node,
        configure_event,
        activate_event,
    ])
