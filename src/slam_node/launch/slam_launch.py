"""
OMNI slam_node launch file.

Starts:
  1. static_transform_publisher — base_link → lidar_link  (matches lidar_node frame)
  2. slam_toolbox async_slam_toolbox_node as a LifecycleNode
     Automatically triggers: configure → activate so it starts mapping immediately.

Physical offsets (measure on hardware and update if needed):
  LiDAR sits on top of the carbon-fibre mast through the head.
  z=0.825 m measured — floor to LiDAR scan plane.
  X/Y offsets are 0 — LiDAR is centred on the robot's vertical axis.
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


def generate_launch_description():

    pkg_dir     = get_package_share_directory('slam_node')
    params_file = os.path.join(pkg_dir, 'config', 'mapper_params_online_async.yaml')

    autostart            = LaunchConfiguration('autostart', default='true')
    use_lifecycle_manager = LaunchConfiguration('use_lifecycle_manager', default='false')
    use_sim_time         = LaunchConfiguration('use_sim_time', default='false')
    slam_params          = LaunchConfiguration('slam_params_file', default=params_file)

    # ── LiDAR static TF ──────────────────────────────────────────────────────
    # LD19 is on top of the mast, centred over the robot.
    # frame_id=base_link  child_frame_id=lidar_link  (matches lidar_node default)
    # x=0  y=0  z=0.825m  roll=0  pitch=0  yaw=0
    # Measured: lidar scan plane is 825mm above the floor.
    lidar_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_lidar',
        arguments=['0', '0', '0.825', '0', '0', '0',
                   'base_link', 'lidar_link'],
        output='screen',
    )

    # ── slam_toolbox async (LifecycleNode) ────────────────────────────────────
    slam_node = LifecycleNode(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        namespace='',
        parameters=[
            slam_params,
            {
                'use_lifecycle_manager': use_lifecycle_manager,
                'use_sim_time': use_sim_time,
            },
        ],
    )

    # ── Lifecycle: configure at t=3s, activate at t=8s ───────────────────────
    # slam_toolbox is a lifecycle node. Using two independent TimerActions is the
    # most reliable approach on the Pi 5 — OnStateTransition fires prematurely and
    # sends ACTIVATE before configure completes, leaving the node unconfigured.
    #
    # The node prints "Node using stack size 40000000" at ~2s after process start.
    # t=3s gives a comfortable margin; configure typically takes ~0.5s; t=8s for
    # activate gives 5s headroom which covers Ceres initialisation.
    configure_event = TimerAction(
        period=3.0,
        actions=[
            LogInfo(msg='[slam_node] sending configure...'),
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
            LogInfo(msg='[slam_node] sending activate...'),
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
        DeclareLaunchArgument(
            'autostart', default_value='true',
            description='Auto configure+activate slam_toolbox on launch'),
        DeclareLaunchArgument(
            'use_lifecycle_manager', default_value='false',
            description='Use external nav2 lifecycle manager instead'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='Use simulation clock'),
        DeclareLaunchArgument(
            'slam_params_file', default_value=params_file,
            description='Path to slam_toolbox params YAML'),
        LogInfo(msg='OMNI slam_node starting — map→odom→base_link, LiDAR z=0.38m'),
        lidar_tf,
        slam_node,
        configure_event,
        activate_event,
    ])
