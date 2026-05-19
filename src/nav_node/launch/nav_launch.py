"""
nav_launch.py — Nav2 navigation stack for OMNI robot

Brings up the full Nav2 stack configured for OMNI's hardware.
Run AFTER slam_node is already running (it provides /map and the
map→odom→base_link TF tree that Nav2 depends on).

CRITICAL: controller_server is remapped so Nav2 publishes velocity
commands to /cmd_vel_raw instead of /cmd_vel. This means all motion
passes through safety_node before reaching the motors:

  Nav2 → /cmd_vel_raw → safety_node → /cmd_vel → motor_control_node

Usage:
  ros2 launch nav_node nav_launch.py
  ros2 launch nav_node nav_launch.py params_file:=/path/to/custom_params.yaml

Nodes started by this launch file:
  /bt_navigator         — behavior tree executor, accepts goal poses
  /planner_server       — global path planner (NavFn A*)
  /controller_server    — local path follower (DWB), publishes /cmd_vel_raw
  /behavior_server      — recovery behaviors (spin, backup, wait)
  /lifecycle_manager_navigation — manages all of the above
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    # ── Arguments ──────────────────────────────────────────────────────────────

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('nav_node'), 'config', 'nav2_params.yaml'
        ]),
        description='Full path to the Nav2 parameters YAML file'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='False',
        description='Use simulation clock (False for real robot)'
    )

    params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')

    # ── Nav2 nodes ─────────────────────────────────────────────────────────────
    #
    # All nodes load their parameters from the single nav2_params.yaml file.
    # The YAML is keyed by node name, so each node picks up its own section.
    #
    # map_server is NOT started here — slam_node's slam_toolbox publishes
    # /map live. If you ever want to navigate with a pre-saved map (no SLAM),
    # you will need a separate localization launch (map_server + amcl).

    bt_navigator = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        # bt_navigator orchestrates planner/controller — it does not publish
        # cmd_vel directly, so no remapping needed here.
    )

    planner_server = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        # planner_server computes global paths — publishes /plan (Path),
        # not velocity, so no cmd_vel remapping needed.
    )

    controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        # *** CRITICAL REMAPPING ***
        # controller_server follows the global path and publishes velocity
        # commands. By default it publishes to /cmd_vel. We remap it to
        # /cmd_vel_raw so safety_node can intercept it before it reaches
        # motor_control_node. Without this, Nav2 would bypass safety_node.
        remappings=[('/cmd_vel', '/cmd_vel_raw')],
    )

    behavior_server = Node(
        package='nav2_behaviors',
        executable='behavior_server',
        name='behavior_server',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        # behavior_server runs recovery behaviors (spin, backup, wait).
        # Backup moves the robot in reverse — it also publishes velocity.
        # Remap its cmd_vel output too so backup goes through safety_node.
        remappings=[('/cmd_vel', '/cmd_vel_raw')],
    )

    smoother_server = Node(
        package='nav2_smoother',
        executable='smoother_server',
        name='smoother_server',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        # smoother_server refines paths — publishes smoothed Path, not vel.
    )

    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            # List of Nav2 nodes to manage. The lifecycle_manager starts them,
            # transitions them through configure→activate, and monitors bonds.
            # Order matters: dependencies must come before dependents.
            'node_names': [
                'planner_server',
                'smoother_server',
                'controller_server',
                'behavior_server',
                'bt_navigator',
            ],
            # autostart: True means lifecycle_manager immediately transitions
            # all nodes to active. Set to False if you want manual control.
            'autostart': True,
            # bond_timeout: seconds before lifecycle_manager declares a node
            # dead if it stops heartbeating. 4.0s is the Nav2 default.
            'bond_timeout': 4.0,
        }],
    )

    return LaunchDescription([
        params_file_arg,
        use_sim_time_arg,
        bt_navigator,
        planner_server,
        controller_server,
        behavior_server,
        smoother_server,
        lifecycle_manager,
    ])
