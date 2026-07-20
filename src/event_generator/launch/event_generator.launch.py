"""Launch the event_generator node.

Needs world_state running — it consumes /omni/world_state, nothing else:

    ros2 launch world_state world_state.launch.py
    ros2 launch event_generator event_generator.launch.py

To capture a replay fixture during a live session (see the package CLAUDE.md):

    ros2 launch event_generator event_generator.launch.py \
        record_path:=/tmp/world_state_live.jsonl
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    absence_grace = LaunchConfiguration("absence_grace")
    unknown_min_snapshots = LaunchConfiguration("unknown_min_snapshots")
    record_path = LaunchConfiguration("record_path")

    return LaunchDescription([
        DeclareLaunchArgument(
            "absence_grace",
            default_value="90.0",
            description=(
                "Seconds of sustained absence before person_left fires. Must stay "
                "well above the longest normal face dropout — world state is "
                "face-anchored, so turning away is not leaving."
            ),
        ),
        DeclareLaunchArgument(
            "unknown_min_snapshots",
            default_value="3",
            description=(
                "Snapshots a stable unknown_N must survive before it is announced"
            ),
        ),
        DeclareLaunchArgument(
            "record_path",
            default_value="",
            description="If set, append every inbound snapshot to this JSONL file",
        ),
        Node(
            package="event_generator",
            executable="event_generator_node",
            name="event_generator_node",
            output="screen",
            parameters=[{
                "absence_grace": absence_grace,
                "unknown_min_snapshots": unknown_min_snapshots,
                "record_path": record_path,
            }],
        ),
    ])
