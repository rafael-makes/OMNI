"""Launch the event_generator node.

Needs world_state running — it consumes /omni/world_state, nothing else:

    ros2 launch world_state world_state.launch.py
    ros2 launch event_generator event_generator.launch.py

To capture a replay fixture during a live session (see the package CLAUDE.md):

    ros2 launch event_generator event_generator.launch.py \
        record_path:=/tmp/world_state_live.jsonl

Dwell events (Session 9) are OFF until dwell_zones names at least one zone, and
that zone must also exist in omni_zones/config/zones.yaml or world_state will
never label anybody with it:

    ros2 launch event_generator event_generator.launch.py dwell_zones:=workbench

Testing the check-in without waiting half an hour — 60s dwell, re-firing every
2 minutes:

    ros2 launch event_generator event_generator.launch.py \
        dwell_zones:=workbench dwell_threshold:=60.0 dwell_refire_interval:=120.0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    absence_grace = LaunchConfiguration("absence_grace")
    unknown_min_snapshots = LaunchConfiguration("unknown_min_snapshots")
    record_path = LaunchConfiguration("record_path")
    dwell_threshold = LaunchConfiguration("dwell_threshold")
    dwell_refire_interval = LaunchConfiguration("dwell_refire_interval")
    dwell_zones = LaunchConfiguration("dwell_zones")

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
        DeclareLaunchArgument(
            "dwell_threshold",
            default_value="1800.0",
            description=(
                "Seconds in one zone before the first person_dwelling. This is a "
                "floor, not the check-in policy — behavior_node's "
                "check_in_min_dwell makes the actual decision."
            ),
        ),
        DeclareLaunchArgument(
            "dwell_refire_interval",
            default_value="1800.0",
            description=(
                "Seconds of continued dwell between re-firings, so the policy "
                "layer gets later chances after declining an earlier one"
            ),
        ),
        DeclareLaunchArgument(
            "dwell_zones",
            default_value="",
            description=(
                "COMMA-SEPARATED zones where a dwell is worth reporting, e.g. "
                "workbench,computer. EMPTY MEANS DWELL IS OFF — nothing fires "
                "until this is set, and the zone must also exist in "
                "omni_zones/config/zones.yaml or world_state never labels "
                "anyone with it. (A string, not a list: an empty list default "
                "cannot be typed in rclpy — see node.py.)"
            ),
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
                "dwell_threshold": dwell_threshold,
                "dwell_refire_interval": dwell_refire_interval,
                "dwell_zones": dwell_zones,
            }],
        ),
    ])
