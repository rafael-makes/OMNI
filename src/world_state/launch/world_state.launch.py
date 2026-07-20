"""Launch the world_state node on its own.

Add the rear camera once the Orin publishes rear-tagged detections:

    ros2 launch world_state world_state.launch.py \
        identity_sources:="['/camera/identities=head','/rear_camera/identities=rear']"
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    identity_sources = LaunchConfiguration("identity_sources")
    detection_sources = LaunchConfiguration("detection_sources")
    visibility_timeout = LaunchConfiguration("visibility_timeout")

    return LaunchDescription([
        DeclareLaunchArgument(
            "identity_sources",
            default_value="['/camera/identities=head']",
            description="Face-recognition topics as 'topic=camera' pairs",
        ),
        DeclareLaunchArgument(
            "detection_sources",
            default_value="[]",
            description=(
                "YOLO Detection2DArray topics as 'topic=camera' pairs. Empty by "
                "default — body boxes double-count people who are also being "
                "tracked by face. Enable only for body-without-face presence."
            ),
        ),
        DeclareLaunchArgument(
            "visibility_timeout",
            default_value="3.0",
            description="Seconds without a detection before a person is 'away'",
        ),
        Node(
            package="world_state",
            executable="world_state_node",
            name="world_state_node",
            output="screen",
            parameters=[{
                "identity_sources": identity_sources,
                "detection_sources": detection_sources,
                "visibility_timeout": visibility_timeout,
            }],
        ),
    ])
