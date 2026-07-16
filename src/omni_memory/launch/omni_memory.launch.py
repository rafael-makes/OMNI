"""Launch the omni_memory node with configurable parameters."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    env_file = LaunchConfiguration("env_file")
    default_k = LaunchConfiguration("default_k")

    return LaunchDescription([
        DeclareLaunchArgument(
            "env_file",
            default_value="/home/pi/omni_ws/src/omni_memory/.env",
            description="Path to the .env with SUPABASE_URL/SUPABASE_SERVICE_KEY (+ GEMINI_API_KEY).",
        ),
        DeclareLaunchArgument(
            "default_k", default_value="5",
            description="Default number of memories returned by retrieve_memories.",
        ),
        Node(
            package="omni_memory",
            executable="omni_memory_node",
            name="omni_memory",
            output="screen",
            parameters=[{"env_file": env_file, "default_k": default_k}],
        ),
    ])
