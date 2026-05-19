"""
behavior.launch.py — Launch file for OMNI behavior_node.

Usage:
    ros2 launch behavior_node behavior.launch.py

Optional overrides:
    ros2 launch behavior_node behavior.launch.py gemini_model:=gemini-2.5-flash
    ros2 launch behavior_node behavior.launch.py mic_device_index:=1
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    pkg_share = get_package_share_directory('behavior_node')
    default_config = os.path.join(pkg_share, 'config', 'omni_config.yaml')

    # ── Launch arguments ───────────────────────────────────────────────────────
    # Each argument maps directly to a ROS2 parameter on the node.
    # Declare them here so they can be overridden from the command line.

    gemini_model_arg = DeclareLaunchArgument(
        'gemini_model',
        default_value='models/gemini-2.5-flash-native-audio-latest',
        description='Gemini model name',
    )
    gemini_voice_arg = DeclareLaunchArgument(
        'gemini_voice',
        default_value='Algieba',
        description='Gemini Live voice name',
    )
    config_file_arg = DeclareLaunchArgument(
        'config_file_path',
        default_value=default_config,
        description='Path to omni_config.yaml',
    )
    wake_word_model_arg = DeclareLaunchArgument(
        'wake_word_model',
        default_value='hey_mycroft',
        description='openwakeword model name (without .onnx extension)',
    )
    wake_word_threshold_arg = DeclareLaunchArgument(
        'wake_word_threshold',
        default_value='0.5',
        description='Wake word score threshold (0.0–1.0)',
    )
    conversation_timeout_arg = DeclareLaunchArgument(
        'conversation_timeout',
        default_value='30.0',
        description='Seconds of silence before closing the Gemini stream',
    )
    mic_device_arg = DeclareLaunchArgument(
        'mic_device_index',
        default_value='0',
        description='sounddevice index for USB microphone',
    )
    speaker_device_arg = DeclareLaunchArgument(
        'speaker_device_index',
        default_value='0',
        description='sounddevice index for USB speaker',
    )
    tcp_mic_port_arg = DeclareLaunchArgument(
        'tcp_mic_port',
        default_value='0',
        description='TCP port to accept stereo PCM from Pi Zero (0 = use local mic)',
    )

    # ── Node ──────────────────────────────────────────────────────────────────
    behavior_node = Node(
        package='behavior_node',
        executable='behavior_node',
        name='behavior_node',
        output='screen',
        parameters=[{
            'gemini_model':          LaunchConfiguration('gemini_model'),
            'gemini_voice':          LaunchConfiguration('gemini_voice'),
            'config_file_path':      LaunchConfiguration('config_file_path'),
            'wake_word_model':       LaunchConfiguration('wake_word_model'),
            'wake_word_threshold':   LaunchConfiguration('wake_word_threshold'),
            'conversation_timeout':  LaunchConfiguration('conversation_timeout'),
            'mic_device_index':      LaunchConfiguration('mic_device_index'),
            'speaker_device_index':  LaunchConfiguration('speaker_device_index'),
            'tcp_mic_port':          LaunchConfiguration('tcp_mic_port'),
        }],
        # GEMINI_API_KEY must be set in the environment before launching.
        # The node logs a clear error if it is missing.
        emulate_tty=True,   # preserves colour in ros2 launch output
    )

    return LaunchDescription([
        gemini_model_arg,
        gemini_voice_arg,
        config_file_arg,
        wake_word_model_arg,
        wake_word_threshold_arg,
        conversation_timeout_arg,
        mic_device_arg,
        speaker_device_arg,
        tcp_mic_port_arg,
        behavior_node,
    ])
