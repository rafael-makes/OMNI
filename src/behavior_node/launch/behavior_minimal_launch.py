"""
behavior_minimal_launch.py — Single-node launch for behavior_node.

Starts only behavior_node with every parameter exposed as a launch argument.
No other nodes, no includes. Use this for isolated testing or bring-up.

Usage:
    ros2 launch behavior_node behavior_minimal_launch.py

Override examples:
    ros2 launch behavior_node behavior_minimal_launch.py gemini_model:=models/gemini-2.5-pro
    ros2 launch behavior_node behavior_minimal_launch.py mic_device_index:=1 tcp_mic_port:=9000
    ros2 launch behavior_node behavior_minimal_launch.py wake_word_model:=hey_mycroft wake_word_threshold:=0.7

GEMINI_API_KEY must be exported in the environment before launching:
    export GEMINI_API_KEY=your_key_here
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

    args = [
        DeclareLaunchArgument(
            'gemini_model',
            # Tool-capable Live model required: native-audio models reject `tools`.
            default_value='models/gemini-3.1-flash-live-preview',
            description='Gemini model name passed to the Live API',
        ),
        DeclareLaunchArgument(
            'gemini_voice',
            default_value='Algieba',
            description='Gemini Live voice name (e.g. Algieba, Charon, Fenrir)',
        ),
        DeclareLaunchArgument(
            'config_file_path',
            default_value=default_config,
            description='Absolute path to omni_config.yaml (system prompt + locations)',
        ),
        DeclareLaunchArgument(
            'wake_word_model',
            default_value='hey_mycroft',
            description='openwakeword model name, without .onnx extension',
        ),
        DeclareLaunchArgument(
            'wake_word_threshold',
            default_value='0.5',
            description='Wake word confidence threshold (0.0–1.0); lower = more sensitive',
        ),
        DeclareLaunchArgument(
            'wake_word_startup_suppress',
            default_value='1.5',
            description='Seconds to suppress wake word scoring after detector restart (drains speaker bleed)',
        ),
        DeclareLaunchArgument(
            'conversation_timeout',
            default_value='30.0',
            description='Seconds of Gemini silence before closing stream and returning to IDLE',
        ),
        DeclareLaunchArgument(
            'idle_return_timeout',
            default_value='30.0',
            description='Seconds before an unacknowledged LISTENING state auto-returns to IDLE',
        ),
        DeclareLaunchArgument(
            'mic_device_index',
            default_value='0',
            description='sounddevice input device index for the local USB microphone',
        ),
        DeclareLaunchArgument(
            'speaker_device_index',
            default_value='0',
            description='sounddevice output device index for the USB speaker',
        ),
        DeclareLaunchArgument(
            'tcp_mic_port',
            default_value='0',
            description='TCP port to receive stereo PCM from Pi Zero mic (0 = use local mic)',
        ),
    ]

    node = Node(
        package='behavior_node',
        executable='behavior_node',
        name='behavior_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'gemini_model':         LaunchConfiguration('gemini_model'),
            'gemini_voice':         LaunchConfiguration('gemini_voice'),
            'config_file_path':     LaunchConfiguration('config_file_path'),
            'wake_word_model':      LaunchConfiguration('wake_word_model'),
            'wake_word_threshold':        LaunchConfiguration('wake_word_threshold'),
            'wake_word_startup_suppress': LaunchConfiguration('wake_word_startup_suppress'),
            'conversation_timeout': LaunchConfiguration('conversation_timeout'),
            'idle_return_timeout':  LaunchConfiguration('idle_return_timeout'),
            'mic_device_index':     LaunchConfiguration('mic_device_index'),
            'speaker_device_index': LaunchConfiguration('speaker_device_index'),
            'tcp_mic_port':         LaunchConfiguration('tcp_mic_port'),
        }],
    )

    return LaunchDescription(args + [node])
