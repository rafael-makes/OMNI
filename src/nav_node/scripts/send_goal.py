#!/usr/bin/env python3
"""
send_goal.py — Send a navigation goal to OMNI's Nav2 stack

Usage:
    python3 send_goal.py <x> <y> [yaw_degrees]

Arguments:
    x             Goal X position in meters (map frame)
    y             Goal Y position in meters (map frame)
    yaw_degrees   Goal heading in degrees (optional — omit for free heading)

Examples:
    python3 send_goal.py 2.0 0.0         # 2m forward, any final heading
    python3 send_goal.py 1.0 0.5 90      # 1m forward, 0.5m left, face +Y
    python3 send_goal.py -0.5 0.0 180    # 0.5m backward, face -X

The goal is sent to Nav2's NavigateToPose action server.
Nav2 will plan a path and drive OMNI to the goal pose.
The script prints live feedback (distance remaining, estimated time).

Prerequisites:
    - slam_node must be running (provides /map and TF)
    - nav_node must be running (Nav2 stack active)
    - source /home/pi/omni_ws/install/setup.bash

This script is standalone — behavior_node will eventually call Nav2's
action server directly instead of using this script.
"""

import sys
import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from builtin_interfaces.msg import Time


def euler_to_quaternion(yaw_rad: float):
    """Convert yaw angle (radians) to quaternion (x, y, z, w)."""
    # For a robot moving in 2D, only yaw matters.
    # Roll = 0, pitch = 0, yaw = angle in XY plane.
    half_yaw = yaw_rad / 2.0
    return (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))


class GoalSender(Node):
    def __init__(self):
        super().__init__('goal_sender')
        self._action_client = ActionClient(
            self,
            NavigateToPose,
            'navigate_to_pose'
        )

    def send_goal(self, x: float, y: float, yaw_deg: float | None = None):
        yaw_rad = math.radians(yaw_deg) if yaw_deg is not None else 0.0
        qx, qy, qz, qw = euler_to_quaternion(yaw_rad)

        # Build the goal pose in the map frame
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y
        goal_pose.pose.position.z = 0.0
        goal_pose.pose.orientation.x = qx
        goal_pose.pose.orientation.y = qy
        goal_pose.pose.orientation.z = qz
        goal_pose.pose.orientation.w = qw

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose

        self.get_logger().info(
            f'Waiting for NavigateToPose action server...'
        )
        if not self._action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(
                'NavigateToPose action server not available. '
                'Is nav_node running?'
            )
            return False

        yaw_str = f'{yaw_deg:.1f}°' if yaw_deg is not None else 'free'
        self.get_logger().info(
            f'Sending goal: x={x:.2f}m  y={y:.2f}m  yaw={yaw_str}'
        )

        send_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self._feedback_callback
        )
        rclpy.spin_until_future_complete(self, send_future)

        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal was REJECTED by Nav2')
            return False

        self.get_logger().info('Goal accepted — navigating...')

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result()
        status = result.status

        # ROS2 action status codes:
        # 4 = SUCCEEDED, 5 = CANCELED, 6 = ABORTED
        if status == 4:
            self.get_logger().info('Goal SUCCEEDED — OMNI reached the target.')
            return True
        elif status == 5:
            self.get_logger().warn('Goal was CANCELED.')
            return False
        else:
            self.get_logger().error(
                f'Goal FAILED (status={status}). '
                'Check if path was blocked or robot got stuck.'
            )
            return False

    def _feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        dist = feedback.distance_remaining
        # estimated_time_remaining is a Duration — convert to seconds
        try:
            eta_sec = Duration.from_msg(
                feedback.estimated_time_remaining
            ).nanoseconds / 1e9
            self.get_logger().info(
                f'  → {dist:.2f}m remaining  (ETA: {eta_sec:.1f}s)',
                throttle_duration_sec=1.0
            )
        except Exception:
            self.get_logger().info(
                f'  → {dist:.2f}m remaining',
                throttle_duration_sec=1.0
            )


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print('ERROR: missing arguments')
        print(f'Usage: python3 {sys.argv[0]} <x> <y> [yaw_degrees]')
        sys.exit(1)

    try:
        x = float(sys.argv[1])
        y = float(sys.argv[2])
        yaw_deg = float(sys.argv[3]) if len(sys.argv) > 3 else None
    except ValueError:
        print('ERROR: x, y, and yaw must be numbers')
        sys.exit(1)

    rclpy.init()
    node = GoalSender()

    try:
        success = node.send_goal(x, y, yaw_deg)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print('\nGoal canceled by user (Ctrl+C)')
        sys.exit(1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
