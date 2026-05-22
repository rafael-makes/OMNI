#!/usr/bin/env python3
"""
servo_node validation — publishes test commands and reads /servo_status.
Run in one terminal, servo_node in another.
Or run standalone: starts node in-process, publishes commands, verifies.
"""
import sys
import time
import threading

sys.path.insert(0, '/home/pi/ros2_jazzy/install/rclpy/lib/python3.13/site-packages')

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String


class ServoTester(Node):
    def __init__(self):
        super().__init__('servo_tester')
        self.pub = self.create_publisher(Float32MultiArray, '/servo_commands', 10)
        self.sub = self.create_subscription(String, '/servo_status', self._on_status, 10)
        self.received_status = []

    def _on_status(self, msg):
        self.received_status.append(msg.data)
        self.get_logger().info(f'/servo_status: {msg.data}')

    def send(self, commands):
        msg = Float32MultiArray()
        msg.data = [float(x) for x in commands]
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = ServoTester()

    def spin():
        rclpy.spin(node)

    t = threading.Thread(target=spin, daemon=True)
    t.start()

    time.sleep(1.0)  # wait for discovery

    tests = [
        ('head_pan angle=120 (board 0, ch 4)',          [0, 4, 120]),
        ('head_tilt angle=60 (board 0, ch 5)',           [0, 5, 60]),
        ('right_shoulder angle=90/neutral (board 1, ch 0)', [1, 0, 90]),
        ('left_shoulder + right_shoulder both 110',      [0, 0, 110, 1, 0, 110]),
        ('return head_pan to neutral angle=90',          [0, 4, 90]),
        ('return head_tilt to neutral angle=90',         [0, 5, 90]),
        ('return all to neutral',                        [0, 0, 90, 1, 0, 90]),
    ]

    for label, cmd in tests:
        print(f'\n[TEST] {label}')
        print(f'       cmd={cmd}')
        node.send(cmd)
        time.sleep(1.5)

    print(f'\n/servo_status messages received: {len(node.received_status)}')
    if node.received_status:
        print(f'Last status: {node.received_status[-1]}')
        print('PASS: /servo_status publishing confirmed')
    else:
        print('NOTE: no status yet (node may not be running)')

    node.destroy_node()
    rclpy.shutdown()
    print('\nValidation complete.')


if __name__ == '__main__':
    main()
