#!/usr/bin/env python3
import json
import os
import time

import board
import busio
import rclpy
import yaml
from adafruit_pca9685 import PCA9685
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

CONFIG_PATH = os.path.expanduser('~/omni_ws/config/servo_config.yaml')
INIT_STEP_DELAY = 0.15   # seconds between each servo at startup


class ServoNode(Node):
    def __init__(self):
        super().__init__('servo_node')
        self.boards = {}          # {board_id: PCA9685}
        self.servo_cfg = {}       # {name: {board, channel, neutral_pw, min_pw, max_pw}}
        self.channel_map = {}     # {(board_id, channel): name}

        self._load_config()
        self._init_hardware()

        self.sub_cmd = self.create_subscription(
            Float32MultiArray, '/servo_commands', self._on_command, 10)
        self.pub_status = self.create_publisher(String, '/servo_status', 10)
        self.timer_status = self.create_timer(1.0, self._publish_status)

        self.get_logger().info(f'servo_node ready — {len(self.servo_cfg)} servos loaded')

    # ------------------------------------------------------------------ config

    def _load_config(self):
        try:
            with open(CONFIG_PATH) as f:
                data = yaml.safe_load(f)
            self.servo_cfg = data.get('servos', {})
            for name, cfg in self.servo_cfg.items():
                self.channel_map[(cfg['board'], cfg['channel'])] = name
            self.get_logger().info(f'Loaded servo config: {CONFIG_PATH}')
        except Exception as e:
            self.get_logger().error(f'Config load failed: {e} — using empty config')
            self.servo_cfg = {}

    # ------------------------------------------------------------------ hardware init

    def _init_hardware(self):
        try:
            i2c = busio.I2C(board.SCL, board.SDA)

            pca0 = PCA9685(i2c, address=0x40)
            pca0.frequency = 50
            self.boards[0] = pca0

            pca1 = PCA9685(i2c, address=0x41)
            pca1.frequency = 50
            self.boards[1] = pca1

            self.get_logger().info('PCA9685 boards up: 0x40 (board 0), 0x41 (board 1)')
        except Exception as e:
            self.get_logger().error(f'PCA9685 init failed: {e}')
            return

        # Move each servo to neutral one at a time — never all at once
        for name, cfg in self.servo_cfg.items():
            try:
                pw = cfg['neutral_pw']
                self._set_pw_raw(cfg['board'], cfg['channel'], pw)
                self.get_logger().info(f'  {name} → neutral {pw} µs')
                time.sleep(INIT_STEP_DELAY)
            except Exception as e:
                self.get_logger().error(f'  {name} init failed: {e}')

    # ------------------------------------------------------------------ PWM helpers

    @staticmethod
    def _pw_to_duty(pw_us: float) -> int:
        # 50 Hz → 20 000 µs period; 16-bit duty_cycle range
        return int(pw_us / 20_000.0 * 65_535)

    def _angle_to_pw(self, name: str, angle: float) -> int:
        cfg = self.servo_cfg[name]
        neutral = cfg['neutral_pw']
        min_pw  = cfg['min_pw']
        max_pw  = cfg['max_pw']
        angle   = max(0.0, min(180.0, angle))

        # 90° always maps to neutral_pw; ranges are asymmetric if neutral is off-center
        if angle < 90.0:
            pw = neutral - (90.0 - angle) / 90.0 * (neutral - min_pw)
        elif angle > 90.0:
            pw = neutral + (angle - 90.0) / 90.0 * (max_pw - neutral)
        else:
            pw = float(neutral)

        return int(max(min_pw, min(max_pw, pw)))

    def _set_pw_raw(self, board_id: int, channel: int, pw_us: float):
        self.boards[board_id].channels[channel].duty_cycle = self._pw_to_duty(pw_us)

    def _set_servo(self, board_id: int, channel: int, angle: float):
        name = self.channel_map.get((board_id, channel))
        if name:
            pw = self._angle_to_pw(name, angle)
        else:
            # Unknown channel — plain linear 500–2500 µs
            pw = int(500 + angle / 180.0 * 2000)

        try:
            self._set_pw_raw(board_id, channel, pw)
            if name:
                self.get_logger().debug(f'{name} angle={angle:.1f} → {pw} µs')
        except Exception as e:
            self.get_logger().error(f'I2C error board{board_id} ch{channel}: {e} — reconnecting')
            self._reconnect()

    # ------------------------------------------------------------------ reconnect

    def _reconnect(self):
        self.get_logger().warning('Reconnecting PCA9685 boards...')
        try:
            i2c = busio.I2C(board.SCL, board.SDA)
            pca0 = PCA9685(i2c, address=0x40)
            pca0.frequency = 50
            self.boards[0] = pca0
            pca1 = PCA9685(i2c, address=0x41)
            pca1.frequency = 50
            self.boards[1] = pca1
            self.get_logger().info('PCA9685 reconnected')
        except Exception as e:
            self.get_logger().error(f'Reconnect failed: {e}')

    # ------------------------------------------------------------------ callbacks

    def _on_command(self, msg: Float32MultiArray):
        data = msg.data
        if len(data) == 0 or len(data) % 3 != 0:
            self.get_logger().warning(f'Bad command length {len(data)} (need multiples of 3)')
            return

        for i in range(0, len(data), 3):
            board_id = int(data[i])
            channel  = int(data[i + 1])
            angle    = float(data[i + 2])

            if board_id not in (0, 1):
                self.get_logger().warning(f'Invalid board_id {board_id}')
                continue
            if not 0 <= channel <= 15:
                self.get_logger().warning(f'Invalid channel {channel}')
                continue

            self._set_servo(board_id, channel, angle)

    def _publish_status(self):
        status = {
            'node': 'servo_node',
            'boards': sorted(self.boards.keys()),
            'servos': len(self.servo_cfg),
        }
        msg = String()
        msg.data = json.dumps(status)
        self.pub_status.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ServoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
