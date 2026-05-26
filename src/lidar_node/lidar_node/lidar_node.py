import json
import math
import struct
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import serial

from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


# CRC8, polynomial 0x4D, non-reflected, init 0x00 — matches LD19 SDK
_CRC8_TABLE = (
    0x00, 0x4d, 0x9a, 0xd7, 0x79, 0x34, 0xe3, 0xae,
    0xf2, 0xbf, 0x68, 0x25, 0x8b, 0xc6, 0x11, 0x5c,
    0xa9, 0xe4, 0x33, 0x7e, 0xd0, 0x9d, 0x4a, 0x07,
    0x5b, 0x16, 0xc1, 0x8c, 0x22, 0x6f, 0xb8, 0xf5,
    0x1f, 0x52, 0x85, 0xc8, 0x66, 0x2b, 0xfc, 0xb1,
    0xed, 0xa0, 0x77, 0x3a, 0x94, 0xd9, 0x0e, 0x43,
    0xb6, 0xfb, 0x2c, 0x61, 0xcf, 0x82, 0x55, 0x18,
    0x44, 0x09, 0xde, 0x93, 0x3d, 0x70, 0xa7, 0xea,
    0x3e, 0x73, 0xa4, 0xe9, 0x47, 0x0a, 0xdd, 0x90,
    0xcc, 0x81, 0x56, 0x1b, 0xb5, 0xf8, 0x2f, 0x62,
    0x97, 0xda, 0x0d, 0x40, 0xee, 0xa3, 0x74, 0x39,
    0x65, 0x28, 0xff, 0xb2, 0x1c, 0x51, 0x86, 0xcb,
    0x21, 0x6c, 0xbb, 0xf6, 0x58, 0x15, 0xc2, 0x8f,
    0xd3, 0x9e, 0x49, 0x04, 0xaa, 0xe7, 0x30, 0x7d,
    0x88, 0xc5, 0x12, 0x5f, 0xf1, 0xbc, 0x6b, 0x26,
    0x7a, 0x37, 0xe0, 0xad, 0x03, 0x4e, 0x99, 0xd4,
    0x7c, 0x31, 0xe6, 0xab, 0x05, 0x48, 0x9f, 0xd2,
    0x8e, 0xc3, 0x14, 0x59, 0xf7, 0xba, 0x6d, 0x20,
    0xd5, 0x98, 0x4f, 0x02, 0xac, 0xe1, 0x36, 0x7b,
    0x27, 0x6a, 0xbd, 0xf0, 0x5e, 0x13, 0xc4, 0x89,
    0x63, 0x2e, 0xf9, 0xb4, 0x1a, 0x57, 0x80, 0xcd,
    0x91, 0xdc, 0x0b, 0x46, 0xe8, 0xa5, 0x72, 0x3f,
    0xca, 0x87, 0x50, 0x1d, 0xb3, 0xfe, 0x29, 0x64,
    0x38, 0x75, 0xa2, 0xef, 0x41, 0x0c, 0xdb, 0x96,
    0x42, 0x0f, 0xd8, 0x95, 0x3b, 0x76, 0xa1, 0xec,
    0xb0, 0xfd, 0x2a, 0x67, 0xc9, 0x84, 0x53, 0x1e,
    0xeb, 0xa6, 0x71, 0x3c, 0x92, 0xdf, 0x08, 0x45,
    0x19, 0x54, 0x83, 0xce, 0x60, 0x2d, 0xfa, 0xb7,
    0x5d, 0x10, 0xc7, 0x8a, 0x24, 0x69, 0xbe, 0xf3,
    0xaf, 0xe2, 0x35, 0x78, 0xd6, 0x9b, 0x4c, 0x01,
    0xf4, 0xb9, 0x6e, 0x23, 0x8d, 0xc0, 0x17, 0x5a,
    0x06, 0x4b, 0x9c, 0xd1, 0x7f, 0x32, 0xe5, 0xa8,
)

_PKT_START  = 0x54
_PKT_VERLEN = 0x2C   # 12-point packet type
_PKT_SIZE   = 47
_NUM_POINTS = 12


def _crc8(data: bytes | bytearray) -> int:
    crc = 0x00
    for b in data:
        crc = _CRC8_TABLE[(crc ^ b) & 0xFF]
    return crc


class LidarNode(Node):

    def __init__(self):
        super().__init__('lidar_node')

        self.declare_parameter('serial_port',  '/dev/lidar')
        self.declare_parameter('baud_rate',    230400)
        self.declare_parameter('frame_id',     'lidar_link')
        self.declare_parameter('range_min',    0.02)
        self.declare_parameter('range_max',    12.0)
        self.declare_parameter('scan_hz',      10.0)
        self.declare_parameter('num_bins',     360)
        # LD19 motor spins clockwise (raw angles increase CW from above).
        # ROS LaserScan convention is CCW-positive, so we mirror the angles
        # to avoid a left-right flipped map. Set False if using a CCW lidar.
        self.declare_parameter('reverse_scan', True)

        self._port         = self.get_parameter('serial_port').value
        self._baud         = self.get_parameter('baud_rate').value
        self._frame_id     = self.get_parameter('frame_id').value
        self._range_min    = self.get_parameter('range_min').value
        self._range_max    = self.get_parameter('range_max').value
        self._scan_hz      = self.get_parameter('scan_hz').value
        self._num_bins     = self.get_parameter('num_bins').value
        self._reverse_scan = self.get_parameter('reverse_scan').value

        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._scan_pub   = self.create_publisher(LaserScan, '/scan',         sensor_qos)
        self._status_pub = self.create_publisher(String,    '/lidar_status', 10)

        self._ser: serial.Serial | None = None
        self._connect_serial()

        self._pending_points: list[tuple[float, int, int]] = []
        self._prev_end_angle_deg: float | None = None
        self._scan_count = 0
        self._crc_errors = 0

        self._running = True
        self._read_thread = threading.Thread(
            target=self._read_loop, daemon=True, name='lidar_read'
        )
        self._read_thread.start()

        self.create_timer(1.0, self._status_cb)

        self.get_logger().info(
            f'lidar_node started — {self._port} @ {self._baud} baud, '
            f'frame={self._frame_id}, bins={self._num_bins}'
        )

    # ── Serial ────────────────────────────────────────────────────────────────

    def _connect_serial(self):
        try:
            self._ser = serial.Serial(
                self._port, self._baud,
                timeout=1.0,
                write_timeout=1.0,
            )
            time.sleep(2.0)
            self._ser.reset_input_buffer()
            self.get_logger().info(f'Serial connected: {self._port}')
        except serial.SerialException as e:
            self.get_logger().error(f'Serial open failed: {e}')
            self._ser = None

    # ── Read loop (background thread) ─────────────────────────────────────────

    def _read_loop(self):
        try:
            self._read_loop_inner()
        except Exception as e:
            import traceback
            self.get_logger().error(f'read_loop crashed: {e}\n{traceback.format_exc()}')

    def _read_loop_inner(self):
        buf = bytearray()

        while self._running:
            if self._ser is None:
                time.sleep(1.0)
                self._connect_serial()
                buf.clear()
                self._prev_end_angle_deg = None
                continue

            try:
                # Read whatever is available (at least 1 byte, at most 256)
                waiting = self._ser.in_waiting
                chunk = self._ser.read(max(1, min(256, waiting if waiting else 1)))
                if not chunk:
                    continue
                buf.extend(chunk)

                # Consume complete packets from front of buffer
                while len(buf) >= _PKT_SIZE:
                    # Find the next 0x54 start byte
                    idx = buf.find(_PKT_START)
                    if idx == -1:
                        buf = buf[-1:]   # keep last byte in case it's a start byte
                        break
                    if idx > 0:
                        del buf[:idx]    # discard bytes before start byte
                        continue

                    # buf[0] == 0x54; verify verlen byte
                    if len(buf) < 2:
                        break
                    if buf[1] != _PKT_VERLEN:
                        del buf[0]       # spurious 0x54, slide forward
                        continue

                    if len(buf) < _PKT_SIZE:
                        break            # wait for rest of packet

                    pkt = bytes(buf[:_PKT_SIZE])
                    del buf[:_PKT_SIZE]
                    self._process_packet(pkt)

            except serial.SerialException as e:
                # CP2102 through USB hubs sends zero-length packets between
                # scans. Pyserial raises SerialException on empty reads, but
                # this is transient — don't close the port for it.
                if 'readiness to read but returned no data' in str(e):
                    continue
                self.get_logger().warn(f'Serial read error: {e}')
                try:
                    self._ser.close()
                except Exception:
                    pass
                self._ser = None
                buf.clear()

    # ── Packet processing ─────────────────────────────────────────────────────

    def _process_packet(self, pkt: bytes):
        # Verify CRC over bytes 0–45
        if _crc8(pkt[:46]) != pkt[46]:
            self._crc_errors += 1
            return

        start_angle_deg = struct.unpack_from('<H', pkt, 4)[0] / 100.0
        end_angle_deg   = struct.unpack_from('<H', pkt, 42)[0] / 100.0

        # Intra-packet wrap: end < start means this packet crosses the 0° boundary.
        # This is the primary revolution trigger for the LD19 — the 0°/360° seam
        # falls inside one packet, not between packets.
        intra_wrap = end_angle_deg < start_angle_deg

        if intra_wrap:
            arc = (360.0 - start_angle_deg) + end_angle_deg
        else:
            arc = end_angle_deg - start_angle_deg

        step = arc / (_NUM_POINTS - 1) if _NUM_POINTS > 1 else 0.0

        for i in range(_NUM_POINTS):
            offset    = 6 + i * 3
            dist_mm   = struct.unpack_from('<H', pkt, offset)[0]
            intensity = pkt[offset + 2]
            angle_deg = (start_angle_deg + i * step) % 360.0
            self._pending_points.append((angle_deg, dist_mm, intensity))

        # Publish on intra-packet 0° crossing (primary) or inter-packet wrap (fallback)
        if intra_wrap:
            self._publish_scan()
            self._pending_points.clear()
        elif self._prev_end_angle_deg is not None and start_angle_deg < self._prev_end_angle_deg:
            self._publish_scan()
            self._pending_points.clear()

        self._prev_end_angle_deg = end_angle_deg

    # ── Scan publish ──────────────────────────────────────────────────────────

    def _publish_scan(self):
        if not self._pending_points:
            return

        ranges_m    = [float('inf')] * self._num_bins
        intensities = [0.0]          * self._num_bins
        angle_inc   = 360.0 / self._num_bins

        for angle_deg, dist_mm, intensity in self._pending_points:
            if dist_mm == 0:
                continue
            dist_m = dist_mm / 1000.0
            if dist_m < self._range_min or dist_m > self._range_max:
                continue
            # Mirror CW raw angles to CCW ROS convention when reverse_scan=True
            mapped_deg = (360.0 - angle_deg) % 360.0 if self._reverse_scan else angle_deg
            bin_idx = int(mapped_deg / angle_inc) % self._num_bins
            if dist_m < ranges_m[bin_idx]:
                ranges_m[bin_idx]    = dist_m
                intensities[bin_idx] = float(intensity)

        msg = LaserScan()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.angle_min       = 0.0
        msg.angle_max       = 2.0 * math.pi
        msg.angle_increment = (2.0 * math.pi) / self._num_bins
        msg.time_increment  = (1.0 / self._scan_hz) / self._num_bins
        msg.scan_time       = 1.0 / self._scan_hz
        msg.range_min       = self._range_min
        msg.range_max       = self._range_max
        msg.ranges          = ranges_m
        msg.intensities     = intensities

        self._scan_pub.publish(msg)
        self._scan_count += 1

    # ── Status ────────────────────────────────────────────────────────────────

    def _status_cb(self):
        self._status_pub.publish(String(data=json.dumps({
            'connected':  self._ser is not None,
            'port':       self._port,
            'scan_count': self._scan_count,
            'crc_errors': self._crc_errors,
        })))

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def destroy_node(self):
        self._running = False
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LidarNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
