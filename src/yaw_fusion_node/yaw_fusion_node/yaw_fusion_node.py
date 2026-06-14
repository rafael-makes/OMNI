"""
yaw_fusion_node — fuses BNO085 yaw with wheel odometry.

Why: Wheel encoder yaw drifts during turns (slip, unequal wheel speeds).
The BNO085 GAME_ROTATION_VECTOR has accurate gyro-integrated yaw with no
long-term drift within a session. This node blends the two:
  - x, y position:     from wheel odometry (encoders measure displacement well)
  - yaw orientation:   from BNO085 IMU (gyro is more accurate than wheel diff)
  - linear velocity:   from wheel odometry
  - angular velocity:  from IMU gyro.z (direct measurement, not derived)

Publishes:
  /odometry/fused  (nav_msgs/Odometry) — for nav2 or diagnostics
  odom → base_link TF — motor_control_node must run with publish_tf:=false

Startup alignment: the BNO085 GAME_ROTATION_VECTOR has an arbitrary zero
heading at power-on. On first IMU message, we record an offset so that the
fused yaw starts at the same heading as wheel odometry (both = 0 at startup).
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import tf2_ros

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu


def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class YawFusionNode(Node):

    def __init__(self):
        super().__init__('yaw_fusion_node')

        # ── State ─────────────────────────────────────────────────────────────
        self._imu_yaw_offset: float | None = None   # IMU→odom frame alignment
        self._last_imu_yaw:   float | None = None
        self._last_gyro_z:    float | None = None
        self._last_odom:      Odometry | None = None

        # ── QoS — match sensor publishers (BEST_EFFORT) ───────────────────────
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ── TF broadcaster ────────────────────────────────────────────────────
        self._tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # ── Publishers ────────────────────────────────────────────────────────
        self._fused_pub = self.create_publisher(Odometry, '/odometry/fused', sensor_qos)

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(Odometry, '/odom',      self._odom_cb, sensor_qos)
        self.create_subscription(Imu,      '/imu/data',  self._imu_cb,  sensor_qos)

        self.get_logger().info('yaw_fusion_node ready — waiting for first IMU message')

    # ── IMU callback (100 Hz) ─────────────────────────────────────────────────

    def _imu_cb(self, msg: Imu):
        q = msg.orientation
        raw_yaw = _quat_to_yaw(q.x, q.y, q.z, q.w)

        if self._imu_yaw_offset is None:
            # Align IMU yaw to odom frame: both should start at 0
            odom_yaw = 0.0
            if self._last_odom is not None:
                oq = self._last_odom.pose.pose.orientation
                odom_yaw = _quat_to_yaw(oq.x, oq.y, oq.z, oq.w)
            self._imu_yaw_offset = raw_yaw - odom_yaw
            self.get_logger().info(
                f'IMU yaw aligned: raw={math.degrees(raw_yaw):.1f}° '
                f'offset={math.degrees(self._imu_yaw_offset):.1f}°')

        self._last_imu_yaw  = raw_yaw - self._imu_yaw_offset
        self._last_gyro_z   = msg.angular_velocity.z

    # ── Odom callback (20 Hz from Pico) ──────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        self._last_odom = msg

        if self._last_imu_yaw is None:
            # IMU not yet ready — pass through raw odom TF so nav2 isn't starved
            self._publish_tf(msg, _quat_to_yaw(
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
            ), msg.twist.twist.angular.z)
            return

        self._publish_tf(msg, self._last_imu_yaw, self._last_gyro_z)

    # ── Publish helpers ───────────────────────────────────────────────────────

    def _publish_tf(self, odom: Odometry, yaw: float, gyro_z: float):
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        now = odom.header.stamp

        # odom → base_link TF
        tf = TransformStamped()
        tf.header.stamp    = now
        tf.header.frame_id = 'odom'
        tf.child_frame_id  = 'base_link'
        tf.transform.translation.x = odom.pose.pose.position.x
        tf.transform.translation.y = odom.pose.pose.position.y
        tf.transform.translation.z = 0.0
        tf.transform.rotation.x = 0.0
        tf.transform.rotation.y = 0.0
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self._tf_broadcaster.sendTransform(tf)

        # /odometry/fused
        fused = Odometry()
        fused.header.stamp    = now
        fused.header.frame_id = 'odom'
        fused.child_frame_id  = 'base_link'
        fused.pose.pose.position   = odom.pose.pose.position
        fused.pose.pose.orientation.x = 0.0
        fused.pose.pose.orientation.y = 0.0
        fused.pose.pose.orientation.z = qz
        fused.pose.pose.orientation.w = qw
        fused.twist.twist.linear.x  = odom.twist.twist.linear.x
        fused.twist.twist.angular.z = gyro_z
        fused.pose.covariance  = odom.pose.covariance
        fused.twist.covariance = odom.twist.covariance
        self._fused_pub.publish(fused)


def main(args=None):
    rclpy.init(args=args)
    node = YawFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
