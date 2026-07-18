"""identity_monitor.py — live view of what the face recogniser is seeing.

Multi-person debugging aid: /camera/identity carries only the PRIMARY (largest)
face, so with several people in frame it is not obvious who is being reported.
This prints the face count, the primary face's size, and the published identity
together, so you can see the hand-over as people move.

Run on the Pi or the Jetson (topics are shared):
    python3 identity_monitor.py
"""
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray


class IdentityMonitor(Node):
    def __init__(self):
        super().__init__("identity_monitor")
        self._identity = ""
        self._faces = []
        self._last_line = ""
        sensor_qos = QoSPresetProfiles.SENSOR_DATA.value
        self.create_subscription(String, "/camera/identity", self._on_identity, 10)
        self.create_subscription(Detection2DArray, "/camera/faces", self._on_faces, sensor_qos)
        self.create_timer(0.5, self._tick)
        print("watching /camera/identity + /camera/faces — Ctrl-C to stop\n")

    def _on_identity(self, msg):
        self._identity = msg.data or ""

    def _on_faces(self, msg):
        # largest first, mirroring the node's own primary-face choice
        self._faces = sorted(
            (d.bbox.size_x * d.bbox.size_y, d.bbox.size_x) for d in msg.detections
        )[::-1]

    def _tick(self):
        n = len(self._faces)
        primary = f"{int(self._faces[0][1])}px" if n else "-"
        second = f"{int(self._faces[1][1])}px" if n > 1 else "-"
        who = self._identity or "(nobody)"
        line = f"faces={n}  primary={primary}  2nd={second}  identity={who}"
        # only reprint when something meaningful changes, plus a heartbeat
        if line != self._last_line:
            print(f"[{time.strftime('%H:%M:%S')}] {line}")
            self._last_line = line


def main():
    rclpy.init()
    node = IdentityMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
