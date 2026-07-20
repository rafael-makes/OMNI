"""ROS2 wrapper around EventGenerator.

Thin by design, same contract as world_state's node: parse the snapshot, hand it
to the core, republish whatever comes back. No behaviour, no LLM calls, no
decisions — behavior_node decides what a greeting is.

Topics in:
  ``/omni/world_state``   std_msgs/String, the world_state snapshot as JSON, 1 Hz

Topics out:
  ``/omni/events``        std_msgs/String, one JSON event per message

Events are published **one message per event**, not batched: a subscriber that
wants to act on person_appeared should not have to unpack an array and work out
which element concerns it. They are published on a depth-10 reliable queue —
these are rare, meaningful transitions, and dropping one means a missed greeting
or a stuck presence belief, so best-effort would be the wrong trade.

Clock note: time comes from the snapshot's own ``stamp`` field, which world_state
set from the **Pi's** clock. This node never calls time.time() for event
timestamps — mixing its own clock with the snapshot's would make away_duration
wrong by however long the topic took to arrive.
"""

from __future__ import annotations

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .generator import (
    DEFAULT_ABSENCE_GRACE,
    DEFAULT_NAMED_OVERLAP_RADIUS,
    DEFAULT_UNKNOWN_MIN_FACE_PX,
    DEFAULT_APPEAR_MIN_SNAPSHOTS,
    DEFAULT_UNKNOWN_CAMERAS,
    DEFAULT_UNKNOWN_MIN_SNAPSHOTS,
    EventGenerator,
)


class EventGeneratorNode(Node):
    def __init__(self) -> None:
        super().__init__("event_generator_node")

        self.declare_parameter("state_topic", "/omni/world_state")
        self.declare_parameter("events_topic", "/omni/events")
        self.declare_parameter("absence_grace", DEFAULT_ABSENCE_GRACE)
        self.declare_parameter("unknown_min_snapshots", DEFAULT_UNKNOWN_MIN_SNAPSHOTS)
        self.declare_parameter("appear_min_snapshots", DEFAULT_APPEAR_MIN_SNAPSHOTS)
        self.declare_parameter("unknown_cameras", list(DEFAULT_UNKNOWN_CAMERAS))
        self.declare_parameter(
            "named_overlap_radius", DEFAULT_NAMED_OVERLAP_RADIUS)
        self.declare_parameter(
            "unknown_min_face_px", DEFAULT_UNKNOWN_MIN_FACE_PX)
        # Writes every inbound snapshot to a JSONL file. Off by default; this is
        # how the replay fixture in tests/fixtures was captured, and how the next
        # one should be.
        self.declare_parameter("record_path", "")

        state_topic = self.get_parameter("state_topic").value
        events_topic = self.get_parameter("events_topic").value

        self._gen = EventGenerator(
            absence_grace=float(self.get_parameter("absence_grace").value),
            unknown_min_snapshots=int(
                self.get_parameter("unknown_min_snapshots").value),
            appear_min_snapshots=int(
                self.get_parameter("appear_min_snapshots").value),
            unknown_cameras=list(self.get_parameter("unknown_cameras").value),
            named_overlap_radius=float(
                self.get_parameter("named_overlap_radius").value),
            unknown_min_face_px=float(
                self.get_parameter("unknown_min_face_px").value),
        )

        self._record_path = str(self.get_parameter("record_path").value) or None
        self._record_fh = None
        if self._record_path:
            try:
                self._record_fh = open(self._record_path, "a", buffering=1)
                self.get_logger().info(
                    f"recording world_state snapshots to {self._record_path}")
            except OSError as exc:
                self.get_logger().warning(f"cannot open record_path: {exc}")

        self._pub = self.create_publisher(String, events_topic, 10)
        self.create_subscription(String, state_topic, self._on_state, 10)

        self.get_logger().info(
            f"event_generator_node started — {state_topic} -> {events_topic}, "
            f"absence_grace={self._gen.absence_grace}s, "
            f"unknown_min_snapshots={self._gen.unknown_min_snapshots}, "
            f"unknown_cameras={sorted(self._gen.unknown_cameras)}, "
            f"named_overlap_radius={self._gen.named_overlap_radius}px, "
            f"unknown_min_face_px={self._gen.unknown_min_face_px}px"
        )

    def _on_state(self, msg: String) -> None:
        try:
            snapshot = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            self.get_logger().warning(
                "unparseable world_state payload, ignoring", once=True)
            return

        if self._record_fh is not None:
            try:
                self._record_fh.write(msg.data + "\n")
            except OSError:
                pass

        for event in self._gen.ingest(snapshot):
            payload = json.dumps(event.as_dict())
            self._pub.publish(String(data=payload))
            self.get_logger().info(
                f"{event.kind}: {event.identity} ({event.camera})"
                + (f" — away {event.away_duration:.0f}s"
                   if event.away_duration is not None else "")
            )

    def destroy_node(self) -> bool:
        if self._record_fh is not None:
            try:
                self._record_fh.close()
            except OSError:
                pass
            self._record_fh = None
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = EventGeneratorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
