"""ROS2 wrapper around the WorldState core.

Thin by design: it translates topics into :class:`Detection` events, ticks the
tracker, and republishes the snapshot. No behaviour, no LLM calls, no decisions.

Topics in (all configurable, each tagged with its source camera):
  ``/camera/identities``  std_msgs/String, JSON ``{"faces": [{track, identity, ...}]}``
  ``/camera/detections``  vision_msgs/Detection2DArray, YOLO boxes (class ``person``)

Topics out:
  ``/omni/world_state``   std_msgs/String, the full snapshot as JSON, 1 Hz

Services:
  ``~/query_world_state``  std_srvs/Trigger -> the same JSON in ``message``

Clock note: detections are stamped with the **Pi's** clock on arrival, not with
the header stamp. The Jetson publishes these and the two clocks are not
synchronised — mixing them would make ages meaningless.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Optional

import rclpy
import rclpy.time
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from vision_msgs.msg import Detection2DArray

import yaml
from omni_zones import bearing_from_bbox, estimate_person_xy, load_zone_map

from .models import Detection
from .tracker import WorldState

# "topic=camera" — the camera tag is what Session 10 needs, so every source
# must carry one. A source listed without "=camera" is rejected at startup.
DEFAULT_IDENTITY_SOURCES = ["/camera/identities=head"]
# Person boxes are OFF by default. /camera/identities already carries the
# Jetson's own per-face track ids and identities, which is a far better basis
# for "who is where" than geometry over anonymous YOLO boxes. Mixing the two
# double-counted every person on the first live run: a face track and a body
# track for one human, permanently separate, because a body track's centre is
# always nearer an incoming body box than a face track's centre is.
# Enable only if you need "a body is present but no face is visible".
DEFAULT_DETECTION_SOURCES: list[str] = []


# Camera model defaults, per camera tag. The head cam looks along robot-forward
# (offset 0); the rear cam is bolted on facing backward (offset 180). These feed
# only the *coarse* person-location estimate — see omni_zones/geometry.py for
# why "coarse" is load-bearing here (assumed distance, ignored head pan).
DEFAULT_CAMERA_OFFSETS_DEG = ["head=0.0", "rear=180.0"]
DEFAULT_CAMERA_HFOV_DEG = ["head=66.0", "rear=66.0"]
DEFAULT_CAMERA_IMAGE_WIDTH = ["head=1920.0", "rear=1920.0"]


def parse_sources(entries: list[str]) -> list[tuple[str, str]]:
    sources: list[tuple[str, str]] = []
    for entry in entries:
        if "=" not in entry:
            raise ValueError(
                f"source {entry!r} must be 'topic=camera' — every detection "
                f"needs a source camera tag"
            )
        topic, camera = entry.split("=", 1)
        topic, camera = topic.strip(), camera.strip()
        if not topic or not camera:
            raise ValueError(f"source {entry!r} has an empty topic or camera")
        sources.append((topic, camera))
    return sources


def parse_camera_floats(entries: list[str]) -> dict[str, float]:
    """Parse a ``['camera=value', ...]`` param into ``{camera: float}``."""
    out: dict[str, float] = {}
    for entry in entries or []:
        if "=" not in entry:
            raise ValueError(f"camera value {entry!r} must be 'camera=number'")
        cam, val = entry.split("=", 1)
        out[cam.strip()] = float(val)
    return out


def _load_zones_config(path: str):
    """Read the ``zones:`` mapping out of a YAML file and build a ZoneMap.

    An empty/absent file (or one without a ``zones:`` key) yields an empty map,
    which disables zone labelling cleanly rather than raising."""
    if not path or not os.path.exists(path):
        return load_zone_map({})
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    return load_zone_map(data.get("zones") or {})


class WorldStateNode(Node):
    def __init__(self) -> None:
        super().__init__("world_state_node")

        self.declare_parameter("identity_sources", DEFAULT_IDENTITY_SOURCES)
        self.declare_parameter("detection_sources", DEFAULT_DETECTION_SOURCES)
        self.declare_parameter("visibility_timeout", 3.0)
        self.declare_parameter("publish_rate", 1.0)
        self.declare_parameter("min_confidence", 0.0)
        self.declare_parameter("match_radius", 160.0)
        self.declare_parameter("rejoin_window", 20.0)
        self.declare_parameter("person_class", "person")
        self.declare_parameter("state_topic", "/omni/world_state")

        # ── Coarse person location (map-frame zone labelling) ──────────────────
        # Off unless a robot pose is available; degrades to identity-only "who"
        # when localisation or the zone map is missing.
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("robot_frame", "base_link")
        # Path to a YAML file with a top-level `zones:` mapping. Empty -> the
        # default shipped with omni_zones (which ships empty until the space is
        # traced into rooms). behavior_node reads the SAME file, so keep one copy.
        self.declare_parameter("zones_config", "")
        self.declare_parameter("camera_offsets_deg", DEFAULT_CAMERA_OFFSETS_DEG)
        self.declare_parameter("camera_hfov_deg", DEFAULT_CAMERA_HFOV_DEG)
        self.declare_parameter("camera_image_width", DEFAULT_CAMERA_IMAGE_WIDTH)
        # Assumed range to a detected person, metres. Monocular vision gives no
        # depth; this is a conversational-distance guess, not a measurement.
        self.declare_parameter("assumed_person_distance", 1.5)

        identity_sources = parse_sources(
            self.get_parameter("identity_sources").value or []
        )
        detection_sources = parse_sources(
            self.get_parameter("detection_sources").value or []
        )
        self._person_class = self.get_parameter("person_class").value.lower()
        publish_rate = float(self.get_parameter("publish_rate").value)

        self._state = WorldState(
            visibility_timeout=float(self.get_parameter("visibility_timeout").value),
            min_confidence=float(self.get_parameter("min_confidence").value),
            match_radius=float(self.get_parameter("match_radius").value),
            rejoin_window=float(self.get_parameter("rejoin_window").value),
        )

        # ── Location model ─────────────────────────────────────────────────────
        self._map_frame = self.get_parameter("map_frame").value
        self._robot_frame = self.get_parameter("robot_frame").value
        zones_path = self.get_parameter("zones_config").value or os.path.join(
            get_package_share_directory("omni_zones"), "config", "zones.yaml"
        )
        self._zones = _load_zones_config(zones_path)
        self._cam_offset = {
            cam: math.radians(v)
            for cam, v in parse_camera_floats(
                self.get_parameter("camera_offsets_deg").value
            ).items()
        }
        self._cam_hfov = {
            cam: math.radians(v)
            for cam, v in parse_camera_floats(
                self.get_parameter("camera_hfov_deg").value
            ).items()
        }
        self._cam_width = parse_camera_floats(
            self.get_parameter("camera_image_width").value
        )
        self._person_distance = float(
            self.get_parameter("assumed_person_distance").value
        )
        # TF is how we learn where the robot is. Missing transforms are normal
        # (localisation not up yet) and simply leave detections unlocated.
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._pub = self.create_publisher(
            String, self.get_parameter("state_topic").value, 10
        )

        # Identities are latched state on a String topic — default (reliable) QoS.
        for topic, camera in identity_sources:
            self.create_subscription(
                String, topic,
                lambda msg, cam=camera: self._on_identities(msg, cam),
                10,
            )
        # Detection2DArray comes off the Jetson on SENSOR_DATA (best effort);
        # subscribing reliable here would silently match nothing.
        sensor_qos = QoSPresetProfiles.SENSOR_DATA.value
        for topic, camera in detection_sources:
            self.create_subscription(
                Detection2DArray, topic,
                lambda msg, cam=camera: self._on_detections(msg, cam),
                sensor_qos,
            )

        self.create_timer(1.0 / publish_rate, self._publish_cb)
        self._srv = self.create_service(
            Trigger, "~/query_world_state", self._on_query
        )

        self.get_logger().info(
            f"world_state_node started — "
            f"identities={identity_sources}, detections={detection_sources}, "
            f"visibility_timeout={self._state.visibility_timeout}s, "
            f"publishing {self._pub.topic_name} at {publish_rate} Hz"
        )
        self.get_logger().info(
            f"zone labelling: {len(self._zones)} zone(s) from {zones_path} "
            f"({', '.join(self._zones.names) or 'none'}); "
            f"assumed person distance {self._person_distance}m. "
            f"Coarse (room-level) — see omni_zones/geometry.py."
        )

    # ── location ────────────────────────────────────────────────────────────────

    def _robot_pose(self) -> Optional[tuple[float, float, float]]:
        """(x, y, yaw) of the robot in the map frame from TF, or None if the
        transform is unavailable (localisation not up, tree not connected). We
        take the *latest* transform, not a stamped one — the detection clocks and
        the TF clock are the same Pi clock, but 'latest' sidesteps the question
        entirely and a few tens of ms of staleness is nothing at room scale."""
        try:
            tf = self._tf_buffer.lookup_transform(
                self._map_frame, self._robot_frame, rclpy.time.Time()
            )
        except Exception:
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        yaw = 2.0 * math.atan2(q.z, q.w)
        return (t.x, t.y, yaw)

    def _locate(
        self,
        pose: Optional[tuple[float, float, float]],
        camera: str,
        cx: Optional[float],
    ) -> tuple[Optional[tuple[float, float]], Optional[str]]:
        """Estimate a detection's map (x, y) and the zone it falls in.

        Returns (None, None) when there is no robot pose. When the estimated
        point lands in no zone but the robot itself is in one, we report the
        robot's zone — a person being spoken to at conversational range is
        almost always in the robot's room, and this is the honest coarse answer
        (see omni_zones/geometry.py)."""
        if pose is None:
            return (None, None)
        rx, ry, ryaw = pose
        offset = self._cam_offset.get(camera, 0.0)
        hfov = self._cam_hfov.get(camera, 0.0)
        width = self._cam_width.get(camera, 0.0)
        bearing = bearing_from_bbox(cx, width, hfov) if cx is not None else 0.0
        xy = estimate_person_xy(rx, ry, ryaw, offset, bearing, self._person_distance)
        zone = self._zones.zone_at(*xy) or self._zones.zone_at(rx, ry)
        return (xy, zone)

    # ── ingest ────────────────────────────────────────────────────────────────

    def _on_identities(self, msg: String, camera: str) -> None:
        """``{"faces": [{track, identity, raw, primary, cx, cy, bw, bh}, ...]}``"""
        try:
            payload = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            self.get_logger().warning(
                f"[{camera}] unparseable identities payload, ignoring", once=True
            )
            return

        now = time.time()
        pose = self._robot_pose()
        for face in payload.get("faces", []):
            if not isinstance(face, dict):
                continue
            bbox = None
            if all(k in face for k in ("cx", "cy", "bw", "bh")):
                bbox = (
                    float(face["cx"]), float(face["cy"]),
                    float(face["bw"]), float(face["bh"]),
                )
            cx = bbox[0] if bbox else None
            map_xy, zone = self._locate(pose, camera, cx)
            self._state.observe(
                Detection(
                    camera=camera,
                    timestamp=now,
                    identity=face.get("identity"),
                    # head_detector publishes no per-face score on this topic —
                    # identity has already survived its own hysteresis and
                    # quality gate upstream, so a recognised face counts as 1.0.
                    confidence=float(face.get("confidence", 1.0)),
                    bbox=bbox,
                    source_track=face.get("track"),
                    map_xy=map_xy,
                    zone=zone,
                )
            )

    def _on_detections(self, msg: Detection2DArray, camera: str) -> None:
        now = time.time()
        pose = self._robot_pose()
        for det in msg.detections:
            score = max(
                (r.hypothesis.score for r in det.results
                 if r.hypothesis.class_id.lower() == self._person_class),
                default=None,
            )
            if score is None:
                continue
            cx = det.bbox.center.position.x
            map_xy, zone = self._locate(pose, camera, cx)
            # A bare person box carries no identity. It is tracked anonymously
            # and gets absorbed if the recognizer later names the same track.
            self._state.observe(
                Detection(
                    camera=camera,
                    timestamp=now,
                    identity=None,
                    confidence=float(score),
                    bbox=(
                        cx, det.bbox.center.position.y,
                        det.bbox.size_x, det.bbox.size_y,
                    ),
                    source_track=self._source_track_of(det),
                    map_xy=map_xy,
                    zone=zone,
                )
            )

    @staticmethod
    def _source_track_of(det) -> int | None:
        """vision_msgs can carry a tracker id in ``det.id``, but head_detector
        never sets it (verified against `_build_array`), so this returns None in
        practice and association falls back to centroid matching. Kept for the
        day the Jetson does track person boxes."""
        raw = getattr(det, "id", "")
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    # ── output ────────────────────────────────────────────────────────────────

    def _snapshot_json(self) -> str:
        now = time.time()
        for event in self._state.tick(now):
            self.get_logger().info(
                f"{event.identity or f'person_{event.track_id}'} "
                f"{event.kind} ({event.camera}): {event.detail}"
            )
        return json.dumps(self._state.snapshot(now))

    def _publish_cb(self) -> None:
        self._pub.publish(String(data=self._snapshot_json()))

    def _on_query(self, _request, response):
        response.success = True
        response.message = self._snapshot_json()
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = WorldStateNode()
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
