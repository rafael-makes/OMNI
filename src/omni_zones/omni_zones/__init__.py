"""omni_zones — named rooms as map-frame polygons, shared by behavior_node and
world_state. Pure Python, no ROS: it imports and runs on a desktop with no ROS
installed, the same convention as world_state's core.
"""

from .geometry import bearing_from_bbox, estimate_person_xy, standoff_pose
from .zones import (
    Point,
    Pose,
    Zone,
    ZoneMap,
    load_zone_map,
    point_in_polygon,
    polygon_centroid,
    zone_from_dict,
)

__all__ = [
    "Point",
    "Pose",
    "Zone",
    "ZoneMap",
    "load_zone_map",
    "point_in_polygon",
    "polygon_centroid",
    "zone_from_dict",
    "bearing_from_bbox",
    "estimate_person_xy",
    "standoff_pose",
]
