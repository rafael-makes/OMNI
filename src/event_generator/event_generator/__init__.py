"""event_generator — semantic events from world-state snapshots.

Importing this package must never pull in rclpy: the ROS wrapper lives in
``node.py`` and is reached only through the console-script entry point. Same
convention as ``world_state`` and ``omni_memory``.
"""

from .generator import EventGenerator
from .models import (
    PERSON_APPEARED,
    PERSON_DWELLING,
    PERSON_LEFT,
    UNKNOWN_PERSON_DETECTED,
    Event,
    is_named,
    is_unknown,
)

__all__ = [
    "EventGenerator",
    "Event",
    "PERSON_APPEARED",
    "PERSON_DWELLING",
    "PERSON_LEFT",
    "UNKNOWN_PERSON_DETECTED",
    "is_named",
    "is_unknown",
]
