"""world_state — OMNI's shared "who is where right now" tracker.

The core (``models``, ``tracker``) is ROS-free and importable on any desktop.
The ROS2 wrapper lives in ``world_state.node`` and is imported only by the
node entry point, so ``import world_state`` never pulls in rclpy.
"""

from .models import Detection, PersonTrack, StateEvent
from .tracker import DEFAULT_VISIBILITY_TIMEOUT, WorldState

__all__ = [
    "Detection",
    "PersonTrack",
    "StateEvent",
    "WorldState",
    "DEFAULT_VISIBILITY_TIMEOUT",
]
