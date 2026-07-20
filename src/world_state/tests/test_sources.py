"""Source-tag parsing. Importable without rclpy — the parser is module-level
in node.py, so this test skips cleanly on a desktop with no ROS."""

import pytest

node = pytest.importorskip(
    "world_state.node", reason="rclpy not installed (ROS-free desktop)"
)


def test_parses_topic_and_camera():
    assert node.parse_sources(["/camera/identities=head"]) == [
        ("/camera/identities", "head")
    ]


def test_parses_multiple_cameras():
    assert node.parse_sources(
        ["/camera/identities=head", "/rear_camera/identities=rear"]
    ) == [("/camera/identities", "head"), ("/rear_camera/identities", "rear")]


def test_tolerates_whitespace():
    assert node.parse_sources([" /a = head "]) == [("/a", "head")]


def test_untagged_source_is_rejected():
    """Every detection must carry a source camera — Session 10 depends on it."""
    with pytest.raises(ValueError, match="source camera"):
        node.parse_sources(["/camera/identities"])


def test_empty_camera_is_rejected():
    with pytest.raises(ValueError):
        node.parse_sources(["/camera/identities="])
