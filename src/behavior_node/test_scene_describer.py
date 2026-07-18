"""Pass test for scene_describer — written before the implementation (CLAUDE.md convention).

Everything here runs with no ROS, no Jetson, and no network: the Gemini client is
faked. The one live test is opt-in and skips itself without GEMINI_API_KEY, matching
how omni_memory gates its Supabase tests.

    python -m pytest test_scene_describer.py -v
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from behavior_node.scene_describer import (  # noqa: E402
    DEFAULT_PROMPT,
    SceneDescriber,
    load_prompt,
)

JPEG_MAGIC = b"\xff\xd8\xff"


# ── Fakes ──────────────────────────────────────────────────────────────────────

class _FakeResponse:
    def __init__(self, text):
        self.text = text


class _FakeModels:
    """Stands in for genai.Client().models — records what it was called with."""

    def __init__(self, text="A tidy workshop with a workbench.", raises=None):
        self.text = text
        self.raises = raises
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.raises:
            raise self.raises
        return _FakeResponse(self.text)


class _FakeClient:
    def __init__(self, models):
        self.models = models


def _describer(models, **kw):
    return SceneDescriber(client=_FakeClient(models), **kw)


# ── The module must not drag ROS in ────────────────────────────────────────────

def test_no_ros_imports():
    """The core module must import on a desktop with no ROS installed.

    Checked against the parsed AST rather than the raw text — the module's own
    docstring names these packages when explaining why it avoids them.
    """
    import ast

    import behavior_node.scene_describer as mod

    banned = {"rclpy", "rosidl", "sensor_msgs", "std_msgs", "omni_vision_msgs"}
    tree = ast.parse(Path(mod.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert not (imported & banned), f"scene_describer must stay ROS-free, found: {imported & banned}"


# ── Core behaviour ─────────────────────────────────────────────────────────────

def test_describe_returns_text():
    models = _FakeModels("A workbench with tools on it.")
    out = _describer(models).describe(JPEG_MAGIC + b"padding")
    assert out == "A workbench with tools on it."


def test_prompt_is_sent_as_system_instruction():
    models = _FakeModels()
    _describer(models, prompt="LOOK AT THIS").describe(JPEG_MAGIC + b"x")
    cfg = models.calls[0]["config"]
    assert "LOOK AT THIS" in str(cfg.system_instruction)


def test_image_bytes_are_forwarded_verbatim():
    """The exact bytes we were handed must reach the API — a resize or re-encode
    bug would show up here as a mismatch."""
    models = _FakeModels()
    payload = JPEG_MAGIC + b"the-actual-image"
    _describer(models).describe(payload)

    part = models.calls[0]["contents"][0]
    assert part.inline_data.data == payload
    assert part.inline_data.mime_type == "image/jpeg"


def test_empty_image_rejected_without_calling_api():
    models = _FakeModels()
    with pytest.raises(ValueError, match="empty"):
        _describer(models).describe(b"")
    assert models.calls == [], "must not burn an API call on an empty image"


def test_non_jpeg_rejected():
    models = _FakeModels()
    with pytest.raises(ValueError, match="JPEG"):
        _describer(models).describe(b"\x89PNG\r\n\x1a\n" + b"nope")
    assert models.calls == []


def test_api_failure_raises_scene_error():
    from behavior_node.scene_describer import SceneDescriptionError

    models = _FakeModels(raises=RuntimeError("503 overloaded"))
    with pytest.raises(SceneDescriptionError):
        _describer(models).describe(JPEG_MAGIC + b"x")


def test_empty_model_reply_raises():
    from behavior_node.scene_describer import SceneDescriptionError

    models = _FakeModels(text="")
    with pytest.raises(SceneDescriptionError):
        _describer(models).describe(JPEG_MAGIC + b"x")


# ── Output length capping — the stated gotcha ──────────────────────────────────

def test_long_description_is_truncated_at_sentence_boundary():
    long = (
        "There is a workbench. Tools hang on the wall. A window lets in light. "
        "A cat sleeps on the chair. Boxes are stacked in the corner."
    )
    out = _describer(_FakeModels(long), max_sentences=2).describe(JPEG_MAGIC + b"x")
    assert out == "There is a workbench. Tools hang on the wall."


def test_truncation_keeps_whole_sentences_not_mid_word():
    models = _FakeModels("One two three. Four five six. Seven eight nine.")
    out = _describer(models, max_sentences=1).describe(JPEG_MAGIC + b"x")
    assert out == "One two three."


def test_short_description_passes_through_untouched():
    models = _FakeModels("A quiet room.")
    assert _describer(models, max_sentences=2).describe(JPEG_MAGIC + b"x") == "A quiet room."


def test_max_output_tokens_is_capped_in_request():
    """Belt and braces: the cap is enforced at the API too, not only post-hoc."""
    models = _FakeModels()
    _describer(models).describe(JPEG_MAGIC + b"x")
    assert models.calls[0]["config"].max_output_tokens is not None


def test_thinking_is_disabled():
    """Regression: thinking tokens come out of max_output_tokens, so leaving
    thinking on makes the model spend the whole budget deliberating and return a
    truncated fragment. Observed live on gemini-3-flash-preview and
    gemini-2.5-flash before thinking_budget=0 was set. It also costs ~0.9s,
    which matters on a ~3s spoken-response budget."""
    models = _FakeModels()
    _describer(models).describe(JPEG_MAGIC + b"x")
    assert models.calls[0]["config"].thinking_config.thinking_budget == 0


def test_whitespace_and_newlines_collapsed_for_speech():
    models = _FakeModels("  A room.\n\n  With a chair.  ")
    out = _describer(models, max_sentences=5).describe(JPEG_MAGIC + b"x")
    assert "\n" not in out
    assert out == "A room. With a chair."


# ── Warmup ─────────────────────────────────────────────────────────────────────

def test_warmup_makes_one_call_and_reports_success():
    models = _FakeModels()
    assert _describer(models).warmup() is True
    assert len(models.calls) == 1


def test_warmup_sends_a_valid_jpeg():
    """The embedded probe must survive describe()'s own JPEG validation."""
    from behavior_node.scene_describer import _WARMUP_JPEG

    assert _WARMUP_JPEG.startswith(JPEG_MAGIC)
    models = _FakeModels()
    _describer(models).warmup()
    assert models.calls[0]["contents"][0].inline_data.data == _WARMUP_JPEG


def test_warmup_never_raises_and_reports_failure():
    """Runs on a startup thread — an exception here must not escape."""
    models = _FakeModels(raises=RuntimeError("no network"))
    assert _describer(models).warmup() is False


# ── Prompt loading ─────────────────────────────────────────────────────────────

def test_load_prompt_reads_file(tmp_path):
    p = tmp_path / "scene_prompt.txt"
    p.write_text("Describe briefly.\n")
    assert load_prompt(str(p)) == "Describe briefly."


def test_load_prompt_falls_back_when_missing(tmp_path):
    assert load_prompt(str(tmp_path / "nope.txt")) == DEFAULT_PROMPT


def test_shipped_prompt_caps_output_length():
    """The real prompt must tell the model to be brief, or spoken output rambles."""
    shipped = Path(__file__).parent / "config" / "scene_prompt.txt"
    assert shipped.exists(), "config/scene_prompt.txt must ship"
    text = shipped.read_text().lower()
    assert any(w in text for w in ("sentence", "brief", "short")), (
        "scene_prompt.txt must constrain output length"
    )


# ── Live test (opt-in) ─────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY") or not os.environ.get("OMNI_LIVE_TESTS"),
    reason="live test: set GEMINI_API_KEY and OMNI_LIVE_TESTS=1",
)
def test_live_describes_a_generated_image():
    """Round-trips a real JPEG through the real API."""
    from PIL import Image, ImageDraw

    buf = io.BytesIO()
    img = Image.new("RGB", (640, 480), (30, 30, 40))
    ImageDraw.Draw(img).ellipse((220, 140, 420, 340), fill=(220, 60, 60))
    img.save(buf, format="JPEG")

    out = SceneDescriber().describe(buf.getvalue())
    assert out and len(out) < 400
    print(f"\nlive description: {out}")
