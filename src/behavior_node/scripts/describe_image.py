#!/usr/bin/env python3
"""describe_image.py — run OMNI's scene description on a saved image.

No ROS, no Jetson, no robot. This is the offline path for the vision logic:
if this works, everything except the frame transport works.

    ./describe_image.py ~/shot.jpg
    ./describe_image.py shot.jpg --prompt ../config/scene_prompt.txt
    ./describe_image.py shot.jpg --max-sentences 1 --show-prompt

Needs GEMINI_API_KEY in the environment. It lives in ~/.bashrc, so run this from
an interactive shell (the same gotcha as every other keyed OMNI script).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Import the module from the source tree without needing behavior_node installed,
# so this runs on a desktop checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from behavior_node.scene_describer import (  # noqa: E402
    DEFAULT_MAX_SENTENCES,
    DEFAULT_MODEL,
    SceneDescriber,
    SceneDescriptionError,
)

# Shipped prompt, relative to this script inside the source tree.
_DEFAULT_PROMPT_PATH = Path(__file__).resolve().parent.parent / "config" / "scene_prompt.txt"

_JPEG_MAGIC = b"\xff\xd8\xff"


def _load_jpeg(path: Path) -> bytes:
    """Read an image, converting to JPEG if it is some other format.

    The describer only accepts JPEG (that is what frame_server sends). Accepting a
    PNG here anyway makes the script useful on whatever screenshot is to hand.
    """
    raw = path.read_bytes()
    if raw.startswith(_JPEG_MAGIC):
        return raw

    try:
        import io

        from PIL import Image
    except ImportError:
        raise SystemExit(
            f"{path.name} is not a JPEG, and Pillow is not installed to convert it.\n"
            f"Either pass a .jpg, or: pip install --break-system-packages pillow"
        )

    buf = io.BytesIO()
    Image.open(io.BytesIO(raw)).convert("RGB").save(buf, format="JPEG", quality=90)
    print(f"[note] converted {path.suffix or 'image'} -> JPEG in memory", file=sys.stderr)
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Describe an image with OMNI's scene-description prompt.",
    )
    ap.add_argument("image", type=Path, help="path to an image file")
    ap.add_argument(
        "--prompt",
        type=Path,
        default=_DEFAULT_PROMPT_PATH,
        help=f"scene prompt file (default: {_DEFAULT_PROMPT_PATH})",
    )
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"default: {DEFAULT_MODEL}")
    ap.add_argument(
        "--max-sentences",
        type=int,
        default=DEFAULT_MAX_SENTENCES,
        help=f"hard cap on the reply (default: {DEFAULT_MAX_SENTENCES})",
    )
    ap.add_argument("--show-prompt", action="store_true", help="print the prompt in use")
    args = ap.parse_args()

    if not args.image.exists():
        print(f"error: no such file: {args.image}", file=sys.stderr)
        return 2
    if not os.environ.get("GEMINI_API_KEY"):
        print(
            "error: GEMINI_API_KEY is not set.\n"
            "It lives in ~/.bashrc — run this from an interactive shell, or:\n"
            "  export GEMINI_API_KEY=...",
            file=sys.stderr,
        )
        return 2

    jpeg = _load_jpeg(args.image)

    if not args.prompt.exists():
        print(
            f"[warn] prompt file not found ({args.prompt}) — using the built-in default",
            file=sys.stderr,
        )

    describer = SceneDescriber(
        model=args.model,
        prompt_path=str(args.prompt),
        max_sentences=args.max_sentences,
    )

    if args.show_prompt:
        print(f"--- prompt ---\n{describer.prompt}\n--------------", file=sys.stderr)

    print(f"[{args.image.name}: {len(jpeg) / 1024:.0f} KB, model {args.model}]", file=sys.stderr)

    started = time.monotonic()
    try:
        description = describer.describe(jpeg)
    except (SceneDescriptionError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    elapsed = time.monotonic() - started

    # Description on stdout alone, so the script pipes cleanly.
    print(description)
    print(f"[{elapsed:.2f}s]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
