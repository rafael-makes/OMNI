#!/usr/bin/env bash
# scene_gate.sh — pass test for the describe_scene feature.
#
# Checks the whole path the demo depends on:
#   1. scene_describer unit tests (offline, no robot)
#   2. the CLI script on a saved image (offline, no robot)
#   3. frame_server reachable from the Pi across the link  [needs the Jetson]
#   4. a live frame -> a real description, timed            [needs the Jetson]
#
# Steps 3-4 skip themselves if the Jetson is not up, so this stays useful with
# the robot powered off — which is the point of the offline half.
#
# Requires GEMINI_API_KEY (it lives in ~/.bashrc, so run from an interactive shell).
set -uo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROMPT="$SRC/config/scene_prompt.txt"
FAILED=0

pass() { echo "  PASS  $1"; }
fail() { echo "  FAIL  $1"; FAILED=1; }
skip() { echo "  SKIP  $1"; }

echo "=== 1. scene_describer unit tests ==========================="
if (cd "$SRC" && python -m pytest test_scene_describer.py -q 2>&1 | tail -3); then
  pass "unit tests"
else
  fail "unit tests"
fi

echo
echo "=== 2. CLI on a saved image (robot may be off) =============="
if [ -z "${GEMINI_API_KEY:-}" ]; then
  skip "no GEMINI_API_KEY — run from an interactive shell"
else
  TESTIMG="$(mktemp /tmp/scene_gate_XXXX.jpg)"
  python - "$TESTIMG" <<'PY'
import sys
from PIL import Image, ImageDraw
img = Image.new('RGB', (640, 480), (235, 232, 225))
d = ImageDraw.Draw(img)
d.rectangle((0, 330, 640, 480), fill=(120, 82, 48))
d.ellipse((300, 200, 420, 320), fill=(60, 110, 180))
img.save(sys.argv[1], quality=90)
PY
  OUT="$("$SRC/scripts/describe_image.py" "$TESTIMG" --prompt "$PROMPT" 2>/dev/null)"
  if [ -n "$OUT" ]; then
    pass "CLI returned: $OUT"
  else
    fail "CLI produced no description"
  fi
  rm -f "$TESTIMG"
fi

echo
echo "=== 3. frame_server reachable from this machine ============="
# Retry: with the full stack up (~35 nodes) a single `ros2 service list` often does
# not finish graph discovery inside a short timeout, which made this skip — and
# silently skip step 4 with it — while the service was perfectly healthy.
find_service() {
  for _ in 1 2 3; do
    if timeout 25 ros2 service list 2>/dev/null | grep -q '^/vision/get_camera_frame$'; then
      return 0
    fi
    sleep 2
  done
  return 1
}

if ! command -v ros2 >/dev/null 2>&1; then
  skip "ros2 not on PATH — source the workspace first"
elif ! find_service; then
  skip "/vision/get_camera_frame not visible after 3 tries — Jetson down, or frame_server not running"
  echo "        start it with: ssh Omni 'ros2 launch omni_jetson_bringup head_detector.launch.py'"
else
  pass "/vision/get_camera_frame discovered across the link"

  echo
  echo "=== 4. live frame -> description, timed ====================="
  if [ -z "${GEMINI_API_KEY:-}" ]; then
    skip "no GEMINI_API_KEY"
  else
    python - "$PROMPT" <<'PY'
import sys, time, rclpy
sys.path.insert(0, __import__('os').path.expanduser('~/omni_ws/src/behavior_node'))
from behavior_node.frame_client import FrameClient
from behavior_node.scene_describer import SceneDescriber

rclpy.init()
node = rclpy.create_node('scene_gate')
fc = FrameClient(node, service_timeout=3.0)
fc._warmed.wait(10)          # exclude one-time DDS warmup from the measurement

t0 = time.monotonic(); res = fc.get_frame('head'); t_frame = time.monotonic() - t0
if not res.ok:
    print(f"  FAIL  no frame: {res.message}"); raise SystemExit(1)

sd = SceneDescriber(prompt_path=sys.argv[1])
t1 = time.monotonic(); desc = sd.describe(res.jpeg); t_vis = time.monotonic() - t1
total = t_frame + t_vis

print(f"  frame  {len(res.jpeg)/1024:.0f} KB, age {res.age_seconds:.2f}s, fetch {t_frame:.2f}s")
print(f"  vision {t_vis:.2f}s")
print(f"  {'PASS' if total < 3.0 else 'WARN'}  total {total:.2f}s (budget 3.0s)")
print(f"  ->  {desc}")

bad = fc.get_frame('nonexistent')
print(f"  {'PASS' if not bad.ok else 'FAIL'}  unknown camera_id refused: {bad.message}")

fc.shutdown(); node.destroy_node(); rclpy.shutdown()
PY
    [ $? -ne 0 ] && FAILED=1
  fi
fi

echo
if [ "$FAILED" -eq 0 ]; then echo "GATE PASSED"; else echo "GATE FAILED"; fi
exit "$FAILED"
