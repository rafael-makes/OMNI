#!/usr/bin/env bash
# Step 6 gate: two people get differentiated memory retrieval (via the services).
#   bash -ic '~/omni_ws/src/behavior_node/scripts/step6_service_gate.sh'
WS=~/omni_ws
LOG=$(mktemp)
source "$WS/install/setup.bash"

ros2 run omni_memory omni_memory_node >"$LOG" 2>&1 &
WRAP_PID=$!
NODE_PID=""
cleanup() {
  [ -n "$NODE_PID" ] && kill "$NODE_PID" 2>/dev/null
  kill "$WRAP_PID" 2>/dev/null
  wait "$WRAP_PID" 2>/dev/null
}
trap cleanup EXIT

for i in $(seq 1 30); do
  ros2 service list 2>/dev/null | grep -q '/omni_memory/store_transcript' && break; sleep 1
done
if ! ros2 service list 2>/dev/null | grep -q '/omni_memory/store_transcript'; then
  echo "!! omni_memory services never appeared"; cat "$LOG"; exit 1
fi
NODE_PID=$(pgrep -f 'lib/omni_memory/omni_memory_node' | head -1)

python3 "$WS/src/behavior_node/scripts/step6_service_client.py"
