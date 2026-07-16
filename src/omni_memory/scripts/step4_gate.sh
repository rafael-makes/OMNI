#!/usr/bin/env bash
# Step 4 pass gate: end-to-end via the ROS2 CLI.
#   store_transcript (summarize + store) -> retrieve_memories (context block).
# Starts the node, calls both services, asserts, cleans up its own rows.
#
# Run from an interactive shell that has GEMINI_API_KEY (~/.bashrc) and the ROS
# overlay sourced:  bash -ic '~/omni_ws/src/omni_memory/scripts/step4_gate.sh'
WS=~/omni_ws
ENV_FILE=/home/pi/omni_ws/src/omni_memory/.env
STAMP=$(date +%s)
SESSION="step4gate-$STAMP"
# Unique per-run identity so retrieval (which filters by person) only ever sees
# THIS run's rows — keeps the gate isolated and repeatable.
PERSON="gateuser$STAMP"
LOG=$(mktemp)
FAIL=0

source "$WS/install/setup.bash"

echo ">> starting omni_memory node (session=$SESSION)"
# `ros2 run` spawns the node as a CHILD; killing only this wrapper PID orphans
# the node (a live extra service server that then double-processes requests).
# We capture the real node PID after startup and kill both.
ros2 run omni_memory omni_memory_node >"$LOG" 2>&1 &
WRAP_PID=$!
NODE_PID=""

cleanup() {
  echo ">> cleanup: deleting test rows + stopping node"
  python3 - "$SESSION" "$ENV_FILE" <<'PY' || true
import sys
from omni_memory import MemoryStore, load_env
session, env_file = sys.argv[1], sys.argv[2]
load_env(env_file)
s = MemoryStore()
s.client.table(s.table).delete().eq("session_id", session).execute()
print("   deleted rows for", session)
PY
  [ -n "$NODE_PID" ] && kill "$NODE_PID" 2>/dev/null   # the actual node process
  kill "$WRAP_PID" 2>/dev/null                          # the ros2 run wrapper
  wait "$WRAP_PID" 2>/dev/null
}
trap cleanup EXIT

echo ">> waiting for services..."
for i in $(seq 1 30); do
  if ros2 service list 2>/dev/null | grep -q '/omni_memory/store_transcript'; then
    break
  fi
  sleep 1
done
if ! ros2 service list 2>/dev/null | grep -q '/omni_memory/store_transcript'; then
  echo "!! services never appeared. node log:"; cat "$LOG"; exit 1
fi
# Capture the actual node PID (child of the ros2 run wrapper) for clean teardown.
NODE_PID=$(pgrep -f 'lib/omni_memory/omni_memory_node' | head -1)

TRANSCRIPT="$PERSON: Just so you know, I always take my coffee black at 6 AM. Also my mountain bike is kept in the garage."

echo ">> calling store_transcript"
STORE_OUT=$(ros2 service call /omni_memory/store_transcript \
  omni_memory_msgs/srv/StoreTranscript \
  "{transcript: \"$TRANSCRIPT\", person: '$PERSON', session_id: '$SESSION'}" 2>&1)
echo "$STORE_OUT" | sed 's/^/   /'

if echo "$STORE_OUT" | grep -q 'success=True' && \
   echo "$STORE_OUT" | grep -qE 'stored_count=[1-9]'; then
  echo "   [PASS] store_transcript stored memories"
else
  echo "   [FAIL] store_transcript did not store"; FAIL=1
fi

echo ">> calling retrieve_memories (query='coffee')"
RET_OUT=$(ros2 service call /omni_memory/retrieve_memories \
  omni_memory_msgs/srv/RetrieveMemories \
  "{query: 'coffee', k: 3, person: '$PERSON'}" 2>&1)
echo "$RET_OUT" | sed 's/^/   /'

if echo "$RET_OUT" | grep -q 'success=True' && \
   echo "$RET_OUT" | grep -qiE 'count=[1-9]' && \
   echo "$RET_OUT" | grep -qi 'coffee'; then
  echo "   [PASS] retrieve_memories returned a context block mentioning coffee"
else
  echo "   [FAIL] retrieve_memories did not return the expected context"; FAIL=1
fi

if [ "$FAIL" -eq 0 ]; then
  echo ">> STEP 4 GATE: PASS"
else
  echo ">> STEP 4 GATE: FAIL"; echo "node log:"; cat "$LOG"
fi
exit $FAIL
