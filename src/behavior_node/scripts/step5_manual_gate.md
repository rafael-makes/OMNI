# Step 5 — Live pass test (manual)

The SPEC Step 5 gate is a real spoken conversation, so it can't be fully
automated. The automated `step5_service_gate.sh` proves the store→retrieve
wiring; this procedure proves the end-to-end experience.

## Prereqs
- omni-core reachable over WireGuard; `omni_memory/.env` filled in.
- `GEMINI_API_KEY` exported (`~/.bashrc`).
- Built + sourced: `colcon build && source install/setup.bash`.

## Launch
```
ros2 launch behavior_node omni_full_launch.py
# (or minimal: run omni_memory + behavior_node)
#   ros2 run omni_memory omni_memory_node &
#   ros2 run behavior_node behavior_node
```
Confirm in the log: `omni_memory ready …` and (from behavior_node)
`memory: service connection warmed up`.

## Test — teach, then recall in a NEW session
1. **Wake** OMNI (wake word) and state a durable fact, e.g.
   *"OMNI, remember that I take my coffee black, no sugar."*
2. Let the conversation **end** (stop talking ~30 s, or ask OMNI to stop).
   In the log you should see `memory: storing transcript (… chars)` then
   `memory: stored N memory record(s)`.
3. **Wake OMNI again** (new session). You should see
   `memory: retrieved N memory record(s) for injection` in the log.
4. Ask something that should surface the fact, e.g. *"How do I take my coffee?"*
   → **PASS** if OMNI answers from memory (black, no sugar) without being retold
   this session.

## Also verify graceful degradation
- Kill the omni_memory node, then wake OMNI and converse.
  → OMNI must still greet and chat normally (log warns
  `retrieve_memories service unavailable`); no hang beyond the ~2 s timeout.

## Notes
- **Step 6 has since shipped** (face recognition on the Jetson), so this gate no
  longer runs person-less. If `head_detector` is up and recognises you, memories are
  keyed to your person id rather than general/household. To exercise the original
  Step 5 path in isolation, leave the Jetson vision stack down — `/camera/identity`
  goes unpublished, person stays null, and retrieval falls back to the configurable
  `memory_seed_query`.
- Toggle memory off entirely with `memory_enabled:=false`.
