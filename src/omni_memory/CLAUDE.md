# CLAUDE.md — omni_memory

Read this and `SPEC.md` at the start of every session. Work exactly one build
step per session. Write the pass test before the implementation.

## What this is
OMNI's persistent, per-person memory layer. Conversations → discrete memory
records → stored with vector embeddings in the `omni-core` Supabase project →
relevant memories retrieved and injected into the Gemini Live session context.

## Layout
- `omni_memory/` — core library. **No ROS, no heavyweight imports at module load.**
  Must run on a desktop with no ROS installed (SPEC convention).
  - `models.py` — `MemoryRecord` dataclass (validation + row round-trip).
  - `store.py` — `MemoryStore` (Supabase behind a swappable interface) + `load_env`.
- `migrations/` — SQL applied to the omni-core Postgres (Supabase SQL editor or psql).
- `tests/` — pytest. Live tests skip themselves if `SUPABASE_URL` /
  `SUPABASE_SERVICE_KEY` are unset.

## Conventions
- Python 3.11+ (Pi runs 3.13). Type hints everywhere.
- `MemoryStore` and `Embedder` (Step 2) stay free of ROS imports.
- All storage access goes through `MemoryStore` — no raw Supabase calls in app code.
- Config via `.env` (gitignored). Never commit the service key.
- supabase-py access uses the service role key over WireGuard only.

## Environment notes
- Pi 5, system Python 3.13; nodes use system python (not a venv) so the future
  Step 4 ROS node can import this package. Install deps with
  `pip install --break-system-packages -r requirements.txt` (matches how other
  OMNI nodes install bleak / google-genai).
- omni-core Supabase is a **separate** project from the VPS trading system —
  never touch that project's DB.

## Build steps (one per session) — status
- [x] Step 1: DB migration + `MemoryStore` (store/get_by_id/recent). Gate: insert
      3, read back, fields round-trip.
- [x] Step 2: `Embedder` + embeddings on store() + `retrieve()` similarity search.
      Gate: "belt tension" ranks the workshop memory first; person filter verified.
- [x] Step 3: `summarize_transcript()`. Gate: 3 fixtures — fact-rich yields sane
      records, small-talk yields [], correction captures the new value.
- [x] Step 4: ROS2 node `omni_memory` (store_transcript / retrieve_memories) +
      `omni_memory_msgs` srv pkg. Gate: `scripts/step4_gate.sh` (CLI e2e). NOTE: when
      testing, kill the real node PID (not just `ros2 run`) or orphans pile up.
- [x] Step 5: Gemini Live integration. Lives in behavior_node (memory_client.py,
      memory_format.py, gemini_bridge.py, behavior_node.py). Gate:
      behavior_node/scripts/step5_service_gate.sh (auto) + step5_manual_gate.md (live).
- [x] Step 6: per-person keying via face recognition. Recognizer on the Jetson
      (head_detector), consumed by the Pi. Gate: recognised across a restart +
      `behavior_node/scripts/step6_service_gate.sh`. Multi-face identification and
      targeted enrolment landed 2026-07-18 — see SPEC.md "Identity contracts".

## Running the Step 1 gate
```
cd ~/omni_ws/src/omni_memory
pip install --break-system-packages -r requirements.txt
# apply migrations/0001_init.sql to omni-core via Supabase SQL editor
cp .env.example .env   # fill SUPABASE_URL + SUPABASE_SERVICE_KEY
python -m pytest -v
```
