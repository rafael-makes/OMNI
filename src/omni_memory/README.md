# omni_memory

OMNI's persistent per-person memory layer. See `SPEC.md` for the full design and
`CLAUDE.md` for working conventions.

## Step 1 (current): database + basic client

`MemoryStore` — a thin, swappable storage abstraction over the `omni-core`
Supabase/pgvector project. Surface:

```python
from omni_memory import MemoryStore, MemoryRecord

store = MemoryStore()  # reads SUPABASE_URL / SUPABASE_SERVICE_KEY from env or .env

saved = store.store(MemoryRecord(
    content="Rafael prefers coffee around 6 AM.",
    person="rafael", source="conversation", location="kitchen", importance=4,
))

store.get_by_id(saved.id)
store.recent(5, person="rafael")
```

Embeddings arrive in Step 2; the `embedding` column already exists but is
nullable and left unset for now.

## Setup

```
pip install --break-system-packages -r requirements.txt
# apply migrations/0001_init.sql to the omni-core Postgres (Supabase SQL editor / psql)
cp .env.example .env    # fill in SUPABASE_URL + SUPABASE_SERVICE_KEY
python -m pytest -v     # pass test: insert 3, read back, verify round-trip
```

Live tests skip themselves when Supabase env vars are absent.

## Step 4: ROS2 node

`omni_memory` is also an ament_python ROS2 package wrapping Steps 1–3 behind two
services (interfaces in the `omni_memory_msgs` package):

- `/omni_memory/store_transcript` (`StoreTranscript`): summarize a transcript into
  memory records and store them. Idempotent per request within a 300s window.
- `/omni_memory/retrieve_memories` (`RetrieveMemories`): query → plain-text context
  block for injection into the Gemini Live session.

```
cd ~/omni_ws
colcon build --packages-select omni_memory_msgs omni_memory
source install/setup.bash
ros2 launch omni_memory omni_memory.launch.py     # or: ros2 run omni_memory omni_memory_node

# end-to-end CLI gate (starts node, calls both services, asserts, cleans up):
bash -ic 'src/omni_memory/scripts/step4_gate.sh'
```

Credentials: the node reads `env_file` (ROS param, default `.env` in this dir) for
Supabase creds; `GEMINI_API_KEY` comes from the environment (`~/.bashrc`).

**Testing note:** kill the actual node process, not just the `ros2 run` wrapper —
orphaned nodes become extra service servers and cause duplicate stores.
