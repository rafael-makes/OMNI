# omni_memory

OMNI's persistent per-person memory layer. See `SPEC.md` for the full design and
`CLAUDE.md` for working conventions.

**Status: all six build steps complete and proven live (2026-07-12 → 07-18).**
OMNI recognises who it is talking to and recalls that person's memories across
restarts. See SPEC.md "Known gaps" for what is still unproven.

## Core library

`MemoryStore` — a thin, swappable storage abstraction over the `omni-core`
Supabase/pgvector project. Free of ROS imports, so it runs on a desktop with no
ROS installed. Surface:

```python
from omni_memory import MemoryStore, MemoryRecord

store = MemoryStore()  # reads SUPABASE_URL / SUPABASE_SERVICE_KEY from env or .env

saved = store.store(MemoryRecord(
    content="Rafael prefers coffee around 6 AM.",
    person="rafael", source="conversation", location="kitchen", importance=4,
))

store.get_by_id(saved.id)
store.recent(5, person="rafael")
store.retrieve("what does he drink?", person="rafael")   # similarity search
```

`store()` embeds content automatically (768-dim, `gemini-embedding-001`).
`person=None` means general/household, not "unknown".

## Setup

```
pip install --break-system-packages -r requirements.txt
# apply migrations/*.sql to the omni-core Postgres (Supabase SQL editor / psql)
#   supabase-py cannot run DDL — these are applied by hand
cp .env.example .env    # fill in SUPABASE_URL + SUPABASE_SERVICE_KEY
python -m pytest -v
```

Live tests skip themselves when Supabase env vars are absent.
`GEMINI_API_KEY` comes from `~/.bashrc`, so keyed code needs an interactive shell.

## ROS2 node

`omni_memory` is also an ament_python ROS2 package wrapping the core library behind
three services (interfaces in the `omni_memory_msgs` package):

- `/omni_memory/store_transcript` (`StoreTranscript`): summarize a transcript into
  memory records and store them. Idempotent per request within a 300s window.
- `/omni_memory/retrieve_memories` (`RetrieveMemories`): query → plain-text context
  block for injection into the Gemini Live session.
- `/omni_memory/rekey_person` (`RekeyPerson`): re-label every memory from one person
  id to another — carries an `unknown_N`'s history over once they're given a name.

Retrieval happens **on wake**, storage at **conversation end**. Memory is a soft
dependency: if this node is down, OMNI still converses.

## Identity

Per-person keying comes from face recognition running on the **Jetson**
(`head_detector`), consumed by the Pi via `/camera/identity`. The Pi never runs
recognition; the Jetson never touches the database. Identity topic contracts and the
enrolment rules live in SPEC.md ("Identity contracts", "Behaviour rules").

```
cd ~/omni_ws
colcon build --packages-select omni_memory_msgs omni_memory
source install/setup.bash
ros2 launch omni_memory omni_memory.launch.py     # or: ros2 run omni_memory omni_memory_node

# end-to-end CLI gate (starts node, calls the services, asserts, cleans up):
bash -ic 'src/omni_memory/scripts/step4_gate.sh'
```

Credentials: the node reads `env_file` (ROS param, default `.env` in this dir) for
Supabase creds; `GEMINI_API_KEY` comes from the environment (`~/.bashrc`).

**Testing note:** kill the actual node process, not just the `ros2 run` wrapper —
orphaned nodes become extra service servers and cause duplicate stores.
