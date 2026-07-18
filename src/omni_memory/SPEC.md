# SPEC — OMNI persistent per-person memory

> **Reconstructed 2026-07-18.** The original SPEC.md was never committed and was lost
> from disk, while `CLAUDE.md` still directed every session to read it. This rebuilds
> the design from the shipped code, the migrations, the service definitions, and the
> commit history. It documents the system **as built**; where the original spec's
> intent can't be recovered from the artefacts, that's marked rather than invented.

## Goal

OMNI remembers the people it talks to. A conversation is distilled into discrete
memory records, stored with vector embeddings, and the relevant ones are injected
back into the next conversation with that person — so OMNI can be told something
once and recall it days later.

The proof this works, from the first live run: told OMNI "I like Monster energy
drinks", ended the chat, re-woke it — it recognised Rafael by face and answered
correctly when asked what he likes to drink.

## Shape of the system

Two machines. The **Jetson** sees and recognises; the **Pi** remembers and talks.

```
Jetson (vision)                          Pi (behaviour + memory)
  head_detector_node
    YOLO26n  -> /camera/detections  ─────► behavior_node
    YuNet    -> /camera/faces       ─────►   (head tracking)
    SFace    -> /camera/identity    ─────►   who am I talking to?
             -> /camera/identities  ─────►   everyone in the room
             ◄──── /camera/enroll_request    learn this face as <name>
             ────► /camera/enroll_result
                                            behavior_node ◄──► gemini_bridge
                                                  │
                                                  ▼  services
                                            omni_memory_node
                                                  │
                                                  ▼
                                       Supabase `omni-core` (pgvector)
```

The Pi consumes an identity; it never runs recognition. The Jetson resolves an
identity; it never touches the database. That split is deliberate — recognition
needs the GPU and the frames, memory needs the network and the API keys.

## Data model

`public.memories` (migration `0001_init.sql`):

| column | notes |
| --- | --- |
| `id` | uuid, primary key |
| `content` | the memory statement itself |
| `embedding` | `vector(768)` — nullable, populated from Step 2 onward |
| `person` | person id; **null = general / household**, not "unknown" |
| `source` | `conversation` \| `observation` \| `system` |
| `location` | room tag if known |
| `session_id` | groups memories from one conversation |
| `importance` | smallint 1–5 |
| `created_at` | timestamptz |

Indexes: `person`, `created_at desc`, `session_id`, plus an HNSW cosine index on
`embedding`. Nulls are skipped by the ANN index, so Step 1 rows with no embedding
were always safe.

`match_memories()` (migration `0002_match_memories.sql`) does the similarity search
as a PostgREST RPC. **Person filtering rule:** `filter_person = null` makes all rows
eligible; when set, results are that person's records **plus** general (null-person)
ones. Recency/importance boosting is applied client-side in `MemoryStore`.

> supabase-py cannot run DDL — migrations are applied **by hand** in Supabase Studio.

## Build steps

Convention: one step per session, pass test written before the implementation.

- **Step 1 — DB + `MemoryStore`.** `store` / `get_by_id` / `recent` over Supabase
  behind a swappable interface. Gate: insert 3, read back, fields round-trip.
- **Step 2 — `Embedder` + retrieval.** Embeddings on `store()`, similarity search in
  `retrieve()`. Gate: "belt tension" ranks the workshop memory first; person filter
  verified.
- **Step 3 — `summarize_transcript()`.** Gate: three fixtures — fact-rich yields sane
  records, small-talk yields `[]`, a correction captures the new value.
- **Step 4 — ROS2 node + `omni_memory_msgs`.** Gate: `scripts/step4_gate.sh` (CLI
  end-to-end).
- **Step 5 — Gemini Live integration** in `behavior_node`. Gate:
  `behavior_node/scripts/step5_service_gate.sh` (auto) + `step5_manual_gate.md` (live).
- **Step 6 — per-person keying via face recognition.** Gate: recognised across a
  restart; `step6_service_gate.sh`.

All six are complete and proven live (2026-07-12 → 07-18).

## Service contracts (`omni_memory_msgs`)

- `/omni_memory/store_transcript` — transcript → summarised records, stored.
  Idempotent per request within a 300s window.
- `/omni_memory/retrieve_memories` — query → plain-text context block for injection.
- `/omni_memory/rekey_person` — re-label every memory from one person id to another.
  This is what carries an `unknown_N`'s history over when they're finally given a name.

## Identity contracts

- `/camera/identity` (String) — the **primary (largest) face** only: a known name, a
  stable `unknown_N`, or `''`. One conversation belongs to one person, so behavior_node
  keys off exactly this.
- `/camera/identities` (String, JSON) — every visible face:
  `{"faces": [{track, identity, raw, primary, cx, cy, bw, bh}, ...]}`, largest first.
  Added 2026-07-18 for group awareness and targeted enrolment.
- `/camera/enroll_request` (String) — a bare name enrols the primary face (legacy
  form); JSON `{"name": ..., "target": "unknown_3"}` or `{"name": ..., "track": N}`
  enrols a **specific** visible face. An off-screen target is refused, never
  substituted with the closest face.
- `/camera/enroll_result` (String, JSON) — echoes `ok` / `detail` / `track`.
- Gemini tool `remember_person(name)` → `behavior_node.learn_person` → enrol + re-key.

## Behaviour rules

- **Retrieve on wake** (seed query, person-filtered); **store at conversation END**,
  so learning someone's name mid-chat needs no database re-key.
- **Memory is a soft dependency.** If `omni_memory` is down, OMNI still converses.
- **Late binding.** If a chat starts with nobody or an unknown in frame and a known
  person appears, adopt them and inject their memories. Safe direction only — never
  name → a different name — and once per chat.
- **Hysteresis** on identity (majority vote over a sliding window) kills per-frame
  flicker. Applied **per tracked face**, so a passer-by can't vote in the identity of
  the person being spoken to.
- **Quality gate.** Only a big, confident, *frontal* face may REGISTER a new identity;
  matching an already-known face is ungated and works at any angle. Frontality =
  nose offset / eye separation. Calibration: usable frontal ≈0.34, true profile ≈0.5+,
  bar = **0.40**. (A first bar of 0.30 rejected good faces — nothing could enrol.)
- **Only the primary face may register** a new `unknown_N`. Secondary faces are
  matched but never minted — without this, marginal detections minted ~10 junk
  unknowns in 4 minutes with one real person present.
- **Multi-crop enrolment.** ~5 good frames over 2s, following the *latched track* from
  the original request so nobody else is learned under that name mid-window.
- **Persistence.** Every crop is saved with a sibling `.npy` embedding and loaded in
  preference to re-detection — the aligned 112×112 crop often can't be re-detected by
  YuNet, which silently lost references (a named person would quietly stop being
  recognised).
- **`consolidate()` at startup** merges unknowns that now match a known person and
  prunes folders yielding no embedding. Only anonymous entries are ever removed.
- Unknown ids are **never reused** after a merge — the number is not a visitor count.

## Constraints that shaped the design

- `MemoryStore` and `Embedder` stay free of ROS imports — the core library must run on
  a desktop with no ROS installed.
- All storage access goes through `MemoryStore`; no raw Supabase calls in app code.
- Embedding dimension **768**, verified. `gemini-embedding-001` at 768 is *not*
  L2-normalised (the 3072 default is), so `GeminiEmbedder` normalises. Task types:
  `RETRIEVAL_DOCUMENT` on store, `RETRIEVAL_QUERY` on retrieve. Summarizer temperature
  0.0.
- The summarizer **normalises person names** (`alice<ts>` → `alice`). Never delete test
  rows by `person` — it misses them and could match a real person. Delete by
  `session_id`.
- Gemini returns transient **503** under load; the summarizer retries 429 **and** 5xx,
  otherwise a whole conversation's memories are silently lost.
- `.env` (`SUPABASE_SERVICE_KEY`) is gitignored and must never be committed.
  `GEMINI_API_KEY` lives in `~/.bashrc`, so keyed code needs an interactive shell.
- omni-core Supabase is a **separate** project from the VPS trading system.

## Known gaps

- **Stranger flow end-to-end in conversation** is still unproven with a real second
  person: unknown wakes OMNI → it asks their name → `remember_person` → enrol →
  recognised next time. Every part is validated in isolation.
- **No dedup.** v1 can store the same fact twice (the live DB has a duplicate row).
- Memories written before 2026-07-16 read "the user" instead of the person's name.
