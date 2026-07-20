# CLAUDE.md — behavior_node

Read this at the start of every session.

## What this is
OMNI's central brain: the state machine, and the wiring between voice, memory,
vision and navigation. Deliberately layered so no file knows more than it must:

| file | owns | must not contain |
|---|---|---|
| `behavior_node.py` | state machine, ROS interface, wiring | any Gemini API code |
| `gemini_bridge.py` | all Gemini **Live** (session, audio duplex, tool loop) | ROS publishing outside `run_in_executor` |
| `function_handlers.py` | Gemini tool implementations | |
| `audio_handler.py` | ALSA mic/speaker, the two audio threads | |
| `memory_client.py` | `omni_memory` services | eager `omni_memory_msgs` import |
| `scene_describer.py` | one-shot Gemini **vision** call | any ROS import |
| `greeting_decider.py` | one-shot Gemini **text** call | any ROS import |

`scene_describer` and `greeting_decider` are ROS-free on purpose: both must
import and run on a desktop with no ROS, so their logic is testable offline
(`test_scene_describer.py`, `test_greeting_decider.py`).

---

## THE DESIGN DECISION: how an unprompted greeting reaches the speaker

This is the question Session 10 existed to answer, so it is written down in
full. When `event_generator` says someone walked in, OMNI must be able to speak
*without anyone having said the wake word*. Three mechanisms were possible.

### Rejected: inject a turn into an idle session
There is no idle session to inject into. `_set_state('IDLE')` closes the Gemini
session **and** hands ALSA device 0 to `WakeWordDetector`. Making this work would
mean holding a Live session open permanently, which:
- fights the one-reader ALSA rule (`audio_handler.py` module docstring) — the
  wake word detector and an open session cannot both own the mic;
- burns Live API quota continuously for a robot that is silent most of the day;
- leans entirely on the reconnect path, and a mid-session 1008 drop leaving the
  robot stuck in SPEAKING is already a known failure mode.

### Rejected: bypass Gemini with local TTS
Fast and free, but it gives up the requirement that *Gemini* choose between a
specific greeting and silence, and a second voice engine would not match the
Live voice (Algieba). Two different voices coming out of the same robot reads as
a malfunction.

### Chosen: decide with a one-shot call, speak by opening a Live session

```
person_appeared
      ↓
_greeting_blocked()      ← code-enforced gates. Nothing reaches the model
      ↓  (permitted)        that is not already permitted.
memory retrieve (RetrieveMemories, scoped to that person)
      ↓
GreetingDecider.decide()  ← ONE-SHOT generateContent, flash-lite, thinking off
      ↓
  ┌───┴────────────────────┐
SILENCE                  a line
  ↓                        ↓
nothing happens        _speak_unprompted(line)
(no session, no mic,       → open_session(initial_prompt=...)
 no wake-word teardown,    → lands in LISTENING; they can just reply
 zero cost)
```

**Why this split.** Deciding "greet or stay quiet, and in what words" is a
one-shot *text* task — no microphone, no duplex, no tool loop. Running it as
`generateContent` (the path `scene_describer` already proves) makes **silence
genuinely free**: no session, no ALSA handoff, no wake-word teardown. That
matters because silence is the *common* outcome. The alternative — open a Live
session and let the prompt decide — pays the full cost on every candidate
greeting and, when the model declines, strands the robot in LISTENING with an
open mic for the full 30s conversation timeout having said nothing at all.

**Why the speaking path is a Live session and not TTS.** Someone greeted by name
usually answers. Opening a Live session with the composed line as
`initial_prompt` leaves them in a conversation they can simply talk to, in the
voice they expect. It is also the exact mechanism `_on_say` and
`_on_safety_fault` already use, so it inherits a path that works.

### Measured end to end, 2026-07-20 — live walk-in, VERIFIED

A genuine 13-minute absence, walking back into frame:

```
person_appeared  ──▶ memory retrieved      1.24s   (5 records)
                 ──▶ greeting decided      0.54s   SPEAK
                 ──▶ session open          0.14s
                 ──▶ FIRST AUDIO           1.07s
                     ────────────────────────────
                     TOTAL                 2.99s   ✅ target ~3s
```

Spoken: *"Good morning, Rafael, I hope you have a Monster ready for the day
ahead."*

`away_duration` arithmetic confirmed exactly: `person_left` → `person_appeared`
measured 716s, plus the 90s grace = **806s reported, 806s expected**. Anchoring
the away clock to the last real *sighting* rather than to the moment we gave up
is what makes that correct — the grace period is part of how long they were
gone, not a free pass.

The same session also produced the short-absence case for free: a 206s absence
went through the full `person_left`/`person_appeared` cycle and the decider
returned **SILENCE** in 0.51s. Under ten minutes, correctly declined, no session
opened, no audio.

Full loop closed afterwards: greeted → conversation → 729-char transcript stored
as 1 memory record attributed to `rafael` (via `_session_person`, set in
`_speak_unprompted`).

### Earlier measurement, 2026-07-19 (sitting still, everything warm)

```
person_appeared  ──▶ memory retrieved      1.83s   (5 records, live VPS call)
                 ──▶ greeting decided      0.59s
                 ──▶ session open          0.14s
                 ──▶ FIRST AUDIO           1.05s
                     ────────────────────────────
                     TOTAL                 3.60s
```

Spoken result: *"Good evening, Rafael, I hope you have a Monster ready for this
evening."* — the personalisation is coming from real stored memories.

**The two-round-trip cost of this design is not what dominates.** The decision
call is 0.59s and the session open is 1.05s; together 1.64s, less than half the
budget. **`RetrieveMemories` at 1.83s is the single largest cost** — it does an
embedding plus a vector search against the VPS on every greeting.

So 3.6s against a ~3s target, and the obvious lever is memory, not the mechanism.
The cheapest fix if it matters: **cache per-person memories with a TTL.** Rafael's
remembered facts do not change minute to minute; the first greeting after boot
pays 1.8s and every later one pays ~0. Do NOT reach for "open the session in
parallel with the decision" — that opens a session before knowing whether the
answer is silence, which throws away the entire reason this design was chosen.

Watch for a false reading here: with `omni_memory` **not running**, this leg
reads 2.01s — that is `memory_service_timeout` expiring, not work being done. If
you measure ~2.0s exactly, check the node is up before concluding anything.

`GreetingDecider` is warmed at startup for the same reason `SceneDescriber` is
(first call in a process ~3.4s vs ~0.7s warm); measured warmup here was 0.85s.

### `_speak_unprompted()` is the one place that owns this
Both `/audio/say` and greetings go through it. The ALSA handoff order is strict
and identical to the safety-fault path — **stop the detector, sleep 100ms, then
start capture**. Without the sleep the capture open races the kernel release and
gets ALSA error -9985. Do not inline this dance a third time.

One trap it works around: **`open_session(memory_context=...)` is silently
ignored whenever `initial_prompt` is set** (see the branch in
`gemini_bridge._run_single_session` — the pending-prompt path never prepends
`_memory_context`). So `_speak_unprompted` folds the memory block into the
prompt text itself. Passing it as the keyword argument looks correct and
delivers nothing.

---

## Greeting suppression — in code, never in the prompt

`_greeting_blocked()` returns a reason string or `None`. Every rule lives there:

- **`NAVIGATING` / `EXPLORING` / `DOCKING` / `ERROR`** — busy or faulted.
- **`LISTENING` / `SPEAKING`** — a wake-word conversation is live. Someone
  walking in must not talk over the person already talking.
- **`bridge.is_session_active()`** — belt and braces for a session open in a
  state the above would wave through.
- **Low power** — `/bms/low_battery`, or `_battery_pct` under
  `greeting_min_battery`.
- **Per-person cooldown** (`greeting_cooldown`, default 600s), keyed on identity.

**None of this is mentioned to the model.** A suppression rule expressed as
prompt text is a suggestion; expressed as a `return`, it is a rule. The model is
only ever asked about greetings that are already permitted — the single thing it
decides is whether a permitted greeting is *warranted*.

Two details that are easy to get wrong:
- The gate is checked **twice** — once on the event, and again after the decision
  call returns. That call takes a second or two, and the user may have said the
  wake word meanwhile. Greeting over the top of a conversation that started while
  we were thinking is precisely the interruption this must never cause.
- The cooldown is recorded **only when OMNI actually speaks**, not on every
  attempt. A SILENCE decision must not burn the cooldown, or a genuine arrival
  ten minutes later gets suppressed by a greeting that never happened.
  API-hammering is bounded anyway: `person_appeared` cannot repeat faster than
  `event_generator`'s 90s `absence_grace`.

### No "docked" signal exists yet
`bms_node` builds `BatteryState` field by field and **never sets
`power_supply_status`** (verified in its source, not assumed). So "is OMNI on the
charger" is not knowable from `/battery/status`. Docked-ness is currently only
representable as the `DOCKING` state, which the state check already covers. Add
a real check to `_greeting_blocked()` when the docking work lands — the comment
there marks the spot.

---

## Parameters (greeting)
| param | default | note |
|---|---|---|
| `greeting_enabled` | `True` | |
| `greeting_events_topic` | `/omni/events` | from `event_generator` |
| `greeting_cooldown` | `600.0` | s, per person, code-enforced |
| `greeting_min_battery` | `20.0` | % |
| `greeting_model` | `gemini-3.1-flash-lite` | one-shot, not the Live model |
| `greeting_prompt_path` | `''` | `''` -> `config/greeting_prompt.txt` |

`config/greeting_prompt.txt` and the parser agree on one literal token:
the model replies **`SILENCE`** to decline, anything else is the spoken line.
A test asserts both halves of that contract — change one and change the other.

Thinking is **off** (`thinking_budget=0`) for the same two reasons as
`scene_describer`: reasoning tokens are drawn from `max_output_tokens`, so a low
cap with thinking on yields a truncated fragment rather than a short answer; and
it costs seconds this path does not have.

---

## Threading — the rules that are load-bearing
- `_set_state()` is the **only** writer of `_current_state`, guarded by
  `_state_lock`. Direct assignment anywhere else is a bug.
- The ROS executor is **single-threaded**. Anything that blocks belongs on a
  daemon thread — `_do_greeting` (memory retrieve + decision call, seconds each)
  runs on one for exactly this reason. A blocking `queue.put` in `stop_capture`
  once froze the whole node with the displays stuck on ERROR.
- `open_session` / `close_session` / `inject_context` marshal onto the asyncio
  loop internally, so they are safe from any thread. Do not call other asyncio
  APIs from ROS or wake-word threads.

## Running
```
cd ~/omni_ws && colcon build --packages-select behavior_node && source install/setup.bash
ros2 launch behavior_node omni_full_launch.py        # full system
python3 -m pytest test_greeting_decider.py test_scene_describer.py -q   # robot off
```

Greetings additionally need the presence pipeline:
```
ros2 run world_state world_state_node                # launch file is broken, see event_generator/CLAUDE.md
ros2 launch event_generator event_generator.launch.py
```
