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
| `suppression.py` | when NOT to speak unprompted | any ROS import |
| `check_in_policy.py` | whether to check in (the manners) | any ROS import |
| `check_in_response.py` | classifying the reply | any ROS import |
| `check_in.py` | the check-in state machine | Gemini API code, Nav2 goal building |

`scene_describer`, `greeting_decider`, `person_nav`, `suppression`,
`check_in_policy` and `check_in_response` are ROS-free on purpose: all must
import and run on a desktop with no ROS, so their logic is testable offline
(`test_scene_describer.py`, `test_greeting_decider.py`, `test_person_nav.py`,
`test_suppression.py`, `test_check_in_policy.py`, `test_check_in_response.py`).

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
`/audio/say`, greetings and check-in openers all go through it. The ALSA handoff
order is strict and identical to the safety-fault path — **stop the detector,
sleep 100ms, then start capture**. Without the sleep the capture open races the
kernel release and gets ALSA error -9985. Do not inline this dance a third time.

**It asks `_wake.is_running()`, NOT `state == 'IDLE'` (fixed in Session 9).**
That inference held for every caller until check-ins existed, and is wrong in
both directions:

- *IDLE but not running* — presence-disarm stops the detector while staying in
  IDLE, so the old check paid a pointless 100 ms sleep.
- *Not IDLE but running* — **nothing stops the detector when navigation starts**
  (`start_navigation` does not touch it, and `_check_presence_timeout` returns
  early unless IDLE). A check-in is the first path in the system that drives
  **from IDLE**, so OMNI arrives beside you in NAVIGATING with the detector still
  holding device 0. The old check skipped the stop, capture raced a device that
  was never released, and the check-in would arrive and say **nothing at all** —
  surfacing as ALSA -9985, which does not mention the wake word anywhere.

The same latent hole is why `_nav_result_callback`'s comment ("in NAVIGATING the
wake word was not running") reads as true: `navigate_to` is only ever reached
from a conversation, where the detector was already stopped. Any new path that
navigates from IDLE must assume the detector is live.

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

### The suppression rules are now SHARED (Session 9)
There are two features that speak without being spoken to — the greeting and the
proactive check-in — and they must agree on what "busy" means. That rule set
lives once, in **`suppression.py`** (`interaction_blocked()` + `RobotStatus`).

`_greeting_blocked()` is now just `'greetings disabled'` plus a call to it;
`CheckInPolicy` calls the same function and adds its own gates. `robot_status()`
on the node is the single builder of the status snapshot.

Do not re-inline these rules into either caller. Two copies would drift, and the
drift is invisible: the failure is not a crash, it is OMNI cheerfully
interrupting a conversation six months from now because only one copy learned
about a new state.

### No "docked" signal exists yet
`bms_node` builds `BatteryState` field by field and **never sets
`power_supply_status`** (verified in its source, not assumed). So "is OMNI on the
charger" is not knowable from `/battery/status`. Docked-ness is currently only
representable as the `DOCKING` state, which the state check already covers.

Session 9 wired the *plumbing* for it: `RobotStatus.docked` exists, is checked by
`interaction_blocked()`, and is fed from `self._docked` in `robot_status()` —
which is hardcoded `False`. When the docking work lands, set `self._docked` in
one place and **both** greetings and check-ins inherit the suppression at once.
That is the whole point of the shared helper.

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

## Person zones + go_to_person (Session 7)

Two navigation surfaces now share one name space and one goal builder:

- **`navigate_to(location)`** — places. Resolves via `node.resolve_location()`,
  which checks **saved point locations** (from `save_location`, in
  `omni_config.yaml`) first, then **zone anchors** (from `omni_zones`). One
  unified `known_place_names()` list drives the "unknown location" reply.
- **`go_to_person(name)`** — people. "Come here" / "come to me" call it with an
  empty name → it resolves to `_session_person` (who OMNI is talking with).

Both dispatch through **`node.start_navigation(x, y, yaw_deg)`** — the single
Nav2 goal builder. `_navigate_to` and `go_to_person` no longer build goals
themselves; don't reintroduce a second copy.

**The decision is ROS-free and tested.** `person_nav.plan_go_to_person()` takes
the cached world_state snapshot and returns an outcome
(`no_name`/`no_world_state`/`unknown`/`unplaced`/`stale`/`go`). The handler owns
only the standoff geometry, the Nav2 goal, and the in-character line — same split
as `scene_describer`/`greeting_decider`. `test_person_nav.py` covers every
outcome with the robot off.

**Honesty is the point, not a fallback.** Location from world_state is
*room-level* (see `world_state/CLAUDE.md` and `omni_zones/geometry.py`:
assumed distance, ignored head pan). So `go_to_person` drives only on a
confident, recent fix; unknown / never-localised / older than
`person_location_stale_after` (120 s) all speak the situation instead. The
success criterion is arriving in the right **room** — Session 6's head tracking
does the looking-at once OMNI is there.

**Goal selection:** a fresh `map_xy` → `standoff_pose()` stops
`person_standoff_distance` (1.0 m) short of the estimate, facing them, computed
from the live robot pose (TF `map`→`base_link`). Zone-only (no `map_xy`) → the
zone's anchor. No robot pose → aim at the estimate itself (still the right room).

**Zones are the shared `omni_zones` config** — the SAME file world_state loads,
so there is one source of truth for where the rooms are. Empty by default (the
space isn't traced into polygons yet): `navigate_to` then falls back to point
locations, and `go_to_person` honestly reports it cannot place anyone. Fill in
`omni_zones/config/zones.yaml` and both consumers pick it up.

`/omni/world_state` is a **soft dependency**, exactly like memory: if the node
isn't publishing, `latest_world_state()` is `None` and `go_to_person` says its
world-tracking isn't reporting rather than crashing.

### Parameters (Session 7)
| param | default | note |
|---|---|---|
| `zones_config_path` | `''` | `''` → `omni_zones`'s shipped `config/zones.yaml` |
| `world_state_topic` | `/omni/world_state` | cached for `go_to_person` |
| `person_location_stale_after` | `120.0` | s; older last-sighting → honest "not sure" |
| `person_standoff_distance` | `1.0` | m; how far short of a person to stop |

## Proactive check-in (Session 9)

"Rafael has been at the workbench for an hour" → drive over, stand **beside**
him, ask **one** question, and leave again if he does not want company. The first
behaviour where OMNI initiates a social interaction with a purpose.

```
person_dwelling (event_generator)
      ↓
CheckInPolicy.decide()     ← code-enforced manners. Dwell ≥60 min, not busy,
      ↓  (approved)           battery, quiet hours, cooldowns. Logged either way.
  SNAPSHOT   remember the current pose (TF map→base_link)
      ↓
  APPROACH   Nav2 goal: side-offset standoff beside them
      ↓
    ASK      ONE soft opener, via _speak_unprompted (the Session 2 path)
      ↓
  LISTEN     engaged? "no"? "not now"? 15 s of silence?
      ↓
  RETURN     Nav2 goal back to the remembered pose
      ↓
  outcome recorded → cooldowns applied, logged to memory
```

### The layering, which is the same one as everywhere else
| what | where | ROS? |
|---|---|---|
| *should* we check in | `check_in_policy.py` | no — 96 tests |
| what did they *say* | `check_in_response.py` | no — 47 tests |
| when NOT to speak at all | `suppression.py` | no — 18 tests |
| driving, speaking, timing out | `check_in.py` | yes — 36 tests on a fake node |

**Gemini triggers nothing here.** The policy decides whether to go, `check_in.py`
decides where to stand and when to leave, and the model is handed exactly one
job: say the opener and then converse. Everything with a consequence is an `if`.

### The seams in behavior_node — all four are load-bearing
- **`_on_presence_event`** routes `person_dwelling` → `CheckInBehavior`. The
  greeting subscription is reused; a second one is only created if greetings are
  off, so a dwell event is never delivered twice.
- **`_nav_result_callback`** offers the result to the mission **first**. The
  generic "announce your arrival in character" prompt is wrong for both legs of a
  check-in — arrival is followed by a specific opener, and the return by silence.
- **`_set_state('IDLE')`** calls `on_conversation_end()` **before**
  `_flush_conversation_to_memory()`. Order is load-bearing: `pop_transcript()`
  empties the buffer that the outcome classifier reads. It also records any
  completed conversation as an interaction, so a chat suppresses a later check-in.
- **`_tick_check_in`** (1 Hz timer) drives the silence timeout, the
  person-left-the-zone abort, and the hard duration ceiling.

Wake word (`_on_wake_word`) and safety faults (`_on_safety_fault`) both abort.
The fault path aborts *before* cancelling the Nav2 goal, so the mission cannot
see the cancelled result and helpfully dispatch a drive home into a live fault.

`abort()` **cancels the in-flight goal** via `node.cancel_navigation()`. Without
that, a wake word mid-APPROACH left OMNI still motoring toward someone who was
already talking to it, and the goal's eventual SUCCEEDED result fell through to
the generic arrival handler — which announces the arrival over the top of the
conversation.

### Decisions worth not re-litigating
- **`abort()` does not drive home.** A wake word means they are talking to OMNI
  now; driving off mid-sentence to restore a pose is worse than staying put. It
  *does* record `yes`, because being talked to is engagement.
- **Silence closes the session immediately** rather than waiting out the 30 s
  conversation timeout. Standing there with an open mic having been ignored is
  exactly the loitering the feature must not do. It never re-asks.
- **An aborted APPROACH records no outcome.** Nothing was asked, so nothing was
  answered — burning a 4-hour zone cooldown on a trip that never arrived would be
  the robot punishing itself.
- **No robot pose → no mission.** Not a degradation: without a pose there is no
  way home, and the brief's "never end stranded" outranks the check-in.
- **`_person_still_in_zone()` fails safe.** `world_state` is a soft dependency,
  so a missing snapshot never aborts a drive; only an explicit *different* zone
  counts as "they got up".

### Side-offset approach
`omni_zones.standoff_pose()` gained a `lateral` param (positive = left of the
approach line). `lateral=0` is bit-identical to Session 7, so `go_to_person` is
untouched. Yaw is recomputed **at the goal**, so OMNI stands to one side and
still looks across at them rather than staring past them.

Which side is a **fixed per-zone convention** in `zones.yaml`
(`check_in_side: left|right`), falling back to `check_in_default_side`. It is
config and not perception because nothing estimates which way a person is
facing — `world_state` has a face box, not a pose model. A bench does not move
and neither does how you sit at it, so a constant is the honest v1.

### What is NOT done
- **Final-approach speed is not capped.** The brief asks for it; there is no
  mechanism without configuring Nav2's Speed Filter (or a controller param
  client), and faking it would be worse than saying so. The approach runs at
  normal Nav2 speed.
- **`_dock_fallback()` cannot dock.** The `DOCKING` state is still a placeholder
  ("no docking logic implemented yet") and the AprilTag dock pipeline is not
  wired to a behaviour. A failed return therefore logs loudly and stands down
  where it is. `check_in.py::_dock_fallback` is the exact spot to wire it, and
  "if `return_pose` was the dock, re-dock properly" belongs there too.
- **The reply classifier is keywords, not a model.** `check_in_response.py`
  documents why (bounded, self-correcting failure) and where it is fragile.
  `NOT_NOW` is matched **before** `NO` so "no, not right now" is a deferral, and
  the default is `YES` so one mumbled sentence cannot cause a 4-hour lockout.
- **Cooldowns do not survive a restart.** They live in the `CheckInPolicy`
  object. Restart the node and a "no" from ten minutes ago is forgotten. The
  memory log written on every outcome is the raw material for fixing this.
- **Never run on hardware.** All of the above is offline logic and stubs.

### Parameters (Session 9)
| param | default | note |
|---|---|---|
| `check_in_enabled` | `True` | |
| `check_in_min_dwell` | `3600.0` | s. Lowering this is how the feature gets hated |
| `check_in_min_battery` | `40.0` | % — deliberately well above the greeting's 20% |
| `check_in_quiet_start` / `_end` | `21:00` / `08:00` | wraps midnight |
| `check_in_global_cooldown` | `7200.0` | s after ANY interaction with that person |
| `check_in_no_cooldown` | `14400.0` | s, per zone, after a "no" |
| `check_in_not_now_cooldown` | `3600.0` | s, per zone, after a "not now"/silence |
| `check_in_zones` | `[]` | empty = trust `event_generator`'s `dwell_zones` |
| `check_in_bias_enabled` | `True` | stretch the threshold where declines dominate |
| `check_in_bias_min_samples` | `3` | outcomes needed before the bias moves |
| `check_in_bias_max_multiplier` | `2.0` | ceiling on that stretch |
| `check_in_standoff_distance` | `1.0` | m short of the person |
| `check_in_lateral_offset` | `0.6` | m to the side |
| `check_in_default_side` | `left` | when `zones.yaml` does not say |
| `check_in_silence_timeout` | `15.0` | s before silence reads as "not now" |
| `check_in_max_duration` | `300.0` | s hard ceiling on the whole mission |

### Testing it live — the recipe
Two configs ship EMPTY and both are required, or nothing can ever fire:
1. `omni_zones/config/zones.yaml` is still `zones: {}`. Trace at least one zone,
   or `world_state` never labels anyone and no dwell can start.
2. `dwell_zones` on `event_generator`.

The production defaults (60 min dwell, 2 h cooldown, quiet after 21:00) are
unusable for a test you want to finish today, so every one of them is a launch
argument — no rebuild needed:

```bash
# Jetson first (vision must be up, or nobody is ever seen)
ssh Omni 'ros2 launch omni_jetson_bringup head_detector.launch.py'

# Pi
ros2 launch behavior_node omni_full_launch.py \
    check_in_min_dwell:=60.0 \
    check_in_global_cooldown:=120.0 \
    check_in_no_cooldown:=300.0 \
    check_in_not_now_cooldown:=180.0 \
    check_in_quiet_start:=00:00 check_in_quiet_end:=00:00 \
    check_in_min_battery:=10.0

ros2 run world_state world_state_node        # launch file is broken
ros2 launch event_generator event_generator.launch.py \
    dwell_zones:=workbench dwell_threshold:=60.0 dwell_refire_interval:=120.0

ros2 topic echo /omni/events                 # watch dwell + check_in phases
```

`check_in_quiet_start == check_in_quiet_end` disables quiet hours — without that,
an evening test is silently suppressed and looks like the feature is broken.

**Smoke-test the drive without waiting for a real dwell** by injecting the event
(world_state must still be running and tracking you, or it cannot place you):
```bash
ros2 topic pub --once /omni/events std_msgs/msg/String \
  '{data: "{\"kind\":\"person_dwelling\",\"identity\":\"rafael\",\"zone\":\"workbench\",\"dwell_duration\":4000}"}'
```
**It will drive.** Clear floor, hand on the e-stop.

### Parameter gotcha: an empty list default cannot be typed in rclpy
`check_in_zones` and `dwell_zones` are **comma-separated strings**, not string
arrays, and must stay that way. `declare_parameter(name, [])` overwrites the
descriptor type from the value, an empty list infers as `BYTE_ARRAY`, and the
parameter then reads back uninitialized *and* rejects any override with
"expecting type BYTE_ARRAY". Passing an explicit `ParameterDescriptor` does not
help — it is overwritten. Both were caught by actually launching the node, not
by the unit tests, which never touch rclpy parameters. This is the same
empty-sequence trap that broke `world_state`'s launch file.

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
- `CheckInBehavior._start()` runs on a daemon thread for the same reason
  `_do_greeting` does — `nav_is_ready()` blocks for up to a second. Its `_lock`
  guards only the mission fields and is **never held across a call into the
  node**: `_set_state()` and `start_navigation()` take their own locks, and
  holding both invites a deadlock.

## Running
```
cd ~/omni_ws && colcon build --packages-select behavior_node && source install/setup.bash
ros2 launch behavior_node omni_full_launch.py        # full system
python3 -m pytest -q                                 # robot off, 270 tests
```

Note: `test_greeting_decider.py::test_missing_api_key_is_a_clear_error` fails in
any shell that sourced `~/.bashrc`, because `GreetingDecider` falls back to
`os.environ['GEMINI_API_KEY']` and the test cannot then produce a missing key.
It passes in a non-interactive shell. Pre-existing, not a Session 9 regression.

Greetings additionally need the presence pipeline:
```
ros2 run world_state world_state_node                # launch file is broken, see event_generator/CLAUDE.md
ros2 launch event_generator event_generator.launch.py
```
