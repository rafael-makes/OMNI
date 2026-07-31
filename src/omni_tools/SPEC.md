# SPEC.md — `omni_tools`

OMNI's tool-use agent: the single router between reasoning (Gemini Live) and
capabilities (ROS 2 actions/services/topics, Home Assistant, memory).

**Status: proposed. Nothing here is built yet.** Review gate before implementation.

---

## 0. Findings from reading the repo — read this first

Four things in the brief do not match what is actually in the tree. Each one
changes the design, so none of them is worked around silently.

### 0.1 The Live model does not support asynchronous function calling

The brief assumes Gemini 2.5 Flash Native Audio. `behavior_node.py:115` actually
runs:

```python
self.declare_parameter('gemini_model', 'models/gemini-3.1-flash-live-preview')
```

Google's model page for `gemini-3.1-flash-live-preview` states plainly that
function calling is **synchronous only** — *"the model will not start responding
until you've sent the tool response"* — and that async function calling is not
supported. The `google-genai` SDK on the Pi (2.0.1) *does* carry the types
(`Behavior.NON_BLOCKING`, `FunctionResponse.will_continue`,
`FunctionResponseScheduling.{SILENT,WHEN_IDLE,INTERRUPT}`), verified by
introspection, so this is a **model** limitation, not an SDK one. Declaring
`behavior=NON_BLOCKING` against 3.1 Flash Live would look correct and change
nothing — exactly the class of bug this repo has been bitten by before (see
`feedback_two_sided_protocol`).

**Consequence, and it is the central design constraint of this spec:** every
function call blocks the model's turn until its `FunctionResponse` arrives. There
is no protocol-level "I'll get back to you". So the acknowledge-now /
report-later pattern must be built *in OMNI*, out of two mechanisms that already
exist and are proven on hardware (§4).

Nothing about the contract in §1 changes if OMNI later moves to a model with
NON_BLOCKING. `ToolSpec.report` deliberately reuses Gemini's own vocabulary
(`interrupt` / `when_idle` / `silent`) so that migration swaps the *transport*
and leaves every tool declaration untouched.

### 0.2 There is no Home Assistant integration in the repo

`grep -ril 'home.assistant|homeassistant|hass'` across `src/` and `config/` hits
exactly one file — `navigation2/nav2_docking/README.md`, unrelated. The brief
offers "a Home Assistant action (real, if the integration exists in-repo)". It
does not. So step 4 of the build ships `ha_call_service` as a **thin REST tool
against the HA `/api/services/<domain>/<service>` endpoint**, config-driven
(`ha_base_url`, `ha_token` from env), with an offline fake for tests. It needs
one thing from you: **the HA base URL and a long-lived access token**, or a
decision to defer it and prove the loop with the memory tool instead.

### 0.3 Memory: `MemoryStore` is the wrong layer for this process

The brief says to read/write "through the existing `MemoryStore` client". In
`behavior_node`'s process that is the wrong object:

| | `omni_memory.store.MemoryStore` | `behavior_node.memory_client.MemoryClient` |
|---|---|---|
| where it runs | inside the `omni_memory` **node** | inside `behavior_node` |
| needs | `SUPABASE_URL` + service key + `Embedder` in-process | nothing |
| talks to | Supabase over WireGuard, blocking | 3 ROS services, own node + executor |
| on failure | raises | degrades to `''` / no-op |

`behavior_node` has never imported `MemoryStore` and must not start: it would
duplicate the Supabase client, put a blocking VPS round-trip on OMNI's own
threads, and break the soft-dependency promise that `memory_client.py`'s
docstring is entirely about. **`omni_tools` goes through `MemoryClient`**, which
*is* "the existing client" from this process's point of view, and which reaches
`MemoryStore` one hop away in the `omni_memory` node. §5 covers a gap this
exposes.

### 0.4 `audio_node` is not the Gemini Live client

The brief describes "Pi 5 runs a WebSocket client streaming 16 kHz PCM". Two
implementations of that exist. `audio_node/audio_node.py` is the older one and is
**deliberately not launched** — `omni_full_launch.py` says so in its docstring:

> `audio_node` is intentionally excluded — `behavior_node` owns the Gemini Live
> session and the USB audio device. Running both creates duplicate WebSocket
> connections and two processes fighting over the same ALSA device.

The live client is `behavior_node/gemini_bridge.py`. `omni_tools` integrates with
that one and ignores `audio_node` entirely.

---

## 1. The tool registration contract

### 1.1 `ToolSpec`

One capability = one `ToolSpec`. Frozen dataclass, no ROS and no `google.genai`
import anywhere in the module that defines it.

```python
@dataclass(frozen=True)
class ToolSpec:
    name: str                      # snake_case, unique, stable — the model sees this
    description: str               # natural language, written FOR the model
    parameters: dict               # JSON Schema, object type. {} = no args
    handler: Handler               # (args: dict, ctx: ToolContext) -> ToolResult | str

    kind: Kind = Kind.FAST         # FAST | LONG                       (§4)
    timeout: float = 8.0           # seconds; FAST budget or LONG watchdog
    ack: str | None = None         # LONG only: spoken immediately     (§4)
    report: Report = Report.WHEN_IDLE   # LONG only: how completion surfaces
    cancel: Cancel | None = None   # LONG only: (job, ctx) -> None. How to stop it

    exposed_to_model: bool = True  # False = agent-initiated only
    retry: Retry = Retry.NEVER     # NEVER | ONCE_IF_UNAVAILABLE       (§6)
    memory_query: str | None = None     # pre-hook retrieval template  (§5)
    memory_note: bool = False           # write the outcome to memory  (§5)
```

`parameters` is **plain JSON Schema as a dict**, never a `genai_types.Schema`.
That is the decoupling the brief asks for, and it buys three things: the registry
imports and unit-tests with no `google.genai` installed; a second consumer (a
future text-mode agent, an HTTP debug endpoint) gets the same declarations for
free; and the conversion lives in exactly one file (§2.1).

Only a documented subset is supported — `type` ∈ object/string/number/integer/
boolean/array, plus `properties`, `required`, `enum`, `items`, `description`,
`nullable`. Anything else raises **at registration time**, not at session-open
time. A malformed schema must fail on the bench, not thirty seconds into a
conversation.

### 1.2 One interface over four kinds of plumbing

The handler signature is the whole abstraction: `(args, ctx) -> ToolResult | str`.
Whether it wraps a Nav2 action, a ROS service, a topic publish, or arithmetic is
invisible to the registry and to Gemini. `adapters.py` provides the three ROS
shapes as small builders so a tool that is *only* a service call is a few lines:

```python
registry.register(ToolSpec(
    name='clear_fault',
    description='Clear an active safety fault...',
    parameters={'type': 'object',
                'properties': {'fault_type': {'type': 'string',
                                              'enum': ['stall', 'estop', 'all']}},
                'required': ['fault_type']},
    handler=publish_topic('/safety/clear_fault', StringMsg,
                          build=lambda a: StringMsg(data=a['fault_type']),
                          speech=_clear_fault_line),
))
```

Adapters are convenience, not a requirement. A handler may be any callable, and
the ten existing handlers in `function_handlers.py` port across essentially
verbatim (§8).

### 1.3 `ToolContext` — the injected seam

Handlers never import `behavior_node`. They receive a context object holding
exactly what a tool is allowed to touch:

```python
class ToolContext:
    node          # the ROS node (publishers, action clients, TF, logger)
    memory        # MemoryClient
    speak         # Reporter.speak(text) — the §4.3 completion channel
    person        # current interlocutor id, or None
    jobs          # JobManager (for LONG tools that need their job handle)
    caller        # Caller.MODEL | Caller.AGENT
```

`caller` is deliberately visible to the handler. A tool sometimes wants to behave
differently when nobody asked — a proactive `map_current_floor` should not narrate
its ack to an empty room.

### 1.4 `ToolResult`

```python
@dataclass(frozen=True)
class ToolResult:
    ok: bool
    speech: str              # ALWAYS safe to say aloud, success or failure
    data: dict = {}          # structured payload, not spoken; for agent callers
    error: ErrorKind | None = None    # §6
    job_id: str | None = None         # LONG only
```

`speech` is mandatory even on failure, and is what goes back to Gemini as the
function response. This preserves the single most important property of the
current handlers: **a tool failure produces a sentence OMNI can say, never an
exception and never silence.** `function_handlers.py:782` already makes this
argument for `describe_scene` ("a tool exception would leave a dead silence
mid-conversation, the worst outcome here"); the registry makes it structural.

Returning a bare `str` from a handler is sugar for `ToolResult(ok=True,
speech=...)`, so the existing handlers need no rewrite.

---

## 2. Gemini Live integration

### 2.1 Registered tools → function declarations

`registry.gemini_tools()` returns `[genai_types.Tool(function_declarations=[...])]`
— a single `Tool` holding every declaration, matching what `OMNI_TOOLS` is today.
`schema.py` is **the only module in the package that imports `google.genai`**, and
it imports it lazily inside the function (the `feedback_lazy_import_soft_deps`
lesson: a module-level import of a soft dependency has already killed this robot
once).

`gemini_bridge._run_single_session` changes by one line:

```python
tools=OMNI_TOOLS                    # before
tools=self._registry.gemini_tools() # after
```

Tools with `exposed_to_model=False` are filtered out here. Declarations are built
**per session open**, not cached at import, so a tool registered at runtime (or a
description tweaked by parameter) takes effect on the next wake word.

`behavior=NON_BLOCKING` is **not** set — see §0.1. When the Live model gains
support, this is the one place that changes.

### 2.2 Function call → dispatch → response

The existing loop in `gemini_bridge._recv_loop` (lines 604–639) is structurally
correct and is kept. Only the dispatch target changes:

```
response.tool_call
  └─ for each fc in response.tool_call.function_calls:
       result = await loop.run_in_executor(None, registry.invoke, fc.name, dict(fc.args), Caller.MODEL)
       fn_responses.append(FunctionResponse(id=fc.id, name=fc.name,
                                            response={'result': result.speech}))
  └─ await session.send(input=LiveClientToolResponse(function_responses=fn_responses))
```

Three properties of that loop are load-bearing and must survive the refactor:

- **One `LiveClientToolResponse` per `tool_call` event, always.** The comment at
  `gemini_bridge.py:600` records what happens otherwise: error 1008, because
  Gemini has already advanced past tool state. Parallel calls in one event are
  batched, never sent individually.
- **Dispatch runs in `run_in_executor`**, never on the asyncio loop. Handlers call
  rclpy and may block for seconds; the audio send/recv coroutines share that loop.
- **Every call gets a response.** `registry.invoke()` catches everything, including
  an unknown tool name and a handler that raises, and returns a speakable
  `ToolResult`. There is no path where `fn_responses` is short an entry.

### 2.3 The response budget

Because the model is blocked (§0.1), the wall-clock cost of a tool call is dead
air in the conversation. The registry enforces this rather than trusting it:

| | budget | on breach |
|---|---|---|
| `Kind.FAST` | `timeout`, default **8 s** | return the timeout line, let the job finish orphaned, log loudly |
| `Kind.LONG` | ack returns in **< 100 ms** | — the handler never runs inline |

Measured reference points for calibrating a FAST budget: `describe_scene` is
~2.5 s (frame fetch + vision call, `function_handlers.py:723`), `RetrieveMemories`
is ~1.8 s live (`behavior_node/CLAUDE.md`). Anything that cannot promise a
ceiling is `Kind.LONG`, not a FAST tool with a generous timeout.

---

## 3. Where it runs, and how it comes up

**In `behavior_node`'s process, on the Pi 5. It is a library, not a node.**

This is the load-bearing deployment decision, so the reasoning is recorded:
everything a tool needs to touch — the Live session, `_set_state`, the single
Nav2 goal builder `start_navigation`, `MemoryClient`, the TF buffer, the
wake-word/ALSA handoff — lives inside `behavior_node`. A separate tool node would
put a ROS service hop in front of every one of them, adding a round trip to the
critical path the brief explicitly protects ("no awkward pauses in the voice
path"), and would need a second copy of the state machine to know whether it is
safe to act. `check_in.py` already established the pattern for a self-contained
behaviour that reaches the node through a narrow seam; `omni_tools` follows it.

- **Package:** `omni_ws/src/omni_tools/` — a colcon *library* package, per
  ARCHITECTURE.md rule 1 (Pi-side, flat under `src/`). No node, no entry point.
- **Bringup:** nothing new. `behavior_node` imports it; `ros2 launch behavior_node
  omni_full_launch.py` is unchanged.
- **Dependency:** `<exec_depend>omni_tools</exec_depend>` in
  `behavior_node/package.xml`.
- **Jetson:** nothing. A tool that needs vision calls the Jetson's existing
  service (`/vision/get_camera_frame`) exactly as `describe_scene` does today.
- **Build:** `colcon build --packages-select omni_tools behavior_node`, then
  re-source. (`feedback_live_param_tuning`: launch files are always copies —
  rebuild before believing a change did nothing.)

---

## 4. The async / long-running problem

The constraint from §0.1 restated: **the model is blocked until the function
response arrives.** So `map_current_floor` cannot hold the call open for minutes.
It must return immediately and report later out of band.

### 4.1 The pattern

```
Gemini calls map_current_floor
        │
        ├─ registry sees Kind.LONG
        │     ├─ validate args                        ─┐
        │     ├─ jobs.start(spec, args, ctx)  ────────  │  < 100 ms, always
        │     └─ return ToolResult(ok=True,             │
        │             speech=spec.ack, job_id=...)     ─┘
        │
        └─ FunctionResponse sent → Gemini says "okay, mapping this floor"
                                   and the conversation continues normally
                                   ─────────────────────────────────────
   meanwhile, on a daemon thread:
        handler runs for minutes
        │
        ├─ succeeds  → Reporter.report(job, ToolResult.ok(speech=...))
        ├─ fails     → Reporter.report(job, ToolResult.fail(speech=...))
        └─ watchdog  → cancel() called, then reported as a timeout   (§6.3)
```

`ack` is a plain string on the `ToolSpec`, not model-generated, because it must be
instant. Gemini renders it in OMNI's voice the same way it renders every other
handler return today.

### 4.2 `JobManager`

Owns one daemon thread per running job (jobs are rare and long; a pool would only
add a queueing failure mode). Per job it holds: id, tool name, args, caller,
`started_at`, state ∈ `running | done | failed | timed_out | cancelled`, result.

- **Watchdog.** A single 1 Hz sweep thread. Past `spec.timeout`, the job is marked
  `timed_out`, `spec.cancel(job, ctx)` is invoked, and the timeout is reported.
  The handler thread is *not* killed — Python cannot safely — so `cancel` is what
  actually stops the underlying work (cancel the Nav2 goal, deactivate
  slam_toolbox). A LONG tool without a `cancel` is a wedge waiting to happen, so
  registration warns when `Kind.LONG` has `cancel=None`.
- **One at a time per tool name.** A second `map_current_floor` while one is
  running returns "I am already mapping" rather than starting a second. Distinct
  tools may run concurrently.
- **Jobs do not survive a restart.** Same honest limitation as check-in cooldowns.
  A `job_started` / `job_finished` memory note (§5) is the raw material for fixing
  that later.

### 4.3 How completion reaches the voice path

`Reporter` is the *only* thing that speaks on a job's behalf, and it is
`_on_say`'s existing two-branch logic factored out
(`behavior_node.py:1347–1381` — session live? inject; else open one):

| `report` | session **open** | session **closed** (IDLE) |
|---|---|---|
| `INTERRUPT` | `bridge.inject_context(text, alert=False)` now | `_speak_unprompted(text)` now |
| `WHEN_IDLE` *(default)* | queue; deliver on the next `SPEAKING → LISTENING` | `_speak_unprompted(text)` now |
| `SILENT` | queue as context for the next session open; never spoken | log + memory only |

`WHEN_IDLE`'s "next LISTENING" hook is the existing
`_await_playback_and_set_listening` transition — the point where playback has
drained and OMNI is not mid-sentence. That is what stops a mapping-complete
announcement landing on top of an answer to an unrelated question.

Both branches are mechanisms this repo has already measured on hardware:
`inject_context` is the safety-fault path, `_speak_unprompted` is the greeting
path (2.99 s to first audio, live-verified 2026-07-20). Neither is new risk.

**Suppression applies.** Before speaking, `Reporter` consults
`suppression.interaction_blocked(node.robot_status())` — the shared rule set that
greetings and check-ins already share. A mapping run that finishes while OMNI is
driving, faulted, or mid-conversation must not barge in; a blocked `INTERRUPT`
degrades to `WHEN_IDLE`. Re-inlining these rules is explicitly warned against in
`behavior_node/CLAUDE.md`, and this is the third consumer.

### 4.4 Rejected: keep the call open and stream progress

Considered, because the SDK exposes `FunctionResponse.will_continue`. Rejected:
the field is meaningless without NON_BLOCKING support on the model (§0.1), so it
would silently do nothing — and even with model support, a minutes-long open tool
call makes the whole conversation hostage to a mapping run. The out-of-band report
is the right shape regardless of what the model supports.

---

## 5. Memory in the loop

All access via `MemoryClient` (§0.3). Memory stays a **soft dependency**: every
path below degrades to a no-op, never an error.

### 5.1 Read — before the handler

`ToolSpec.memory_query` is a format template over the tool's own args:

```python
memory_query='{location} preferences and past visits'
```

When set, the registry calls `ctx.memory.retrieve_context(query, k=3,
person=ctx.person)` before the handler and passes the block in
`ctx.recalled`. Off by default, and deliberately: retrieval is **~1.8 s live**
(`behavior_node/CLAUDE.md`, the single largest cost in the greeting path), which
is most of a FAST tool's budget. Only tools that genuinely need it should opt in,
and any tool that does should be `Kind.LONG` or have a widened `timeout`.

Ordinary conversational recall is unaffected — it already happens once at session
open, via the existing wake-word path.

### 5.2 Write — after the handler

`memory_note=True` writes the outcome as an observation. **This is the one gap in
the existing plumbing.** `StoreTranscript` takes a raw *transcript* and runs it
through `GeminiSummarizer` (`omni_memory/node.py:150`) — sending it a one-line
tool outcome means paying an LLM call to summarise a sentence, and the summariser
is entitled to discard it as small talk. There is no direct "store this record"
service; `MemoryStore.store()` is a `MemoryRecord` write available only inside the
`omni_memory` node.

Two options, and I need your call (§9):

- **(a) Add `StoreFact.srv`** to `omni_memory_msgs` — `content`, `person`,
  `source`, `importance` → straight to `MemoryStore.store()`, no summariser. Small
  (one srv, one handler, one client method), correct, and useful well beyond this
  module. Requires a rebuild on **both** machines, message package rule.
- **(b) Defer.** `memory_note` logs only until (a) exists. The tool contract is
  written now and the write turns on later with no tool changes.

Recommendation: **(b) for this build, (a) as its own small follow-up.** It keeps
the tool agent from dragging a wire-format change through its first landing.

### 5.3 The memory tool itself

`remember_this` / `recall` are ordinary registered tools wrapping the same
client — they get no privileged path. `recall` is FAST with `timeout=4.0` (1.8 s
typical + headroom). `remember_this` is FAST and fire-and-forget:
`store_transcript` is already non-blocking by design.

---

## 6. Error and recovery contract

### 6.1 What a tool returns on failure

`ToolResult(ok=False, speech=<in-character sentence>, error=ErrorKind.X)`.

| `ErrorKind` | meaning | retryable |
|---|---|---|
| `INVALID_ARGS` | schema-valid but nonsense (unknown location) | no — the model must ask |
| `UNAVAILABLE` | dependency down (Nav2, omni_memory, Jetson, HA) | once, if idempotent |
| `DENIED` | refused by a code-enforced rule (faulted, low battery) | no |
| `FAILED` | it ran and did not work | no |
| `TIMEOUT` | budget or watchdog expired | no |

Handlers never raise. If one does, `registry.invoke` catches it, logs with a
traceback, and returns `FAILED` with a generic in-character line. **An exception
must never reach the Gemini loop** — it would either kill `_recv_loop` or leave
the turn unanswered, which with a blocking model is a hung conversation.

### 6.2 Retry vs. surface

The registry does **not** retry by default. `Retry.ONCE_IF_UNAVAILABLE` retries
exactly once, only on `UNAVAILABLE`, only for tools declared idempotent, and only
within the remaining budget. Everything else surfaces immediately, because the
model is a better retry policy than a loop is: it can ask a clarifying question,
try a different tool, or tell the user the truth.

Two things must never auto-retry: **anything that moves the robot**, and anything
with an external side effect (an HA service call). A duplicated
`navigate_to` is a second drive.

The one blessed retry is the ROS 2 cold-start case — `feedback_ros2_service_cold_start`
and `memory_client.py:99` both document that a fresh `rmw_fastrtps` client drops
its first reply. That is exactly `UNAVAILABLE`-once-then-succeeds, and adapters
built by `call_service()` get it by default.

### 6.3 Wedged or timed-out long-running tools

The unattended-autonomous-mapping case, which is where this has to be right:

1. Watchdog trips at `spec.timeout` (mapping: 30 min, generous and finite).
2. `spec.cancel(job, ctx)` runs — the *only* thing that actually stops the work.
   For mapping: cancel the Nav2 goal, stop the exploration driver, leave
   slam_toolbox alone so the partial map survives.
3. Job → `timed_out`. **Always** logged at ERROR and, once §5.2(a) lands, written
   to memory — the report may be spoken to nobody, and a silent unattended failure
   that leaves no trace is the worst outcome here.
4. Reported per `spec.report`. If suppression blocks it, it is delivered on the
   next session open instead of dropped.
5. Robot state is left safe: the registry never leaves a state it set. Any tool
   that sets `EXPLORING` restores `IDLE` on every exit path, including timeout.
   (`_check_state_watchdog` is the backstop, but relying on it means 5+ s of a
   robot in a state that is over.)

`report_status` gains running-job awareness so "how are you doing?" answers
truthfully mid-mapping. Uses the registry as its source, not a second bookkeeping
copy.

---

## 7. Agent-initiated invocation

The same registry, the same handlers, no Gemini in the path:

```python
result = registry.invoke('map_current_floor', {'floor': 'basement'},
                         caller=Caller.AGENT)
```

- **Identical dispatch.** Validation, memory hooks, error contract, job
  management are all shared. A tool cannot behave differently by accident; if it
  wants to, it reads `ctx.caller`.
- **Silent by default.** A model call's `speech` is consumed by Gemini as the
  function response. An agent call returns `speech` to the *caller*, which decides
  whether to pass it to `Reporter`. Proactive invocation must not narrate itself
  unless the behaviour asks it to.
- **`exposed_to_model=False`** registers a tool the agent can call and the model
  cannot — the right home for anything with a consequence that should never be
  reachable by conversation.
- **Callable from any thread.** ROS executor, check-in daemon thread, a timer.
  `invoke` takes no ROS locks and does not spin.
- **Long tools work the same way**: immediate `deferred` result carrying `job_id`,
  completion through `Reporter` under the same suppression rules.

Step 5 of the build proves this with `CheckInBehavior` — which today calls
`node.start_navigation()` and `node._speak_unprompted()` directly — invoking one
registered tool through this entry point. That is the exact seam proactive
check-in needs, validated against a real caller rather than a toy one.

---

## 8. Relationship to `function_handlers.py`

The ten tools there are the reference implementation of this contract and must
keep working. **This build does not rewrite them.** `OMNI_TOOLS` and
`FunctionHandlers.handle()` are replaced by registration of the *same* handler
methods, unchanged, wrapped as `ToolSpec`s:

```python
registry.register(ToolSpec(name='navigate_to', description=<existing text>,
                           parameters=<same schema, as a dict>,
                           handler=lambda a, ctx: handlers._navigate_to(a)))
```

Behaviour is bit-identical: same names, same descriptions, same returned strings.
Migrating them to native `(args, ctx)` handlers, and moving `explore_area` (a stub
since it was written, `function_handlers.py:567`) onto the LONG path, is
follow-up work — not part of proving the loop.

---

## 9. Open questions — I need answers before implementing

1. **Home Assistant (§0.2).** HA base URL + a long-lived token, and which entity
   to prove it on? Or defer HA and prove the loop with `map_current_floor` +
   memory only?
2. **Memory writes (§5.2).** Confirm (b) — defer `StoreFact.srv` to a follow-up —
   or say the word and I fold it in.
3. **`map_current_floor` scope.** Confirmed as a *stub* per the brief: acquires
   the job, sleeps/streams fake progress, honours cancel, reports completion. It
   proves the async path without pretending to map. Real mapping is its own build.
4. **Live model.** Stay on `gemini-3.1-flash-live-preview` and build the
   report-later path in OMNI (my recommendation — it is needed for
   agent-initiated work regardless), or is moving the Live model back to a
   2.5-series native-audio model on the table? It changes §4's transport, not its
   contract.

---

## 10. Build order (after approval)

Matches the brief. Each step is independently testable.

| # | deliverable | gate |
|---|---|---|
| 1 | `spec.py`, `result.py`, `registry.py`, `schema.py` | pytest, no ROS, no `google.genai`: register / duplicate name / bad schema / dispatch / unknown tool / handler raises / JSON Schema → `genai_types.Schema` shape |
| 2 | `gemini_bridge` wired to the registry | `_recv_loop` dispatches through `invoke`; one `LiveClientToolResponse` per event asserted; live gate — wake word, "what's your status", OMNI answers |
| 3 | `jobs.py` + `reporter.py` | fake clock unit tests: ack < 100 ms, completion routed per `report`, watchdog fires `cancel`, suppression degrades `INTERRUPT` |
| 4 | example tools: `ha_call_service` (pending Q1), `map_current_floor` stub, `recall` / `remember_this` | live: ask OMNI to map — it acks immediately, conversation continues, completion is spoken later unprompted |
| 5 | agent-initiated path | `CheckInBehavior` invokes one registered tool; existing 270 behavior_node tests still pass |
