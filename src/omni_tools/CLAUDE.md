# CLAUDE.md — omni_tools

Read this and `SPEC.md` at the start of every session.

**Status: spec only. No implementation exists yet.** If you are reading this and
`omni_tools/` contains Python, update this line.

## What this is
The router between OMNI's reasoning (Gemini Live, in `behavior_node`) and its
capabilities. Every capability is a registered `ToolSpec`. Adding a behaviour
should mean registering a tool — never adding a branch to the voice path.

It is a **library, not a node**, and it runs inside `behavior_node`'s process on
the Pi 5. See `SPEC.md` §3 for why; do not turn it into a node without reading
that section, because the reason is latency in the voice path.

## Layout
```
omni_tools/
  spec.py       ToolSpec, Kind, Report, Retry, Caller   — no imports beyond stdlib
  result.py     ToolResult, ErrorKind                   — no imports beyond stdlib
  registry.py   ToolRegistry: register / gemini_tools / invoke
  schema.py     JSON Schema dict -> genai_types.Schema  — THE ONLY genai import
  jobs.py       JobManager: background execution + watchdog
  reporter.py   completion -> voice path
  adapters.py   call_service / send_action_goal / publish_topic builders
tests/          pytest, robot off, no google.genai required
```

## The rules that are load-bearing

**1. `registry.py` imports neither `rclpy` nor `google.genai`.** The whole point
of `parameters` being a plain JSON Schema dict is that the registry is testable
on a desktop with neither installed. If you find yourself importing either one
there, the design has drifted — put it behind `ctx` or in `schema.py`.

**2. `schema.py` imports `google.genai` lazily, inside the function.** Module-level
imports of soft dependencies have killed this robot before (see
`memory_client.py`'s docstring and `frame_client.py`). Do not hoist it.

**3. A handler never raises into the Gemini loop.** `invoke()` catches everything
and returns a speakable `ToolResult`. With the current Live model the call is
*blocking* (§0.1), so an unanswered function call is a hung conversation, not a
dropped one. `describe_scene` makes the same argument at
`function_handlers.py:782` — a tool exception is dead silence mid-sentence.

**4. One `LiveClientToolResponse` per `tool_call` event.** Never one message per
call. Gemini closes the socket with error 1008 if you send a second — it has
already advanced past tool state. The comment at `gemini_bridge.py:600` is the
scar tissue.

**5. Dispatch runs in `run_in_executor`, never on the asyncio loop.** Handlers
call rclpy and block for seconds; the mic send loop and the Gemini recv loop
share that loop. This is already how `gemini_bridge` dispatches — keep it.

**6. `ToolResult.speech` is mandatory, including on failure.** It must be a
sentence OMNI can say in character. That is the entire failure UX.

**7. Suppression is shared, not re-implemented.** `Reporter` calls
`suppression.interaction_blocked()` — the same function greetings and check-ins
use. `behavior_node/CLAUDE.md` warns explicitly against a second copy; the drift
is invisible until OMNI interrupts a conversation six months later.

## How to add a new tool

1. Write the handler. Signature `(args: dict, ctx: ToolContext) -> ToolResult | str`.
   Returning a plain string means success. No ROS imports at module scope if the
   handler can be kept ROS-free — prefer reaching ROS through `ctx.node`.
2. Write the `ToolSpec`. `description` is written **for the model**, in the second
   person, saying when to call it and when not to. Read the existing descriptions
   in `function_handlers.py` — `describe_scene`'s is the model of the form (it
   spends most of its length on *when not to*, and that is why it works).
3. Pick `Kind`. Can you promise a wall-clock ceiling under ~8 s? `FAST`. Otherwise
   `LONG`, and then `ack`, `report` and `cancel` are all required — a `LONG` tool
   with no `cancel` cannot be un-wedged.
4. Register it in `behavior_node`'s tool registration module.
5. Test it with the robot off (below). Then, if it moves anything, test it live
   with a clear floor and a hand on the e-stop.

Set `exposed_to_model=False` for anything the agent may call but conversation must
never reach.

## Testing

```bash
cd ~/omni_ws/src/omni_tools && python3 -m pytest -q     # robot off, no ROS needed
cd ~/omni_ws && colcon build --packages-select omni_tools behavior_node
source install/setup.bash
cd ~/omni_ws/src/behavior_node && python3 -m pytest -q  # must stay green (270 tests)
```

The registry, the schema conversion, `JobManager` and the error contract are all
unit-testable with no ROS and no Gemini — that is a design requirement, not a
convenience. Anything that needs a live session is a *manual* gate with a written
recipe, in the style of `behavior_node/scripts/step5_manual_gate.md`.

`JobManager` tests inject a clock. Do not write a test that sleeps for a watchdog.

## Gotchas found while reading the repo (2026-07-21)

**The Live model does not support async function calling.** `behavior_node`
runs `models/gemini-3.1-flash-live-preview`, which is documented as synchronous
function calling only — the model will not speak until the tool response is sent.
`google-genai` 2.0.1 *does* expose `Behavior.NON_BLOCKING`,
`FunctionResponse.will_continue` and `FunctionResponseScheduling`; setting them
against this model looks correct and does nothing. This is why the long-running
pattern is built in OMNI (`SPEC.md` §4) rather than delegated to the protocol.
Re-check this before assuming it is still true — if the model gains support, the
only file that changes is `schema.py`.

**Gemini 3.1 also dropped `thinkingBudget` for `thinkingLevel`, and now sends
multiple content parts per server event.** Relevant if you touch the recv loop.

**`MemoryStore` is not reachable from this process.** Use `MemoryClient` (ROS
services). `SPEC.md` §0.3 has the table. And `StoreTranscript` runs an LLM
summariser over whatever you send it — it is not a "write this record" call.

**No Home Assistant integration exists in the repo.** The brief implies one does.
`SPEC.md` §0.2.

**`audio_node` is not the Gemini Live client** and is deliberately not launched.
`gemini_bridge.py` in `behavior_node` is. Editing `audio_node` changes nothing on
the running robot.

**`explore_area` has always been a stub** (`function_handlers.py:567`) — it sets
`EXPLORING` and drives nothing. Do not treat it as a working example of a
long-running tool; it is the reason the pattern is needed.

**An empty-list ROS 2 parameter default is untypeable.** If tools ever take a
list-valued parameter, it must be a comma-separated string. `declare_parameter(n, [])`
infers `BYTE_ARRAY` and then rejects every override. This broke `world_state`'s
launch file and `check_in_zones`; unit tests never catch it, so `ros2 launch` once.

**Launch files and installed packages are copies.** `omni_ws` mixes editable
egg-link and copy installs; rebuild before concluding an edit had no effect
(`feedback_live_param_tuning`).

## Conventions
- Python 3.13 (Pi system Python, no venv). Type hints everywhere.
- `from __future__ import annotations` at the top of every module.
- Frozen dataclasses for anything that crosses a module boundary.
- Docstrings explain **why**, not what. This repo's files are read months later
  by someone deciding whether they may change something.
