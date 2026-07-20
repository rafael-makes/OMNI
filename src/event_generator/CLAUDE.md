# CLAUDE.md — event_generator

Read this at the start of every session. Write the pass test before the
implementation.

## What this is
Semantic **presence events** derived from `world_state`'s snapshots:
`person_appeared`, `person_left`, `unknown_person_detected`. It turns a 1 Hz
stream of "who is visible right now" into the far rarer "something actually
changed."

Derivation **only** — no behaviour, no LLM calls, no decisions. `behavior_node`
decides whether an arrival is worth greeting; this package only says an arrival
happened. Same split as `world_state`, one layer up.

## Layout
- `event_generator/` — core library. **No ROS imports.** Must run on a desktop
  with no ROS installed (SPEC convention, same as `world_state`/`omni_memory`).
  - `models.py` — `Event`, plus `is_named` / `is_unknown`.
  - `generator.py` — `EventGenerator`: ingest a snapshot, get events.
- `node.py` — the ROS2 wrapper. Imported only by the entry point, so
  `import event_generator` never pulls in rclpy.
- `tests/` — pytest, passes with the robot off.
  - `fixtures/world_state_live.jsonl` — a captured live snapshot sequence.

## Data flow
```
/omni/world_state (String JSON, 1 Hz)  ──→  EventGenerator  ──→  /omni/events
                                                                 (String JSON,
                                                                  one msg per event)
```

One message per event, not batched: a subscriber acting on `person_appeared`
should not have to unpack an array and find the element that concerns it.
Reliable QoS, depth 10 — these are rare and meaningful, and a dropped one means
a missed greeting or a stuck presence belief.

## The constraint that shaped everything here

**World state is face-anchored.** `world_state` runs on `/camera/identities`
alone (its CLAUDE.md, "Why person boxes are off by default"), so a track is
visible only while the Jetson can see *and recognise a face*. Someone at the
workbench who turns to pick up a screwdriver stops being visible. Someone who
looks down at their hands stops being visible. This is constant and normal and
it is **not the person leaving**.

So: **visibility is not presence.** Every debounce exists to put distance
between the two.

| rule | default | why |
|---|---|---|
| `absence_grace` | `90.0` s | sustained absence before `person_left`. Must sit well above "turned away for a while" and well below "went to make a coffee". `world_state`'s own `visibility_timeout` is 3s — three orders of magnitude too twitchy to mean *left*. |
| `unknown_min_snapshots` | `3` | a stable `unknown_N` must survive ~3s of **consecutive** sightings |
| `appear_min_snapshots` | `1` | named people fire immediately — see below |
| `unknown_cameras` | `['head']` | stranger detection is head-only for now |
| `named_overlap_radius` | `160.0` px | an `unknown_N` this close to a known person **is** that person — see below |
| `unknown_min_face_px` | `50.0` px | smaller than this (shorter side) is clutter, not a stranger |

### Why flicker cannot re-fire person_appeared
There is no "recently greeted" suppression in this library, and there must not
be. A flickering person **never leaves `PRESENT`**, so they can never re-enter
it, so `person_appeared` cannot fire twice. That is the whole mechanism. The
trap it avoids: grace is measured from the last **sighting**, not from the last
event, so ten turn-away/turn-back cycles spanning far more than 90 seconds still
produce nothing (`test_repeated_flicker_never_re_fires_appeared`).

Greeting *cooldown* is a different concern and lives in `behavior_node`.

### Why named people fire on the first snapshot
An identity has already passed the Jetson recognizer's hysteresis and frontality
gate before it ever reaches `world_state`. Re-confirming it here buys nothing and
costs a full second against a greeting latency budget of ~3s. Unknowns get the
3-snapshot gate because a one-frame recognizer id is not evidence of a person.

### The phantom-stranger problem, and why overlap suppression exists
The spec anticipated one upstream duplicate-face bug: rows with an **empty**
identity. Live capture found a second, worse variant wearing an `unknown_N` id.

The Jetson recognizer **drops to a fresh `unknown_N` whenever confidence dips** —
a head turn is enough — and `world_state` cannot merge it back, by design ("no
re-identification... that is a new track"). So one person sitting still mints a
stream of anonymous ids *on their own face*. Measured 2026-07-19, 129 seconds,
one seated person: **five stranger announcements**, four of them Rafael.

Caught live: `unknown_58` at bbox (606.2, 187.9) while `rafael` sat at
(606.3, 187.4) — same frame, both flagged visible.

Three things were needed, and all three are load-bearing:

1. **Consecutive, not cumulative, snapshot counting.** `unknown_30` was announced
   on 4 visible snapshots scattered across 98. "Persist across N snapshots" means
   a run, not a tally.
2. **Suppress an unknown overlapping a named person** — containment either way.
3. **Compare against the named person's *remembered* box, not just this frame's.**
   This is the subtle one. The phantom appears *because* recognition failed, so in
   that very frame the real person is usually not visible and contributes no box.
   Our own debounced `PRESENT` belief is the only thing that still knows they are
   there. Plus a centroid radius, because the remembered box freezes at wherever
   they were at dropout — which is very often mid-movement, since moving is why
   recognition failed. (Rafael leaned to (699.7, 220.1) in his last visible frame;
   the phantom then appeared 85px away at his usual seated spot.)

Result on the same recording: **five announcements down to one**, and that one
(`unknown_46` at (321, 530)) is a genuinely distinct, stationary detection.

**The trade, stated plainly:** a real stranger standing within ~160px of someone
OMNI knows — roughly touching, at conversational range — will not be announced.
That is deliberate. A phantom stranger beside every recognised person is a much
worse failure than a missed one. `test_overlap_suppression_is_what_does_it` pins
the mechanism so it cannot be "simplified" away without the suite going red.

### And a fourth thing: tiny boxes are not faces
Overlap suppression alone still left phantoms — but only at the frame edges, and
they were *small*. Measured live in the 1280x720 publish space:

| | box | where |
|---|---|---|
| Rafael, real face | 88 x 115 px | centre |
| `unknown_18` | 34.7 x 34.8 px | far left edge |
| `unknown_27` | 23.0 x 25.1 px | far right edge |

A 23px "face" is background clutter that YuNet scores above its 0.6 threshold —
a pattern on a shelf, a photo, a reflection. **No temporal debounce can ever
remove these**, because they are perfectly stationary and perfectly persistent;
they look exactly like a very patient person. The only separating signal is size,
and the two populations do not overlap. Hence `unknown_min_face_px` (50.0,
judged on the **shorter** side — area would let a 20x200px sliver through).

After all four filters: **65 seconds live, one seated person, exactly one event
(`person_appeared: rafael`) and zero phantoms.**

## What is deliberately ignored — do not "fix" without asking

- **Rows with no identity generate nothing.** `/camera/identities` is known to
  emit two rows for one face — one identified, one with an empty identity a few
  px away (`world_state` CLAUDE.md, "Upstream note"). Treating those as people
  would announce a phantom stranger beside every person OMNI recognises. **No
  debounce fixes this**, because the phantom is perfectly stable; not looking at
  it is the only correct handling. It is a `head_detector` bug.
- **Identity is the key, not `track_id`.** Tracks get re-minted; identities are
  stable across cameras and across re-minting. Consequence: when the recognizer
  promotes `unknown_3` to `rafael`, those are two independent keys — `rafael`
  appears with `away_duration=None`, and `unknown_3` goes absent silently.
  `person_left` fires for **named people only**, so the promotion emits no
  spurious departure.
- **`unknown_person_detected` fires once per stranger, ever** (per process). A
  returning `unknown_N` is the same stranger the recognizer already told us
  about; re-announcing is noise. Note the recognizer persists unknowns across
  reboots (43 of them at last count), so these ids are long-lived.
- **Strangers leaving is not an event.** No `person_left` for `unknown_N`.
- **`away_duration` includes the grace period.** It is measured from the last
  real sighting, so a 10-minute absence reads ~600s, not ~510s. The greeting
  prompt reasons about how long they were *actually* gone, so the grace period
  must not be a free pass.
- **Out-of-order snapshots are dropped.** A backwards `stamp` would make every
  absence gap negative and silently freeze all departures.
- **Cold start is briefly blind.** Overlap suppression checks an unknown against
  named people *this node believes are present*, and a freshly started node
  believes nothing. Observed 2026-07-19: a restart with Rafael out of frame
  announced `unknown_24` — plausibly Rafael himself, unrecognised, with no known
  track to suppress against. It settles within seconds of the first recognised
  face and does not recur, so this is a note, not a bug to engineer around. It
  only shows up during development restarts; in production the node runs for
  days.

## Clock
Time comes from each snapshot's own `stamp` field, which `world_state` set from
the **Pi's** clock. The library never calls `time.time()` — that is what lets
the replay test run a ten-minute absence in a millisecond, and what stops
`away_duration` being wrong by however long the topic took to arrive.

## Parameters
| param | default | note |
|---|---|---|
| `state_topic` | `/omni/world_state` | |
| `events_topic` | `/omni/events` | |
| `absence_grace` | `90.0` | s of sustained absence before `person_left` |
| `unknown_min_snapshots` | `3` | |
| `appear_min_snapshots` | `1` | |
| `unknown_cameras` | `['head']` | rear publishes no identities topic yet |
| `record_path` | `''` | append inbound snapshots to JSONL — how fixtures are captured |

## Running
```
cd ~/omni_ws && colcon build --packages-select event_generator && source install/setup.bash
ros2 run world_state world_state_node          # NOT the launch file — see below
ros2 launch event_generator event_generator.launch.py
ros2 topic echo /omni/events
python3 -m pytest tests/ -q                    # robot off
```

Capturing a new replay fixture during a live run:
```
ros2 launch event_generator event_generator.launch.py record_path:=/tmp/ws_capture.jsonl
```

### Gotcha: world_state's launch file is broken
`ros2 launch world_state world_state.launch.py` fails with *"Expected a
non-empty sequence... Got inconsistent input for `detection_sources`"* — the
`default_value="[]"` cannot be type-inferred by the launch substitution
machinery. Use `ros2 run world_state world_state_node` until that is fixed. It
is a `world_state` bug, not one of ours.

## Status
- [x] Core library + 25 debounce tests (appear / flicker / sustained absence /
      return / unknown stability / camera scope / malformed input).
- [x] ROS2 wrapper: `/omni/world_state` -> `/omni/events`.
- [x] **Live gate on real hardware 2026-07-19.** Jetson vision stack up, Pi
      consuming across the link: `person_appeared: rafael` fired on first
      sighting, `unknown_person_detected: unknown_29` fired exactly 3 snapshots
      after that id stabilised, and nothing re-fired while Rafael stayed put.
- [x] Replay gate against a captured live sequence (`tests/fixtures/`): 130
      snapshots / 129s / 471 null-identity phantom rows / 23 snapshots of real
      face dropout. Asserts Rafael is greeted **exactly once** and never "leaves".
- [x] Phantom-stranger suppression, found only by that capture: 5 false stranger
      announcements in 2 minutes reduced to 1 genuine one.
- [ ] Not yet exercised against a **rear** identities source — that topic does
      not exist yet. The `unknown_cameras` scope is a guard for when the rear
      recognizer lands, not a filter doing anything today.
- [ ] The DDS discovery blocker recorded in `world_state`'s CLAUDE.md **did not
      reproduce** on 2026-07-19 — identities crossed the link first try. It is
      not fixed, just not firing; verify data flow (not `ros2 topic list`) on
      every fresh bringup.
