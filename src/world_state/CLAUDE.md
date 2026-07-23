# CLAUDE.md — world_state

Read this at the start of every session. Write the pass test before the
implementation.

## What this is
One shared, live answer to **"who is where right now."** Detections from every
camera fold into a single set of person tracks; anything that wants to know who
is around reads one topic instead of re-deriving it from raw vision.

State tracking **only** — no behaviour, no LLM calls, no decisions. Consumers
(behavior_node, future session work) decide what to *do* with the state.

Since Session 7 each track also carries a **coarse zone** — the named room the
person is estimated to be in — so "who is *where*" is answerable, not just "who".

## Zones — the coarse "where" (Session 7)

Each `PersonTrack` carries `map_xy` (estimated map-frame position) and `zone`
(the named room it falls in). The ROS wrapper computes both from the robot pose
(TF `map`→`base_link`) and the detection's camera + horizontal bearing, using
the shared **`omni_zones`** library (polygons, point-in-zone). The core tracker
never does geometry — it carries `map_xy`/`zone` through exactly like `bbox`.

**This is room-level, not metric.** Read `omni_zones/geometry.py` before trusting
a coordinate. Error sources, largest first: **distance is assumed** (`1.5 m`,
monocular vision gives no range), **head pan is ignored** (the head cam is on a
pan/tilt `head_tracking` swings ±40°; only the mounting offset is modelled), and
**bearing needs the image width** (param `camera_image_width`, defaults 1920 px).
When the estimate lands in no zone but the robot is in one, the wrapper reports
the **robot's** zone — a person at conversational range is almost always in the
robot's room. That fallback is what makes "come here" land in the right room.

Degrades cleanly: no TF (localisation down) → `map_xy`/`zone` are `null` and the
tracker is identity-only, exactly as before. A frame with no pose never wipes a
good last-known zone (that is how "where did I last see Rafael" survives him
leaving). Empty zone map (the shipped default) → every point is in no zone → all
`zone`s are `null`; fill in `omni_zones/config/zones.yaml` once the space is
traced into rooms. `test_zone_tracking.py` covers all of this.

## Layout
- `world_state/` — core library. **No ROS imports.** Must run on a desktop with
  no ROS installed (SPEC convention, same as `omni_memory`).
  - `models.py` — `Detection` (an observation), `PersonTrack` (a believed
    person), `StateEvent` (a transition).
  - `tracker.py` — `WorldState`: ingest, timeout, snapshot.
- `node.py` — the ROS2 wrapper. Imported only by the entry point, so
  `import world_state` never pulls in rclpy.
- `tests/` — pytest, passes with the robot off. `test_sources.py` skips itself
  when rclpy is absent.

## Data flow
```
Jetson: /camera/identities  (String JSON, per-face identity)  ─┐
        /camera/detections  (Detection2DArray, YOLO persons)  ─┼→ WorldState
        (rear-camera equivalents, once they exist)            ─┘      │
                                                                      ▼
                          /omni/world_state (String JSON, 1 Hz)
                          ~/query_world_state (std_srvs/Trigger)
```

Every source is configured as **`topic=camera`** and an untagged source is
rejected at startup. Session 10 depends on knowing which camera saw what, so
the tag is mandatory rather than optional.

## Matching rules (v1)
1. A detection carrying an **identity** matches that identity's track on *any*
   camera → camera handoff produces one person, not two.
2. A detection with **no identity** matches by `(camera, source_track)`.
3. **Containment**, both directions: a body box that swallows an existing
   track's face — or a face that lands inside an existing body — is that same
   person. This is what stops `/camera/detections` and `/camera/identities`
   from counting one human twice.
4. **Centroid**, within `match_radius` px on the same camera, for untracked
   boxes that nothing contains.
5. An anonymous track that later arrives with an identity is **absorbed** into
   the identified track, keeping the earlier `first_seen`.

Rules 3 and 4 skip any track already updated at the same timestamp, so two
boxes in one frame can never collapse into one person.

### Why person boxes are off by default
Live run 2026-07-19, one person in frame, `present_count` 2–3. Rules 3 and 4
cannot fuse a face track with a body track reliably: `_track_containing` picks
the nearest centre, and a body track's centre is *always* nearer an incoming
body box than a face track's centre is. So the first body box mints a body
track, and every subsequent body box goes to it rather than to the person's
face track. Face and body stay separate permanently, and the identified track's
bbox flip-flops between the two.

`/camera/identities` alone is the better basis: it carries the Jetson's own
per-face `track` ids *and* identities, so no geometry is needed. The cost is
that a person facing away (no detectable face) is not tracked at all. Enable
`detection_sources` only if you specifically need body-without-face presence,
and expect double counting when you do.

Upstream note: `/camera/identities` sometimes emits **two rows for one face** —
one identified, one with an empty identity a few px away. That shows up as a
phantom `person_N` next to a named track. It is a head_detector issue, not
something the tracker can resolve.

### Why rules 3 and 4 exist
`head_detector._build_array` **never sets `Detection2D.id`** — verified in the
source, not assumed. So every person box arrives with no track id. Without
centroid association each frame minted a fresh person (~10/s); without
containment the body box and the recognised face were two people standing in
the same spot. Both were caught by replaying the real payload shapes before the
first live run.

`unknown_N` from the Jetson recognizer is a *stable id*, so it handoffs like a
name — but `identified` is false for it. `known_present` lists real names only.

## Known limits — deliberate, do not "fix" without asking
- **No cross-camera dedup for anonymous people.** Two cameras seeing the same
  unnamed stranger read as two people. Fixing this needs appearance embeddings
  or geometry; v1 keeps it simple on purpose. Covered by a test that asserts
  the current (limited) behaviour so a future change is a conscious one.
- **No re-identification.** If the recognizer changes its mind about who
  someone is, that is a new track. Hysteresis lives upstream in the Jetson
  recognizer (see `omni_memory/SPEC.md` "Behaviour rules"), not here.
- **Named people are never pruned** from history; anonymous away-tracks are
  capped at `max_history` (50). Knowing Rafael left an hour ago is the point.
- **`first_seen` is first *sighting*, not first arrival in the room.**
- **Association is 2D pixel geometry, no depth.** Someone walking directly
  behind someone else can be absorbed into their track. Acceptable while the
  head camera sees one or two people at conversational range.
- **`/camera/identities` carries no per-face score**, so a recognised face is
  ingested at confidence 1.0. Identity has already passed hysteresis and the
  frontality gate on the Jetson, so `min_confidence` only ever filters YOLO
  person boxes.

## Clock
Detections are stamped with the **Pi's** `time.time()` on arrival, never with
the message header stamp. The Jetson publishes these and the two clocks are not
synchronised — mixing them makes "seconds since seen" meaningless. The core
library is clock-agnostic: the caller always supplies the timestamp.

## Parameters
| param | default | note |
|---|---|---|
| `identity_sources` | `['/camera/identities=head']` | `topic=camera` pairs |
| `detection_sources` | `[]` | person boxes; **off by default**, see below |
| `visibility_timeout` | `3.0` | s without a detection → away |
| `publish_rate` | `1.0` | Hz |
| `min_confidence` | `0.0` | floor for ingesting a detection (YOLO boxes only) |
| `match_radius` | `160.0` | px; centroid association. `0` disables it |
| `person_class` | `person` | YOLO class filter |
| `map_frame` | `map` | TF frame for the robot pose |
| `robot_frame` | `base_link` | TF child frame |
| `zones_config` | `''` | YAML with a top-level `zones:` map; `''` → `omni_zones`'s shipped default |
| `camera_offsets_deg` | `['head=0.0','rear=180.0']` | camera mounting yaw vs robot-forward |
| `camera_hfov_deg` | `['head=66.0','rear=66.0']` | horizontal FOV per camera |
| `camera_image_width` | `['head=1920.0','rear=1920.0']` | px; needed to turn a column into a bearing |
| `assumed_person_distance` | `1.5` | m; monocular range guess, not a measurement |

QoS: identities on default (reliable) depth 10; `Detection2DArray` on
**SENSOR_DATA (best effort)** — the Jetson publishes best-effort, and a
reliable subscription silently matches nothing.

## Running
```
cd ~/omni_ws && colcon build --packages-select world_state && source install/setup.bash
ros2 launch world_state world_state.launch.py
ros2 topic echo --full-length /omni/world_state          # --full-length: JSON is long
ros2 service call /world_state_node/query_world_state std_srvs/srv/Trigger
python3 -m pytest tests/ -q                              # robot off
```

Adding the rear camera when the Orin publishes it:
```
ros2 launch world_state world_state.launch.py \
  identity_sources:="['/camera/identities=head','/rear_camera/identities=rear']"
```

## Status
- [x] Core library + tests (appear / disappear / timeout / camera handoff).
- [x] ROS2 wrapper: 1 Hz JSON topic + Trigger service.
- [x] Live gate on the Pi: appear → identified → head-to-rear handoff stays
      **one track** → timeout to away.
- [x] Replay gate against **head_detector's real payload shapes** (both topics
      at 10 Hz, `det.id` unset): ~160 detections over 8 s collapse to exactly
      one track, then time out to away. `scratchpad/replay.py` pattern — worth
      recreating in `scripts/` if this needs regression cover.
- [x] Deployed Jetson build **confirmed identical** to the Pi's source of truth
      (matching md5 of `head_detector_node.py`), so the payload shapes the
      replay gate shrugs off are the real ones.
- [ ] Not yet run against the **live Jetson** — the vision stack was not up
      (`ros2 launch omni_jetson_bringup head_detector.launch.py` on the Orin,
      reachable as `ssh Omni`).
- [ ] **Blocker: cross-machine DDS discovery is unreliable.** On 2026-07-19 the
      Jetson could not see the Pi's long-running `bms_node` publisher at all,
      though a publisher created *after* the subscriber worked. Both hosts are
      dual-homed (WiFi + direct eth) and discovery traffic was riding WiFi. This
      node subscribes across that boundary, so it can come up permanently deaf —
      verify actual data flow, not `ros2 topic list`, on the first live run.
- [ ] **Rear camera publishes no identities topic yet.** The `=rear` path is
      exercised only by synthetic publishers. Wire it up when the rear
      recognizer lands, then re-run the handoff gate for real.
