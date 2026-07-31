#!/usr/bin/env python3
"""Build the dwell replay fixture from the real world_state capture.

PROVENANCE — READ THIS BEFORE TRUSTING THE FIXTURE
--------------------------------------------------
`tests/fixtures/work_session_dwell.jsonl` is **derived, not recorded**. It is
`tests/fixtures/world_state_live.jsonl` — 130 snapshots / 129 s of genuine
hardware output, Rafael at the workbench, recorded 2026-07-19 — tiled end to end
until it spans long enough to cross a 20-minute dwell threshold, with Rafael's
rows tagged `zone: workbench`.

What that buys, and what it does not:

  * REAL: the face-dropout pattern (23 of 127 rows non-visible from one person
    sitting still), the 471 null-identity phantom rows, the five churning
    `unknown_N` ids. Every nasty property the live capture has, the dwell replay
    inherits — and those are exactly the properties that break naive dwell logic.
  * NOT REAL: the *duration*. A tiled 129 s loop is not the same as 25 unbroken
    minutes at a bench. In particular it cannot contain a dropout longer than the
    original capture's worst one, so it cannot prove the 90 s absence_grace is
    the right number for a long work session.

So this fixture is a regression gate on dwell mechanics over realistic noise, and
NOT evidence about real-world dwell durations. Replace it with a genuine 20+
minute recording when one exists — the session brief asks for exactly that:

    ros2 launch event_generator event_generator.launch.py \
        record_path:=/tmp/bench_session.jsonl
    # ...then sit at the bench for 20 minutes...
    python3 scripts/make_dwell_fixture.py --source /tmp/bench_session.jsonl \
        --tiles 1 --out tests/fixtures/work_session_dwell.jsonl

With --tiles 1 this script only tags zones and rewrites nothing else, so a real
recording passes through essentially untouched.

Usage:
    python3 scripts/make_dwell_fixture.py            # regenerate the shipped fixture
"""

from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(os.path.dirname(HERE), "tests", "fixtures")

DEFAULT_SOURCE = os.path.join(FIXTURES, "world_state_live.jsonl")
DEFAULT_OUT = os.path.join(FIXTURES, "work_session_dwell.jsonl")


def load(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def shift(snapshot: dict, offset: float, person: str, zone: str) -> dict:
    """One snapshot moved forward in time by `offset`, with `person` zone-tagged.

    Every absolute time in the snapshot moves by the same offset so the internal
    arithmetic stays consistent: a `last_seen` left unshifted would read as an
    enormous absence and fire a spurious person_left on the very first tile.
    """
    out = json.loads(json.dumps(snapshot))   # deep copy, it is small
    out["stamp"] = round(snapshot["stamp"] + offset, 3)

    for row in out.get("people", []) or []:
        for key in ("first_seen", "last_seen"):
            if isinstance(row.get(key), (int, float)):
                row[key] = round(row[key] + offset, 3)
        # Keep the derived field honest rather than stale.
        if isinstance(row.get("last_seen"), (int, float)):
            row["seconds_since_seen"] = round(
                max(0.0, out["stamp"] - row["last_seen"]), 3)
        # The zone world_state would have attached had zones.yaml been filled in
        # when this was recorded. Only the named person gets one: dwell is
        # named-only, and leaving the phantoms unplaced is both realistic and a
        # useful extra guard.
        if (row.get("identity") or "") == person:
            row["zone"] = zone
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--tiles", type=int, default=13,
                    help="repeats of the source; 13 x 129s = ~28 min")
    ap.add_argument("--person", default="rafael")
    ap.add_argument("--zone", default="workbench")
    args = ap.parse_args()

    src = load(args.source)
    if not src:
        raise SystemExit(f"no snapshots in {args.source}")

    # One tile spans first..last stamp; add the median inter-snapshot gap so the
    # seam between tiles looks like one more ordinary 1 Hz step rather than a
    # duplicate timestamp.
    span = src[-1]["stamp"] - src[0]["stamp"]
    step = span / max(1, len(src) - 1)
    tile_span = span + step

    rows = []
    for tile in range(args.tiles):
        offset = tile * tile_span - src[0]["stamp"]
        for snapshot in src:
            rows.append(shift(snapshot, offset, args.person, args.zone))

    # Stamps must be strictly increasing or the generator drops snapshots as
    # out-of-order and the fixture silently proves nothing.
    stamps = [r["stamp"] for r in rows]
    assert stamps == sorted(stamps), "tiling produced out-of-order stamps"

    with open(args.out, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    total = stamps[-1] - stamps[0]
    print(f"wrote {len(rows)} snapshots spanning {total:.0f}s "
          f"({total / 60.0:.1f} min) -> {args.out}")


if __name__ == "__main__":
    main()
