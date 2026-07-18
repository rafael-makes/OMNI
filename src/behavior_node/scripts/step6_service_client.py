"""Step 6 pass test: two different people get DIFFERENTIATED memory retrieval.

Stores a fact for each of two people via MemoryClient (person-keyed, exactly as
behavior_node does once /camera/identity resolves them), then retrieves as each
and verifies each sees only their own fact (+ general), never the other's.

Assumes omni_memory is running. Run via step6_service_gate.sh.
"""
import sys
import threading
import time

import rclpy
from rclpy.node import Node

from behavior_node.memory_client import MemoryClient

TS = int(time.time())
ALICE = f"alice{TS}"
BOB = f"bob{TS}"

# Semantically SIMILAR facts (both pets) so only the person filter differentiates.
# Generic "User:" labels (as real Gemini transcription produces) so the summarizer
# attributes to default_person verbatim — matching how behavior_node keys by the
# /camera/identity person. (A real name in the label gets normalized by the model.)
ALICE_TX = "User: Just so you know, I have a cat named Whiskers."
BOB_TX = "User: For the record, I have a dog named Rex."


def retrieve_warm(mem, query, person, want, deadline=30):
    """Retrieve, retrying until the connection is warm (see step5 notes)."""
    end = time.time() + deadline
    block = ""
    while time.time() < end:
        block = mem.retrieve_context(query, k=5, person=person)
        if block and want in block.lower():
            break
        time.sleep(2)
    return block


def main():
    rclpy.init()
    node = Node("step6_gate_client")
    mem = MemoryClient(node, enabled=True, service_timeout=5.0)
    threading.Thread(target=lambda: rclpy.spin(node), daemon=True).start()

    ok = True
    try:
        print(f">> store fact for {ALICE} (cat) and {BOB} (dog)")
        mem.store_transcript(ALICE_TX, person=ALICE, session_id=f"s6-{ALICE}")
        mem.store_transcript(BOB_TX, person=BOB, session_id=f"s6-{BOB}")
        time.sleep(14)  # let both summarize+store land

        # Alice's view
        a_block = retrieve_warm(mem, "what pet do I have", ALICE, "whiskers")
        print(f"---- retrieve as {ALICE} ----\n{a_block or '(empty)'}")
        if "whiskers" in a_block.lower() and "rex" not in a_block.lower():
            print("   [PASS] Alice sees her cat, not Bob's dog")
        else:
            print("   [FAIL] Alice's retrieval not differentiated"); ok = False

        # Bob's view
        b_block = retrieve_warm(mem, "what pet do I have", BOB, "rex")
        print(f"---- retrieve as {BOB} ----\n{b_block or '(empty)'}")
        if "rex" in b_block.lower() and "whiskers" not in b_block.lower():
            print("   [PASS] Bob sees his dog, not Alice's cat")
        else:
            print("   [FAIL] Bob's retrieval not differentiated"); ok = False
    finally:
        try:
            from omni_memory import MemoryStore, load_env
            load_env("/home/pi/omni_ws/src/omni_memory/.env")
            s = MemoryStore()
            # Clean up by SESSION_ID, never by person: the summarizer normalises a
            # person name (alice<ts> -> "alice"), so a person-based delete both MISSES
            # the stored rows (they orphan) and could match a REAL person of that name.
            # session_id is stored verbatim, so it is exact and safe.
            deleted = 0
            for sid in (f"s6-{ALICE}", f"s6-{BOB}"):
                deleted += len(
                    s.client.table(s.table).delete().eq("session_id", sid).execute().data or []
                )
            print(f"   cleaned up {deleted} test row(s)")
        except Exception as exc:  # noqa: BLE001
            print(f"   cleanup warning: {exc}")
        mem.shutdown()          # stop the dedicated memory executor cleanly
        node.destroy_node()
        rclpy.shutdown()

    print(">> STEP 6 GATE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
