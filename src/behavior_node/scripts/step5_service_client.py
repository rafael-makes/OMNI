"""Step 5 component gate: exercise behavior_node's MemoryClient against the live
omni_memory services (no audio/Gemini). Proves the store->retrieve wiring that
the Live integration relies on.

Assumes the omni_memory node is already running. Run via step5_service_gate.sh.
"""
import sys
import threading
import time

import rclpy
from rclpy.node import Node

from behavior_node.memory_client import MemoryClient
from behavior_node.memory_format import wrap_memory_context

PERSON = f"step5user{int(time.time())}"
SESSION = f"step5-{int(time.time())}"

# A simulated end-of-conversation transcript (what the bridge would accumulate).
TRANSCRIPT = (
    f"User: By the way, remember that my favourite tea is jasmine green tea.\n"
    f"OMNI: Noted — jasmine green tea is your favourite.\n"
    f"User: And I usually drink it in the evening.\n"
    f"OMNI: Understood, jasmine green tea in the evening."
)


def main():
    rclpy.init()
    node = Node("step5_gate_client")
    mem = MemoryClient(node, enabled=True, service_timeout=5.0)

    # Mirror behavior_node exactly: single-threaded spin on a background thread,
    # while MemoryClient calls are made from this (the "wake-word") thread.
    spin = threading.Thread(target=lambda: rclpy.spin(node), daemon=True)
    spin.start()

    ok = True
    try:
        # 1) store a conversation transcript (fire-and-forget) ...
        print(f">> store_transcript (person={PERSON})")
        mem.store_transcript(TRANSCRIPT, person=PERSON, session_id=SESSION)
        # ... wait for summarize+store to land (fire-and-forget has no return).
        time.sleep(12)

        # 2) retrieve with a related query — should surface the jasmine-tea fact.
        # Retry until the connection is warm (a settled client is 100% reliable;
        # see diag notes / step5_manual_gate.md). Cold-start reply-path matching
        # is racy for fresh DDS participants — production never hits this because
        # the first wake word is minutes after boot.
        print(">> retrieve_context (query='favourite tea'), waiting for warm connection")
        block = ""
        for attempt in range(12):
            block = mem.retrieve_context("what is my favourite tea", k=5, person=PERSON)
            if block and "jasmine" in block.lower():
                break
            time.sleep(2)
        print("---- context block ----")
        print(block if block else "(empty)")
        print("-----------------------")

        wrapped = wrap_memory_context(block)
        if block and "jasmine" in block.lower():
            print("   [PASS] stored fact retrieved via MemoryClient")
        else:
            print("   [FAIL] jasmine fact not retrieved"); ok = False
        if wrapped.startswith("[MEMORY]"):
            print("   [PASS] context wrapped for injection")
        else:
            print("   [FAIL] wrap_memory_context did not wrap"); ok = False

        # 3) graceful degradation: disabled client returns '' and no-ops.
        mem.enabled = False
        if mem.retrieve_context("anything", person=PERSON) == "":
            print("   [PASS] disabled retrieve returns ''")
        else:
            print("   [FAIL] disabled retrieve non-empty"); ok = False
        mem.enabled = True
    finally:  # noqa: PLW0603
        # cleanup this run's rows directly
        try:
            from omni_memory import MemoryStore, load_env
            load_env("/home/pi/omni_ws/src/omni_memory/.env")
            s = MemoryStore()
            # Clean up by SESSION_ID, never by person: the summarizer normalises a
            # person name, so a person-based delete both MISSES the stored rows (they
            # orphan) and could match a REAL person of that name. session_id is stored
            # verbatim, so it is exact and safe.
            n = len(
                s.client.table(s.table).delete().eq("session_id", SESSION).execute().data or []
            )
            print(f"   cleaned up {n} test row(s)")
        except Exception as exc:  # noqa: BLE001
            print(f"   cleanup warning: {exc}")
        node.destroy_node()
        rclpy.shutdown()

    print(">> STEP 5 SERVICE GATE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
