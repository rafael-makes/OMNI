"""memory_client.py — behavior_node's bridge to the omni_memory ROS services.

Wraps the two omni_memory services with GRACEFUL DEGRADATION: memory is an
enhancement, never a hard dependency. If the omni_memory node (or the VPS behind
it) is unavailable, retrieve() returns '' and store() is a no-op — OMNI keeps
conversing normally.

  /omni_memory/retrieve_memories  (RetrieveMemories) — query -> context block
  /omni_memory/store_transcript   (StoreTranscript)  — transcript -> memories

DEDICATED NODE + EXECUTOR (important):
  These calls come from several threads — retrieve() from the wake-word thread,
  store() from _set_state (ROS executor or asyncio thread). Sharing behavior_node's
  node/executor made service responses intermittently drop (cross-thread contention
  on one node's clients + wait set). So MemoryClient runs its OWN node on its OWN
  single-threaded executor thread; a lock serializes call submission. This isolates
  all memory ROS traffic from behavior_node entirely.

LAZY IMPORT (see also frame_client):
  The omni_memory_msgs import lives inside __init__, not at module scope. At module
  scope it runs during `import behavior_node.behavior_node` — before main() executes
  — so an unbuilt or unsourced omni_memory_msgs would kill the WHOLE robot with a
  ModuleNotFoundError. The SPEC says memory is a soft dependency ("if omni_memory is
  down, OMNI still converses"); an eager import quietly broke that promise, since a
  missing srv package is exactly the case it was meant to survive. Do not hoist it up.
"""
from __future__ import annotations

import threading
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor


class MemoryClient:
    def __init__(self, parent_node, *, enabled: bool = True, service_timeout: float = 2.0):
        self._log = parent_node.get_logger()
        self.enabled = enabled
        self._timeout = service_timeout

        # Set before any early return so shutdown() and the call paths stay safe.
        self._node = None
        self._exec = None
        self._spin_thread = None
        self._retrieve_cli = self._store_cli = self._rekey_cli = None
        self._srv_retrieve = self._srv_store = self._srv_rekey = None
        self._lock = threading.Lock()  # serialize call_async submissions
        self._warmed = threading.Event()

        # LAZY IMPORT — see the module docstring. Memory is a soft dependency;
        # a missing omni_memory_msgs must not stop OMNI from starting.
        try:
            from omni_memory_msgs.srv import (
                RekeyPerson,
                RetrieveMemories,
                StoreTranscript,
            )
        except ImportError as exc:
            self._log.warn(
                f"memory: omni_memory_msgs unavailable ({exc}) — persistent memory "
                f"disabled. Build it and re-source the workspace (a shell sourced "
                f"before the package was built will not see it)."
            )
            self.enabled = False
            self._warmed.set()   # release anyone waiting on warmup
            return
        self._srv_retrieve = RetrieveMemories
        self._srv_store = StoreTranscript
        self._srv_rekey = RekeyPerson

        # Own node + executor, spun on a dedicated daemon thread.
        self._node = rclpy.create_node("behavior_memory_client")
        self._retrieve_cli = self._node.create_client(
            RetrieveMemories, "/omni_memory/retrieve_memories"
        )
        self._store_cli = self._node.create_client(
            StoreTranscript, "/omni_memory/store_transcript"
        )
        self._rekey_cli = self._node.create_client(
            RekeyPerson, "/omni_memory/rekey_person"
        )
        self._exec = SingleThreadedExecutor()
        self._exec.add_node(self._node)
        self._spin_thread = threading.Thread(
            target=self._exec.spin, name="memory-exec", daemon=True
        )
        self._spin_thread.start()

        # Warm up the DDS request+reply paths in the background so the first real
        # retrieve round-trips. A freshly created service client matches its
        # request path before its reply path; until the reply path is up, the
        # server's response is dropped and the call times out. In production this
        # completes long before the first wake word; harmless if omni_memory is
        # absent (it just times out and retries on the real call).
        if self.enabled:
            threading.Thread(target=self._warmup, daemon=True).start()

    def _warmup(self) -> None:
        try:
            got_r = self._retrieve_cli.wait_for_service(timeout_sec=30.0)
            self._store_cli.wait_for_service(timeout_sec=5.0)
            if got_r:
                # Passive settle after discovery so the DDS reply path finishes
                # matching before the first real call (an active warmup call that
                # times out on the cold path can wedge the client instead).
                time.sleep(2.0)
                self._log.info("memory: service connection warmed up")
        except Exception:  # noqa: BLE001 - warmup is best-effort
            pass
        finally:
            # Always release waiters; if omni_memory was absent, the real call
            # will just degrade gracefully.
            self._warmed.set()

    def shutdown(self) -> None:
        """Stop the executor, WAIT for the spin thread to exit, then destroy the node.

        The join is load-bearing — see FrameClient.shutdown for the full note.
        executor.shutdown() returns before the spin loop has actually stopped, and
        destroying the node underneath a live spin thread aborts the process.
        """
        # _node/_exec are None if the lazy import failed — nothing to tear down.
        if self._node is None:
            return
        try:
            self._exec.shutdown()
            if self._spin_thread is not None:
                self._spin_thread.join(timeout=2.0)
            self._node.destroy_node()
        except Exception:  # noqa: BLE001 - best effort on teardown
            pass

    def _ready(self, client) -> bool:
        """Poll the non-blocking readiness flag up to the timeout (no wait_for_service:
        our dedicated executor keeps the graph current, and service_is_ready() avoids
        any guard-condition wait)."""
        deadline = time.monotonic() + self._timeout
        while not client.service_is_ready():
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)
        return True

    # ── retrieve (blocking, short timeout) ──────────────────────────────────────

    def retrieve_context(self, query: str, *, k: int = 5, person: str | None = None) -> str:
        """Return a formatted memory context block, or '' if unavailable/empty."""
        if not self.enabled:
            return ""
        # Let the one-time background warmup finish establishing the DDS paths
        # before the first real call. Set long before the first wake word in
        # production; only actually waits during cold start (e.g. the test gate).
        self._warmed.wait(self._timeout)
        if not self._ready(self._retrieve_cli):
            self._log.warn(
                "memory: retrieve_memories service unavailable — continuing without memory"
            )
            return ""
        req = self._srv_retrieve.Request(query=query, k=int(k), person=person or "")
        with self._lock:
            future = self._retrieve_cli.call_async(req)
        done = threading.Event()
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(self._timeout):
            self._log.warn("memory: retrieve_memories timed out — continuing without memory")
            return ""
        try:
            resp = future.result()
        except Exception as exc:  # noqa: BLE001
            self._log.warn(f"memory: retrieve_memories failed: {exc}")
            return ""
        if not resp or not resp.success:
            self._log.warn(
                f"memory: retrieve unsuccessful: {getattr(resp, 'message', 'no response')}"
            )
            return ""
        self._log.info(f"memory: retrieved {resp.count} memory record(s) for injection")
        return resp.context_block or ""

    # ── store (fire-and-forget) ─────────────────────────────────────────────────

    def store_transcript(
        self,
        transcript: str,
        *,
        person: str | None = None,
        session_id: str | None = None,
        source: str = "conversation",
    ) -> None:
        """Summarize + store a transcript. Non-blocking; logs the outcome.

        Submitted on a short-lived thread so the caller (often _set_state on the
        ROS executor thread) never blocks; the dedicated executor completes it.
        """
        if not self.enabled:
            return
        if not transcript or not transcript.strip():
            return
        threading.Thread(
            target=self._do_store,
            args=(transcript, person or "", session_id or "", source),
            daemon=True,
        ).start()

    def _do_store(self, transcript: str, person: str, session_id: str, source: str) -> None:
        if not self._ready(self._store_cli):
            self._log.warn(
                "memory: store_transcript service unavailable — dropping transcript "
                "(conversation not persisted)"
            )
            return
        req = self._srv_store.Request(
            transcript=transcript, person=person, session_id=session_id, source=source
        )
        with self._lock:
            future = self._store_cli.call_async(req)
        future.add_done_callback(self._on_store_done)
        self._log.info(f"memory: storing transcript ({len(transcript)} chars)")

    def rekey_person(self, from_person: str, to_person: str) -> None:
        """Re-label a person's memories (unknown_N -> name). Non-blocking, best-effort."""
        if not self.enabled or not from_person or not to_person or from_person == to_person:
            return
        threading.Thread(
            target=self._do_rekey, args=(from_person, to_person), daemon=True
        ).start()

    def _do_rekey(self, from_person: str, to_person: str) -> None:
        if not self._ready(self._rekey_cli):
            self._log.warn("memory: rekey_person service unavailable — memories not merged")
            return
        req = self._srv_rekey.Request(from_person=from_person, to_person=to_person)
        with self._lock:
            future = self._rekey_cli.call_async(req)

        def _done(fut):
            try:
                resp = fut.result()
                self._log.info(f"memory: {resp.message}" if resp and resp.success
                               else f"memory: rekey unsuccessful: {getattr(resp,'message','')}")
            except Exception as exc:  # noqa: BLE001
                self._log.warn(f"memory: rekey_person failed: {exc}")

        future.add_done_callback(_done)

    def _on_store_done(self, future) -> None:
        try:
            resp = future.result()
        except Exception as exc:  # noqa: BLE001
            self._log.warn(f"memory: store_transcript failed: {exc}")
            return
        if resp and resp.success:
            self._log.info(f"memory: {resp.message}")
        else:
            self._log.warn(
                f"memory: store unsuccessful: {getattr(resp, 'message', 'no response')}"
            )
