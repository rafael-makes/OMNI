"""ROS2 node wrapping the OMNI memory layer (SPEC Step 4).

Exposes two services on the Pi, talking to the omni-core Supabase over WireGuard:
  - /omni_memory/store_transcript  (StoreTranscript): summarize + store
  - /omni_memory/retrieve_memories (RetrieveMemories): query -> context block

Only this module imports rclpy / the msgs package; the core library
(models/store/embedder/summarizer/context_format) stays ROS-free.
"""
from __future__ import annotations

import hashlib
import threading
import time

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from omni_memory_msgs.srv import RekeyPerson, RetrieveMemories, StoreTranscript

from .context_format import format_context
from .embedder import GeminiEmbedder
from .models import VALID_SOURCES
from .store import MemoryStore, load_env
from .summarizer import GeminiSummarizer, summarize_transcript

DEFAULT_ENV_FILE = "/home/pi/omni_ws/src/omni_memory/.env"

# store_transcript is a slow (~seconds) non-idempotent write. ROS2 services have
# no built-in request dedup, and a slow reply can make the transport re-deliver
# the same request, causing duplicate stores. We dedup on request content within
# this window so retransmits are absorbed.
_DEDUPE_TTL_SECONDS = 300.0


def _opt(value: str):
    """Treat empty ROS string fields as 'unset'."""
    value = (value or "").strip()
    return value or None


class OmniMemoryNode(Node):
    def __init__(self) -> None:
        super().__init__("omni_memory")

        # --- parameters ---
        env_file = self.declare_parameter("env_file", DEFAULT_ENV_FILE).value
        self._default_k = int(self.declare_parameter("default_k", 5).value)
        table = self.declare_parameter("table", "memories").value
        embed_model = self.declare_parameter("embed_model", "gemini-embedding-001").value
        text_model = self.declare_parameter("text_model", "gemini-2.5-flash").value

        # Load creds from the project .env (Supabase + Gemini) if present.
        if env_file:
            load_env(env_file)

        # --- backends (Steps 1-3) ---
        self._embedder = GeminiEmbedder(model=embed_model)
        self._store = MemoryStore(table=table, embedder=self._embedder)
        self._summarizer = GeminiSummarizer(model=text_model)

        # Idempotency cache for store_transcript (a slow, non-idempotent networked
        # write): request-hash -> (timestamp, result); result is None while a
        # request is in flight. Absorbs any duplicate/retransmitted delivery of an
        # identical request so it is stored only once.
        self._dedupe_lock = threading.Lock()
        self._recent_stores: dict[str, tuple] = {}

        # One MutuallyExclusive group per service: a handler is never re-entered
        # while it's still running, and keeping store/retrieve in separate groups
        # lets them run concurrently under the MultiThreadedExecutor (a long store
        # won't block a retrieve).
        store_group = MutuallyExclusiveCallbackGroup()
        retrieve_group = MutuallyExclusiveCallbackGroup()
        self._store_srv = self.create_service(
            StoreTranscript,
            "~/store_transcript",
            self._on_store_transcript,
            callback_group=store_group,
        )
        self._retrieve_srv = self.create_service(
            RetrieveMemories,
            "~/retrieve_memories",
            self._on_retrieve_memories,
            callback_group=retrieve_group,
        )
        self._rekey_srv = self.create_service(
            RekeyPerson,
            "~/rekey_person",
            self._on_rekey_person,
            callback_group=retrieve_group,
        )
        self.get_logger().info(
            f"omni_memory ready — table='{table}', embed='{embed_model}', text='{text_model}'"
        )

    # --- service callbacks ---

    def _dedupe_key(self, request) -> str:
        raw = "\x00".join(
            [request.transcript, request.person, request.session_id, request.source]
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _claim_or_cached(self, key: str):
        """Return (claimed, cached_result). claimed=True means we own the work;
        otherwise a duplicate is in flight or was just processed."""
        now = time.time()
        with self._dedupe_lock:
            self._recent_stores = {
                k: v for k, v in self._recent_stores.items() if now - v[0] < _DEDUPE_TTL_SECONDS
            }
            if key in self._recent_stores:
                return False, self._recent_stores[key][1]
            self._recent_stores[key] = (now, None)  # claim, in-flight
            return True, None

    def _record_result(self, key: str, result: tuple) -> None:
        with self._dedupe_lock:
            self._recent_stores[key] = (time.time(), result)

    def _on_store_transcript(self, request, response):
        key = self._dedupe_key(request)
        claimed, cached = self._claim_or_cached(key)
        if not claimed:
            # Duplicate delivery of an identical request. Do NOT store again.
            if cached is not None:
                response.success, response.stored_count, response.ids, response.message = cached
            else:
                response.success = True
                response.stored_count = 0
                response.ids = []
                response.message = "duplicate request ignored (identical store in progress)"
            self.get_logger().warn("store_transcript: ignored duplicate request")
            return response
        try:
            source = _opt(request.source)
            if source is not None and source not in VALID_SOURCES:
                response.success = False
                response.message = (
                    f"invalid source '{source}'; must be one of {VALID_SOURCES}"
                )
                self._record_result(
                    key, (response.success, 0, [], response.message)
                )
                return response

            records = summarize_transcript(
                request.transcript,
                session_id=_opt(request.session_id),
                default_person=_opt(request.person),
                summarizer=self._summarizer,
            )
            ids = []
            for rec in records:
                if source is not None:
                    rec.source = source
                saved = self._store.store(rec)
                ids.append(saved.id)

            response.success = True
            response.stored_count = len(ids)
            response.ids = ids
            response.message = f"stored {len(ids)} memory record(s)"
            self.get_logger().info(response.message)
            self._record_result(key, (True, len(ids), list(ids), response.message))
        except Exception as exc:  # noqa: BLE001 - report failure to the caller
            response.success = False
            response.stored_count = 0
            response.ids = []
            response.message = f"store_transcript failed: {exc}"
            self.get_logger().error(response.message)
            # Failures are not cached: drop the claim so a genuine retry can run.
            with self._dedupe_lock:
                self._recent_stores.pop(key, None)
        return response

    def _on_rekey_person(self, request, response):
        try:
            from_person = _opt(request.from_person)
            to_person = _opt(request.to_person)
            if from_person is None or to_person is None:
                response.success = False
                response.updated = 0
                response.message = "from_person and to_person are both required"
                return response
            n = self._store.rekey(from_person, to_person)
            response.success = True
            response.updated = n
            response.message = f"re-keyed {n} memory record(s): {from_person} -> {to_person}"
            self.get_logger().info(response.message)
        except Exception as exc:  # noqa: BLE001
            response.success = False
            response.updated = 0
            response.message = f"rekey_person failed: {exc}"
            self.get_logger().error(response.message)
        return response

    def _on_retrieve_memories(self, request, response):
        try:
            k = request.k if request.k > 0 else self._default_k
            records = self._store.retrieve(
                request.query, k=k, person=_opt(request.person)
            )
            response.success = True
            response.context_block = format_context(records)
            response.count = len(records)
            response.message = f"retrieved {len(records)} memory record(s)"
            self.get_logger().info(response.message)
        except Exception as exc:  # noqa: BLE001 - report failure to the caller
            response.success = False
            response.context_block = ""
            response.count = 0
            response.message = f"retrieve_memories failed: {exc}"
            self.get_logger().error(response.message)
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OmniMemoryNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
