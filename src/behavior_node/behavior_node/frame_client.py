"""frame_client.py — behavior_node's bridge to the Jetson's frame_server service.

  /vision/get_camera_frame  (GetCameraFrame) — camera_id -> JPEG bytes

This crosses machines: the client runs on the Pi, the server on the Orin, over the
direct 192.168.50.0/24 link that Fast DDS is whitelisted to (see fastdds_eth.xml on
both hosts). Same wire as /camera/detections, which has run reliably for months.

DEDICATED NODE + EXECUTOR + WARMUP
  Deliberately mirrors MemoryClient, for the same two hard-won reasons:
    1. Sharing behavior_node's node/executor made service responses intermittently
       drop under cross-thread contention.
    2. A freshly created service client matches its REQUEST path before its REPLY
       path. Until the reply path is up the server's response is silently dropped
       and the first call times out — so the paths are warmed in the background at
       construction, long before anyone asks OMNI what it can see.

GRACEFUL DEGRADATION
  get_frame() returns None rather than raising when the Jetson is unreachable. The
  caller turns that into something OMNI can say out loud.

  The GetCameraFrame import is LAZY (inside __init__, not at module scope) and on
  purpose. Imported at module scope it runs during `import behavior_node.behavior_node`,
  before main() executes a single line — so a missing or unbuilt omni_vision_msgs
  took down the ENTIRE robot with a ModuleNotFoundError, not just scene description.
  Observed 2026-07-18: a shell sourced before omni_vision_msgs was built had a stale
  AMENT_PREFIX_PATH and behavior_node refused to start at all. Vision is an optional
  capability; losing it must cost one tool, not the brain. Do not hoist this back up.
"""
from __future__ import annotations

import threading
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor

SERVICE_NAME = '/vision/get_camera_frame'


class FrameResult:
    """What get_frame() gives back. jpeg is None on failure; message says why.

    `views` is the full reply: [(camera_id, jpeg), ...] in server order, head
    first. For a single-camera request it holds exactly one entry and `jpeg`
    mirrors it, so existing single-frame callers are unaffected.
    """

    __slots__ = ('jpeg', 'age_seconds', 'message', 'ok', 'views')

    def __init__(self, jpeg=None, age_seconds=0.0, message='', ok=False, views=None):
        self.jpeg = jpeg
        self.age_seconds = age_seconds
        self.message = message
        self.ok = ok
        self.views = views or []

    @property
    def camera_ids(self):
        return [cid for cid, _ in self.views]

    def __repr__(self):
        n = len(self.jpeg) if self.jpeg else 0
        return (f'FrameResult(ok={self.ok}, {n}B, age={self.age_seconds:.2f}s, '
                f'views={self.camera_ids}, {self.message!r})')


class FrameClient:
    def __init__(self, parent_node, *, enabled: bool = True, service_timeout: float = 2.5):
        """
        service_timeout — budget for one frame fetch. Kept tight: this sits inside
                          a spoken exchange with a ~3s target, and the vision call
                          still has to happen afterwards. Better to admit we cannot
                          see than to leave a silence.
        """
        self._log = parent_node.get_logger()
        self.enabled = enabled
        self._timeout = service_timeout

        # Set before any early return so shutdown()/get_frame() are always safe.
        self._srv = None
        self._node = None
        self._cli = None
        self._exec = None
        self._spin_thread = None
        self._lock = threading.Lock()
        self._warmed = threading.Event()

        # LAZY IMPORT — see the module docstring. A missing omni_vision_msgs
        # disables scene description; it must never stop OMNI from starting.
        try:
            from omni_vision_msgs.srv import GetCameraFrame
        except ImportError as exc:
            self._log.warn(
                f'frame_server: omni_vision_msgs unavailable ({exc}) — scene '
                f'description disabled. Build it and re-source the workspace '
                f'(a shell sourced before the package was built will not see it).'
            )
            self.enabled = False
            self._warmed.set()   # release anyone waiting on warmup
            return
        self._srv = GetCameraFrame

        self._node = rclpy.create_node('behavior_frame_client')
        self._cli = self._node.create_client(GetCameraFrame, SERVICE_NAME)
        self._exec = SingleThreadedExecutor()
        self._exec.add_node(self._node)
        self._spin_thread = threading.Thread(
            target=self._exec.spin, name='frame-exec', daemon=True)
        self._spin_thread.start()

        if self.enabled:
            threading.Thread(target=self._warmup, daemon=True).start()

    def _warmup(self) -> None:
        try:
            # 30s: the Jetson may still be booting when the Pi's behavior_node comes up.
            if self._cli.wait_for_service(timeout_sec=30.0):
                # Passive settle so the reply path finishes matching. An active
                # warmup call that times out on the cold path can wedge the client
                # instead of helping — do not "improve" this into a real request.
                time.sleep(2.0)
                self._log.info('frame_server: service connection warmed up')
            else:
                self._log.warn(
                    f'frame_server: {SERVICE_NAME} not seen within 30s — '
                    f'scene description will be unavailable until the Jetson is up')
        except Exception:  # noqa: BLE001 - warmup is best-effort
            pass
        finally:
            self._warmed.set()

    def shutdown(self) -> None:
        """Tear down in strict order: stop the executor, WAIT for the spin thread
        to actually exit, then destroy the node.

        The join is load-bearing. executor.shutdown() only asks the spin loop to
        stop; it returns immediately. Destroying the node while that thread is
        still inside spin() aborts the process with 'terminate called without an
        active exception' (a C++ thread destroyed while joinable). Measured 3 in 6
        runs before the join was added.
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

    def _ready(self) -> bool:
        """Poll the non-blocking readiness flag (our own executor keeps the graph
        current, so this avoids a guard-condition wait)."""
        deadline = time.monotonic() + self._timeout
        while not self._cli.service_is_ready():
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)
        return True

    def get_frame(self, camera_id: str = 'head') -> FrameResult:
        """Fetch the newest JPEG from the Jetson. Never raises."""
        if not self.enabled:
            return FrameResult(message='frame client is disabled')

        self._warmed.wait(self._timeout)

        if not self._ready():
            self._log.warn('frame_server: service unavailable')
            return FrameResult(
                message='the frame service is not reachable — is the Jetson up?')

        req = self._srv.Request(camera_id=camera_id)
        started = time.monotonic()
        with self._lock:
            future = self._cli.call_async(req)

        done = threading.Event()
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(self._timeout):
            self._log.warn(f'frame_server: request timed out after {self._timeout}s')
            return FrameResult(message='the camera did not respond in time')

        try:
            resp = future.result()
        except Exception as exc:  # noqa: BLE001
            self._log.warn(f'frame_server: request failed: {exc}')
            return FrameResult(message='the camera request failed')

        if resp is None or not resp.success:
            msg = getattr(resp, 'message', 'no response')
            self._log.warn(f'frame_server: unsuccessful: {msg}')
            return FrameResult(message=msg)

        # resp.jpeg / frame.jpeg are uint8[] — array('B') or bytes depending on
        # the rmw path, so bytes() is not redundant.
        #
        # getattr on `frames`, not resp.frames directly: the Jetson may still be
        # running a frame_server built before multi-camera existed, in which case
        # the reply has only the legacy fields. Falling back to them keeps 'front'
        # working against an old server instead of failing with AttributeError —
        # the two machines are built separately and do drift apart.
        views = []
        for frame in getattr(resp, 'frames', None) or []:
            views.append((frame.camera_id, bytes(frame.jpeg)))

        jpeg = bytes(resp.jpeg)
        if not views and jpeg:
            views = [(camera_id, jpeg)]
        if not jpeg and views:
            jpeg = views[0][1]

        elapsed = time.monotonic() - started
        summary = ', '.join(f'{cid} {len(data) / 1024:.0f}KB' for cid, data in views)
        self._log.info(
            f'frame_server: got [{summary}] '
            f'(frame age {resp.age_seconds:.2f}s, fetch {elapsed:.2f}s)')
        return FrameResult(
            jpeg=jpeg, age_seconds=float(resp.age_seconds),
            message=resp.message, ok=True, views=views)
