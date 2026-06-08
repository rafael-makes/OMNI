"""
behavior_node.py — OMNI central brain. State machine and ROS2 interface.

This file deliberately contains NO Gemini API code. All Gemini details live in
gemini_bridge.py. All function call logic lives in function_handlers.py. All
audio hardware access lives in audio_handler.py. This file wires them together
and manages the robot's state machine.

STATE MACHINE:
  IDLE       Wake word listening active. Gemini stream closed.
  LISTENING  Wake word heard, Gemini stream open, waiting for user speech.
  SPEAKING   Gemini is generating or playing an audio response.
  NAVIGATING Nav2 driving to a goal. Gemini stream closed.
  EXPLORING  Autonomous exploration mode. Gemini stream closed.
  DOCKING    Reserved for future use. Transitions gracefully, no logic yet.
  ERROR      Safety fault received. Gemini stream open to react in character.

THREAD SAFETY (read this before editing):
  _set_state() is the ONLY place _current_state is written.
  It is protected by _state_lock (threading.Lock).
  Never assign self._current_state = ... anywhere else — that is a bug.

  _on_wake_word() fires on the WakeWordDetector daemon thread.
  It calls bridge.open_session() which internally uses call_soon_threadsafe —
  safe from any thread. Do not call asyncio methods directly from this callback.

  _on_safety_fault() fires on the ROS2 executor thread (normal subscription
  callback). It is safe to call _set_state() and bridge.inject_context() here.

  Function handlers run in the GeminiBridge asyncio thread-pool executor and
  call _set_state() and nav action client methods — all thread-safe.
"""

import os
import threading
import time

import yaml

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Float32MultiArray, String
from vision_msgs.msg import Detection2DArray

from behavior_node.audio_handler import AudioHandler
from behavior_node.function_handlers import FunctionHandlers, VALID_STATES
from behavior_node.gemini_bridge import GeminiBridge
from behavior_node.wake_word import WakeWordDetector

# States that cause the Gemini stream to close (robot is busy, can't converse)
_STATES_CLOSE_STREAM = {'NAVIGATING', 'EXPLORING', 'DOCKING'}


class BehaviorNode(Node):

    def __init__(self):
        super().__init__('behavior_node')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('gemini_model',        'models/gemini-2.5-flash-native-audio-latest')
        self.declare_parameter('gemini_voice',        'Algieba')
        self.declare_parameter('config_file_path',    '~/omni_ws/src/behavior_node/config/omni_config.yaml')
        self.declare_parameter('wake_word_model',              'hey_mycroft')
        self.declare_parameter('wake_word_threshold',          0.5)
        self.declare_parameter('wake_word_startup_suppress',   1.5)
        self.declare_parameter('conversation_timeout', 30.0)
        self.declare_parameter('idle_return_timeout', 30.0)
        self.declare_parameter('presence_timeout',    10.0)
        self.declare_parameter('mic_device_index',    0)
        self.declare_parameter('speaker_device_index', 0)
        self.declare_parameter('tcp_mic_port',        0)

        model           = self.get_parameter('gemini_model').value
        voice           = self.get_parameter('gemini_voice').value
        config_path     = os.path.expanduser(self.get_parameter('config_file_path').value)
        ww_model        = self.get_parameter('wake_word_model').value
        ww_threshold    = self.get_parameter('wake_word_threshold').value
        ww_suppress     = self.get_parameter('wake_word_startup_suppress').value
        mic_dev         = self.get_parameter('mic_device_index').value
        spk_dev         = self.get_parameter('speaker_device_index').value
        tcp_mic_port    = self.get_parameter('tcp_mic_port').value
        self._conv_timeout     = self.get_parameter('conversation_timeout').value
        self._presence_timeout = self.get_parameter('presence_timeout').value

        # ── Gemini API key ─────────────────────────────────────────────────────
        # Loaded from environment — never hardcoded. Set in ~/.bashrc:
        #   export GEMINI_API_KEY=your_key_here
        self._api_key = os.environ.get('GEMINI_API_KEY', '')
        if not self._api_key:
            self.get_logger().error(
                'GEMINI_API_KEY environment variable is not set. '
                'Gemini will not connect. '
                'Fix: add "export GEMINI_API_KEY=your_key_here" to ~/.bashrc, '
                'then source ~/.bashrc or restart the terminal.'
            )

        # ── Config file ────────────────────────────────────────────────────────
        config        = self._load_config(config_path)
        omni_cfg      = config.get('omni', {})
        system_prompt = omni_cfg.get('system_prompt', '')
        self._locations  = omni_cfg.get('locations', {})   # used by navigate_to()
        self._config_path = config_path                    # used by save_location()

        if not system_prompt:
            self.get_logger().warn(
                f'System prompt is empty — check {config_path}. '
                f'OMNI will connect but have no personality instructions.'
            )
        if not self._locations:
            self.get_logger().info(
                'No locations configured in omni_config.yaml. '
                'navigate_to() calls will return an in-character "unknown location" response.'
            )

        # ── State machine ──────────────────────────────────────────────────────
        # _state_lock protects _current_state. ALL writes go through _set_state().
        # Direct assignment to _current_state anywhere else in this file is a bug.
        self._state_lock    = threading.Lock()
        self._current_state = 'IDLE'
        self._prev_state    = 'IDLE'   # used to restore state after ERROR clears

        # ── Cached sensor data ─────────────────────────────────────────────────
        self._battery_pct   = None   # float 0.0–100.0 or None before first message
        self._last_fault    = None   # most recent fault string, or None if clear

        # ── Conversation timeout tracking ──────────────────────────────────────
        # Updated by _reset_conversation_timeout(), called by gemini_bridge on
        # any incoming Gemini message (audio, function calls, server content).
        # CPython's GIL makes float assignment atomic, so no lock needed here.
        self._last_activity_time  = time.monotonic()
        self._state_entered_time  = time.monotonic()  # updated by _set_state()
        self._fault_active        = False              # True from fault until safety clears

        # ── Presence tracking ──────────────────────────────────────────────────
        # _last_person_seen: monotonic timestamp of the most recent camera frame
        #   containing a person. Initialized to now so the first presence_timeout
        #   window starts at boot, not at time zero.
        # _presence_armed: True when the wake word detector should be running.
        #   Both written only under CPython's GIL (atomic assignment) from the
        #   ROS2 executor thread and from _set_state() — same pattern as
        #   _last_activity_time.
        self._last_person_seen = time.monotonic()
        self._presence_armed   = True

        # ── Nav2 action client ─────────────────────────────────────────────────
        # Used by function_handlers._navigate_to() to send goals to bt_navigator.
        # _current_goal_handle is set in _nav_goal_response_callback() and cleared
        # by _nav_cancel_callback() or when navigation completes.
        self._nav_action_client   = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._current_goal_handle = None   # ClientGoalHandle, or None if not navigating

        # ── Publishers ─────────────────────────────────────────────────────────
        self._state_pub  = self.create_publisher(String,             '/robot_state',    10)
        self._speech_pub = self.create_publisher(String,             '/audio/speech',   10)
        self._servo_pub  = self.create_publisher(Float32MultiArray,  '/servo_commands', 10)
        self._levels_pub = self.create_publisher(Float32MultiArray,  '/audio/levels',   10)

        # ── Subscribers ────────────────────────────────────────────────────────
        self.create_subscription(String,       '/safety/fault',   self._on_safety_fault,  10)
        self.create_subscription(String,       '/safety/status',  self._on_safety_status,  10)
        self.create_subscription(BatteryState, '/battery/status', self._on_battery_status, 10)
        self.create_subscription(OccupancyGrid,'/map/coverage',   self._on_map_coverage,  10)
        self.create_subscription(String,       '/map/location',   self._on_map_location,  10)
        self.create_subscription(String,       '/motor_status',   self._on_motor_status,  10)
        self.create_subscription(String,       '/wifi_config',    self._on_wifi_config,   10)
        # camera_node publishes with BEST_EFFORT sensor QoS — subscriber must match
        _sensor_qos = QoSPresetProfiles.SENSOR_DATA.value
        self.create_subscription(
            Detection2DArray, '/camera/detections', self._on_detections, _sensor_qos
        )

        # ── Timers ─────────────────────────────────────────────────────────────
        # 10Hz state publisher — fast enough that any node coming online quickly
        # gets the current state without waiting long.
        self.create_timer(0.1, self._publish_state)
        # Check conversation timeout every 5 seconds. Fine-grained enough at 30s timeout.
        self.create_timer(5.0, self._check_conversation_timeout)
        # State watchdog: recovers from stuck NAVIGATING/SPEAKING/LISTENING states.
        self.create_timer(5.0, self._check_state_watchdog)
        # Check presence timeout every 1 second. Gives ~1s re-arm latency when a
        # person reappears, which is imperceptible in normal use.
        self.create_timer(1.0, self._check_presence_timeout)

        # ── Audio handler ──────────────────────────────────────────────────────
        self._audio = AudioHandler(
            mic_device=mic_dev,
            speaker_device=spk_dev,
            logger=self.get_logger(),
            tcp_mic_port=tcp_mic_port,
            on_levels=self._publish_levels,
        )
        self._audio.start()

        # ── Function handlers ──────────────────────────────────────────────────
        self._fn = FunctionHandlers(self)

        # ── Gemini bridge ──────────────────────────────────────────────────────
        self._bridge = GeminiBridge(
            node=self,
            audio_handler=self._audio,
            function_handlers=self._fn,
            system_prompt=system_prompt,
            model=model,
            voice=voice,
            on_activity=self._reset_conversation_timeout,
        )
        self._bridge.start()

        # ── Wake word detector ─────────────────────────────────────────────────
        self._wake = WakeWordDetector(
            callback=self._on_wake_word,
            mic_device=mic_dev,
            model_name=ww_model,
            score_threshold=ww_threshold,
            logger=self.get_logger(),
            audio_handler=self._audio,
            startup_suppress_secs=ww_suppress,
        )
        self._wake.start()

        self.get_logger().info(
            f'BehaviorNode ready — state=IDLE, wake word model={ww_model}, '
            f'conversation timeout={self._conv_timeout}s'
        )

    # ── State machine ──────────────────────────────────────────────────────────

    def _set_state(self, new_state: str):
        """
        The ONLY method that writes to _current_state.
        Protected by _state_lock so concurrent calls from different threads
        (wake word thread, asyncio executor, ROS2 timer) are safe.

        Side effects (stream close, wake word restart) happen OUTSIDE the lock
        to avoid holding the lock while calling into other subsystems.
        """
        if new_state not in VALID_STATES:
            self.get_logger().warn(
                f'_set_state: ignoring invalid state {new_state!r}. '
                f'Valid states: {sorted(VALID_STATES)}'
            )
            return

        with self._state_lock:
            if new_state == self._current_state:
                return  # no-op: already in this state
            old_state = self._current_state
            # Save previous state so ERROR recovery can return to the right place
            if old_state != 'ERROR':
                self._prev_state = old_state
            self._current_state = new_state
            self._state_entered_time = time.monotonic()

        self.get_logger().info(f'State: {old_state} → {new_state}')

        # Clear the fault record when leaving ERROR so report_status() and any
        # subsequent inject_context calls don't keep referencing a resolved fault.
        if old_state == 'ERROR' and new_state != 'ERROR':
            self._last_fault = None

        # Close the Gemini stream when the robot is busy driving.
        # The bridge handles reconnect logic when we return to IDLE.
        if new_state in _STATES_CLOSE_STREAM:
            self._bridge.close_session()

        # Returning to IDLE: stop mic capture (wake word detector needs exclusive access),
        # wait 100ms for ALSA to fully release device 0, then restart the detector.
        # Order is strict: stop_capture → sleep → wake_word.start().
        # Without the sleep, the detector's sd.InputStream open races the kernel release
        # and gets ALSA error -9985 (Device unavailable).
        if new_state == 'IDLE':
            self._last_activity_time = time.monotonic()
            # Reset presence state so the wake word detector always runs for a
            # full presence_timeout window after returning to IDLE, regardless of
            # how long the robot was away from this state.
            self._last_person_seen = time.monotonic()
            self._presence_armed   = True
            self._audio.stop_capture()
            time.sleep(0.1)
            self._wake.start()

        # DOCKING: reserved for future use. Just transition and log.
        if new_state == 'DOCKING':
            self.get_logger().info(
                'DOCKING state entered — no docking logic implemented yet.'
            )

    # ── Conversation timeout ───────────────────────────────────────────────────

    def _reset_conversation_timeout(self):
        """
        Called by gemini_bridge on every incoming Gemini message.
        Resets the silence clock so the timeout only fires after a genuine
        30-second gap — not mid-function-call or mid-response.
        Float assignment is atomic in CPython (GIL), so no lock needed.
        """
        self._last_activity_time = time.monotonic()

    def _check_conversation_timeout(self):
        """
        ROS2 timer callback — fires every 5 seconds.
        If the Gemini session has been open but silent for longer than
        conversation_timeout seconds, close the stream and return to IDLE
        so wake word detection resumes.
        Only active during LISTENING or SPEAKING — ignore in all other states.
        """
        with self._state_lock:
            state = self._current_state

        if state not in ('LISTENING', 'SPEAKING'):
            return  # not in a conversation — nothing to time out

        elapsed = time.monotonic() - self._last_activity_time
        if elapsed >= self._conv_timeout:
            self.get_logger().info(
                f'Conversation timeout after {elapsed:.1f}s of silence — '
                f'returning to IDLE and restarting wake word detection'
            )
            self._bridge.close_session()
            self._set_state('IDLE')

    def _check_state_watchdog(self):
        """
        Fires every 5 seconds. Recovers from states that should not last forever:
          NAVIGATING — Nav2 goal callback failed to fire; cancel and return to IDLE.
          SPEAKING   — Gemini stream died without closing cleanly; return to IDLE.
        LISTENING is already covered by _check_conversation_timeout.
        """
        with self._state_lock:
            state   = self._current_state
            entered = self._state_entered_time

        elapsed = time.monotonic() - entered

        if state == 'NAVIGATING' and elapsed > 120.0:
            # Navigation taking more than 2 minutes with no state transition —
            # goal callbacks are likely lost. Cancel and recover.
            self.get_logger().warn(
                f'Watchdog: stuck in NAVIGATING for {elapsed:.0f}s — cancelling and returning to IDLE'
            )
            if self._current_goal_handle:
                self._current_goal_handle.cancel_goal_async()
                self._current_goal_handle = None
            self._set_state('IDLE')
            self._wake.start()

        elif state == 'SPEAKING' and elapsed > 60.0:
            # Speaking for more than 60s — Gemini stream likely died.
            self.get_logger().warn(
                f'Watchdog: stuck in SPEAKING for {elapsed:.0f}s — closing session and returning to IDLE'
            )
            self._bridge.close_session()
            self._audio.stop_capture()
            self._set_state('IDLE')
            self._wake.start()

    # ── Wake word callback ─────────────────────────────────────────────────────

    def _on_wake_word(self):
        """
        Called by WakeWordDetector on its daemon thread when the wake word fires.
        The detector has already set _running=False and closed its ALSA stream
        before calling this — so the mic device is now free for audio_handler.

        Must NOT call bridge methods directly from this thread — that would put
        asyncio operations on a non-asyncio thread, which is unpredictably unsafe.
        bridge.open_session() already uses loop.call_soon_threadsafe() internally,
        so calling it here is safe: it schedules the session open on the asyncio
        loop rather than executing it on this thread.
        """
        self.get_logger().info('Wake word detected — opening Gemini session')
        self._set_state('LISTENING')
        self._reset_conversation_timeout()
        # Detector has released the mic — start capture before opening the session
        self._audio.start_capture()
        # open_session() uses call_soon_threadsafe internally — safe from any thread
        self._bridge.open_session()

    # ── Safety fault callbacks ─────────────────────────────────────────────────

    def _on_safety_fault(self, msg: String):
        """
        Fires on the ROS2 executor thread when /safety/fault receives a message.
        Transitions to ERROR and injects a [SYSTEM ALERT] into the Gemini session
        so OMNI can react in character. inject_context() is thread-safe.
        """
        fault_text = msg.data
        self._last_fault   = fault_text
        self._fault_active = True
        self.get_logger().error(f'Safety fault received: {fault_text}')

        # Snapshot state BEFORE calling _set_state so we know what was happening
        with self._state_lock:
            state_before_fault = self._current_state

        self._set_state('ERROR')

        # If we were IDLE, the wake word detector owns the mic.
        # We must stop it and hand the mic to audio_handler before opening the session.
        # If we were already in LISTENING/SPEAKING, capture is already running — skip.
        if state_before_fault == 'IDLE':
            self._wake.stop()
            time.sleep(0.1)             # 100ms for ALSA to release device 0
            self._audio.start_capture() # now safe to open the mic

        # Pass the fault as initial_prompt so it is the very first thing Gemini
        # sees when the session opens — no race with inject_context().
        # [SYSTEM ALERT] prefix triggers the system-prompt urgency rule.
        self._bridge.open_session(
            initial_prompt=(
                f'[SYSTEM ALERT] Safety fault detected: {fault_text}. '
                f'Announce this fault immediately, in character — alarmed, urgent, '
                f'C-3PO-ish. Do not wait for the user to ask.'
            )
        )

    def _on_safety_status(self, msg: String):
        """
        Fires at 1 Hz from safety_node. If safety returns to OK while behavior_node
        is stuck in a fault-related state (ERROR or SPEAKING from fault announcement),
        auto-recover to IDLE so OMNI responds to the wake word again.
        """
        if msg.data.startswith('OK') and self._fault_active:
            if self._current_state in ('ERROR', 'SPEAKING', 'LISTENING'):
                self.get_logger().info(
                    f'Safety cleared — auto-recovering from {self._current_state} to IDLE'
                )
                self._bridge.close_session()
                self._audio.stop_capture()
                self._fault_active = False
                self._last_fault   = None
                self._set_state('IDLE')
                self._wake.start()

    # ── Sensor subscribers ─────────────────────────────────────────────────────

    def _on_battery_status(self, msg: BatteryState):
        """Cache battery percentage for report_status() function handler."""
        # BatteryState.percentage is 0.0–1.0 in ROS2 — convert to 0–100 for readability
        self._battery_pct = msg.percentage * 100.0

    def _on_map_coverage(self, msg: OccupancyGrid):
        # Not yet used — reserved for explore_area() reporting when frontier
        # exploration is implemented. Store for future use.
        pass

    def _on_map_location(self, msg: String):
        # Semantic location string (e.g. "kitchen") — could be injected into
        # Gemini context in a future session. Log for now.
        self.get_logger().debug(f'Map location update: {msg.data}')

    def _on_motor_status(self, msg: String):
        import json
        raw = msg.data.strip()
        if not raw:
            return
        try:
            # motor_control_node publishes JSON odometry telemetry at high frequency.
            # Parse it and log at DEBUG so it never spams the console.
            json.loads(raw)
            self.get_logger().debug(f'Motor telemetry: {raw}')
        except json.JSONDecodeError:
            # Non-JSON messages are human-readable status strings.
            # Only warn if they are not one of the known-nominal values.
            if raw.lower() not in ('ok', 'nominal'):
                self.get_logger().warn(f'Motor status: {raw}')

    def _on_wifi_config(self, msg: String):
        # Configuration updates from chest_node — log only, no behavior yet
        self.get_logger().info(f'WiFi config update received: {msg.data}')

    def _on_detections(self, msg: Detection2DArray):
        """
        Update presence timestamp whenever a person appears in camera frame.
        Intentionally lightweight — all arm/disarm decisions happen in the
        _check_presence_timeout timer so this callback never sleeps.
        """
        if any(
            result.hypothesis.class_id.lower() == 'person'
            for det in msg.detections
            for result in det.results
        ):
            self._last_person_seen = time.monotonic()

    def _check_presence_timeout(self):
        """
        1Hz timer — manages wake word arm/disarm based on person presence.
        Only acts in IDLE; never interrupts an active conversation or navigation.

        Armed → disarmed: no person detected for presence_timeout seconds.
        Disarmed → armed: person seen more recently than presence_timeout ago.

        The 200ms sleep before _wake.start() in the re-arm path gives the
        previously stopped detector thread time to fully release the ALSA device
        (thread exits within one 80ms chunk after _running is set False).
        """
        with self._state_lock:
            state = self._current_state
        if state != 'IDLE':
            return

        elapsed        = time.monotonic() - self._last_person_seen
        person_present = elapsed < self._presence_timeout

        if person_present and not self._presence_armed:
            self.get_logger().info(
                f'Person detected — re-arming wake word detector '
                f'({elapsed:.1f}s since last detection)'
            )
            self._presence_armed = True
            time.sleep(0.2)
            self._wake.start()

        elif not person_present and self._presence_armed:
            self.get_logger().info(
                f'No person detected for {elapsed:.1f}s — '
                f'disarming wake word detector '
                f'(presence_timeout={self._presence_timeout}s)'
            )
            self._presence_armed = False
            self._wake.stop()

    # ── Nav2 action callbacks ──────────────────────────────────────────────────

    def _nav_goal_response_callback(self, future):
        """
        Called by the Nav2 action client when the action server accepts or
        rejects our goal. Stores the goal handle so stop_navigation() can
        cancel it later.
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn(
                'Nav2 rejected the navigation goal. '
                'Nav2 may not be ready or the goal pose is invalid.'
            )
            self._set_state('IDLE')
            return

        self.get_logger().info('Nav2 accepted navigation goal')
        self._current_goal_handle = goal_handle

        # Request the result future so we know when navigation completes
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._nav_result_callback)

    def _nav_result_callback(self, future):
        """Called when the navigation goal finishes (success, abort, or cancel)."""
        result   = future.result()
        status   = result.status
        # Nav2 status codes: 4=SUCCEEDED, 5=CANCELED, 6=ABORTED
        status_names = {4: 'SUCCEEDED', 5: 'CANCELED', 6: 'ABORTED'}
        status_str = status_names.get(status, f'UNKNOWN({status})')
        self.get_logger().info(f'Navigation finished — status: {status_str}')
        self._current_goal_handle = None
        # Return to IDLE regardless of outcome so wake word resumes
        self._set_state('IDLE')

    def _nav_feedback_callback(self, feedback_msg):
        """Receives periodic distance-remaining feedback from bt_navigator."""
        fb = feedback_msg.feedback
        self.get_logger().debug(
            f'Nav feedback — distance remaining: {fb.distance_remaining:.2f}m'
        )

    def _nav_cancel_callback(self, future):
        """Called when a cancel request completes."""
        self.get_logger().info('Nav2 goal cancellation acknowledged')
        self._current_goal_handle = None

    # ── Nav2 readiness check ───────────────────────────────────────────────────

    def nav_is_ready(self) -> bool:
        """
        Non-blocking check: returns True if the Nav2 action server is available.
        Uses a 1-second timeout so it does not hang forever if Nav2 is not running.
        Called by function_handlers._navigate_to() before sending a goal.
        Safe to call from a thread-pool worker (run_in_executor context).
        """
        ready = self._nav_action_client.wait_for_server(timeout_sec=1.0)
        if not ready:
            self.get_logger().warn(
                'Nav2 action server not available after 1s. '
                'Is nav_node running? Start it with: ros2 launch nav_node nav_launch.py'
            )
        return ready

    # ── Publishers ─────────────────────────────────────────────────────────────

    def _publish_state(self):
        """Timer callback — publishes current state at 10Hz."""
        with self._state_lock:
            state = self._current_state
        msg      = String()
        msg.data = state
        self._state_pub.publish(msg)

    def _publish_levels(self, levels: list):
        """Called from audio-play thread; publish amplitude bands for chest LED matrix."""
        msg      = Float32MultiArray()
        msg.data = levels
        self._levels_pub.publish(msg)

    # ── Config loader ──────────────────────────────────────────────────────────

    def _load_config(self, path: str) -> dict:
        """
        Load and return the YAML config dict.
        Returns an empty dict on failure so the node starts anyway
        (it will have no personality but will not crash).
        """
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
            self.get_logger().info(f'Loaded config from {path}')
            return config or {}
        except FileNotFoundError:
            self.get_logger().error(
                f'Config file not found: {path}. '
                f'OMNI will start without a system prompt or location data.'
            )
            return {}
        except yaml.YAMLError as exc:
            self.get_logger().error(
                f'YAML parse error in {path}: {exc}. '
                f'Check indentation — YAML is strict about spaces.'
            )
            return {}


def main(args=None):
    rclpy.init(args=args)
    node = BehaviorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._audio.stop()
        node._bridge.stop()
        node._wake.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
