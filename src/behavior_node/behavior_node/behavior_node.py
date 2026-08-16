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

import json
import math
import os
import threading
import time
import uuid

import yaml

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSPresetProfiles, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Bool, Float32MultiArray, Float64, String
from std_srvs.srv import Trigger
from vision_msgs.msg import Detection2DArray

from datetime import datetime, time as dt_time

from omni_zones import load_zone_map

from behavior_node.audio_handler import AudioHandler
from behavior_node.check_in import CheckInBehavior
from behavior_node.check_in_policy import (
    OUTCOME_NOT_NOW,
    OUTCOME_NO_RESPONSE,
    OUTCOME_YES,
    CheckInConfig,
    CheckInPolicy,
)
from behavior_node.function_handlers import FunctionHandlers, VALID_STATES
from behavior_node.gemini_bridge import GeminiBridge
from behavior_node.frame_client import FrameClient
from behavior_node.greeting_decider import GreetingDecider
from behavior_node.memory_client import MemoryClient
from behavior_node.memory_format import wrap_memory_context
from behavior_node.scene_describer import SceneDescriber
from behavior_node.suppression import RobotStatus, interaction_blocked
from behavior_node.wake_word import WakeWordDetector

# States that cause the Gemini stream to close (robot is busy, can't converse)
_STATES_CLOSE_STREAM = {'NAVIGATING', 'EXPLORING', 'DOCKING'}


def _split_csv(raw) -> list:
    """"a, b" -> ['a', 'b']; "" -> []. Also accepts a real list, so a params file
    supplying a proper YAML sequence still works — only the *default* has to be a
    string (see check_in_zones for why)."""
    if isinstance(raw, (list, tuple)):
        return [str(item).strip() for item in raw if str(item).strip()]
    return [part.strip() for part in str(raw or '').split(',') if part.strip()]


def _parse_clock(text: str, *, default: dt_time) -> dt_time:
    """Parse an "HH:MM" parameter into a time, falling back to `default`.

    Never raises: a typo in quiet_hours must not stop the robot from booting. A
    bad value falls back to the safe default (quiet hours still enforced) rather
    than to "no quiet hours at all", because the failure that matters here is
    OMNI talking at midnight.
    """
    try:
        hh, _, mm = str(text).strip().partition(':')
        return dt_time(int(hh), int(mm or 0))
    except (ValueError, TypeError):
        return default


class BehaviorNode(Node):

    def __init__(self):
        super().__init__('behavior_node')

        # ── Parameters ────────────────────────────────────────────────────────
        # NOTE: native-audio models (gemini-2.5-flash-native-audio-*) reject `tools`
        # in the Live API and drop the session with WS 1008 mid-turn. behavior_node
        # needs function calling, so it must use a tool-capable Live model.
        self.declare_parameter('gemini_model',        'models/gemini-3.1-flash-live-preview')
        self.declare_parameter('gemini_voice',        'Algieba')
        self.declare_parameter('config_file_path',    '~/omni_ws/src/behavior_node/config/omni_config.yaml')
        self.declare_parameter('wake_word_model',              'hey_mycroft')
        self.declare_parameter('wake_word_threshold',          0.5)
        self.declare_parameter('wake_word_startup_suppress',   1.5)
        self.declare_parameter('conversation_timeout', 30.0)
        self.declare_parameter('idle_return_timeout', 30.0)
        self.declare_parameter('presence_timeout',    10.0)
        # Accept either a numeric sounddevice index OR a case-insensitive name
        # substring (e.g. 'reSpeaker', 'PnP'). Names survive USB index reshuffles
        # across reboots/replugs; see _resolve_audio_device().
        self.declare_parameter('mic_device_index',    '0')
        self.declare_parameter('speaker_device_index', '0')
        self.declare_parameter('tcp_mic_port',        0)
        # ── Persistent memory (Step 5) — soft dependency on the omni_memory node ─
        self.declare_parameter('memory_enabled',         True)
        self.declare_parameter('memory_retrieve_k',      5)
        self.declare_parameter('memory_service_timeout', 2.0)
        self.declare_parameter(
            'memory_seed_query',
            'important preferences, facts, habits, and recent events about the people here',
        )
        # ── Per-person memory keying (Step 6) ────────────────────────────────────
        # Identity of the person OMNI is talking to comes from the Jetson recognizer
        # on /camera/identity (a resolved name or stable 'unknown_N', '' = none).
        self.declare_parameter('person_keying_enabled', True)
        self.declare_parameter('identity_topic',        '/camera/identity')
        self.declare_parameter('identity_timeout',      5.0)   # seconds an id stays "fresh"

        # ── Scene description — soft dependency on the Jetson's frame_server ─────
        # describe_scene() fetches a JPEG from the Orin and sends it to the Gemini
        # vision endpoint. Whole path must fit inside a spoken exchange, hence the
        # tight service timeout; if the Jetson is down OMNI just says it cannot see.
        self.declare_parameter('scene_enabled',         True)
        self.declare_parameter('scene_service_timeout', 2.5)
        self.declare_parameter('scene_camera_id',       'head')
        # Vision model. NOT the Live model above — this is a one-shot generateContent
        # call. flash-lite with thinking disabled is the fastest option that still
        # returns a complete sentence (see scene_describer for why that matters).
        self.declare_parameter('scene_model',           'gemini-3.1-flash-lite')
        self.declare_parameter('scene_max_sentences',   2)
        # '' -> config/scene_prompt.txt from this package's share dir.
        self.declare_parameter('scene_prompt_path',     '')
        # Rear-view and both-views prompts, same convention. Separate files rather
        # than one prompt with a direction hint: the orientation rules genuinely
        # differ ("behind me" vs "in front of me" vs "one room, two halves"), and
        # a single prompt trying to cover all three is what makes OMNI say it can
        # see something in front of it that is actually behind it.
        self.declare_parameter('scene_rear_prompt_path',   '')
        self.declare_parameter('scene_fusion_prompt_path', '')

        # ── Unprompted greetings (Session 10) ────────────────────────────────────
        # Driven by /omni/events from event_generator, which has already done all
        # the presence debouncing — by the time person_appeared arrives, the person
        # genuinely went away and genuinely came back. Everything here is about
        # whether a greeting is APPROPRIATE, not whether they are really there.
        self.declare_parameter('greeting_enabled',   True)
        self.declare_parameter('greeting_events_topic', '/omni/events')
        # Per-person cooldown, enforced in code and never in the prompt: a rule the
        # model can be talked out of is not a rule. 10 minutes.
        self.declare_parameter('greeting_cooldown',  600.0)
        # Below this battery percentage OMNI has better things to spend power on
        # than saying hello. bms_node also publishes /bms/low_battery at 20%.
        self.declare_parameter('greeting_min_battery', 20.0)
        self.declare_parameter('greeting_model',     'gemini-3.1-flash-lite')
        # '' -> config/greeting_prompt.txt from this package's share dir.
        self.declare_parameter('greeting_prompt_path', '')

        # ── Person zones + go_to_person (Session 7) ──────────────────────────────
        # navigate_to() resolves zone anchors too, and go_to_person() reads who is
        # in which room from world_state. Zones are the SHARED omni_zones config —
        # the SAME file world_state loads. '' -> omni_zones's shipped default.
        self.declare_parameter('zones_config_path',  '')
        self.declare_parameter('world_state_topic',  '/omni/world_state')
        # Beyond this age, a last-known person location is reported as uncertain
        # rather than driven to — "go to Rafael" then honestly says it's not sure.
        self.declare_parameter('person_location_stale_after', 120.0)
        # How far short of a person's estimated spot to stop, metres.
        self.declare_parameter('person_standoff_distance',    1.0)

        # ── Proactive check-in (Session 9) ───────────────────────────────────────
        # Driven by person_dwelling on /omni/events. The DECISION lives in
        # check_in_policy (ROS-free, exhaustively tested); these are its knobs,
        # plus the geometry and timeouts of the drive-over itself.
        #
        # The bar is deliberately high. Interrupting focus is how this feature
        # gets turned off — permanently, by the person it was built for.
        self.declare_parameter('check_in_enabled',       True)
        # Seconds at one spot before a check-in is even considered. 60 minutes.
        self.declare_parameter('check_in_min_dwell',     3600.0)
        # A check-in costs a round trip across the room; the greeting floor (20%)
        # is far too generous for that.
        self.declare_parameter('check_in_min_battery',   40.0)
        # Quiet hours, local time, "HH:MM". The child sleeps.
        self.declare_parameter('check_in_quiet_start',   '21:00')
        self.declare_parameter('check_in_quiet_end',     '08:00')
        # Cooldowns, seconds. Global = after ANY interaction with that person.
        # The per-zone ones are what make "no" and "not now" mean different things.
        self.declare_parameter('check_in_global_cooldown',  7200.0)    # 2 h
        self.declare_parameter('check_in_no_cooldown',      14400.0)   # 4 h
        self.declare_parameter('check_in_not_now_cooldown', 3600.0)    # 1 h
        # Zones a check-in may happen in, COMMA-SEPARATED ("workbench,computer").
        # Empty = trust event_generator's own dwell_zones (this is a second,
        # independent belt). A string rather than a list because an empty list
        # default cannot be typed in rclpy — declare_parameter overwrites the
        # descriptor type from the value, an empty list infers as BYTE_ARRAY, and
        # any override is then rejected as "expecting type BYTE_ARRAY".
        self.declare_parameter('check_in_zones',         '')
        # v1.5 learning: stretch the dwell threshold in zones where declines
        # dominate, up to this multiple of check_in_min_dwell.
        self.declare_parameter('check_in_bias_enabled',      True)
        self.declare_parameter('check_in_bias_min_samples',  3)
        self.declare_parameter('check_in_bias_max_multiplier', 2.0)
        # Approach geometry. Stand BESIDE them, not in front — blocking someone's
        # bench to ask if they need help is its own answer.
        self.declare_parameter('check_in_standoff_distance', 1.0)
        self.declare_parameter('check_in_lateral_offset',    0.6)
        # Which side to arrive on when zones.yaml does not say. See the
        # check_in_side note in omni_zones: world_state has no facing estimate,
        # so v1 encodes a fixed side per zone in config.
        self.declare_parameter('check_in_default_side',      'left')
        # Seconds of silence after the opener before it is read as "not now" and
        # OMNI quietly leaves. Never re-asks.
        self.declare_parameter('check_in_silence_timeout',   15.0)
        # Hard ceiling on the whole behaviour, so a wedged state can never strand
        # OMNI mid-room believing it is still checking in.
        self.declare_parameter('check_in_max_duration',      300.0)

        model           = self.get_parameter('gemini_model').value
        voice           = self.get_parameter('gemini_voice').value
        config_path     = os.path.expanduser(self.get_parameter('config_file_path').value)
        ww_model        = self.get_parameter('wake_word_model').value
        ww_threshold    = self.get_parameter('wake_word_threshold').value
        ww_suppress     = self.get_parameter('wake_word_startup_suppress').value
        mic_dev         = self._resolve_audio_device(
            self.get_parameter('mic_device_index').value, want_input=True)
        spk_dev         = self._resolve_audio_device(
            self.get_parameter('speaker_device_index').value, want_input=False)
        tcp_mic_port    = self.get_parameter('tcp_mic_port').value
        self._conv_timeout     = self.get_parameter('conversation_timeout').value
        self._presence_timeout = self.get_parameter('presence_timeout').value
        self._memory_enabled     = self.get_parameter('memory_enabled').value
        self._memory_k           = int(self.get_parameter('memory_retrieve_k').value)
        self._memory_seed_query  = self.get_parameter('memory_seed_query').value
        memory_timeout           = self.get_parameter('memory_service_timeout').value
        self._person_keying      = self.get_parameter('person_keying_enabled').value
        identity_topic           = self.get_parameter('identity_topic').value
        self._identity_timeout   = self.get_parameter('identity_timeout').value

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

        # ── Zones (Session 7) — named rooms as map-frame polygons ────────────────
        # Shared with world_state via the omni_zones library. navigate_to()
        # resolves a zone's anchor as a destination, and go_to_person() maps a
        # person's zone to a goal. Empty map (the shipped default) is fine: it
        # just means navigate_to falls back to point locations and go_to_person
        # honestly reports it cannot place anyone.
        self._zones = self._load_zones()
        # Latest /omni/world_state snapshot (parsed JSON) + when it arrived.
        # go_to_person reads it; None until the first message or if world_state
        # is not running.
        self._world_state       = None
        self._world_state_time  = 0.0
        self._person_stale_after = float(
            self.get_parameter('person_location_stale_after').value)
        self._person_standoff = float(
            self.get_parameter('person_standoff_distance').value)

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
        # bms_node latches this True below 20% SOC. Note it never sets
        # BatteryState.power_supply_status (verified in bms_node.py — it builds the
        # message field by field and that one is not among them), so "is OMNI on
        # the charger" is NOT knowable from /battery/status. See _greeting_blocked.
        self._low_battery   = False

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
        self._nav_intent          = None   # (kind, label) of the current drive, or None
        self._nav_started         = None   # monotonic start time of the current drive

        # ── Publishers ─────────────────────────────────────────────────────────
        self._state_pub       = self.create_publisher(String,             '/robot_state',         10)
        self._speech_pub      = self.create_publisher(String,             '/audio/speech',        10)
        self._servo_pub       = self.create_publisher(Float32MultiArray,  '/servo_commands',      10)
        self._levels_pub      = self.create_publisher(Float32MultiArray,  '/audio/levels',        10)
        self._clear_fault_pub = self.create_publisher(String,             '/safety/clear_fault',  10)
        # Check-in state transitions go back onto the same semantic event bus
        # event_generator publishes on, so anything watching OMNI's behaviour
        # (chest LEDs, logging, a future dashboard) sees them without a new
        # topic. Reliable depth 10, matching event_generator's publisher — these
        # are rare and meaningful. Our own callback ignores kinds it does not
        # handle, so publishing here cannot feed back into the greeting path.
        self._events_pub      = self.create_publisher(String,             '/omni/events',         10)

        # ── Subscribers ────────────────────────────────────────────────────────
        self.create_subscription(String,       '/safety/fault',   self._on_safety_fault,  10)
        self.create_subscription(String,       '/safety/status',  self._on_safety_status,  10)
        self.create_subscription(BatteryState, '/battery/status', self._on_battery_status, 10)
        self.create_subscription(OccupancyGrid,'/map/coverage',   self._on_map_coverage,  10)
        self.create_subscription(String,       '/map/location',   self._on_map_location,  10)
        self.create_subscription(String,       '/motor_status',   self._on_motor_status,  10)
        self.create_subscription(String,       '/wifi_config',    self._on_wifi_config,   10)
        self.create_subscription(String,       '/audio/say',      self._on_say,           10)
        self.create_subscription(Bool,         '/bms/low_battery', self._on_low_battery,  10)
        # camera_node publishes with BEST_EFFORT sensor QoS — subscriber must match
        _sensor_qos = QoSPresetProfiles.SENSOR_DATA.value
        self.create_subscription(
            Detection2DArray, '/camera/detections', self._on_detections, _sensor_qos
        )
        # world_state (Session 7): who is in which room. Cached for go_to_person.
        self.create_subscription(
            String, self.get_parameter('world_state_topic').value,
            self._on_world_state, 10,
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
        # Drives the check-in mission's timeouts: the 15s silence after the
        # opener, "they got up mid-approach", and the hard duration ceiling. 1 Hz
        # so the silence timeout is accurate to a second — it is short, and
        # loitering after being ignored is precisely what it exists to prevent.
        self.create_timer(1.0, self._tick_check_in)

        # ── Audio handler ──────────────────────────────────────────────────────
        self._audio = AudioHandler(
            mic_device=mic_dev,
            speaker_device=spk_dev,
            logger=self.get_logger(),
            tcp_mic_port=tcp_mic_port,
            on_levels=self._publish_levels,
        )
        self._audio.start()

        # ── Persistent memory client (Step 5) ───────────────────────────────────
        # Soft dependency: if the omni_memory node is down, retrieve()/store() are
        # no-ops and OMNI converses normally. _session_id groups one conversation's
        # memories; assigned per wake-word event.
        self._memory = MemoryClient(
            self, enabled=self._memory_enabled, service_timeout=memory_timeout
        )
        self._session_id = None

        # ── Scene description client + describer ────────────────────────────────
        # Both are soft: a missing Jetson or a missing API key degrades to OMNI
        # saying it cannot see, never to a crash or a silence.
        self._scene_enabled = bool(self.get_parameter('scene_enabled').value)
        self._scene_camera_id = str(self.get_parameter('scene_camera_id').value)
        self._frames = FrameClient(
            self,
            enabled=self._scene_enabled,
            service_timeout=float(self.get_parameter('scene_service_timeout').value),
        )
        self._scene = None
        if self._scene_enabled:
            share_cfg = os.path.join(
                get_package_share_directory('behavior_node'), 'config')

            def _prompt_path(param: str, filename: str) -> str:
                path = str(self.get_parameter(param).value)
                if not path:
                    path = os.path.join(share_cfg, filename)
                if not os.path.exists(path):
                    self.get_logger().warn(
                        f'scene: prompt file not found at {path} — using the '
                        f'built-in default prompt')
                return path

            prompt_path = _prompt_path('scene_prompt_path', 'scene_prompt.txt')
            rear_prompt_path = _prompt_path(
                'scene_rear_prompt_path', 'scene_rear_prompt.txt')
            fusion_prompt_path = _prompt_path(
                'scene_fusion_prompt_path', 'scene_fusion_prompt.txt')

            self._scene = SceneDescriber(
                model=str(self.get_parameter('scene_model').value),
                prompt_path=prompt_path,
                rear_prompt_path=rear_prompt_path,
                fusion_prompt_path=fusion_prompt_path,
                max_sentences=int(self.get_parameter('scene_max_sentences').value),
            )
            self.get_logger().info(
                f'scene description enabled '
                f'(model={self.get_parameter("scene_model").value}, prompt={prompt_path})')

            # Warm the vision endpoint in the background. The first call in a process
            # pays ~3.4s of connection setup vs ~0.7s thereafter, and without this it
            # is the user's first "what do you see?" that pays it. Daemon thread so a
            # slow or failed warmup never delays startup — describe_scene works either
            # way, it is just slower the first time.
            threading.Thread(
                target=self._warm_scene, name='scene-warmup', daemon=True
            ).start()
        # ── Unprompted greetings (Session 10) ────────────────────────────────────
        # Soft, like every other enhancement here: a missing event_generator means
        # no events arrive and OMNI simply never greets anyone unprompted.
        self._greeting_enabled  = bool(self.get_parameter('greeting_enabled').value)
        self._greeting_cooldown = float(self.get_parameter('greeting_cooldown').value)
        self._greeting_min_batt = float(self.get_parameter('greeting_min_battery').value)
        # person -> monotonic timestamp of their last SPOKEN greeting. Guarded by
        # _greeting_lock, which also guards _greeting_in_flight: the event callback
        # is on the ROS executor but the work runs on a daemon thread, so two
        # arrivals in quick succession would otherwise both pass the gate.
        self._greeting_last     = {}
        self._greeting_in_flight = False
        self._greeting_lock     = threading.Lock()
        self._greeter           = None
        if self._greeting_enabled:
            greet_prompt_path = str(self.get_parameter('greeting_prompt_path').value)
            if not greet_prompt_path:
                greet_prompt_path = os.path.join(
                    get_package_share_directory('behavior_node'),
                    'config', 'greeting_prompt.txt')
            if not os.path.exists(greet_prompt_path):
                self.get_logger().warn(
                    f'greeting: prompt file not found at {greet_prompt_path} — '
                    f'using the built-in default prompt')
            self._greeter = GreetingDecider(
                model=str(self.get_parameter('greeting_model').value),
                prompt_path=greet_prompt_path,
            )
            self.create_subscription(
                String,
                str(self.get_parameter('greeting_events_topic').value),
                self._on_presence_event,
                10,
            )
            # Warm the text endpoint for the same reason scene description warms
            # the vision one — except the budget here is tighter, because nobody
            # asked for this greeting and a late one lands after they walked past.
            threading.Thread(
                target=self._warm_greeter, name='greeting-warmup', daemon=True
            ).start()
            self.get_logger().info(
                f'unprompted greetings enabled '
                f'(cooldown={self._greeting_cooldown}s, prompt={greet_prompt_path})')

        # ── Proactive check-in (Session 9) ───────────────────────────────────────
        # There is still no charging signal to read (bms_node never sets
        # power_supply_status), so this stays False until the docking work lands.
        # It is read by robot_status(), so setting it suppresses BOTH greetings and
        # check-ins in one move.
        self._docked = False

        # ── Docking mission (navigate to pre-dock pose → dock_node visual back-in) ──
        # "Go dock yourself" drives to a SAVED pre-dock location (map: gross approach),
        # then hands off to dock_node's tag+ToF back-in (vision: precision). Save the
        # pre-dock pose (rear toward the dock) as `dock_location_name` with save_location.
        self._dock_location_name = self.declare_parameter('dock_location_name', 'dock').value
        self._docking    = False
        self._dock_phase = None            # 'APPROACH' | 'BACKING' | None
        self._dock_nav_retried = False     # one-shot retry of a transient approach abort
        self._dock_lock  = threading.Lock()
        self._dock_start_cli  = self.create_client(Trigger, '/dock/start')
        self._dock_cancel_cli = self.create_client(Trigger, '/dock/cancel')
        self.create_subscription(String, '/dock/result', self._on_dock_result, 10)
        # Standoff heading → dock_node, so it turns to face the tag before searching
        # (Nav2 arrives heading-free). Latched to match dock_node's TRANSIENT_LOCAL sub.
        _dock_latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._dock_orient_pub = self.create_publisher(
            Float64, '/dock/orient_target', _dock_latched)
        # Undock: Nav2 can't plan out of the dock (start pose inside robot_radius of the
        # wall), so drive forward off it first via dock_node, then navigate.
        self._dock_undock_cli = self.create_client(Trigger, '/dock/undock')
        self.create_subscription(String, '/dock/undock_result',
                                 self._on_undock_result, 10)
        # Live docked state from dock_node (rear ToF) — the real "on the dock" signal
        # (see CLAUDE.md "no docked signal exists yet"). Drives self._docked for
        # greeting/check-in suppression AND the undock-before-nav trigger.
        self.create_subscription(Bool, '/dock/docked', self._on_dock_docked, _dock_latched)
        # Foxglove/RViz "2D Goal Pose" publishes /goal_pose, but nothing in this stack
        # converts it to a nav goal (OMNI's nav is action-based via voice). Bridge it
        # here so a clicked goal drives OMNI through the SAME undock-aware path.
        # QoS: BEST_EFFORT so it matches ANY publisher — Foxglove publishes /goal_pose
        # BEST_EFFORT (verified 2026-08-01: a RELIABLE sub silently dropped it, a
        # RELIABLE CLI pub worked), and a BEST_EFFORT sub also accepts a RELIABLE
        # RViz/CLI goal. Depth 1 — only the latest clicked goal matters.
        _goal_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(PoseStamped, '/goal_pose', self._on_goal_pose, _goal_qos)
        # Foxglove's "Publish 2D pose" defaults to /move_base_simple/goal (the ROS1
        # name); RViz uses /goal_pose. Accept BOTH so either tool works out of the box.
        self.create_subscription(
            PoseStamped, '/move_base_simple/goal', self._on_goal_pose, _goal_qos)
        self._undock_lock  = threading.Lock()
        self._undocking    = False
        self._pending_nav  = None      # (x, y, yaw, intent) deferred behind an undock

        self._check_in_enabled = bool(self.get_parameter('check_in_enabled').value)
        # Both stay None when check-ins are disabled; every call site guards on
        # `is not None`, so the feature is genuinely absent rather than inert.
        self._check_in = None
        self._check_in_policy = None
        if self._check_in_enabled:
            self._check_in_policy = CheckInPolicy(
                CheckInConfig(
                    enabled=True,
                    min_dwell=float(self.get_parameter('check_in_min_dwell').value),
                    battery_floor=float(
                        self.get_parameter('check_in_min_battery').value),
                    quiet_start=_parse_clock(
                        str(self.get_parameter('check_in_quiet_start').value),
                        default=dt_time(21, 0)),
                    quiet_end=_parse_clock(
                        str(self.get_parameter('check_in_quiet_end').value),
                        default=dt_time(8, 0)),
                    global_cooldown=float(
                        self.get_parameter('check_in_global_cooldown').value),
                    no_cooldown=float(
                        self.get_parameter('check_in_no_cooldown').value),
                    not_now_cooldown=float(
                        self.get_parameter('check_in_not_now_cooldown').value),
                    bias_enabled=bool(
                        self.get_parameter('check_in_bias_enabled').value),
                    bias_min_samples=int(
                        self.get_parameter('check_in_bias_min_samples').value),
                    bias_max_multiplier=float(
                        self.get_parameter('check_in_bias_max_multiplier').value),
                ),
                zones=_split_csv(self.get_parameter('check_in_zones').value),
            )
            self._check_in = CheckInBehavior(self, self._check_in_policy)
            # Greetings already subscribe to /omni/events; only add a second
            # subscription if greetings are off, so a dwell event is never
            # delivered twice to the same callback.
            if not self._greeting_enabled:
                self.create_subscription(
                    String,
                    str(self.get_parameter('greeting_events_topic').value),
                    self._on_presence_event,
                    10,
                )
            self.get_logger().info(
                f'proactive check-ins enabled '
                f'(min_dwell={self._check_in_policy.config.min_dwell:.0f}s, '
                f'quiet {self._check_in_policy.config.quiet_start}–'
                f'{self._check_in_policy.config.quiet_end})')

        # Step 6: latest recognized identity + when it arrived (monotonic). Written
        # by the /camera/identity callback, read at wake time. _session_person is the
        # person latched for the CURRENT conversation so a mid-chat identity change
        # doesn't re-key the store.
        self._current_person      = None
        self._current_person_time = 0.0
        self._session_person      = None
        self._late_bind_done      = False   # one late identity adoption per chat
        self._enroll_pub          = None
        if self._person_keying:
            self.create_subscription(String, identity_topic, self._on_identity, 10)
            # Step 6 on-the-fly learning: ask the Jetson recognizer to enroll the
            # face it currently sees under a name (published by learn_person()).
            self._enroll_pub = self.create_publisher(String, '/camera/enroll_request', 10)

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

    # ── Audio device resolution ────────────────────────────────────────────────

    def _resolve_audio_device(self, value, want_input: bool):
        """Resolve a device param to a sounddevice index.

        `value` may be a numeric index ('0', 1) or a case-insensitive name
        substring ('reSpeaker', 'PnP'). Names are matched against sounddevice's
        device list, restricted to the needed direction (input vs output), so
        the right physical device is selected even if USB indices shuffle across
        reboots/replugs. Returns an int index, or None (system default) if a name
        cannot be matched — logged loudly so the failure is visible.
        """
        s = str(value).strip()
        if s == '':
            return None
        if s.lstrip('-').isdigit():
            return int(s)
        try:
            import sounddevice as sd
            want = s.lower()
            for i, dev in enumerate(sd.query_devices()):
                chans = dev['max_input_channels'] if want_input else dev['max_output_channels']
                if want in dev['name'].lower() and chans > 0:
                    self.get_logger().info(
                        f"Audio device {s!r} resolved to index {i}: {dev['name']!r} "
                        f"({'in' if want_input else 'out'}={chans})"
                    )
                    return i
            self.get_logger().error(
                f"Audio device name {s!r} not found among sounddevice "
                f"{'inputs' if want_input else 'outputs'} — using system default"
            )
        except Exception as exc:
            self.get_logger().error(
                f"Audio device resolution for {s!r} failed ({exc}) — using system default"
            )
        return None

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
            # A check-in conversation just ended: classify what they said and set
            # off home. MUST run before _flush_conversation_to_memory(), which
            # pops the transcript buffer that on_conversation_end() reads.
            if self._check_in is not None and self._check_in.is_active():
                self._check_in.on_conversation_end()
            # Any completed conversation counts as an interaction for check-in
            # purposes — OMNI having just talked to you is the most reliable
            # signal that it does not need to walk over and talk to you again.
            if self._check_in_policy is not None and self._session_person:
                self._check_in_policy.record_interaction(
                    self._session_person, datetime.now())
            # Conversation ended → persist it to memory (Step 5). Pops the bridge's
            # transcript buffer (empty if nothing was said) and stores it via the
            # omni_memory service. Non-blocking and best-effort. Done before the
            # close_session() below; the buffer survives session close regardless.
            self._flush_conversation_to_memory()
            # Enforce the IDLE invariant (see module docstring): the Gemini stream
            # must be closed in IDLE. Routes like the conversation timeout close it
            # themselves, but reaching IDLE any other way — notably Gemini calling
            # set_robot_state('IDLE') to end the chat — must also close it. Otherwise
            # the session task keeps running and the next wake word's open_session()
            # is rejected ("already running — ignoring"), leaving the robot stuck in
            # LISTENING until the conversation timeout. close_session() is idempotent,
            # so the double-close from those other routes is harmless.
            self._bridge.close_session()
            # Reset presence state so the wake word detector always runs for a
            # full presence_timeout window after returning to IDLE, regardless of
            # how long the robot was away from this state.
            self._last_person_seen = time.monotonic()
            self._presence_armed   = True
            self._audio.stop_capture()
            time.sleep(0.1)
            self._wake.start()

        # DOCKING: entered by the docking mission for the visual back-in (dock_node
        # drives via /cmd_vel_raw; self._docked is set on the /dock/result outcome).
        if new_state == 'DOCKING':
            self.get_logger().info('DOCKING — visual back-in in progress')

    # ── Conversation timeout ───────────────────────────────────────────────────

    def _reset_conversation_timeout(self):
        """
        Called by gemini_bridge on every incoming Gemini message.
        Resets the silence clock so the timeout only fires after a genuine
        30-second gap — not mid-function-call or mid-response.
        Float assignment is atomic in CPython (GIL), so no lock needed.
        """
        self._last_activity_time = time.monotonic()

    def _on_identity(self, msg: String):
        """
        Latest recognized identity from the Jetson recognizer (/camera/identity).
        Payload is a resolved person id — a known name or a stable 'unknown_N',
        or '' when no face is recognized. String assignment is atomic under the GIL.
        """
        person = (msg.data or '').strip().lower()
        self._current_person = person or None
        self._current_person_time = time.monotonic()
        # Someone may walk into frame AFTER the wake word (e.g. you wake OMNI from
        # off-camera). Adopt them mid-conversation instead of staying "stranger".
        self._maybe_late_bind(self._current_person)

    def _maybe_late_bind(self, person):
        """Bind a person recognised DURING a conversation that started without one.

        _session_person is latched at wake so mid-chat identity noise can't re-key the
        store — but if nobody was in frame at wake, that left the whole conversation
        anonymous even once the person appeared. Upgrade only in the safe direction
        (nobody/unknown -> a real name), never name -> different name.
        """
        if not self._person_keying or self._late_bind_done or not person:
            return
        with self._state_lock:
            state = self._current_state
        if state not in ('LISTENING', 'SPEAKING'):
            return                                   # no live conversation
        have = self._session_person
        if have and not have.startswith('unknown'):
            return                                   # already know who this is
        if person.startswith('unknown'):
            # Still anonymous, but key the conversation to them if we had nobody.
            if not have:
                self._session_person = person
            return
        self._late_bind_done = True
        prev = have
        self._session_person = person
        self.get_logger().info(
            f'Late-bound conversation person: {person} (was {prev!r})'
        )
        # Fetching memories blocks, so do it off the executor thread, then tell OMNI
        # who it is now looking at — otherwise it keeps treating them as a stranger.
        threading.Thread(
            target=self._inject_late_identity, args=(person,), daemon=True
        ).start()

    def _inject_late_identity(self, person):
        try:
            block = ''
            if self._memory_enabled:
                block = self._memory.retrieve_context(
                    self._memory_seed_query, k=self._memory_k, person=person
                )
            msg = (
                f'[MEMORY] You can now see who you are talking to: '
                f'{person.capitalize()}. Greet them by name naturally — you no longer '
                f'need to ask their name.'
            )
            wrapped = wrap_memory_context(block)
            if wrapped:
                msg = f'{msg}\n\n{wrapped}'
            self._bridge.inject_context(msg, alert=False)
        except Exception as exc:  # noqa: BLE001 - best effort, never break the chat
            self.get_logger().warn(f'late identity inject failed: {exc}')

    def _current_identity(self):
        """The recognized person if the identity is still fresh, else None."""
        if not self._person_keying or self._current_person is None:
            return None
        if time.monotonic() - self._current_person_time > self._identity_timeout:
            return None   # stale — treat as unknown/general
        return self._current_person

    def learn_person(self, name: str) -> bool:
        """Enroll the currently-seen face under `name` (Step 6 on-the-fly learning),
        called from the remember_person Gemini tool. Publishes an enroll request to
        the Jetson recognizer and re-keys THIS conversation to the name — because
        the transcript is stored at conversation end, that alone attributes all of
        it to the real person (no DB re-keying needed within the session).
        Returns False if person keying is disabled."""
        if not self._person_keying or self._enroll_pub is None:
            return False
        name = (name or '').strip().lower()
        if not name:
            return False
        prev = self._session_person
        # Target the person we are TALKING TO, not whoever is closest to the camera.
        # remember_person means "the one I'm conversing with is called X"; with two
        # people in frame the speaker often isn't the largest face, and a bare name
        # tells the recognizer to grab the primary face. That is how a daughter's
        # name was learned onto her father's face — he was simply nearer the lens.
        # Only an anonymous unknown_N is a meaningful target; otherwise fall back to
        # the legacy bare-name (primary face) form.
        if prev and prev.startswith('unknown'):
            request = json.dumps({'name': name, 'target': prev})
        else:
            request = name
        self._enroll_pub.publish(String(data=request))
        # Carry over this person's existing memories: if they were an anonymous
        # unknown_N, re-label those records to the new name (persisted unknowns keep
        # the same id across reboots, so this merges their whole history).
        if prev and prev.startswith('unknown') and self._memory_enabled:
            self._memory.rekey_person(prev, name)
        self._session_person = name
        # Treat them as recognized for the rest of the session, too.
        self._current_person = name
        self._current_person_time = time.monotonic()
        self.get_logger().info(f'Learning person on the fly: {name} (was {prev!r})')
        return True

    # ── Unprompted greetings ───────────────────────────────────────────────────

    def _on_low_battery(self, msg: Bool):
        """bms_node latches this True below 20% SOC."""
        self._low_battery = bool(msg.data)

    def _warm_greeter(self):
        """Background: prime the text endpoint so the first greeting is fast."""
        started = time.monotonic()
        if self._greeter.warmup():
            self.get_logger().info(
                f'greeting: decision endpoint warmed up in '
                f'{time.monotonic() - started:.2f}s')
        else:
            self.get_logger().warn(
                'greeting: decision warmup failed (check GEMINI_API_KEY / network) '
                '— the first greeting will be slower')

    def _greeting_blocked(self) -> str | None:
        """Why a greeting must not happen right now, or None if it may.

        Every one of these is enforced HERE, in code, and none of them is
        mentioned to the model. A suppression rule expressed as prompt text is a
        suggestion; a suppression rule expressed as a return statement is a rule.

        The state / session / battery rules are SHARED with the Session 9
        check-in, via suppression.interaction_blocked(). Two copies of that list
        would drift, and the drift would be invisible — the failure mode is not a
        crash, it is OMNI cheerfully interrupting a conversation months from now
        because only one copy learned about a new state. What stays here is only
        what is genuinely greeting-specific.
        """
        if not self._greeting_enabled or self._greeter is None:
            return 'greetings disabled'

        return interaction_blocked(
            self.robot_status(), min_battery=self._greeting_min_batt)

    def robot_status(self) -> RobotStatus:
        """Snapshot everything the shared suppression rules need to see.

        One builder so greetings and check-ins can never disagree about what
        "busy" means. Takes _state_lock — do not call while already holding it.

        `docked` is wired but currently always False: bms_node builds
        BatteryState field by field and never sets power_supply_status (verified
        in its source), so "OMNI is on the charger" is not knowable from
        /battery/status. The transient DOCKING state is covered by the state
        check. When the docking work lands, set self._docked and BOTH features
        inherit the suppression at once — that is the point of the shared helper.
        """
        with self._state_lock:
            state = self._current_state
        return RobotStatus(
            state=state,
            session_active=self._bridge.is_session_active(),
            low_battery=self._low_battery,
            battery_pct=self._battery_pct,
            docked=self._docked,
        )

    def _on_presence_event(self, msg: String):
        """A semantic event from event_generator on /omni/events.

        Fires on the ROS executor thread. Everything expensive — the memory
        lookup and the decision call — is handed to a daemon thread, because both
        block for seconds and this executor is single-threaded (see the
        behavior_node deadlock note in feedback_behavior_node).
        """
        try:
            event = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            self.get_logger().warn('unparseable /omni/events payload, ignoring')
            return

        kind = event.get('kind')

        # Session 9: someone has been settled at one spot long enough to be worth
        # walking over to. The manners live in CheckInPolicy; this only routes.
        if kind == 'person_dwelling':
            if self._check_in is not None:
                self._check_in.on_dwell_event(event)
            return

        if kind != 'person_appeared':
            # person_left, unknown_person_detected and our own check_in phase
            # events are published for other consumers; greeting only cares about
            # arrivals.
            self.get_logger().debug(f'presence event ignored: {kind}')
            return

        identity = (event.get('identity') or '').strip().lower()
        if not identity or identity.startswith('unknown'):
            # event_generator only emits person_appeared for named people, so this
            # is defensive — but greeting a stranger BY NAME is impossible and
            # greeting them generically is a different feature.
            return

        away = event.get('away_duration')
        away = float(away) if isinstance(away, (int, float)) else None

        blocked = self._greeting_blocked()
        if blocked:
            self.get_logger().info(
                f'greeting: {identity} appeared but suppressed — {blocked}')
            return

        now = time.monotonic()
        with self._greeting_lock:
            if self._greeting_in_flight:
                self.get_logger().info(
                    f'greeting: {identity} appeared but another greeting is in flight')
                return
            last = self._greeting_last.get(identity)
            if last is not None and (now - last) < self._greeting_cooldown:
                self.get_logger().info(
                    f'greeting: {identity} appeared but greeted '
                    f'{now - last:.0f}s ago (cooldown {self._greeting_cooldown:.0f}s)')
                return
            self._greeting_in_flight = True

        threading.Thread(
            target=self._do_greeting, args=(identity, away),
            name='greeting', daemon=True,
        ).start()

    def _do_greeting(self, identity: str, away_duration):
        """Fetch this person's memories, ask Gemini to greet them or not, act.

        Runs on its own daemon thread — both the memory retrieve and the decision
        call block for up to a couple of seconds each.
        """
        try:
            memory_context = ''
            if self._memory_enabled:
                memory_context = self._memory.retrieve_context(
                    self._memory_seed_query, k=self._memory_k, person=identity)

            hour = time.localtime().tm_hour
            period = ('morning' if hour < 12
                      else 'afternoon' if hour < 17 else 'evening')

            started = time.monotonic()
            decision = self._greeter.decide(
                identity=identity,
                away_duration=away_duration,
                memory_context=memory_context,
                period=period,
            )
            self.get_logger().info(
                f'greeting: decision for {identity} in '
                f'{time.monotonic() - started:.2f}s — '
                f'{"SPEAK" if decision.speak else "SILENCE"} ({decision.reason})')

            if not decision.speak:
                return

            # Re-check the gate: the decision call took a second or two, and the
            # user may have said the wake word in the meantime. Greeting over the
            # top of a conversation that started while we were thinking is exactly
            # the interruption this feature must never cause.
            blocked = self._greeting_blocked()
            if blocked:
                self.get_logger().info(
                    f'greeting: {identity} line composed but dropped — {blocked}')
                return

            with self._greeting_lock:
                self._greeting_last[identity] = time.monotonic()

            # A greeting is an interaction: it starts the check-in global
            # cooldown, so OMNI saying hello this morning is a reason not to walk
            # over and interrupt this afternoon.
            if self._check_in_policy is not None:
                self._check_in_policy.record_interaction(identity, datetime.now())

            self.get_logger().info(f'greeting {identity}: "{decision.line}"')
            self._speak_unprompted(
                f'Say exactly this out loud right now, in character, and then stop '
                f'and wait for them to reply: "{decision.line}". Do not add '
                f'commentary and do not repeat yourself.',
                person=identity,
                memory_context=memory_context,
            )
        except Exception as exc:  # noqa: BLE001 - a greeting must never kill the node
            self.get_logger().warn(f'greeting failed for {identity}: {exc}')
        finally:
            with self._greeting_lock:
                self._greeting_in_flight = False

    def _speak_unprompted(self, prompt: str, *, person=None, memory_context=''):
        """Open a Gemini Live session with `prompt` as the opening turn.

        The shared mechanism behind /audio/say and unprompted greetings. In IDLE
        the wake word detector owns ALSA device 0, so the handoff order is strict
        and identical to the safety-fault path: stop the detector, let the kernel
        release the device, then start capture. Without the sleep the capture open
        races the release and gets ALSA -9985.

        Deliberately leaves the robot in LISTENING rather than closing straight
        after speaking: someone greeted by name will often answer, and they should
        be able to just talk. The existing conversation timeout returns to IDLE if
        they do not.

        `memory_context` is folded into the prompt HERE rather than passed through
        to open_session(memory_context=...), because the bridge ignores that
        argument whenever initial_prompt is set — see the branch in
        gemini_bridge._run_single_session. Passing it there would look correct and
        silently deliver nothing, and the greeted person's follow-up ("did you
        finish that board?") is exactly where those memories earn their keep.

        Safe to call from any thread — _set_state, _wake and _audio are all
        internally locked or thread-confined, and open_session marshals onto the
        asyncio loop itself.
        """
        if person:
            who = (
                f'[MEMORY] You recognise the person in front of you: their name is '
                f'{person.capitalize()}. Address them by name naturally; do not '
                f'announce that you recognised their face.'
            )
            wrapped = wrap_memory_context(memory_context)
            preamble = f'{who}\n\n{wrapped}' if wrapped else who
            prompt = f'{preamble}\n\n{prompt}'

        # Ask the detector whether it actually holds the mic, rather than
        # inferring it from the state being IDLE. That inference was true for
        # every caller until Session 9 and is now wrong in both directions:
        #
        #   * IDLE but NOT running — presence-disarm stops the detector while
        #     staying in IDLE, so the old check paid a pointless 100ms sleep.
        #   * NOT IDLE but running — nothing stops the detector when navigation
        #     starts, and a check-in is the first path that drives FROM IDLE. So
        #     OMNI arrives beside you in NAVIGATING with the detector still on
        #     device 0, and the old check skipped the stop entirely. The capture
        #     open then races a device that was never released: ALSA -9985, and
        #     the check-in arrives and says nothing at all.
        if self._wake.is_running():
            self._wake.stop()
            time.sleep(0.1)   # 100ms for ALSA to release device 0
        self._audio.start_capture()

        # Attribute anything said in this exchange to the right person, exactly as
        # the wake-word path does — otherwise a conversation that began with a
        # greeting gets stored as general/household memory.
        self._session_id = uuid.uuid4().hex
        self._session_person = person
        self._late_bind_done = False

        self._reset_conversation_timeout()
        self._set_state('LISTENING')
        self._bridge.open_session(initial_prompt=prompt)

    def _warm_scene(self):
        """Background: prime the vision endpoint so the first describe_scene is fast."""
        started = time.monotonic()
        if self._scene.warmup():
            self.get_logger().info(
                f'scene: vision endpoint warmed up in {time.monotonic() - started:.2f}s'
            )
        else:
            # Not fatal — the first real call just pays the setup cost instead.
            self.get_logger().warn(
                'scene: vision warmup failed (check GEMINI_API_KEY / network) — '
                'the first description will be slower'
            )

    def _tick_check_in(self):
        """1 Hz timer — hands the check-in mission a chance to time out or abort.

        Wrapped because a raised exception in a ROS timer callback kills the
        timer silently, which would strand a mission with no way to end it.
        """
        if self._check_in is None:
            return
        try:
            self._check_in.tick()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'check-in tick failed: {exc}')

    def _flush_conversation_to_memory(self):
        """
        On conversation end, hand the accumulated transcript to omni_memory to be
        summarized and stored. Best-effort and non-blocking:
          • pop_transcript() returns '' if nothing was said → store is a no-op.
          • MemoryClient.store_transcript() drops silently if the service is down.
        Person is unknown until Step 6, so memories are stored as general/household.
        Safe from any thread (store uses call_async, no spinning here).
        """
        if not self._memory_enabled:
            return
        transcript = self._bridge.pop_transcript()
        if not transcript.strip():
            return
        self._memory.store_transcript(
            transcript, person=self._session_person,
            session_id=self._session_id, source='conversation'
        )

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

        if state == 'NAVIGATING' and elapsed > 180.0:
            # Navigation taking more than 3 minutes with no state transition —
            # Nav2 goal result callback likely lost. Cancel and recover.
            self.get_logger().warn(
                f'Watchdog: stuck in NAVIGATING for {elapsed:.0f}s — cancelling and returning to IDLE'
            )
            if self._current_goal_handle:
                self._current_goal_handle.cancel_goal_async()
                self._current_goal_handle = None
            self._set_state('IDLE')

        elif state == 'SPEAKING' and elapsed > 4.0 and not self._audio.is_playing():
            # SPEAKING but the speaker has been idle past a normal turn boundary —
            # the playback-drain transition (gemini_bridge._await_playback_and_set_
            # listening) never fired. This happens when SPEAKING was entered by a
            # path that doesn't arm that watcher, e.g. Gemini calling
            # set_robot_state('SPEAKING') by hand, or a turn ending without
            # turn_complete. Self-heal to LISTENING so the conversation continues
            # instead of stranding until the conversation/45s timeout.
            self.get_logger().warn(
                f'Watchdog: SPEAKING for {elapsed:.0f}s with idle speaker — '
                f'recovering to LISTENING'
            )
            self._set_state('LISTENING')

        elif state == 'SPEAKING' and elapsed > 45.0:
            # Still SPEAKING with the speaker active after 45s — playback itself is
            # wedged, not just a missed transition. Close the session and reset.
            self.get_logger().warn(
                f'Watchdog: stuck in SPEAKING for {elapsed:.0f}s — closing session and returning to IDLE'
            )
            self._bridge.close_session()
            self._audio.stop_capture()
            self._fault_active = False
            self._set_state('IDLE')

        elif state == 'ERROR' and elapsed > 25.0:
            # ERROR for more than 25s — fault session timed out without recovery.
            # Safety may still be faulted but restore the wake word so the user
            # can talk to OMNI and ask it to clear the fault.
            self.get_logger().warn(
                f'Watchdog: stuck in ERROR for {elapsed:.0f}s — restoring wake word'
            )
            self._bridge.close_session()
            self._audio.stop_capture()
            self._fault_active = False
            self._set_state('IDLE')

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
        # A wake word outranks a check-in from any state: the person is talking to
        # OMNI now, which is a better outcome than whatever the mission was doing.
        # abort() deliberately does not drive home — see its docstring.
        if self._check_in is not None and self._check_in.is_active():
            self._check_in.abort('wake word')
        # A wake word also outranks a docking mission — the person is talking to OMNI
        # now. Stop the approach drive or the visual back-in before opening a session.
        if self._docking:
            self._abort_docking('wake word')
        self._set_state('LISTENING')
        self._reset_conversation_timeout()
        # Detector has released the mic — start capture before opening the session
        self._audio.start_capture()

        # New conversation id, and retrieve remembered context to inject (Step 5).
        # retrieve blocks up to memory_service_timeout on THIS wake-word thread
        # (never the ROS executor); if omni_memory is down it returns '' fast-ish
        # and we open the session without memory.
        self._session_id = uuid.uuid4().hex
        # Latch the recognized person for this whole conversation (Step 6): retrieval
        # is scoped to this person + general memories, and the end-of-chat store is
        # attributed to them. None when no fresh identity → general/household.
        self._session_person = self._current_identity()
        # Re-arm late binding for this conversation (see _maybe_late_bind): if nobody
        # is in frame at wake, whoever appears next gets adopted.
        self._late_bind_done = False
        if self._session_person:
            self.get_logger().info(f'Conversation person: {self._session_person}')
        memory_context = ''
        if self._memory_enabled:
            block = self._memory.retrieve_context(
                self._memory_seed_query, k=self._memory_k, person=self._session_person
            )
            memory_context = wrap_memory_context(block)
        # Tell OMNI WHO it is talking to (Step 6). Without this it only ever sees the
        # memory statements and has no idea of the person's name — retrieval is keyed
        # by identity, but the model was never told the identity.
        if self._person_keying:
            known = self._session_person and not self._session_person.startswith('unknown')
            if known:
                who = (
                    f'[MEMORY] You recognise the person in front of you: their name is '
                    f'{self._session_person.capitalize()}. Address them by name naturally '
                    f'when it fits; do not announce that you recognised their face.'
                )
                memory_context = f'{who}\n\n{memory_context}' if memory_context else who
            else:
                hint = (
                    '[MEMORY] You do not recognise this person yet. If it fits naturally, '
                    'ask their name, and once you learn it call remember_person(name) so '
                    'you know their face next time.'
                )
                memory_context = f'{memory_context}\n\n{hint}' if memory_context else hint

        # open_session() uses call_soon_threadsafe internally — safe from any thread
        self._bridge.open_session(memory_context=memory_context)

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

        # An e-stop or safety fault ends a check-in immediately, from any state.
        # Abort BEFORE cancelling the goal: the mission must not see the cancelled
        # result and helpfully dispatch a drive home into an active fault.
        if self._check_in is not None and self._check_in.is_active():
            self._check_in.abort(f'safety fault: {fault_text}')
        # Same for a docking mission: stop the approach or the visual back-in before
        # ERROR cancels the goal, so the mission cannot dispatch anything into a fault.
        if self._docking:
            self._abort_docking(f'safety fault: {fault_text}')

        self._set_state('ERROR')

        # Cancel any active Nav2 goal so the robot stops immediately.
        if self._current_goal_handle is not None:
            self._current_goal_handle.cancel_goal_async()
            self._current_goal_handle = None

        # Always ensure mic is available for the fault announcement.
        # - IDLE: wake word detector owns the mic — stop it first.
        # - NAVIGATING: mic was never started — start it now.
        # - LISTENING/SPEAKING: capture already running — start_capture() is idempotent.
        if state_before_fault == 'IDLE':
            self._wake.stop()
            time.sleep(0.1)   # 100ms for ALSA to release device 0
        self._audio.start_capture()

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
                self._set_state('IDLE')   # _set_state('IDLE') starts the wake word internally

    # ── Scripted speech ────────────────────────────────────────────────────────

    def _on_say(self, msg: String):
        """
        Fires on the ROS2 executor thread when /audio/say receives a message.
        Makes OMNI speak the given text aloud, in character, so scripts, cron
        jobs, or other nodes can drive speech:

            ros2 topic pub --once /audio/say std_msgs/String "data: 'Hello'"

        Speech is generated by Gemini Live, not literal TTS — OMNI says the line
        in its own voice and may lightly paraphrase. inject_context() and
        open_session() are both thread-safe, so this is safe from the executor.
        """
        text = msg.data.strip()
        if not text:
            return
        self.get_logger().info(f'/audio/say received: {text}')

        prompt = (
            f'Say the following out loud right now, in character and as close to '
            f'verbatim as you can: "{text}". Do not add commentary, and do not '
            f'wait for the user to speak.'
        )

        # If a Gemini session is already live, drop the line straight in
        # (alert=False so it is not framed as an urgent fault).
        if self._bridge.is_session_active():
            self._bridge.inject_context(prompt, alert=False)
            return

        # No session — open one with the line as the first thing Gemini sees.
        # _speak_unprompted owns the mic handoff (the strict stop-detector →
        # sleep → start-capture order) and leaves the robot in LISTENING so the
        # existing conversation timeout closes the session and returns to IDLE.
        self._speak_unprompted(prompt)

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

        # A docking mission owns its approach arrival: SUCCEEDED hands off to the
        # visual back-in, not the generic "announce arrival" prompt below.
        with self._dock_lock:
            docking_approach = self._docking and self._dock_phase == 'APPROACH'
        if docking_approach:
            self._on_dock_approach_result(status)
            return

        # A check-in owns its own arrivals: reaching the person is followed by a
        # specific opener, and getting home again is followed by silence. The
        # generic "announce your arrival" prompt below is wrong for both, so the
        # mission gets first refusal on the result.
        if self._check_in is not None and self._check_in.on_nav_result(status):
            return

        # Consume the destination context once, whatever the outcome.
        intent  = self._nav_intent
        started = self._nav_started
        self._nav_intent  = None
        self._nav_started = None
        if intent and intent[0] == 'silent':
            # Foxglove/RViz goal — arrive without announcing (no Gemini chatter/cost).
            self.get_logger().info(f'Nav (silent /goal_pose) finished — status {status}')
            self._set_state('IDLE')
            return
        if intent and intent[0] == 'place':
            dest = f'the {intent[1]}'
        elif intent and intent[0] == 'person':
            dest = intent[1]
        else:
            dest = 'your destination'

        # How long the drive took, phrased loosely so OMNI can drop it in if it
        # fits the one sentence. Rounded to avoid false precision; omitted when
        # the start time is unknown (e.g. a drive that began before this field).
        took = ''
        if started is not None:
            secs = time.monotonic() - started
            if secs < 90:
                took = f'The trip took about {max(5, round(secs / 5) * 5)} seconds. '
            else:
                took = f'The trip took about {round(secs / 60)} minutes. '

        if status == 4:
            prompt = (
                f'You have just arrived at {dest}, where the user asked you to go. '
                f'{took}'
                f'Announce it briefly and in character — name where you are rather '
                f'than saying "the destination", and you may mention how long it '
                f'took if it fits naturally. One sentence.'
            )
        elif status == 5:
            prompt = None  # cancelled by user — no announcement needed
        else:
            prompt = (
                f'You were driving to {dest} but could not get there — you got '
                f'stuck or found no path. Apologise briefly in character and name '
                f'where you were headed. One sentence only.'
            )

        if prompt:
            # Transition to LISTENING so conversation_timeout handles session cleanup.
            # Do NOT go through IDLE first — that would start the wake word detector
            # and immediately need to stop it again (race condition + ALSA conflict).
            # In NAVIGATING state: mic was not running, wake word was not running.
            # Just start capture, set LISTENING, open session. Timeout closes cleanly.
            self._audio.start_capture()
            self._reset_conversation_timeout()
            self._set_state('LISTENING')
            self._bridge.open_session(initial_prompt=prompt)
        else:
            # Cancelled — return to IDLE directly (wake word resumes)
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

    # ── Docking mission ─────────────────────────────────────────────────────────
    # navigate_to the saved pre-dock pose (MAP: gross approach) → dock_node visual
    # back-in (VISION: precision). The map only has to get OMNI close enough for the
    # rear camera to see the tag; the tag + rear ToF correct for localisation error.
    def start_docking(self) -> str:
        """Begin the docking mission. Called from the `dock` Gemini function (in the
        function-handler thread pool, so the 1 s nav_is_ready() below is fine).
        Returns an in-character line for OMNI to speak. Never raises."""
        with self._dock_lock:
            if self._docking:
                return "I am already in the middle of docking, thank you."
        with self._state_lock:
            state = self._current_state
        if state in ('NAVIGATING', 'EXPLORING', 'DOCKING', 'ERROR'):
            return f"I cannot dock right now — I am currently {state.lower()}."

        pose = self.resolve_location(self._dock_location_name)
        if pose is None:
            return (
                "I'm afraid I don't have a dock location saved yet. Park me at the "
                "pre-dock spot — rear toward the dock — and save it as "
                f"'{self._dock_location_name}', then I can return to it on my own.")
        if not self.nav_is_ready():
            return ("I'm afraid my navigation systems are not available, so I cannot "
                    "drive to the dock just now.")

        with self._dock_lock:
            self._docking    = True
            self._dock_phase = 'APPROACH'
            self._dock_nav_retried = False
        # NOTE: self._docked is driven by dock_node's live /dock/docked (rear ToF) — do
        # NOT clear it here, or start_navigation below won't know to undock first.
        x, y, yaw_deg = pose
        # Tell dock_node the standoff heading (map frame) so it turns to face the tag
        # before searching — Nav2 delivers OMNI to the standoff POSITION heading-free.
        # Latched + published now, well before the visual back-in reads it at /dock/start.
        try:
            self._dock_orient_pub.publish(Float64(data=float(yaw_deg)))
        except Exception as exc:  # noqa: BLE001 — never let this kill the mission
            self.get_logger().warn(f'could not publish dock orient target: {exc}')
        self.get_logger().info('Docking mission: driving to pre-dock pose')
        self.start_navigation(x, y, yaw_deg, intent=('dock', self._dock_location_name))
        return ("Very well, I shall return to my dock — driving to the approach "
                "point now, then backing in.")

    def _on_dock_approach_result(self, status: int):
        """Nav2 result while docking is in APPROACH. On arrival, trigger the visual
        back-in; on failure, apologise. Called from _nav_result_callback (executor)."""
        if status == 4:                     # SUCCEEDED — at the pre-dock pose
            with self._dock_lock:
                self._dock_phase = 'BACKING'
            self._set_state('DOCKING')
            if not self._dock_start_cli.service_is_ready():
                self.get_logger().error('dock_node /dock/start unavailable')
                self._clear_docking()
                self._announce_via_session(
                    'You reached the dock approach point but the docking controller '
                    'is not responding. Apologise briefly, in character. One sentence.')
                return
            self.get_logger().info('At pre-dock pose — triggering visual back-in')
            self._dock_start_cli.call_async(Trigger.Request())
            # the outcome arrives on /dock/result → _on_dock_result
        elif status == 5:                   # CANCELED
            self._clear_docking()
            self._set_state('IDLE')
        else:                               # ABORTED / unknown — couldn't reach it
            # Nav2's FIRST goal after a boot/teleop can abort transiently (costmap/
            # DDS still settling — observed live 2026-08-01, an immediate retry then
            # succeeded). So retry the approach ONCE before giving up. The nav server
            # was ready moments ago, so skip the blocking nav_is_ready() here (this
            # runs on the single-threaded executor). Phase stays APPROACH, so the
            # next result routes back through here; a second abort exhausts the retry.
            with self._dock_lock:
                retry = self._docking and not self._dock_nav_retried
                if retry:
                    self._dock_nav_retried = True
            pose = self.resolve_location(self._dock_location_name) if retry else None
            if retry and pose is not None:
                self.get_logger().warn(
                    'Docking approach aborted — retrying the drive once (a first nav '
                    'goal after boot/teleop can abort transiently).')
                x, y, yaw_deg = pose
                self.start_navigation(x, y, yaw_deg,
                                      intent=('dock', self._dock_location_name))
                return
            self._clear_docking()
            self._announce_via_session(
                'You tried to drive to your dock but could not find a clear path and '
                'stopped. You do NOT know what is blocking the route — do NOT guess, '
                'invent, or name any object. Say only that you could not find a clear '
                'path to the dock right now. One sentence, in character.')

    def _on_dock_result(self, msg: String):
        """dock_node terminal outcome: 'docked' | 'failed: ...' | 'cancelled'. Acted
        on only during BACKING, so the latched startup value and strays are ignored."""
        with self._dock_lock:
            if not self._docking or self._dock_phase != 'BACKING':
                return
        result = (msg.data or '').strip()
        self.get_logger().info(f'/dock/result: {result}')
        if result == 'docked':
            self._docked = True
            self._clear_docking()
            self._announce_via_session(
                'You have just successfully docked. Announce it briefly and with '
                'quiet satisfaction, in character. One sentence.')
        elif result.startswith('failed'):
            self._clear_docking()
            self._announce_via_session(
                'You tried to back onto the dock but it did not complete. Apologise '
                'briefly, in character. One sentence only.')
        else:                               # cancelled
            self._clear_docking()
            self._set_state('IDLE')

    def _abort_docking(self, reason: str):
        """Stop an in-progress docking mission (wake word / safety fault). Cancels the
        Nav2 approach or the visual back-in per phase. Does NOT speak — the caller owns
        what OMNI says next."""
        with self._dock_lock:
            if not self._docking:
                return
            phase = self._dock_phase
            self._docking    = False
            self._dock_phase = None
        # If the approach is still deferred behind an undock (driving off the dock),
        # cancel that drive and drop the pending goal. /dock/cancel stops the undock.
        with self._undock_lock:
            was_undocking = self._undocking
            self._undocking = False
            self._pending_nav = None
        if was_undocking and self._dock_cancel_cli.service_is_ready():
            self._dock_cancel_cli.call_async(Trigger.Request())
        self.get_logger().info(f'Docking aborted — {reason} (phase {phase})')
        if phase == 'APPROACH':
            self.cancel_navigation()
        elif phase == 'BACKING' and self._dock_cancel_cli.service_is_ready():
            self._dock_cancel_cli.call_async(Trigger.Request())

    def _clear_docking(self):
        with self._dock_lock:
            self._docking    = False
            self._dock_phase = None

    def _on_dock_docked(self, msg: Bool):
        """Live docked state from dock_node (rear ToF). The authoritative 'on the dock'
        signal — feeds greeting/check-in suppression and the undock-before-nav trigger."""
        self._docked = bool(msg.data)

    def _on_undock_result(self, msg: String):
        """dock_node finished driving off the dock. On success, dispatch the nav goal
        that was deferred in start_navigation; on failure, stand down honestly."""
        result = (msg.data or '').strip()
        with self._undock_lock:
            if not self._undocking:
                return
            self._undocking = False
            pending = self._pending_nav
            self._pending_nav = None
        if result == 'undocked':
            self._docked = False
            self.get_logger().info('Undocked — dispatching the deferred navigation goal.')
            if pending is not None:
                x, y, yaw_deg, intent = pending
                self.start_navigation(x, y, yaw_deg, intent=intent)
        else:
            self.get_logger().warn(f'Undock failed ({result}) — cannot leave the dock.')
            with self._dock_lock:
                was_docking = self._docking
            if was_docking:
                self._clear_docking()
            self._set_state('IDLE')
            self._announce_via_session(
                'You tried to move off your dock but could not. Say briefly and in '
                'character that you are unable to leave the dock right now. One sentence.')

    def _announce_via_session(self, prompt: str):
        """Speak a one-off line by opening a Live session with it as the first thing
        Gemini sees, landing in LISTENING so the conversation timeout closes it — the
        same mechanism as the nav-arrival announcement. ROS-executor thread only."""
        self._audio.start_capture()
        self._reset_conversation_timeout()
        self._set_state('LISTENING')
        self._bridge.open_session(initial_prompt=prompt)

    # ── Zones + world state (Session 7) ─────────────────────────────────────────

    def _load_zones(self):
        """Build the shared ZoneMap. Path param wins; else omni_zones' shipped
        config. Never raises — a bad or missing file degrades to an empty map so
        the robot still boots (navigate_to then uses point locations only)."""
        path = os.path.expanduser(self.get_parameter('zones_config_path').value or '')
        if not path:
            try:
                path = os.path.join(
                    get_package_share_directory('omni_zones'), 'config', 'zones.yaml')
            except Exception as exc:
                self.get_logger().warn(f'omni_zones share dir not found: {exc}')
                return load_zone_map({})
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f) or {}
            zone_map = load_zone_map(data.get('zones') or {})
            self.get_logger().info(
                f'Loaded {len(zone_map)} zone(s) from {path}'
                + (f': {", ".join(zone_map.names)}' if zone_map else ' (empty)'))
            return zone_map
        except FileNotFoundError:
            self.get_logger().info(
                f'No zones file at {path} — go_to_person will report unknown '
                f'locations and navigate_to uses point locations only.')
            return load_zone_map({})
        except Exception as exc:
            self.get_logger().warn(f'Failed to load zones from {path}: {exc}')
            return load_zone_map({})

    def _on_world_state(self, msg: String):
        """Cache the latest world_state snapshot for go_to_person()."""
        try:
            self._world_state = json.loads(msg.data)
            self._world_state_time = time.monotonic()
        except (json.JSONDecodeError, TypeError):
            self.get_logger().warn('unparseable /omni/world_state payload, ignoring')

    def latest_world_state(self):
        """The most recent world_state snapshot dict, or None if world_state is
        not publishing (the node is a soft dependency, exactly like memory)."""
        return self._world_state

    def known_place_names(self) -> list:
        """Every name navigate_to can resolve: saved point locations plus zones."""
        return sorted(set(self._locations.keys()) | set(self._zones.names))

    def resolve_location(self, name: str):
        """Resolve a named place to a goal pose (x, y, yaw_deg), or None.

        Saved point locations (from save_location) win over zone anchors when a
        name exists as both — a point the user deliberately taught is a more
        specific intent than a room's default parking spot."""
        name = (name or '').lower().strip()
        loc = self._locations.get(name)
        if loc:
            yaw = float(loc[2]) if len(loc) > 2 else 0.0
            return (float(loc[0]), float(loc[1]), yaw)
        return self._zones.nav_pose(name)   # None if the zone is unknown

    def _on_goal_pose(self, msg: PoseStamped):
        """A goal clicked in Foxglove/RViz (2D Goal Pose -> /goal_pose). Route it through
        start_navigation, so it undocks first if OMNI is on the dock. Arrival is silent —
        it's a dev tool, no Gemini announcement/cost."""
        with self._dock_lock:
            if self._docking:
                self.get_logger().info('/goal_pose ignored — docking mission in progress')
                return
        q = msg.pose.orientation
        yaw_deg = math.degrees(math.atan2(
            2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)))
        x, y = msg.pose.position.x, msg.pose.position.y
        self.get_logger().info(
            f'/goal_pose (Foxglove/RViz) -> navigating to x={x:.2f} y={y:.2f} '
            f'yaw={yaw_deg:.0f}°')
        self.start_navigation(x, y, yaw_deg, intent=('silent', 'goal_pose'))

    def start_navigation(self, x: float, y: float, yaw_deg: float,
                         *, intent=None) -> None:
        """Build and dispatch a Nav2 goal, and enter NAVIGATING. Shared by
        navigate_to() and go_to_person(). Callers must have checked
        nav_is_ready() first. Thread-safe: send_goal_async schedules on the
        executor without blocking.

        intent — optional (kind, label) describing WHERE and WHY, e.g.
        ('place', 'kitchen') or ('person', 'Rafael'). Stashed so the arrival
        announcement can name the destination instead of saying 'the
        destination'. None keeps the generic prompt (e.g. the check-in drives
        through here but owns its own arrival via _nav_result_callback)."""
        # If OMNI is on the dock, Nav2 cannot plan out (the docked start pose is inside
        # robot_radius of the wall). Undock first (drive forward off the dock via
        # dock_node), then dispatch this goal from _on_undock_result. This branch is
        # DORMANT unless docked, so ordinary navigation is unchanged.
        with self._undock_lock:
            if self._undocking:
                self._pending_nav = (x, y, yaw_deg, intent)   # newer goal supersedes
                return
            if self._docked and self._dock_undock_cli.service_is_ready():
                self._pending_nav = (x, y, yaw_deg, intent)
                self._undocking = True
                self.get_logger().info('On the dock — undocking before navigating.')
                self._dock_undock_cli.call_async(Trigger.Request())
                return

        self._nav_intent  = intent         # consumed + cleared in _nav_result_callback
        self._nav_started = time.monotonic()
        goal_msg      = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp    = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.position.z = 0.0
        yaw_rad = math.radians(float(yaw_deg))
        goal_msg.pose.pose.orientation.z = math.sin(yaw_rad / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(yaw_rad / 2.0)

        self.get_logger().info(
            f'Sending navigation goal: x={x:.3f}, y={y:.3f}, yaw={yaw_deg:.1f}°')
        self._set_state('NAVIGATING')
        send_future = self._nav_action_client.send_goal_async(
            goal_msg, feedback_callback=self._nav_feedback_callback)
        send_future.add_done_callback(self._nav_goal_response_callback)

    # ── Nav2 readiness check ───────────────────────────────────────────────────

    def cancel_navigation(self) -> bool:
        """Cancel any in-flight Nav2 goal. Returns True if there was one.

        Used by the check-in when it is aborted mid-drive: without this, a wake
        word during APPROACH leaves OMNI still driving to the person while the
        person is already talking to it, and the goal's eventual SUCCEEDED result
        falls through to the generic arrival handler — which cheerfully announces
        that it has arrived, over the top of the conversation.

        Safe from any thread; cancel_goal_async schedules on the executor.
        """
        handle = self._current_goal_handle
        if handle is None:
            return False
        self._current_goal_handle = None
        try:
            handle.cancel_goal_async()
        except Exception as exc:  # noqa: BLE001 - cancelling is best-effort
            self.get_logger().warn(f'nav cancel failed: {exc}')
            return False
        return True

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
        node._memory.shutdown()
        node._frames.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
