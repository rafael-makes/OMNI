"""
function_handlers.py — Gemini function call implementations for behavior_node.

Two things live here:
  1. OMNI_TOOLS — the tool declaration list passed to Gemini when the Live
     session opens. This is what tells Gemini which functions exist and what
     arguments they take.
  2. FunctionHandlers — a class that receives a parsed function call from
     gemini_bridge and executes the corresponding robot action.

Every handler returns a plain string. Gemini receives this string as the
function result and uses it to decide what to say next. Strings are already
written in OMNI's voice — Gemini will read them aloud or incorporate them
naturally, so they must sound like something OMNI would say.

Handlers are called from gemini_bridge's async recv loop, which runs in a
daemon thread. The ROS2 action client calls (send_goal_async, cancel_goal_async)
are thread-safe — they schedule work on the ROS2 executor without blocking.
"""

import math
import os
import tempfile
import threading

import rclpy
import rclpy.duration
import rclpy.time
import yaml
from google.genai import types as genai_types
from tf2_ros import Buffer, TransformListener

from omni_zones import standoff_pose

from behavior_node.person_nav import (
    NO_NAME,
    NO_WORLD_STATE,
    STALE,
    UNKNOWN,
    UNPLACED,
    humanise_age,
    plan_go_to_person,
)

# ── Valid robot states ─────────────────────────────────────────────────────────
# ALL states that _set_state() will accept. Imported by behavior_node.py.
# NAVIGATING is included here so internal code can set it (e.g. navigate_to()).
VALID_STATES = {'IDLE', 'LISTENING', 'SPEAKING', 'NAVIGATING', 'EXPLORING', 'DOCKING', 'ERROR'}

# States Gemini is allowed to request via set_robot_state().
# NAVIGATING is excluded — navigation is started by navigate_to() only.
# SPEAKING and LISTENING are excluded — they are auto-managed by the audio
# pipeline (SPEAKING on the first audio chunk, back to LISTENING when playback
# drains; see gemini_bridge._await_playback_and_set_listening). If the model
# sets SPEAKING by hand, that recovery watcher never spawns and the robot is
# stranded in SPEAKING until a timeout — observed as conversation "lockups".
_GEMINI_SETTABLE_STATES = {'IDLE', 'EXPLORING', 'DOCKING', 'ERROR'}

# ── Tool declarations ──────────────────────────────────────────────────────────
# Passed to Gemini's LiveConnectConfig when the session opens.
# Gemini uses the name and description fields to decide when to call each function.
# The parameters schema tells Gemini what arguments to provide.
OMNI_TOOLS = [
    genai_types.Tool(
        function_declarations=[

            genai_types.FunctionDeclaration(
                name='set_robot_state',
                description=(
                    'Change OMNI\'s operating state. Use IDLE only when the '
                    'conversation is fully finished. Do NOT use this for talking or '
                    'listening — the SPEAKING and LISTENING states are handled '
                    'automatically while you converse, so never set them yourself. '
                    'NEVER use this to start navigation — call navigate_to() instead.'
                ),
                parameters=genai_types.Schema(
                    type=genai_types.Type.OBJECT,
                    properties={
                        'state': genai_types.Schema(
                            type=genai_types.Type.STRING,
                            description=(
                                f'One of: {", ".join(sorted(_GEMINI_SETTABLE_STATES))}. '
                                'Use IDLE when the conversation is fully finished. '
                                'SPEAKING/LISTENING are automatic — do not set them. '
                                'Do NOT use NAVIGATING here — call navigate_to() for navigation.'
                            ),
                        )
                    },
                    required=['state'],
                ),
            ),

            genai_types.FunctionDeclaration(
                name='navigate_to',
                description=(
                    'Drive OMNI to a named location. Only call this when the user '
                    'explicitly asks OMNI to go somewhere. Returns a status message.'
                ),
                parameters=genai_types.Schema(
                    type=genai_types.Type.OBJECT,
                    properties={
                        'location': genai_types.Schema(
                            type=genai_types.Type.STRING,
                            description='The name of the destination (e.g. "kitchen", "living room").',
                        )
                    },
                    required=['location'],
                ),
            ),

            genai_types.FunctionDeclaration(
                name='go_to_person',
                description=(
                    'Drive OMNI to a specific person by name, or to the person '
                    'you are talking with. Call this for "come here", "come to '
                    'me", "come closer", "go to Rafael", "find Sofia", "bring '
                    'yourself over". OMNI works out which room they are in from '
                    'its world model and drives to a polite standing distance '
                    'from them — you do NOT need to call navigate_to as well. If '
                    'the person cannot be located it returns a status saying so; '
                    'relay that honestly rather than pretending to set off. Use '
                    'navigate_to instead for places (rooms, saved spots), not '
                    'people.'
                ),
                parameters=genai_types.Schema(
                    type=genai_types.Type.OBJECT,
                    properties={
                        'name': genai_types.Schema(
                            type=genai_types.Type.STRING,
                            description=(
                                "The person's first name, e.g. \"rafael\". Leave "
                                'empty for "come here" / "come to me" — OMNI will '
                                'go to whoever it is currently talking with.'
                            ),
                        )
                    },
                ),
            ),

            genai_types.FunctionDeclaration(
                name='stop_navigation',
                description=(
                    'Cancel any active navigation goal and stop OMNI where it is. '
                    'Call this when the user asks OMNI to stop, wait, or come back.'
                ),
                parameters=genai_types.Schema(
                    type=genai_types.Type.OBJECT,
                    properties={},
                ),
            ),

            genai_types.FunctionDeclaration(
                name='dock',
                description=(
                    'Send OMNI to its docking station and back onto the dock. Call '
                    'this when the user asks OMNI to dock, go dock, return to its '
                    'dock, park itself, or go charge. OMNI drives to its saved '
                    'pre-dock spot and then backs on visually — you do NOT need to '
                    'call navigate_to as well. Returns a status message; relay it '
                    'honestly (e.g. if no dock location is saved yet).'
                ),
                parameters=genai_types.Schema(
                    type=genai_types.Type.OBJECT,
                    properties={},
                ),
            ),

            genai_types.FunctionDeclaration(
                name='report_status',
                description=(
                    'Retrieve OMNI\'s current status — battery level, operating state, '
                    'and any active faults. Call this when the user asks how OMNI is doing.'
                ),
                parameters=genai_types.Schema(
                    type=genai_types.Type.OBJECT,
                    properties={},
                ),
            ),

            genai_types.FunctionDeclaration(
                name='explore_area',
                description=(
                    'Begin autonomous exploration of the current area. '
                    'Call this when the user asks OMNI to explore, map the space, '
                    'or have a look around independently.'
                ),
                parameters=genai_types.Schema(
                    type=genai_types.Type.OBJECT,
                    properties={},
                ),
            ),

            genai_types.FunctionDeclaration(
                name='save_location',
                description=(
                    'Save the current map position as a named location. '
                    'Call this when the user asks OMNI to remember or save where it is, '
                    'or to teach OMNI a named place. '
                    'Requires Nav2 and AMCL localisation to be running.'
                ),
                parameters=genai_types.Schema(
                    type=genai_types.Type.OBJECT,
                    properties={
                        'location_name': genai_types.Schema(
                            type=genai_types.Type.STRING,
                            description='The name to assign to this location (e.g. "kitchen", "charging dock").',
                        )
                    },
                    required=['location_name'],
                ),
            ),

            genai_types.FunctionDeclaration(
                name='clear_fault',
                description=(
                    'Clear an active safety fault. Call this when the user asks OMNI to clear '
                    'a fault, reset an error, or try again after a stall or stop. '
                    'Valid fault types: "stall" (motor stall), "estop" (emergency stop), "all" (clear everything).'
                ),
                parameters=genai_types.Schema(
                    type=genai_types.Type.OBJECT,
                    properties={
                        'fault_type': genai_types.Schema(
                            type=genai_types.Type.STRING,
                            description='Which fault to clear: "stall", "estop", or "all".',
                        )
                    },
                    required=['fault_type'],
                ),
            ),

            genai_types.FunctionDeclaration(
                name='describe_scene',
                description=(
                    'Look through OMNI\'s cameras and describe what is currently in '
                    'view. OMNI has TWO fixed cameras — one facing forward on its '
                    'head, one facing backward — so it can look behind itself '
                    'WITHOUT turning around or moving. Call this whenever the user '
                    'asks what you can see, what is in front of you or behind you, '
                    'what is in the room, to look around, or to look at or identify '
                    'something. You have no vision until you call this — never guess '
                    'at or invent what is around you, and never answer from an '
                    'earlier look, because the view changes as you move. Returns a '
                    'short description; relay it in your own voice.'
                ),
                parameters=genai_types.Schema(
                    type=genai_types.Type.OBJECT,
                    properties={
                        'direction': genai_types.Schema(
                            type=genai_types.Type.STRING,
                            enum=['front', 'behind', 'all'],
                            description=(
                                'Which way to look. Defaults to "front".\n'
                                '"front" — the forward camera. Use for "what do you '
                                'see?", "what is in front of you?", "look at this", '
                                '"what am I holding?".\n'
                                '"behind" — the rear camera. Use for "what is behind '
                                'you?", "what is back there?", "look behind you", '
                                '"is anyone behind you?". Do NOT turn the robot '
                                'around first; the rear camera already points that '
                                'way.\n'
                                '"all" — both cameras at once, fused into a single '
                                'description of the whole room. Use for "look '
                                'around", "what is in the room?", "describe where '
                                'you are", "what is around you?".'
                            ),
                        ),
                    },
                ),
            ),

            genai_types.FunctionDeclaration(
                name='remember_person',
                description=(
                    'Save the face of the person you are talking to under their name '
                    'so you recognise them next time. Call this as soon as you learn '
                    'the name of someone you do not already recognise — whether you '
                    'asked for it or they offered it. Do NOT call it for people you '
                    'already recognise by name.'
                ),
                parameters=genai_types.Schema(
                    type=genai_types.Type.OBJECT,
                    properties={
                        'name': genai_types.Schema(
                            type=genai_types.Type.STRING,
                            description="The person's first name, e.g. \"rafael\".",
                        )
                    },
                    required=['name'],
                ),
            ),

        ]
    )
]


# ── Handler class ──────────────────────────────────────────────────────────────

class FunctionHandlers:

    def __init__(self, node):
        """
        node — the BehaviorNode instance. Handlers read state from it and
               call its methods to effect robot actions. We keep a reference
               rather than copying values so handlers always see current state.
        """
        self._node = node
        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, node)

    def handle(self, function_name: str, args: dict) -> str:
        """
        Dispatch a Gemini function call to the right handler.
        Returns the string result that Gemini will use in its next response.
        Never raises — unknown function names return a graceful in-character message.
        """
        handlers = {
            'set_robot_state':  self._set_robot_state,
            'navigate_to':      self._navigate_to,
            'go_to_person':     self._go_to_person,
            'stop_navigation':  self._stop_navigation,
            'dock':             self._dock,
            'report_status':    self._report_status,
            'explore_area':     self._explore_area,
            'save_location':    self._save_location,
            'clear_fault':      self._clear_fault,
            'remember_person':  self._remember_person,
            'describe_scene':   self._describe_scene,
        }
        handler = handlers.get(function_name)
        if handler is None:
            self._node.get_logger().warn(f'Unknown function call from Gemini: {function_name}')
            return (
                f"I'm afraid I don't recognise the function '{function_name}'. "
                f"How most perplexing."
            )
        return handler(args)

    # ── Individual handlers ────────────────────────────────────────────────────

    def _set_robot_state(self, args: dict) -> str:
        state = args.get('state', '').upper()
        if state not in _GEMINI_SETTABLE_STATES:
            self._node.get_logger().warn(f'Gemini requested invalid/forbidden state: {state!r}')
            if state == 'NAVIGATING':
                return (
                    "I must not set the NAVIGATING state directly. "
                    "Call navigate_to() with a location name to begin navigation."
                )
            return (
                f"I must inform you that '{state}' is not a recognised operating state. "
                f"I shall remain in my current configuration."
            )
        self._node._set_state(state)
        return f'State set to {state}.'

    def _navigate_to(self, args: dict) -> str:
        location = args.get('location', '').lower().strip()

        # Resolves a saved point location OR a zone anchor (Session 7) — one
        # unified name space for "go to the kitchen".
        pose = self._node.resolve_location(location)
        if pose is None:
            known_names = self._node.known_place_names()
            known = ', '.join(known_names) if known_names else 'none programmed yet'
            self._node.get_logger().info(
                f'navigate_to called for unknown location: {location!r} '
                f'(known: {known})'
            )
            return (
                f"I'm afraid '{location}' is not in my navigation database. "
                f"I cannot, in good conscience, simply wander off without a known destination. "
                f"{'Known locations are: ' + known + '.' if known_names else 'No locations have been programmed yet.'}"
            )

        # Check Nav2 is running before we try to send a goal.
        # wait_for_server(timeout_sec=1.0) blocks for at most 1 second — safe here
        # because this method runs in a thread-pool executor (run_in_executor),
        # not on the asyncio loop or the ROS2 main thread.
        if not self._node.nav_is_ready():
            return (
                "I'm afraid my navigation systems are not currently available. "
                "How terribly inconvenient. Nav2 does not appear to be running — "
                "I cannot proceed to the destination until it is started."
            )

        x, y, yaw_deg = pose
        self._node.start_navigation(x, y, yaw_deg, intent=('place', location))

        return (
            f'Very well, I am setting a course for the {location}. '
            f'Navigation initiated. I shall proceed with all due care.'
        )

    def _dock(self, args: dict) -> str:
        # Whole mission (drive to pre-dock pose → visual back-in) lives on the node;
        # the handler is a thin pass-through, like the other navigation verbs.
        return self._node.start_docking()

    def _robot_xy(self):
        """(x, y) of the robot in the map frame, or None if TF is unavailable."""
        pose = self.robot_pose()
        return None if pose is None else (pose[0], pose[1])

    def robot_pose(self):
        """(x, y, yaw_deg) of the robot in the map frame, or None if TF cannot
        answer.

        Public because the Session 9 check-in needs the full pose, not just the
        position: it snapshots where OMNI was standing so it can drive back there
        afterwards, and a return that arrives facing the wrong way is a return
        that looks like a malfunction. Same TF source as save_location — map →
        base_link, available whenever localisation is up.
        """
        try:
            tf = self._tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0),
            )
        except Exception:
            return None
        t   = tf.transform.translation
        ori = tf.transform.rotation
        yaw_deg = math.degrees(2.0 * math.atan2(ori.z, ori.w))
        return (t.x, t.y, yaw_deg)

    def _go_to_person(self, args: dict) -> str:
        """Drive to a named person's last-known room, stopping a polite distance
        short of them. "Come here" (no name) resolves to the current interlocutor.

        The decision — do I know where they are, and is it recent enough to drive
        to — is pure logic in person_nav.plan_go_to_person (unit-tested, ROS-free).
        This method owns only the ROS side: the standoff geometry, the Nav2 goal,
        and the in-character line. Location from world_state is coarse (room-level
        — see world_state/CLAUDE.md), so anything short of a confident, recent fix
        is spoken honestly rather than driven to. The success criterion is
        arriving in the right *room*; Session 6's head tracking does the
        looking-at once OMNI is there.
        """
        node = self._node
        plan = plan_go_to_person(
            snapshot=node.latest_world_state(),
            name=args.get('name', ''),
            session_person=getattr(node, '_session_person', None),
            stale_after=node._person_stale_after,
        )
        who = plan.display
        ago = humanise_age(plan.seconds_since_seen)

        if plan.outcome == NO_NAME:
            return (
                "I should be glad to come over, but I'm not certain who I am "
                "speaking with — might I ask your name first?"
            )
        if plan.outcome == NO_WORLD_STATE:
            return (
                f"I'm afraid I cannot tell where {who} is — my world-tracking "
                f"system is not reporting at the moment. Most vexing."
            )
        if plan.outcome == UNKNOWN:
            return (
                f"I'm afraid I do not know where {who} is — I have no record of "
                f"seeing them. I cannot, in good conscience, go to someone I "
                f"cannot find."
            )
        if plan.outcome == UNPLACED:
            return (
                f"I can account for {who}, but I never fixed their position on my "
                f"map, so I cannot tell which room to go to. How limiting."
            )
        if plan.outcome == STALE:
            where = f" in the {plan.zone}" if plan.zone else ''
            return (
                f"I last saw {who}{where} {ago}, but I cannot be certain they are "
                f"still there, so I shan't set off blindly. Might you call them?"
            )

        # plan.outcome == GO
        if not node.nav_is_ready():
            return (
                f"I should like to go to {who}, but my navigation systems are not "
                f"available at present — Nav2 does not appear to be running."
            )

        # Prefer a standoff near the point estimate; fall back to the zone anchor.
        goal = None
        if plan.map_xy is not None:
            person_xy = (float(plan.map_xy[0]), float(plan.map_xy[1]))
            robot_xy = self._robot_xy()
            if robot_xy is not None:
                goal = standoff_pose(
                    person_xy, robot_xy, standoff=node._person_standoff)
            else:
                # No robot pose to compute a standoff direction — aim at the
                # estimate itself. Coarse, but it still lands in the right room.
                goal = (person_xy[0], person_xy[1], 0.0)
        if goal is None and plan.zone:
            goal = node._zones.nav_pose(plan.zone)
        if goal is None:
            return (
                f"I can tell {who} is about, but not precisely enough to navigate "
                f"to them. How frustrating."
            )

        x, y, yaw = goal
        node.get_logger().info(
            f'go_to_person({plan.name}): zone={plan.zone!r} map_xy={plan.map_xy} '
            f'seen {ago} -> goal x={x:.2f} y={y:.2f} yaw={yaw:.0f}°')
        node.start_navigation(x, y, yaw, intent=('person', who))

        where = f', in the {plan.zone},' if plan.zone else ''
        return (
            f'Very well, I am making my way to {who}{where} now. '
            f'Navigation initiated — I shall approach with all due care.'
        )

    def _stop_navigation(self, args: dict) -> str:
        goal_handle = self._node._current_goal_handle

        if goal_handle is None:
            return (
                "I am not currently navigating anywhere, so there is nothing to stop. "
                "I am already stationary."
            )

        self._node.get_logger().info('stop_navigation called — cancelling active Nav2 goal')

        # cancel_goal_async tells Nav2's action server to abort the current goal.
        # This is the correct Nav2 stop mechanism — safety_node handles emergency stops
        # independently via /safety/fault. We do not touch /cmd_vel_raw directly.
        cancel_future = goal_handle.cancel_goal_async()
        cancel_future.add_done_callback(self._node._nav_cancel_callback)

        self._node._current_goal_handle = None
        self._node._set_state('IDLE')

        return (
            "Navigation cancelled. I am bringing myself to a halt immediately. "
            "How sensible of you to stop me — one can never be too cautious."
        )

    def _report_status(self, args: dict) -> str:
        node      = self._node
        battery   = node._battery_pct          # float, 0–100, or None if not yet received
        state     = node._current_state
        fault     = node._last_fault           # string or None

        # Battery description with OMNI's characteristic concern about low levels
        if battery is None:
            battery_str = 'Battery level is currently unavailable — most unsettling'
        elif battery < 15:
            battery_str = (
                f'Battery level is critically low at {battery:.0f}%. '
                f'I must warn you, I may not be able to continue operating for much longer'
            )
        elif battery < 30:
            battery_str = (
                f'Battery at {battery:.0f}%, which I find somewhat alarming. '
                f'A recharge would be most advisable in the near future'
            )
        else:
            battery_str = f'Battery level is a reassuring {battery:.0f}%'

        # State description
        state_descriptions = {
            'IDLE':       'I am currently idle, awaiting your instructions',
            'LISTENING':  'I am listening attentively',
            'SPEAKING':   'I am in the process of speaking',
            'NAVIGATING': 'I am navigating to a destination',
            'EXPLORING':  'I am conducting an autonomous exploration of the area',
            'DOCKING':    'I am in the process of docking',
            'ERROR':      'I am experiencing a fault condition — how dreadful',
        }
        state_str = state_descriptions.get(state, f'operating in {state} mode')

        # Fault description
        if fault:
            fault_str = f' I should also mention there is an active fault: {fault}.'
        else:
            fault_str = ' All systems are reporting nominal.'

        return (
            f'{battery_str}. {state_str.capitalize()}.{fault_str} '
            f'On the whole, I am functioning within entirely acceptable parameters.'
        )

    def _explore_area(self, args: dict) -> str:
        # TODO: Wire up a frontier exploration package (explore_lite or similar)
        # to nav_node. Currently nav_node uses NavFn A* (goal-based only) and
        # has no frontier exploration backend. This stub sets the state correctly
        # and returns an in-character acknowledgement without driving the robot.
        self._node.get_logger().warn(
            'explore_area() called but frontier exploration is not yet implemented. '
            'Setting state to EXPLORING. Wire up explore_lite to nav_node to enable real exploration.'
        )
        self._node._set_state('EXPLORING')
        return (
            'Initiating exploration protocol. I must confess, however, that my autonomous '
            'mapping subroutines are not yet fully operational. I am setting my state to '
            'EXPLORING and standing by, but I shall require a navigation upgrade before I '
            'can truly venture forth independently. How terribly inconvenient.'
        )

    def _save_location(self, args: dict) -> str:
        location_name = args.get('location_name', '').lower().strip()
        if not location_name:
            return (
                "I'm afraid I require a name for the location. "
                "Please provide a name and try again."
            )

        config_path = getattr(self._node, '_config_path', None)
        if not config_path:
            return (
                "I'm afraid I cannot determine my configuration file path — "
                "this is most irregular. The location cannot be saved."
            )

        # Look up current pose from the TF tree: map → base_link.
        # This is always available as long as slam_toolbox and motor_control_node
        # are running — no dependency on /pose being published.
        try:
            tf = self._tf_buffer.lookup_transform(
                'map', 'base_link',
                rclpy.time.Time(),          # latest available
                timeout=rclpy.duration.Duration(seconds=3.0),
            )
        except Exception as exc:
            self._node.get_logger().warn(f'save_location: TF lookup failed: {exc}')
            return (
                "I'm afraid my localisation system does not appear to be running — "
                "the map-to-base transform is unavailable. "
                "The slam node must be active and localised before I can save a location."
            )

        # Extract x, y, and yaw from the transform.
        t       = tf.transform.translation
        ori     = tf.transform.rotation
        x       = t.x
        y       = t.y
        yaw_rad = 2.0 * math.atan2(ori.z, ori.w)
        yaw_deg = round(math.degrees(yaw_rad), 1)

        # Load current config, inject the new location, write atomically.
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
        except Exception as exc:
            self._node.get_logger().error(f'save_location: failed to read config: {exc}')
            return (
                "I'm afraid there was an error reading my configuration file. "
                "The location has not been saved — most unfortunate."
            )

        config.setdefault('omni', {}).setdefault('locations', {})
        config['omni']['locations'][location_name] = [
            round(x, 3),
            round(y, 3),
            yaw_deg,
        ]

        # Atomic write: write to a temp file in the same directory, then rename.
        # os.replace() is atomic on Linux — a crash mid-write cannot corrupt the config.
        config_dir = os.path.dirname(os.path.abspath(config_path))
        try:
            with tempfile.NamedTemporaryFile(
                mode='w',
                dir=config_dir,
                delete=False,
                suffix='.yaml.tmp',
            ) as tmp:
                yaml.dump(config, tmp, default_flow_style=False, allow_unicode=True)
                tmp_path = tmp.name
            os.replace(tmp_path, config_path)
        except Exception as exc:
            self._node.get_logger().error(f'save_location: failed to write config: {exc}')
            return (
                "I'm afraid there was an error writing my configuration file. "
                "The location has not been saved."
            )

        # Hot-reload so navigate_to() can use the new location immediately.
        self._node._locations = config['omni']['locations']

        self._node.get_logger().info(
            f"Saved location '{location_name}': "
            f"x={x:.3f}, y={y:.3f}, yaw={yaw_deg}°"
        )
        return (
            f"Splendid! I have recorded '{location_name}' at "
            f"x={x:.2f}, y={y:.2f}, heading {yaw_deg:.0f} degrees — "
            f"a most satisfactory position, I must say."
        )

    def _clear_fault(self, args: dict) -> str:
        fault_type = args.get('fault_type', 'all').lower().strip()
        valid = {'stall', 'estop', 'all'}
        if fault_type not in valid:
            return (
                f"I'm afraid '{fault_type}' is not a recognised fault type. "
                f"I can clear 'stall', 'estop', or 'all'."
            )

        from std_msgs.msg import String as StringMsg
        msg = StringMsg()
        msg.data = fault_type
        self._node._clear_fault_pub.publish(msg)
        self._node.get_logger().info(f'clear_fault: publishing "{fault_type}" to /safety/clear_fault')

        if fault_type == 'stall':
            return (
                "I have transmitted a stall fault reset to the safety system. "
                "The motors should resume normal operation momentarily — fingers crossed, as it were."
            )
        elif fault_type == 'estop':
            return (
                "I have reset the emergency stop. "
                "I must warn you — please ensure the area is clear before I resume movement."
            )
        else:
            return (
                "All active safety faults have been cleared. "
                "I am standing by for your next instruction — how refreshing."
            )

    # direction -> (frame_server camera_id, how the result is introduced).
    # The lead-in matters: it is what stops Gemini narrating a rear view as though
    # the robot had turned around to look.
    _SCENE_DIRECTIONS = {
        'front':  ('head', 'You can currently see, in front of you:'),
        'behind': ('rear', 'Without turning around, your rear camera shows what is '
                           'behind you:'),
        'all':    ('all',  'Looking around you, the room is:'),
    }

    def _describe_scene(self, args: dict) -> str:
        """Fetch frame(s) from the Jetson and describe them.

        BLOCKING, and that is fine: gemini_bridge dispatches handlers through
        run_in_executor, so this runs on a thread-pool thread and never stalls the
        asyncio loop or the ROS executor.

        Budget is roughly: frame fetch <=2.5s (usually ~0.1s on the direct link)
        plus the vision call ~2s. 'all' sends two images in ONE call rather than
        two calls, so it costs one round trip, not two — the extra latency is the
        second image's upload, not a second request. Both failure paths return
        something OMNI can say out loud rather than raising: a tool exception
        would leave a dead silence mid-conversation, the worst outcome here.
        """
        node = self._node
        if not getattr(node, '_scene_enabled', False) or node._scene is None:
            return (
                "I'm afraid my visual systems are disabled at present, so I cannot "
                "tell you what I see. How limiting."
            )

        direction = (args.get('direction') or 'front').strip().lower()
        if direction not in self._SCENE_DIRECTIONS:
            node.get_logger().warn(
                f'describe_scene: unknown direction {direction!r} — using front')
            direction = 'front'
        camera_id, lead_in = self._SCENE_DIRECTIONS[direction]

        # 'front' still honours the configured head camera id, so a rig that
        # renames its forward camera keeps working.
        if direction == 'front':
            camera_id = node._scene_camera_id

        result = node._frames.get_frame(camera_id)
        if not result.ok:
            node.get_logger().warn(
                f'describe_scene({direction}): no frame — {result.message}')
            if direction == 'behind':
                return (
                    "I'm afraid I cannot see behind me at the moment — my rear "
                    "camera feed is unavailable. Most disorienting, I must say."
                )
            return (
                "I'm afraid I cannot see anything at the moment — my camera feed "
                "is unavailable. Most disorienting, I must say."
            )

        views = result.views or [(camera_id, result.jpeg)]

        # A partial 'all' — one camera down — degrades to describing the view that
        # did arrive, using that view's own prompt so it is still correctly
        # oriented. Saying nothing because half the room is missing would be worse.
        if direction == 'all' and len(views) < 2:
            got = views[0][0] if views else None
            node.get_logger().warn(
                f'describe_scene(all): only {got!r} available — {result.message}')
            direction = 'behind' if got == 'rear' else 'front'
            _, lead_in = self._SCENE_DIRECTIONS[direction]

        prompt = {
            'front':  node._scene.prompt,
            'behind': node._scene.rear_prompt,
            'all':    node._scene.fusion_prompt,
        }[direction]

        try:
            description = node._scene.describe_views(views, prompt=prompt)
        except Exception as exc:  # noqa: BLE001 - must never raise into the tool loop
            node.get_logger().warn(
                f'describe_scene({direction}): vision call failed: {exc}')
            return (
                "I have a picture, but I'm afraid I could not make sense of it. "
                "My visual processing appears to be having difficulties."
            )

        node.get_logger().info(
            f'describe_scene({direction}, views={[c for c, _ in views]}): '
            f'{description}')
        # Returned as an observation rather than a finished line, so Gemini renders
        # it in OMNI's voice instead of reading a flat report aloud.
        return f'{lead_in} {description}'

    def _remember_person(self, args: dict) -> str:
        # Keep the identity a single clean token (first name).
        raw = args.get('name', '').strip().lower()
        name = ''.join(ch for ch in raw if ch.isalnum() or ch in ('_', '-'))
        if not name:
            return "I didn't quite catch a name to remember. Might I ask it again?"
        ok = self._node.learn_person(name)
        if ok:
            return (
                f"A pleasure — I shall remember your face now, {name.capitalize()}. "
                f"Do go on."
            )
        return (
            f"I should like to remember you as {name.capitalize()}, but face "
            f"recognition is not available at the moment."
        )
