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

import yaml
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from google.genai import types as genai_types
from nav2_msgs.action import NavigateToPose

# ── Valid robot states ─────────────────────────────────────────────────────────
# Kept here as the single source of truth — imported by behavior_node.py too.
VALID_STATES = {'IDLE', 'LISTENING', 'SPEAKING', 'NAVIGATING', 'EXPLORING', 'DOCKING', 'ERROR'}

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
                    'Change OMNI\'s operating state. Call this to signal what '
                    'OMNI is currently doing — SPEAKING when talking, LISTENING '
                    'when waiting for input, IDLE when conversation ends.'
                ),
                parameters=genai_types.Schema(
                    type=genai_types.Type.OBJECT,
                    properties={
                        'state': genai_types.Schema(
                            type=genai_types.Type.STRING,
                            description=(
                                f'One of: {", ".join(sorted(VALID_STATES))}. '
                                'Use IDLE when the conversation is fully finished.'
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

    def handle(self, function_name: str, args: dict) -> str:
        """
        Dispatch a Gemini function call to the right handler.
        Returns the string result that Gemini will use in its next response.
        Never raises — unknown function names return a graceful in-character message.
        """
        handlers = {
            'set_robot_state':  self._set_robot_state,
            'navigate_to':      self._navigate_to,
            'stop_navigation':  self._stop_navigation,
            'report_status':    self._report_status,
            'explore_area':     self._explore_area,
            'save_location':    self._save_location,
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
        if state not in VALID_STATES:
            self._node.get_logger().warn(f'Gemini requested invalid state: {state!r}')
            return (
                f"I must inform you that '{state}' is not a recognised operating state. "
                f"I shall remain in my current configuration."
            )
        self._node._set_state(state)
        return f'State set to {state}.'

    def _navigate_to(self, args: dict) -> str:
        location = args.get('location', '').lower().strip()
        locations = self._node._locations   # dict loaded from omni_config.yaml

        if location not in locations:
            known = ', '.join(sorted(locations.keys())) if locations else 'none programmed yet'
            self._node.get_logger().info(
                f'navigate_to called for unknown location: {location!r} '
                f'(known: {known})'
            )
            return (
                f"I'm afraid '{location}' is not in my navigation database. "
                f"I cannot, in good conscience, simply wander off without a known destination. "
                f"{'Known locations are: ' + known + '.' if locations else 'No locations have been programmed yet.'}"
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

        coords = locations[location]   # [x, y, yaw_degrees]
        x, y   = float(coords[0]), float(coords[1])
        yaw_deg = float(coords[2]) if len(coords) > 2 else 0.0

        goal_msg      = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp    = self._node.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0

        # For 2D navigation only the z and w quaternion components matter (yaw only).
        yaw_rad = math.radians(yaw_deg)
        goal_msg.pose.pose.orientation.z = math.sin(yaw_rad / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(yaw_rad / 2.0)

        self._node.get_logger().info(
            f'Sending navigation goal: {location} → x={x}, y={y}, yaw={yaw_deg}°'
        )
        self._node._set_state('NAVIGATING')

        # send_goal_async is thread-safe — schedules the goal on the ROS2 executor.
        # The callback stores the goal handle so stop_navigation() can cancel it.
        send_future = self._node._nav_action_client.send_goal_async(
            goal_msg,
            feedback_callback=self._node._nav_feedback_callback,
        )
        send_future.add_done_callback(self._node._nav_goal_response_callback)

        return (
            f'Very well, I am setting a course for the {location}. '
            f'Navigation initiated. I shall proceed with all due care.'
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

        # Create a one-shot subscription to /amcl_pose and wait up to 3 seconds.
        # The callback runs on the ROS2 executor thread; we block here (thread-pool
        # executor) on a threading.Event so the two threads coordinate safely.
        pose_event  = threading.Event()
        holder      = [None]  # mutable container so the nested callback can write to it

        def _pose_cb(msg):
            holder[0] = msg
            pose_event.set()

        sub = self._node.create_subscription(
            PoseWithCovarianceStamped,
            '/pose',
            _pose_cb,
            1,
        )
        try:
            received = pose_event.wait(timeout=3.0)
        finally:
            self._node.destroy_subscription(sub)

        if not received:
            self._node.get_logger().warn(
                'save_location: timed out waiting for /pose — '
                'is slam_toolbox running in localization mode?'
            )
            return (
                "I'm afraid my localisation system does not appear to be running — "
                "no pose was received from slam_toolbox after three seconds. "
                "The slam node must be active and localised before I can save a location."
            )

        pose_msg = holder[0]

        # pose.covariance is a flat 36-element row-major 6×6 matrix.
        # Diagonal indices: x=0, y=7, yaw=35.  Threshold of 0.5 m²/rad².
        cov     = pose_msg.pose.covariance
        cov_x   = cov[0]
        cov_y   = cov[7]
        cov_yaw = cov[35]
        _COV_THRESHOLD = 0.5
        if max(cov_x, cov_y, cov_yaw) >= _COV_THRESHOLD:
            self._node.get_logger().warn(
                f'save_location: covariance too high — '
                f'cov_x={cov_x:.3f}, cov_y={cov_y:.3f}, cov_yaw={cov_yaw:.3f}'
            )
            return (
                "I must warn you — my localisation is currently rather uncertain, "
                "with a covariance I find most alarming. "
                "I cannot, in good conscience, save an unreliable position. "
                "Please allow the localisation to settle and try again."
            )

        # Extract x, y, and yaw from the pose.
        p       = pose_msg.pose.pose.position
        ori     = pose_msg.pose.pose.orientation
        x       = p.x
        y       = p.y
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
