"""OMNI head camera — USB webcam person detector (YOLO26n / TensorRT on the Jetson).

Publishes /camera/detections + /camera/status for the Pi's head_tracking_node.
The head cam is the HHWei USB webcam, opened by its stable /dev/v4l/by-id path so it
never swaps device nodes with the rear (2K) docking cam.

USB gotcha baked in here: capture_fps=30 (USB webcams max out at 30). Captures
848x480 (native MJPG mode, ~29fps), publishes in 1280x720 space so head_tracking's
pixel->angle mapping is unchanged.

Face recognition is ON (publish_identity): it publishes /camera/identity, which the
Pi's behavior_node uses to key OMNI's per-person memory — recognising who it is
talking to and recalling that person's memories. Without it OMNI treats everyone as a
stranger. Fail-safe: a missing SFace model or gallery disables identity only, leaving
person/face detection untouched. Pass publish_identity:=false to opt out.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_HEAD_CAM = ('/dev/v4l/by-id/'
             'usb-HHWei_Technology_Co.__Ltd._USB_Camera_HHW001-video-index0')
_REAR_CAM = ('/dev/v4l/by-id/'
             'usb-GENERAL_2K_HD_Camera-video-index0')


def generate_launch_description():
    usb_device = LaunchConfiguration('usb_device')
    rear_device = LaunchConfiguration('rear_device')
    publish_identity = LaunchConfiguration('publish_identity')
    scene_vision = LaunchConfiguration('scene_vision')
    rear_vision = LaunchConfiguration('rear_vision')
    clean_max_width = LaunchConfiguration('clean_max_width')
    capture_width = LaunchConfiguration('capture_width')
    detect_max_width = LaunchConfiguration('detect_max_width')
    capture_height = LaunchConfiguration('capture_height')

    return LaunchDescription([
        DeclareLaunchArgument(
            'usb_device', default_value=_HEAD_CAM,
            description='V4L2 by-id path of the head USB camera'),
        DeclareLaunchArgument(
            'publish_identity', default_value='true',
            description='Publish /camera/identity (SFace face recognition) for OMNI'
                        "'s per-person memory. false disables recognition."),
        DeclareLaunchArgument(
            'scene_vision', default_value='true',
            description='Publish the clean frame and run frame_server, so the Pi can '
                        'answer "what do you see?". false disables scene description.'),
        DeclareLaunchArgument(
            'rear_device', default_value=_REAR_CAM,
            description='V4L2 by-id path of the rear 2K camera'),
        DeclareLaunchArgument(
            'rear_vision', default_value='true',
            description='Run rear_camera so "what is behind you?" works. Set false '
                        'if you need to run the legacy dock_apriltag.launch.py, '
                        'which opens the rear device directly and cannot share it.'),
        # Capture size feeds YOLO, YuNet AND the clean scene-description feed off a
        # single stream, so raising it costs decode+resize on EVERY frame. Baseline at
        # 848x480: infer_ms 21.4 + face_ms 35.5 = ~57ms, comfortably inside the 10 Hz
        # detection target. Watch /camera/status when changing this — if the total
        # approaches 100ms, head tracking gets sluggish and face recognition (and so
        # per-person memory) degrades. Camera tops out at 1920x1080 MJPG.
        DeclareLaunchArgument('capture_width', default_value='1280'),
        DeclareLaunchArgument('capture_height', default_value='720'),
        DeclareLaunchArgument(
            'detect_max_width', default_value='848',
            description='Width YOLO/YuNet see, independent of capture. Holds face\n'
                        'detection at its 848 cost while the scene feed keeps full\n'
                        'resolution. 0 = detect at capture resolution.'),
        DeclareLaunchArgument(
            'clean_max_width', default_value='0',
            description='Longest edge of the frame served for scene description. '
                        '848 = the native capture width, i.e. no downscale at all. '
                        'Measured 2026-07-18 with an interleaved A/B on identical '
                        'frames: 848 costs only +0.22s median over 640 (0.99s vs '
                        '0.77s) for 1.8x the pixels, which is what lets OMNI read a '
                        'drink can label at desk distance instead of just seeing a '
                        'cylinder. Drop to 640 if latency ever gets tight. '
                        '0 = capture resolution, whatever that is.'),
        Node(
            package='head_detector',
            executable='head_detector_node',
            name='head_detector_node',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'source': 'usb',
                'usb_device': usb_device,
                'detect_max_width': detect_max_width,
                'capture_width': capture_width,
                'capture_height': capture_height,
                'capture_fps': 30,      # USB webcams cap at 30fps MJPG
                'publish_identity': publish_identity,
                # Undrawn frame on /camera/image_clean/compressed for frame_server.
                # NOT the annotated debug feed — a vision model handed that one
                # describes the detection boxes instead of the room.
                'publish_clean_image': scene_vision,
                'clean_image_fps': 2.0,
                'clean_max_width': clean_max_width,
            }],
        ),
        # Rear 2K cam owner, so describe_scene(direction='behind'|'all') has a
        # cached frame to serve. Streams MJPG continuously but only DECODES at
        # decode_fps — this camera cannot negotiate anything below 30fps on the
        # wire (asking for 5 fails the open outright), so the throttle has to be
        # downstream. Measured stable alongside the head cam and the OAK-D on the
        # shared USB 2.0 bus: 25.00 fps loaded vs 24.99 unloaded, zero USB errors.
        Node(
            package='rear_camera',
            executable='rear_camera_node',
            name='rear_camera',
            output='screen',
            emulate_tty=True,
            condition=IfCondition(rear_vision),
            parameters=[{
                'device': rear_device,
                'capture_width': 640,
                'capture_height': 480,
                'capture_fps': 30,      # the ONLY MJPG rate this cam negotiates
                'decode_fps': 5.0,
                'publish_clean_image': True,
                'clean_image_fps': 2.0,
                # Raw feed is for apriltag docking only — see
                # dock_apriltag_shared.launch.py. Off here to save CPU/DDS traffic.
                'publish_raw_image': False,
            }],
        ),
        # Serves the newest clean frame(s) as JPEG on /vision/get_camera_frame. The
        # Pi's behavior_node calls this across the 192.168.50.0/24 link when Gemini
        # invokes describe_scene. Subscribes Jetson-locally, so only the reply
        # crosses the wire. camera_id: head | rear | all.
        Node(
            package='frame_server',
            executable='frame_server_node',
            name='frame_server',
            output='screen',
            emulate_tty=True,
            condition=IfCondition(scene_vision),
            parameters=[{
                'head_topic': '/camera/image_clean/compressed',
                'rear_topic': '/camera/rear/image_clean/compressed',
                'rear_enabled': rear_vision,
                'max_frame_age': 5.0,
            }],
        ),
    ])
