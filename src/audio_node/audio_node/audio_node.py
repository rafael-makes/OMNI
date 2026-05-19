"""
OMNI audio_node
Direct USB audio capture and playback via Gemini Live API.

Captures microphone audio from the USB audio device, streams it to the
Gemini Live API WebSocket, and plays the response back through the same
USB device. No Pi Zero W or TCP bridge involved.

USB device auto-detected by scanning for 'usb' in the device name, or
set explicitly via the 'audio_device' ROS parameter.

ROS2 interface:
  Publishes:   /audio/text   (std_msgs/String)            — Gemini response transcription
               /audio/levels (std_msgs/Float32MultiArray) — 16-band mic RMS at 10 Hz
  Subscribes:  /robot_state  (std_msgs/String)            — current robot state
"""

import asyncio
import os
import queue
import threading
import time

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

try:
    import sounddevice as sd
    _SD_OK = True
except OSError:
    _SD_OK = False

try:
    from google import genai
    from google.genai import types as genai_types
    _GENAI_OK = True
except ImportError:
    _GENAI_OK = False

_PCM_IN_RATE   = 16000
_PCM_OUT_RATE  = 24000   # Gemini streams 24 kHz audio
_PCM_DEV_RATE  = 44100   # USB device native rate — resample before writing
_CHUNK_FRAMES  = 1600          # 100 ms at 16 kHz
_LEVELS_BANDS  = 16
_INT16_MAX     = 32768.0
_EMA_ALPHA     = 0.3
_SILENCE_CHUNK = bytes(_CHUNK_FRAMES * 2)   # pre-built zeros for mute-while-speaking


class AudioNode(Node):

    def __init__(self):
        super().__init__('audio_node')

        self.declare_parameter('audio_device', '')
        self.declare_parameter('audio_levels_hz', 10.0)
        self.declare_parameter('gemini_model', 'models/gemini-2.5-flash-native-audio-latest')
        self.declare_parameter(
            'personality_config',
            os.path.expanduser('~/omni_ws/config/personality.yaml'),
        )

        audio_device_param = self.get_parameter('audio_device').value
        audio_levels_hz    = self.get_parameter('audio_levels_hz').value
        self._gemini_model = self.get_parameter('gemini_model').value
        personality_path   = self.get_parameter('personality_config').value

        self._system_prompt = self._load_personality(personality_path)

        api_key = os.environ.get('GEMINI_API_KEY', '')
        if not api_key:
            self.get_logger().error('GEMINI_API_KEY not set — Gemini will not connect')
        else:
            self.get_logger().info(f'GEMINI_API_KEY loaded ({len(api_key)} chars)')
        self._api_key = api_key

        self._audio_device = self._resolve_device(audio_device_param)

        # ── Publishers / subscribers ──────────────────────────────────────────
        self._text_pub   = self.create_publisher(String,            '/audio/text',   10)
        self._levels_pub = self.create_publisher(Float32MultiArray, '/audio/levels', 10)
        self.create_subscription(String, '/robot_state', self._state_cb, 10)
        self.create_timer(1.0 / audio_levels_hz, self._levels_timer_cb)

        # ── Shared state ──────────────────────────────────────────────────────
        self._robot_state  = ''
        self._audio_levels = [0.0] * _LEVELS_BANDS
        self._levels_lock  = threading.Lock()
        self._running      = True
        self._playing      = threading.Event()  # set while speaker is active → mutes mic

        # ── Queues ────────────────────────────────────────────────────────────
        self._loop: asyncio.AbstractEventLoop | None = None
        self._async_mic_queue: asyncio.Queue | None  = None
        self._response_queue: queue.Queue            = queue.Queue()  # unbounded — Gemini generates faster than real-time; capping drops chunks → audio skips

        # ── Background threads ────────────────────────────────────────────────
        threading.Thread(target=self._gemini_thread_func, daemon=True, name='audio-gemini').start()
        threading.Thread(target=self._capture_loop,       daemon=True, name='audio-capture').start()
        threading.Thread(target=self._playback_loop,      daemon=True, name='audio-play').start()

        self.get_logger().info(
            f'audio_node started — device={self._audio_device!r}, '
            f'model {self._gemini_model}, levels {audio_levels_hz:.0f} Hz'
        )

    # ── Device detection ──────────────────────────────────────────────────────

    def _resolve_device(self, param: str):
        if param:
            try:
                return int(param)
            except ValueError:
                return param

        if not _SD_OK:
            return None

        try:
            for i, dev in enumerate(sd.query_devices()):
                if 'usb' in dev['name'].lower() and dev['max_input_channels'] > 0:
                    self.get_logger().info(
                        f'USB audio device auto-detected: [{i}] {dev["name"]}'
                    )
                    return i
        except Exception as e:
            self.get_logger().warn(f'Device query failed: {e}')

        self.get_logger().info('No USB device found — using system default')
        return None

    # ── Personality ───────────────────────────────────────────────────────────

    def _load_personality(self, path: str) -> str:
        try:
            with open(os.path.expanduser(path)) as f:
                data = yaml.safe_load(f)
            prompt = data['omni']['system_prompt']
            self.get_logger().info(f'System prompt loaded: {len(prompt)} chars')
            return prompt
        except Exception as e:
            self.get_logger().error(f'Failed to load personality ({e}) — using default')
            return 'You are OMNI, a friendly autonomous indoor robot.'

    # ── ROS callbacks ─────────────────────────────────────────────────────────

    def _state_cb(self, msg: String):
        self._robot_state = msg.data.strip()

    def _levels_timer_cb(self):
        with self._levels_lock:
            levels = list(self._audio_levels)
        out = Float32MultiArray()
        out.data = levels
        self._levels_pub.publish(out)

    # ── Audio levels ──────────────────────────────────────────────────────────

    def _update_audio_levels(self, pcm_bytes: bytes):
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        if len(samples) < _LEVELS_BANDS:
            return
        band_size = len(samples) // _LEVELS_BANDS
        bands = [
            min(1.0, float(np.sqrt(np.mean(
                samples[i * band_size:(i + 1) * band_size] ** 2
            ))) / _INT16_MAX)
            for i in range(_LEVELS_BANDS)
        ]
        with self._levels_lock:
            for i in range(_LEVELS_BANDS):
                self._audio_levels[i] = (
                    _EMA_ALPHA * bands[i] + (1.0 - _EMA_ALPHA) * self._audio_levels[i]
                )

    # ── Capture thread ────────────────────────────────────────────────────────

    def _capture_loop(self):
        """Read mic → feed Gemini. Sends silence while response audio is playing."""
        if not _SD_OK:
            self.get_logger().error('sounddevice not available — capture disabled')
            return

        while self._running:
            try:
                with sd.InputStream(
                    samplerate=_PCM_IN_RATE,
                    channels=1,
                    dtype='int16',
                    blocksize=_CHUNK_FRAMES,
                    device=self._audio_device,
                ) as stream:
                    self.get_logger().info(
                        f'Mic open: {_PCM_IN_RATE} Hz, device={self._audio_device!r}'
                    )
                    while self._running:
                        pcm, _ = stream.read(_CHUNK_FRAMES)
                        if self._playing.is_set():
                            raw = _SILENCE_CHUNK          # mute mic during playback
                        else:
                            raw = pcm.tobytes()
                            self._update_audio_levels(raw)

                        if self._loop is not None and self._async_mic_queue is not None:
                            asyncio.run_coroutine_threadsafe(
                                self._async_mic_queue.put(raw), self._loop
                            )
            except Exception as e:
                self.get_logger().warn(f'Mic error ({e}) — retrying in 2s')
                if self._running:
                    time.sleep(2.0)

    # ── Playback thread ───────────────────────────────────────────────────────

    def _playback_loop(self):
        """Play Gemini response audio. Sets _playing while active to mute mic."""
        if not _SD_OK:
            self.get_logger().error('sounddevice not available — playback disabled')
            return

        while self._running:
            try:
                with sd.OutputStream(
                    samplerate=_PCM_DEV_RATE,
                    channels=1,
                    dtype='int16',
                    device=self._audio_device,
                ) as stream:
                    self.get_logger().info(
                        f'Speaker open: {_PCM_DEV_RATE} Hz (resampled from {_PCM_OUT_RATE} Hz), '
                        f'device={self._audio_device!r}'
                    )
                    play_until = 0.0   # monotonic deadline when last written chunk finishes

                    while self._running:
                        try:
                            audio_bytes = self._response_queue.get(timeout=1.0)
                            self._playing.set()
                            self._update_audio_levels(audio_bytes)  # drive chest bars to OMNI's voice
                            # Resample 24 kHz → 44.1 kHz via linear interpolation.
                            # 24000:44100 = 80:147 — not a clean ratio, so we must
                            # resample in software rather than rely on the device.
                            src = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                            n_out = int(round(len(src) * _PCM_DEV_RATE / _PCM_OUT_RATE))
                            x_in  = np.arange(len(src))
                            x_out = np.linspace(0, len(src) - 1, n_out)
                            pcm = np.interp(x_out, x_in, src).astype(np.int16).reshape(-1, 1)
                            # Track how long until this audio will have finished playing
                            chunk_secs = len(pcm) / _PCM_DEV_RATE
                            play_until = max(play_until, time.monotonic()) + chunk_secs
                            stream.write(pcm)
                        except queue.Empty:
                            # Queue drained — wait for hardware buffer to finish playing
                            # before unmuting the mic (stream.write is non-blocking;
                            # audio lives in the device buffer after write() returns)
                            drain_deadline = play_until + stream.latency
                            remaining = drain_deadline - time.monotonic()
                            if remaining > 0:
                                time.sleep(remaining)
                            self._playing.clear()
                            play_until = 0.0
            except Exception as e:
                self.get_logger().warn(f'Speaker error ({e}) — retrying in 2s')
                if self._running:
                    time.sleep(2.0)

    # ── Gemini asyncio thread ─────────────────────────────────────────────────

    def _gemini_thread_func(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        loop.run_until_complete(self._gemini_main())

    async def _gemini_main(self):
        backoff = 1.0
        while self._running:
            self._async_mic_queue = asyncio.Queue()
            try:
                await self._gemini_session()
                backoff = 1.0
            except Exception as e:
                self.get_logger().warn(
                    f'Gemini session ended ({e}) — reconnecting in {backoff:.0f}s'
                )
                self._async_mic_queue = None
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)

    async def _gemini_session(self):
        if not _GENAI_OK:
            self.get_logger().error('google-genai not installed — sleeping 60s')
            await asyncio.sleep(60.0)
            return
        if not self._api_key:
            self.get_logger().error('No GEMINI_API_KEY — sleeping 60s')
            await asyncio.sleep(60.0)
            return

        client = genai.Client(api_key=self._api_key)
        config = genai_types.LiveConnectConfig(
            response_modalities=['AUDIO'],
            system_instruction=self._system_prompt,
            output_audio_transcription=genai_types.AudioTranscriptionConfig(),
            speech_config=genai_types.SpeechConfig(
                voice_config=genai_types.VoiceConfig(
                    prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                        voice_name='Algieba',
                    )
                )
            ),
        )

        async with client.aio.live.connect(model=self._gemini_model, config=config) as session:
            self.get_logger().info('Gemini Live connected')
            await asyncio.gather(
                self._send_audio_loop(session),
                self._recv_response_loop(session),
            )

    async def _send_audio_loop(self, session):
        while True:
            chunk: bytes = await self._async_mic_queue.get()
            await session.send_realtime_input(
                audio=genai_types.Blob(
                    data=chunk,
                    mime_type=f'audio/pcm;rate={_PCM_IN_RATE}',
                )
            )

    async def _recv_response_loop(self, session):
        while True:
            async for response in session.receive():
                if response.data:
                    self._response_queue.put_nowait(response.data)
                sc = response.server_content
                if sc and sc.output_transcription and sc.output_transcription.text:
                    text = sc.output_transcription.text.strip()
                    if text:
                        self._text_pub.publish(String(data=text))
                        self.get_logger().info(f'Gemini: {text}')

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def destroy_node(self):
        self._running = False
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AudioNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
