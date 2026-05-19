"""
audio_handler.py — Mic capture and speaker playback for behavior_node.

DEVICE LIFECYCLE (important — read before editing):
  The USB mic (device 0) can only be opened by ONE reader at a time under ALSA.
  WakeWordDetector opens the mic while OMNI is IDLE (listening for wake word).
  AudioHandler opens the mic only when a Gemini session is active.
  These two must NEVER run at the same time.

  Correct order when wake word fires:
    1. WakeWordDetector stops (sets _running=False, closes its stream)
    2. behavior_node calls audio_handler.start_capture()
    3. behavior_node calls bridge.open_session()

  Correct order when returning to IDLE:
    1. bridge.close_session() finishes
    2. behavior_node calls audio_handler.stop_capture()
    3. behavior_node calls wake_word.start()

  The speaker thread (audio-play) is NOT affected by this — it holds a separate
  ALSA OUTPUT handle and can stay open continuously.

TCP MIC MODE (tcp_mic_port != 0):
  Instead of a local sounddevice capture, a TCP server listens for a Pi Zero
  running mic_bridge.py to push raw S16_LE 16kHz stereo PCM. The stereo stream
  is averaged to mono before entering the queue. The TCP server thread runs
  continuously (started in start()), so the Pi Zero can stay connected across
  multiple Gemini sessions. start_capture()/stop_capture() gate whether incoming
  chunks are enqueued or discarded — they do NOT tear down the TCP connection.
  WakeWordDetector still uses the local ALSA mic (device mic_device_index) in
  this mode; set that to a valid local mic index for wake word detection to work.

gemini_bridge.py uses three methods:
  get_mic_chunk()    → bytes   blocking, returns the next 100ms of mic audio
  play_audio(bytes)  → None    enqueue PCM for speaker playback
  stop()             → None    graceful shutdown of both threads

behavior_node.py also uses:
  start_capture()    → None    open mic (or enable TCP gate) and start capture
  stop_capture()     → None    signal capture thread to stop (or close TCP gate)
"""

import queue
import socket
import threading
import time

import numpy as np
import sounddevice as sd

# ── Audio constants ────────────────────────────────────────────────────────────
# These match audio_node.py exactly so the two nodes are consistent.
_PCM_IN_RATE   = 16_000   # Gemini expects 16kHz mono PCM from the mic
_PCM_OUT_RATE  = 24_000   # Gemini streams 24kHz PCM to us
_PCM_DEV_RATE  = 44_100   # USB audio device native rate (must resample to this)
_CHUNK_FRAMES  = 1_600    # 100ms at 16kHz — one send unit for Gemini
_INT16_MAX     = 32_768.0

# Pre-built silence chunk: 100ms of zeros at 16kHz, int16, mono = 3200 bytes.
# Sent instead of real mic audio while the speaker is playing (echo suppression).
_SILENCE_CHUNK = bytes(_CHUNK_FRAMES * 2)


class AudioHandler:
    """
    Owns the sounddevice streams and the two audio threads.
    Constructed by behavior_node.py and passed to GeminiBridge.
    """

    def __init__(self, mic_device: int, speaker_device: int, logger, tcp_mic_port: int = 0):
        """
        mic_device     — sounddevice index for USB mic (ROS2 param mic_device_index)
        speaker_device — sounddevice index for USB speaker (ROS2 param speaker_device_index)
        logger         — rclpy logger so we can use self.get_logger() style logs
        tcp_mic_port   — if nonzero, accept stereo PCM from Pi Zero on this TCP port
                         instead of opening a local sounddevice input stream
        """
        self._mic_device     = mic_device
        self._speaker_device = speaker_device
        self._tcp_mic_port   = tcp_mic_port
        self._log            = logger

        # Thread-safe queues.
        # _mic_queue holds raw 16kHz int16 bytes ready for Gemini.
        # _play_queue holds 24kHz int16 bytes coming back from Gemini.
        self._mic_queue  = queue.Queue(maxsize=10)   # drop old chunks if bridge is slow
        self._play_queue = queue.Queue()

        # threading.Event is safe to set/clear/check from any thread without a lock.
        # Set   → speaker is currently playing; mic should send silence.
        # Clear → speaker is idle; mic should send real audio.
        self._muted = threading.Event()

        # Separate flags so capture and playback can be started/stopped independently.
        self._capture_running = False
        self._speaker_running = False

        # Monotonic timestamp of when the play queue last drained fully.
        # Used by in_suppression_window() so the wake word detector can ignore
        # audio that arrives shortly after OMNI finishes speaking.
        self._last_play_end_time = 0.0

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self):
        """
        Start the speaker playback thread only.
        Do NOT start capture here — the mic is owned by WakeWordDetector while
        OMNI is IDLE. Call start_capture() only after the wake word fires and
        the detector has stopped.
        """
        self._speaker_running = True
        self._play_thread = threading.Thread(
            target=self._play_loop,
            name='audio-play',
            daemon=True,
        )
        self._play_thread.start()
        self._log.info(f'AudioHandler speaker thread started — device {self._speaker_device}')

        if self._tcp_mic_port:
            self._tcp_thread = threading.Thread(
                target=self._tcp_capture_loop,
                name='audio-tcp',
                daemon=True,
            )
            self._tcp_thread.start()
            self._log.info(f'AudioHandler TCP mic server started — port {self._tcp_mic_port}')

    def start_capture(self):
        """
        Open the USB mic and start the capture thread.
        Call AFTER WakeWordDetector has stopped (its ALSA stream must be closed
        first). Safe to call multiple times — creates a new thread each time.
        """
        if self._capture_running:
            self._log.warn('start_capture() called but capture is already running — ignoring')
            return
        self._muted.clear()   # reset echo-suppression gate for new session
        self._capture_running = True
        if self._tcp_mic_port:
            # TCP server thread is already running — just open the gate
            self._log.info('AudioHandler capture gate open — TCP mic active')
            return
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name='audio-capture',
            daemon=True,
        )
        self._capture_thread.start()
        self._log.info(f'AudioHandler capture thread started — mic device {self._mic_device}')

    def stop_capture(self):
        """
        Signal the capture thread to stop and release the ALSA input device.
        Returns immediately — the thread exits on its next loop iteration.
        Call this before starting WakeWordDetector so the mic is free.
        """
        self._capture_running = False
        # Unblock get_mic_chunk() if it is waiting inside the queue
        self._mic_queue.put(b'')
        self._log.info('AudioHandler capture stopping')

    def get_mic_chunk(self) -> bytes:
        """
        Blocking call — waits until a 100ms mic chunk is ready.
        Returns the raw chunk regardless of mute state — the send loop in
        gemini_bridge decides whether to transmit based on is_playing().
        Called by gemini_bridge._send_audio_loop() in a tight loop.
        """
        return self._mic_queue.get()   # blocks until capture thread puts one in

    def play_audio(self, pcm_bytes: bytes):
        """
        Enqueue a chunk of 24kHz int16 mono PCM for speaker playback.
        Also sets _muted so the capture thread sends silence while we speak.
        Called by gemini_bridge._recv_loop() for every audio chunk from Gemini.
        """
        was_playing = self._muted.is_set()
        self._muted.set()
        if not was_playing:
            self._log.info('Speaker gate: CLOSED — mic blocked, playback starting')
        qsize = self._play_queue.qsize()
        if qsize > 5:
            self._log.warn(f'play_audio(): play queue backing up — depth {qsize}')
        self._play_queue.put(pcm_bytes)

    def stop(self):
        """Signal both threads to stop. Returns immediately; threads finish on their own."""
        self._capture_running = False
        self._speaker_running = False
        self._muted.clear()
        # Unblock get_mic_chunk() if it is waiting
        self._mic_queue.put(b'')
        self._log.info('AudioHandler stopping')

    def is_playing(self) -> bool:
        """
        Returns True while the speaker is actively playing audio.
        Used by gemini_bridge._send_audio_loop to gate mic transmission —
        don't send audio to Gemini while OMNI is speaking (echo prevention).
        Safe to call from any thread.
        """
        return self._muted.is_set()

    def in_suppression_window(self, window: float = 2.0) -> bool:
        """
        Returns True if the wake word detector should ignore detections.
        True while the speaker is still playing (muted event set) OR within
        `window` seconds of the play queue draining — prevents OMNI's voice
        from echoing back through the mic and re-triggering the wake word.
        Safe to call from any thread.
        """
        if self._muted.is_set():
            return True
        return time.monotonic() - self._last_play_end_time < window

    # ── Internal threads ───────────────────────────────────────────────────────

    def _capture_loop(self):
        """
        Reads from the USB mic in 100ms blocks.
        sounddevice.RawInputStream gives us int16 bytes directly —
        no conversion needed before sending to Gemini.
        """
        try:
            with sd.RawInputStream(
                samplerate=_PCM_IN_RATE,
                blocksize=_CHUNK_FRAMES,
                dtype='int16',
                channels=1,
                device=self._mic_device,
            ) as stream:
                self._log.info('Mic capture stream open')
                while self._capture_running:
                    data, overflowed = stream.read(_CHUNK_FRAMES)
                    if overflowed:
                        # Overflows happen when the CPU is briefly busy.
                        # Log once but do not crash — Gemini tolerates small gaps.
                        self._log.warn('Mic buffer overflow — audio gap')
                    try:
                        self._mic_queue.put_nowait(bytes(data))
                    except queue.Full:
                        # Bridge is not consuming fast enough. Drop the oldest chunk.
                        try:
                            self._mic_queue.get_nowait()
                        except queue.Empty:
                            pass
                        self._mic_queue.put_nowait(bytes(data))
        except Exception as exc:
            self._log.error(f'Mic capture thread crashed: {exc}')

    def _play_loop(self):
        """
        Reads 24kHz int16 PCM chunks from _play_queue, resamples to 44100Hz,
        and writes to the USB speaker.

        Resampling: linear interpolation via numpy. Not audiophile quality but
        perfectly fine for speech. A proper library (soxr, librosa) could be
        used later if the output sounds wrong.

        _muted is cleared after the queue drains — this is the signal to
        the capture thread to resume sending real mic audio.
        """
        try:
            with sd.RawOutputStream(
                samplerate=_PCM_DEV_RATE,
                dtype='int16',
                channels=1,
                device=self._speaker_device,
            ) as stream:
                self._log.info('Speaker playback stream open')
                while self._speaker_running:
                    try:
                        # Wait up to 0.5s for a chunk. If nothing arrives,
                        # clear _muted so the mic comes back online.
                        chunk = self._play_queue.get(timeout=0.5)
                    except queue.Empty:
                        # 0.5s elapsed with no new chunk — speaker is idle.
                        # Backup clear path: post-write check below handles the common case.
                        if self._muted.is_set():
                            self._last_play_end_time = time.monotonic()
                            self._muted.clear()
                            self._log.info('Speaker gate: OPEN (0.5s idle timeout)')
                        continue

                    if not chunk:
                        self._log.info('_play_loop: dequeued empty chunk — skipping')
                        continue

                    resampled = self._resample(chunk)
                    stream.write(resampled)
                    # Primary clear path: as soon as write() returns, check whether
                    # the queue is empty. If so, clear _muted immediately rather than
                    # waiting for the 0.5s queue.Empty timeout. This matters when
                    # Gemini sends the greeting as one large chunk — stream.write()
                    # can block for the full audio duration, so by the time it
                    # returns the queue has already been empty for a while.
                    if self._play_queue.empty() and self._muted.is_set():
                        self._last_play_end_time = time.monotonic()
                        self._muted.clear()
                        self._log.info('Speaker gate: OPEN (queue empty after write)')

        except Exception as exc:
            self._log.error(f'Speaker playback thread crashed: {exc}')

    def _tcp_capture_loop(self):
        """
        TCP server for Pi Zero mic bridge (mic_bridge.py).
        Accepts one client at a time. Incoming stream is S16_LE 16kHz stereo;
        each 100ms chunk (6400 bytes) is averaged to mono (3200 bytes) before
        being placed on _mic_queue.

        Runs for the lifetime of the node (started in start()). Data is enqueued
        only when _capture_running is True — during IDLE the Pi Zero stays
        connected but chunks are discarded so the Gemini session doesn't receive
        stale audio when it reopens.
        """
        # Stereo chunk: 1600 samples × 2 channels × 2 bytes/sample = 6400 bytes
        STEREO_CHUNK = _CHUNK_FRAMES * 2 * 2

        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(('', self._tcp_mic_port))
        srv.listen(1)
        srv.settimeout(1.0)   # so accept() doesn't block node shutdown
        self._log.info(f'TCP mic server listening on :{self._tcp_mic_port}')

        try:
            while self._speaker_running:
                # Wait for Pi Zero to connect (or reconnect after a drop)
                try:
                    conn, addr = srv.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break

                self._log.info(f'TCP mic: connected from {addr[0]}:{addr[1]}')
                conn.settimeout(1.0)
                buf = bytearray()

                try:
                    while self._speaker_running:
                        try:
                            data = conn.recv(STEREO_CHUNK * 4)
                        except socket.timeout:
                            continue
                        if not data:
                            break
                        buf.extend(data)

                        while len(buf) >= STEREO_CHUNK:
                            chunk = bytes(buf[:STEREO_CHUNK])
                            del buf[:STEREO_CHUNK]

                            if not self._capture_running:
                                continue   # session not active — discard

                            mono = self._stereo_to_mono(chunk)
                            try:
                                self._mic_queue.put_nowait(mono)
                            except queue.Full:
                                try:
                                    self._mic_queue.get_nowait()
                                except queue.Empty:
                                    pass
                                self._mic_queue.put_nowait(mono)
                except Exception as exc:
                    self._log.error(f'TCP mic client error: {exc}')
                finally:
                    try:
                        conn.close()
                    except OSError:
                        pass

                self._log.warn('TCP mic: Pi Zero disconnected — waiting for reconnect')
        finally:
            srv.close()

    @staticmethod
    def _stereo_to_mono(stereo_bytes: bytes) -> bytes:
        """Extract left channel only from S16_LE stereo PCM.
        ReSpeaker mics are out of phase — averaging L+R cancels to near-zero."""
        samples = np.frombuffer(stereo_bytes, dtype=np.int16)
        return samples[0::2].tobytes()

    @staticmethod
    def _resample(pcm_24k: bytes) -> bytes:
        """
        Convert 24kHz int16 mono PCM → 44100Hz int16 mono PCM.
        Uses numpy linear interpolation. Fast enough for real-time speech on Pi 5.
        """
        samples_in  = np.frombuffer(pcm_24k, dtype=np.int16).astype(np.float32)
        n_in        = len(samples_in)
        # Calculate exactly how many output samples we need for this chunk
        n_out       = int(np.round(n_in * _PCM_DEV_RATE / _PCM_OUT_RATE))
        x_in        = np.linspace(0, n_in - 1, n_in)
        x_out       = np.linspace(0, n_in - 1, n_out)
        samples_out = np.interp(x_out, x_in, samples_in)
        return samples_out.astype(np.int16).tobytes()
