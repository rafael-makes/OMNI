"""
wake_word.py — Local wake word detection for behavior_node.

Listens on the USB mic for the wake word and fires a callback when detected.
Runs in its own daemon thread. Uses openwakeword with the hey_mycroft model
(closest available phonetic match to "hey omni"). A custom hey_omni model
can be trained with openwakeword's training tools and dropped in via the
wake_word_model ROS2 parameter.

openwakeword is STRICT about input size: exactly 1280 samples at 16kHz
(80ms) per call to Model.predict(). We use blocksize=1280 on the sounddevice
stream so every read delivers exactly that — no accumulation needed.

If openwakeword fails to import or load, falls back to RMS energy threshold
detection. The fallback is noisier (false triggers on loud sounds) but keeps
OMNI functional without a reboot. A warning is logged explaining why.

This class opens its own mic stream and is never active at the same time as
AudioHandler — WakeWordDetector runs during IDLE, AudioHandler runs when the
Gemini stream is open. No device conflict.

Usage:
    detector = WakeWordDetector(
        callback=my_callback,
        mic_device=0,
        model_name='hey_mycroft',
        score_threshold=0.5,
        logger=self.get_logger(),
    )
    detector.start()   # begin listening
    ...
    detector.stop()    # stop listening (call before opening Gemini stream)
"""

import threading
import time

import numpy as np
import sounddevice as sd

# ── Audio constants ────────────────────────────────────────────────────────────
_SAMPLE_RATE   = 16_000   # openwakeword requires 16kHz input
_CHUNK_SAMPLES = 1_280    # openwakeword requires EXACTLY 1280 samples (80ms) per predict()
_CHUNK_BYTES   = _CHUNK_SAMPLES * 2   # int16 = 2 bytes per sample

# ── Energy fallback constants ──────────────────────────────────────────────────
# RMS level (0–32768 scale) above which a chunk counts as "loud".
# 800 is a reasonable default for a quiet indoor environment.
# Tune upward if OMNI triggers on background noise.
_ENERGY_THRESHOLD   = 800
# Number of consecutive loud chunks required before triggering wake.
# At 80ms per chunk, 4 chunks = 320ms of sustained sound.
# This prevents a single clap or door slam from waking OMNI.
_ENERGY_CONSEC_MIN  = 4


class WakeWordDetector:

    def __init__(
        self,
        callback,
        mic_device: int,
        model_name: str = 'hey_mycroft',
        score_threshold: float = 0.5,
        logger=None,
        audio_handler=None,
    ):
        """
        callback        — called (no arguments) when wake word detected.
                          Fired on the detector thread — behavior_node must
                          handle this in a thread-safe way.
        mic_device      — sounddevice index for the USB mic
        model_name      — openwakeword model to load (default 'hey_mycroft').
                          Must match one of the .onnx filenames in
                          openwakeword/resources/models/ (without extension).
        score_threshold — confidence score (0.0–1.0) above which we trigger.
                          Lower = more sensitive but more false positives.
        logger          — rclpy logger
        audio_handler   — AudioHandler instance; if provided, detections are
                          suppressed while the speaker is playing or within 2s
                          of playback ending (prevents OMNI's own voice re-triggering).
        """
        self._callback        = callback
        self._mic_device      = mic_device
        self._model_name      = model_name
        self._score_threshold = score_threshold
        self._log             = logger
        self._audio           = audio_handler
        self._running         = False

        # Decide which detection path to use
        self._use_oww = self._try_load_oww()

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self):
        """Open the mic and begin listening. Returns immediately."""
        self._running = True
        target = self._oww_loop if self._use_oww else self._energy_loop
        self._thread = threading.Thread(
            target=target,
            name='wake-word',
            daemon=True,
        )
        self._thread.start()
        mode = f'openwakeword ({self._model_name})' if self._use_oww else 'energy threshold fallback'
        self._log.info(f'Wake word detector started — mode: {mode}')

    def stop(self):
        """Signal the detector thread to stop. Returns immediately."""
        self._running = False
        self._log.info('Wake word detector stopped')

    # ── Setup ──────────────────────────────────────────────────────────────────

    def _try_load_oww(self) -> bool:
        """
        Attempt to import openwakeword and load the requested model.
        Returns True if successful, False if we must fall back to energy threshold.

        openwakeword 0.4.0 API notes:
          - Model() takes wakeword_model_paths (list of .onnx file paths), not model names.
          - No inference_framework parameter — 0.4.0 uses onnx directly.
          - Prediction keys are filenames without extension (e.g. 'hey_mycroft_v0.1').
          - We find the matching key by checking if model_name appears as a substring.
        """
        try:
            import os
            from openwakeword.model import Model  # noqa: PLC0415

            # Find the .onnx file for the requested model name
            resources_dir = os.path.join(
                os.path.dirname(os.path.abspath(Model.__module__.replace('.', '/'))),
                'resources', 'models',
            )
            # Fallback: search site-packages directly
            import openwakeword
            resources_dir = os.path.join(
                os.path.dirname(os.path.abspath(openwakeword.__file__)),
                'resources', 'models',
            )
            matches = [
                os.path.join(resources_dir, f)
                for f in os.listdir(resources_dir)
                if f.endswith('.onnx') and self._model_name in f
            ]
            if not matches:
                raise FileNotFoundError(
                    f"No .onnx file containing '{self._model_name}' found in {resources_dir}. "
                    f"Available: {[f for f in os.listdir(resources_dir) if f.endswith('.onnx')]}"
                )
            model_path = matches[0]

            # wakeword_model_paths takes full file paths, not model names
            oww_model = Model(wakeword_model_paths=[model_path])

            # Do one dummy predict to discover the prediction key
            # (key is the filename without .onnx, e.g. 'hey_mycroft_v0.1')
            dummy = np.zeros(_CHUNK_SAMPLES, dtype=np.int16)
            preds = oww_model.predict(dummy)

            # Find the key that matches our model name (substring match handles version suffix)
            matching_keys = [k for k in preds if self._model_name in k]
            if not matching_keys:
                raise KeyError(
                    f"No prediction key containing '{self._model_name}' found. "
                    f"Available keys: {list(preds.keys())}"
                )
            self._oww_key   = matching_keys[0]
            self._oww_model = oww_model
            self._log.info(
                f"openwakeword loaded — model file: {os.path.basename(model_path)}, "
                f"prediction key: '{self._oww_key}'"
            )
            return True
        except Exception as exc:
            self._log.warn(
                f'openwakeword failed to load ({exc}). '
                f'Falling back to energy threshold detection. '
                f'OMNI will still wake up but may trigger on loud sounds. '
                f'Fix: check that openwakeword is installed and the model name is correct.'
            )
            return False

    # ── Detection loops ────────────────────────────────────────────────────────

    def _oww_loop(self):
        """
        openwakeword detection loop.

        Reads exactly 1280-sample chunks from the mic (openwakeword's required
        input size — 80ms at 16kHz). Passes each chunk to Model.predict().
        When the score for our model exceeds the threshold, fires the callback
        and pauses listening until behavior_node calls stop().

        blocksize=1280 on the sounddevice stream guarantees every read()
        returns exactly 1280 samples. No accumulation buffer required.
        """
        wake_detected = False
        try:
            with sd.InputStream(
                samplerate=_SAMPLE_RATE,
                blocksize=_CHUNK_SAMPLES,   # exactly 1280 — strict openwakeword requirement
                dtype='int16',
                channels=1,
                device=self._mic_device,
            ) as stream:
                while self._running:
                    # read() returns (frames, overflowed). frames is a numpy array
                    # of shape (1280, 1) int16 — we flatten to 1D for openwakeword.
                    frames, _ = stream.read(_CHUNK_SAMPLES)
                    audio_1d  = frames[:, 0]   # shape (1280,) int16

                    preds = self._oww_model.predict(audio_1d)
                    # Use the discovered prediction key (e.g. 'hey_mycroft_v0.1'),
                    # not the raw model_name string ('hey_mycroft')
                    score = preds.get(self._oww_key, 0.0)

                    if score >= self._score_threshold:
                        # Suppress if OMNI was recently speaking — prevents its own
                        # audio from echoing through the mic and re-triggering.
                        if self._audio and self._audio.in_suppression_window():
                            self._log.debug(
                                f'Wake word suppressed (post-playback window) — '
                                f'score={score:.3f}'
                            )
                            continue
                        self._log.info(
                            f'Wake word detected — key={self._oww_key} '
                            f'score={score:.3f} threshold={self._score_threshold}'
                        )
                        self._running  = False   # exits the while loop
                        wake_detected  = True
                        # Do NOT call callback here — we are still inside the
                        # `with sd.InputStream` block, so the ALSA device is still
                        # held open. Callback fires AFTER the with block exits below.
        except Exception as exc:
            self._log.error(f'Wake word detection thread crashed: {exc}')

        # Stream is now closed — ALSA device 0 has been released.
        # Wait 100ms for the kernel to fully free the handle before audio_handler
        # opens the same device. Without this, start_capture() races the release.
        if wake_detected:
            time.sleep(0.1)
            self._callback()

    def _energy_loop(self):
        """
        Energy threshold fallback.

        Computes RMS of each 1280-sample chunk. Counts how many consecutive
        chunks exceed _ENERGY_THRESHOLD. When _ENERGY_CONSEC_MIN consecutive
        loud chunks are seen, treats it as a wake event.

        This will false-trigger on sustained loud sounds (music, TV, vacuum).
        It will NOT false-trigger on a single clap or door slam.
        Log a warning each time it triggers so the user can diagnose false positives.
        """
        consecutive_loud = 0
        wake_detected    = False
        try:
            with sd.InputStream(
                samplerate=_SAMPLE_RATE,
                blocksize=_CHUNK_SAMPLES,   # same 1280-sample chunk size for consistency
                dtype='int16',
                channels=1,
                device=self._mic_device,
            ) as stream:
                while self._running:
                    frames, _ = stream.read(_CHUNK_SAMPLES)
                    audio_1d  = frames[:, 0].astype(np.float32)
                    rms       = float(np.sqrt(np.mean(audio_1d ** 2)))

                    if rms >= _ENERGY_THRESHOLD:
                        consecutive_loud += 1
                        if consecutive_loud >= _ENERGY_CONSEC_MIN:
                            if self._audio and self._audio.in_suppression_window():
                                self._log.debug(
                                    f'Energy trigger suppressed (post-playback window) — '
                                    f'RMS={rms:.0f}'
                                )
                                consecutive_loud = 0
                            else:
                                self._log.warn(
                                    f'Wake triggered by energy fallback — '
                                    f'RMS={rms:.0f} (threshold={_ENERGY_THRESHOLD}). '
                                    f'This may be a false positive.'
                                )
                                consecutive_loud = 0
                                self._running    = False
                                wake_detected    = True
                            # Same rule as _oww_loop: do not call callback inside
                            # the with block — ALSA device must be released first.
                    else:
                        consecutive_loud = 0   # reset on any quiet chunk
        except Exception as exc:
            self._log.error(f'Energy fallback detection thread crashed: {exc}')

        # Stream closed — 100ms pause so ALSA fully releases device 0 before
        # audio_handler opens it in start_capture().
        if wake_detected:
            time.sleep(0.1)
            self._callback()
