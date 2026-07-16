"""
gemini_bridge.py — All Gemini Live API code for behavior_node.

behavior_node.py never imports google.genai directly — everything lives here.

THREAD MODEL (read this before editing):
  The asyncio event loop runs in a single daemon thread ('gemini-asyncio').
  ROS2 lives on a completely separate thread (the rclpy executor).
  These two worlds must not call each other's blocking APIs directly.

  Safe crossing points used here:
    ROS2 → asyncio:  loop.call_soon_threadsafe()  or  run_coroutine_threadsafe()
    asyncio → ROS2:  loop.run_in_executor()   — runs in a thread pool worker,
                     safe for rclpy publish() and send_goal_async() which are
                     internally thread-safe.

  If you ever see a direct rclpy publisher.publish() call inside a coroutine
  (not wrapped in run_in_executor), that is a bug waiting to happen.

SESSION LIFECYCLE:
  behavior_node calls open_session() when wake word fires.
  behavior_node calls close_session() when state moves to NAVIGATING/EXPLORING.
  If the WebSocket drops, we reconnect automatically after 5 seconds,
  but only resume full audio if the robot is IDLE or LISTENING.
  If mid-task (NAVIGATING, EXPLORING, ERROR, DOCKING), we wait until
  the state returns to IDLE before resuming.
"""

import asyncio
import datetime
import threading
import wave

import numpy as np
from google import genai
from google.genai import types as genai_types

from behavior_node.audio_handler import _PCM_IN_RATE
from behavior_node.function_handlers import OMNI_TOOLS
from behavior_node.memory_format import coalesce_transcript

_RECONNECT_DELAY   = 5.0    # seconds to wait before reconnecting after a drop
_STATES_BLOCK_AUDIO = {'NAVIGATING', 'EXPLORING', 'DOCKING', 'ERROR'}


class GeminiBridge:

    def __init__(self, node, audio_handler, function_handlers, system_prompt, model, voice,
                 on_activity=None):
        """
        node              — BehaviorNode reference (for state checks and logger)
        audio_handler     — AudioHandler instance (mic chunks in, speaker out)
        function_handlers — FunctionHandlers instance (dispatches Gemini tool calls)
        system_prompt     — string loaded from omni_config.yaml
        model             — Gemini model string e.g. 'gemini-2.5-flash'
        voice             — Gemini voice name e.g. 'Algieba'
        on_activity       — optional callable(), called on every incoming Gemini message.
                            behavior_node passes _reset_conversation_timeout here so the
                            timeout resets on function calls too, not just audio chunks.
        """
        self._node        = node
        self._audio       = audio_handler
        self._fn          = function_handlers
        self._prompt      = system_prompt
        self._model       = model
        self._voice       = voice
        self._on_activity = on_activity  # may be None if caller doesn't need it

        # asyncio event loop — created in start(), runs forever in _loop_thread
        self._loop        = None
        self._loop_thread = None

        # The currently running session task (asyncio.Task).
        # Stored so close_session() can cancel it from outside the loop.
        self._session_task  = None
        self._session_active = False  # True only while inside an open Live session

        # Greeting is injected once per wake-word event via session.send().
        # Reset in open_session() (called on wake word), NOT in the reconnect loop —
        # so a crash-reconnect cycle doesn't re-greet mid-conversation.
        self._greeting_sent = False

        # When set, replaces the normal greeting on the next session open.
        # Used by _on_safety_fault so the fault is the first thing Gemini sees,
        # with no race between open_session() and inject_context().
        self._pending_prompt: str | None = None

        # Memory context (Step 5): a block of remembered facts, injected as part
        # of the FIRST message so OMNI knows them from the start of the chat.
        # Set by open_session(memory_context=...); consumed once per session.
        self._memory_context: str = ''

        # Conversation transcript (Step 5): (speaker, text_fragment) pairs captured
        # from Gemini input/output transcription, flushed to memory at chat end.
        # Written from the asyncio recv loop, read/cleared from the ROS thread —
        # guarded by _transcript_lock.
        self._transcript_lock = threading.Lock()
        self._transcript_segments: list[tuple[str, str]] = []

        self._running = False

    # ── Lifecycle — called from the ROS2 thread ────────────────────────────────

    def start(self):
        """
        Create the asyncio event loop and start its thread.
        Call once during BehaviorNode.__init__.
        The loop runs forever — individual sessions are started/stopped on demand.
        """
        self._running = True
        self._loop    = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_loop,
            name='gemini-asyncio',
            daemon=True,
        )
        self._loop_thread.start()
        self._node.get_logger().info('GeminiBridge asyncio loop started')

    def open_session(self, initial_prompt: str | None = None, memory_context: str | None = None):
        """
        Schedule a new Gemini Live session on the asyncio loop.
        Returns immediately — the session opens asynchronously.
        Safe to call from the ROS2 thread.

        initial_prompt — if set, replaces the normal wake-word greeting with this
        text on the next session open. Used by _on_safety_fault so the fault is
        the very first thing Gemini sees, eliminating the race between scheduling
        open_session() and calling inject_context() a moment later.

        memory_context — optional remembered-facts block (already wrapped for the
        model). Prepended to the greeting on the next session open so OMNI knows
        the facts from the first word. Ignored when initial_prompt is set (faults
        stay clean/urgent).
        """
        def _schedule():
            # Guard: only block if the session is both task-running AND logically active.
            # _session_active is set to False in the finally block of _run_single_session
            # (quickly), but _session_task.done() can stay False for ~10s while the
            # async-with context manager's __aexit__ closes the WebSocket. Using
            # _session_active means we don't drop wake words during that cleanup window.
            if self._session_task and not self._session_task.done() and self._session_active:
                self._node.get_logger().warn(
                    'open_session() called but a session task is already running — ignoring'
                )
                return
            self._greeting_sent  = False        # new wake-word event → fresh greeting
            self._pending_prompt = initial_prompt  # None clears any leftover fault prompt
            self._memory_context = memory_context or ''  # None clears any leftover context
            # New conversation → start with a clean transcript buffer. (Reconnects
            # go through _session_with_reconnect, not open_session, so a mid-chat
            # drop does not wipe what was said before it.)
            with self._transcript_lock:
                self._transcript_segments = []
            self._session_task = self._loop.create_task(self._session_with_reconnect())

        self._loop.call_soon_threadsafe(_schedule)

    def close_session(self):
        """
        Cancel the running session task, closing the WebSocket.
        Returns immediately. Safe to call from the ROS2 thread.
        """
        def _cancel():
            if self._session_task and not self._session_task.done():
                self._node.get_logger().info('GeminiBridge: closing session')
                self._session_task.cancel()

        self._loop.call_soon_threadsafe(_cancel)

    def is_session_active(self) -> bool:
        """True while a Gemini Live session is open. Safe to read from any thread."""
        return self._session_active

    def inject_context(self, text: str, *, alert: bool = True):
        """
        Send a text message into the active Gemini session.
        Used by behavior_node to inject [SYSTEM ALERT] fault messages so
        Gemini can react in character. With alert=True (default) the text is
        prefixed with [SYSTEM ALERT] to trigger the system-prompt urgency rule;
        pass alert=False to send the text as-is (e.g. a scripted /audio/say line).
        Safe to call from the ROS2 thread.
        """
        async def _inject():
            if not self._session_active or self._session_ref is None:
                self._node.get_logger().warn(
                    'inject_context() called but no session is active — ignoring'
                )
                return
            payload = f'[SYSTEM ALERT] {text}' if alert else text
            self._node.get_logger().info(f'Injecting context: {payload}')
            try:
                await self._session_ref.send(input=payload)
            except Exception as exc:
                self._node.get_logger().error(f'inject_context failed: {exc}')

        asyncio.run_coroutine_threadsafe(_inject(), self._loop)

    def pop_transcript(self) -> str:
        """Return the accumulated conversation transcript and clear the buffer.

        Called from the ROS thread when a conversation ends, to hand the text to
        the memory store. Safe to call anytime; returns '' when empty.
        """
        with self._transcript_lock:
            segments = self._transcript_segments
            self._transcript_segments = []
        return coalesce_transcript(segments)

    def _append_transcript(self, speaker: str, text: str) -> None:
        """Record a streamed transcription fragment (called from the recv loop)."""
        if not text:
            return
        with self._transcript_lock:
            self._transcript_segments.append((speaker, text))

    def stop(self):
        """Shut down the asyncio loop entirely. Call during node shutdown."""
        self._running = False
        self.close_session()
        self._loop.call_soon_threadsafe(self._loop.stop)

    # ── Internal asyncio methods ───────────────────────────────────────────────

    def _run_loop(self):
        """Entry point for the gemini-asyncio daemon thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _session_with_reconnect(self):
        """
        Outer reconnect wrapper. Opens a session, runs it, and if it drops
        unexpectedly waits 5 seconds then reconnects — but only if the robot
        state allows resuming audio (IDLE or LISTENING). If mid-task
        (NAVIGATING, EXPLORING, ERROR, DOCKING) it holds and waits for state
        to return to IDLE before reconnecting.
        """
        while self._running:
            try:
                await self._run_single_session()
                # If we reach here, close_session() cancelled the task cleanly
                break
            except asyncio.CancelledError:
                # Explicit close requested by behavior_node — do not reconnect
                self._node.get_logger().info('Gemini session closed cleanly')
                break
            except Exception as exc:
                self._node.get_logger().error(
                    f'Gemini session dropped unexpectedly: {exc}. '
                    f'Reconnecting in {_RECONNECT_DELAY}s...'
                )

                # The session died mid-turn, so neither the turn_complete branch
                # nor the clean receive()-exhaust fallback in _recv_loop ran — the
                # SPEAKING→LISTENING transition was skipped and the robot is orphaned
                # in SPEAKING. Recover immediately instead of leaving it frozen for
                # ~45s until _check_state_watchdog fires. Only touch SPEAKING: a fault
                # (ERROR) or a function handler (NAVIGATING) may have already moved us
                # elsewhere, in which case leave that state alone.
                if self._node._current_state == 'SPEAKING':
                    self._node.get_logger().warn(
                        'Session dropped while SPEAKING — recovering to LISTENING '
                        '(previously stayed stuck until the 45s watchdog).'
                    )
                    await self._loop.run_in_executor(
                        None, self._node._set_state, 'LISTENING'
                    )

                await asyncio.sleep(_RECONNECT_DELAY)

                if not self._running:
                    break

                # Check whether to resume immediately or wait for a better state
                state = self._node._current_state
                if state in _STATES_BLOCK_AUDIO:
                    self._node.get_logger().info(
                        f'Robot is in {state} — holding reconnect until state is IDLE. '
                        f'Connection will resume automatically when the task finishes.'
                    )
                    # Poll until state clears — 1s granularity is fine here
                    while self._running and self._node._current_state in _STATES_BLOCK_AUDIO:
                        await asyncio.sleep(1.0)

                    if not self._running:
                        break

                    self._node.get_logger().info(
                        f'State is now {self._node._current_state} — resuming Gemini session'
                    )

    async def _run_single_session(self):
        """
        Open one Gemini Live session and run send/recv until cancelled.
        Uses asyncio.gather() so send and recv run truly concurrently —
        neither blocks the other, so audio flows smoothly while function
        calls are being processed.
        """
        # No http_options here — audio_node (confirmed working) omits them.
        # Adding api_version='v1alpha' was causing silent connection failure.
        client = genai.Client(api_key=self._node._api_key)

        hour = datetime.datetime.now().hour
        if hour < 12:
            period = 'morning'
        elif hour < 17:
            period = 'afternoon'
        else:
            period = 'evening'

        config = genai_types.LiveConnectConfig(
            response_modalities=['AUDIO'],
            system_instruction=self._prompt,
            tools=OMNI_TOOLS,
            # Request transcription of OMNI's audio output — shows in logs and
            # lets us publish /audio/text if needed in future.
            output_audio_transcription=genai_types.AudioTranscriptionConfig(),
            # Request transcription of the USER's speech too (Step 5). Both are
            # accumulated into the conversation transcript that gets summarized
            # and stored to memory at the end of the chat.
            input_audio_transcription=genai_types.AudioTranscriptionConfig(),
            speech_config=genai_types.SpeechConfig(
                voice_config=genai_types.VoiceConfig(
                    prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                        voice_name=self._voice,
                    )
                )
            ),
            # Voice-activity detection (VAD) tuning. Without this, Gemini uses a
            # default start-of-speech threshold that only triggered on our louder
            # turns (~450+ mic RMS) and ignored quieter speech (~300-400 RMS),
            # making OMNI appear to "go deaf" mid-conversation. START_SENSITIVITY_HIGH
            # lowers that threshold to catch softer speech while staying above the
            # ~90-130 quiet-room floor. silence_duration_ms=1000 gives a full second
            # of pause tolerance before Gemini decides the user's turn has ended, so
            # natural mid-sentence pauses don't get cut off.
            realtime_input_config=genai_types.RealtimeInputConfig(
                automatic_activity_detection=genai_types.AutomaticActivityDetection(
                    start_of_speech_sensitivity=genai_types.StartSensitivity.START_SENSITIVITY_HIGH,
                    end_of_speech_sensitivity=genai_types.EndSensitivity.END_SENSITIVITY_HIGH,
                    silence_duration_ms=1000,
                )
            ),
        )

        self._node.get_logger().info(
            f'Opening Gemini Live session — model={self._model}, voice={self._voice}'
        )

        # _session_ref lets inject_context() reach the session from outside the gather
        self._session_ref = None

        async with client.aio.live.connect(model=self._model, config=config) as session:
            self._session_ref    = session
            self._session_active = True
            self._node.get_logger().info('Gemini Live session open — audio streaming active')

            try:
                # First message on session open — either a fault alert or the normal
                # wake-word greeting. _pending_prompt is set by open_session() when
                # a fault arrives so the alert fires the instant the session is ready,
                # with no race against inject_context(). Cleared after sending.
                if not self._greeting_sent:
                    if self._pending_prompt:
                        opening = self._pending_prompt
                        self._pending_prompt = None
                    else:
                        opening = (
                            f'The user just activated you with the wake word. '
                            f'Say a single brief {period} greeting in character — '
                            f'one sentence only — then stop and wait for their request.'
                        )
                        # Prepend remembered facts (Step 5) so OMNI knows them
                        # from the first word. Consumed once per session.
                        if self._memory_context:
                            opening = f'{self._memory_context}\n\n{opening}'
                            self._memory_context = ''
                    await session.send(input=opening, end_of_turn=True)
                    self._greeting_sent = True

                # gather() runs both coroutines concurrently on this event loop.
                # _send_audio_loop reads mic and sends to Gemini.
                # _recv_loop reads Gemini responses and handles audio + function calls.
                # If either raises, gather cancels the other and propagates the exception.
                await asyncio.gather(
                    self._send_audio_loop(session),
                    self._recv_loop(session),
                )
            finally:
                self._session_active = False
                self._session_ref    = None
                self._node.get_logger().info('Gemini Live session closed')

    async def _send_audio_loop(self, session):
        """
        Continuously read mic chunks from AudioHandler and send them to Gemini.

        get_mic_chunk() is a blocking call (waits for the next 100ms of audio).
        We run it in the default thread-pool executor so it does not block the
        asyncio event loop — _recv_loop() must stay responsive during this wait.

        SILENCE SUPPRESSION: while the speaker is playing, we consume mic chunks
        from the queue but do NOT send them. Sending silence (zeros) during
        playback causes Gemini's VAD to miscalibrate — it adapts to silence as
        the baseline and then fails to detect real speech afterward.

        VAD WARMUP: after the speaker gate opens, the first 500ms (5 chunks) of
        real ambient audio is sent before the user speaks. This re-establishes
        Gemini's VAD baseline with real room noise after the silence gap.
        """
        self._node.get_logger().info('_send_audio_loop started')
        gate_was_blocked = False
        vad_warmup_remaining = 0

        # ── Diagnostics (reset each time the gate opens) ──────────────────────
        _RMS_WINDOW_CHUNKS = 20    # 20 × 100ms = 2s RMS report interval
        _WAV_CAPTURE_CHUNKS = 50   # 50 × 100ms = 5s WAV file
        _WAV_PATH = '/tmp/gemini_input.wav'
        rms_buf     = []
        wav_buf     = []
        wav_saved   = False

        while self._session_active:
            chunk = await self._loop.run_in_executor(None, self._audio.get_mic_chunk)

            if not chunk or not self._session_active:
                break

            if self._audio.is_playing():
                if not gate_was_blocked:
                    self._node.get_logger().info(
                        'Mic gate: CLOSED — discarding mic audio while speaker plays'
                    )
                    gate_was_blocked = True
                # Do not send anything — silence poisons Gemini's VAD baseline
                continue

            if gate_was_blocked:
                self._node.get_logger().info(
                    'Mic gate: OPEN — sending 500ms VAD warmup burst'
                )
                gate_was_blocked = False
                vad_warmup_remaining = 5   # 5 × 100ms = 500ms
                # Reset diagnostics for this open window
                rms_buf.clear()
                wav_buf.clear()
                wav_saved = False

            if vad_warmup_remaining > 0:
                vad_warmup_remaining -= 1
                if vad_warmup_remaining == 0:
                    self._node.get_logger().info(
                        'VAD warmup complete — listening for user speech'
                    )

            # ── RMS amplitude log — every 2 seconds ───────────────────────────
            rms_buf.append(chunk)
            if len(rms_buf) >= _RMS_WINDOW_CHUNKS:
                samples = np.frombuffer(b''.join(rms_buf), dtype=np.int16).astype(np.float32)
                rms = float(np.sqrt(np.mean(samples ** 2)))
                self._node.get_logger().info(
                    f'Mic RMS (2s): {rms:.1f}  '
                    f'[0=silence | ~200=quiet room | ~800=speech | ~3000=loud speech]'
                )
                rms_buf.clear()

            # ── WAV capture — first 5 seconds after gate opens ────────────────
            if not wav_saved:
                wav_buf.append(chunk)
                if len(wav_buf) >= _WAV_CAPTURE_CHUNKS:
                    try:
                        with wave.open(_WAV_PATH, 'wb') as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(_PCM_IN_RATE)
                            wf.writeframes(b''.join(wav_buf))
                        self._node.get_logger().info(
                            f'Saved 5s mic capture → {_WAV_PATH}  '
                            f'(scp to PC and listen: aplay {_WAV_PATH})'
                        )
                    except Exception as exc:
                        self._node.get_logger().error(f'WAV save failed: {exc}')
                    wav_saved = True

            try:
                await session.send_realtime_input(
                    audio=genai_types.Blob(
                        data=chunk,
                        mime_type=f'audio/pcm;rate={_PCM_IN_RATE}',
                    )
                )
            except Exception as exc:
                self._node.get_logger().error(f'Error sending audio to Gemini: {exc}')
                break

    async def _recv_loop(self, session):
        """
        Receive responses from Gemini and dispatch them.

        Three response types we handle:
          response.data        — raw audio PCM (24kHz int16 mono) → play on speaker
          response.tool_call   — Gemini wants to call a function → dispatch to handlers
          response.server_content — transcription, end-of-turn marker, or turn_complete

        SPEAKING / LISTENING auto-transitions (do not rely on set_robot_state function call):
          First audio chunk of a turn while state is LISTENING → set SPEAKING immediately,
          before the chunk is enqueued on the speaker. This is reliable because it fires
          on real data arrival, not on an optional function call from the model.

          When Gemini's turn ends, _await_playback_and_set_listening() is spawned as an
          asyncio task. It polls AudioHandler.is_playing() every 100ms and transitions
          back to LISTENING once the speaker queue drains. Two signals can trigger this:
            • server_content.turn_complete=True  — clean end of Gemini's turn
            • receive() generator exhausting without turn_complete — barge-in / interruption
          Both cases are handled; speaking_triggered guards against spawning duplicate tasks.

        IMPORTANT — double loop required (matches audio_node pattern):
          session.receive() is an async generator that exhausts after delivering
          one turn's worth of responses — it does NOT run forever like a WebSocket
          reader. Without the outer while loop, _recv_loop exits after Gemini's
          first response and all subsequent turns (user speech → Gemini reply) are
          silently dropped. The outer while re-enters receive() for each new turn.

        THREAD SAFETY: _set_state() and function handlers are called via run_in_executor()
        so that rclpy calls execute in a thread pool worker, not on the asyncio loop.
        Direct reads of _current_state without the lock are safe in CPython — string
        assignment is atomic under the GIL (same pattern as _session_with_reconnect).
        """
        self._node.get_logger().info('_recv_loop started — waiting for Gemini responses')
        _audio_chunk_count = 0
        while self._session_active:
            # speaking_triggered resets each turn so each new Gemini response gets
            # a fresh LISTENING→SPEAKING transition and its own playback watcher.
            speaking_triggered = False

            # session.receive() is an async generator that exhausts after one
            # turn's responses — it does NOT stay open like a continuous stream.
            # The outer while re-enters receive() so subsequent turns are received.
            async for response in session.receive():
                if not self._session_active:
                    break

                # ── Log every message type received ───────────────────────────
                present = [
                    attr for attr in ('data', 'tool_call', 'server_content',
                                      'setup_complete', 'go_away', 'session_resumption_update')
                    if getattr(response, attr, None)
                ]
                if present:
                    self._node.get_logger().info(f'Gemini response — fields present: {present}')
                else:
                    self._node.get_logger().info('Gemini response — all fields empty/None (heartbeat?)')

                # Notify behavior_node of any activity so the conversation timeout resets.
                if self._on_activity and (response.data or response.tool_call or response.server_content):
                    self._on_activity()

                # ── Audio response ────────────────────────────────────────────
                if response.data:
                    _audio_chunk_count += 1
                    self._node.get_logger().info(
                        f'Audio chunk #{_audio_chunk_count}: {len(response.data)} bytes received '
                        f'from Gemini → passing to play_audio()'
                    )

                    # Auto-transition LISTENING → SPEAKING on the first audio chunk
                    # of each turn. Done before play_audio() so state is correct
                    # the moment the speaker starts. Only fires from LISTENING —
                    # other states (ERROR, NAVIGATING) are left untouched.
                    if not speaking_triggered and self._node._current_state == 'LISTENING':
                        speaking_triggered = True
                        await self._loop.run_in_executor(
                            None, self._node._set_state, 'SPEAKING'
                        )

                    self._audio.play_audio(response.data)
                    self._node.get_logger().info(
                        f'Audio chunk #{_audio_chunk_count}: play_audio() returned '
                        f'(enqueued for speaker)'
                    )

                # ── Function calls ────────────────────────────────────────────
                # All results from a single tool_call event must be batched into
                # ONE LiveClientToolResponse. Sending separate messages per call
                # causes error 1008 — Gemini treats the second message as a
                # protocol violation because it already advanced past tool state.
                if response.tool_call:
                    fn_responses = []
                    for fc in response.tool_call.function_calls:
                        self._node.get_logger().info(
                            f'Gemini function call: {fc.name}({dict(fc.args)})'
                        )
                        result = await self._loop.run_in_executor(
                            None,
                            lambda n=fc.name, a=dict(fc.args): self._fn.handle(n, a),
                        )
                        fn_responses.append(
                            genai_types.FunctionResponse(
                                id=fc.id,
                                name=fc.name,
                                response={'result': result},
                            )
                        )
                        self._node.get_logger().debug(
                            f'Function result queued for {fc.name}: '
                            f'{result[:80]}{"..." if len(result) > 80 else ""}'
                        )
                    try:
                        await session.send(
                            input=genai_types.LiveClientToolResponse(
                                function_responses=fn_responses
                            )
                        )
                        self._node.get_logger().info(
                            f'Tool response sent: '
                            f'{[r.name for r in fn_responses]}'
                        )
                    except Exception as exc:
                        self._node.get_logger().error(
                            f'Failed to send tool response '
                            f'{[r.name for r in fn_responses]}: {exc}'
                        )

                # ── Server content (transcription / end-of-turn) ──────────────
                if response.server_content:
                    sc = response.server_content

                    # turn_complete=True is Gemini's explicit signal that it has
                    # finished generating audio for this turn. Spawn a watcher that
                    # waits for the speaker queue to drain, then sets LISTENING.
                    # Clear speaking_triggered so the post-loop fallback below does
                    # not create a duplicate watcher.
                    if getattr(sc, 'turn_complete', False) and speaking_triggered:
                        self._loop.create_task(self._await_playback_and_set_listening())
                        speaking_triggered = False

                    # Transcription fragments (Step 5) — accumulate both sides of
                    # the conversation for the end-of-chat memory store. Streamed
                    # in small pieces; coalesce_transcript() reassembles them.
                    if getattr(sc, 'input_transcription', None) and sc.input_transcription.text:
                        self._append_transcript('User', sc.input_transcription.text)
                    if getattr(sc, 'output_transcription', None) and sc.output_transcription.text:
                        self._append_transcript('OMNI', sc.output_transcription.text)
                        self._node.get_logger().debug(
                            f'Gemini said: {sc.output_transcription.text}'
                        )

            # receive() exhausted without turn_complete — barge-in or interruption.
            # If we had started speaking, still need to drain the speaker queue.
            if speaking_triggered:
                self._loop.create_task(self._await_playback_and_set_listening())

    async def _await_playback_and_set_listening(self):
        """
        Wait for AudioHandler to finish playing all enqueued audio, then
        transition from SPEAKING back to LISTENING.

        Spawned as an asyncio task (not awaited inline) so _recv_loop can
        immediately re-enter session.receive() for the next turn while playback
        is still draining. Polls is_playing() every 100ms — fine-grained enough
        for a responsive mic-gate re-open without busy-looping.

        Guards:
          • Exits early if the session closes while waiting (close_session() called).
          • Only transitions if state is still SPEAKING — function_handlers or a
            safety fault may have already changed state, in which case we leave it alone.
        """
        while self._session_active and self._audio.is_playing():
            await asyncio.sleep(0.1)

        if not self._session_active:
            return

        if self._node._current_state == 'SPEAKING':
            await self._loop.run_in_executor(None, self._node._set_state, 'LISTENING')
