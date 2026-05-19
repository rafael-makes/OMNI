#!/usr/bin/env python3
"""
OMNI audio bridge — runs on Pi Zero W with ReSpeaker 2-mic HAT.

Captures microphone audio and streams it to the Pi 5 audio_node over TCP.
Receives Gemini response audio from the Pi 5 and plays it back.

TCP protocol (matches audio_node):
  Outbound (Pi Zero W → Pi 5):  4-byte big-endian length | 16kHz 16-bit mono PCM
  Inbound  (Pi 5 → Pi Zero W):  4-byte big-endian length | 24kHz 16-bit mono PCM

Environment variables:
  OMNI_HOST      Pi 5 hostname or IP  (default: omni.local)
  OMNI_PORT      TCP port             (default: 9876)
  AUDIO_DEVICE   sounddevice device name or index (default: auto)
"""

import os
import queue
import socket
import struct
import threading
import time

import numpy as np
import sounddevice as sd

# ── Config ────────────────────────────────────────────────────────────────────

HOST = os.environ.get('OMNI_HOST', 'omni.local')
PORT = int(os.environ.get('OMNI_PORT', '9876'))

MIC_RATE      = 16000
PLAY_RATE     = 24000
CHUNK_FRAMES  = 1600       # 100 ms at 16 kHz
MAX_FRAME     = 65536
PLAY_QUEUE_MAX = 30        # ~1.25s of 24kHz audio at 100ms chunks

RECONNECT_DELAY_INIT = 2.0
RECONNECT_DELAY_MAX  = 30.0

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_device():
    dev = os.environ.get('AUDIO_DEVICE')
    if dev is None:
        return None
    try:
        return int(dev)
    except ValueError:
        return dev


def _enable_keepalive(sock: socket.socket) -> None:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    if hasattr(socket, 'TCP_KEEPIDLE'):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 10)
    if hasattr(socket, 'TCP_KEEPINTVL'):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 5)
    if hasattr(socket, 'TCP_KEEPCNT'):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = b''
    while len(buf) < n:
        part = conn.recv(n - len(buf))
        if not part:
            raise ConnectionResetError('connection closed')
        buf += part
    return buf

# ── Threads ───────────────────────────────────────────────────────────────────

_SILENCE_CHUNK = bytes(CHUNK_FRAMES * 2)   # pre-built zero buffer, reused every frame


def _capture_and_send(conn: socket.socket, stop: threading.Event,
                      playing: threading.Event):
    """Open mic → read 100ms chunks → send length-prefixed frames to Pi 5.

    While response audio is playing back through the speaker, sends silence
    instead of real mic audio so Gemini cannot hear its own voice and
    interrupt itself.
    """
    device = _get_device()
    try:
        with sd.InputStream(
            samplerate=MIC_RATE,
            channels=1,
            dtype='int16',
            blocksize=CHUNK_FRAMES,
            device=device,
        ) as stream:
            print(f'[bridge] Mic open: {MIC_RATE} Hz, device={device!r}')
            while not stop.is_set():
                pcm, _ = stream.read(CHUNK_FRAMES)
                raw = _SILENCE_CHUNK if playing.is_set() else pcm.tobytes()
                conn.sendall(struct.pack('>I', len(raw)) + raw)
    except sd.PortAudioError as e:
        print(f'[bridge] Mic error: {e}')
    except OSError as e:
        print(f'[bridge] Send error: {e}')
    finally:
        stop.set()


def _recv_and_play(conn: socket.socket, stop: threading.Event,
                   playing: threading.Event):
    """Read length-prefixed response audio from Pi 5 → play via speaker.

    Sets `playing` while audio is queued or actively playing so the capture
    thread can mute the mic and prevent echo.
    """
    device = _get_device()
    play_q: queue.Queue = queue.Queue(maxsize=PLAY_QUEUE_MAX)

    def _player():
        try:
            with sd.OutputStream(
                samplerate=PLAY_RATE,
                channels=1,
                dtype='int16',
                device=device,
            ) as stream:
                print(f'[bridge] Speaker open: {PLAY_RATE} Hz, device={device!r}')
                while not stop.is_set():
                    try:
                        chunk = play_q.get(timeout=0.3)
                        stream.write(chunk)
                    except queue.Empty:
                        # Queue drained — playback finished, unmute mic
                        playing.clear()
        except sd.PortAudioError as e:
            print(f'[bridge] Speaker error: {e}')
            stop.set()

    threading.Thread(target=_player, daemon=True).start()

    try:
        while not stop.is_set():
            header = _recv_exact(conn, 4)
            length = struct.unpack('>I', header)[0]
            if length == 0 or length > MAX_FRAME:
                print(f'[bridge] Bad frame length {length}, skipping')
                continue
            data = _recv_exact(conn, length)
            pcm = np.frombuffer(data, dtype=np.int16).reshape(-1, 1)
            playing.set()   # mute mic before queuing audio
            try:
                play_q.put_nowait(pcm)
            except queue.Full:
                pass
    except (ConnectionResetError, OSError) as e:
        print(f'[bridge] Recv error: {e}')
    finally:
        playing.clear()
        stop.set()

# ── Session ───────────────────────────────────────────────────────────────────

def _run_session():
    print(f'[bridge] Connecting to {HOST}:{PORT}...')
    conn = socket.create_connection((HOST, PORT), timeout=10.0)
    conn.settimeout(None)   # clear connect-phase timeout — socket is now blocking
    _enable_keepalive(conn)
    print(f'[bridge] Connected.')

    stop = threading.Event()
    playing = threading.Event()
    t_send = threading.Thread(target=_capture_and_send, args=(conn, stop, playing), daemon=True)
    t_recv = threading.Thread(target=_recv_and_play,   args=(conn, stop, playing), daemon=True)
    t_send.start()
    t_recv.start()

    try:
        while not stop.is_set():
            time.sleep(0.5)
    finally:
        stop.set()
        conn.close()
        t_send.join(timeout=3.0)
        t_recv.join(timeout=3.0)
    print('[bridge] Session ended.')


def main():
    backoff = RECONNECT_DELAY_INIT
    while True:
        try:
            _run_session()
            backoff = RECONNECT_DELAY_INIT
        except (ConnectionRefusedError, socket.timeout, OSError) as e:
            print(f'[bridge] Connection failed: {e}')
        print(f'[bridge] Reconnecting in {backoff:.0f}s...')
        time.sleep(backoff)
        backoff = min(backoff * 2.0, RECONNECT_DELAY_MAX)


if __name__ == '__main__':
    main()
