"""
Pi Zero W audio bridge simulator.

Connects to audio_node TCP server, streams a 16kHz 16-bit mono WAV file
(or a speech-like chirp sweep if no file is given), then saves the Gemini
audio response to /tmp/gemini_response.wav.

Usage:
  python3 test_audio_sim.py [input.wav]

The input WAV must be 16kHz, 16-bit, mono. If omitted a 2-second voiced
speech-like audio is generated, followed by silence to trigger Gemini's VAD.
"""

import math
import socket
import struct
import sys
import time
import wave
import threading

HOST = '127.0.0.1'
PORT = 9876
CHUNK_SAMPLES = 1600       # 100 ms at 16 kHz
SAMPLE_RATE = 16000
RESPONSE_RATE = 24000
SILENCE_AFTER_S = 4.0      # seconds of silence after speech to trigger VAD
RESPONSE_WAIT_S = 25.0     # seconds to wait for Gemini response


def generate_speech_like(duration_s: float = 2.0) -> bytes:
    """Generate a voiced speech-like signal (fundamental + harmonics with AM)."""
    n = int(SAMPLE_RATE * duration_s)
    samples = []
    for i in range(n):
        t = i / SAMPLE_RATE
        # Voiced fundamental at 150 Hz with harmonics (speech-like spectrum)
        sig = (
            0.5 * math.sin(2 * math.pi * 150 * t) +
            0.3 * math.sin(2 * math.pi * 300 * t) +
            0.2 * math.sin(2 * math.pi * 600 * t) +
            0.15 * math.sin(2 * math.pi * 900 * t) +
            0.1 * math.sin(2 * math.pi * 1800 * t)
        )
        # Amplitude modulation at ~4 Hz to simulate syllable rhythm
        am = 0.5 + 0.5 * math.sin(2 * math.pi * 4 * t)
        # Fade in/out
        fade = min(t / 0.05, 1.0, (duration_s - t) / 0.05)
        v = int(32767 * 0.4 * sig * am * fade)
        samples.append(struct.pack('<h', max(-32767, min(32767, v))))
    return b''.join(samples)


def generate_silence(duration_s: float) -> bytes:
    n = int(SAMPLE_RATE * duration_s)
    return b'\x00\x00' * n


def load_wav(path: str) -> bytes:
    with wave.open(path, 'rb') as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != SAMPLE_RATE:
            print(f'WARNING: {path} should be 16kHz 16-bit mono. Streaming anyway.')
        return wf.readframes(wf.getnframes())


def send_frames(conn: socket.socket, pcm: bytes, label: str = ''):
    chunk_bytes = CHUNK_SAMPLES * 2
    offset = 0
    while offset < len(pcm):
        chunk = pcm[offset:offset + chunk_bytes]
        offset += chunk_bytes
        conn.sendall(struct.pack('>I', len(chunk)) + chunk)
        time.sleep(len(chunk) / (SAMPLE_RATE * 2) * 0.95)
    if label:
        print(label)


def recv_responses(conn: socket.socket, out_path: str, stop_event: threading.Event):
    received: list[bytes] = []
    conn.settimeout(2.0)
    try:
        while not stop_event.is_set():
            try:
                header = b''
                while len(header) < 4:
                    part = conn.recv(4 - len(header))
                    if not part:
                        return
                    header += part
                length = struct.unpack('>I', header)[0]
                data = b''
                while len(data) < length:
                    part = conn.recv(length - len(data))
                    if not part:
                        return
                    data += part
                received.append(data)
                ms = length // 2 / RESPONSE_RATE * 1000
                print(f'  received audio chunk: {length} bytes ({ms:.0f} ms)')
            except socket.timeout:
                continue
    finally:
        if received:
            all_pcm = b''.join(received)
            with wave.open(out_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(RESPONSE_RATE)
                wf.writeframes(all_pcm)
            dur = len(all_pcm) // 2 / RESPONSE_RATE
            print(f'Response saved to {out_path} ({dur:.1f}s of 24kHz audio)')
        else:
            print('No audio response received.')


def main():
    wav_path = sys.argv[1] if len(sys.argv) > 1 else None
    if wav_path:
        print(f'Loading {wav_path}')
        speech_pcm = load_wav(wav_path)
        silence_pcm = generate_silence(SILENCE_AFTER_S)
    else:
        print('No WAV file given — generating 2s speech-like audio + 4s silence')
        speech_pcm = generate_speech_like(2.0)
        silence_pcm = generate_silence(SILENCE_AFTER_S)

    print(f'Connecting to {HOST}:{PORT}...')
    conn = socket.create_connection((HOST, PORT), timeout=10.0)
    print('Connected.')

    stop_event = threading.Event()
    out_path = '/tmp/gemini_response.wav'
    recv_thread = threading.Thread(
        target=recv_responses, args=(conn, out_path, stop_event), daemon=True
    )
    recv_thread.start()

    send_frames(conn, speech_pcm, 'Speech audio sent.')
    send_frames(conn, silence_pcm, 'Silence sent — waiting for VAD end-of-turn...')

    print(f'Waiting up to {RESPONSE_WAIT_S:.0f}s for Gemini response...')
    recv_thread.join(timeout=RESPONSE_WAIT_S)
    stop_event.set()
    conn.close()


if __name__ == '__main__':
    main()
