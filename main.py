"""
JARVIS — Local Voice-Activated AI Assistant
============================================
Main orchestrator script. Runs a continuous listen-wait loop:

    IDLE → (wake word) → LISTENING → (silence) → THINKING → SPEAKING → IDLE

Usage:
    python main.py

Prerequisites:
    - Ollama running locally with llama3.1:latest loaded
    - Piper executable + voice model in ./piper/
"""

import sys
import time
import struct
import logging

import pyaudio

from config import (
    AUDIO_RATE,
    AUDIO_CHANNELS,
    AUDIO_FORMAT_WIDTH,
    SILENCE_THRESHOLD,
    SILENCE_DURATION,
    MAX_LISTEN_SECONDS,
)
from audio_utils import is_silence, play_chime
from ears import WakeWordDetector, SpeechTranscriber
from brain import OllamaBrain
from voice import PiperVoice

# ──────────────────────────────────────────────
#  Logging Setup
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-12s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("jarvis")


# ──────────────────────────────────────────────
#  State Labels (for clarity in logs)
# ──────────────────────────────────────────────
STATE_IDLE = "IDLE"
STATE_LISTENING = "LISTENING"
STATE_THINKING = "THINKING"
STATE_SPEAKING = "SPEAKING"


def record_until_silence(
    stream: pyaudio.Stream,
    frame_length: int,
) -> bytes:
    """
    Record audio from the stream until the user stops speaking.

    Detects end-of-speech by monitoring for sustained silence.
    Returns early if MAX_LISTEN_SECONDS is exceeded.

    Args:
        stream: An open PyAudio input stream.
        frame_length: Number of samples per frame.

    Returns:
        Raw PCM16 audio bytes of the recorded speech.
        Empty bytes if no speech was detected.
    """
    frames: list[bytes] = []
    silence_start: float | None = None
    has_speech = False

    # Calculate timing
    frame_bytes = frame_length * AUDIO_FORMAT_WIDTH
    frame_duration = frame_length / AUDIO_RATE
    max_frames = int(MAX_LISTEN_SECONDS / frame_duration)

    for _ in range(max_frames):
        try:
            audio_frame = stream.read(frame_length, exception_on_overflow=False)
        except IOError:
            continue

        frames.append(audio_frame)

        if is_silence(audio_frame, SILENCE_THRESHOLD):
            # Track how long we've been in silence
            if silence_start is None:
                silence_start = time.time()
            elif time.time() - silence_start >= SILENCE_DURATION:
                if has_speech:
                    logger.info("Silence detected — stopping recording.")
                    break
                else:
                    # No speech detected at all, bail out
                    logger.info("No speech detected — timeout.")
                    return b""
        else:
            # Audio above threshold — user is speaking
            silence_start = None
            has_speech = True

    if not has_speech:
        return b""

    return b"".join(frames)


def print_banner() -> None:
    """Print a startup banner."""
    banner = r"""
==================================================
              J.A.R.V.I.S.
    Local Voice AI Assistant  •  v1.0
==================================================
    """
    print(banner)


def main() -> None:
    """
    Main orchestrator loop.
    Initializes all modules, opens the audio stream,
    and runs the wake-word → listen → think → speak cycle.
    """
    print_banner()

    # ── Initialize Modules ──────────────────────
    logger.info("Initializing modules...")

    detector = WakeWordDetector()

    transcriber = SpeechTranscriber()
    brain = OllamaBrain()

    try:
        voice = PiperVoice()
    except FileNotFoundError as e:
        logger.error(f"TTS init failed: {e}")
        sys.exit(1)

    # ── Open Audio Stream ───────────────────────
    pa = pyaudio.PyAudio()
    frame_length = detector.frame_length

    stream = pa.open(
        format=pyaudio.paInt16,
        channels=AUDIO_CHANNELS,
        rate=detector.sample_rate,
        input=True,
        frames_per_buffer=frame_length,
    )

    logger.info(
        f"Audio stream opened — rate={detector.sample_rate}Hz, "
        f"frame_length={frame_length} samples"
    )

    print("\n" + "=" * 50)
    print("  Say 'Hey Jarvis' to activate...")
    print("  Press Ctrl+C to quit.")
    print("=" * 50 + "\n")

    state = STATE_IDLE

    # ── Main Loop ───────────────────────────────
    try:
        while True:
            # ── IDLE: Listen for wake word ──────
            if state == STATE_IDLE:
                try:
                    audio_frame = stream.read(
                        frame_length, exception_on_overflow=False
                    )
                except IOError:
                    continue

                if detector.process(audio_frame):
                    state = STATE_LISTENING
                    logger.info("Wake word detected!")
                    print("\n[ Listening... ]")
                    play_chime()

            # ── LISTENING: Record user speech ───
            elif state == STATE_LISTENING:
                audio_data = record_until_silence(stream, frame_length)

                if not audio_data:
                    print("   (no speech detected)")
                    state = STATE_IDLE
                    print("\n[ Say 'Hey Jarvis' to activate... ]\n")
                    continue

                state = STATE_THINKING

                # ── THINKING: Transcribe + LLM ──
                print("[ Thinking... ]")
                transcript = transcriber.transcribe(audio_data)

                if not transcript:
                    print("   (couldn't understand audio)")
                    state = STATE_IDLE
                    print("\n[ Say 'Hey Jarvis' to activate... ]\n")
                    continue

                print(f"You: {transcript}")

                response_stream = brain.think(transcript)

                state = STATE_SPEAKING

                # ── SPEAKING: Stream to TTS ─────
                print("Jarvis: ", end="", flush=True)

                # Tee the response for console display and TTS
                def tee_and_print(gen):
                    """Yield chunks while also printing them to console."""
                    for chunk in gen:
                        print(chunk, end="", flush=True)
                        yield chunk
                    print()  # newline after response

                voice.speak_streamed(tee_and_print(response_stream))

                state = STATE_IDLE
                print("\n[ Say 'Hey Jarvis' to activate... ]\n")

    except KeyboardInterrupt:
        print("\n\n[ Shutting down Jarvis... ]")
    finally:
        # ── Cleanup ─────────────────────────────
        stream.stop_stream()
        stream.close()
        pa.terminate()
        detector.cleanup()
        logger.info("All resources released. Goodbye!")


if __name__ == "__main__":
    main()
