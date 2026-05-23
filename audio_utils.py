"""
Audio Utilities
===============
Shared helpers for audio processing — RMS calculation,
silence detection, and activation chime generation.
"""

import struct
import math
import numpy as np
import pyaudio

from config import (
    AUDIO_RATE,
    CHIME_FREQUENCY,
    CHIME_DURATION,
    PLAY_ACTIVATION_CHIME,
)


def compute_rms(audio_chunk: bytes) -> float:
    """
    Calculate the Root Mean Square (RMS) energy of a raw PCM16 audio chunk.

    Args:
        audio_chunk: Raw bytes of signed 16-bit little-endian PCM audio.

    Returns:
        RMS energy as a float. Higher values = louder audio.
    """
    if not audio_chunk:
        return 0.0

    # Unpack raw bytes into signed 16-bit integers
    sample_count = len(audio_chunk) // 2
    samples = struct.unpack(f"<{sample_count}h", audio_chunk)

    if sample_count == 0:
        return 0.0

    # RMS = sqrt(mean(x^2))
    sum_squares = sum(s * s for s in samples)
    return math.sqrt(sum_squares / sample_count)


def is_silence(audio_chunk: bytes, threshold: int) -> bool:
    """
    Determine whether an audio chunk is silence.

    Args:
        audio_chunk: Raw PCM16 audio bytes.
        threshold: RMS threshold below which audio is considered silent.

    Returns:
        True if the audio is below the silence threshold.
    """
    return compute_rms(audio_chunk) < threshold


def play_chime() -> None:
    """
    Play a short activation chime to indicate the wake word was detected.
    Generates a simple sine wave tone and plays it through PyAudio.
    Skipped if PLAY_ACTIVATION_CHIME is False in config.
    """
    if not PLAY_ACTIVATION_CHIME:
        return

    pa = pyaudio.PyAudio()
    try:
        # Generate sine wave samples
        num_samples = int(AUDIO_RATE * CHIME_DURATION)
        t = np.linspace(0, CHIME_DURATION, num_samples, endpoint=False)
        waveform = (np.sin(2 * np.pi * CHIME_FREQUENCY * t) * 16000).astype(np.int16)

        # Apply a quick fade-in/fade-out to avoid clicks
        fade_len = min(200, num_samples // 4)
        fade_in = np.linspace(0, 1, fade_len)
        fade_out = np.linspace(1, 0, fade_len)
        waveform[:fade_len] = (waveform[:fade_len] * fade_in).astype(np.int16)
        waveform[-fade_len:] = (waveform[-fade_len:] * fade_out).astype(np.int16)

        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=AUDIO_RATE,
            output=True,
        )
        stream.write(waveform.tobytes())
        stream.stop_stream()
        stream.close()
    finally:
        pa.terminate()
