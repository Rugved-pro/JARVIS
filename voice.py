"""
Voice Module
============
Text-to-Speech using the Piper TTS engine.
Runs Piper as a subprocess, pipes text in via stdin,
captures WAV audio from stdout, and plays it through PyAudio.
"""

import subprocess
import io
import wave
import os
import re
import logging
from typing import Generator

import pyaudio

from config import (
    PIPER_EXE_PATH,
    PIPER_VOICE_MODEL,
    AUDIO_RATE,
)

logger = logging.getLogger(__name__)

# Regex pattern to split text into sentences
SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


class PiperVoice:
    """
    Text-to-Speech engine using a local Piper executable.
    Converts text to speech by piping to Piper's stdin and
    capturing raw WAV audio from its stdout.
    """

    def __init__(
        self,
        piper_exe_path: str = PIPER_EXE_PATH,
        voice_model_path: str = PIPER_VOICE_MODEL,
    ):
        # Resolve paths relative to the script's directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self._piper_exe = os.path.normpath(
            os.path.join(base_dir, piper_exe_path)
        )
        self._voice_model = os.path.normpath(
            os.path.join(base_dir, voice_model_path)
        )

        # Validate paths
        if not os.path.isfile(self._piper_exe):
            raise FileNotFoundError(
                f"Piper executable not found: {self._piper_exe}\n"
                f"Download it from: https://github.com/rhasspy/piper/releases"
            )
        if not os.path.isfile(self._voice_model):
            raise FileNotFoundError(
                f"Piper voice model not found: {self._voice_model}\n"
                f"Download from: https://huggingface.co/rhasspy/piper-voices"
            )

        logger.info(
            f"PiperVoice initialized — exe='{self._piper_exe}', "
            f"model='{os.path.basename(self._voice_model)}'"
        )

    def _synthesize(self, text: str) -> bytes:
        """
        Run Piper to synthesize text into raw WAV audio bytes.

        Args:
            text: The text to convert to speech.

        Returns:
            Raw WAV file bytes.
        """
        if not text.strip():
            return b""

        cmd = [
            self._piper_exe,
            "--model", self._voice_model,
            "--output-raw",  # Output raw PCM instead of WAV header
        ]

        try:
            result = subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=30,
            )

            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", errors="replace")
                logger.error(f"Piper failed (exit {result.returncode}): {stderr}")
                return b""

            return result.stdout

        except subprocess.TimeoutExpired:
            logger.error("Piper subprocess timed out.")
            return b""
        except FileNotFoundError:
            logger.error(f"Piper executable not found at: {self._piper_exe}")
            return b""

    def _play_audio(self, raw_audio: bytes, sample_rate: int = 22050) -> None:
        """
        Play raw PCM16 audio through the default speaker.

        Args:
            raw_audio: Raw signed 16-bit PCM audio bytes.
            sample_rate: Sample rate of the audio (Piper default is 22050).
        """
        if not raw_audio:
            return

        pa = pyaudio.PyAudio()
        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                output=True,
            )

            # Play in chunks to avoid buffer issues
            chunk_size = 4096
            for i in range(0, len(raw_audio), chunk_size):
                stream.write(raw_audio[i:i + chunk_size])

            stream.stop_stream()
            stream.close()
        finally:
            pa.terminate()

    def speak(self, text: str) -> None:
        """
        Synthesize and play a complete text string.

        Args:
            text: The text to speak aloud.
        """
        if not text.strip():
            return

        logger.info(f"Speaking: '{text[:60]}...'")
        raw_audio = self._synthesize(text)
        self._play_audio(raw_audio)

    def speak_streamed(self, text_generator: Generator[str, None, None]) -> None:
        """
        Accept a streaming text generator (from the LLM) and speak
        sentence by sentence for near-real-time TTS output.

        Buffers incoming chunks and flushes to TTS whenever a sentence
        boundary (. ! ?) is detected, so the user hears the first
        sentence while the LLM is still generating the rest.

        Args:
            text_generator: A generator yielding text chunks.
        """
        buffer = ""
        full_text = ""

        for chunk in text_generator:
            buffer += chunk
            full_text += chunk

            # Check if buffer contains complete sentences
            sentences = SENTENCE_SPLIT_RE.split(buffer)

            if len(sentences) > 1:
                # Speak all complete sentences, keep the remainder
                complete = " ".join(sentences[:-1])
                buffer = sentences[-1]

                if complete.strip():
                    self.speak(complete.strip())

        # Speak any remaining text in the buffer
        if buffer.strip():
            self.speak(buffer.strip())

        logger.info(f"Finished speaking ({len(full_text)} chars total)")
