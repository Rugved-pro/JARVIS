"""
Ears Module
===========
Handles all audio input — wake word detection via openWakeWord
and speech-to-text transcription via faster-whisper.
Both run on CPU to preserve GPU VRAM for the LLM.
"""

import wave
import tempfile
import os
import logging

import numpy as np
from openwakeword.model import Model as OWWModel
from faster_whisper import WhisperModel

from config import (
    WAKEWORD_MODEL_PATH,
    WAKEWORD_THRESHOLD,
    WAKEWORD_VAD_THRESHOLD,
    WAKEWORD_FRAME_LENGTH,
    WHISPER_MODEL,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    AUDIO_RATE,
    AUDIO_CHANNELS,
    AUDIO_FORMAT_WIDTH,
)

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """
    Listens for "Hey Jarvis" using openWakeWord.
    Processes one audio frame (1280 samples / 80ms) at a time.
    Uses the downloaded ONNX model to avoid Windows TFLite issues.
    """

    def __init__(
        self,
        model_path: str = WAKEWORD_MODEL_PATH,
        threshold: float = WAKEWORD_THRESHOLD,
        vad_threshold: float = WAKEWORD_VAD_THRESHOLD,
    ):
        self._model_path = os.path.abspath(model_path)
        self._threshold = threshold
        
        # openwakeword's keys in its prediction dict are based on the filename
        # e.g., 'hey_jarvis' for 'hey_jarvis.onnx'
        self._model_name = os.path.splitext(os.path.basename(self._model_path))[0]

        if not os.path.isfile(self._model_path):
            raise FileNotFoundError(
                f"ONNX wake word model not found: {self._model_path}"
            )

        self._model = OWWModel(
            model_paths=[self._model_path],
            vad_threshold=vad_threshold,
            inference_framework="onnx",  # Force ONNX to avoid tflite_runtime dependency errors
        )

        logger.info(
            f"openWakeWord initialized — model='{wakeword_model}', "
            f"threshold={threshold}, vad_threshold={vad_threshold}, "
            f"frame_length={WAKEWORD_FRAME_LENGTH}"
        )

    @property
    def frame_length(self) -> int:
        """Number of audio samples per frame expected by openWakeWord (1280)."""
        return WAKEWORD_FRAME_LENGTH

    @property
    def sample_rate(self) -> int:
        """Sample rate expected by openWakeWord (16000 Hz)."""
        return AUDIO_RATE

    def process(self, audio_frame: bytes) -> bool:
        """
        Process a single audio frame for wake word detection.

        Args:
            audio_frame: Raw PCM16 bytes of length `frame_length * 2` (2560 bytes).

        Returns:
            True if "Hey Jarvis" was detected above the confidence threshold.
        """
        # Convert raw bytes to numpy int16 array
        pcm = np.frombuffer(audio_frame, dtype=np.int16)

        # Run prediction
        prediction = self._model.predict(pcm)

        # Check confidence against threshold
        confidence = prediction.get(self._model_name, 0)

        if confidence > self._threshold:
            logger.info(
                f"Wake word detected! confidence={confidence:.3f} "
                f"(threshold={self._threshold})"
            )
            # Reset the model's internal state to avoid repeated triggers
            self._model.reset()
            return True

        return False

    def cleanup(self) -> None:
        """Release openWakeWord resources."""
        self._model = None
        logger.info("openWakeWord resources released.")


class SpeechTranscriber:
    """
    Transcribes recorded audio to text using faster-whisper.
    Runs on CPU with int8 quantization for low latency.
    """

    def __init__(
        self,
        model_size: str = WHISPER_MODEL,
        device: str = WHISPER_DEVICE,
        compute_type: str = WHISPER_COMPUTE_TYPE,
    ):
        logger.info(
            f"Loading Whisper model '{model_size}' "
            f"(device={device}, compute_type={compute_type})..."
        )
        self._model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
        logger.info("Whisper model loaded and ready.")

    def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = AUDIO_RATE,
    ) -> str:
        """
        Transcribe raw PCM16 audio to text.

        Args:
            audio_data: Raw PCM16 audio bytes (mono, 16kHz).
            sample_rate: Sample rate of the audio.

        Returns:
            Transcribed text string. Empty string if nothing detected.
        """
        if not audio_data:
            return ""

        # Write audio to a temporary WAV file (faster-whisper needs a file path)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as tmp_file:
                tmp_path = tmp_file.name
                with wave.open(tmp_file, "wb") as wf:
                    wf.setnchannels(AUDIO_CHANNELS)
                    wf.setsampwidth(AUDIO_FORMAT_WIDTH)
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio_data)

            # Transcribe with speed-optimized settings
            segments, info = self._model.transcribe(
                tmp_path,
                beam_size=1,           # Greedy decoding for speed
                vad_filter=True,       # Filter out non-speech segments
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                ),
                language="en",
            )

            # Collect all segment texts
            transcript = " ".join(
                segment.text.strip() for segment in segments
            ).strip()

            logger.info(f"Transcription ({info.duration:.1f}s audio): '{transcript}'")
            return transcript

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""

        finally:
            # Clean up temp file
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
