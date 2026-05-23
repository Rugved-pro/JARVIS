"""
JARVIS Configuration
====================
Central configuration for all tunable parameters.
"""

# ──────────────────────────────────────────────
#  Wake Word (openWakeWord)
# ──────────────────────────────────────────────
# No API key needed — fully open source!
# Pointing directly to the downloaded ONNX model to avoid TFLite dependency issues on Windows
WAKEWORD_MODEL_PATH: str = "./hey_jarvis.onnx"
WAKEWORD_THRESHOLD: float = 0.5             # Confidence threshold (0.0–1.0)
WAKEWORD_VAD_THRESHOLD: float = 0.5         # Silero VAD threshold to reduce false positives
WAKEWORD_FRAME_LENGTH: int = 1280           # Samples per frame (80ms at 16kHz) — required by openWakeWord

# ──────────────────────────────────────────────
#  Speech-to-Text (faster-whisper)
# ──────────────────────────────────────────────
WHISPER_MODEL: str = "tiny.en"       # Smallest English-only model (~75MB)
WHISPER_DEVICE: str = "cpu"          # Keep STT on CPU to save GPU VRAM
WHISPER_COMPUTE_TYPE: str = "int8"   # Quantized for maximum CPU speed

# ──────────────────────────────────────────────
#  LLM (Ollama)
# ──────────────────────────────────────────────
OLLAMA_BASE_URL: str = "http://localhost:11434"
OLLAMA_MODEL: str = "llama3.1:latest"

SYSTEM_PROMPT: str = (
    "You are Jarvis, a highly intelligent and helpful AI assistant. "
    "You respond concisely and conversationally — keep answers short unless "
    "the user asks for detail. You do NOT have the ability to execute commands, "
    "open applications, browse the web, or access the terminal or file system. "
    "If the user asks you to perform a system action, politely explain that "
    "this capability is planned via MCP (Model Context Protocol) integration "
    "with VS Code but is not yet available. Focus on being knowledgeable, "
    "witty, and helpful within the bounds of conversation."
)

# ──────────────────────────────────────────────
#  Text-to-Speech (Piper)
# ──────────────────────────────────────────────
PIPER_EXE_PATH: str = "./piper/piper.exe"
PIPER_VOICE_MODEL: str = "./piper/en_US-lessac-medium.onnx"

# ──────────────────────────────────────────────
#  Audio Settings
# ──────────────────────────────────────────────
AUDIO_RATE: int = 16000          # 16kHz sample rate (required by Whisper & Porcupine)
AUDIO_CHANNELS: int = 1         # Mono
AUDIO_FORMAT_WIDTH: int = 2     # 16-bit (2 bytes per sample)

# ──────────────────────────────────────────────
#  Silence / Listening Detection
# ──────────────────────────────────────────────
SILENCE_THRESHOLD: int = 500    # RMS energy below this = silence
SILENCE_DURATION: float = 1.5   # Seconds of silence before we stop recording
MAX_LISTEN_SECONDS: float = 15  # Hard cap on recording duration

# ──────────────────────────────────────────────
#  UI / Feedback
# ──────────────────────────────────────────────
PLAY_ACTIVATION_CHIME: bool = True  # Audible beep when wake word detected
CHIME_FREQUENCY: int = 880          # Hz (A5 note)
CHIME_DURATION: float = 0.15        # Seconds
