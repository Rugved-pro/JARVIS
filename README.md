# JARVIS — Local Voice-Activated AI Assistant (Ongoing)

A fully local, low-latency voice assistant powered by:
- **Wake Word**: openWakeWord (`"Hey Jarvis"`) — no API key needed!
- **Speech-to-Text**: faster-whisper (`tiny.en` model)
- **LLM Brain**: Ollama (`llama3.1:latest`)
- **Text-to-Speech**: Piper TTS

## Objective
**This project aims to experiment just how seamless SLM integration can be and how much power can SLMs safely handle.
The challenge is the efficient and smooth operation of SLMs in normal laptops
Runs on a machine with an **RTX 4050 (6GB VRAM)** — the LLM lives on the GPU, everything else runs on CPU.**

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| Ollama | Latest |
| PortAudio | System library (needed by PyAudio) |
| Piper TTS | v2023.11.14-2 or later |

---

## Step-by-Step Setup

### 1. Install Ollama & Pull the Model

1. **Download Ollama** from [ollama.com](https://ollama.com/).
2. Install and open it — the server starts automatically.
3. Pull the LLM model:
   ```bash
   ollama pull llama3.1:latest
   ```
4. Verify the server is running:
   ```bash
   curl http://localhost:11434/api/tags
   ```
   You should see a JSON response listing `llama3.1:latest`.

### 2. Download Piper TTS

1. Go to the **[Piper Releases](https://github.com/rhasspy/piper/releases)** page.
2. Download the **Windows** release (e.g., `piper_windows_amd64.zip`).
3. Extract and place the files so your directory looks like:
   ```
   JARVIS/
   └── piper/
       ├── piper.exe
       ├── espeak-ng-data/   (included in the release)
       └── ...
   ```
4. **Download a voice model** from **[Piper Voices](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/lessac/medium)**:
   - Download both `en_US-lessac-medium.onnx` and `en_US-lessac-medium.onnx.json`
   - Place both files in the `piper/` directory:
     ```
     JARVIS/
     └── piper/
         ├── piper.exe
         ├── en_US-lessac-medium.onnx
         ├── en_US-lessac-medium.onnx.json
         └── espeak-ng-data/
     ```

### 3. Install PortAudio (Required by PyAudio)

**Windows** — PyAudio wheels usually bundle PortAudio. If `pip install pyaudio` fails:
```bash
pip install pipwin
pipwin install pyaudio
```

**macOS**:
```bash
brew install portaudio
```

**Linux (Debian/Ubuntu)**:
```bash
sudo apt-get install portaudio19-dev
```

### 4. Install Python Dependencies

```bash
cd JARVIS
pip install -r requirements.txt
```

### 5. Run JARVIS

```bash
python main.py
```

You should see:
```
     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
     ██║███████║██████╔╝██║   ██║██║███████╗
██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
 ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝
    Local Voice AI Assistant  •  v1.0

==================================================
  Say 'Hey Jarvis' to activate...
  Press Ctrl+C to quit.
==================================================
```

---

## How It Works

```
Mic → [openWakeWord] → "Hey Jarvis!" → [Record Speech] → [Whisper STT]
                                                              ↓
                                                    [Ollama llama3.1]
                                                              ↓
                                                     [Piper TTS] → Speaker
```

1. **IDLE** — The mic stream continuously feeds 80ms frames to openWakeWord. Near-zero CPU.
2. **LISTENING** — On wake word detection, audio is recorded until silence is detected (configurable in `config.py`).
3. **THINKING** — The audio buffer is transcribed by faster-whisper, then sent to Ollama for a response.
4. **SPEAKING** — The LLM response streams sentence-by-sentence to Piper TTS for near-real-time playback.

---

## Configuration

All tunable parameters are in **`config.py`**:

| Parameter | Default | Description |
|---|---|---|
| `WAKEWORD_THRESHOLD` | `0.5` | Confidence threshold for wake word detection |
| `SILENCE_THRESHOLD` | `500` | RMS threshold for silence detection |
| `SILENCE_DURATION` | `1.5s` | How long silence must last to stop recording |
| `MAX_LISTEN_SECONDS` | `15s` | Hard cap on recording duration |
| `WHISPER_MODEL` | `tiny.en` | Whisper model size (tiny/base/small) |
| `OLLAMA_MODEL` | `llama3.1:latest` | Which Ollama model to use |
| `PLAY_ACTIVATION_CHIME` | `True` | Play a beep on wake word detection |

---

## Troubleshooting

### PyAudio won't install
```bash
pip install pipwin && pipwin install pyaudio
```
Or download a pre-built wheel from [Unofficial Windows Binaries](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio).

### "Cannot connect to Ollama"
Make sure Ollama is running:
```bash
ollama serve
```
And verify the model is downloaded:
```bash
ollama list
```

### Wake word not detecting
- Check your microphone is the default input device.
- Try lowering `WAKEWORD_THRESHOLD` in `config.py` (e.g., `0.3`).
- Speak clearly: say **"Hey Jarvis"** (not just "Jarvis").
- The model auto-downloads on first run — ensure you have internet for the initial setup.

### Piper not producing audio
- Ensure both `.onnx` and `.onnx.json` files are in the `piper/` directory.
- Test Piper standalone: `echo "Hello world" | piper\piper.exe --model piper\en_US-lessac-medium.onnx --output-raw > test.raw`

---

## Future Plans

- **MCP Integration**: Interface with VS Code via Model Context Protocol for system control.
- **Tool Use**: Allow Jarvis to execute actions (file operations, app launching) through a secure MCP bridge.
- **GUI**: Add a minimal overlay UI showing state and transcript.

---

## License

MIT
