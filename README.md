# 🤖 Agentic Voice Assistant

A fully local, privacy-first voice assistant powered by **CrewAI**, **OpenAI Whisper** (STT), and **Piper** (TTS). Speak to it, and a crew of AI agents figures out what you need — web search, weather, math, reminders, and more — then speaks back the answer.

---

## ✨ Features

- 🎙️ **Wake-word activation** — say "Hey Luna" to trigger listening (or disable it entirely)
- 🧠 **Agentic reasoning** — a CrewAI crew with an Orchestrator, Research Specialist, and Task Agent handles your query
- 🔊 **Offline TTS** — Piper synthesises natural-sounding speech locally; no cloud required
- 📝 **Whisper STT** — OpenAI Whisper transcribes your voice with high accuracy
- 🌐 **Web search** — live results via DuckDuckGo API
- ☁️ **Weather** — current conditions for any city via wttr.in
- 🧮 **Calculator** — evaluates math expressions safely
- 🕐 **Time & date** — current time in any timezone
- 🔗 **Open URLs** — launch any URL in your default browser
- ⏰ **Reminders** — schedule spoken reminders after a delay
- 💬 **Text mode** — `--text` flag runs the assistant without a microphone (great for testing)
- 🔒 **100% local LLM** — powered by [Ollama](https://ollama.com); your data never leaves your machine

---

## 🏗️ Architecture

```
User Speech
    │
    ▼
┌─────────────────┐
│   stt.py        │  Whisper transcription (base model, CPU/CUDA)
│  (Whisper STT)  │
└────────┬────────┘
         │  Transcribed text
         ▼
┌─────────────────────────────────────────────────────┐
│                    agents.py (CrewAI)               │
│                                                     │
│  ┌─────────────────────┐                            │
│  │  Orchestrator Agent │  ← Understands intent,     │
│  │                     │    delegates tasks         │
│  └──────────┬──────────┘                            │
│             │                                       │
│    ┌────────┴────────┐                              │
│    ▼                 ▼                              │
│  ┌──────────┐  ┌───────────────┐                   │
│  │ Research │  │  Task Agent   │                   │
│  │ Specialist│  │ (calc/remind/ │                   │
│  │(web/wthr/ │  │  open URL)   │                   │
│  │  time)   │  └───────────────┘                   │
│  └──────────┘                                       │
└─────────────────────┬───────────────────────────────┘
                      │  Plain-text response
                      ▼
             ┌─────────────────┐
             │    tts.py       │  Piper synthesis → audio playback
             │  (Piper TTS)    │
             └─────────────────┘
```

---

## 📁 Project Structure

```
agentic-voice-assistant/
├── main.py          # Entry point — voice loop & text loop
├── agents.py        # CrewAI agents (Orchestrator, Researcher, Task Agent)
├── tools.py         # All CrewAI tools (search, weather, calc, time, URL, reminder)
├── stt.py           # Whisper-based speech-to-text & wake word detection
├── tts.py           # Piper-based text-to-speech
├── config.py        # All configuration constants
├── audio/           # Temporary WAV files (input/output)
├── piper_models/    # Downloaded Piper voice models
├── requirements.txt
└── pyproject.toml
```

---

## ⚙️ Prerequisites

- **Python 3.10+** (a `.python-version` file is included for `pyenv`)
- **[Ollama](https://ollama.com)** installed and running locally
- A Piper-compatible voice model (see setup below)
- A working microphone (for voice mode)

---

## 🚀 Setup

### 1. Clone the repository

```bash
git clone https://github.com/yahyazoom17/agentic-voice-assistant.git
cd agentic-voice-assistant
```

### 2. Install dependencies

Using `pip`:
```bash
pip install -r requirements.txt
```

Or using `uv` (recommended — a `uv.lock` is included):
```bash
uv sync
```

### 3. Pull an Ollama model

The default model is `gemma4:latest`. Pull it with:
```bash
ollama pull gemma4
```

You can use any model supported by Ollama — just update `LLM_MODEL` in `config.py`.

### 4. Download a Piper voice model

The default voice is `en_US-amy-medium`. Download it into `piper_models/`:
```bash
python -m piper.download en_US-amy-medium
```

Browse all available voices at [rhasspy/piper](https://github.com/rhasspy/piper).

### 5. Start Ollama

```bash
ollama serve
```

---

## ▶️ Running the Assistant

**Voice mode (default):**
```bash
python main.py
```

Say **"Hey Luna"** to wake the assistant, then speak your query. Say *goodbye*, *bye*, or *exit* to quit.

**Text mode (no microphone needed):**
```bash
python main.py --text
```

Type your queries at the prompt and press Enter.

---

## 🛠️ Configuration

All settings live in `config.py`:

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `"base"` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large` |
| `WHISPER_DEVICE` | `"cpu"` | Use `"cuda"` for GPU acceleration |
| `PIPER_VOICE` | `"en_US-amy-medium"` | Piper voice model name |
| `LLM_MODEL` | `"ollama/gemma4:latest"` | Any Ollama-hosted model |
| `OLLAMA_BASE_URL` | `"http://localhost:11434"` | Overridable via `OLLAMA_BASE_URL` env var |
| `WAKE_WORD` | `"hey luna"` | Set to `""` to disable wake-word mode |
| `SILENCE_THRESHOLD` | `500` | RMS threshold for end-of-speech detection |
| `SILENCE_DURATION` | `1.5` | Seconds of silence before stopping recording |
| `MAX_RECORD_SECONDS` | `30` | Hard cap per utterance |

---

## 🧰 Available Tools

The agents have access to the following tools defined in `tools.py`:

| Tool | What it does |
|---|---|
| `web_search` | Searches DuckDuckGo for current information |
| `get_weather` | Fetches live weather for any city via wttr.in |
| `calculator` | Safely evaluates Python math expressions |
| `get_time_date` | Returns current date/time in any timezone |
| `open_url` | Opens a URL in the system's default browser |
| `set_reminder` | Schedules a spoken reminder after N seconds |

---

## 🤝 Agents

| Agent | Role |
|---|---|
| **Voice Assistant Orchestrator** | Central brain — understands intent, delegates tasks, synthesises the final spoken-friendly response |
| **Research Specialist** | Handles web search, weather, and time/date queries |
| **Task Execution Specialist** | Handles calculations, reminders, and opening URLs |

---

## 📦 Key Dependencies

- [CrewAI](https://github.com/joaomdmoura/crewai) — multi-agent orchestration framework
- [openai-whisper](https://github.com/openai/whisper) — speech-to-text
- [piper-tts](https://github.com/rhasspy/piper) — fast, local text-to-speech
- [Ollama](https://ollama.com) — local LLM inference server

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
