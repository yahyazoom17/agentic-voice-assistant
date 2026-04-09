import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
AUDIO_DIR = BASE_DIR / "audio"
AUDIO_DIR.mkdir(exist_ok=True)

INPUT_AUDIO_PATH  = AUDIO_DIR / "input.wav"
OUTPUT_AUDIO_PATH = AUDIO_DIR / "output.wav"

# ── Whisper STT ────────────────────────────────────────────────────────────
WHISPER_MODEL      = "base"          # tiny | base | small | medium | large
WHISPER_LANGUAGE   = None            # None = auto-detect, or e.g. "en"
WHISPER_DEVICE     = "cpu"           # "cpu" or "cuda"

# ── Piper TTS ──────────────────────────────────────────────────────────────
PIPER_MODEL_DIR    = BASE_DIR / "piper_models"
PIPER_MODEL_DIR.mkdir(exist_ok=True)
# Default voice — download with: python -m piper.download <voice>
PIPER_VOICE        = "en_US-amy-medium"

# ── Audio recording ────────────────────────────────────────────────────────
SAMPLE_RATE        = 16_000
CHANNELS           = 1
SILENCE_THRESHOLD  = 500            # RMS threshold to detect end-of-speech
SILENCE_DURATION   = 1.5            # seconds of silence before stopping
MAX_RECORD_SECONDS = 30             # hard cap per utterance

# ── CrewAI / LLM (Ollama) ─────────────────────────────────────────────────
OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL          = "ollama/gemma4:latest"  # any model pulled in Ollama

# ── Wake word (optional) ───────────────────────────────────────────────────
WAKE_WORD          = "hey luna"   # set to "" to disable
