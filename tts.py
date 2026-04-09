"""
tts.py — Text-to-Speech via Piper TTS
"""

from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Optional

from config import OUTPUT_AUDIO_PATH, PIPER_MODEL_DIR, PIPER_VOICE

logger = logging.getLogger(__name__)

# ── Piper TTS ──────────────────────────────────────────────────────────────


def _piper_model_path() -> tuple[Path, Path]:
    """Return (onnx_path, config_path) for the configured voice."""
    onnx   = PIPER_MODEL_DIR / f"{PIPER_VOICE}.onnx"
    config = PIPER_MODEL_DIR / f"{PIPER_VOICE}.onnx.json"
    return onnx, config


def ensure_voice_downloaded() -> None:
    """Download the Piper voice model if not already present."""
    onnx, config = _piper_model_path()
    if onnx.exists() and config.exists():
        return

    logger.info("Downloading Piper voice '%s' …", PIPER_VOICE)
    try:
        subprocess.run(
            [
                sys.executable, "-m", "piper.download",
                "--voice", PIPER_VOICE,
                "--data-dir", str(PIPER_MODEL_DIR),
            ],
            check=True,
        )
        logger.info("Voice downloaded to %s", PIPER_MODEL_DIR)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Failed to download Piper voice '{PIPER_VOICE}'. "
            "Run manually: python -m piper.download --voice <voice> --data-dir piper_models/"
        ) from exc


def synthesize(text: str, output_path: Optional[Path] = None) -> Path:
    """
    Convert text to speech with Piper TTS.

    Args:
        text:        The text to synthesize.
        output_path: Where to save the WAV. Defaults to OUTPUT_AUDIO_PATH.

    Returns:
        Path to the generated WAV file.
    """
    ensure_voice_downloaded()
    out = output_path or OUTPUT_AUDIO_PATH
    onnx, config = _piper_model_path()

    logger.info("Synthesizing speech …")
    cmd = [
        sys.executable, "-m", "piper",
        "--model",      str(onnx),
        "--config",     str(config),
        "--output_file", str(out),
    ]
    proc = subprocess.run(
        cmd,
        input=text.encode(),
        capture_output=True,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace")
        raise RuntimeError(f"Piper TTS failed:\n{stderr}")

    logger.info("Speech saved → %s", out)
    return out


def speak(text: str, blocking: bool = True) -> None:
    """
    Synthesize text and play it through the system speaker.

    Args:
        text:     Text to speak.
        blocking: If True, wait until playback finishes.
    """
    wav_path = synthesize(text)
    _play(wav_path, blocking=blocking)


def _play(wav_path: Path, blocking: bool = True) -> None:
    """Play a WAV file using the best available method."""
    try:
        import sounddevice as sd
        import soundfile as sf

        data, samplerate = sf.read(str(wav_path), dtype="float32")
        if blocking:
            sd.play(data, samplerate)
            sd.wait()
        else:
            def _bg():
                sd.play(data, samplerate)
                sd.wait()
            threading.Thread(target=_bg, daemon=True).start()
        return
    except ImportError:
        pass

    # Fallback: aplay (Linux) / afplay (macOS) / powershell (Windows)
    import platform
    system = platform.system()
    if system == "Linux":
        player_cmd = ["aplay", str(wav_path)]
    elif system == "Darwin":
        player_cmd = ["afplay", str(wav_path)]
    else:
        player_cmd = ["powershell", "-c", f"(New-Object Media.SoundPlayer '{wav_path}').PlaySync()"]

    if blocking:
        subprocess.run(player_cmd, check=True)
    else:
        subprocess.Popen(player_cmd)
