"""
stt.py — Speech-to-Text via OpenAI Whisper
"""

from __future__ import annotations

import io
import logging
import struct
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import pyaudio
import whisper

from config import (
    CHANNELS,
    INPUT_AUDIO_PATH,
    MAX_RECORD_SECONDS,
    SAMPLE_RATE,
    SILENCE_DURATION,
    SILENCE_THRESHOLD,
    WAKE_WORD,
    WHISPER_DEVICE,
    WHISPER_LANGUAGE,
    WHISPER_MODEL,
)

logger = logging.getLogger(__name__)

# ── Whisper model singleton ────────────────────────────────────────────────

_model: Optional[whisper.Whisper] = None


def load_model() -> whisper.Whisper:
    global _model
    if _model is None:
        logger.info("Loading Whisper model '%s' on %s …", WHISPER_MODEL, WHISPER_DEVICE)
        _model = whisper.load_model(WHISPER_MODEL, device=WHISPER_DEVICE)
        logger.info("Whisper model loaded.")
    return _model


# ── Audio recording ────────────────────────────────────────────────────────

CHUNK = 1024  # frames per PyAudio buffer read


def _rms(data: bytes) -> float:
    """Root-mean-square of a raw PCM (int16) byte string."""
    count = len(data) // 2
    if count == 0:
        return 0.0
    shorts = struct.unpack(f"{count}h", data)
    return float(np.sqrt(sum(s * s for s in shorts) / count))


def record_until_silence() -> Path:
    """
    Record microphone audio until SILENCE_DURATION seconds of silence,
    or MAX_RECORD_SECONDS total. Saves WAV to INPUT_AUDIO_PATH.
    """
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    logger.info("🎙  Recording … (speak now)")
    frames: list[bytes] = []
    silent_chunks = 0
    max_silent_chunks = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK)
    max_chunks = int(MAX_RECORD_SECONDS * SAMPLE_RATE / CHUNK)
    speech_detected = False

    for _ in range(max_chunks):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        if _rms(data) > SILENCE_THRESHOLD:
            speech_detected = True
            silent_chunks = 0
        elif speech_detected:
            silent_chunks += 1
            if silent_chunks >= max_silent_chunks:
                break

    stream.stop_stream()
    stream.close()
    pa.terminate()

    # Save WAV
    with wave.open(str(INPUT_AUDIO_PATH), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))

    logger.info("Recording saved → %s", INPUT_AUDIO_PATH)
    return INPUT_AUDIO_PATH


# ── Transcription ──────────────────────────────────────────────────────────


def transcribe(audio_path: Optional[Path] = None) -> str:
    """
    Transcribe audio file with Whisper.

    Args:
        audio_path: Path to WAV/MP3/etc. Defaults to INPUT_AUDIO_PATH.

    Returns:
        Transcribed text string (stripped, lower-cased).
    """
    model = load_model()
    path = str(audio_path or INPUT_AUDIO_PATH)
    opts: dict = {}
    if WHISPER_LANGUAGE:
        opts["language"] = WHISPER_LANGUAGE
    result = model.transcribe(path, fp16=False, **opts)
    text = result.get("text", "").strip()
    logger.info("Transcription: %r", text)
    return text


# ── Wake-word gate ─────────────────────────────────────────────────────────


def wait_for_wake_word() -> None:
    """
    Continuously record short audio clips until the wake word is detected.
    Only active when WAKE_WORD is non-empty.
    """
    if not WAKE_WORD:
        return

    logger.info("⏳  Waiting for wake word: %r", WAKE_WORD)
    while True:
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )
        frames = [stream.read(CHUNK, exception_on_overflow=False) for _ in range(int(3 * SAMPLE_RATE / CHUNK))]
        stream.stop_stream()
        stream.close()
        pa.terminate()

        with wave.open(str(INPUT_AUDIO_PATH), "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(frames))

        text = transcribe(INPUT_AUDIO_PATH)
        if WAKE_WORD.lower() in text.lower():
            logger.info("✅  Wake word detected.")
            return


# ── Convenience: record + transcribe in one call ───────────────────────────


def listen() -> str:
    """Record microphone input and return transcription."""
    record_until_silence()
    return transcribe()
