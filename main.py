"""
main.py — Entry point for the Agentic AI Voice Assistant
"""

from __future__ import annotations

import logging
import sys

# ── Logging setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def check_ollama() -> None:
    import requests
    from config import LLM_MODEL, OLLAMA_BASE_URL
    model_name = LLM_MODEL.removeprefix("ollama/")
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        # Ollama names may include a ":latest" tag
        pulled = any(m.split(":")[0] == model_name.split(":")[0] for m in models)
        if not pulled:
            logger.warning(
                "Model '%s' not found locally. Pull it with: ollama pull %s",
                model_name, model_name,
            )
    except Exception:
        logger.error(
            "Cannot reach Ollama at %s. Make sure Ollama is running: ollama serve",
            OLLAMA_BASE_URL,
        )
        sys.exit(1)


def run_voice_loop() -> None:
    """Main loop: listen → transcribe → agent → speak → repeat."""
    from config import WAKE_WORD
    from stt import listen, wait_for_wake_word
    from tts import speak, synthesize
    from agents import VoiceAssistantCrew

    crew = VoiceAssistantCrew()

    print("\n" + "═" * 60)
    print("  🤖  Agentic Voice Assistant  (Whisper + Piper + CrewAI)")
    print("═" * 60)
    if WAKE_WORD:
        print(f"  Wake word: '{WAKE_WORD}'  |  Say 'goodbye' to exit")
    else:
        print("  Press  Ctrl-C  to exit  |  Say 'goodbye' to exit")
    print("═" * 60 + "\n")

    speak("Hello! I am Luna, your voice assistant. How can I help you?")

    while True:
        try:
            # ── Wait for wake word (if enabled) ───────────────────────────
            if WAKE_WORD:
                wait_for_wake_word()

            # ── Listen ─────────────────────────────────────────────────────
            print("🎙  Listening …")
            user_text = listen()

            if not user_text.strip():
                speak("I didn't catch that. Could you repeat?")
                continue

            print(f"👤  You said: {user_text}")

            # ── Exit command ────────────────────────────────────────────────
            if any(kw in user_text.lower() for kw in ("goodbye", "bye", "exit", "quit", "stop")):
                speak("Goodbye! Have a great day!")
                break

            # ── Agent processing ────────────────────────────────────────────
            print("🤔  Thinking …")
            response = crew.process_query(user_text)
            print(f"🤖  Assistant: {response}\n")

            # ── Speak response ──────────────────────────────────────────────
            speak(response)

        except KeyboardInterrupt:
            speak("Goodbye!")
            break
        except Exception as exc:
            logger.exception("Unexpected error: %s", exc)
            speak("Sorry, something went wrong. Please try again.")


def run_text_loop() -> None:
    """Text-only loop for testing without a microphone."""
    from tts import speak
    from agents import VoiceAssistantCrew

    crew = VoiceAssistantCrew()
    print("\n[TEXT MODE]  Type your query (or 'quit' to exit)\n")
    speak("Hello! I am Luna, your voice assistant. Running in text mode. Type your query.")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_text:
            continue
        if user_text.lower() in ("quit", "exit", "goodbye"):
            speak("Goodbye!")
            break

        response = crew.process_query(user_text)
        print(f"Assistant: {response}\n")
        speak(response)


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    check_ollama()
    mode = "voice"
    
    if "--text" in sys.argv:
        mode = "text"

    if mode == "text":
        run_text_loop()
    else:
        run_voice_loop()
