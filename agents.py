"""
agents.py — CrewAI agents and crew for the voice assistant
"""

from __future__ import annotations

import logging

from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM

from config import LLM_MODEL, OLLAMA_BASE_URL
from tools import ALL_TOOLS

logger = logging.getLogger(__name__)

# ── LLM ───────────────────────────────────────────────────────────────────


def _make_llm() -> LLM:
    return LLM(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.4,
    )


# ── Agents ─────────────────────────────────────────────────────────────────


def build_orchestrator(llm: LLM) -> Agent:
    return Agent(
        role="Voice Assistant Orchestrator",
        goal=(
            "Understand what the user wants from their spoken request "
            "and coordinate specialist agents to produce a helpful, concise response "
            "suitable for text-to-speech playback (no markdown, no bullet points)."
        ),
        backstory=(
            "You are the central brain of a voice assistant. "
            "You receive transcribed speech, reason about intent, delegate tasks, "
            "and synthesize a final spoken-friendly answer."
        ),
        llm=llm,
        tools=ALL_TOOLS,
        verbose=True,
        allow_delegation=True,
        max_iter=6,
    )


def build_researcher(llm: LLM) -> Agent:
    return Agent(
        role="Research Specialist",
        goal="Find accurate, up-to-date information using available tools.",
        backstory=(
            "You are an expert at retrieving factual information quickly. "
            "You use web search, weather, and time tools to answer factual queries."
        ),
        llm=llm,
        tools=[t for t in ALL_TOOLS if t.name in ("web_search", "get_weather", "get_time_date")],
        verbose=True,
        allow_delegation=False,
        max_iter=4,
    )


def build_task_agent(llm: LLM) -> Agent:
    return Agent(
        role="Task Execution Specialist",
        goal="Execute user-requested actions such as calculations, reminders, and opening URLs.",
        backstory=(
            "You handle practical tasks for the user: doing maths, setting reminders, "
            "and launching web pages."
        ),
        llm=llm,
        tools=[t for t in ALL_TOOLS if t.name in ("calculator", "set_reminder", "open_url")],
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )


# ── Crew ───────────────────────────────────────────────────────────────────


class VoiceAssistantCrew:
    """Wraps CrewAI Crew with a simple process_query interface."""

    def __init__(self) -> None:
        llm = _make_llm()
        self.orchestrator = build_orchestrator(llm)
        self.researcher   = build_researcher(llm)
        self.task_agent   = build_task_agent(llm)

    def process_query(self, user_input: str) -> str:
        """
        Run the crew on a user query and return a TTS-friendly response string.

        Args:
            user_input: Transcribed user speech.

        Returns:
            Plain-text response for the TTS engine.
        """
        logger.info("Processing query: %r", user_input)

        understand_task = Task(
            description=(
                f"The user said: '{user_input}'\n\n"
                "Understand the intent. Decide whether this needs web research, "
                "a task action (calculation / reminder / open URL), or can be answered "
                "from general knowledge. If delegation is needed, call the right specialist."
            ),
            expected_output=(
                "A clear, complete, spoken-friendly answer to the user's request. "
                "No markdown, no lists, no asterisks. Write as you would speak aloud."
            ),
            agent=self.orchestrator,
        )

        crew = Crew(
            agents=[self.orchestrator, self.researcher, self.task_agent],
            tasks=[understand_task],
            process=Process.sequential,
            verbose=True,
        )

        result = crew.kickoff()

        # CrewAI may return an object; extract the string
        if hasattr(result, "raw"):
            return str(result.raw).strip()
        return str(result).strip()
