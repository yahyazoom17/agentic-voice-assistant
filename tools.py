"""
tools.py — CrewAI tools available to the voice assistant agents
"""

from __future__ import annotations

import datetime
import json
import subprocess
import sys
import webbrowser
from typing import Type

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


# ── Tool schemas ───────────────────────────────────────────────────────────


class WebSearchInput(BaseModel):
    query: str = Field(..., description="The search query string")


class WeatherInput(BaseModel):
    city: str = Field(..., description="City name, e.g. 'London' or 'New York'")


class CalculatorInput(BaseModel):
    expression: str = Field(..., description="A valid Python math expression, e.g. '2 ** 10 + 5'")


class TimeDateInput(BaseModel):
    timezone: str = Field(default="local", description="Timezone name, e.g. 'UTC', 'US/Eastern'. 'local' uses system time.")


class OpenURLInput(BaseModel):
    url: str = Field(..., description="The full URL to open in the default browser")


class ReminderInput(BaseModel):
    message: str = Field(..., description="Reminder message text")
    delay_seconds: int = Field(..., description="Seconds from now to trigger the reminder")


# ── Tool implementations ───────────────────────────────────────────────────


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "Search the internet for current information. "
        "Use this for news, facts, or anything that may have changed recently."
    )
    args_schema: Type[BaseModel] = WebSearchInput

    def _run(self, query: str) -> str:
        try:
            resp = requests.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
                timeout=8,
            )
            data = resp.json()
            abstract = data.get("AbstractText", "")
            related  = [r.get("Text", "") for r in data.get("RelatedTopics", [])[:3] if "Text" in r]
            if abstract:
                return abstract
            if related:
                return " | ".join(related)
            return f"No direct results found for: {query}"
        except Exception as exc:
            return f"Search failed: {exc}"


class WeatherTool(BaseTool):
    name: str = "get_weather"
    description: str = "Get current weather conditions for a given city."
    args_schema: Type[BaseModel] = WeatherInput

    def _run(self, city: str) -> str:
        try:
            resp = requests.get(
                "https://wttr.in/" + requests.utils.quote(city),
                params={"format": "j1"},
                timeout=8,
            )
            data = resp.json()
            current = data["current_condition"][0]
            desc   = current["weatherDesc"][0]["value"]
            temp_c = current["temp_C"]
            feels  = current["FeelsLikeC"]
            humid  = current["humidity"]
            return (
                f"Weather in {city}: {desc}. "
                f"Temperature: {temp_c}°C (feels like {feels}°C), Humidity: {humid}%."
            )
        except Exception as exc:
            return f"Could not fetch weather for '{city}': {exc}"


class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = (
        "Evaluate a safe mathematical expression and return the result. "
        "Supports +, -, *, /, **, //, %, abs(), round(), etc."
    )
    args_schema: Type[BaseModel] = CalculatorInput

    _SAFE_GLOBALS = {"__builtins__": {}, "abs": abs, "round": round, "pow": pow}

    def _run(self, expression: str) -> str:
        try:
            result = eval(expression, self._SAFE_GLOBALS, {})  # noqa: S307
            return f"{expression} = {result}"
        except Exception as exc:
            return f"Calculation error: {exc}"


class TimeDateTool(BaseTool):
    name: str = "get_time_date"
    description: str = "Return the current date and time, optionally in a specific timezone."
    args_schema: Type[BaseModel] = TimeDateInput

    def _run(self, timezone: str = "local") -> str:
        try:
            if timezone == "local":
                now = datetime.datetime.now()
            else:
                import zoneinfo
                tz  = zoneinfo.ZoneInfo(timezone)
                now = datetime.datetime.now(tz)
            return now.strftime("Today is %A, %B %d %Y. The time is %I:%M %p.")
        except Exception as exc:
            return f"Time lookup error: {exc}"


class OpenURLTool(BaseTool):
    name: str = "open_url"
    description: str = "Open a URL in the user's default web browser."
    args_schema: Type[BaseModel] = OpenURLInput

    def _run(self, url: str) -> str:
        try:
            webbrowser.open(url)
            return f"Opened {url} in the browser."
        except Exception as exc:
            return f"Failed to open URL: {exc}"


class ReminderTool(BaseTool):
    name: str = "set_reminder"
    description: str = "Schedule a spoken reminder after a delay (in seconds)."
    args_schema: Type[BaseModel] = ReminderInput

    def _run(self, message: str, delay_seconds: int) -> str:
        import threading

        def _fire():
            import time
            time.sleep(delay_seconds)
            # Lazy import to avoid circular deps
            from tts import speak
            speak(f"Reminder: {message}")

        threading.Thread(target=_fire, daemon=True).start()
        return f"Reminder set: '{message}' in {delay_seconds} seconds."


# ── Exported tool list ─────────────────────────────────────────────────────

ALL_TOOLS = [
    WebSearchTool(),
    WeatherTool(),
    CalculatorTool(),
    TimeDateTool(),
    OpenURLTool(),
    ReminderTool(),
]
