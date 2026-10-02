"""Planner agent — converts a natural-language RCA query into structured
search parameters that the retriever can execute deterministically.

Design:
  - Single Gemini call with a strict JSON schema in the system prompt.
  - Python post-processes the output: datetime parsing, ±15 min buffer.
  - All failures fall back to safe defaults so the graph never crashes here.
"""

import logging
import textwrap
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field, field_validator

from agents.state import RCAState
from agents.utils import parse_llm_json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic model for strict output validation
# ---------------------------------------------------------------------------


class PlannerOutput(BaseModel):
    """Validated, typed representation of the planner's Gemini response."""

    time_window_start: Optional[str] = None
    time_window_end: Optional[str] = None
    search_keywords: list[str] = Field(default_factory=list)
    error_levels: list[str] = Field(default=["ERROR", "WARN"])
    components_of_interest: list[str] = Field(default_factory=list)

    @field_validator("error_levels")
    @classmethod
    def validate_levels(cls, v: list[str]) -> list[str]:
        valid = {"DEBUG", "INFO", "WARN", "ERROR", "FATAL"}
        return [lvl.upper() for lvl in v if lvl.upper() in valid] or ["ERROR", "WARN"]


# ---------------------------------------------------------------------------
# Safe defaults
# ---------------------------------------------------------------------------

_DEFAULT_ERROR_LEVELS = ["ERROR", "WARN"]
_TIME_BUFFER = timedelta(minutes=15)

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a log analysis query parser for an enterprise incident RCA system.
    The log data covers the date 2026-05-26 (UTC). Times without a date use this date.

    Parse the user's natural-language incident query into structured search parameters.
    Return ONLY a valid JSON object with EXACTLY these fields — no markdown, no explanation:

    {
        "time_window_start": "<ISO8601 datetime string or null>",
        "time_window_end":   "<ISO8601 datetime string or null>",
        "search_keywords":   ["keyword1", "keyword2"],
        "error_levels":      ["ERROR", "WARN"],
        "components_of_interest": ["ComponentShortName"]
    }

    Rules:
    - time_window_start / time_window_end:
        * "around <HH:MM>" → start = HH:MM - 15 min, end = HH:MM + 15 min.
        * "between X and Y" → use that range directly.
        * No time mentioned → null for both.
        * Always use date 2026-05-26 unless the user specifies otherwise.
        * Format: 2026-05-26THH:MM:SS (no timezone suffix needed).
    - search_keywords: Short technical tokens to grep for (e.g. "503", "ConnectionTimeout",
      "rate limit", "DB-Cluster-01", "pool"). Empty list is fine.
    - error_levels: Default ["ERROR", "WARN"]. Add "INFO" only if broad activity is needed.
    - components_of_interest: Java class short-names from these options:
        DataSource, Breaker, Ingress, Router, PolicyEnforcer, RateLimiter,
        TokenValidator, QueueManager, RedisClient, Reporter.
      Map natural language: "database" → "DataSource", "circuit breaker" → "Breaker",
      "gateway" → "Ingress", "auth" → "TokenValidator", "rate limit" → "RateLimiter".
      Empty list means "all components".
""")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class PlannerAgent:
    """Decomposes a natural-language query into structured retrieval parameters.

    Args:
        llm: Shared ``ChatGoogleGenerativeAI`` instance.
    """

    def __init__(self, llm: ChatGoogleGenerativeAI) -> None:
        self._llm = llm

    async def run(self, state: RCAState) -> dict[str, Any]:
        """Parse ``state["query"]`` and return structured search parameters.

        Returns a partial state update dict. Never raises — falls back to
        safe defaults if Gemini is unavailable or returns malformed JSON.
        """
        query = state["query"]
        logger.info("[Planner] Parsing query: %r", query)

        try:
            raw = await self._call_gemini(query)
            output = self._parse_and_validate(raw)
        except Exception:
            logger.warning("[Planner] Gemini call failed; using safe defaults.", exc_info=True)
            output = PlannerOutput()

        tw_start, tw_end = self._resolve_time_window(
            query, output.time_window_start, output.time_window_end
        )

        logger.info(
            "[Planner] Result: window=%s→%s keywords=%s levels=%s components=%s",
            tw_start,
            tw_end,
            output.search_keywords,
            output.error_levels,
            output.components_of_interest,
        )

        return {
            "time_window_start": tw_start,
            "time_window_end": tw_end,
            "search_keywords": output.search_keywords,
            "error_levels": output.error_levels,
            "components_of_interest": output.components_of_interest,
        }

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    async def _call_gemini(self, query: str) -> str:
        """Invoke Gemini and return the raw response text."""
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"Query: {query}"),
        ]
        response = await self._llm.ainvoke(messages)
        return str(response.content)

    def _parse_and_validate(self, raw: str) -> PlannerOutput:
        """Strip fences, parse JSON, validate with Pydantic."""
        data = parse_llm_json(raw)
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object, got {type(data)}")
        return PlannerOutput(**data)

    @staticmethod
    def _resolve_time_window(
        query: str,
        raw_start: Optional[str],
        raw_end: Optional[str],
    ) -> tuple[Optional[datetime], Optional[datetime]]:
        """Parse ISO strings from Gemini and optionally apply ±15 min buffer.

        The buffer is re-applied in Python (even if Gemini already did it)
        only when the query contains "around" or "about", ensuring consistent
        ±15 min semantics regardless of Gemini's interpretation.
        """
        tw_start: Optional[datetime] = None
        tw_end: Optional[datetime] = None

        if raw_start:
            try:
                tw_start = datetime.fromisoformat(raw_start.replace("Z", "")).replace(
                    tzinfo=None
                )
            except ValueError:
                logger.debug("[Planner] Could not parse time_window_start: %r", raw_start)

        if raw_end:
            try:
                tw_end = datetime.fromisoformat(raw_end.replace("Z", "")).replace(
                    tzinfo=None
                )
            except ValueError:
                logger.debug("[Planner] Could not parse time_window_end: %r", raw_end)

        # Python-side buffer when "around" / "about" phrasing is detected
        # and Gemini returned the same value for start/end (i.e. a point, not range).
        around_keywords = ("around", "about", "near", "approximately")
        if any(kw in query.lower() for kw in around_keywords):
            if tw_start and tw_end and abs((tw_end - tw_start).total_seconds()) < 60:
                # Gemini returned essentially the same point — apply buffer
                tw_start = tw_start - _TIME_BUFFER
                tw_end = tw_end + _TIME_BUFFER
            elif tw_start and not tw_end:
                tw_end = tw_start + _TIME_BUFFER
                tw_start = tw_start - _TIME_BUFFER

        return tw_start, tw_end
