"""Shared utilities for agent LLM calls and response parsing.

Every agent that talks to Gemini goes through ``parse_llm_json`` so JSON
extraction logic lives in exactly one place.
"""

import json
import logging
import re
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


def parse_llm_json(text: str) -> Any:
    """Strip markdown code fences from *text* and return parsed JSON.

    Handles the three common Gemini wrapping patterns:
      ```json { ... } ```
      ``` { ... } ```
      { ... }          (bare JSON — returned directly)

    Raises ``json.JSONDecodeError`` if the stripped text is not valid JSON.
    """
    stripped = text.strip()
    # Remove opening fence (with or without language tag)
    stripped = re.sub(r"^```(?:json|JSON)?\s*\n?", "", stripped)
    # Remove closing fence
    stripped = re.sub(r"\n?```\s*$", "", stripped)
    return json.loads(stripped.strip())


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------


def build_llm() -> ChatGoogleGenerativeAI:
    """Return a zero-temperature Gemini Flash instance.

    Uses ``gemini-2.5-flash`` for fast, deterministic structured-output
    generation.  All agents share the same model but call it independently
    so timeouts in one node don't affect others.
    """
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=settings.GEMINI_API_KEY,
    )
