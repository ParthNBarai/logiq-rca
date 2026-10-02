"""Hypothesis agent — generates ranked root-cause hypotheses from evidence.

Sends the top 20 evidence lines (formatted for readability) to Gemini and
asks it to produce 2–3 hypotheses with confidence scores and references to
specific log lines by ``ingestion_id``.

The ``ingestion_id`` references are validated in the Evidence agent —
Gemini cannot cite a log line that wasn't in the prompt, preventing
hallucination of fabricated log entries.
"""

import logging
import textwrap
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from agents.state import RCAState
from agents.utils import parse_llm_json

logger = logging.getLogger(__name__)

_EVIDENCE_WINDOW = 20   # max lines sent to Gemini (context efficiency)

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert Site Reliability Engineer performing root cause analysis.

    You will be given a list of log lines retrieved for a specific tenant, each prefixed
    with its ingestion_id. Based on these logs and the user's query, generate 2–3 ranked
    hypotheses about what caused the incident.

    Return ONLY a valid JSON array — no markdown, no explanation.
    Each element must have exactly these fields:
    {
        "hypothesis": "<one clear sentence describing the root cause>",
        "confidence": <float 0.0 to 1.0>,
        "supporting_log_ids": ["<ingestion_id>", ...]
    }

    Rules:
    - supporting_log_ids MUST reference ingestion_ids from the provided log lines only.
      Do NOT invent ingestion_ids that are not in the list.
    - Rank hypotheses by confidence descending (highest first).
    - If evidence is ambiguous, assign lower confidence and say so in the hypothesis text.
    - Focus on ROOT CAUSES (what triggered the failure chain), not symptoms.
""")

_FALLBACK_HYPOTHESIS = [
    {
        "hypothesis": "Insufficient evidence to determine root cause. Broaden the time window or check additional tenants.",
        "confidence": 0.1,
        "supporting_log_ids": [],
    }
]


class HypothesisItem(BaseModel):
    """Single validated hypothesis from Gemini."""

    hypothesis: str
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_log_ids: list[str] = Field(default_factory=list)


class HypothesisAgent:
    """Generates candidate root-cause hypotheses from merged evidence.

    Args:
        llm: Shared ``ChatGoogleGenerativeAI`` instance.
    """

    def __init__(self, llm: ChatGoogleGenerativeAI) -> None:
        self._llm = llm

    async def run(self, state: RCAState) -> dict[str, Any]:
        """Generate hypotheses from ``state["merged_evidence"]``.

        Returns a partial state update with ``hypotheses``.  Falls back
        to a single low-confidence hypothesis on any failure.
        """
        evidence = state.get("merged_evidence") or []
        tenant_id = state["tenant_id"]

        if not evidence:
            logger.warning("[Hypothesis] No evidence available — returning fallback.")
            return {"hypotheses": _FALLBACK_HYPOTHESIS}

        top_evidence = evidence[:_EVIDENCE_WINDOW]
        evidence_text = self._format_evidence(top_evidence)

        user_content = (
            f"Tenant: {tenant_id}\n"
            f"Query: {state['query']}\n\n"
            f"Log evidence:\n{evidence_text}"
        )

        try:
            raw = await self._call_gemini(user_content)
            hypotheses = self._parse_and_validate(raw)
        except Exception:
            logger.warning("[Hypothesis] Gemini call failed; using fallback.", exc_info=True)
            return {"hypotheses": _FALLBACK_HYPOTHESIS}

        if not hypotheses:
            hypotheses = _FALLBACK_HYPOTHESIS

        logger.info("[Hypothesis] Generated %d hypotheses.", len(hypotheses))
        return {"hypotheses": hypotheses}

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    async def _call_gemini(self, user_content: str) -> str:
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]
        response = await self._llm.ainvoke(messages)
        return str(response.content)

    def _parse_and_validate(self, raw: str) -> list[dict]:
        """Strip fences, parse JSON array, validate each item with Pydantic."""
        data = parse_llm_json(raw)

        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array, got {type(data)}")

        validated: list[dict] = []
        for item in data:
            try:
                h = HypothesisItem(**item)
                validated.append(h.model_dump())
            except Exception as exc:
                logger.debug("[Hypothesis] Skipping malformed item: %s — %s", item, exc)

        return validated

    @staticmethod
    def _format_evidence(evidence: list[dict]) -> str:
        """Format evidence dicts as numbered, human-readable log entries.

        Each entry is prefixed with its ingestion_id so Gemini can reference
        specific lines in ``supporting_log_ids``.
        """
        lines: list[str] = []
        for i, e in enumerate(evidence, 1):
            lines.append(
                f"[{i}] ingestion_id: {e.get('ingestion_id', 'N/A')}\n"
                f"     {e.get('timestamp', '')} {e.get('level', '')} "
                f"{e.get('component', '')}: {e.get('message', '')}"
            )
        return "\n".join(lines)
