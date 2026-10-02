"""Synthesizer agent — produces the final grounded RCA narrative.

Takes the top validated hypothesis and asks Gemini to write a concise
RCA: root cause, confidence, causal chain, and recommended fix.

GROUNDING RULE (anti-hallucination):
  After Gemini responds, the ``evidence_sources`` field is REPLACED with
  the actual log lines from ``validated_hypotheses[0].evidence_lines``.
  This means evidence_sources are always real retrieved logs — Gemini
  cannot fabricate fake log entries in the final report.
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

_MAX_EVIDENCE_SOURCES = 5   # max grounded log lines in final output

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a senior Site Reliability Engineer writing a post-incident Root Cause Analysis.

    You will receive the top validated hypothesis for a production incident, along with
    supporting log evidence. Write a concise, actionable RCA.

    Return ONLY a valid JSON object — no markdown, no explanation:
    {
        "root_cause": "<one clear sentence — the technical root cause, not the symptom>",
        "confidence": <float 0.0 to 1.0>,
        "causal_chain": [
            "<earliest event>",
            "<intermediate event>",
            "<final symptom visible to users>"
        ],
        "evidence_sources": [],
        "recommended_fix": "<one specific, actionable remediation step>"
    }

    Rules:
    - root_cause: the underlying trigger, not the end-user visible symptom.
      Good: "Database connection pool exhausted due to long-running queries under load."
      Bad:  "Users saw 503 errors."
    - causal_chain: ordered list, earliest event first. 3–5 steps typical.
    - evidence_sources: leave as empty list [] — it will be replaced programmatically.
    - recommended_fix: concrete and specific (e.g. "Increase DB connection pool size from
      default to 50 and add a 30s query timeout on DataSource configuration.").
    - confidence: carry through from the hypothesis; reduce slightly if evidence is partial.
""")

_FALLBACK_RCA = {
    "root_cause": "Unable to synthesise RCA — LLM unavailable or evidence insufficient.",
    "confidence": 0.0,
    "causal_chain": [],
    "evidence_sources": [],
    "recommended_fix": "Investigate manually using the retrieved evidence.",
}


class SynthesizerOutput(BaseModel):
    """Validated Gemini response for the final RCA."""

    root_cause: str
    confidence: float = Field(ge=0.0, le=1.0)
    causal_chain: list[str] = Field(default_factory=list)
    evidence_sources: list[Any] = Field(default_factory=list)
    recommended_fix: str = ""


class SynthesizerAgent:
    """Produces the final grounded RCA from validated hypotheses.

    Args:
        llm: Shared ``ChatGoogleGenerativeAI`` instance.
    """

    def __init__(self, llm: ChatGoogleGenerativeAI) -> None:
        self._llm = llm

    async def run(self, state: RCAState) -> dict[str, Any]:
        """Synthesise ``validated_hypotheses`` into the final ``rca_result``.

        Returns a partial state update with ``rca_result``.
        """
        validated = state.get("validated_hypotheses") or []
        tenant_id = state["tenant_id"]

        if not validated:
            logger.warning("[Synthesizer] No validated hypotheses — returning fallback RCA.")
            return {"rca_result": {**_FALLBACK_RCA, "tenant_id": tenant_id}}

        top = validated[0]  # already sorted by confidence desc in EvidenceAgent
        user_content = self._build_prompt(state["query"], tenant_id, top)

        try:
            raw = await self._call_gemini(user_content)
            rca = self._parse_and_validate(raw)
        except Exception:
            logger.warning("[Synthesizer] Gemini call failed; building fallback from hypothesis.", exc_info=True)
            rca = self._fallback_from_hypothesis(top)

        # GROUNDING: replace whatever Gemini put in evidence_sources with
        # the actual retrieved log lines — prevents hallucination.
        rca["evidence_sources"] = self._ground_evidence(top)
        rca["tenant_id"] = tenant_id

        logger.info(
            "[Synthesizer] RCA complete. root_cause=%r confidence=%.2f",
            rca.get("root_cause", "")[:80],
            rca.get("confidence", 0.0),
        )

        return {"rca_result": rca}

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

    def _parse_and_validate(self, raw: str) -> dict:
        """Parse and validate Gemini's JSON response."""
        data = parse_llm_json(raw)
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object, got {type(data)}")
        output = SynthesizerOutput(**data)
        return output.model_dump()

    @staticmethod
    def _build_prompt(query: str, tenant_id: str, top_hypothesis: dict) -> str:
        """Format the top hypothesis and its evidence into the user message."""
        evidence_text = "\n".join(
            f"  [{i+1}] {e.get('timestamp', '')} {e.get('level', '')} "
            f"{e.get('component', '')}: {e.get('message', '')}"
            for i, e in enumerate(top_hypothesis.get("evidence_lines", [])[:10])
        )

        return (
            f"Tenant: {tenant_id}\n"
            f"Original query: {query}\n\n"
            f"Top hypothesis (confidence={top_hypothesis.get('confidence', 0):.2f}, "
            f"verdict={top_hypothesis.get('verdict', 'N/A')}):\n"
            f"  {top_hypothesis.get('hypothesis', '')}\n\n"
            f"Supporting evidence (chronological):\n"
            f"{evidence_text or '  (none)'}"
        )

    @staticmethod
    def _fallback_from_hypothesis(top: dict) -> dict:
        """Build a minimal RCA dict from the hypothesis when Gemini is unavailable."""
        return {
            "root_cause": top.get("hypothesis", _FALLBACK_RCA["root_cause"]),
            "confidence": top.get("confidence", 0.0),
            "causal_chain": [
                e.get("message", "")
                for e in top.get("evidence_lines", [])[:3]
            ],
            "evidence_sources": [],
            "recommended_fix": "Review the evidence lines and investigate the identified component.",
        }

    @staticmethod
    def _ground_evidence(top: dict) -> list[dict]:
        """Extract grounded evidence dicts from the top hypothesis.

        Returns at most ``_MAX_EVIDENCE_SOURCES`` lines, each as a minimal
        dict suitable for API serialisation.
        """
        lines = top.get("evidence_lines") or []
        return [
            {
                "timestamp": e.get("timestamp", ""),
                "level": e.get("level", ""),
                "component": e.get("component", ""),
                "message": e.get("message", ""),
            }
            for e in lines[:_MAX_EVIDENCE_SOURCES]
        ]
