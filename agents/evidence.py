"""Evidence agent — deterministic hypothesis validation (zero LLM calls).

For each hypothesis produced by the Hypothesis agent, this node:
  1. Looks up every ``supporting_log_id`` in the ``merged_evidence`` index.
  2. Computes ``evidence_strength`` = found_citations / claimed_citations.
  3. Assigns a verdict: SUPPORTED / PARTIAL / UNSUPPORTED.
  4. Attaches the actual evidence dicts (sorted chronologically) so the
     Synthesizer has grounded material to work with without re-querying.

No LLM is involved. This node cannot be influenced by hallucinated log ids
because it only accepts lines that genuinely exist in ``merged_evidence``.
The iteration_count is incremented by the graph node wrapper.
"""

import logging
from typing import Any

from agents.state import RCAState

logger = logging.getLogger(__name__)

_SUPPORTED_THRESHOLD = 0.6
_PARTIAL_THRESHOLD = 0.3


class EvidenceAgent:
    """Validates hypotheses against retrieved evidence using deterministic logic."""

    async def run(self, state: RCAState) -> dict[str, Any]:
        """Cross-check each hypothesis against ``merged_evidence``.

        Returns a partial state update with ``validated_hypotheses``,
        sorted by confidence descending so the synthesizer always receives
        the strongest hypothesis first.
        """
        hypotheses = state.get("hypotheses") or []
        merged = state.get("merged_evidence") or []

        # Build O(1) lookup index from ingestion_id → evidence dict
        evidence_index: dict[str, dict] = {
            e["ingestion_id"]: e for e in merged if e.get("ingestion_id")
        }

        validated: list[dict] = []
        for hyp in hypotheses:
            validated.append(
                self._validate_one(hyp, evidence_index)
            )

        # Sort descending by confidence so synthesizer always picks [0] as top
        validated.sort(key=lambda h: h["confidence"], reverse=True)

        logger.info(
            "[Evidence] Validated %d hypotheses. Top confidence=%.2f verdict=%s",
            len(validated),
            validated[0]["confidence"] if validated else 0.0,
            validated[0]["verdict"] if validated else "N/A",
        )

        return {"validated_hypotheses": validated}

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _validate_one(
        self, hypothesis: dict, evidence_index: dict[str, dict]
    ) -> dict:
        """Validate a single hypothesis dict and return the enriched result."""
        claimed_ids: list[str] = hypothesis.get("supporting_log_ids") or []

        found_evidence: list[dict] = [
            evidence_index[id_]
            for id_ in claimed_ids
            if id_ in evidence_index
        ]

        # Sort found evidence chronologically — this becomes the causal timeline
        found_evidence.sort(key=lambda e: e.get("timestamp", ""))

        evidence_strength = (
            len(found_evidence) / len(claimed_ids) if claimed_ids else 0.0
        )

        verdict = self._compute_verdict(evidence_strength)

        # Adjust confidence by evidence strength — penalise uncorroborated claims
        raw_confidence: float = hypothesis.get("confidence", 0.0)
        adjusted_confidence = raw_confidence * evidence_strength if claimed_ids else raw_confidence * 0.5

        logger.debug(
            "[Evidence] hyp=%r claimed=%d found=%d strength=%.2f verdict=%s",
            hypothesis.get("hypothesis", "")[:60],
            len(claimed_ids),
            len(found_evidence),
            evidence_strength,
            verdict,
        )

        return {
            "hypothesis": hypothesis.get("hypothesis", ""),
            "confidence": round(adjusted_confidence, 3),
            "supporting_log_ids": claimed_ids,
            "evidence_lines": found_evidence,   # actual dicts — used for grounding
            "evidence_strength": round(evidence_strength, 3),
            "verdict": verdict,
        }

    @staticmethod
    def _compute_verdict(strength: float) -> str:
        if strength >= _SUPPORTED_THRESHOLD:
            return "SUPPORTED"
        if strength >= _PARTIAL_THRESHOLD:
            return "PARTIAL"
        return "UNSUPPORTED"
