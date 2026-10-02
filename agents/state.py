"""LangGraph shared state for the RCA workflow.

``RCAState`` is the single mutable context that flows through every node.
Each agent reads from and writes to this TypedDict; LangGraph merges
partial-dict returns so nodes only need to include keys they change.
"""

from datetime import datetime
from typing import Optional, TypedDict


class RCAState(TypedDict):
    """Full state schema for the five-node RCA graph.

    Lifecycle of fields:
      startup   → query, tenant_id, iteration_count, max_iterations, error
      planner   → time_window_start, time_window_end, search_keywords,
                  error_levels, components_of_interest
      retriever → structural_hits, semantic_hits, merged_evidence
      hypothesis→ hypotheses
      evidence  → validated_hypotheses, iteration_count (incremented)
      synthesizer→ rca_result
    """

    # -- Input ----------------------------------------------------------------
    query: str
    tenant_id: str

    # -- Planner output -------------------------------------------------------
    time_window_start: Optional[datetime]
    time_window_end: Optional[datetime]
    search_keywords: list[str]
    error_levels: list[str]           # e.g. ["ERROR", "WARN"]
    components_of_interest: list[str] # e.g. ["DataSource", "Breaker"]

    # -- Retrieval output -----------------------------------------------------
    structural_hits: list[dict]   # from SQLite (deterministic, exact filters)
    semantic_hits: list[dict]     # from ChromaDB (embedding similarity)
    merged_evidence: list[dict]   # deduped + ranked, max 50

    # -- Hypothesis output ----------------------------------------------------
    hypotheses: list[dict]        # [{hypothesis, confidence, supporting_log_ids}]

    # -- Evidence validation output -------------------------------------------
    validated_hypotheses: list[dict]  # [{hypothesis, confidence, evidence_lines,
                                      #   evidence_strength, verdict}]

    # -- Final output ---------------------------------------------------------
    rca_result: dict  # {root_cause, confidence, causal_chain,
                      #  evidence_sources, recommended_fix, tenant_id}

    # -- Control --------------------------------------------------------------
    iteration_count: int
    max_iterations: int   # hard cap = 3, prevents infinite loops
    error: Optional[str]
