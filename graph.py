"""LangGraph RCA workflow assembly.

Wires five agents into a directed graph:

    planner → retriever → hypothesis → evidence ─┐
                                                  ├── (conditional) → synthesizer → END
                                                  └── (future: loop back to retriever)

The ``should_continue_or_end`` routing function is the explicit guard against
the infinite-loop problem: ``max_iterations = 3`` is set in the initial state
and enforced here regardless of hypothesis quality.
"""

import logging
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agents.evidence import EvidenceAgent
from agents.hypothesis import HypothesisAgent
from agents.planner import PlannerAgent
from agents.retriever import RetrieverAgent
from agents.state import RCAState
from agents.synthesizer import SynthesizerAgent
from storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def should_continue_or_end(state: RCAState) -> str:
    """Decide whether to synthesise or loop back for broader retrieval.

    Guards:
      1. Hard cap: if iteration_count >= max_iterations → always synthesise.
      2. High confidence: top hypothesis confidence > 0.6 → synthesise.
      3. Otherwise: synthesise (placeholder — future: return "retriever" to
         widen the time window and retry with lower confidence threshold).

    The guard at (1) is the explicit solution to the infinite-loop bug: even
    if no hypothesis reaches 0.6 confidence, we still terminate after 3 passes.
    """
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 3)

    if iteration >= max_iter:
        logger.info(
            "[Graph] Max iterations (%d) reached — forcing synthesise.", max_iter
        )
        return "synthesize"

    validated = state.get("validated_hypotheses") or []
    if not validated:
        return "synthesize"

    top_confidence = max(
        (h.get("confidence", 0.0) for h in validated), default=0.0
    )

    if top_confidence > 0.6:
        logger.info(
            "[Graph] Top hypothesis confidence=%.2f > 0.6 — synthesising.", top_confidence
        )
        return "synthesize"

    # Future enhancement: return "retriever" here to widen the time window
    # and re-run retrieval when confidence is insufficient.
    logger.info(
        "[Graph] Low confidence (%.2f) but no retry logic yet — synthesising.", top_confidence
    )
    return "synthesize"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph(
    session_factory: async_sessionmaker[AsyncSession],
    vector_store: VectorStore,
    llm: ChatGoogleGenerativeAI,
) -> Any:
    """Instantiate agents, build the LangGraph, and return a compiled graph.

    All dependencies are injected here so the node closures capture them.
    The returned compiled graph is safe to share across async requests.

    Args:
        session_factory: Async SQLAlchemy session factory for the retriever.
        vector_store:    Shared ``VectorStore`` instance for semantic search.
        llm:             Gemini LLM instance used by planner, hypothesis,
                         and synthesizer agents.

    Returns:
        A compiled ``StateGraph`` ready for ``await graph.ainvoke(state)``.
    """
    # Instantiate agents (LLM is shared — loading once is sufficient)
    planner = PlannerAgent(llm)
    retriever = RetrieverAgent(session_factory, vector_store)
    hypothesis_agent = HypothesisAgent(llm)
    evidence_agent = EvidenceAgent()
    synthesizer = SynthesizerAgent(llm)

    # ------------------------------------------------------------------
    # Node wrappers — thin async closures that delegate to agent.run()
    # ------------------------------------------------------------------

    async def planner_node(state: RCAState) -> dict:
        """Parse query into structured search parameters."""
        return await planner.run(state)

    async def retriever_node(state: RCAState) -> dict:
        """Execute hybrid SQL + semantic retrieval and merge evidence."""
        return await retriever.run(state)

    async def hypothesis_node(state: RCAState) -> dict:
        """Generate ranked root-cause hypotheses from merged evidence."""
        return await hypothesis_agent.run(state)

    async def evidence_node(state: RCAState) -> dict:
        """Validate hypotheses against evidence; increment iteration counter."""
        result = await evidence_agent.run(state)
        # Increment here — this is the decision point for the routing function
        result["iteration_count"] = state.get("iteration_count", 0) + 1
        return result

    async def synthesizer_node(state: RCAState) -> dict:
        """Produce the final grounded RCA narrative."""
        return await synthesizer.run(state)

    # ------------------------------------------------------------------
    # Graph assembly
    # ------------------------------------------------------------------

    builder: StateGraph = StateGraph(RCAState)

    builder.add_node("planner", planner_node)
    builder.add_node("retriever", retriever_node)
    builder.add_node("hypothesis", hypothesis_node)
    builder.add_node("evidence", evidence_node)
    builder.add_node("synthesizer", synthesizer_node)

    builder.set_entry_point("planner")
    builder.add_edge("planner", "retriever")
    builder.add_edge("retriever", "hypothesis")
    builder.add_edge("hypothesis", "evidence")
    builder.add_conditional_edges(
        "evidence",
        should_continue_or_end,
        {"synthesize": "synthesizer"},
    )
    builder.add_edge("synthesizer", END)

    compiled = builder.compile()
    logger.info("LangGraph RCA workflow compiled successfully.")
    return compiled
