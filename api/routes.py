"""FastAPI route handlers for ingestion, RCA query, and health check.

All heavy objects (pipeline, vector_store, graph) live on ``app.state``
and are resolved via ``request.app.state`` — constructed once at startup,
never per-request, so the sentence-transformer model and Gemini client
are initialised exactly once.
"""

import logging
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, status
from sqlalchemy import text

from api.schemas import (
    EvidenceItem,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    RCAResponse,
)
from agents.state import RCAState
from ingestion.pipeline import IngestPipeline
from storage.database import async_session_factory
from storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse, tags=["ops"])
async def health(request: Request) -> HealthResponse:
    """Liveness + readiness probe.

    Checks:
      - SQLite connectivity (SELECT 1)
      - ChromaDB / VectorStore availability
    """
    # DB check
    try:
        async with async_session_factory() as session:
            await session.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as exc:
        logger.error("Health: DB check failed: %s", exc)
        db_status = f"error: {exc}"

    # Vector store check
    try:
        vs: VectorStore = request.app.state.vector_store
        tenant_count = len(vs.list_tenants())
        vs_status = f"ready ({tenant_count} tenant collections)"
    except Exception as exc:
        logger.error("Health: VectorStore check failed: %s", exc)
        vs_status = f"error: {exc}"

    return HealthResponse(status="ok", db=db_status, vector_store=vs_status)


# ---------------------------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------------------------


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    tags=["ingestion"],
)
async def ingest(request: Request, body: IngestRequest) -> IngestResponse:
    """Ingest a log file into SQLite and the ChromaDB vector store.

    The operation is idempotent — re-ingesting the same file skips already-stored
    records and returns the skip count.  Use this endpoint to add new log files
    at runtime without restarting the server.
    """
    pipeline: IngestPipeline = request.app.state.pipeline
    source_name = body.file_source_name or Path(body.file_path).name

    try:
        result = await pipeline.ingest_file(body.file_path, source_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        logger.exception("Ingestion error for %s", body.file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        )

    return IngestResponse(**result.to_dict())


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------


@router.post(
    "/query",
    response_model=RCAResponse,
    status_code=status.HTTP_200_OK,
    tags=["rca"],
)
async def query(request: Request, body: QueryRequest) -> RCAResponse:
    """Run a natural-language RCA query and return a full incident analysis.

    The graph executes five agents in sequence:
      planner → retriever → hypothesis → evidence → synthesizer

    ``tenant_id`` is validated to match ``TENANT-[A-Z]`` before the graph is
    invoked — the tenant filter is the first condition on every DB query.
    Execution time is measured end-to-end and returned in ``duration_ms``.
    """
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "GEMINI_API_KEY is not configured. "
                "Set it in your .env file and restart the server."
            ),
        )

    initial_state: RCAState = {
        "query": body.query,
        "tenant_id": body.tenant_id,
        # Planner will populate these
        "time_window_start": None,
        "time_window_end": None,
        "search_keywords": [],
        "error_levels": ["ERROR", "WARN"],
        "components_of_interest": [],
        # Retriever will populate these
        "structural_hits": [],
        "semantic_hits": [],
        "merged_evidence": [],
        # Hypothesis agent will populate these
        "hypotheses": [],
        # Evidence agent will populate these
        "validated_hypotheses": [],
        # Synthesizer will populate this
        "rca_result": {},
        # Loop control — max_iterations = 3 prevents infinite loops
        "iteration_count": 0,
        "max_iterations": 3,
        "error": None,
    }

    t0 = time.monotonic()
    try:
        final_state: RCAState = await graph.ainvoke(initial_state)
    except Exception as exc:
        logger.exception("Graph execution failed for tenant %s", body.tenant_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RCA graph execution failed: {exc}",
        )
    duration_ms = (time.monotonic() - t0) * 1000

    rca = final_state.get("rca_result") or {}

    # Coerce evidence_sources into EvidenceItem objects for the response model
    evidence_sources = [
        EvidenceItem(**e)
        for e in rca.get("evidence_sources", [])
        if isinstance(e, dict)
    ]

    return RCAResponse(
        query=body.query,
        tenant_id=body.tenant_id,
        root_cause=rca.get("root_cause", "Unable to determine root cause."),
        confidence=float(rca.get("confidence", 0.0)),
        causal_chain=rca.get("causal_chain", []),
        evidence_sources=evidence_sources,
        recommended_fix=rca.get("recommended_fix", ""),
        duration_ms=round(duration_ms, 1),
        iteration_count=final_state.get("iteration_count", 0),
    )
