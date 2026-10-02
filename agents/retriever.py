"""Retriever agent — hybrid SQLite + ChromaDB evidence gathering.

Three-phase execution:
  Phase 1 — Structural (SQLite, deterministic):
    Exact-match filters on tenant, time window, log level, keywords.
    Returns chronologically ordered records. Solves Scenario 1 by pulling
    the low-frequency DB pool warnings that precede the 503 flood.

  Phase 2 — Semantic (ChromaDB, per-tenant collection):
    Embedding-based nearest-neighbour search on the NL query.
    Surfaces related events that don't share keywords with the query.
    Tenant isolation is absolute — cross-collection search never happens.

  Phase 3 — Merge + RCA Ranking:
    Deduplicates by ingestion_id (structural wins over semantic for same id).
    Applies RCA-specific scoring: exceptions and state transitions score
    highest; high-frequency repetitive messages are demoted to prevent the
    rate-limit flood (Scenario 2) from burying the real root cause.
    Keeps top 50.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agents.state import RCAState
from storage.models import LogRecord
from storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

_STRUCTURAL_LIMIT = 200
_SEMANTIC_LIMIT = 20
_MERGE_TOP_N = 50
_REPETITION_THRESHOLD = 10
_NULL_INT_SENTINEL = -1


class RetrieverAgent:
    """Executes hybrid retrieval and returns ranked merged evidence.

    Args:
        session_factory: Async SQLAlchemy session factory.
        vector_store:    Shared ``VectorStore`` instance.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        vector_store: VectorStore,
    ) -> None:
        self._session_factory = session_factory
        self._vector_store = vector_store

    async def run(self, state: RCAState) -> dict[str, Any]:
        """Execute all three retrieval phases and return evidence state updates.

        Returns a partial state update with ``structural_hits``,
        ``semantic_hits``, and ``merged_evidence``.
        """
        tenant_id = state["tenant_id"]
        logger.info("[Retriever] Starting hybrid retrieval for tenant %s", tenant_id)

        # Phases 1 & 2 run concurrently — neither depends on the other
        structural, semantic = await asyncio.gather(
            self._structural_search(state),
            self._semantic_search(state),
        )

        logger.info(
            "[Retriever] structural=%d semantic=%d", len(structural), len(semantic)
        )

        merged = self._merge_and_rank(structural, semantic)
        logger.info("[Retriever] merged_evidence=%d (top %d)", len(merged), _MERGE_TOP_N)

        return {
            "structural_hits": structural,
            "semantic_hits": semantic,
            "merged_evidence": merged,
        }

    # -----------------------------------------------------------------------
    # Phase 1 — Structural (SQLite)
    # -----------------------------------------------------------------------

    async def _structural_search(self, state: RCAState) -> list[dict]:
        """Build and execute a deterministic SQLite query from planner output.

        The tenant_id filter is ALWAYS the first condition and is never optional.
        """
        try:
            async with self._session_factory() as session:
                filters = [
                    LogRecord.tenant_id == state["tenant_id"]  # ← always first
                ]

                tw_start: Optional[datetime] = state.get("time_window_start")
                tw_end: Optional[datetime] = state.get("time_window_end")

                if tw_start:
                    filters.append(LogRecord.timestamp >= tw_start)
                if tw_end:
                    filters.append(LogRecord.timestamp <= tw_end)

                error_levels = state.get("error_levels") or []
                if error_levels:
                    filters.append(LogRecord.level.in_(error_levels))

                keywords = state.get("search_keywords") or []
                if keywords:
                    kw_filters = [
                        LogRecord.message.ilike(f"%{kw}%") for kw in keywords
                    ]
                    filters.append(or_(*kw_filters))

                result = await session.execute(
                    select(LogRecord)
                    .where(and_(*filters))
                    .order_by(LogRecord.timestamp.asc())  # chronological order is critical
                    .limit(_STRUCTURAL_LIMIT)
                )
                records = result.scalars().all()
                return [self._record_to_evidence(r, "structural") for r in records]

        except Exception:
            logger.warning("[Retriever] Structural search failed.", exc_info=True)
            return []

    # -----------------------------------------------------------------------
    # Phase 2 — Semantic (ChromaDB)
    # -----------------------------------------------------------------------

    async def _semantic_search(self, state: RCAState) -> list[dict]:
        """Run embedding-based search scoped strictly to the tenant's collection."""
        try:
            hits = await asyncio.to_thread(
                self._vector_store.search,
                state["tenant_id"],
                state["query"],
                _SEMANTIC_LIMIT,
            )
            return [self._semantic_hit_to_evidence(h) for h in hits]
        except Exception:
            logger.warning("[Retriever] Semantic search failed.", exc_info=True)
            return []

    # -----------------------------------------------------------------------
    # Phase 3 — Merge + RCA ranking
    # -----------------------------------------------------------------------

    def _merge_and_rank(
        self, structural: list[dict], semantic: list[dict]
    ) -> list[dict]:
        """Deduplicate by ingestion_id, score for RCA relevance, return top N.

        Structural hits take precedence over semantic hits for the same
        ingestion_id because they carry the full SQLAlchemy field set.
        """
        # Deduplicate — structural first so it wins over semantic
        seen: dict[str, dict] = {}
        for r in structural:
            seen[r["ingestion_id"]] = r
        for r in semantic:
            if r["ingestion_id"] not in seen:
                seen[r["ingestion_id"]] = r

        all_records = list(seen.values())

        # Count message frequency for repetition-penalty detection
        msg_count: dict[str, int] = {}
        for r in all_records:
            msg = r.get("message", "")
            msg_count[msg] = msg_count.get(msg, 0) + 1

        # Score each record
        scored = [self._score(r, msg_count) for r in all_records]

        # Sort: score DESC, then timestamp ASC for ties (chronological among equals)
        scored.sort(key=lambda x: (-x["_score"], x.get("timestamp", "")))

        return scored[:_MERGE_TOP_N]

    @staticmethod
    def _score(record: dict, msg_count: dict[str, int]) -> dict:
        """Compute RCA relevance score and return record with ``_score`` set."""
        base = record.get("_score", 50.0)

        # High-value RCA signals
        if record.get("exception_type"):
            base += 30   # exceptions are almost always root-cause proximate
        if record.get("circuit_breaker_state"):
            base += 20   # state transitions mark trigger points
        if record.get("level") == "ERROR":
            base += 20
        elif record.get("level") == "WARN":
            base += 10
        elif record.get("level") == "INFO" and not record.get("is_error"):
            base -= 10   # demote routine INFO noise

        # Repetition penalty — prevents rate-limit flood from dominating
        if msg_count.get(record.get("message", ""), 0) > _REPETITION_THRESHOLD:
            base -= 5

        return {**record, "_score": base}

    # -----------------------------------------------------------------------
    # Evidence dict converters
    # -----------------------------------------------------------------------

    @staticmethod
    def _record_to_evidence(record: LogRecord, source: str) -> dict:
        """Convert a SQLAlchemy ``LogRecord`` to a canonical evidence dict."""
        return {
            "ingestion_id": record.ingestion_id,
            "tenant_id": record.tenant_id,
            "timestamp": record.timestamp.isoformat(),
            "timestamp_unix": int(record.timestamp.timestamp()),
            "level": record.level,
            "component": record.component,
            "logger_class": record.logger_class,
            "message": record.message,
            "is_error": record.is_error,
            "http_status": record.http_status,
            "latency_ms": record.latency_ms,
            "db_cluster": record.db_cluster,
            "circuit_breaker_state": record.circuit_breaker_state,
            "exception_type": record.exception_type,
            "file_source": record.file_source,
            "raw_line": record.raw_line,
            "source": source,
            "_score": 50.0,  # base; _score() will mutate this during ranking
        }

    @staticmethod
    def _semantic_hit_to_evidence(hit: dict) -> dict:
        """Convert a ChromaDB search result to a canonical evidence dict.

        Cosine distance (0 = identical, 1 = orthogonal) is inverted to a
        similarity score so high-similarity hits start with a higher base score.
        """
        meta = hit.get("metadata") or {}
        distance = hit.get("distance", 1.0)
        ts_unix = meta.get("timestamp_unix", 0)

        # Convert sentinels back to None
        http_status: Optional[int] = meta.get("http_status")
        if http_status == _NULL_INT_SENTINEL:
            http_status = None
        latency_ms: Optional[int] = meta.get("latency_ms")
        if latency_ms == _NULL_INT_SENTINEL:
            latency_ms = None

        ts = datetime.utcfromtimestamp(ts_unix) if ts_unix else datetime.utcnow()

        return {
            "ingestion_id": hit.get("ingestion_id", ""),
            "tenant_id": meta.get("tenant_id", ""),
            "timestamp": ts.isoformat(),
            "timestamp_unix": ts_unix,
            "level": meta.get("level", "INFO"),
            "component": meta.get("component", ""),
            "logger_class": meta.get("logger_class", ""),
            "message": meta.get("message", hit.get("document", "")),
            "is_error": bool(meta.get("is_error", False)),
            "http_status": http_status,
            "latency_ms": latency_ms,
            "db_cluster": meta.get("db_cluster") or None,
            "circuit_breaker_state": meta.get("circuit_breaker_state") or None,
            "exception_type": meta.get("exception_type") or None,
            "file_source": meta.get("file_source", ""),
            "raw_line": hit.get("document", ""),
            "source": "semantic",
            # Convert distance → similarity score: closer = higher base
            "_score": max(0.0, 100.0 * (1.0 - distance)),
        }
