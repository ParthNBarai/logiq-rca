"""Async ingestion pipeline: file → parse → normalise → SQLite + ChromaDB.

Design decisions:
  - Reads the file in chunks (``LOG_CHUNK_SIZE`` lines) to bound memory usage.
  - Each line in a chunk is parsed concurrently via ``asyncio.gather`` with a
    semaphore capped at ``MAX_CONCURRENT_INGESTION`` to prevent thread-pool
    saturation.
  - SQLite inserts use ``INSERT OR IGNORE`` (on_conflict_do_nothing) so the
    pipeline is fully idempotent — re-ingesting a file skips existing rows.
  - ChromaDB upserts are also idempotent by design (document id = ingestion_id).
  - Every run is recorded in ``IngestionRun`` for audit and restart-tracking.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiofiles
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from config import settings
from ingestion.normalizer import LogNormalizer
from ingestion.parser import BaseLogParser, LogParserFactory, ParsedLogLine
from storage.database import async_session_factory
from storage.models import IngestionRun, LogRecord
from storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class IngestionResult:
    """Summary returned after a completed ingestion run."""

    total_lines: int
    records_inserted: int
    records_skipped: int
    tenants_found: set[str]
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        """Serialisable dict for API responses and logging."""
        return {
            "total_lines": self.total_lines,
            "records_inserted": self.records_inserted,
            "records_skipped": self.records_skipped,
            "tenants_found": sorted(self.tenants_found),
            "duration_seconds": round(self.duration_seconds, 3),
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class IngestPipeline:
    """Orchestrates end-to-end ingestion of a log file.

    Args:
        vector_store: Shared ``VectorStore`` instance (model loaded once).
    """

    def __init__(self, vector_store: VectorStore) -> None:
        self._vector_store = vector_store
        self._normalizer = LogNormalizer()
        self._semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_INGESTION)

    # -----------------------------------------------------------------------
    # Public
    # -----------------------------------------------------------------------

    async def ingest_file(
        self, file_path: str, file_source_name: str
    ) -> IngestionResult:
        """Ingest *file_path* into SQLite and ChromaDB.

        Creates an ``IngestionRun`` audit record, processes the file in chunks,
        and marks the run ``completed`` or ``failed`` on exit.

        Args:
            file_path:        Absolute or relative path to the log file.
            file_source_name: Human-readable label stored on every ``LogRecord``
                              (typically the filename without path).

        Returns:
            ``IngestionResult`` with counts and tenant list.

        Raises:
            FileNotFoundError: When *file_path* does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Log file not found: {file_path}")

        start = time.monotonic()
        run_id = await self._create_run(file_path)

        try:
            result = await self._process_file(path, file_source_name)
            await self._finish_run(run_id, result, "completed")
        except Exception:
            logger.exception("Ingestion failed for %s", file_path)
            await self._fail_run(run_id)
            raise

        result.duration_seconds = time.monotonic() - start
        logger.info(
            "Ingestion complete: %s — %s",
            file_source_name,
            result.to_dict(),
        )
        return result

    # -----------------------------------------------------------------------
    # File-level processing
    # -----------------------------------------------------------------------

    async def _process_file(
        self, path: Path, file_source_name: str
    ) -> IngestionResult:
        """Stream *path* in chunks and process each chunk."""
        total_lines = 0
        records_inserted = 0
        records_skipped = 0
        tenants_found: set[str] = set()
        parser: Optional[BaseLogParser] = None
        chunk: list[str] = []

        async with aiofiles.open(path, "r", encoding="utf-8", errors="replace") as fh:
            async for raw_line in fh:
                total_lines += 1
                stripped = raw_line.strip()
                if not stripped:
                    continue

                # Detect format from the first parseable line
                if parser is None:
                    try:
                        parser = LogParserFactory.get_parser(stripped)
                        logger.info(
                            "Detected parser %s for %s",
                            type(parser).__name__,
                            path.name,
                        )
                    except ValueError:
                        logger.warning(
                            "Could not detect parser on line %d; skipping.", total_lines
                        )
                        continue

                chunk.append(raw_line)

                if len(chunk) >= settings.LOG_CHUNK_SIZE:
                    ins, skp, tnts = await self._process_chunk(
                        chunk, parser, file_source_name
                    )
                    records_inserted += ins
                    records_skipped += skp
                    tenants_found.update(tnts)
                    chunk = []

            # Tail chunk (fewer than LOG_CHUNK_SIZE lines)
            if chunk and parser:
                ins, skp, tnts = await self._process_chunk(
                    chunk, parser, file_source_name
                )
                records_inserted += ins
                records_skipped += skp
                tenants_found.update(tnts)

        return IngestionResult(
            total_lines=total_lines,
            records_inserted=records_inserted,
            records_skipped=records_skipped,
            tenants_found=tenants_found,
        )

    # -----------------------------------------------------------------------
    # Chunk-level processing
    # -----------------------------------------------------------------------

    async def _process_chunk(
        self,
        lines: list[str],
        parser: BaseLogParser,
        file_source_name: str,
    ) -> tuple[int, int, set[str]]:
        """Parse, normalise, and store one chunk of raw lines.

        Returns:
            Tuple of (inserted_count, skipped_count, tenant_ids_in_chunk).
        """
        # -- concurrent parse with bounded semaphore -------------------------
        async def parse_one(line: str) -> Optional[ParsedLogLine]:
            async with self._semaphore:
                return await asyncio.to_thread(parser.parse, line)

        raw_parsed: list[Optional[ParsedLogLine]] = await asyncio.gather(
            *[parse_one(line) for line in lines]
        )

        # Filter parse failures, then normalise
        parsed = self._normalizer.normalize_batch(
            [p for p in raw_parsed if p is not None]
        )

        if not parsed:
            return 0, len(lines), set()

        # -- build DB row dicts ----------------------------------------------
        records_data = [self._to_db_dict(p, file_source_name) for p in parsed]

        # -- idempotent SQLite bulk insert -----------------------------------
        inserted, skipped = await self._bulk_insert(records_data)

        # -- ChromaDB upsert per tenant -------------------------------------
        tenant_map: dict[str, list[ParsedLogLine]] = {}
        for p in parsed:
            tenant_map.setdefault(p.tenant_id, []).append(p)

        for tenant_id, tenant_parsed in tenant_map.items():
            tenant_records = [
                self._parsed_to_log_record(p, file_source_name)
                for p in tenant_parsed
            ]
            await asyncio.to_thread(
                self._vector_store.upsert_logs, tenant_id, tenant_records
            )

        return inserted, skipped, set(tenant_map.keys())

    # -----------------------------------------------------------------------
    # SQLite helpers
    # -----------------------------------------------------------------------

    async def _bulk_insert(self, records_data: list[dict]) -> tuple[int, int]:
        """Insert records, silently skipping those with duplicate ingestion_id.

        Uses SQLite's ``INSERT OR IGNORE`` semantics via
        ``on_conflict_do_nothing`` so the operation is always idempotent.
        Returns (inserted_count, skipped_count).
        """
        if not records_data:
            return 0, 0

        async with async_session_factory() as session:
            # Pre-check which ingestion_ids already exist — rowcount is
            # unreliable for OR IGNORE in some SQLite/aiosqlite versions.
            incoming_ids = [r["ingestion_id"] for r in records_data]
            existing_result = await session.execute(
                select(LogRecord.ingestion_id).where(
                    LogRecord.ingestion_id.in_(incoming_ids)
                )
            )
            existing_ids: set[str] = set(existing_result.scalars().all())

            new_records = [
                r for r in records_data if r["ingestion_id"] not in existing_ids
            ]
            skipped = len(records_data) - len(new_records)

            if new_records:
                await session.execute(
                    sqlite_insert(LogRecord).values(new_records).on_conflict_do_nothing(
                        index_elements=["ingestion_id"]
                    )
                )
                await session.commit()

            logger.debug(
                "Chunk insert: %d new, %d skipped.", len(new_records), skipped
            )
            return len(new_records), skipped

    # -----------------------------------------------------------------------
    # IngestionRun audit helpers
    # -----------------------------------------------------------------------

    async def _create_run(self, file_path: str) -> int:
        """Insert a new IngestionRun row and return its PK."""
        async with async_session_factory() as session:
            run = IngestionRun(
                file_path=file_path,
                started_at=datetime.now(timezone.utc).replace(tzinfo=None),
                status="running",
            )
            session.add(run)
            await session.commit()
            await session.refresh(run)
            return run.id

    async def _finish_run(
        self, run_id: int, result: IngestionResult, status: str
    ) -> None:
        """Mark the IngestionRun as completed with final counts."""
        async with async_session_factory() as session:
            run = await session.get(IngestionRun, run_id)
            if run:
                run.completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
                run.records_processed = result.records_inserted
                run.records_skipped = result.records_skipped
                run.status = status
                await session.commit()

    async def _fail_run(self, run_id: int) -> None:
        """Mark the IngestionRun as failed."""
        async with async_session_factory() as session:
            run = await session.get(IngestionRun, run_id)
            if run:
                run.completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
                run.status = "failed"
                await session.commit()

    # -----------------------------------------------------------------------
    # Conversion helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _to_db_dict(p: ParsedLogLine, file_source: str) -> dict:
        """Convert a ``ParsedLogLine`` to a flat dict for SQLAlchemy bulk insert."""
        return {
            "tenant_id": p.tenant_id,
            # Strip timezone info — SQLite stores as naive UTC strings
            "timestamp": p.timestamp.replace(tzinfo=None),
            "level": p.level,
            "logger_class": p.logger_class,
            "component": p.component,
            "message": p.message,
            "is_error": p.is_error,
            "http_status": p.http_status,
            "latency_ms": p.latency_ms,
            "db_cluster": p.db_cluster,
            "circuit_breaker_state": p.circuit_breaker_state,
            "exception_type": p.exception_type,
            "raw_line": p.raw_line,
            "file_source": file_source,
            "ingestion_id": p.ingestion_id,
        }

    @staticmethod
    def _parsed_to_log_record(p: ParsedLogLine, file_source: str) -> LogRecord:
        """Build an unsaved ``LogRecord`` ORM instance from a parsed line.

        Used solely to pass typed objects to ``VectorStore.upsert_logs``.
        """
        return LogRecord(
            tenant_id=p.tenant_id,
            timestamp=p.timestamp.replace(tzinfo=None),
            level=p.level,
            logger_class=p.logger_class,
            component=p.component,
            message=p.message,
            is_error=p.is_error,
            http_status=p.http_status,
            latency_ms=p.latency_ms,
            db_cluster=p.db_cluster,
            circuit_breaker_state=p.circuit_breaker_state,
            exception_type=p.exception_type,
            raw_line=p.raw_line,
            file_source=file_source,
            ingestion_id=p.ingestion_id,
        )
