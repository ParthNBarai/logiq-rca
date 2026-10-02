"""SQLAlchemy ORM models for persistent log storage.

Two tables:
  - ``LogRecord``      — one row per parsed log line, idempotency-keyed by SHA-256
  - ``IngestionRun``   — audit trail of every file ingestion attempt
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


class LogRecord(Base):
    """Parsed and normalised log line persisted to SQLite.

    ``ingestion_id`` (SHA-256 of the raw line) carries a UNIQUE constraint so
    re-ingesting the same file is a no-op at the database level.
    """

    __tablename__ = "log_records"
    __table_args__ = (
        UniqueConstraint("ingestion_id", name="uq_log_records_ingestion_id"),
        # Composite indexes for the most common query patterns
        Index("ix_log_records_tenant_ts", "tenant_id", "timestamp"),
        Index("ix_log_records_tenant_level", "tenant_id", "level"),
        Index("ix_log_records_tenant_error", "tenant_id", "is_error"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Core identity fields
    tenant_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    level: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    logger_class: Mapped[str] = mapped_column(String, nullable=False)
    component: Mapped[str] = mapped_column(String, nullable=False, index=True)
    message: Mapped[str] = mapped_column(String, nullable=False)
    is_error: Mapped[bool] = mapped_column(Boolean, nullable=False, index=True)

    # Optional extracted sub-fields
    http_status: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    db_cluster: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    circuit_breaker_state: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    exception_type: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # Provenance
    raw_line: Mapped[str] = mapped_column(String, nullable=False)
    file_source: Mapped[str] = mapped_column(String, nullable=False)
    ingestion_id: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True
    )

    def __repr__(self) -> str:
        return (
            f"<LogRecord id={self.id} tenant={self.tenant_id} "
            f"level={self.level} component={self.component}>"
        )


class IngestionRun(Base):
    """Audit record for a single file ingestion attempt.

    ``status`` is one of: ``running`` | ``completed`` | ``failed``.
    """

    __tablename__ = "ingestion_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    records_processed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    records_skipped: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="running")

    def __repr__(self) -> str:
        return (
            f"<IngestionRun id={self.id} file={self.file_path!r} status={self.status}>"
        )
