"""Pydantic request and response schemas for all API endpoints."""

import re
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# /ingest
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    """Trigger ingestion of a log file at the given path."""

    file_path: str = Field(..., description="Absolute or relative path to the log file.")
    file_source_name: Optional[str] = Field(
        None,
        description=(
            "Label stored on every LogRecord for this file. "
            "Defaults to the filename if omitted."
        ),
    )


class IngestResponse(BaseModel):
    """Summary of a completed ingestion run."""

    total_lines: int
    records_inserted: int
    records_skipped: int
    tenants_found: list[str]
    duration_seconds: float


# ---------------------------------------------------------------------------
# /query
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Natural-language RCA query scoped to a single tenant."""

    query: str = Field(..., min_length=5, description="Natural-language question about the incident.")
    tenant_id: str = Field(
        ...,
        description="Tenant to analyse (e.g. TENANT-X). Must match TENANT-[A-Z].",
    )

    @field_validator("tenant_id")
    @classmethod
    def validate_tenant_id(cls, v: str) -> str:
        if not re.match(r"^TENANT-[A-Z]$", v.strip().upper()):
            raise ValueError(
                "tenant_id must be in the format TENANT-[A-Z] (e.g. TENANT-X)."
            )
        return v.strip().upper()


class EvidenceItem(BaseModel):
    """A single grounded log line included in the final RCA."""

    timestamp: str
    level: str
    component: str
    message: str


class RCAResponse(BaseModel):
    """Full RCA report returned by the /query endpoint."""

    query: str
    tenant_id: str
    root_cause: str
    confidence: float
    causal_chain: list[str]
    evidence_sources: list[EvidenceItem]
    recommended_fix: str
    duration_ms: float
    iteration_count: int


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Extended health-check payload including dependency statuses."""

    status: str
    db: str = "unknown"
    vector_store: str = "unknown"
    version: str = "0.1.0"
