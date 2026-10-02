"""LogIQ — FastAPI application entry point.

Startup sequence (via lifespan):
  1. Create / verify all SQLite tables.
  2. Initialise the VectorStore (downloads sentence-transformer model on first run).
  3. Ingest both seed log files if not already ingested.
  4. Build the LangGraph RCA workflow (requires GEMINI_API_KEY).
  5. Attach singletons to app.state for dependency injection in routes.

Run with:
    uvicorn main:app --reload --port 8000

Environment variables (see .env.example):
    GEMINI_API_KEY    — required for /query; optional for ingestion-only use
    DATABASE_URL      — defaults to sqlite+aiosqlite:///./logiq.db
    CHROMA_PERSIST_DIR — defaults to ./chroma_db
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from sqlalchemy import select

from api.routes import router
from config import settings
from ingestion.pipeline import IngestPipeline
from storage.database import async_session_factory, create_all_tables
from storage.models import IngestionRun
from storage.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Seed log files — resolved relative to this file's parent directory
# so the app works regardless of which directory uvicorn is started from.
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent
_SEED_FILES: list[tuple[str, str]] = [
    (str(_PROJECT_ROOT / "auth_rate_limit_noise.log"), "auth_rate_limit_noise"),
    (str(_PROJECT_ROOT / "production_incident_01.log"), "production_incident_01"),
    (str(_PROJECT_ROOT / "Memory Leak GC Pause Log.log"), "memory_leak_gc_pause_log"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _is_successfully_ingested(file_path: str) -> bool:
    """Return True when *file_path* has at least one completed IngestionRun."""
    async with async_session_factory() as session:
        result = await session.execute(
            select(IngestionRun.id)
            .where(
                IngestionRun.file_path == file_path,
                IngestionRun.status == "completed",
            )
            .limit(1)
        )
        return result.scalar_one_or_none() is not None


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan — startup and graceful shutdown."""

    logger.info("LogIQ starting up…")

    # 1. Database
    await create_all_tables()

    # 2. VectorStore (sentence-transformer model loaded here)
    vector_store = VectorStore()
    pipeline = IngestPipeline(vector_store)

    app.state.vector_store = vector_store
    app.state.pipeline = pipeline

    # 3. Seed ingestion
    for file_path, source_name in _SEED_FILES:
        if not Path(file_path).exists():
            logger.warning("Seed log file not found, skipping: %s", file_path)
            continue
        if await _is_successfully_ingested(file_path):
            logger.info("Already ingested %s — skipping.", source_name)
            continue
        logger.info("Ingesting seed file: %s", source_name)
        try:
            result = await pipeline.ingest_file(file_path, source_name)
            logger.info(
                "Seed ingestion complete [%s]: inserted=%d skipped=%d "
                "tenants=%s duration=%.2fs",
                source_name,
                result.records_inserted,
                result.records_skipped,
                sorted(result.tenants_found),
                result.duration_seconds,
            )
        except Exception:
            logger.exception("Failed to ingest seed file %s", file_path)

    # 4. LangGraph RCA workflow (requires GEMINI_API_KEY)
    if not settings.GEMINI_API_KEY:
        logger.warning(
            "GEMINI_API_KEY not set — POST /query will return 503. "
            "Set it in your .env file and restart."
        )
        app.state.graph = None
    else:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            from graph import build_graph

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0,
                google_api_key=settings.GEMINI_API_KEY,
            )
            app.state.graph = build_graph(async_session_factory, vector_store, llm)
            logger.info("LangGraph RCA workflow ready.")
        except Exception:
            logger.exception("Failed to build LangGraph — /query will be unavailable.")
            app.state.graph = None

    logger.info("LogIQ ready. 🟢")

    yield  # ---- application is live ----

    # 5. Shutdown
    logger.info("LogIQ shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LogIQ",
    description=(
        "Enterprise log intelligence and RCA engine. "
        "POST /ingest to load logs, POST /query to run RCA."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)
