"""Async SQLAlchemy engine and session management.

``create_all_tables()`` is called once at application startup.
``get_session()`` is a FastAPI-compatible dependency that yields an
``AsyncSession`` and commits / rolls back automatically.
"""

import logging
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from config import settings
from storage.models import Base

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engine — one per process, shared across all requests
# ---------------------------------------------------------------------------

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,         # flip to True for SQL debug output
    future=True,
    # aiosqlite requires check_same_thread=False via connect_args
    connect_args={"check_same_thread": False},
)

# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

async_session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


async def create_all_tables() -> None:
    """Create all ORM-mapped tables if they don't already exist.

    Safe to call on every startup — uses ``CREATE TABLE IF NOT EXISTS``
    semantics via SQLAlchemy's ``checkfirst`` behaviour.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables verified / created at %s", settings.DATABASE_URL)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields a scoped ``AsyncSession``.

    Commits on clean exit, rolls back on exception, always closes the session.
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
