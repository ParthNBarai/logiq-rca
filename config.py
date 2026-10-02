"""Application configuration via pydantic-settings.

All values can be overridden via environment variables or a .env file.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central settings object loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    GEMINI_API_KEY: str = ""

    DATABASE_URL: str = "sqlite+aiosqlite:///./logiq.db"
    CHROMA_PERSIST_DIR: str = "./chroma_db"

    # Ingestion tuning
    MAX_CONCURRENT_INGESTION: int = 10
    LOG_CHUNK_SIZE: int = 1000


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached singleton Settings instance."""
    return Settings()


settings: Settings = get_settings()
