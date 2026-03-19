# src/config.py
"""Application configuration via pydantic-settings.

All settings are read from environment variables or a .env file.
No credentials are hardcoded.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "get_settings"]


class Settings(BaseSettings):
    """Application settings read from environment / .env file.

    Attributes:
        app_name: Display name for the API.
        debug: Enable SQLAlchemy echo and verbose logging.
        database_url: Async PostgreSQL connection URL.
        api_prefix: URL prefix for all routes.
        default_n_simulations: Default Monte Carlo iterations.
        quality_threshold: Default passing quality score.
        log_level: Python logging level string.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Monte Carlo DQ"
    debug: bool = False

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://dq_user:dq_pass@localhost:5432/dq_db",
        description="Async PostgreSQL DSN",
    )

    # API
    api_prefix: str = "/api/v1"

    # Simulation defaults
    default_n_simulations: int = Field(default=10_000, ge=100)
    quality_threshold: float = Field(default=0.8, ge=0.0, le=1.0)

    # Logging
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings singleton.

    Returns:
        Fully initialised Settings object.
    """
    return Settings()
