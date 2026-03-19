# src/api/main.py
"""FastAPI application factory and startup logic."""

from __future__ import annotations

import logging
import logging.config
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src import __version__
from src.api.routes.health import router as health_router
from src.api.routes.quality import router as quality_router
from src.config import get_settings

__all__ = ["create_app", "app"]

logger = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    """Set up structured logging for the application.

    Args:
        level: Python logging level string (e.g. 'INFO', 'DEBUG').
    """
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    "datefmt": "%Y-%m-%dT%H:%M:%S",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                }
            },
            "root": {"level": level, "handlers": ["console"]},
        }
    )


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application.

    Returns:
        Configured FastAPI instance with routes and middleware.
    """
    settings = get_settings()
    _configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        logger.info("Starting %s v%s", settings.app_name, __version__)
        yield
        logger.info("Shutting down %s", settings.app_name)

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "Monte Carlo Data Quality API — bootstrap-based confidence intervals "
            "and statistical tests for data quality monitoring."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS — restrict in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    prefix = settings.api_prefix
    app.include_router(health_router, prefix=prefix)
    app.include_router(quality_router, prefix=prefix)

    return app


app = create_app()
