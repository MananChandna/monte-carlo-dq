# src/db/connection.py
"""Database connection and session management via SQLAlchemy async."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from src.config import get_settings

__all__ = [
    "Base",
    "get_engine",
    "get_session",
    "init_db",
]

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):  # type: ignore[misc]
    """SQLAlchemy declarative base for ORM models."""

    pass


_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Return (or create) the shared async SQLAlchemy engine.

    Returns:
        Configured AsyncEngine connected to PostgreSQL.
    """
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.database_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            echo=settings.debug,
        )
        logger.info("Database engine created: %s", settings.database_url)
    return _engine


def _get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager yielding a database session.

    Yields:
        An AsyncSession that is committed on clean exit and rolled
        back on exception.

    Example:
        async with get_session() as session:
            result = await session.execute(text("SELECT 1"))
    """
    factory = _get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Create all tables defined via the ORM Base metadata.

    This is called during application startup.
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialised")
