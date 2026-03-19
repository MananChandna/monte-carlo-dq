# src/api/routes/health.py
"""Health-check endpoint."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from src.models.schemas import HealthResponse

__all__ = ["router"]

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Health check")  # type: ignore[misc]
async def health() -> HealthResponse:
    """Return service health status.

    Returns:
        HealthResponse with status, version, and current timestamp.
    """
    from src import __version__

    return HealthResponse(
        status="ok",
        version=__version__,
        timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
    )
