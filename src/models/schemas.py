# src/models/schemas.py
"""Pydantic v2 request and response schemas for the DQ API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "HealthResponse",
    "RunRequest",
    "DimensionResultSchema",
    "SimulationResultSchema",
    "ColumnProfileSchema",
    "DatasetProfileSchema",
    "DriftReportSchema",
    "QualityRunRecord",
    "HistoryResponse",
]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """API health-check response."""

    model_config = ConfigDict(frozen=True)

    status: str = "ok"
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Run request / response
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    """Request body for POST /api/v1/quality/run.

    Attributes:
        dataset_name: Logical name for the dataset being evaluated.
        n_simulations: Monte Carlo iteration count.
        quality_threshold: Minimum passing score.
        key_columns: Columns used for uniqueness checks.
        timestamp_col: Column for timeliness / freshness checks.
        max_age_hours: Staleness threshold in hours.
        sample_fraction: Bootstrap sample fraction.
    """

    dataset_name: str = Field(..., min_length=1, max_length=128)
    n_simulations: int = Field(default=1_000, ge=100, le=100_000)
    quality_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    key_columns: list[str] | None = None
    timestamp_col: str | None = None
    max_age_hours: float = Field(default=24.0, gt=0)
    sample_fraction: float = Field(default=0.8, gt=0.0, le=1.0)


class DimensionResultSchema(BaseModel):
    """Schema for a single DQ dimension result."""

    model_config = ConfigDict(frozen=True)

    dimension: str
    mean_score: float
    std_dev: float
    p5: float
    p95: float
    p_value: float
    observed_score: float


class SimulationResultSchema(BaseModel):
    """Schema for a complete Monte Carlo run result."""

    model_config = ConfigDict(frozen=True)

    n_simulations: int
    overall_score: float
    dimensions: list[DimensionResultSchema]
    passed: bool
    threshold: float


# ---------------------------------------------------------------------------
# Profile schemas
# ---------------------------------------------------------------------------


class ColumnProfileSchema(BaseModel):
    """Schema for a single column statistical profile."""

    model_config = ConfigDict(frozen=True)

    name: str
    dtype: str
    null_rate: float
    cardinality: int
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    skew: float | None = None
    kurtosis: float | None = None
    min_value: Any = None
    max_value: Any = None
    top_values: dict[str, int] = Field(default_factory=dict)


class DatasetProfileSchema(BaseModel):
    """Schema for a full dataset profile response."""

    model_config = ConfigDict(frozen=True)

    n_rows: int
    n_columns: int
    columns: list[ColumnProfileSchema]
    memory_mb: float


class DriftReportSchema(BaseModel):
    """Schema for a drift report entry."""

    model_config = ConfigDict(frozen=True)

    column: str
    ks_statistic: float | None = None
    ks_p_value: float | None = None
    js_divergence: float | None = None
    schema_changed: bool
    baseline_null_rate: float
    current_null_rate: float
    drift_detected: bool


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


class QualityRunRecord(BaseModel):
    """Persisted record of a completed DQ run."""

    run_id: str
    dataset_name: str
    created_at: datetime
    result: SimulationResultSchema


class HistoryResponse(BaseModel):
    """Paginated history of past DQ runs."""

    total: int
    runs: list[QualityRunRecord]
