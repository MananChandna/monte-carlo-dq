# src/api/routes/quality.py
"""Data quality REST endpoints."""

from __future__ import annotations

import io
import logging
import uuid
from datetime import datetime, timezone
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

from src.core.profiler import StatisticalProfiler
from src.core.simulation import MonteCarloEngine, SimulationResult
from src.models.schemas import (
    ColumnProfileSchema,
    DatasetProfileSchema,
    DimensionResultSchema,
    DriftReportSchema,
    HistoryResponse,
    QualityRunRecord,
    SimulationResultSchema,
)

__all__ = ["router"]

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/quality", tags=["quality"])

# In-memory store for POC (replace with PostgreSQL in production)
_run_history: list[QualityRunRecord] = []


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _parse_upload(file: UploadFile) -> pd.DataFrame:
    """Read an uploaded CSV or JSON file into a DataFrame.

    Args:
        file: FastAPI UploadFile object.

    Returns:
        Parsed DataFrame.

    Raises:
        HTTPException: If the file format is unsupported or parsing fails.
    """
    content = file.file.read()
    filename = file.filename or ""
    try:
        if filename.endswith(".csv"):
            return pd.read_csv(io.BytesIO(content))
        if filename.endswith(".json"):
            return pd.read_json(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to parse file: {exc}",
        ) from exc
    raise HTTPException(
        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        detail="Only .csv and .json uploads are supported",
    )


def _sim_result_to_schema(result: SimulationResult) -> SimulationResultSchema:
    """Convert internal SimulationResult to API schema.

    Args:
        result: SimulationResult dataclass instance.

    Returns:
        SimulationResultSchema Pydantic model.
    """
    return SimulationResultSchema(
        n_simulations=result.n_simulations,
        overall_score=result.overall_score,
        dimensions=[
            DimensionResultSchema(
                dimension=d.dimension,
                mean_score=d.mean_score,
                std_dev=d.std_dev,
                p5=d.p5,
                p95=d.p95,
                p_value=d.p_value,
                observed_score=d.observed_score,
            )
            for d in result.dimensions
        ],
        passed=result.passed,
        threshold=result.threshold,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/run",
    response_model=SimulationResultSchema,
    status_code=status.HTTP_200_OK,
    summary="Run full DQ check on an uploaded dataset",
)  # type: ignore[misc]
async def run_quality_check(
    file: Annotated[UploadFile, File(description="CSV or JSON dataset")],
    dataset_name: Annotated[str, Query(min_length=1, max_length=128)] = "dataset",
    n_simulations: Annotated[int, Query(ge=100, le=100_000)] = 1_000,
    quality_threshold: Annotated[float, Query(ge=0.0, le=1.0)] = 0.8,
    key_columns: Annotated[str | None, Query(description="Comma-separated key columns")] = None,
    timestamp_col: Annotated[str | None, Query()] = None,
    sample_fraction: Annotated[float, Query(gt=0.0, le=1.0)] = 0.8,
) -> SimulationResultSchema:
    """Execute a full Monte Carlo data quality simulation on the uploaded file.

    Upload a CSV or JSON dataset.  The engine runs bootstrap simulations to
    estimate quality score confidence intervals across completeness,
    uniqueness, validity, and (optionally) timeliness dimensions.

    Args:
        file: Dataset file (CSV or JSON).
        dataset_name: Logical identifier for this dataset.
        n_simulations: Number of Monte Carlo iterations.
        quality_threshold: Minimum overall score to pass.
        key_columns: Comma-separated column names for uniqueness check.
        timestamp_col: Column name for timeliness check.
        sample_fraction: Fraction of rows per bootstrap sample.

    Returns:
        SimulationResultSchema with per-dimension statistics.
    """
    df = _parse_upload(file)

    key_cols: list[str] | None = None
    if key_columns:
        key_cols = [c.strip() for c in key_columns.split(",") if c.strip()]

    engine = MonteCarloEngine(
        n_simulations=n_simulations,
        sample_fraction=sample_fraction,
        quality_threshold=quality_threshold,
    )

    try:
        result = engine.run(
            df,
            key_columns=key_cols,
            timestamp_col=timestamp_col,
        )
    except Exception as exc:
        logger.exception("Simulation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Simulation error: {exc}",
        ) from exc

    schema_result = _sim_result_to_schema(result)
    record = QualityRunRecord(
        run_id=str(uuid.uuid4()),
        dataset_name=dataset_name,
        created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        result=schema_result,
    )
    _run_history.append(record)

    logger.info(
        "DQ run complete: dataset=%s, score=%.4f, passed=%s",
        dataset_name,
        result.overall_score,
        result.passed,
    )
    return schema_result


@router.get(
    "/history",
    response_model=HistoryResponse,
    summary="Retrieve past DQ run results",
)  # type: ignore[misc]
async def get_history(
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    dataset_name: Annotated[str | None, Query()] = None,
) -> HistoryResponse:
    """Return paginated history of completed DQ runs.

    Args:
        limit: Maximum number of records to return.
        offset: Number of records to skip.
        dataset_name: Optional filter by dataset name.

    Returns:
        HistoryResponse with total count and run list.
    """
    filtered = _run_history
    if dataset_name:
        filtered = [r for r in _run_history if r.dataset_name == dataset_name]

    total = len(filtered)
    page = filtered[offset : offset + limit]
    return HistoryResponse(total=total, runs=page)


@router.post(
    "/profile",
    response_model=DatasetProfileSchema,
    summary="Get column-level statistical profile",
)  # type: ignore[misc]
async def get_profile(
    file: Annotated[UploadFile, File(description="CSV or JSON dataset")],
) -> DatasetProfileSchema:
    """Generate a statistical profile for an uploaded dataset.

    Args:
        file: Dataset file (CSV or JSON).

    Returns:
        DatasetProfileSchema with per-column statistics.
    """
    df = _parse_upload(file)
    profiler = StatisticalProfiler()
    profile = profiler.profile(df)

    return DatasetProfileSchema(
        n_rows=profile.n_rows,
        n_columns=profile.n_columns,
        memory_mb=profile.memory_mb,
        columns=[
            ColumnProfileSchema(
                name=c.name,
                dtype=c.dtype,
                null_rate=c.null_rate,
                cardinality=c.cardinality,
                mean=c.mean,
                median=c.median,
                std=c.std,
                skew=c.skew,
                kurtosis=c.kurtosis,
                min_value=c.min_value,
                max_value=c.max_value,
                top_values=c.top_values,
            )
            for c in profile.columns
        ],
    )


@router.post(
    "/drift",
    response_model=list[DriftReportSchema],
    summary="Detect drift between baseline and current dataset",
)  # type: ignore[misc]
async def detect_drift(
    baseline_file: Annotated[UploadFile, File(description="Baseline CSV or JSON")],
    current_file: Annotated[UploadFile, File(description="Current CSV or JSON")],
) -> list[DriftReportSchema]:
    """Compare a current dataset against a historical baseline for drift.

    Args:
        baseline_file: Reference dataset (historical snapshot).
        current_file: Current dataset to evaluate.

    Returns:
        List of DriftReportSchema, one per shared column.
    """
    baseline = _parse_upload(baseline_file)
    current = _parse_upload(current_file)

    profiler = StatisticalProfiler()
    reports = profiler.detect_drift(baseline, current)

    return [
        DriftReportSchema(
            column=r.column,
            ks_statistic=r.ks_statistic,
            ks_p_value=r.ks_p_value,
            js_divergence=r.js_divergence,
            schema_changed=r.schema_changed,
            baseline_null_rate=r.baseline_null_rate,
            current_null_rate=r.current_null_rate,
            drift_detected=r.drift_detected,
        )
        for r in reports
    ]
