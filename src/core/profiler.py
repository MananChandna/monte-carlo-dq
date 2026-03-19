# src/core/profiler.py
"""Statistical profiling for data quality monitoring.

Computes column-level descriptive statistics, detects schema drift between
baseline and current snapshots, and runs distribution-level tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

__all__ = [
    "StatisticalProfiler",
    "ColumnProfile",
    "DatasetProfile",
    "DriftReport",
]

logger = logging.getLogger(__name__)


@dataclass
class ColumnProfile:
    """Statistical profile for a single column.

    Attributes:
        name: Column name.
        dtype: Pandas dtype string.
        null_rate: Fraction of null values.
        cardinality: Number of distinct non-null values.
        mean: Arithmetic mean (numeric columns only).
        median: Median value (numeric columns only).
        std: Standard deviation (numeric columns only).
        skew: Skewness (numeric columns only).
        kurtosis: Excess kurtosis (numeric columns only).
        min_value: Minimum value.
        max_value: Maximum value.
        top_values: Most frequent values with counts.
    """

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
    top_values: dict[str, int] = field(default_factory=dict)


@dataclass
class DatasetProfile:
    """Full profile for a DataFrame.

    Attributes:
        n_rows: Total row count.
        n_columns: Total column count.
        columns: Per-column profiles.
        memory_mb: Approximate DataFrame memory usage in MB.
    """

    n_rows: int
    n_columns: int
    columns: list[ColumnProfile]
    memory_mb: float


@dataclass
class DriftReport:
    """Drift comparison between baseline and current dataset.

    Attributes:
        column: Column being compared.
        ks_statistic: KS test statistic (numeric columns).
        ks_p_value: KS test p-value; low value indicates drift.
        js_divergence: Jensen-Shannon divergence (categorical columns).
        schema_changed: True if dtype changed between snapshots.
        baseline_null_rate: Null rate in baseline.
        current_null_rate: Null rate in current dataset.
        drift_detected: True if any drift signal fires.
    """

    column: str
    ks_statistic: float | None = None
    ks_p_value: float | None = None
    js_divergence: float | None = None
    schema_changed: bool = False
    baseline_null_rate: float = 0.0
    current_null_rate: float = 0.0
    drift_detected: bool = False


class StatisticalProfiler:
    """Profiles DataFrames and detects distributional drift.

    Args:
        top_n: Number of top values to include in column profiles.
        ks_alpha: Significance level for KS test drift detection.
        js_threshold: JS divergence threshold for categorical drift.
    """

    def __init__(
        self,
        top_n: int = 10,
        ks_alpha: float = 0.05,
        js_threshold: float = 0.1,
    ) -> None:
        self.top_n = top_n
        self.ks_alpha = ks_alpha
        self.js_threshold = js_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile(self, df: pd.DataFrame) -> DatasetProfile:
        """Generate a full statistical profile of a DataFrame.

        Args:
            df: DataFrame to profile.

        Returns:
            DatasetProfile containing per-column statistics.
        """
        columns = [self._profile_column(df[col]) for col in df.columns]
        return DatasetProfile(
            n_rows=len(df),
            n_columns=len(df.columns),
            columns=columns,
            memory_mb=round(df.memory_usage(deep=True).sum() / 1024**2, 4),
        )

    def detect_drift(
        self,
        baseline: pd.DataFrame,
        current: pd.DataFrame,
    ) -> list[DriftReport]:
        """Compare current data against a baseline for drift signals.

        Runs KS tests for numeric columns and Jensen-Shannon divergence for
        categorical columns.  Also flags schema changes (dtype mismatches).

        Args:
            baseline: Reference DataFrame (historical snapshot).
            current: Current DataFrame to evaluate.

        Returns:
            List of DriftReport objects, one per shared column.
        """
        reports: list[DriftReport] = []
        shared_cols = set(baseline.columns) & set(current.columns)

        for col in sorted(shared_cols):
            report = self._compare_column(col, baseline[col], current[col])
            reports.append(report)

        logger.info(
            "Drift detection complete: %d columns checked, %d drifted",
            len(reports),
            sum(r.drift_detected for r in reports),
        )
        return reports

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _profile_column(self, series: pd.Series) -> ColumnProfile:
        """Compute statistics for a single column.

        Args:
            series: Column Series to profile.

        Returns:
            ColumnProfile with computed statistics.
        """
        null_rate = float(series.isna().mean())
        clean = series.dropna()
        cardinality = int(clean.nunique())

        top_values: dict[str, int] = {}
        if len(clean) > 0:
            vc = clean.value_counts().head(self.top_n)
            top_values = {str(k): int(v) for k, v in vc.items()}

        profile = ColumnProfile(
            name=series.name,
            dtype=str(series.dtype),
            null_rate=null_rate,
            cardinality=cardinality,
            top_values=top_values,
        )

        if pd.api.types.is_numeric_dtype(series):
            if len(clean) > 0:
                profile.mean = float(clean.mean())
                profile.median = float(clean.median())
                profile.std = float(clean.std())
                profile.skew = float(clean.skew())
                profile.kurtosis = float(clean.kurtosis())
                profile.min_value = float(clean.min())
                profile.max_value = float(clean.max())
        else:
            if len(clean) > 0:
                profile.min_value = str(clean.min())
                profile.max_value = str(clean.max())

        return profile

    def _compare_column(
        self,
        col: str,
        baseline_series: pd.Series,
        current_series: pd.Series,
    ) -> DriftReport:
        """Run drift tests for a single column.

        Args:
            col: Column name.
            baseline_series: Historical series.
            current_series: Current series.

        Returns:
            DriftReport with test results.
        """
        schema_changed = str(baseline_series.dtype) != str(current_series.dtype)
        b_null = float(baseline_series.isna().mean())
        c_null = float(current_series.isna().mean())

        report = DriftReport(
            column=col,
            schema_changed=schema_changed,
            baseline_null_rate=b_null,
            current_null_rate=c_null,
        )

        b_clean = baseline_series.dropna()
        c_clean = current_series.dropna()

        if len(b_clean) == 0 or len(c_clean) == 0:
            report.drift_detected = schema_changed
            return report

        # If schema changed, skip distribution test to avoid type errors
        if schema_changed:
            report.drift_detected = True
            return report

        if pd.api.types.is_numeric_dtype(baseline_series):
            ks_stat, ks_p = stats.ks_2samp(
                b_clean.astype(float).values,
                c_clean.astype(float).values,
            )
            report.ks_statistic = float(ks_stat)
            report.ks_p_value = float(ks_p)
            report.drift_detected = schema_changed or (ks_p < self.ks_alpha)
        else:
            js_div = self._js_divergence(b_clean, c_clean)
            report.js_divergence = js_div
            report.drift_detected = schema_changed or (js_div > self.js_threshold)

        return report

    def _js_divergence(
        self,
        s1: pd.Series,
        s2: pd.Series,
    ) -> float:
        """Compute Jensen-Shannon divergence between two categorical series.

        Args:
            s1: First categorical series.
            s2: Second categorical series.

        Returns:
            JS divergence in [0, 1].
        """
        all_cats = sorted(set(s1.astype(str)) | set(s2.astype(str)))
        p = s1.astype(str).value_counts(normalize=True)
        q = s2.astype(str).value_counts(normalize=True)

        p_vec = np.array([p.get(c, 0.0) for c in all_cats], dtype=float)
        q_vec = np.array([q.get(c, 0.0) for c in all_cats], dtype=float)

        # jensenshannon returns the square root of JS divergence
        return float(jensenshannon(p_vec, q_vec) ** 2)
