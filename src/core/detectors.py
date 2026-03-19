# src/core/detectors.py
"""Anomaly and drift detection for data quality monitoring.

Implements Z-score, IQR, Isolation Forest, volume anomaly, and
freshness checks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

__all__ = [
    "AnomalyDetector",
    "AnomalyReport",
    "VolumeReport",
    "FreshnessReport",
]

logger = logging.getLogger(__name__)


@dataclass
class AnomalyReport:
    """Results from column-level outlier detection.

    Attributes:
        column: Column name.
        method: Detection method used ('zscore', 'iqr', 'isolation_forest').
        n_anomalies: Count of detected outliers.
        anomaly_rate: Fraction of rows flagged.
        anomaly_indices: Row indices of detected outliers.
        threshold_used: The threshold/parameter applied.
    """

    column: str
    method: str
    n_anomalies: int
    anomaly_rate: float
    anomaly_indices: list[int] = field(default_factory=list)
    threshold_used: float = 3.0


@dataclass
class VolumeReport:
    """Row-count volume anomaly results.

    Attributes:
        current_count: Observed row count.
        baseline_mean: Rolling mean of historical row counts.
        baseline_std: Rolling std of historical row counts.
        z_score: Z-score of current count vs baseline.
        anomaly_detected: True if z_score exceeds threshold.
        threshold: Z-score threshold for detection.
    """

    current_count: int
    baseline_mean: float
    baseline_std: float
    z_score: float
    anomaly_detected: bool
    threshold: float = 3.0


@dataclass
class FreshnessReport:
    """Freshness / timeliness anomaly results.

    Attributes:
        timestamp_col: Column checked.
        max_ts: Latest timestamp in the dataset.
        age_hours: Age of the freshest record in hours.
        stale: True if age exceeds max_age_hours.
        max_age_hours: Configured staleness threshold.
    """

    timestamp_col: str
    max_ts: datetime | None
    age_hours: float
    stale: bool
    max_age_hours: float


class AnomalyDetector:
    """Multi-method anomaly detection for tabular data.

    Args:
        zscore_threshold: Z-score threshold for outlier detection.
        iqr_multiplier: IQR fence multiplier (default 1.5 = Tukey fences).
        isolation_contamination: Expected outlier fraction for Isolation Forest.
        volume_z_threshold: Z-score threshold for volume anomalies.
        random_seed: RNG seed for Isolation Forest.
    """

    def __init__(
        self,
        zscore_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
        isolation_contamination: float = 0.05,
        volume_z_threshold: float = 3.0,
        random_seed: int = 42,
    ) -> None:
        self.zscore_threshold = zscore_threshold
        self.iqr_multiplier = iqr_multiplier
        self.isolation_contamination = isolation_contamination
        self.volume_z_threshold = volume_z_threshold
        self.random_seed = random_seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_zscore(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> list[AnomalyReport]:
        """Detect outliers using Z-score thresholding.

        Args:
            df: DataFrame to analyse.
            columns: Numeric columns to check. Defaults to all numeric cols.

        Returns:
            List of AnomalyReport, one per checked column.
        """
        numeric_cols = self._select_numeric(df, columns)
        reports: list[AnomalyReport] = []

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 2:
                continue
            z = np.abs(stats_zscore(series.values))
            mask = z > self.zscore_threshold
            indices = series.index[mask].tolist()
            reports.append(
                AnomalyReport(
                    column=col,
                    method="zscore",
                    n_anomalies=int(mask.sum()),
                    anomaly_rate=float(mask.mean()),
                    anomaly_indices=[int(i) for i in indices],
                    threshold_used=self.zscore_threshold,
                )
            )

        logger.debug("Z-score detection: %d columns checked", len(reports))
        return reports

    def detect_iqr(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> list[AnomalyReport]:
        """Detect outliers using Tukey IQR fences.

        Args:
            df: DataFrame to analyse.
            columns: Numeric columns to check. Defaults to all numeric cols.

        Returns:
            List of AnomalyReport, one per checked column.
        """
        numeric_cols = self._select_numeric(df, columns)
        reports: list[AnomalyReport] = []

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 4:
                continue
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - self.iqr_multiplier * iqr
            upper = q3 + self.iqr_multiplier * iqr
            mask = (series < lower) | (series > upper)
            indices = series.index[mask].tolist()
            reports.append(
                AnomalyReport(
                    column=col,
                    method="iqr",
                    n_anomalies=int(mask.sum()),
                    anomaly_rate=float(mask.mean()),
                    anomaly_indices=[int(i) for i in indices],
                    threshold_used=self.iqr_multiplier,
                )
            )

        return reports

    def detect_isolation_forest(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> AnomalyReport:
        """Detect multivariate anomalies via Isolation Forest.

        Args:
            df: DataFrame to analyse.
            columns: Numeric columns to use as features. Defaults to all
                numeric columns.

        Returns:
            Single AnomalyReport with row-level outlier flags.
        """
        numeric_cols = self._select_numeric(df, columns)
        feature_df = df[numeric_cols].dropna()

        if len(feature_df) < 10 or not numeric_cols:
            return AnomalyReport(
                column=",".join(numeric_cols) or "none",
                method="isolation_forest",
                n_anomalies=0,
                anomaly_rate=0.0,
            )

        clf = IsolationForest(
            contamination=self.isolation_contamination,
            random_state=self.random_seed,
            n_jobs=-1,
        )
        preds = clf.fit_predict(feature_df.values)
        mask = preds == -1
        indices = feature_df.index[mask].tolist()

        return AnomalyReport(
            column=",".join(numeric_cols),
            method="isolation_forest",
            n_anomalies=int(mask.sum()),
            anomaly_rate=float(mask.mean()),
            anomaly_indices=[int(i) for i in indices],
            threshold_used=self.isolation_contamination,
        )

    def detect_volume_anomaly(
        self,
        current_count: int,
        historical_counts: list[int],
    ) -> VolumeReport:
        """Detect if current row count is anomalous vs historical baseline.

        Args:
            current_count: Observed row count in current batch.
            historical_counts: List of row counts from previous batches.

        Returns:
            VolumeReport with z-score and anomaly flag.
        """
        if len(historical_counts) < 2:
            return VolumeReport(
                current_count=current_count,
                baseline_mean=float(current_count),
                baseline_std=0.0,
                z_score=0.0,
                anomaly_detected=False,
                threshold=self.volume_z_threshold,
            )

        arr = np.array(historical_counts, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std())
        # When historical counts are perfectly constant (std=0), only flag if
        # the deviation is large in absolute terms (>10% of mean).
        if std == 0.0:
            pct_dev = abs(current_count - mean) / (mean if mean > 0 else 1.0)
            anomaly = pct_dev > 0.10
            z = float("inf") if anomaly else 0.0
        else:
            z = abs(current_count - mean) / std
            anomaly = z > self.volume_z_threshold

        return VolumeReport(
            current_count=current_count,
            baseline_mean=mean,
            baseline_std=std,
            z_score=float(z),
            anomaly_detected=anomaly,
            threshold=self.volume_z_threshold,
        )

    def check_freshness(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
        max_age_hours: float = 24.0,
    ) -> FreshnessReport:
        """Check whether the most recent record is within acceptable age.

        Args:
            df: DataFrame containing a timestamp column.
            timestamp_col: Name of the timestamp column.
            max_age_hours: Maximum acceptable record age in hours.

        Returns:
            FreshnessReport with staleness verdict.
        """
        if timestamp_col not in df.columns or df.empty:
            return FreshnessReport(
                timestamp_col=timestamp_col,
                max_ts=None,
                age_hours=float("inf"),
                stale=True,
                max_age_hours=max_age_hours,
            )

        ts = pd.to_datetime(df[timestamp_col], errors="coerce").dropna()
        if ts.empty:
            return FreshnessReport(
                timestamp_col=timestamp_col,
                max_ts=None,
                age_hours=float("inf"),
                stale=True,
                max_age_hours=max_age_hours,
            )

        max_ts = ts.max()
        # Normalise timezone
        if max_ts.tzinfo is not None:
            max_ts = max_ts.tz_convert("UTC").tz_localize(None)
        now = pd.Timestamp.utcnow().tz_localize(None)
        age_hours = float((now - max_ts).total_seconds() / 3600)

        return FreshnessReport(
            timestamp_col=timestamp_col,
            max_ts=max_ts.to_pydatetime(),
            age_hours=age_hours,
            stale=age_hours > max_age_hours,
            max_age_hours=max_age_hours,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _select_numeric(
        df: pd.DataFrame,
        columns: list[str] | None,
    ) -> list[str]:
        """Return list of numeric column names from df.

        Args:
            df: Source DataFrame.
            columns: Explicit list, or None to use all numeric columns.

        Returns:
            Filtered list of numeric column names present in df.
        """
        if columns is not None:
            return [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        return list(df.select_dtypes(include="number").columns)


# ---------------------------------------------------------------------------
# Thin wrapper around scipy so callers don't import scipy directly
# ---------------------------------------------------------------------------


def stats_zscore(arr: np.ndarray) -> np.ndarray:
    """Compute element-wise Z-scores for a 1-D array.

    Args:
        arr: 1-D numeric array.

    Returns:
        Array of Z-scores with the same shape.
    """
    from scipy import stats as _stats

    return np.abs(_stats.zscore(arr, nan_policy="omit"))
