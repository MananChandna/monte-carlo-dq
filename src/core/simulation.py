# src/core/simulation.py
"""Monte Carlo simulation engine for data quality scoring.

This module provides bootstrap-based Monte Carlo simulations to estimate
confidence intervals and p-values across multiple data quality dimensions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "MonteCarloEngine",
    "SimulationResult",
    "QualityDimension",
    "DimensionResult",
]

logger = logging.getLogger(__name__)


@dataclass
class DimensionResult:
    """Result for a single quality dimension simulation.

    Attributes:
        dimension: Name of the quality dimension.
        mean_score: Mean quality score across all simulations.
        std_dev: Standard deviation of scores.
        p5: 5th percentile (lower confidence bound).
        p95: 95th percentile (upper confidence bound).
        p_value: P-value against null hypothesis of perfect quality.
        observed_score: Actual observed score (no simulation).
    """

    dimension: str
    mean_score: float
    std_dev: float
    p5: float
    p95: float
    p_value: float
    observed_score: float


@dataclass
class SimulationResult:
    """Aggregated result from a full Monte Carlo DQ run.

    Attributes:
        n_simulations: Number of bootstrap iterations run.
        overall_score: Weighted mean of all dimension scores.
        dimensions: Per-dimension simulation results.
        passed: Whether overall score meets the threshold.
        threshold: Minimum acceptable quality score.
    """

    n_simulations: int
    overall_score: float
    dimensions: list[DimensionResult]
    passed: bool
    threshold: float = 0.8


class QualityDimension:
    """Computes raw quality scores for individual DQ dimensions.

    All score methods return a float in [0.0, 1.0] where 1.0 is perfect.
    """

    @staticmethod
    def completeness(df: pd.DataFrame) -> float:
        """Compute completeness score as fraction of non-null values.

        Args:
            df: Input DataFrame to evaluate.

        Returns:
            Fraction of non-null cells across the entire DataFrame.
        """
        if df.empty:
            return 0.0
        total = df.size
        non_null = df.notna().sum().sum()
        return float(non_null / total)

    @staticmethod
    def uniqueness(df: pd.DataFrame, key_columns: list[str] | None = None) -> float:
        """Compute uniqueness score based on duplicate rows.

        Args:
            df: Input DataFrame to evaluate.
            key_columns: Columns to check for uniqueness. Uses all columns
                if None.

        Returns:
            Fraction of unique rows.
        """
        if df.empty:
            return 0.0
        cols = key_columns or df.columns.tolist()
        cols = [c for c in cols if c in df.columns]
        if not cols:
            return 1.0
        n_unique = df[cols].drop_duplicates().shape[0]
        return float(n_unique / len(df))

    @staticmethod
    def validity(
        df: pd.DataFrame,
        rules: dict[str, Callable[[pd.Series], pd.Series]] | None = None,
    ) -> float:
        """Compute validity score by applying column-level rules.

        Args:
            df: Input DataFrame to evaluate.
            rules: Mapping of column name → validator function.  Each
                function should accept a Series and return a boolean Series.

        Returns:
            Fraction of cells passing all provided rules.  Returns 1.0 when
            no rules are supplied.
        """
        if not rules or df.empty:
            return 1.0
        passed = 0
        total = 0
        for col, rule_fn in rules.items():
            if col not in df.columns:
                continue
            mask = rule_fn(df[col])
            passed += int(mask.sum())
            total += len(mask)
        return float(passed / total) if total > 0 else 1.0

    @staticmethod
    def timeliness(
        df: pd.DataFrame,
        timestamp_col: str,
        max_age_hours: float = 24.0,
    ) -> float:
        """Compute timeliness score based on age of records.

        Args:
            df: Input DataFrame to evaluate.
            timestamp_col: Column containing record timestamps.
            max_age_hours: Records older than this are considered stale.

        Returns:
            Fraction of records within acceptable age.
        """
        if df.empty or timestamp_col not in df.columns:
            return 0.0
        now = pd.Timestamp.utcnow().tz_localize(None)
        raw = pd.to_datetime(df[timestamp_col], errors="coerce")
        # Strip timezone if present so comparison is always tz-naive
        if hasattr(raw, "dt"):
            ts = (
                raw.dt.tz_localize(None)
                if raw.dt.tz is None
                else raw.dt.tz_convert("UTC").dt.tz_localize(None)
            )
        else:
            ts = raw
        cutoff = now - pd.Timedelta(hours=max_age_hours)
        fresh = (ts >= cutoff).sum()
        return float(fresh / len(df))


class MonteCarloEngine:
    """Bootstrap-based Monte Carlo engine for DQ confidence intervals.

    Runs repeated bootstrap samples of the dataset, computes quality scores
    on each sample, and derives statistical summaries including p-values
    against the null hypothesis that the dataset has perfect quality.

    Args:
        n_simulations: Number of bootstrap iterations (default 10_000).
        sample_fraction: Fraction of rows per bootstrap draw (default 0.8).
        random_seed: RNG seed for reproducibility.
        quality_threshold: Minimum mean score to pass (default 0.8).
    """

    def __init__(
        self,
        n_simulations: int = 10_000,
        sample_fraction: float = 0.8,
        random_seed: int = 42,
        quality_threshold: float = 0.8,
    ) -> None:
        self.n_simulations = n_simulations
        self.sample_fraction = sample_fraction
        self.rng = np.random.default_rng(random_seed)
        self.quality_threshold = quality_threshold
        self._dimension = QualityDimension()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        key_columns: list[str] | None = None,
        validity_rules: dict[str, Callable[[pd.Series], pd.Series]] | None = None,
        timestamp_col: str | None = None,
        max_age_hours: float = 24.0,
    ) -> SimulationResult:
        """Execute the full Monte Carlo DQ simulation.

        Args:
            df: Dataset to evaluate.
            key_columns: Columns used for uniqueness check.
            validity_rules: Column-level validity rules.
            timestamp_col: Column name for timeliness checks.
            max_age_hours: Stale record threshold in hours.

        Returns:
            SimulationResult with per-dimension statistics.
        """
        logger.info(
            "Starting Monte Carlo simulation: n=%d, rows=%d",
            self.n_simulations,
            len(df),
        )

        dim_scores: dict[str, list[float]] = {
            "completeness": [],
            "uniqueness": [],
            "validity": [],
        }
        if timestamp_col:
            dim_scores["timeliness"] = []

        n_sample = max(1, int(len(df) * self.sample_fraction))

        for _ in range(self.n_simulations):
            idx = self.rng.integers(0, len(df), size=n_sample)
            sample = df.iloc[idx]

            dim_scores["completeness"].append(self._dimension.completeness(sample))
            dim_scores["uniqueness"].append(self._dimension.uniqueness(sample, key_columns))
            dim_scores["validity"].append(self._dimension.validity(sample, validity_rules))
            if timestamp_col:
                dim_scores["timeliness"].append(
                    self._dimension.timeliness(sample, timestamp_col, max_age_hours)
                )

        # Observed (no sampling) scores for each dimension
        observed: dict[str, float] = {
            "completeness": self._dimension.completeness(df),
            "uniqueness": self._dimension.uniqueness(df, key_columns),
            "validity": self._dimension.validity(df, validity_rules),
        }
        if timestamp_col:
            observed["timeliness"] = self._dimension.timeliness(df, timestamp_col, max_age_hours)

        results = [
            self._build_dimension_result(dim, scores, observed[dim])
            for dim, scores in dim_scores.items()
        ]

        overall = float(np.mean([r.mean_score for r in results]))
        return SimulationResult(
            n_simulations=self.n_simulations,
            overall_score=overall,
            dimensions=results,
            passed=overall >= self.quality_threshold,
            threshold=self.quality_threshold,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_dimension_result(
        self,
        dimension: str,
        scores: list[float],
        observed_score: float,
    ) -> DimensionResult:
        """Summarise bootstrap score distribution for one dimension.

        Args:
            dimension: Dimension name.
            scores: List of bootstrap scores.
            observed_score: Score on the full unsampled dataset.

        Returns:
            DimensionResult with CI and p-value.
        """
        arr = np.array(scores)
        mean = float(arr.mean())
        std = float(arr.std())
        p5 = float(np.percentile(arr, 5))
        p95 = float(np.percentile(arr, 95))

        # When all scores are identical (std == 0), ttest is undefined.
        # A perfectly degenerate distribution at 1.0 means no evidence against
        # null of perfect quality → p_value = 1.0; any other degenerate value
        # is maximally significant → p_value = 0.0.
        if std == 0.0:
            p_value = 1.0 if mean == 1.0 else 0.0
        else:
            _, raw_p = stats.ttest_1samp(arr, popmean=1.0)
            p_value = float(raw_p) if not np.isnan(raw_p) else 1.0

        return DimensionResult(
            dimension=dimension,
            mean_score=mean,
            std_dev=std,
            p5=p5,
            p95=p95,
            p_value=p_value,
            observed_score=observed_score,
        )
