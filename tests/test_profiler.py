# tests/test_profiler.py
"""Unit tests for StatisticalProfiler and drift detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.profiler import DatasetProfile, StatisticalProfiler

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def profiler() -> StatisticalProfiler:
    return StatisticalProfiler(top_n=5, ks_alpha=0.05, js_threshold=0.1)


@pytest.fixture()
def numeric_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "a": rng.normal(0, 1, 500),
            "b": rng.exponential(2, 500),
            "c": rng.integers(1, 10, 500).astype(float),
        }
    )


@pytest.fixture()
def mixed_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "num": rng.normal(50, 10, 300),
            "cat": rng.choice(["X", "Y", "Z"], 300),
            "nullable_num": list(rng.normal(5, 1, 250)) + [None] * 50,
        }
    )


# ---------------------------------------------------------------------------
# Profile tests
# ---------------------------------------------------------------------------


class TestStatisticalProfiler:
    def test_returns_dataset_profile(
        self, profiler: StatisticalProfiler, numeric_df: pd.DataFrame
    ) -> None:
        result = profiler.profile(numeric_df)
        assert isinstance(result, DatasetProfile)

    def test_row_and_col_counts(
        self, profiler: StatisticalProfiler, numeric_df: pd.DataFrame
    ) -> None:
        result = profiler.profile(numeric_df)
        assert result.n_rows == len(numeric_df)
        assert result.n_columns == len(numeric_df.columns)

    def test_numeric_stats_populated(
        self, profiler: StatisticalProfiler, numeric_df: pd.DataFrame
    ) -> None:
        result = profiler.profile(numeric_df)
        col_a = next(c for c in result.columns if c.name == "a")
        assert col_a.mean is not None
        assert col_a.std is not None
        assert col_a.skew is not None
        assert col_a.kurtosis is not None

    def test_null_rate_computed(
        self, profiler: StatisticalProfiler, mixed_df: pd.DataFrame
    ) -> None:
        result = profiler.profile(mixed_df)
        nullable_col = next(c for c in result.columns if c.name == "nullable_num")
        assert nullable_col.null_rate == pytest.approx(50 / 300, abs=1e-4)

    def test_cardinality(self, profiler: StatisticalProfiler, mixed_df: pd.DataFrame) -> None:
        result = profiler.profile(mixed_df)
        cat_col = next(c for c in result.columns if c.name == "cat")
        assert cat_col.cardinality == 3

    def test_top_values_limited(
        self, profiler: StatisticalProfiler, numeric_df: pd.DataFrame
    ) -> None:
        result = profiler.profile(numeric_df)
        for col in result.columns:
            assert len(col.top_values) <= profiler.top_n

    def test_memory_mb_positive(
        self, profiler: StatisticalProfiler, numeric_df: pd.DataFrame
    ) -> None:
        result = profiler.profile(numeric_df)
        assert result.memory_mb > 0

    def test_empty_dataframe(self, profiler: StatisticalProfiler) -> None:
        result = profiler.profile(pd.DataFrame())
        assert result.n_rows == 0
        assert result.columns == []


# ---------------------------------------------------------------------------
# Drift detection tests
# ---------------------------------------------------------------------------


class TestDriftDetection:
    def test_no_drift_identical(
        self, profiler: StatisticalProfiler, numeric_df: pd.DataFrame
    ) -> None:
        reports = profiler.detect_drift(numeric_df, numeric_df.copy())
        for r in reports:
            # Identical data — KS should not flag drift
            assert r.ks_p_value is None or r.ks_p_value > profiler.ks_alpha

    def test_drift_detected_different_dist(self, profiler: StatisticalProfiler) -> None:
        rng = np.random.default_rng(0)
        baseline = pd.DataFrame({"x": rng.normal(0, 1, 500)})
        current = pd.DataFrame({"x": rng.normal(10, 1, 500)})  # clearly different
        reports = profiler.detect_drift(baseline, current)
        assert len(reports) == 1
        assert reports[0].drift_detected

    def test_schema_change_detected(self, profiler: StatisticalProfiler) -> None:
        baseline = pd.DataFrame({"col": [1.0, 2.0, 3.0]})
        current = pd.DataFrame({"col": ["a", "b", "c"]})
        reports = profiler.detect_drift(baseline, current)
        assert reports[0].schema_changed is True

    def test_categorical_no_drift(self, profiler: StatisticalProfiler) -> None:
        rng = np.random.default_rng(7)
        b = pd.DataFrame({"cat": rng.choice(["A", "B", "C"], 500)})
        c = pd.DataFrame({"cat": rng.choice(["A", "B", "C"], 500)})
        reports = profiler.detect_drift(b, c)
        # JS divergence should be low for same distribution
        assert reports[0].js_divergence is not None
        assert reports[0].js_divergence < profiler.js_threshold

    def test_categorical_drift_detected(self, profiler: StatisticalProfiler) -> None:
        b = pd.DataFrame({"cat": ["A"] * 500})
        c = pd.DataFrame({"cat": ["B"] * 500})
        reports = profiler.detect_drift(b, c)
        assert reports[0].drift_detected is True

    def test_only_shared_columns_compared(self, profiler: StatisticalProfiler) -> None:
        b = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        c = pd.DataFrame({"a": [1.0, 2.0], "c": [5.0, 6.0]})
        reports = profiler.detect_drift(b, c)
        reported_cols = {r.column for r in reports}
        assert reported_cols == {"a"}

    def test_null_rate_tracked(self, profiler: StatisticalProfiler) -> None:
        b = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0]})
        c = pd.DataFrame({"v": [1.0, None, 3.0, None, 5.0]})
        reports = profiler.detect_drift(b, c)
        assert reports[0].baseline_null_rate == pytest.approx(0.0)
        assert reports[0].current_null_rate == pytest.approx(0.4)
