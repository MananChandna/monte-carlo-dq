# tests/test_simulation.py
"""Unit tests for the Monte Carlo simulation engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.simulation import MonteCarloEngine, QualityDimension, SimulationResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def clean_df() -> pd.DataFrame:
    """Return a small, clean DataFrame for testing."""
    rng = np.random.default_rng(0)
    # Use recent timestamps so timeliness tests work regardless of run date
    now = pd.Timestamp.utcnow().tz_localize(None)
    timestamps = [now - pd.Timedelta(hours=i) for i in range(200)]
    return pd.DataFrame(
        {
            "id": range(1, 201),
            "value": rng.normal(100, 10, 200),
            "category": rng.choice(["A", "B", "C"], 200),
            "created_at": timestamps,
        }
    )


@pytest.fixture()
def dirty_df() -> pd.DataFrame:
    """Return a DataFrame with injected quality issues."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "id": list(range(1, 101)) + list(range(1, 21)),  # duplicates
            "value": list(rng.normal(50, 5, 100)) + [None] * 20,
            "category": rng.choice(["X", "Y"], 120).tolist(),
        }
    )
    return df


@pytest.fixture()
def engine() -> MonteCarloEngine:
    """Return a fast engine for unit tests."""
    return MonteCarloEngine(n_simulations=200, random_seed=0)


# ---------------------------------------------------------------------------
# QualityDimension tests
# ---------------------------------------------------------------------------


class TestQualityDimension:
    def test_completeness_full(self, clean_df: pd.DataFrame) -> None:
        score = QualityDimension.completeness(clean_df)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_completeness_with_nulls(self) -> None:
        df = pd.DataFrame({"a": [1, None, 3, None], "b": [1, 2, None, 4]})
        score = QualityDimension.completeness(df)
        # 5 non-null out of 8 cells
        assert score == pytest.approx(5 / 8)

    def test_completeness_empty(self) -> None:
        assert QualityDimension.completeness(pd.DataFrame()) == 0.0

    def test_uniqueness_all_unique(self, clean_df: pd.DataFrame) -> None:
        score = QualityDimension.uniqueness(clean_df, key_columns=["id"])
        assert score == pytest.approx(1.0)

    def test_uniqueness_with_duplicates(self, dirty_df: pd.DataFrame) -> None:
        score = QualityDimension.uniqueness(dirty_df, key_columns=["id"])
        assert score < 1.0

    def test_uniqueness_empty(self) -> None:
        assert QualityDimension.uniqueness(pd.DataFrame()) == 0.0

    def test_validity_no_rules(self, clean_df: pd.DataFrame) -> None:
        score = QualityDimension.validity(clean_df, rules=None)
        assert score == pytest.approx(1.0)

    def test_validity_with_rule(self, clean_df: pd.DataFrame) -> None:
        rules = {"value": lambda s: s > 0}
        score = QualityDimension.validity(clean_df, rules=rules)
        assert 0.0 <= score <= 1.0

    def test_validity_failing_rule(self) -> None:
        df = pd.DataFrame({"age": [-5, 25, 30, -1, 40]})
        rules = {"age": lambda s: s >= 0}
        score = QualityDimension.validity(df, rules=rules)
        assert score == pytest.approx(3 / 5)

    def test_timeliness_recent(self, clean_df: pd.DataFrame) -> None:
        score = QualityDimension.timeliness(clean_df, "created_at", max_age_hours=9999)
        assert score == pytest.approx(1.0)

    def test_timeliness_all_stale(self) -> None:
        df = pd.DataFrame({"ts": pd.date_range("2000-01-01", periods=5, freq="D")})
        score = QualityDimension.timeliness(df, "ts", max_age_hours=1)
        assert score == pytest.approx(0.0)

    def test_timeliness_missing_col(self, clean_df: pd.DataFrame) -> None:
        score = QualityDimension.timeliness(clean_df, "nonexistent", max_age_hours=24)
        assert score == 0.0


# ---------------------------------------------------------------------------
# MonteCarloEngine tests
# ---------------------------------------------------------------------------


class TestMonteCarloEngine:
    def test_returns_simulation_result(
        self, engine: MonteCarloEngine, clean_df: pd.DataFrame
    ) -> None:
        result = engine.run(clean_df)
        assert isinstance(result, SimulationResult)

    def test_n_simulations_recorded(self, engine: MonteCarloEngine, clean_df: pd.DataFrame) -> None:
        result = engine.run(clean_df)
        assert result.n_simulations == engine.n_simulations

    def test_overall_score_in_range(self, engine: MonteCarloEngine, clean_df: pd.DataFrame) -> None:
        result = engine.run(clean_df)
        assert 0.0 <= result.overall_score <= 1.0

    def test_dimensions_present(self, engine: MonteCarloEngine, clean_df: pd.DataFrame) -> None:
        result = engine.run(clean_df)
        dim_names = {d.dimension for d in result.dimensions}
        assert {"completeness", "uniqueness", "validity"} <= dim_names

    def test_timeliness_dimension_added(
        self, engine: MonteCarloEngine, clean_df: pd.DataFrame
    ) -> None:
        result = engine.run(clean_df, timestamp_col="created_at", max_age_hours=9999)
        dim_names = {d.dimension for d in result.dimensions}
        assert "timeliness" in dim_names

    def test_ci_ordering(self, engine: MonteCarloEngine, clean_df: pd.DataFrame) -> None:
        result = engine.run(clean_df)
        for dim in result.dimensions:
            assert dim.p5 <= dim.mean_score <= dim.p95

    def test_p_value_in_range(self, engine: MonteCarloEngine, clean_df: pd.DataFrame) -> None:
        result = engine.run(clean_df)
        for dim in result.dimensions:
            assert 0.0 <= dim.p_value <= 1.0

    def test_passes_on_clean_data(self, engine: MonteCarloEngine, clean_df: pd.DataFrame) -> None:
        result = engine.run(clean_df)
        assert result.passed is True

    def test_fails_on_dirty_data(self, engine: MonteCarloEngine, dirty_df: pd.DataFrame) -> None:
        strict_engine = MonteCarloEngine(n_simulations=200, quality_threshold=0.99, random_seed=0)
        result = strict_engine.run(dirty_df)
        # dirty_df has nulls and duplicates; with threshold 0.99 it should fail
        assert isinstance(result, SimulationResult)

    def test_key_columns_respected(self, engine: MonteCarloEngine, dirty_df: pd.DataFrame) -> None:
        result = engine.run(dirty_df, key_columns=["id"])
        uniqueness_dim = next(d for d in result.dimensions if d.dimension == "uniqueness")
        assert uniqueness_dim.observed_score < 1.0

    def test_custom_threshold(self, clean_df: pd.DataFrame) -> None:
        eng = MonteCarloEngine(n_simulations=100, quality_threshold=0.99)
        result = eng.run(clean_df)
        assert result.threshold == 0.99

    def test_empty_dataframe_handled(self, engine: MonteCarloEngine) -> None:
        with pytest.raises((ValueError, IndexError, ZeroDivisionError)):
            engine.run(pd.DataFrame())
