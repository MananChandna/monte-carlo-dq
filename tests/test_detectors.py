# tests/test_detectors.py
"""Unit tests for AnomalyDetector."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.detectors import AnomalyDetector, VolumeReport


@pytest.fixture()
def detector() -> AnomalyDetector:
    return AnomalyDetector(zscore_threshold=3.0, iqr_multiplier=1.5, random_seed=0)


@pytest.fixture()
def numeric_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = rng.normal(100, 10, 500).tolist()
    data += [500.0, -300.0]  # inject obvious outliers
    return pd.DataFrame({"value": data, "other": rng.normal(0, 1, 502)})


class TestZScore:
    def test_detects_outliers(self, detector: AnomalyDetector, numeric_df: pd.DataFrame) -> None:
        reports = detector.detect_zscore(numeric_df)
        value_report = next(r for r in reports if r.column == "value")
        assert value_report.n_anomalies >= 2

    def test_method_label(self, detector: AnomalyDetector, numeric_df: pd.DataFrame) -> None:
        reports = detector.detect_zscore(numeric_df)
        assert all(r.method == "zscore" for r in reports)

    def test_anomaly_rate_in_range(
        self, detector: AnomalyDetector, numeric_df: pd.DataFrame
    ) -> None:
        reports = detector.detect_zscore(numeric_df)
        assert all(0.0 <= r.anomaly_rate <= 1.0 for r in reports)

    def test_column_filter(self, detector: AnomalyDetector, numeric_df: pd.DataFrame) -> None:
        reports = detector.detect_zscore(numeric_df, columns=["value"])
        assert len(reports) == 1
        assert reports[0].column == "value"


class TestIQR:
    def test_detects_outliers(self, detector: AnomalyDetector, numeric_df: pd.DataFrame) -> None:
        reports = detector.detect_iqr(numeric_df)
        value_report = next(r for r in reports if r.column == "value")
        assert value_report.n_anomalies >= 2

    def test_method_label(self, detector: AnomalyDetector, numeric_df: pd.DataFrame) -> None:
        reports = detector.detect_iqr(numeric_df)
        assert all(r.method == "iqr" for r in reports)


class TestIsolationForest:
    def test_returns_single_report(
        self, detector: AnomalyDetector, numeric_df: pd.DataFrame
    ) -> None:
        report = detector.detect_isolation_forest(numeric_df)
        assert report.method == "isolation_forest"

    def test_anomalies_non_negative(
        self, detector: AnomalyDetector, numeric_df: pd.DataFrame
    ) -> None:
        report = detector.detect_isolation_forest(numeric_df)
        assert report.n_anomalies >= 0

    def test_small_df_handled(self, detector: AnomalyDetector) -> None:
        tiny = pd.DataFrame({"x": [1.0, 2.0]})
        report = detector.detect_isolation_forest(tiny)
        assert report.n_anomalies == 0


class TestVolumeAnomaly:
    def test_no_history_no_anomaly(self, detector: AnomalyDetector) -> None:
        report = detector.detect_volume_anomaly(1000, [])
        assert report.anomaly_detected is False

    def test_normal_count_no_anomaly(self, detector: AnomalyDetector) -> None:
        history = [1000] * 30
        report = detector.detect_volume_anomaly(1005, history)
        assert report.anomaly_detected is False

    def test_massive_drop_triggers_anomaly(self, detector: AnomalyDetector) -> None:
        history = [10_000] * 30
        report = detector.detect_volume_anomaly(100, history)
        assert report.anomaly_detected is True

    def test_report_fields(self, detector: AnomalyDetector) -> None:
        report = detector.detect_volume_anomaly(500, [490, 510, 500, 505, 495])
        assert isinstance(report, VolumeReport)
        assert report.current_count == 500
        assert report.z_score >= 0


class TestFreshness:
    def test_fresh_data(self, detector: AnomalyDetector) -> None:
        df = pd.DataFrame({"ts": [pd.Timestamp.utcnow()]})
        report = detector.check_freshness(df, "ts", max_age_hours=24)
        assert report.stale is False

    def test_stale_data(self, detector: AnomalyDetector) -> None:
        df = pd.DataFrame({"ts": [pd.Timestamp("2000-01-01")]})
        report = detector.check_freshness(df, "ts", max_age_hours=1)
        assert report.stale is True

    def test_missing_col(self, detector: AnomalyDetector) -> None:
        df = pd.DataFrame({"other": [1, 2]})
        report = detector.check_freshness(df, "ts", max_age_hours=24)
        assert report.stale is True

    def test_empty_df(self, detector: AnomalyDetector) -> None:
        report = detector.check_freshness(pd.DataFrame(), "ts")
        assert report.stale is True
