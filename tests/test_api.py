# tests/test_api.py
"""Integration tests for the FastAPI REST layer."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


def _csv_upload(df: pd.DataFrame, filename: str = "test.csv") -> tuple[str, tuple]:
    """Return a files dict suitable for TestClient multipart upload."""
    buf = io.BytesIO(df.to_csv(index=False).encode())
    return ("file", (filename, buf, "text/csv"))


@pytest.fixture()
def orders_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "order_id": [f"ORD-{i}" for i in range(500)],
            "amount": rng.lognormal(4, 1, 500).round(2),
            "status": rng.choice(["pending", "shipped", "delivered"], 500),
            "created_at": pd.date_range("2024-01-01", periods=500, freq="h").astype(str),
        }
    )


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient) -> None:
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_health_body(self, client: TestClient) -> None:
        resp = client.get("/api/v1/health")
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "timestamp" in data


# ---------------------------------------------------------------------------
# Quality run endpoint
# ---------------------------------------------------------------------------


class TestQualityRunEndpoint:
    def test_run_returns_200(self, client: TestClient, orders_df: pd.DataFrame) -> None:
        resp = client.post(
            "/api/v1/quality/run?n_simulations=200&dataset_name=orders",
            files=[_csv_upload(orders_df)],
        )
        assert resp.status_code == 200

    def test_run_response_shape(self, client: TestClient, orders_df: pd.DataFrame) -> None:
        resp = client.post(
            "/api/v1/quality/run?n_simulations=200",
            files=[_csv_upload(orders_df)],
        )
        data = resp.json()
        assert "overall_score" in data
        assert "dimensions" in data
        assert "passed" in data
        assert isinstance(data["dimensions"], list)

    def test_run_score_in_range(self, client: TestClient, orders_df: pd.DataFrame) -> None:
        resp = client.post(
            "/api/v1/quality/run?n_simulations=200",
            files=[_csv_upload(orders_df)],
        )
        data = resp.json()
        assert 0.0 <= data["overall_score"] <= 1.0

    def test_run_with_timestamp_col(self, client: TestClient, orders_df: pd.DataFrame) -> None:
        resp = client.post(
            "/api/v1/quality/run?n_simulations=200&timestamp_col=created_at&max_age_hours=99999",
            files=[_csv_upload(orders_df)],
        )
        assert resp.status_code == 200
        dims = {d["dimension"] for d in resp.json()["dimensions"]}
        assert "timeliness" in dims

    def test_run_unsupported_file_type(self, client: TestClient) -> None:
        buf = io.BytesIO(b"not a csv")
        resp = client.post(
            "/api/v1/quality/run?n_simulations=100",
            files=[("file", ("test.txt", buf, "text/plain"))],
        )
        assert resp.status_code == 415

    def test_run_stores_history(self, client: TestClient, orders_df: pd.DataFrame) -> None:
        client.post(
            "/api/v1/quality/run?n_simulations=100&dataset_name=hist_test",
            files=[_csv_upload(orders_df)],
        )
        resp = client.get("/api/v1/quality/history?dataset_name=hist_test")
        assert resp.status_code == 200
        assert resp.json()["total"] >= 1


# ---------------------------------------------------------------------------
# History endpoint
# ---------------------------------------------------------------------------


class TestHistoryEndpoint:
    def test_history_returns_200(self, client: TestClient) -> None:
        resp = client.get("/api/v1/quality/history")
        assert resp.status_code == 200

    def test_history_shape(self, client: TestClient) -> None:
        resp = client.get("/api/v1/quality/history")
        data = resp.json()
        assert "total" in data
        assert "runs" in data
        assert isinstance(data["runs"], list)

    def test_history_pagination(self, client: TestClient, orders_df: pd.DataFrame) -> None:
        # Seed a few runs
        for _ in range(3):
            client.post(
                "/api/v1/quality/run?n_simulations=100&dataset_name=paginate_test",
                files=[_csv_upload(orders_df)],
            )
        resp = client.get("/api/v1/quality/history?limit=2&offset=0&dataset_name=paginate_test")
        data = resp.json()
        assert len(data["runs"]) <= 2


# ---------------------------------------------------------------------------
# Profile endpoint
# ---------------------------------------------------------------------------


class TestProfileEndpoint:
    def test_profile_returns_200(self, client: TestClient, orders_df: pd.DataFrame) -> None:
        resp = client.post(
            "/api/v1/quality/profile",
            files=[_csv_upload(orders_df)],
        )
        assert resp.status_code == 200

    def test_profile_columns(self, client: TestClient, orders_df: pd.DataFrame) -> None:
        resp = client.post(
            "/api/v1/quality/profile",
            files=[_csv_upload(orders_df)],
        )
        data = resp.json()
        assert data["n_rows"] == len(orders_df)
        assert data["n_columns"] == len(orders_df.columns)
        assert len(data["columns"]) == len(orders_df.columns)

    def test_profile_col_stats(self, client: TestClient, orders_df: pd.DataFrame) -> None:
        resp = client.post(
            "/api/v1/quality/profile",
            files=[_csv_upload(orders_df)],
        )
        amount_col = next(c for c in resp.json()["columns"] if c["name"] == "amount")
        assert amount_col["mean"] is not None
        assert amount_col["null_rate"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Drift endpoint
# ---------------------------------------------------------------------------


class TestDriftEndpoint:
    def test_drift_returns_200(self, client: TestClient, orders_df: pd.DataFrame) -> None:
        resp = client.post(
            "/api/v1/quality/drift",
            files=[
                (
                    "baseline_file",
                    (
                        "baseline.csv",
                        io.BytesIO(orders_df.to_csv(index=False).encode()),
                        "text/csv",
                    ),
                ),
                (
                    "current_file",
                    ("current.csv", io.BytesIO(orders_df.to_csv(index=False).encode()), "text/csv"),
                ),
            ],
        )
        assert resp.status_code == 200

    def test_drift_report_fields(self, client: TestClient, orders_df: pd.DataFrame) -> None:
        resp = client.post(
            "/api/v1/quality/drift",
            files=[
                (
                    "baseline_file",
                    (
                        "baseline.csv",
                        io.BytesIO(orders_df.to_csv(index=False).encode()),
                        "text/csv",
                    ),
                ),
                (
                    "current_file",
                    ("current.csv", io.BytesIO(orders_df.to_csv(index=False).encode()), "text/csv"),
                ),
            ],
        )
        for entry in resp.json():
            assert "column" in entry
            assert "drift_detected" in entry
            assert "schema_changed" in entry
