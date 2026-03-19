# Monte Carlo Data Quality

[![CI](https://github.com/YOUR_USERNAME/monte-carlo-dq/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/monte-carlo-dq/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/YOUR_USERNAME/monte-carlo-dq/branch/main/graph/badge.svg)](https://codecov.io/gh/YOUR_USERNAME/monte-carlo-dq)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Production-grade Monte Carlo simulation engine for data quality monitoring.**
> Bootstrap-based confidence intervals, drift detection, anomaly scoring, and a
> full REST API — all containerised and CI-ready.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client (HTTP)                            │
└───────────────────────────┬─────────────────────────────────────┘
                            │  REST / JSON
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI  (port 8000)                        │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │  POST /run       │  │  GET  /history   │  │  GET /health │  │
│  │  POST /profile   │  │  POST /drift     │  │              │  │
│  └────────┬─────────┘  └────────┬─────────┘  └──────────────┘  │
└───────────┼─────────────────────┼────────────────────────────────┘
            │                     │
            ▼                     ▼
┌───────────────────────┐  ┌─────────────────────────────────────┐
│   Monte Carlo Core    │  │         In-Memory / PostgreSQL       │
│                       │  │          (Run History Store)         │
│  ┌─────────────────┐  │  └─────────────────────────────────────┘
│  │  simulation.py  │  │
│  │  (Bootstrap CI, │  │
│  │   p-values)     │  │
│  └────────┬────────┘  │
│           │           │
│  ┌────────▼────────┐  │
│  │   profiler.py   │  │
│  │  (KS test, JS   │  │
│  │   divergence)   │  │
│  └────────┬────────┘  │
│           │           │
│  ┌────────▼────────┐  │
│  │  detectors.py   │  │
│  │  (Z-score, IQR, │  │
│  │  IsoForest,     │  │
│  │  Volume, Fresh) │  │
│  └─────────────────┘  │
└───────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PostgreSQL 16  (port 5432)                    │
│            [Run history · Profiles · Drift reports]             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/monte-carlo-dq.git
cd monte-carlo-dq

# 2. Copy environment config
cp .env.example .env

# 3. Install dev dependencies + pre-commit hooks
make install-dev

# 4. Generate seed data and run tests
make seed-data && make test

# 5. Start the full stack (API + PostgreSQL)
make docker-up
```

> API docs: <http://localhost:8000/docs>

---

## Project Structure

```
monte-carlo-dq/
├── .github/workflows/ci.yml      # GitHub Actions CI pipeline
├── src/
│   ├── api/
│   │   ├── main.py               # FastAPI app factory
│   │   └── routes/
│   │       ├── health.py         # GET /health
│   │       └── quality.py        # POST /run, /profile, /drift; GET /history
│   ├── core/
│   │   ├── simulation.py         # Monte Carlo engine (bootstrap CI)
│   │   ├── profiler.py           # Statistical profiling + drift detection
│   │   └── detectors.py          # Anomaly detection (Z-score, IQR, IF)
│   ├── models/schemas.py         # Pydantic v2 request/response schemas
│   ├── db/connection.py          # Async SQLAlchemy engine & session
│   └── config.py                 # pydantic-settings configuration
├── tests/                        # pytest test suite (>80% coverage target)
├── data/samples/                 # Synthetic seed CSV datasets
├── notebooks/exploration.ipynb   # Interactive demo notebook
├── docker/Dockerfile             # Multi-stage Docker build
├── docker-compose.yml            # PostgreSQL + API services
├── pyproject.toml                # Pinned dependencies + tool config
├── .pre-commit-config.yaml       # black, flake8, isort, mypy hooks
├── Makefile                      # Developer shortcuts
└── README.md
```

---

## API Reference

All endpoints are prefixed with `/api/v1`.

### `POST /quality/run`

Run a full Monte Carlo DQ simulation on an uploaded file.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | `UploadFile` | required | CSV or JSON dataset |
| `dataset_name` | `str` | `"dataset"` | Logical dataset name |
| `n_simulations` | `int` | `1000` | Bootstrap iterations (100–100 000) |
| `quality_threshold` | `float` | `0.8` | Minimum passing score |
| `key_columns` | `str` | `null` | Comma-separated columns for uniqueness |
| `timestamp_col` | `str` | `null` | Column for timeliness dimension |
| `sample_fraction` | `float` | `0.8` | Bootstrap sample fraction |

**Response** `200 OK`
```json
{
  "n_simulations": 1000,
  "overall_score": 0.9312,
  "passed": true,
  "threshold": 0.8,
  "dimensions": [
    {
      "dimension": "completeness",
      "mean_score": 0.971,
      "std_dev": 0.008,
      "p5": 0.957,
      "p95": 0.984,
      "p_value": 0.0001,
      "observed_score": 0.970
    }
  ]
}
```

---

### `POST /quality/profile`

Generate a column-level statistical profile.

| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | `UploadFile` | CSV or JSON dataset |

---

### `POST /quality/drift`

Detect distributional drift between a baseline and current dataset.

| Parameter | Type | Description |
|-----------|------|-------------|
| `baseline_file` | `UploadFile` | Historical reference dataset |
| `current_file` | `UploadFile` | Current dataset to compare |

---

### `GET /quality/history`

Paginated run history.

| Query param | Type | Default |
|-------------|------|---------|
| `limit` | `int` | `50` |
| `offset` | `int` | `0` |
| `dataset_name` | `str` | `null` |

---

### `GET /health`

```json
{ "status": "ok", "version": "0.1.0", "timestamp": "2024-07-01T12:00:00" }
```

---

## Quality Dimensions

| Dimension | Method | Description |
|-----------|--------|-------------|
| **Completeness** | Non-null fraction | What fraction of cells are populated? |
| **Uniqueness** | Deduplication rate | What fraction of key-column rows are unique? |
| **Validity** | Rule evaluation | What fraction of values pass domain rules? |
| **Timeliness** | Timestamp delta | What fraction of records are within `max_age_hours`? |

Each dimension score is estimated via **10 000 bootstrap samples** (configurable) to produce a mean, standard deviation, 5th/95th percentile CI, and a t-test p-value against the null hypothesis of perfect quality (μ = 1.0).

---

## Drift Detection

| Column type | Test | Signal |
|-------------|------|--------|
| Numeric | Kolmogorov-Smirnov | `p_value < 0.05` |
| Categorical | Jensen-Shannon divergence | `js_div > 0.1` |
| Any | dtype check | dtype changed between snapshots |

---

## Development

```bash
make install-dev     # install all deps + pre-commit hooks
make format          # black + isort (auto-fix)
make lint            # flake8 + mypy + isort (check-only)
make test            # pytest with coverage
make test-fast       # pytest, no coverage (faster iteration)
make seed-data       # generate synthetic CSVs in data/samples/
make run             # uvicorn hot-reload dev server
make docker-up       # start Postgres + API in Docker
make docker-down     # tear down containers
make clean           # remove all generated artifacts
```

---

## CI/CD Pipeline

The GitHub Actions pipeline (`.github/workflows/ci.yml`) runs on every push
and pull request to `main`:

```
push / PR ──► lint (3.11, 3.12)
                    │
                    ▼
             test (3.11, 3.12)
             ├── pytest + coverage
             └── upload to Codecov
                    │
                    ▼
           build-docker (multi-stage)
```

---

## Contributing

1. Fork the repository and create a feature branch:
   ```bash
   git checkout -b feat/your-feature
   ```
2. Install dev dependencies: `make install-dev`
3. Make your changes — all functions need type hints, Google-style docstrings,
   and at least one unit test.
4. Run the full suite: `make lint && make test`
5. Commit using [Conventional Commits](https://www.conventionalcommits.org/):
   ```
   feat(core): add Chi-squared test for categorical validity
   fix(api): handle empty CSV uploads gracefully
   ```
6. Open a pull request — CI must be green before review.

---

## License

MIT © Your Name
