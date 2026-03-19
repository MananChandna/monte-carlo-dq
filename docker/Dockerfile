# docker/Dockerfile
# ---------------------------------------------------------------------------
# Stage 1: builder — install dependencies into a venv
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel
RUN pip install --upgrade pip wheel

# Copy dependency manifest first for layer caching
COPY pyproject.toml ./

# Install project dependencies into an isolated prefix
RUN pip install --prefix=/install \
        pandas \
        numpy \
        scipy \
        scikit-learn \
        fastapi \
        "uvicorn[standard]" \
        python-multipart \
        pydantic \
        pydantic-settings \
        sqlalchemy \
        asyncpg \
        alembic

# ---------------------------------------------------------------------------
# Stage 2: runtime — lean final image
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL maintainer="Monte Carlo DQ Team"
LABEL description="Monte Carlo Data Quality API"

# Runtime system deps (libpq for asyncpg)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY src/ ./src/
COPY data/ ./data/

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
