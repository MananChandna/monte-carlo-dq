# Makefile — developer ergonomics for Monte Carlo DQ
.DEFAULT_GOAL := help
SHELL         := /bin/bash
PYTHON        := python3
PIP           := pip3
APP_MODULE    := src.api.main:app
DOCKER_COMPOSE := docker compose

.PHONY: help install install-dev lint format test test-fast run \
        docker-up docker-down docker-logs docker-build \
        seed-data clean clean-pyc clean-cov

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------
install: ## Install production dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -e .

install-dev: ## Install all dev + production dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev,notebooks]"
	pre-commit install
	@echo "✅  Dev environment ready"

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------
lint: ## Run flake8, mypy, isort (check-only)
	flake8 src/ tests/
	mypy src/
	isort --check-only src/ tests/

format: ## Auto-format code with black and isort
	black src/ tests/ data/
	isort src/ tests/ data/

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
test: ## Run full test suite with coverage
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=xml

test-fast: ## Run tests without coverage (faster)
	pytest tests/ -v -x --no-cov

test-unit: ## Run only unit tests (no API tests)
	pytest tests/test_simulation.py tests/test_profiler.py tests/test_detectors.py -v

# ---------------------------------------------------------------------------
# Local development server
# ---------------------------------------------------------------------------
run: ## Start the FastAPI development server with hot-reload
	uvicorn $(APP_MODULE) --reload --host 0.0.0.0 --port 8000

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
docker-build: ## Build the Docker image
	$(DOCKER_COMPOSE) build

docker-up: ## Start all services (PostgreSQL + API) in the background
	$(DOCKER_COMPOSE) up -d
	@echo "🚀  Services started — API at http://localhost:8000/docs"

docker-down: ## Stop and remove all services
	$(DOCKER_COMPOSE) down

docker-logs: ## Tail logs from all running containers
	$(DOCKER_COMPOSE) logs -f

docker-shell: ## Open a shell in the running API container
	$(DOCKER_COMPOSE) exec api /bin/bash

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
seed-data: ## Generate synthetic seed datasets in data/samples/
	$(PYTHON) data/samples/generate_seed_data.py
	@echo "✅  Seed data generated in data/samples/"

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
clean-pyc: ## Remove Python bytecode files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

clean-cov: ## Remove coverage artifacts
	rm -rf .coverage coverage.xml htmlcov/ .pytest_cache/

clean: clean-pyc clean-cov ## Remove all generated artifacts
	rm -rf dist/ build/ .mypy_cache/
	@echo "✅  Clean complete"
