.PHONY: help install test test-cov lint typecheck format build run train-model clean docker-build docker-run

# ── Variables ─────────────────────────────────────────────────────────────────
PYTHON     := python
PIP        := pip
PYTEST     := pytest
IMAGE_NAME := kairos-engine
IMAGE_TAG  := latest

# Default target
help:
	@echo "=================================================================="
	@echo "  THE KAIROS ENGINE — Development Makefile"
	@echo "=================================================================="
	@echo "  install       Install all dependencies"
	@echo "  test          Run full test suite"
	@echo "  test-cov      Run tests with HTML coverage report"
	@echo "  lint          Run ruff linter"
	@echo "  typecheck     Run mypy static type checker"
	@echo "  format        Auto-format with ruff"
	@echo "  run           Launch the Kairos Engine"
	@echo "  train-model   Build/retrain XGBoost crash predictor"
	@echo "  clean         Remove caches, logs, bytecode"
	@echo "  docker-build  Build Docker image"
	@echo "  docker-run    Run Docker container (requires model volume)"
	@echo "=================================================================="

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "[OK] Dependencies installed."

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -v --tb=short -x

test-cov:
	$(PYTHON) -m pytest tests/ -v --tb=short \
		--cov=src --cov-report=term-missing --cov-report=html:htmlcov
	@echo "[OK] Coverage report -> htmlcov/index.html"

test-fast:
	$(PYTHON) -m pytest tests/ -v --tb=short -x \
		--ignore=tests/test_risk_model.py \
		--ignore=tests/test_integration.py
	@echo "[OK] Fast tests (no model required) complete."

# ── Code Quality ──────────────────────────────────────────────────────────────
lint:
	$(PYTHON) -m ruff check src/ tests/ main.py build_crash_predictor.py

typecheck:
	$(PYTHON) -m mypy src/ --ignore-missing-imports --no-strict-optional

format:
	$(PYTHON) -m ruff format src/ tests/ main.py build_crash_predictor.py
	@echo "[OK] Code formatted."

# ── Runtime ───────────────────────────────────────────────────────────────────
run:
	$(PYTHON) main.py

run-verbose:
	$(PYTHON) main.py --verbose

run-preonly:
	$(PYTHON) main.py --no-inflight

# ── ML Pipeline ───────────────────────────────────────────────────────────────
train-model:
	$(PYTHON) build_crash_predictor.py
	@echo "[OK] Model trained -> kairos_crash_predictor.json"

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	rm -rf logs/*.log logs/*.jsonl 2>/dev/null || true
	@echo "[OK] Clean complete."

clean-model:
	rm -f kairos_crash_predictor.json kairos_crash_predictor_metadata.json
	@echo "[WARN] Model files deleted. Run 'make train-model' to rebuild."

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .
	@echo "[OK] Docker image: $(IMAGE_NAME):$(IMAGE_TAG)"

docker-run:
	docker run --rm \
		-v "$(PWD)/models:/app/models" \
		-v "$(PWD)/logs:/app/logs" \
		$(IMAGE_NAME):$(IMAGE_TAG)

docker-test:
	docker run --rm \
		-v "$(PWD)/models:/app/models" \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		python run_tests.py

# ── CI shortcuts ──────────────────────────────────────────────────────────────
ci: lint test
	@echo "[OK] CI checks passed."
