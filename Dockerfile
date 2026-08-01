# ==============================================================================
# The Kairos Engine — Multi-Stage Docker Build
# Stage 1: dependency builder, Stage 2: minimal runtime
# Models are mounted as volumes, never baked into the image.
# ==============================================================================

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Install system dependencies for llama-cpp-python compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make cmake \
    libopenblas-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements early for better layer caching
COPY requirements.txt .

# Install Python dependencies into /usr/local (no venv needed in container)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Install minimal runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash kairos

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code
COPY src/ src/
COPY tests/ tests/
COPY main.py build_crash_predictor.py run_tests.py ./
COPY README.md pyproject.toml requirements.txt LICENSE ./

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV KAIROS_LOG_LEVEL=INFO

# Create directories
RUN mkdir -p /app/models /app/logs && \
    chown -R kairos:kairos /app

USER kairos

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src.config import ENGINE_NAME; print(ENGINE_NAME)" || exit 1

# Models are mounted as a volume at runtime:
# docker run -v /path/to/models:/app/models kairos-engine
VOLUME ["/app/models", "/app/logs"]

# Default: run the engine (requires model mounted)
CMD ["python", "main.py"]

# Alternative entry points:
# docker run kairos-engine python build_crash_predictor.py  (train model)
# docker run kairos-engine python run_tests.py              (run tests)
