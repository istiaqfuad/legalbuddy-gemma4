FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install dependencies ONLY (no workspace source).
# This layer is cached and only rebuilds when manifests/lockfile change,
# so editing Python source no longer re-downloads/recompiles torch et al.
# All workspace member manifests are needed for uv to resolve the graph.
COPY pyproject.toml uv.lock ./
COPY apps/api/pyproject.toml ./apps/api/pyproject.toml
COPY apps/shared/pyproject.toml ./apps/shared/pyproject.toml
COPY apps/ingestion/pyproject.toml ./apps/ingestion/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace --package api

# Copy source for api + its editable workspace dep (shared), then install the
# api package itself (fast — external deps already present).
COPY apps/api ./apps/api
COPY apps/shared ./apps/shared
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package api

# Enable bytecode compilation and Python optimization
ENV PYTHONOPTIMIZE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/apps/api/src:$PYTHONPATH"

# HuggingFace/sentence-transformers cache the model under $HOME at runtime.
# A real, writable HOME is required or model download fails with
# "Permission denied: '/nonexistent'".
ENV HOME="/home/app"
ENV HF_HOME="/home/app/.cache/huggingface"

# Create non-root user with a writable home + cache dir
RUN addgroup --system app && \
    adduser --system --ingroup app --home /home/app app && \
    mkdir -p /home/app/.cache/huggingface && \
    chown -R app:app /home/app

# Switch to non-root user
USER app

# Expose the FastAPI port
EXPOSE 8000

WORKDIR /app/apps/api/src

# Command to run the application
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]