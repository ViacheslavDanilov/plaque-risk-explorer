FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Build tools required by statsforecast (autogluon transitive dep)
RUN apt-get update && \
    apt-get install -y --no-install-recommends g++ && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install dependencies (cached layer)
COPY pyproject.toml ./
RUN uv sync --no-dev --no-install-project

# Copy application code
COPY src/ src/
COPY models/ models/
COPY data/features.csv data/features.csv

# Install the project itself
RUN uv sync --no-dev

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "plaque_risk_explorer.main:app", "--host", "0.0.0.0", "--port", "8000"]
