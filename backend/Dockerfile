# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS base

ENV UV_CACHE_DIR=/root/.cache/uv \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app


FROM base AS builder

COPY pyproject.toml ./

RUN uv sync


FROM base AS runtime

ENV PYTHONPATH="/app/src:${PYTHONPATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libgomp1 \
    libopenblas0 \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv

COPY . .

RUN groupadd --system appuser \
    && useradd --system --gid appuser --home /app appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

WORKDIR /app/src
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
