# syntax=docker/dockerfile:1

# ---- Stage 1: build the dashboard (Vite + React) --------------------------
FROM node:22-alpine AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# ---- Stage 2: Python runtime (FastAPI serves the API and the SPA) ---------
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Install the package (numpy/scipy ship manylinux wheels, no compiler needed).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# api.py resolves the SPA at <package-root>/web/dist, i.e. /app/web/dist.
COPY --from=web /web/dist ./web/dist

EXPOSE 8000

# Bind 0.0.0.0 so the platform router can reach it; honor $PORT when injected.
CMD ["sh", "-c", "uvicorn agenteval.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
