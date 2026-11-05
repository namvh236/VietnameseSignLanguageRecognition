# Single image serves both the AI core (src/) and the serving layer (serving/).
# Build context is this directory (backend/).
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3-pip libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# uv syncs deps; the torch cu126 index is declared in pyproject.toml.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install deps first to leverage Docker layer caching.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Host .venv is excluded via .dockerignore; the container builds its own.
COPY . /app

EXPOSE 8000
# recognition_root defaults to /app; no extra ENV needed.
CMD ["/app/.venv/bin/python", "-m", "uvicorn", "serving.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]
