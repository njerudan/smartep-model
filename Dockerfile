# SMARTEP API: ingest webhook + worker + gap model v0.1 (models/gap-model), CPU only.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN pip install --no-cache-dir uv==0.13.9

WORKDIR /app
# Dependencies first (cached layer): core + serve extra only; no torch, no MiniLM.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project --extra serve

COPY smartep ./smartep
COPY pipelines ./pipelines
COPY models ./models
RUN uv sync --locked --extra serve

RUN useradd --system --uid 10001 --home-dir /app smartep && mkdir -p /data && chown smartep /data
USER smartep

ENV SMARTEP_MODEL_DIR=/app/models/gap-model \
    SMARTEP_DB_PATH=/data/smartep.db \
    PATH=/opt/venv/bin:$PATH
EXPOSE 8000
VOLUME ["/data"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# One process: the worker thread shares the loaded model with the web server.
CMD ["uvicorn", "smartep.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--no-server-header"]
