# API image: FastAPI app serving POST /ask (agent_eval.api:app).
#
#   docker build -t agent-eval-demo .
#   # offline demo (fake LLM, in-memory index built at startup, no key):
#   docker run --rm -p 8000:8000 -e LLM_PROVIDER=mock -e EMBEDDING_PROVIDER=mock \
#       -e VECTOR_STORE=memory agent-eval-demo
#   # real models, existing pgvector index (secrets passed at run time, never baked in):
#   docker run --rm -p 8000:8000 --env-file .env agent-eval-demo
#
# The image holds the code, config.yaml and data/ (corpus + deadline table). It has no
# secrets: .env is excluded by .dockerignore. The eval harness and tests are not included.

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependencies first (cached layer), from the lock file only.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Project. Installed editable on purpose: the code finds config.yaml and data/ relative to
# the project root (agent_eval.config.PROJECT_ROOT).
COPY config.yaml ./
COPY config ./config
COPY data ./data
COPY src ./src
RUN uv sync --frozen --no-dev \
    && useradd --create-home --uid 10001 app \
    && mkdir -p runs && chown app runs

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "agent_eval.api:app", "--host", "0.0.0.0", "--port", "8000"]
