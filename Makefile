.PHONY: help install up down ingest test test-integration lint typecheck eval eval-mock compare report serve

UV ?= uv
CHUNK ?= 500
OVERLAP ?= 50

help:
	@echo "install    install dependencies (uv sync)"
	@echo "up / down  start / stop PostgreSQL + pgvector (docker compose)"
	@echo "ingest     chunk, embed and index data/corpus (CHUNK=500 OVERLAP=50)"
	@echo "test       run the test suite (mock mode, no API key; DB tests skip if no DB)"
	@echo "test-integration  pgvector tests only (needs make up)"
	@echo "lint       ruff + mypy"
	@echo "eval       real evaluation run (asks for cost confirmation; ARGS=...)"
	@echo "eval-mock  offline evaluation run with fake LLM/embeddings/judge"
	@echo "compare    compare result files into evals/results/summary.md [milestone 7]"
	@echo "report     inject the generated tables into README.md        [milestone 7]"
	@echo "serve      run the FastAPI app                               [milestone 7]"

install:
	$(UV) sync

up:
	docker compose up -d --wait

down:
	docker compose down

test:
	LLM_PROVIDER=mock EMBEDDING_PROVIDER=mock $(UV) run pytest

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	$(UV) run mypy

typecheck:
	$(UV) run mypy

ingest:
	$(UV) run python -m agent_eval.ingest --chunk-size $(CHUNK) --overlap $(OVERLAP)

test-integration:
	RUN_INTEGRATION=1 LLM_PROVIDER=mock EMBEDDING_PROVIDER=mock $(UV) run pytest -m integration

# Real run: prints the estimated cost and asks for confirmation. Extra flags via ARGS, e.g.
#   make eval ARGS="--provider anthropic --chunk-size 1000 --overlap 100"
eval:
	$(UV) run python -m evals.run_eval $(ARGS)

eval-mock:
	$(UV) run python -m evals.run_eval --mock $(ARGS)

compare report serve:
	@echo "'make $@' is not implemented yet (see the milestone noted in 'make help')." && exit 1
