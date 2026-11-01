# agent-eval-demo

RAG and agentic workflows over the French e-invoicing reform (facturation électronique), with
an evaluation harness for accuracy, safety, latency and cost.

**Status: work in progress (milestone 2 of 8: ingestion and retrieval).** No evaluation has
been run yet; there are no results to report.

## Quick start (offline, no API key)

```bash
uv sync
make test
```

Everything runs in mock mode by default (deterministic fake LLM and embeddings). Copy
`.env.example` to `.env` and set `LLM_PROVIDER` / `EMBEDDING_PROVIDER` and API keys for real runs.

## Vector database

- **Hosted (real runs): Supabase** (PostgreSQL with the `vector` extension). Put the
  "Session pooler" connection string in `DATABASE_URL` in `.env`.
- **Local and CI:** `make up` starts `pgvector/pgvector:pg16` from `docker-compose.yml`; CI uses
  the same image as a service container.

`make ingest` creates one table per embedding model and chunk config
(e.g. `chunks_text_embedding_3_small_c500_o50`) with an HNSW index on cosine distance
(`m=16`, `ef_construction=64`, `ef_search=40`).

The corpus is listed in [data/SOURCES.md](data/SOURCES.md). The full README (architecture,
design decisions, generated results, limitations) is written in milestone 8.
