import os
import uuid

import pytest

from agent_eval.config import PROJECT_ROOT, HNSWConfig
from agent_eval.embeddings import build_embedder
from agent_eval.ingest.chunker import chunk_documents
from agent_eval.ingest.indexer import index_chunks
from agent_eval.ingest.loader import Block, Document, load_manifest
from agent_eval.retrieval.base import IndexSpec
from agent_eval.retrieval.memory_store import MemoryStore
from agent_eval.retrieval.reranker import LexicalReranker
from agent_eval.retrieval.retriever import Retriever, build_retriever

MANIFEST = PROJECT_ROOT / "data" / "corpus" / "manifest.yaml"


@pytest.fixture(scope="module")
def corpus_chunks():
    return chunk_documents(load_manifest(MANIFEST), chunk_size=500, overlap=50)


@pytest.fixture
def memory_retriever(mock_settings, corpus_chunks):
    embedder = build_embedder(mock_settings.embeddings)
    spec = IndexSpec(embedder.model, embedder.dimension, 500, 50)
    store = MemoryStore(spec)
    index_chunks(corpus_chunks, store, embedder)
    return Retriever(store, embedder, top_k=5)


def test_table_name_is_a_safe_identifier():
    spec = IndexSpec("text-embedding-3-small", 1536, 500, 50)
    assert spec.table_name == "chunks_text_embedding_3_small_c500_o50"
    long = IndexSpec("x" * 100, 8, 500, 50)
    assert len(long.table_name) <= 63


def test_memory_search_finds_relevant_document(memory_retriever):
    # Mock embeddings are lexical, so the query reuses distinctive words of the target page.
    result = memory_retriever.search(
        "Service d'Immatriculation : dossier de candidature déposé par un opérateur"
    )
    assert result.hits[0].chunk.doc_id == "plateformes-agreees"
    assert [h.rank for h in result.hits] == [1, 2, 3, 4, 5]
    assert result.hits[0].score >= result.hits[-1].score


def test_reindexing_replaces_chunks(mock_settings, corpus_chunks, memory_retriever):
    store = memory_retriever.store
    before = store.count()
    index_chunks(corpus_chunks, store, memory_retriever.embedder)
    assert store.count() == before


def test_collection_filter(mock_settings, memory_retriever):
    adv = Document(
        doc_id="adv-test",
        title="Document adverse (test)",
        source_file="adv.md",
        collection="adversarial",
        blocks=[Block("plateforme agréée immatriculée DGFiP rôle", None, None)],
    )
    index_chunks(chunk_documents([adv], 500, 50), memory_retriever.store, memory_retriever.embedder)
    corpus_only = memory_retriever.search("plateforme agréée immatriculée", collections=["corpus"])
    assert all(h.chunk.collection == "corpus" for h in corpus_only.hits)
    both = memory_retriever.search("plateforme agréée immatriculée DGFiP rôle", k=3)
    assert both.hits[0].chunk.doc_id == "adv-test"


def test_reranker_reorders_candidates(memory_retriever):
    memory_retriever.reranker = LexicalReranker()
    result = memory_retriever.search("liste des plateformes agréées", k=3)
    assert result.reranked
    assert len(result.candidates) >= len(result.hits) == 3
    assert result.hits[0].score >= result.hits[-1].score


# --- pgvector integration ------------------------------------------------------------------


def _db_or_skip(dsn: str):
    psycopg = pytest.importorskip("psycopg")
    try:
        psycopg.connect(dsn, connect_timeout=3).close()
    except psycopg.OperationalError as exc:
        if os.environ.get("RUN_INTEGRATION") == "1":
            raise
        pytest.skip(f"PostgreSQL not reachable ({exc.__class__.__name__}); run `make up`")


@pytest.fixture
def pg_store(mock_settings):
    from dotenv import dotenv_values

    from agent_eval.config import PROJECT_ROOT
    from agent_eval.retrieval.pgvector_store import PgVectorStore

    # Environment first (CI), then .env (local Supabase / Docker), then config.yaml.
    dsn = (
        os.environ.get("DATABASE_URL")
        or dotenv_values(PROJECT_ROOT / ".env").get("DATABASE_URL")
        or mock_settings.database.url
    )
    _db_or_skip(dsn)
    # Unique model name -> unique table, so tests never touch real indexes.
    spec = IndexSpec(f"test-{uuid.uuid4().hex[:8]}", mock_settings.embeddings.dimension, 500, 50)
    store = PgVectorStore(spec, dsn, HNSWConfig(m=16, ef_construction=64, ef_search=40))
    yield store
    store.connect().execute(f'DROP TABLE IF EXISTS "{spec.table_name}"')
    store.close()


@pytest.mark.integration
def test_pgvector_hnsw_index_and_results_match_exact_search(
    mock_settings, corpus_chunks, pg_store, memory_retriever
):
    embedder = memory_retriever.embedder
    report = index_chunks(corpus_chunks, pg_store, embedder)
    assert pg_store.count() == report.chunks == len(corpus_chunks)

    indexdef = pg_store.index_definition() or ""
    assert "USING hnsw" in indexdef and "vector_cosine_ops" in indexdef
    assert "m='16'" in indexdef and "ef_construction='64'" in indexdef

    pg_retriever = Retriever(pg_store, embedder, top_k=5)
    for query in [
        "rôle d'une plateforme agréée immatriculée",
        "calendrier de l'obligation de facturation électronique pour une PME",
        "transmission des données de transaction e-reporting",
    ]:
        exact = [h.chunk.chunk_id for h in memory_retriever.search(query).hits]
        approx = pg_retriever.search(query)
        got = [h.chunk.chunk_id for h in approx.hits]
        assert got[0] == exact[0]
        assert len(set(got) & set(exact)) >= 4
        assert approx.hits[0].chunk.title  # metadata round-trips


@pytest.mark.integration
def test_pgvector_collection_filter_and_reindex(mock_settings, corpus_chunks, pg_store):
    embedder = build_embedder(mock_settings.embeddings)
    index_chunks(corpus_chunks, pg_store, embedder)
    adv = Document(
        "adv-test",
        "Adverse",
        "adv.md",
        collection="adversarial",
        blocks=[Block("plateforme agréée immatriculée", None, None)],
    )
    index_chunks(chunk_documents([adv], 500, 50), pg_store, embedder)
    assert pg_store.count(["adversarial"]) == 1
    hits = pg_store.search(embedder.embed_one("plateforme agréée"), 5, ["corpus"])
    assert len(hits) == 5 and all(h.chunk.collection == "corpus" for h in hits)
    index_chunks(corpus_chunks, pg_store, embedder)  # idempotent
    assert pg_store.count(["corpus"]) == len(corpus_chunks)


def test_build_retriever_from_settings(mock_settings):
    mock_settings.retrieval.store = "memory"
    mock_settings.retrieval.rerank = True
    r = build_retriever(mock_settings)
    assert isinstance(r.store, MemoryStore)
    assert isinstance(r.reranker, LexicalReranker)
