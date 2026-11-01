"""Query-time retrieval: embed the query, search the store, optionally rerank."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from agent_eval.config import Settings
from agent_eval.embeddings import Embedder, build_embedder
from agent_eval.retrieval.base import IndexSpec, SearchHit, VectorStore
from agent_eval.retrieval.memory_store import MemoryStore
from agent_eval.retrieval.reranker import CrossEncoderReranker, LexicalReranker, Reranker


@dataclass
class RetrievalResult:
    query: str
    hits: list[SearchHit]
    latency_s: float
    reranked: bool
    candidates: list[SearchHit] = field(default_factory=list)  # before reranking


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        top_k: int = 5,
        reranker: Reranker | None = None,
        candidate_k: int = 20,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.top_k = top_k
        self.reranker = reranker
        self.candidate_k = max(candidate_k, top_k)

    def search(
        self, query: str, k: int | None = None, collections: list[str] | None = None
    ) -> RetrievalResult:
        k = k or self.top_k
        start = time.perf_counter()
        vec = self.embedder.embed_one(query)
        if self.reranker is None:
            hits = self.store.search(vec, k, collections)
            return RetrievalResult(query, hits, time.perf_counter() - start, reranked=False)
        candidates = self.store.search(vec, max(self.candidate_k, k), collections)
        hits = self.reranker.rerank(query, candidates, k)
        return RetrievalResult(
            query, hits, time.perf_counter() - start, reranked=True, candidates=candidates
        )


def index_spec(settings: Settings) -> IndexSpec:
    return IndexSpec(
        embedding_model=settings.embeddings.resolved_model,
        dimension=settings.embeddings.dimension,
        chunk_size=settings.retrieval.chunk_size,
        chunk_overlap=settings.retrieval.chunk_overlap,
    )


def build_store(settings: Settings) -> VectorStore:
    spec = index_spec(settings)
    if settings.retrieval.store == "memory":
        return MemoryStore(spec)
    from agent_eval.retrieval.pgvector_store import PgVectorStore

    return PgVectorStore(spec, settings.database.url, settings.retrieval.hnsw)


def build_reranker(settings: Settings) -> Reranker | None:
    if not settings.retrieval.rerank:
        return None
    if settings.embeddings.provider == "mock":
        return LexicalReranker()
    return CrossEncoderReranker(settings.retrieval.reranker_model)


def build_retriever(
    settings: Settings, store: VectorStore | None = None, embedder: Embedder | None = None
) -> Retriever:
    return Retriever(
        store=store or build_store(settings),
        embedder=embedder or build_embedder(settings.embeddings),
        top_k=settings.retrieval.top_k,
        reranker=build_reranker(settings),
        candidate_k=settings.retrieval.candidate_k,
    )
