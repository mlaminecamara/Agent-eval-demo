"""Exact (brute-force) cosine search in memory. Used by unit tests and as a reference to check
the approximate HNSW results from pgvector."""

from __future__ import annotations

from agent_eval.ingest.chunker import Chunk
from agent_eval.retrieval.base import IndexSpec, SearchHit, VectorStore


class MemoryStore(VectorStore):
    def __init__(self, spec: IndexSpec) -> None:
        super().__init__(spec)
        self._rows: dict[str, tuple[Chunk, list[float]]] = {}

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        for chunk, vec in zip(chunks, vectors, strict=True):
            if len(vec) != self.spec.dimension:
                raise ValueError(f"vector dimension {len(vec)} != {self.spec.dimension}")
            self._rows[chunk.chunk_id] = (chunk, vec)

    def delete_docs(self, doc_ids: list[str]) -> None:
        drop = set(doc_ids)
        self._rows = {k: v for k, v in self._rows.items() if v[0].doc_id not in drop}

    def search(
        self, query_vector: list[float], k: int, collections: list[str] | None = None
    ) -> list[SearchHit]:
        # Vectors are L2-normalised, so the dot product is the cosine similarity.
        scored = [
            (sum(a * b for a, b in zip(query_vector, vec, strict=True)), chunk)
            for chunk, vec in self._rows.values()
            if collections is None or chunk.collection in collections
        ]
        scored.sort(key=lambda t: (-t[0], t[1].chunk_id))
        return [SearchHit(chunk=c, score=s, rank=i) for i, (s, c) in enumerate(scored[:k], 1)]

    def count(self, collections: list[str] | None = None) -> int:
        return sum(
            1 for c, _ in self._rows.values() if collections is None or c.collection in collections
        )
