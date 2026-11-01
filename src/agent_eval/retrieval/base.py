"""Vector store interface and the index naming scheme."""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from agent_eval.ingest.chunker import Chunk


@dataclass(frozen=True)
class IndexSpec:
    """One index per embedding model and chunking config, so experiments never overwrite
    each other. Collections (corpus, adversarial) share an index and are filtered by column."""

    embedding_model: str
    dimension: int
    chunk_size: int
    chunk_overlap: int

    @property
    def table_name(self) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", self.embedding_model.lower()).strip("_")
        name = f"chunks_{slug}_c{self.chunk_size}_o{self.chunk_overlap}"
        if len(name) > 63:  # PostgreSQL identifier limit
            digest = hashlib.sha1(name.encode()).hexdigest()[:8]
            name = f"{name[:54]}_{digest}"
        return name


@dataclass
class SearchHit:
    chunk: Chunk
    score: float  # cosine similarity (higher is closer), or reranker score after reranking
    rank: int  # 1-based


class VectorStore(ABC):
    def __init__(self, spec: IndexSpec) -> None:
        self.spec = spec

    @abstractmethod
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    @abstractmethod
    def delete_docs(self, doc_ids: list[str]) -> None: ...

    @abstractmethod
    def search(
        self, query_vector: list[float], k: int, collections: list[str] | None = None
    ) -> list[SearchHit]: ...

    @abstractmethod
    def count(self, collections: list[str] | None = None) -> int: ...
