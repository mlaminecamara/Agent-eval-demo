"""Chunk, embed and index documents. Re-running replaces the documents' previous chunks."""

from __future__ import annotations

from dataclasses import dataclass

from agent_eval.embeddings import Embedder
from agent_eval.ingest.chunker import Chunk, chunk_documents, count_tokens
from agent_eval.ingest.loader import Document
from agent_eval.observability.logging import get_logger
from agent_eval.retrieval.base import VectorStore

log = get_logger(__name__)


@dataclass
class IndexReport:
    table: str
    documents: int
    chunks: int
    embedding_tokens: int
    per_document: dict[str, int]


def plan_chunks(docs: list[Document], chunk_size: int, overlap: int) -> list[Chunk]:
    return chunk_documents(docs, chunk_size, overlap)


def embedding_token_estimate(chunks: list[Chunk]) -> int:
    return sum(count_tokens(c.embedding_text()) for c in chunks)


def index_chunks(
    chunks: list[Chunk], store: VectorStore, embedder: Embedder, batch_size: int = 64
) -> IndexReport:
    doc_ids = sorted({c.doc_id for c in chunks})
    store.delete_docs(doc_ids)
    tokens_before = embedder.tokens_used
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embedder.embed([c.embedding_text() for c in batch])
        store.upsert(batch, vectors)
        log.info("indexed batch", extra={"batch_start": start, "batch_size": len(batch)})
    per_doc: dict[str, int] = {}
    for c in chunks:
        per_doc[c.doc_id] = per_doc.get(c.doc_id, 0) + 1
    return IndexReport(
        table=store.spec.table_name,
        documents=len(doc_ids),
        chunks=len(chunks),
        embedding_tokens=embedder.tokens_used - tokens_before,
        per_document=per_doc,
    )
