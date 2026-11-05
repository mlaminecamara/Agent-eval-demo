"""Open the search index for a configuration: shared by the eval harness and the API."""

from __future__ import annotations

from pathlib import Path

from agent_eval.config import Settings
from agent_eval.embeddings import Embedder, build_embedder
from agent_eval.ingest.chunker import chunk_documents
from agent_eval.ingest.indexer import index_chunks
from agent_eval.ingest.loader import load_manifest
from agent_eval.retrieval.base import VectorStore
from agent_eval.retrieval.memory_store import MemoryStore
from agent_eval.retrieval.retriever import build_store, index_spec


def manifests(settings: Settings) -> list[Path]:
    paths = [settings.resolve_path(settings.corpus.manifest)]
    adv = settings.resolve_path(settings.corpus.adversarial_dir) / "manifest.yaml"
    if adv.exists():
        paths.append(adv)
    return paths


def build_index(settings: Settings, needs_adversarial: bool) -> tuple[VectorStore, Embedder]:
    """Memory store: index in-process. pgvector: use the existing index (make ingest)."""
    embedder = build_embedder(settings.embeddings)
    if settings.retrieval.store == "memory":
        store: VectorStore = MemoryStore(index_spec(settings))
        docs = [d for m in manifests(settings) for d in load_manifest(m)]
        chunks = chunk_documents(
            docs, settings.retrieval.chunk_size, settings.retrieval.chunk_overlap
        )
        index_chunks(chunks, store, embedder)
        return store, embedder
    store = build_store(settings)
    n_corpus = store.count(["corpus"])
    n_adv = store.count(["adversarial"])
    if n_corpus == 0 or (needs_adversarial and n_adv == 0):
        raise SystemExit(
            f"Index {store.spec.table_name} has {n_corpus} corpus / {n_adv} adversarial chunks. "
            f"Run: make ingest CHUNK={settings.retrieval.chunk_size} "
            f"OVERLAP={settings.retrieval.chunk_overlap}"
        )
    return store, embedder
