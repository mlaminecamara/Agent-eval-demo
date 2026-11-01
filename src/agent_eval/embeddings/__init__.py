"""Embedding providers behind one interface, selected by config."""

from __future__ import annotations

from agent_eval.config import EmbeddingConfig
from agent_eval.embeddings.base import Embedder

__all__ = ["Embedder", "build_embedder"]


def build_embedder(cfg: EmbeddingConfig) -> Embedder:
    if cfg.provider == "mock":
        from agent_eval.embeddings.mock import MockEmbedder

        return MockEmbedder(cfg.resolved_model, cfg.dimension)
    if cfg.provider == "openai":
        from agent_eval.embeddings.openai_emb import OpenAIEmbedder

        return OpenAIEmbedder(cfg.resolved_model, cfg.dimension)
    raise ValueError(f"Unknown embedding provider {cfg.provider!r}")
