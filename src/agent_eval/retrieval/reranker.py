"""Optional second-stage reranking of retrieved candidates.

- CrossEncoderReranker: local cross-encoder (default BAAI/bge-reranker-v2-m3, multilingual).
  Needs the optional extra: `uv sync --extra rerank` (installs sentence-transformers + torch).
- LexicalReranker: deterministic word-overlap scorer used in mock mode and tests.
"""

from __future__ import annotations

import re
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Any

from agent_eval.retrieval.base import SearchHit

_WORD = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set[str]:
    norm = unicodedata.normalize("NFKD", text.lower())
    norm = "".join(c for c in norm if not unicodedata.combining(c))
    return {w for w in _WORD.findall(norm) if len(w) > 2}


class Reranker(ABC):
    name: str

    @abstractmethod
    def scores(self, query: str, texts: list[str]) -> list[float]: ...

    def rerank(self, query: str, hits: list[SearchHit], k: int) -> list[SearchHit]:
        if not hits:
            return []
        scores = self.scores(query, [h.chunk.embedding_text() for h in hits])
        order = sorted(range(len(hits)), key=lambda i: (-scores[i], hits[i].rank))
        return [
            replace(hits[i], score=float(scores[i]), rank=r) for r, i in enumerate(order[:k], 1)
        ]


class LexicalReranker(Reranker):
    name = "lexical"

    def scores(self, query: str, texts: list[str]) -> list[float]:
        q = _words(query)
        return [len(q & _words(t)) / (len(q) or 1) for t in texts]


class CrossEncoderReranker(Reranker):
    name = "cross-encoder"

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "Reranking needs the optional extra: uv sync --extra rerank"
            ) from exc
        self.model_name = model_name
        self._model: Any = CrossEncoder(model_name)

    def scores(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self._model.predict([(query, t) for t in texts])]
