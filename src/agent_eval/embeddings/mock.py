"""Deterministic fake embeddings (signed feature hashing over word unigrams and bigrams).

Texts that share words get similar vectors, so retrieval in mock mode behaves like a crude
lexical search. Stable across processes and platforms (blake2b, not Python's salted hash).
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re
import unicodedata

from agent_eval.embeddings.base import Embedder
from agent_eval.llm.mock import estimate_tokens

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    norm = unicodedata.normalize("NFKD", text.lower())
    norm = "".join(c for c in norm if not unicodedata.combining(c))
    words = [w for w in _WORD.findall(norm) if len(w) > 1]
    return words + [f"{a}_{b}" for a, b in itertools.pairwise(words)]


class MockEmbedder(Embedder):
    name = "mock"

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        for tok in _tokens(text):
            digest = hashlib.blake2b(tok.encode(), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            vec[0] = 1.0  # empty text: arbitrary but valid unit vector
            return vec
        return [v / norm for v in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.tokens_used += sum(estimate_tokens(t) for t in texts)
        return [self._vector(t) for t in texts]
