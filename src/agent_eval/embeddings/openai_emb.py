"""OpenAI embeddings."""

from __future__ import annotations

from agent_eval.embeddings.base import Embedder


class OpenAIEmbedder(Embedder):
    name = "openai"
    batch_size = 100

    def __init__(self, model: str, dimension: int, timeout_s: float = 60) -> None:
        super().__init__(model, dimension)
        from openai import OpenAI  # imported lazily so mock mode needs no key

        self._client = OpenAI(timeout=timeout_s)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            resp = self._client.embeddings.create(model=self.model, input=batch)
            self.tokens_used += resp.usage.prompt_tokens
            vectors.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        if vectors and len(vectors[0]) != self.dimension:
            raise ValueError(
                f"{self.model} returned dimension {len(vectors[0])}, config says {self.dimension}"
            )
        return vectors
