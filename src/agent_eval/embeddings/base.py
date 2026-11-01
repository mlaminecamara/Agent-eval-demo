"""Embedding interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Embedder(ABC):
    name: str

    def __init__(self, model: str, dimension: int) -> None:
        self.model = model
        self.dimension = dimension
        self.tokens_used = 0  # cumulative, for cost accounting

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one L2-normalised vector per input text."""

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
