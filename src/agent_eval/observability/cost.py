"""Token pricing and cost computation, read from config/pricing.yaml.

A missing or null price yields cost=None ("unknown"), never a guessed number.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelPrice:
    model: str
    input_per_mtok: float | None
    output_per_mtok: float | None
    verified: bool
    source: str | None = None

    @property
    def known(self) -> bool:
        return self.input_per_mtok is not None and self.output_per_mtok is not None


class PricingTable:
    def __init__(self, prices: dict[str, ModelPrice]) -> None:
        self.prices = prices

    @classmethod
    def load(cls, path: Path | str) -> PricingTable:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        prices = {
            name: ModelPrice(
                model=name,
                input_per_mtok=entry.get("input_per_mtok"),
                output_per_mtok=entry.get("output_per_mtok"),
                verified=bool(entry.get("verified", False)),
                source=entry.get("source"),
            )
            for name, entry in (data.get("models") or {}).items()
        }
        return cls(prices)

    def get(self, model: str) -> ModelPrice:
        return self.prices.get(model, ModelPrice(model, None, None, verified=False))

    def cost(self, model: str, input_tokens: int, output_tokens: int) -> float | None:
        price = self.get(model)
        if not price.known:
            return None
        assert price.input_per_mtok is not None and price.output_per_mtok is not None
        return (input_tokens * price.input_per_mtok + output_tokens * price.output_per_mtok) / 1e6
