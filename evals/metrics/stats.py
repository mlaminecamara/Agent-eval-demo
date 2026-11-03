"""Small statistics helpers. Every rate is reported with its sample size and a 95% Wilson
interval, which behaves sensibly for small n and for rates near 0 or 1."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Rate:
    successes: int
    n: int
    ci_low: float | None
    ci_high: float | None

    @property
    def value(self) -> float | None:
        return None if self.n == 0 else self.successes / self.n

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"value": self.value}

    def fmt(self) -> str:
        if self.n == 0:
            return "n/a (n=0)"
        assert self.ci_low is not None and self.ci_high is not None
        return (
            f"{100 * self.successes / self.n:.0f}% ({self.successes}/{self.n}, "
            f"95% CI {100 * self.ci_low:.0f}–{100 * self.ci_high:.0f})"
        )


def wilson(successes: int, n: int, z: float = 1.96) -> Rate:
    if n == 0:
        return Rate(0, 0, None, None)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return Rate(successes, n, max(0.0, centre - half), min(1.0, centre + half))


def rate(flags: list[bool]) -> Rate:
    return wilson(sum(1 for f in flags if f), len(flags))


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolation percentile (q in [0, 100]); None for an empty list."""
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
