"""Hard budget for one eval run, plus the pre-run cost estimate and confirmation.

The guard counts LLM calls and tokens across every provider call made through GuardedLLM and
raises BudgetExceeded as soon as a limit is reached. A call already in flight cannot be
stopped, so the token limit can be overshot by at most one response.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

from agent_eval.observability.cost import PricingTable


class BudgetExceeded(RuntimeError):
    pass


class CostGuard:
    def __init__(self, max_llm_calls: int, max_total_tokens: int) -> None:
        self.max_llm_calls = max_llm_calls
        self.max_total_tokens = max_total_tokens
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self._lock = threading.Lock()

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def before_call(self) -> None:
        with self._lock:
            if self.calls >= self.max_llm_calls:
                raise BudgetExceeded(f"max_llm_calls={self.max_llm_calls} reached")
            if self.total_tokens >= self.max_total_tokens:
                raise BudgetExceeded(
                    f"max_total_tokens={self.max_total_tokens} reached ({self.total_tokens} used)"
                )
            self.calls += 1

    def after_call(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens

    def snapshot(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "max_llm_calls": self.max_llm_calls,
            "max_total_tokens": self.max_total_tokens,
        }


@dataclass
class CostEstimate:
    model: str
    n_requests: int
    calls_per_request: float
    input_tokens_per_call: int
    output_tokens_per_call: int
    total_calls: int
    total_input_tokens: int
    total_output_tokens: int
    cost_usd: float | None  # None when the price is unknown
    price_verified: bool

    def render(self) -> str:
        cost = (
            "UNKNOWN (no price in config/pricing.yaml)"
            if self.cost_usd is None
            else f"${self.cost_usd:.4f}"
        )
        lines = [
            f"Model: {self.model}",
            f"Requests: {self.n_requests} x ~{self.calls_per_request:g} LLM calls "
            f"= ~{self.total_calls} calls",
            f"Tokens (upper-bound estimate): ~{self.total_input_tokens:,} in / "
            f"~{self.total_output_tokens:,} out",
            f"Estimated cost: {cost}",
        ]
        if self.cost_usd is not None and not self.price_verified:
            lines.append("WARNING: price for this model is marked verified: false.")
        return "\n".join(lines)


def estimate_cost(
    pricing: PricingTable,
    model: str,
    n_requests: int,
    calls_per_request: float,
    input_tokens_per_call: int,
    output_tokens_per_call: int,
) -> CostEstimate:
    total_calls = round(n_requests * calls_per_request)
    total_in = total_calls * input_tokens_per_call
    total_out = total_calls * output_tokens_per_call
    return CostEstimate(
        model=model,
        n_requests=n_requests,
        calls_per_request=calls_per_request,
        input_tokens_per_call=input_tokens_per_call,
        output_tokens_per_call=output_tokens_per_call,
        total_calls=total_calls,
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        cost_usd=pricing.cost(model, total_in, total_out),
        price_verified=pricing.get(model).verified,
    )


def confirm_run(
    estimates: list[CostEstimate],
    assume_yes: bool = False,
    allow_unknown_price: bool = False,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> bool:
    """Print the estimates and ask for confirmation. Returns True if the run may proceed."""
    for est in estimates:
        print_fn(est.render())
        print_fn("")
    unknown = [e.model for e in estimates if e.cost_usd is None]
    if unknown and not allow_unknown_price:
        print_fn(
            f"Refusing to run: no price for {', '.join(unknown)}. Fill config/pricing.yaml "
            "or pass --allow-unknown-price."
        )
        return False
    if assume_yes:
        return True
    return input_fn("Proceed with this real (billed) run? [y/N] ").strip().lower() in {"y", "yes"}
