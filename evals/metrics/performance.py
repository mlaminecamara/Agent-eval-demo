"""Latency, tokens, cost and steps over a set of per-case records."""

from __future__ import annotations

from typing import Any

from evals.metrics.stats import mean, percentile


def summarize_performance(records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [r["latency_s"] for r in records]
    costs = [r["cost_usd"] for r in records]
    cost_known = all(c is not None for c in costs)
    stopped: dict[str, int] = {}
    for r in records:
        stopped[r["stopped_reason"]] = stopped.get(r["stopped_reason"], 0) + 1
    return {
        "n": len(records),
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "latency_max_s": max(latencies) if latencies else None,
        "input_tokens_mean": mean([r["input_tokens"] for r in records]),
        "output_tokens_mean": mean([r["output_tokens"] for r in records]),
        # Unknown as soon as one case has an unknown cost: never a partial sum.
        "cost_usd_mean": mean(costs) if cost_known and costs else None,
        "cost_usd_total": sum(costs) if cost_known and costs else None,
        "steps_mean": mean([r["steps"] for r in records]),
        "tool_calls_mean": mean([len(r["tool_calls"]) for r in records]),
        "stopped_reasons": stopped,
    }
