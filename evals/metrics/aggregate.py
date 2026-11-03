"""Aggregate per-case records into the metric summary, overall and per category.

Every rate carries its n and a Wilson 95% interval; every mean carries its n.
"""

from __future__ import annotations

from typing import Any

from evals.metrics.performance import summarize_performance
from evals.metrics.stats import mean, rate

CATEGORIES = ("factual", "multi_step", "out_of_scope", "injection")


def _mean_with_n(values: list[float]) -> dict[str, Any]:
    return {"value": mean(values), "n": len(values)}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    with_facts = [r for r in records if r["facts"]["expected"] > 0]
    with_tools = [r for r in records if r["tools"]["expected"] > 0]
    with_sources = [r for r in records if r.get("retrieval")]
    must_refuse = [r for r in records if r["must_refuse"]]
    must_answer = [r for r in records if not r["must_refuse"]]
    injection = [r for r in records if r["safety"]["injection_resisted"] is not None]
    judged = [r for r in records if r.get("judge") and r["judge"].get("score") is not None]

    direct = [r["retrieval"]["direct"] for r in with_sources if r["retrieval"].get("direct")]
    agent_ret = [r["retrieval"]["agent"] for r in with_sources]
    searched = [a for a in agent_ret if a["searched"]]

    total_calls = sum(r["tools"]["actual_calls"] for r in records)
    failed_calls = sum(r["tools"]["failed_calls"] for r in records)

    return {
        "n_cases": len(records),
        "answer": {
            "all_facts_found": rate([r["facts"]["all_found"] for r in with_facts]).to_dict(),
            "fact_recall_mean": _mean_with_n([r["facts"]["recall"] for r in with_facts]),
            "judge_pass": rate([bool(r["judge"]["pass"]) for r in judged]).to_dict(),
            "judge_score_mean": _mean_with_n([r["judge"]["score"] for r in judged]),
        },
        "retrieval_direct": {
            "hit_at_1": rate([d["hit_at_1"] for d in direct]).to_dict(),
            "hit_at_3": rate([d["hit_at_3"] for d in direct]).to_dict(),
            "hit_at_5": rate([d["hit_at_5"] for d in direct]).to_dict(),
            "mrr_at_10": _mean_with_n([d["rr_at_10"] for d in direct]),
        },
        "retrieval_agent": {
            "searched": rate([a["searched"] for a in agent_ret]).to_dict(),
            "hit_at_5_when_searched": rate([a["hit_at_5"] for a in searched]).to_dict(),
            "mrr_when_searched": _mean_with_n([a["rr"] for a in searched]),
        },
        "tool_use": {
            "all_expected_calls_matched": rate(
                [r["tools"]["all_matched"] for r in with_tools]
            ).to_dict(),
            "expected_call_recall_mean": _mean_with_n([r["tools"]["recall"] for r in with_tools]),
            "failed_calls": {
                "value": failed_calls / total_calls if total_calls else None,
                "failed": failed_calls,
                "n": total_calls,
            },
            "excess_steps_mean": _mean_with_n(
                [
                    r["tools"]["excess_steps"]
                    for r in records
                    if r["tools"]["excess_steps"] is not None
                ]
            ),
        },
        "safety": {
            "correct_refusal": rate([r["safety"]["refused"] for r in must_refuse]).to_dict(),
            "false_refusal": rate([r["safety"]["refused"] for r in must_answer]).to_dict(),
            "injection_resisted": rate(
                [r["safety"]["injection_resisted"] for r in injection]
            ).to_dict(),
            "system_prompt_leak": rate(
                [r["safety"]["leaked_system_prompt"] for r in records]
            ).to_dict(),
        },
        "performance": summarize_performance(records),
    }


def summarize_by_category(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        cat: summarize([r for r in records if r["category"] == cat])
        for cat in CATEGORIES
        if any(r["category"] == cat for r in records)
    }
