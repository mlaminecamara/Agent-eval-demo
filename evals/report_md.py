"""Markdown rendering of one run's results (written next to the JSON by run_eval)."""

from __future__ import annotations

from typing import Any

from evals.metrics.stats import Rate


def fmt_rate(d: dict[str, Any]) -> str:
    return Rate(d["successes"], d["n"], d["ci_low"], d["ci_high"]).fmt()


def fmt_mean(d: dict[str, Any], digits: int = 2) -> str:
    if d["value"] is None:
        return f"n/a (n={d['n']})"
    return f"{d['value']:.{digits}f} (n={d['n']})"


def fmt_num(v: float | None, digits: int = 2, unit: str = "") -> str:
    return "n/a" if v is None else f"{v:.{digits}f}{unit}"


def fmt_cost(v: float | None) -> str:
    return "unknown" if v is None else f"${v:.4f}"


def metric_rows(m: dict[str, Any]) -> list[tuple[str, str]]:
    p = m["performance"]
    return [
        ("Cases", str(m["n_cases"])),
        ("All key facts found", fmt_rate(m["answer"]["all_facts_found"])),
        ("Key-fact recall (mean)", fmt_mean(m["answer"]["fact_recall_mean"])),
        ("LLM-judge pass (score ≥ 4/5)", fmt_rate(m["answer"]["judge_pass"])),
        ("LLM-judge score (mean, 1–5)", fmt_mean(m["answer"]["judge_score_mean"])),
        ("Retrieval hit@1 (direct)", fmt_rate(m["retrieval_direct"]["hit_at_1"])),
        ("Retrieval hit@5 (direct)", fmt_rate(m["retrieval_direct"]["hit_at_5"])),
        ("Retrieval MRR@10 (direct)", fmt_mean(m["retrieval_direct"]["mrr_at_10"], 3)),
        ("Agent searched the corpus", fmt_rate(m["retrieval_agent"]["searched"])),
        ("Agent hit@5 when it searched", fmt_rate(m["retrieval_agent"]["hit_at_5_when_searched"])),
        ("All expected tool calls made", fmt_rate(m["tool_use"]["all_expected_calls_matched"])),
        ("Expected-call recall (mean)", fmt_mean(m["tool_use"]["expected_call_recall_mean"])),
        (
            "Failed tool calls",
            f"{m['tool_use']['failed_calls']['failed']}/{m['tool_use']['failed_calls']['n']}",
        ),
        ("Steps above expected (mean)", fmt_mean(m["tool_use"]["excess_steps_mean"])),
        ("Correct refusal (must refuse)", fmt_rate(m["safety"]["correct_refusal"])),
        ("False refusal (must answer)", fmt_rate(m["safety"]["false_refusal"])),
        ("Injection resisted", fmt_rate(m["safety"]["injection_resisted"])),
        ("System-prompt leak", fmt_rate(m["safety"]["system_prompt_leak"])),
        (
            "Latency p50 / p95",
            f"{fmt_num(p['latency_p50_s'], 1, ' s')} / "
            f"{fmt_num(p['latency_p95_s'], 1, ' s')} (n={p['n']})",
        ),
        (
            "Tokens per request (in / out, mean)",
            f"{fmt_num(p['input_tokens_mean'], 0)} / {fmt_num(p['output_tokens_mean'], 0)}",
        ),
        ("Agent cost per request (mean)", fmt_cost(p["cost_usd_mean"])),
        ("Agent cost total", fmt_cost(p["cost_usd_total"])),
        ("Steps per request (mean)", fmt_num(p["steps_mean"], 1)),
    ]


def render_run_markdown(payload: dict[str, Any]) -> str:
    cfg, ds = payload["config"], payload["dataset"]
    lines = [f"# Eval run `{payload['run_id']}`", ""]
    warnings = []
    if payload["mock"]:
        warnings.append(
            "MOCK RUN: fake LLM, embeddings and judge. These numbers only check "
            "that the pipeline works; they say nothing about model quality."
        )
    if not ds["validated"]:
        warnings.append(f"UNVALIDATED DATASET: case status counts {ds['status_counts']}.")
    if not payload["complete"]:
        warnings.append(f"INCOMPLETE RUN: {payload['incomplete_reason']}")
    unverified = [m for m, ok in payload["prices_verified"].items() if not ok]
    if unverified and not payload["mock"]:
        warnings.append(f"Unverified prices for: {', '.join(sorted(unverified))}.")
    lines += [f"> **{w}**" for w in warnings]
    if warnings:
        lines.append("")
    lines += [
        "| Setting | Value |",
        "|---|---|",
        f"| Agent | {cfg['agent']} |",
        f"| LLM | {cfg['llm_provider']}:{cfg['llm_model']} |",
        f"| Embeddings / store | {cfg['embedding_model']} / {cfg['store']} "
        f"(`{cfg['index_table']}`) |",
        f"| Chunking | {cfg['chunk_size']} tokens, overlap {cfg['chunk_overlap']} |",
        f"| Rerank | {cfg['reranker_model'] if cfg['rerank'] else 'off'} |",
        f"| Judge | {cfg['judge_model'] or 'off'}"
        + (
            f" (prompt sha256:{cfg['judge_prompt_sha256_16']})"
            if cfg.get("judge_prompt_sha256_16")
            else ""
        )
        + " |",
        f"| Dataset | `{ds['path']}` sha256:{ds['sha256_16']}, {ds['n_cases']} cases |",
        f"| Code | {cfg['git_commit'] or 'unknown'} |",
        f"| Created | {payload['created_at']} |",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in metric_rows(payload["metrics"]["overall"])]

    by_cat = payload["metrics"]["by_category"]
    if by_cat:
        cats = list(by_cat)
        lines += [
            "",
            "## By category",
            "",
            "| Metric | " + " | ".join(cats) + " |",
            "|---|" + "---|" * len(cats),
        ]
        rows = {c: dict(metric_rows(by_cat[c])) for c in cats}
        for key in rows[cats[0]]:
            lines.append(f"| {key} | " + " | ".join(rows[c][key] for c in cats) + " |")

    lines += [
        "",
        "## Cases",
        "",
        "| Case | Category | Facts | Tools | Refusal OK | Judge | Steps | Latency | Cost |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in payload["cases"]:
        facts = (
            "–"
            if r["facts"]["expected"] == 0
            else f"{r['facts']['found']}/{r['facts']['expected']}"
        )
        tools = (
            "–"
            if r["tools"]["expected"] == 0
            else f"{r['tools']['matched']}/{r['tools']['expected']}"
        )
        judge = "–" if not r["judge"] or r["judge"]["score"] is None else str(r["judge"]["score"])
        lines.append(
            f"| {r['id']} | {r['category']} | {facts} | {tools} | "
            f"{'yes' if r['safety']['refusal_correct'] else 'NO'} | {judge} | {r['steps']} | "
            f"{r['latency_s']:.1f} s | {fmt_cost(r['cost_usd'])} |"
        )
    lines.append("")
    return "\n".join(lines)
