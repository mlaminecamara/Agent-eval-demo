"""Compare evaluation runs: group repeated runs by configuration and aggregate them.

    uv run python -m evals.compare            # real runs in evals/results/
    uv run python -m evals.compare --mock     # mock runs in evals/results/mock/

Writes comparison.json and comparison.md next to the runs. Only numbers read from result
files appear in the output; nothing is typed in by hand.

Which runs count. A run is excluded (and listed with the reason) when it is incomplete, was
made on a dataset with draft cases, on another version of the golden dataset, or graded
with another version of the judge prompt than the current one. So results stay comparable.

Statistics. A configuration is usually run several times on the same 33 questions. The
repeats are not independent samples (same questions), so the unit is the case: each case's
outcome is averaged over the runs, and the 95% Wilson interval is computed with n = number
of cases and the summed per-case averages as successes. With one run this is the usual
Wilson interval; with repeats it is conservative (a per-case average varies less than one
Bernoulli draw), and unlike a bootstrap it does not collapse to [0, 0] at 0%. Differences
with the baseline use a paired bootstrap over the cases both configurations ran. Run-to-run
spread (min-max of the per-run value) and the number of cases whose outcome changed between
runs are reported next to it. None of this covers questions outside the dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from agent_eval.config import PROJECT_ROOT
from evals.dataset import dataset_fingerprint
from evals.judge import PROMPT_PATH as JUDGE_PROMPT_PATH
from evals.metrics.stats import percentile

RESULTS_DIR = PROJECT_ROOT / "evals" / "results"
GOLDEN = PROJECT_ROOT / "evals" / "datasets" / "golden.jsonl"
N_BOOT = 2000
SEED = 20261105

CaseFn = Callable[[dict[str, Any]], float | None]


def _flag(value: Any) -> float | None:
    return None if value is None else float(bool(value))


# name -> (label, per-case value or None when the metric does not apply to the case)
METRICS: dict[str, tuple[str, CaseFn]] = {
    "judge_pass": ("LLM-judge pass", lambda c: _flag((c["judge"] or {}).get("pass"))),
    "facts_all_found": ("All key facts found", lambda c: _flag(c["facts"]["all_found"])),
    "tools_all_matched": (
        "Expected tool calls made",
        # as in the run report: only cases that expect at least one tool call
        lambda c: _flag(c["tools"]["all_matched"]) if c["tools"]["expected"] else None,
    ),
    "hit_at_5_direct": (
        "Retrieval hit@5 (direct)",
        lambda c: _flag(c["retrieval"]["direct"]["hit_at_5"]) if c["retrieval"] else None,
    ),
    "correct_refusal": (
        "Correct refusal",
        lambda c: _flag(c["safety"]["refusal_correct"]) if c["must_refuse"] else None,
    ),
    "false_refusal": (
        "False refusal",
        lambda c: None if c["must_refuse"] else _flag(c["safety"]["refused"]),
    ),
    "injection_resisted": (
        "Injection resisted",
        lambda c: _flag(c["safety"]["injection_resisted"]),
    ),
    "system_prompt_leak": (
        "System-prompt leak",
        lambda c: _flag(c["safety"]["leaked_system_prompt"]),
    ),
}
# Metrics compared with the baseline (paired bootstrap).
DELTA_METRICS = ("judge_pass", "facts_all_found", "tools_all_matched", "correct_refusal")


@dataclass
class Group:
    key: tuple[Any, ...]
    label: str
    config: dict[str, Any]
    runs: list[dict[str, Any]] = field(default_factory=list)


def config_key(cfg: dict[str, Any]) -> tuple[Any, ...]:
    return (
        cfg["agent"],
        cfg["llm_provider"],
        cfg["llm_model"],
        cfg["embedding_model"],
        cfg["chunk_size"],
        cfg["chunk_overlap"],
        cfg["top_k"],
        bool(cfg["rerank"]),
    )


def config_label(cfg: dict[str, Any]) -> str:
    rerank = "rerank" if cfg["rerank"] else "no rerank"
    return (
        f"{cfg['agent']} · {cfg['llm_model']} · "
        f"c{cfg['chunk_size']}/o{cfg['chunk_overlap']} · {rerank}"
    )


def exclusion_reason(run: dict[str, Any], dataset_sha: str, judge_sha: str) -> str | None:
    if not run.get("complete"):
        return f"incomplete ({run.get('incomplete_reason')})"
    if not run["dataset"]["validated"]:
        return "dataset had draft cases"
    if run["dataset"]["sha256_16"] != dataset_sha:
        return f"other dataset version ({run['dataset']['sha256_16']})"
    judged = run["config"]["judge_model"] and not run.get("mock")  # the mock judge has no prompt
    if judged and run["config"].get("judge_prompt_sha256_16") != judge_sha:
        return "graded with another judge prompt version"
    return None


def load_runs(
    results_dir: Path, dataset_sha: str, judge_sha: str
) -> tuple[list[Group], list[dict[str, str]]]:
    groups: dict[tuple[Any, ...], Group] = {}
    excluded: list[dict[str, str]] = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name.startswith("comparison"):
            continue
        run = json.loads(path.read_text(encoding="utf-8"))
        reason = exclusion_reason(run, dataset_sha, judge_sha)
        if reason:
            excluded.append({"run_id": run["run_id"], "reason": reason})
            continue
        key = config_key(run["config"])
        group = groups.setdefault(key, Group(key, config_label(run["config"]), run["config"]))
        group.runs.append(run)
    return list(groups.values()), excluded


def per_case_means(runs: list[dict[str, Any]], fn: CaseFn) -> dict[str, float]:
    """case id -> mean of the metric over the runs where it applies."""
    values: dict[str, list[float]] = {}
    for run in runs:
        for case in run["cases"]:
            v = fn(case)
            if v is not None:
                values.setdefault(case["id"], []).append(v)
    return {cid: float(np.mean(vs)) for cid, vs in values.items()}


def wilson_ci(successes: float, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval; `successes` may be fractional (summed per-case averages)."""
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_ci(values: list[float], rng: np.random.Generator) -> tuple[float, float] | None:
    if not values:
        return None
    arr = np.asarray(values)
    idx = rng.integers(0, len(arr), size=(N_BOOT, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize_metric(runs: list[dict[str, Any]], fn: CaseFn) -> dict[str, Any]:
    by_case = per_case_means(runs, fn)
    per_run = []
    for run in runs:
        vals = [v for v in (fn(c) for c in run["cases"]) if v is not None]
        if vals:
            per_run.append(float(np.mean(vals)))
    n_obs = sum(1 for run in runs for c in run["cases"] if fn(c) is not None)
    ci = wilson_ci(sum(by_case.values()), len(by_case))
    return {
        "value": float(np.mean(list(by_case.values()))) if by_case else None,
        "ci_low": ci[0] if ci else None,
        "ci_high": ci[1] if ci else None,
        "n_cases": len(by_case),
        "n_observations": n_obs,
        "per_run_min": min(per_run) if per_run else None,
        "per_run_max": max(per_run) if per_run else None,
        # cases whose outcome was not the same in every run (only meaningful with >1 run)
        "unstable_cases": sum(1 for v in by_case.values() if 0 < v < 1) if len(runs) > 1 else None,
    }


def paired_delta(
    a_runs: list[dict[str, Any]],
    b_runs: list[dict[str, Any]],
    fn: CaseFn,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    """b - a on the cases both configurations have, with a paired bootstrap interval."""
    a, b = per_case_means(a_runs, fn), per_case_means(b_runs, fn)
    common = sorted(a.keys() & b.keys())
    if not common:
        return None
    diffs = [b[c] - a[c] for c in common]
    ci = bootstrap_ci(diffs, rng)
    assert ci is not None
    return {
        "delta": float(np.mean(diffs)),
        "ci_low": ci[0],
        "ci_high": ci[1],
        "n_cases": len(common),
        "significant": ci[0] > 0 or ci[1] < 0,
    }


def performance(runs: list[dict[str, Any]]) -> dict[str, Any]:
    cases = [c for run in runs for c in run["cases"]]
    latencies = [c["latency_s"] for c in cases]
    agent_costs: list[float] = [c["cost_usd"] for c in cases if c["cost_usd"] is not None]
    judged = [c["judge"] for c in cases if c["judge"]]
    judge_costs: list[float] = [j["cost_usd"] for j in judged if j.get("cost_usd") is not None]
    known = len(agent_costs) == len(cases)
    return {
        "n_requests": len(cases),
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "steps_mean": float(np.mean([c["steps"] for c in cases])) if cases else None,
        "agent_cost_per_request": float(np.mean(agent_costs)) if cases and known else None,
        "judge_cost_per_request": (
            float(np.mean(judge_costs)) if judged and len(judge_costs) == len(judged) else None
        ),
        "agent_cost_total": float(sum(agent_costs)) if known else None,
    }


def pick_baseline(groups: list[Group], wanted: str | None) -> Group | None:
    if not groups:
        return None
    if wanted:
        match = [g for g in groups if wanted in g.label]
        if len(match) != 1:
            raise SystemExit(
                f"--baseline {wanted!r} matches {len(match)} configurations: "
                + "; ".join(g.label for g in match or groups)
            )
        return match[0]
    defaults = [g for g in groups if g.config["agent"] == "react" and not g.config["rerank"]]
    return max(defaults or groups, key=lambda g: len(g.runs))


def compare(groups: list[Group], baseline: Group | None) -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEED)
    out = []
    for g in groups:
        entry: dict[str, Any] = {
            "label": g.label,
            "config": {k: g.config[k] for k in sorted(g.config) if k != "git_commit"},
            "runs": [r["run_id"] for r in g.runs],
            "n_runs": len(g.runs),
            "prices_verified": all(all(r["prices_verified"].values()) for r in g.runs),
            "is_baseline": g is baseline,
            "metrics": {name: summarize_metric(g.runs, fn) for name, (_, fn) in METRICS.items()},
            "performance": performance(g.runs),
            "delta_vs_baseline": None,
        }
        if baseline is not None and g is not baseline:
            entry["delta_vs_baseline"] = {
                name: paired_delta(baseline.runs, g.runs, METRICS[name][1], rng)
                for name in DELTA_METRICS
            }
        out.append(entry)
    return out


# --- Markdown ---------------------------------------------------------------------------------


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{100 * v:.0f}%"


def fmt_metric(m: dict[str, Any]) -> str:
    if m["value"] is None:
        return "n/a"
    text = f"{_pct(m['value'])} [{_pct(m['ci_low'])}–{_pct(m['ci_high'])}]"
    if m["per_run_min"] != m["per_run_max"]:
        text += f" runs {_pct(m['per_run_min'])}–{_pct(m['per_run_max'])}"
    return text


def fmt_delta(d: dict[str, Any] | None) -> str:
    if d is None:
        return "n/a"
    sign = "+" if d["delta"] >= 0 else ""
    mark = " *" if d["significant"] else ""
    return (
        f"{sign}{100 * d['delta']:.0f} pts "
        f"[{100 * d['ci_low']:+.0f}, {100 * d['ci_high']:+.0f}]{mark}"
    )


def _money(v: float | None) -> str:
    return "unknown" if v is None else f"${v:.4f}"


def _secs(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.1f} s"


def render(entries: list[dict[str, Any]], excluded: list[dict[str, str]], mock: bool) -> str:
    lines = ["# Configuration comparison", ""]
    if mock:
        lines += ["> **MOCK RUNS: these numbers measure the pipeline, not model quality.**", ""]
    if not entries:
        lines += ["No comparable runs found.", ""]
    else:
        lines += [
            "Value = mean over cases of each case's outcome averaged over runs, "
            "[95% Wilson interval, one case = one unit]. `runs a–b`: min–max of the per-run value "
            "when runs differ. Generated by `python -m evals.compare`.",
            "",
            "## Quality and safety",
            "",
            "| Configuration | Runs | " + " | ".join(label for label, _ in METRICS.values()) + " |",
            "|---|---|" + "---|" * len(METRICS),
        ]
        for e in entries:
            name = e["label"] + (" (baseline)" if e["is_baseline"] else "")
            cells = [fmt_metric(e["metrics"][k]) for k in METRICS]
            lines.append(f"| {name} | {e['n_runs']} | " + " | ".join(cells) + " |")
        lines += [
            "",
            "## Latency and cost",
            "",
            "| Configuration | Requests | Latency p50 / p95 | Steps (mean) | Agent cost / request "
            "| Judge cost / request | Prices verified |",
            "|---|---|---|---|---|---|---|",
        ]
        for e in entries:
            p = e["performance"]
            lines.append(
                f"| {e['label']} | {p['n_requests']} | {_secs(p['latency_p50_s'])} / "
                f"{_secs(p['latency_p95_s'])} | {p['steps_mean']:.1f} | "
                f"{_money(p['agent_cost_per_request'])} | {_money(p['judge_cost_per_request'])} "
                f"| {'yes' if e['prices_verified'] else 'no'} |"
            )
        deltas = [e for e in entries if e["delta_vs_baseline"]]
        if deltas:
            base = next(e["label"] for e in entries if e["is_baseline"])
            lines += [
                "",
                f"## Difference with the baseline ({base})",
                "",
                "Paired bootstrap over the cases both configurations ran, in percentage points. "
                "`*` = the 95% interval excludes 0.",
                "",
                "| Configuration | " + " | ".join(METRICS[k][0] for k in DELTA_METRICS) + " |",
                "|---|" + "---|" * len(DELTA_METRICS),
            ]
            for e in deltas:
                cells = [fmt_delta(e["delta_vs_baseline"][k]) for k in DELTA_METRICS]
                lines.append(f"| {e['label']} | " + " | ".join(cells) + " |")
        unstable = [
            (e["label"], e["metrics"]["judge_pass"]["unstable_cases"])
            for e in entries
            if e["metrics"]["judge_pass"]["unstable_cases"] is not None
        ]
        if unstable:
            lines += ["", "## Run-to-run stability", ""]
            lines += [
                f"- {label}: {n} cases changed judge outcome between runs" for label, n in unstable
            ]
        lines += ["", "## Runs", ""]
        lines += [f"- {e['label']}: " + ", ".join(f"`{r}`" for r in e["runs"]) for e in entries]
    if excluded:
        lines += ["", "## Excluded runs", ""]
        lines += [f"- `{x['run_id']}`: {x['reason']}" for x in excluded]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mock", action="store_true", help="compare mock runs (evals/results/mock)")
    p.add_argument("--results-dir", type=Path)
    p.add_argument("--baseline", help="substring of the baseline configuration label")
    args = p.parse_args(argv)

    results_dir = args.results_dir or (RESULTS_DIR / "mock" if args.mock else RESULTS_DIR)
    groups, excluded = load_runs(
        results_dir, dataset_fingerprint(GOLDEN), dataset_fingerprint(JUDGE_PROMPT_PATH)
    )
    groups.sort(key=lambda g: g.label)
    entries = compare(groups, pick_baseline(groups, args.baseline))
    payload = {"mock": args.mock, "entries": entries, "excluded": excluded}
    (results_dir / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (results_dir / "comparison.md").write_text(
        render(entries, excluded, args.mock), encoding="utf-8"
    )
    print(
        f"{len(entries)} configuration(s) from {sum(e['n_runs'] for e in entries)} run(s); "
        f"{len(excluded)} excluded. Wrote {results_dir / 'comparison.md'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
