"""Run a matrix of configurations x repeats, with one cost confirmation for the whole matrix.

    uv run python -m evals.run_matrix --dry-run      # plan and cost estimate only
    uv run python -m evals.run_matrix                # asks once, then runs everything
    uv run python -m evals.run_matrix --only react-openai,langgraph-openai

Each run is an ordinary `evals.run_eval` run (same hard caps, same result files), labelled
"matrix:<name>". Comparable runs already in evals/results/ count towards the repeats.

Cost estimate per run, in this order:
1. observed: mean agent + judge cost per request in earlier runs of the same configuration
   (any judge prompt version), times the number of cases, times a 1.5 safety margin;
2. otherwise an upper bound: calls per case for that agent x 5,000 input / 500 output tokens
   per call, plus the judge, at config/pricing.yaml prices.
Prices marked verified: false are flagged. After the runs, `evals.compare` is regenerated.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from agent_eval.config import PROJECT_ROOT, Settings, load_settings
from agent_eval.observability.cost import PricingTable
from evals import compare, run_eval
from evals.dataset import dataset_fingerprint, load_dataset

MATRIX = PROJECT_ROOT / "evals" / "matrix.yaml"
SAFETY_MARGIN = 1.5
# Upper-bound LLM calls per case: plan + up to 4 steps + synthesis; tool loop + verify retry.
CALLS_PER_CASE = {"react": 4, "plan_execute": 6, "langgraph": 6}
OVERRIDES = {
    "provider": "llm.provider",
    "model": "llm.model",
    "chunk_size": "retrieval.chunk_size",
    "chunk_overlap": "retrieval.chunk_overlap",
    "rerank": "retrieval.rerank",
}


@dataclass
class Planned:
    name: str
    agent: str
    settings: Settings
    repeats: int
    existing: int
    cost_per_run: float | None
    estimate_source: str
    prices_verified: bool

    @property
    def to_run(self) -> int:
        return max(0, self.repeats - self.existing)

    @property
    def label(self) -> str:
        return compare.config_label(self.config_view())

    def config_view(self) -> dict[str, Any]:
        s = self.settings
        return {
            "agent": self.agent,
            "llm_provider": s.llm.provider,
            "llm_model": s.llm.resolved_model,
            "embedding_model": s.embeddings.resolved_model,
            "chunk_size": s.retrieval.chunk_size,
            "chunk_overlap": s.retrieval.chunk_overlap,
            "top_k": s.retrieval.top_k,
            "rerank": s.retrieval.rerank,
        }


def settings_for(cfg: dict[str, Any]) -> Settings:
    overrides = {OVERRIDES[k]: v for k, v in cfg.items() if k in OVERRIDES}
    overrides["embeddings.provider"] = "openai"
    return load_settings(overrides=overrides)


def _past_runs(results_dir: Path) -> list[dict[str, Any]]:
    runs = []
    for path in sorted(results_dir.glob("*.json")):
        if not path.name.startswith("comparison"):
            runs.append(json.loads(path.read_text(encoding="utf-8")))
    return runs


def observed_cost_per_request(runs: list[dict[str, Any]], key: tuple[Any, ...]) -> float | None:
    """Mean agent + judge cost per request over earlier complete runs of this configuration."""
    totals: list[float] = []
    for run in runs:
        if not run.get("complete") or compare.config_key(run["config"]) != key:
            continue
        for case in run["cases"]:
            judge = (case["judge"] or {}).get("cost_usd") or 0.0
            if case["cost_usd"] is None:
                return None
            totals.append(case["cost_usd"] + judge)
    return sum(totals) / len(totals) if totals else None


def upper_bound_cost(
    settings: Settings, agent: str, n_cases: int, pricing: PricingTable
) -> float | None:
    calls = CALLS_PER_CASE.get(agent, run_eval.EST_AGENT_CALLS_PER_CASE) * n_cases
    agent_cost = pricing.cost(
        settings.llm.resolved_model,
        calls * run_eval.EST_AGENT_INPUT_TOKENS,
        calls * run_eval.EST_AGENT_OUTPUT_TOKENS,
    )
    judge_cost = pricing.cost(
        settings.judge.model,
        n_cases * run_eval.EST_JUDGE_INPUT_TOKENS,
        n_cases * run_eval.EST_JUDGE_OUTPUT_TOKENS,
    )
    if agent_cost is None or judge_cost is None:
        return None
    return agent_cost + judge_cost


def plan(matrix: dict[str, Any], only: set[str] | None, reuse: bool) -> list[Planned]:
    cases = load_dataset(run_eval.DEFAULT_DATASET)
    runs = _past_runs(run_eval.RESULTS_DIR)
    dataset_sha = dataset_fingerprint(run_eval.DEFAULT_DATASET)
    judge_sha = dataset_fingerprint(compare.JUDGE_PROMPT_PATH)
    comparable = [r for r in runs if compare.exclusion_reason(r, dataset_sha, judge_sha) is None]
    out = []
    for cfg in matrix["configs"]:
        if only and cfg["name"] not in only:
            continue
        settings = settings_for(cfg)
        pricing = PricingTable.load(settings.resolve_path(settings.cost_guard.pricing_file))
        planned = Planned(
            name=cfg["name"],
            agent=cfg.get("agent", "react"),
            settings=settings,
            repeats=int(cfg.get("repeats", matrix.get("repeats", 1))),
            existing=0,
            cost_per_run=None,
            estimate_source="",
            prices_verified=all(
                pricing.get(m).verified for m in (settings.llm.resolved_model, settings.judge.model)
            ),
        )
        key = compare.config_key(planned.config_view())
        if reuse:
            planned.existing = sum(1 for r in comparable if compare.config_key(r["config"]) == key)
        observed = observed_cost_per_request(runs, key)
        if observed is not None:
            planned.cost_per_run = observed * len(cases) * SAFETY_MARGIN
            planned.estimate_source = f"observed x{SAFETY_MARGIN}"
        else:
            planned.cost_per_run = upper_bound_cost(settings, planned.agent, len(cases), pricing)
            planned.estimate_source = "upper bound"
        out.append(planned)
    return out


def preflight(planned: list[Planned]) -> list[str]:
    problems = []
    needs_rerank = any(p.settings.retrieval.rerank and p.to_run for p in planned)
    if needs_rerank and importlib.util.find_spec("sentence_transformers") is None:
        problems.append(
            "rerank configs need the local cross-encoder: run `uv sync --extra rerank` "
            "(downloads torch and, on first use, the reranker model)."
        )
    return problems


def render_plan(planned: list[Planned], n_cases: int) -> tuple[str, float | None]:
    lines = [
        f"{'config':<22} {'model':<28} {'done':>4} {'to run':>6} {'$/run':>9} "
        f"{'$ total':>9}  estimate",
    ]
    total: float | None = 0.0
    for p in planned:
        per_run = "unknown" if p.cost_per_run is None else f"${p.cost_per_run:.3f}"
        sub = None if p.cost_per_run is None else p.cost_per_run * p.to_run
        if sub is None:
            total = None
        elif total is not None:
            total += sub
        flag = "" if p.prices_verified else "  (unverified prices)"
        lines.append(
            f"{p.name:<22} {p.settings.llm.resolved_model:<28} {p.existing:>4} {p.to_run:>6} "
            f"{per_run:>9} {('unknown' if sub is None else f'${sub:.3f}'):>9}  "
            f"{p.estimate_source}{flag}"
        )
    total_txt = "unknown" if total is None else f"${total:.2f}"
    runs = sum(p.to_run for p in planned)
    lines.append(f"\n{runs} run(s) of {n_cases} cases. Estimated total: {total_txt}")
    return "\n".join(lines), total


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--matrix", type=Path, default=MATRIX)
    p.add_argument("--only", help="comma-separated config names")
    p.add_argument("--dry-run", action="store_true", help="print the plan and estimate only")
    p.add_argument("--no-reuse", action="store_true", help="ignore existing comparable runs")
    p.add_argument("--yes", action="store_true", help="skip the confirmation")
    args = p.parse_args(argv)

    matrix = yaml.safe_load(args.matrix.read_text(encoding="utf-8"))
    only = set(args.only.split(",")) if args.only else None
    planned = plan(matrix, only, reuse=not args.no_reuse)
    n_cases = len(load_dataset(run_eval.DEFAULT_DATASET))
    text, total = render_plan(planned, n_cases)
    print(text)
    for problem in preflight(planned):
        print(f"\nCannot run: {problem}")
        return 2
    if args.dry_run or not any(x.to_run for x in planned):
        return 0
    if total is None:
        print("Refusing to run: a price is unknown (fill config/pricing.yaml).")
        return 2
    if not args.yes and input(
        "Proceed with these real (billed) runs? [y/N] "
    ).strip().lower() not in {
        "y",
        "yes",
    }:
        print("Aborted.")
        return 1

    for x in planned:
        for i in range(x.to_run):
            s = x.settings
            run_args = [
                "--agent",
                x.agent,
                "--provider",
                s.llm.provider,
                "--model",
                s.llm.resolved_model,
                "--chunk-size",
                str(s.retrieval.chunk_size),
                "--overlap",
                str(s.retrieval.chunk_overlap),
                "--rerank" if s.retrieval.rerank else "--no-rerank",
                "--label",
                f"matrix:{x.name}",
                "--yes",
            ]
            print(f"\n=== {x.name} run {i + 1}/{x.to_run}")
            code = run_eval.main(run_args)
            if code != 0:
                print(f"Stopping the matrix: run_eval exited with {code}.")
                compare.main([])
                return code
    return compare.main([])


if __name__ == "__main__":
    sys.exit(main())
