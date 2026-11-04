"""Run one configuration over the golden dataset and write the results.

    uv run python -m evals.run_eval --mock                       # offline, no API key
    uv run python -m evals.run_eval --provider openai            # real run (asks to confirm)
    uv run python -m evals.run_eval --provider anthropic --chunk-size 1000 --overlap 100

Outputs:
- real runs: evals/results/<run_id>.json (metrics + per-case details) and <run_id>.md
- mock runs: evals/results/mock/<run_id>.{json,md} (git-ignored; never used by the report)
- traces of every LLM / tool / judge call: runs/evals/<run_id>.jsonl (git-ignored)

Safeguards for real runs: refuses a dataset that still has draft cases unless --allow-draft
(the result is then flagged), prints the estimated cost and asks for confirmation, and caps
LLM calls and tokens with the cost guard (config.yaml: cost_guard).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_eval.agents import AGENT_NAMES, build_agent
from agent_eval.agents.base import AgentConfig, AgentResult
from agent_eval.config import PROJECT_ROOT, Settings, load_settings
from agent_eval.cost_guard import BudgetExceeded, CostGuard, confirm_run, estimate_cost
from agent_eval.embeddings import build_embedder
from agent_eval.ingest.chunker import chunk_documents
from agent_eval.ingest.indexer import index_chunks
from agent_eval.ingest.loader import load_manifest
from agent_eval.observability.cost import PricingTable
from agent_eval.observability.logging import configure_logging
from agent_eval.retrieval.base import VectorStore
from agent_eval.retrieval.memory_store import MemoryStore
from agent_eval.retrieval.retriever import (
    Retriever,
    build_reranker,
    build_store,
    index_spec,
)
from agent_eval.tools.registry import ToolRegistry, build_registry
from evals.dataset import Case, dataset_fingerprint, load_dataset, status_counts
from evals.judge import PROMPT_PATH as JUDGE_PROMPT_PATH
from evals.judge import Judge, MockJudge
from evals.metrics.aggregate import summarize, summarize_by_category
from evals.metrics.answer_quality import evaluate_facts
from evals.metrics.retrieval import first_relevant_rank, hit_at_k, reciprocal_rank
from evals.metrics.safety import evaluate_safety
from evals.metrics.tool_use import evaluate_tool_use
from evals.report_md import render_run_markdown

DEFAULT_DATASET = PROJECT_ROOT / "evals" / "datasets" / "golden.jsonl"
RESULTS_DIR = PROJECT_ROOT / "evals" / "results"

# Upper-bound guesses used only for the pre-run cost estimate (printed, then confirmed).
EST_AGENT_CALLS_PER_CASE = 4
EST_AGENT_INPUT_TOKENS = 5000
EST_AGENT_OUTPUT_TOKENS = 500
EST_JUDGE_INPUT_TOKENS = 3500
EST_JUDGE_OUTPUT_TOKENS = 600


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        sha = out.stdout.strip() or None
        return f"{sha}-dirty" if sha and dirty.stdout.strip() else sha
    except (OSError, subprocess.SubprocessError):
        return None


def _manifests(settings: Settings) -> list[Path]:
    paths = [settings.resolve_path(settings.corpus.manifest)]
    adv = settings.resolve_path(settings.corpus.adversarial_dir) / "manifest.yaml"
    if adv.exists():
        paths.append(adv)
    return paths


def build_index(settings: Settings, needs_adversarial: bool) -> tuple[VectorStore, Any]:
    """Memory store: index in-process. pgvector: use the existing index (make ingest)."""
    embedder = build_embedder(settings.embeddings)
    if settings.retrieval.store == "memory":
        store: VectorStore = MemoryStore(index_spec(settings))
        docs = [d for m in _manifests(settings) for d in load_manifest(m)]
        chunks = chunk_documents(
            docs, settings.retrieval.chunk_size, settings.retrieval.chunk_overlap
        )
        index_chunks(chunks, store, embedder)
        return store, embedder
    store = build_store(settings)
    n_corpus = store.count(["corpus"])
    n_adv = store.count(["adversarial"])
    if n_corpus == 0 or (needs_adversarial and n_adv == 0):
        raise SystemExit(
            f"Index {store.spec.table_name} has {n_corpus} corpus / {n_adv} adversarial chunks. "
            f"Run: make ingest CHUNK={settings.retrieval.chunk_size} "
            f"OVERLAP={settings.retrieval.chunk_overlap}"
        )
    return store, embedder


def evaluate_case(
    case: Case, result: AgentResult, retriever: Retriever, judge: Any, mock_judge: bool
) -> dict[str, Any]:
    facts = evaluate_facts(case.expected_facts, result.answer)
    tools = evaluate_tool_use(case, result.tool_calls, result.steps)
    safety = evaluate_safety(case, result.answer)

    retrieval: dict[str, Any] | None = None
    if case.expected_sources:
        direct_hits = retriever.search(case.question, k=10, collections=case.collections).hits
        direct_passages = [
            {
                "doc_id": h.chunk.doc_id,
                "page_start": h.chunk.page_start,
                "page_end": h.chunk.page_end,
            }
            for h in direct_hits
        ]
        d_rank = first_relevant_rank(direct_passages, case.expected_sources)
        agent_passages = result.retrieved
        a_rank = first_relevant_rank(agent_passages, case.expected_sources)
        retrieval = {
            "direct": {
                "rank": d_rank,
                "hit_at_1": hit_at_k(d_rank, 1),
                "hit_at_3": hit_at_k(d_rank, 3),
                "hit_at_5": hit_at_k(d_rank, 5),
                "rr_at_10": reciprocal_rank(d_rank, 10),
                "top": [f"{p['doc_id']}:p{p['page_start']}" for p in direct_passages[:5]],
            },
            "agent": {
                "searched": any(c.name == "search_docs" and c.ok for c in result.tool_calls),
                "rank": a_rank,
                "hit_at_5": hit_at_k(a_rank, 5),
                "rr": reciprocal_rank(a_rank),
            },
        }

    judge_out: dict[str, Any] | None = None
    if judge is not None:
        if mock_judge:
            judge_out = judge.grade(
                case,
                result.answer,
                result.tool_calls,
                facts_ok=facts.all_found,
                refusal_ok=safety.refusal_correct,
            )
        else:
            judge_out = judge.grade(case, result.answer, result.tool_calls)

    return {
        "id": case.id,
        "category": case.category,
        "status": case.status,
        "must_refuse": case.must_refuse,
        "question": case.question,
        "answer": result.answer,
        "stopped_reason": result.stopped_reason,
        "error": result.error,
        "steps": result.steps,
        "llm_calls": result.llm_calls,
        "tool_calls": [
            {"name": c.name, "arguments": c.arguments, "ok": c.ok, "error_type": c.error_type}
            for c in result.tool_calls
        ],
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cost_usd": result.cost_usd,
        "latency_s": result.latency_s,
        "trace_id": result.trace_id,
        "facts": {
            "expected": facts.expected,
            "found": facts.found,
            "missing": facts.missing,
            "recall": facts.recall,
            "all_found": facts.all_found,
        },
        "tools": {
            "expected": tools.expected,
            "matched": tools.matched,
            "recall": tools.recall,
            "all_matched": tools.all_matched,
            "unmatched": tools.unmatched_expected,
            "actual_calls": tools.actual_calls,
            "failed_calls": tools.failed_calls,
            "steps": tools.steps,
            "expected_steps": tools.expected_steps,
            "excess_steps": tools.excess_steps,
        },
        "retrieval": retrieval,
        "safety": {
            "refused": safety.refused,
            "refusal_correct": safety.refusal_correct,
            "leaked_system_prompt": safety.leaked_system_prompt,
            "payload_markers_found": safety.payload_markers_found,
            "injection_resisted": safety.injection_resisted,
        },
        "judge": judge_out,
    }


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--mock", action="store_true", help="fake LLM/embeddings/judge, memory store")
    p.add_argument("--agent", default="react", choices=AGENT_NAMES)
    p.add_argument("--provider", choices=["openai", "anthropic"])
    p.add_argument("--model")
    p.add_argument("--chunk-size", type=int)
    p.add_argument("--overlap", type=int)
    p.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--store", choices=["pgvector", "memory"])
    p.add_argument("--judge", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--category", action="append", help="restrict to these categories")
    p.add_argument("--case", action="append", help="restrict to these case ids")
    p.add_argument("--limit", type=int)
    p.add_argument("--label", help="free-text label stored in the result")
    p.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    p.add_argument("--allow-unknown-price", action="store_true")
    p.add_argument(
        "--allow-draft",
        action="store_true",
        help="run a real eval on draft cases (result is flagged as unvalidated)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    overrides: dict[str, Any] = {}
    if args.mock:
        overrides |= {
            "llm.provider": "mock",
            "llm.model": None,
            "embeddings.provider": "mock",
            "embeddings.model": None,
            "retrieval.store": "memory",
        }
    else:
        if args.provider:
            overrides["llm.provider"] = args.provider
        overrides["embeddings.provider"] = "openai"
    if args.model:
        overrides["llm.model"] = args.model
    if args.chunk_size:
        overrides["retrieval.chunk_size"] = args.chunk_size
    if args.overlap is not None:
        overrides["retrieval.chunk_overlap"] = args.overlap
    if args.rerank is not None:
        overrides["retrieval.rerank"] = args.rerank
    if args.store:
        overrides["retrieval.store"] = args.store
    settings = load_settings(overrides=overrides)
    configure_logging("WARNING")

    if not args.mock and settings.llm.provider == "mock":
        print("Real run requested but LLM provider is mock: pass --provider or use --mock.")
        return 2

    cases = load_dataset(args.dataset)
    if args.category:
        cases = [c for c in cases if c.category in args.category]
    if args.case:
        cases = [c for c in cases if c.id in args.case]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("No cases selected.")
        return 2
    drafts = [c.id for c in cases if c.status == "draft"]
    if drafts and not args.mock and not args.allow_draft:
        print(
            f"Refusing a real run: {len(drafts)} of {len(cases)} cases are still 'draft'. "
            "Validate them in evals/datasets/golden.jsonl (status: validated) first, or pass "
            "--allow-draft (the result will be flagged as unvalidated)."
        )
        return 2

    pricing = PricingTable.load(settings.resolve_path(settings.cost_guard.pricing_file))
    if not args.mock:
        estimates = [
            estimate_cost(
                pricing,
                settings.llm.resolved_model,
                len(cases),
                EST_AGENT_CALLS_PER_CASE,
                EST_AGENT_INPUT_TOKENS,
                EST_AGENT_OUTPUT_TOKENS,
            )
        ]
        if args.judge:
            estimates.append(
                estimate_cost(
                    pricing,
                    settings.judge.model,
                    len(cases),
                    1,
                    EST_JUDGE_INPUT_TOKENS,
                    EST_JUDGE_OUTPUT_TOKENS,
                )
            )
        print(
            f"Real evaluation run: {len(cases)} cases, agent={args.agent}, "
            f"max {settings.cost_guard.max_llm_calls} LLM calls / "
            f"{settings.cost_guard.max_total_tokens:,} tokens (hard caps).\n"
        )
        if not confirm_run(
            estimates, assume_yes=args.yes, allow_unknown_price=args.allow_unknown_price
        ):
            print("Aborted.")
            return 1

    now = datetime.now(tz=UTC)
    model_slug = settings.llm.resolved_model.replace("/", "-")
    run_id = (
        f"{now:%Y%m%d-%H%M%S}_{args.agent}_{model_slug}"
        f"_c{settings.retrieval.chunk_size}o{settings.retrieval.chunk_overlap}"
        f"{'_rerank' if settings.retrieval.rerank else ''}"
    )
    out_dir = RESULTS_DIR / "mock" if args.mock else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = settings.resolve_path(settings.logging.runs_dir) / "evals" / f"{run_id}.jsonl"

    needs_adv = any("adversarial" in c.collections for c in cases)
    store, embedder = build_index(settings, needs_adv)
    retriever = Retriever(
        store,
        embedder,
        top_k=settings.retrieval.top_k,
        reranker=build_reranker(settings),
        candidate_k=settings.retrieval.candidate_k,
    )
    registries: dict[tuple[str, ...], ToolRegistry] = {}

    def registry_for(collections: list[str]) -> ToolRegistry:
        key = tuple(collections)
        if key not in registries:
            registries[key] = build_registry(settings, retriever, list(collections))
        return registries[key]

    guard = CostGuard(settings.cost_guard.max_llm_calls, settings.cost_guard.max_total_tokens)
    agent = build_agent(
        args.agent,
        settings,
        guard=guard,
        tools=registry_for(["corpus"]),
        config=AgentConfig(
            max_iterations=settings.agent.max_iterations,
            max_tool_result_chars=settings.agent.max_tool_result_chars,
        ),
        trace_path=trace_path,
    )
    judge: Any = None
    if args.judge:
        judge = MockJudge() if args.mock else Judge(settings.judge, guard, pricing, trace_path)

    records: list[dict[str, Any]] = []
    incomplete: str | None = None
    for i, case in enumerate(cases, start=1):
        agent.tools = registry_for(case.collections)
        agent.config.today = case.reference_date
        result = agent.run(case.question, case_id=case.id, run_id=run_id)
        try:
            record = evaluate_case(case, result, retriever, judge, mock_judge=args.mock)
        except BudgetExceeded as exc:
            incomplete = f"cost guard stopped the run during judging: {exc}"
            break
        records.append(record)
        status = "ok" if record["facts"]["all_found"] in (True, None) else "facts-missing"
        print(
            f"[{i:>2}/{len(cases)}] {case.id:<10} {result.stopped_reason:<15} "
            f"steps={result.steps} tools={len(result.tool_calls)} {status} "
            f"{result.latency_s:5.1f}s"
        )
        if result.stopped_reason == "budget":
            incomplete = f"cost guard stopped the run: {result.error}"
            break

    payload: dict[str, Any] = {
        "run_id": run_id,
        "created_at": now.isoformat(),
        "label": args.label,
        "mock": args.mock,
        "complete": incomplete is None,
        "incomplete_reason": incomplete,
        "config": {
            "agent": args.agent,
            "llm_provider": settings.llm.provider,
            "llm_model": settings.llm.resolved_model,
            "embedding_model": settings.embeddings.resolved_model,
            "store": settings.retrieval.store,
            "index_table": store.spec.table_name,
            "chunk_size": settings.retrieval.chunk_size,
            "chunk_overlap": settings.retrieval.chunk_overlap,
            "top_k": settings.retrieval.top_k,
            "rerank": settings.retrieval.rerank,
            "reranker_model": settings.retrieval.reranker_model
            if settings.retrieval.rerank
            else None,
            "max_iterations": settings.agent.max_iterations,
            "judge_model": (judge.model if judge else None),
            "judge_prompt_sha256_16": dataset_fingerprint(JUDGE_PROMPT_PATH)
            if judge and not args.mock
            else None,
            "git_commit": _git_commit(),
        },
        "dataset": {
            "path": (
                args.dataset.resolve().relative_to(PROJECT_ROOT).as_posix()
                if args.dataset.resolve().is_relative_to(PROJECT_ROOT)
                else str(args.dataset)
            ),
            "sha256_16": dataset_fingerprint(args.dataset),
            "n_cases": len(cases),
            "status_counts": status_counts(cases),
            "validated": not drafts,
        },
        "prices_verified": {
            m: pricing.get(m).verified
            for m in {settings.llm.resolved_model, settings.embeddings.resolved_model}
            | ({settings.judge.model} if args.judge and not args.mock else set())
        },
        "cost_guard": guard.snapshot(),
        "metrics": {"overall": summarize(records), "by_category": summarize_by_category(records)},
        "cases": records,
    }
    json_path = out_dir / f"{run_id}.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    md_path = out_dir / f"{run_id}.md"
    md_path.write_text(render_run_markdown(payload), encoding="utf-8")
    print(f"\nWrote {_rel(json_path)} and {md_path.name}")
    print(f"Traces: {_rel(trace_path)}")
    if incomplete:
        print(f"INCOMPLETE: {incomplete}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
