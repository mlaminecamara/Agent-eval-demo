"""CLI: ask one question to an agent.

    uv run python -m agent_eval.agents "Question ?" [--agent react] [--provider openai|anthropic]

Uses the providers from config.yaml/.env unless --provider is given. Real providers are
billed: the run is capped by the cost guard (max_iterations + 1 LLM calls) and the cost is
printed at the end. The trace is appended to runs/traces/<date>.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys

from agent_eval.agents import AGENT_NAMES, build_agent
from agent_eval.config import load_settings
from agent_eval.cost_guard import CostGuard
from agent_eval.observability.logging import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--agent", default="react", choices=AGENT_NAMES)
    parser.add_argument("--provider", choices=["mock", "openai", "anthropic"])
    parser.add_argument("--model")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = parser.parse_args(argv)
    # Corpus passages contain characters (e.g. "➢") that cp1252 consoles cannot encode.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    overrides: dict[str, object] = {}
    if args.provider:
        overrides["llm.provider"] = args.provider
        overrides["embeddings.provider"] = "mock" if args.provider == "mock" else "openai"
    if args.model:
        overrides["llm.model"] = args.model
    settings = load_settings(overrides=overrides)
    configure_logging(settings.logging.level)

    guard = CostGuard(
        max_llm_calls=settings.agent.max_iterations + 1,
        max_total_tokens=settings.cost_guard.max_total_tokens,
    )
    agent = build_agent(args.agent, settings, guard=guard)
    result = agent.run(args.question)

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str))
        return 0
    print(result.answer or f"(no answer: {result.stopped_reason} {result.error or ''})")
    print("\n---")
    tools = ", ".join(f"{c.name}{'' if c.ok else '(error)'}" for c in result.tool_calls) or "-"
    cost = "unknown" if result.cost_usd is None else f"${result.cost_usd:.5f}"
    print(
        f"agent={result.agent} model={result.provider}:{result.model} "
        f"stopped={result.stopped_reason} steps={result.steps} tools=[{tools}]\n"
        f"tokens={result.input_tokens}+{result.output_tokens} cost={cost} "
        f"latency={result.latency_s:.2f}s trace_id={result.trace_id}"
    )
    return 0 if result.stopped_reason in ("final_answer", "max_iterations") else 1


if __name__ == "__main__":
    sys.exit(main())
