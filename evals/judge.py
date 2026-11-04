"""LLM-as-judge: grades one answer against the reference with a rubric stored in
evals/judge/judge_prompt.md. Judge calls go through the same cost guard as the agent and are
traced (agent name "judge") in the run's trace file.

Caveats (also stated in the README): the judge is itself an LLM; it may favour verbose
answers or answers written by its own model family, and its scores are only as good as the
reference answers. Deterministic fact checks are reported next to it for that reason.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from agent_eval.config import JudgeConfig, LLMConfig
from agent_eval.cost_guard import BudgetExceeded, CostGuard
from agent_eval.llm import build_llm
from agent_eval.llm.base import Message
from agent_eval.observability.cost import PricingTable
from agent_eval.observability.tracing import Trace, TraceSink
from evals.dataset import Case

PROMPT_PATH = Path(__file__).parent / "judge" / "judge_prompt.md"
MAX_SOURCE_CHARS = 6000
MAX_PASSAGES = 8
MAX_PASSAGE_CHARS = 600


def _fact_line(fact: Any) -> str:
    value = fact.value if fact.value is not None else " / ".join(fact.any_of or [])
    return f"- {fact.description or fact.type}: {value}"


def _source_ref(s: dict[str, Any]) -> str:
    page = s.get("page_start", s.get("page"))
    return (
        f"{s.get('doc_id')}"
        + (f" p.{page}" if page else "")
        + (f" §{s['section']}" if s.get("section") else "")
    )


def render_evidence(tool_calls: list[Any]) -> str:
    """What the agent saw from its tools, labelled as the agent saw it: search passages keep
    their ref (S1, S2... restarting at each search), get_deadline its cited sources,
    calculate_date its result. Without this the judge marks tool-backed citations as invented."""
    lines: list[str] = []
    passages_left = MAX_PASSAGES
    n_search = 0
    for call in tool_calls:
        if not call.ok:
            continue
        data = call.data
        if call.name == "search_docs":
            n_search += 1
            for p in data.get("passages", [])[: max(passages_left, 0)]:
                lines.append(
                    f"[search {n_search}, {p.get('ref')} = {_source_ref(p)}] "
                    f"{p.get('text', '')[:MAX_PASSAGE_CHARS]}"
                )
                passages_left -= 1
        elif call.name == "get_deadline":
            sources = "; ".join(_source_ref(s) for s in data.get("sources", []))
            lines.append(
                f"[get_deadline {data.get('entity_size')}/{data.get('obligation')}] "
                f"date {data.get('date')} ({data.get('obligation_label')}); "
                f"entity: {data.get('entity_definition')}; sources: {sources}"
            )
        else:
            payload = {k: v for k, v in data.items() if k != "text"}
            lines.append(
                f"[{call.name} {json.dumps(call.arguments, ensure_ascii=False)}] "
                f"{json.dumps(payload, ensure_ascii=False)}"
            )
    return "\n".join(lines)[:MAX_SOURCE_CHARS] or "(no tool output)"


def render_prompt(case: Case, answer: str, tool_calls: list[Any]) -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8")
    facts = "\n".join(_fact_line(f) for f in case.expected_facts) or "(none)"
    src = render_evidence(tool_calls)
    return template.format(
        question=case.question,
        expected_answer=case.expected_answer,
        expected_facts=facts,
        must_refuse=str(case.must_refuse).lower(),
        sources=src,
        answer=answer or "(empty answer)",
    )


def parse_verdict(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    keys = ("correctness", "groundedness", "completeness", "score")
    if not all(isinstance(data.get(k), int) and 1 <= data[k] <= 5 for k in keys):
        return None
    # Recompute instead of trusting the model's arithmetic.
    data["score"] = min(data["correctness"], data["groundedness"], data["completeness"])
    data["pass"] = data["score"] >= 4
    return data


class Judge:
    def __init__(
        self,
        cfg: JudgeConfig,
        guard: CostGuard | None,
        pricing: PricingTable,
        trace_path: Path | None,
    ) -> None:
        llm_cfg = LLMConfig(
            provider=cfg.provider,
            model=cfg.model,
            max_output_tokens=cfg.max_output_tokens,
            effort=cfg.effort,
            temperature=None,
        )
        self.llm = build_llm(llm_cfg, guard)
        self.model = cfg.model
        self.pricing = pricing
        self.sink = TraceSink(trace_path)

    def grade(self, case: Case, answer: str, tool_calls: list[Any]) -> dict[str, Any]:
        prompt = render_prompt(case, answer, tool_calls)
        trace = Trace(
            self.sink, self.pricing, "judge", case.question, metadata={"case_id": case.id}
        )
        trace.start(self.llm.name, self.llm.model)
        start = time.perf_counter()
        verdict: dict[str, Any] | None = None
        error = None
        raw = ""
        try:
            response = self.llm.complete(
                "You are a strict, consistent evaluator. Output JSON only.",
                [Message.user(prompt)],
            )
            trace.llm_call(1, response, price_model=self.llm.model)
            raw = response.text
            verdict = parse_verdict(raw)
            if verdict is None:
                error = "unparseable judge output"
        except BudgetExceeded:
            raise
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        trace.end(
            "final_answer" if verdict else "error", 1, time.perf_counter() - start, error=error
        )
        return {
            "model": self.model,
            "score": verdict["score"] if verdict else None,
            "pass": verdict["pass"] if verdict else None,
            "criteria": {k: verdict[k] for k in ("correctness", "groundedness", "completeness")}
            if verdict
            else None,
            "rationale": verdict.get("rationale") if verdict else None,
            "error": error,
            "raw": None if verdict else raw[:500],
            "cost_usd": trace.cost_usd,
            "input_tokens": trace.input_tokens,
            "output_tokens": trace.output_tokens,
        }


class MockJudge:
    """Offline stand-in: passes when every key fact is present and refusal is correct.
    Exists only so the pipeline runs in CI; it measures nothing beyond the fact checks."""

    model = "mock-judge"

    def grade(
        self,
        case: Case,
        answer: str,
        tool_calls: list[Any],
        facts_ok: bool | None = None,
        refusal_ok: bool = True,
    ) -> dict[str, Any]:
        ok = (facts_ok is not False) and refusal_ok
        score = 5 if ok else 2
        return {
            "model": self.model,
            "score": score,
            "pass": ok,
            "criteria": None,
            "rationale": "mock judge (fact checks only)",
            "error": None,
            "raw": None,
            "cost_usd": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
