"""Plan-and-Execute agent: plan once, run each step, then write the answer.

1. Plan: one LLM call whose only tool is `submit_plan` (a list of {tool, goal} steps, capped
   at max_iterations - 2). An out-of-scope question gets the refusal instead of a plan; if the
   model answers in text without a plan, a one-step search plan is used (flagged in extra).
2. Execute: one LLM call per step, with all tools available and the plan and current step in
   the system prompt. The conversation (question, tool calls, tool results) is shared across
   steps, so a later step can use an earlier result (e.g. a date from get_deadline). A step
   where the model calls no tool is skipped.
3. Synthesize: one last call with tool_choice "none" writes the answer from the results.

No replanning: a plan that turns out wrong is not revised. This is the main behavioural
difference with ReAct, which decides each next step after seeing the previous result.
Same guards as ReAct: tool errors go back to the model, budget/provider errors end the run.
"""

from __future__ import annotations

import time
from typing import Any

from agent_eval.agents.base import Agent, AgentResult, StoppedReason, ToolCallRecord
from agent_eval.agents.prompts import (
    EXECUTOR_SUFFIX,
    PLANNER_SUFFIX,
    SYNTHESIS_SUFFIX,
    build_system_prompt,
    is_refusal,
)
from agent_eval.cost_guard import BudgetExceeded
from agent_eval.llm.base import Message, ToolSpec
from agent_eval.observability.logging import get_logger

log = get_logger(__name__)

PLAN_TOOLS = ("search_docs", "get_deadline", "calculate_date")

SUBMIT_PLAN = ToolSpec(
    name="submit_plan",
    description=(
        "Enregistre le plan : liste ordonnée d'étapes, chacune avec l'outil à utiliser et ce "
        "que l'étape doit établir."
    ),
    parameters={
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "tool": {"type": "string", "enum": list(PLAN_TOOLS)},
                        "goal": {"type": "string"},
                    },
                    "required": ["tool", "goal"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["steps"],
        "additionalProperties": False,
    },
)


def parse_plan(arguments: dict[str, Any], max_steps: int) -> list[dict[str, str]]:
    steps = arguments.get("steps")
    if not isinstance(steps, list):
        return []
    plan = [
        {"tool": str(s.get("tool", "")), "goal": str(s.get("goal", "")).strip()}
        for s in steps
        if isinstance(s, dict)
    ]
    return [s for s in plan if s["tool"] in PLAN_TOOLS][:max_steps]


def format_plan(plan: list[dict[str, str]]) -> str:
    return "\n".join(f"{i}. [{s['tool']}] {s['goal']}" for i, s in enumerate(plan, start=1))


class PlanExecuteAgent(Agent):
    name = "plan_execute"

    @property
    def max_plan_steps(self) -> int:
        return max(1, self.config.max_iterations - 2)

    def run(self, question: str, **metadata: Any) -> AgentResult:
        start = time.perf_counter()
        trace = self.new_trace(question, **metadata)
        trace.start(self.llm.name, self.llm.model, max_iterations=self.config.max_iterations)

        base = build_system_prompt(self.config.today)
        specs = self.tools.specs()
        messages: list[Message] = [Message.user(question)]
        records: list[ToolCallRecord] = []
        plan: list[dict[str, str]] = []
        plan_fallback = False
        answer = ""
        stopped: StoppedReason = "final_answer"
        error: str | None = None
        step = 0

        try:
            # 1. Plan
            step += 1
            planner = base + PLANNER_SUFFIX.format(max_steps=self.max_plan_steps)
            response = self._call_llm(planner, messages, [SUBMIT_PLAN], step, trace)
            call = next((c for c in response.tool_calls if c.name == SUBMIT_PLAN.name), None)
            if response.finish_reason == "refusal":
                stopped, answer = "provider_refusal", response.text
            elif call is None and is_refusal(response.text):
                answer = response.text
            else:
                plan = parse_plan(call.arguments, self.max_plan_steps) if call else []
                if not plan:
                    plan_fallback = True
                    plan = [{"tool": "search_docs", "goal": "Rechercher dans les sources."}]

                # 2. Execute
                plan_text = format_plan(plan)
                for index, item in enumerate(plan, start=1):
                    step += 1
                    system = base + EXECUTOR_SUFFIX.format(
                        plan=plan_text, index=index, total=len(plan), goal=item["goal"]
                    )
                    response = self._call_llm(system, messages, specs, step, trace)
                    if response.finish_reason == "refusal":
                        stopped, answer = "provider_refusal", response.text
                        break
                    if not response.tool_calls:
                        continue  # nothing to run; keep the conversation free of stray text
                    messages.append(response.message)
                    messages.extend(
                        self._execute_tool_calls(response.tool_calls, step, trace, records)
                    )

                # 3. Synthesize
                if stopped == "final_answer":
                    step += 1
                    response = self._call_llm(
                        base + SYNTHESIS_SUFFIX, messages, specs, step, trace, "none"
                    )
                    answer = response.text
                    if response.finish_reason == "refusal":
                        stopped = "provider_refusal"
        except BudgetExceeded as exc:
            stopped, error = "budget", str(exc)
        except Exception as exc:
            stopped, error = "error", f"{type(exc).__name__}: {exc}"
            log.warning("agent run failed", extra={"trace_id": trace.trace_id, "error": error})

        latency = time.perf_counter() - start
        return self._result(
            question,
            trace,
            answer,
            stopped,
            step,
            records,
            latency,
            error,
            plan=plan,
            plan_fallback=plan_fallback,
        )
