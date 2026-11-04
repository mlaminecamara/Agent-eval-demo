"""LangGraph agent: the tool loop as an explicit state graph, plus a verification node.

    START -> agent --(tool calls)--> tools --(step < max)--> agent
               |                       \\--(step = max)--> force_final -> END
               |--(text answer)--> verify --(unsupported dates, one retry)--> agent
               |                      \\--(ok, refusal or no retry left)--> END
               \\--(provider refusal)--> END

- agent / tools / force_final do what the native ReAct loop does (same prompt, tools, guards,
  same LLM providers: LangGraph only orchestrates, no LangChain model wrappers).
- verify is the addition: every date written in the answer must appear in the question, a
  tool result or be today's date (agents/grounding.py, deterministic, no LLM call). If not,
  the model is told which dates are unsupported and gets one more turn. Differences between
  this agent and "react" therefore measure this check, not the framework.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from agent_eval.agents.base import Agent, AgentResult, StoppedReason, ToolCallRecord
from agent_eval.agents.grounding import unsupported_dates
from agent_eval.agents.prompts import (
    FORCE_FINAL_MESSAGE,
    VERIFY_MESSAGE,
    build_system_prompt,
    is_refusal,
)
from agent_eval.cost_guard import BudgetExceeded
from agent_eval.llm.base import Message
from agent_eval.observability.logging import get_logger
from agent_eval.observability.tracing import Trace

log = get_logger(__name__)

MAX_VERIFY_RETRIES = 1


class GraphState(TypedDict):
    messages: list[Message]
    answer: str
    stopped: StoppedReason
    verify_retries: int


@dataclass
class _RunContext:
    """Per-run values the nodes share outside the graph state, so that they survive an
    exception raised inside a node (budget, API error)."""

    question: str
    system: str
    trace: Trace
    step: int = 0
    records: list[ToolCallRecord] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)


class LangGraphAgent(Agent):
    name = "langgraph"

    def _build_graph(self, ctx: _RunContext) -> Any:
        specs = self.tools.specs()
        max_iter = self.config.max_iterations

        def agent(state: GraphState) -> dict[str, Any]:
            ctx.step += 1
            response = self._call_llm(ctx.system, state["messages"], specs, ctx.step, ctx.trace)
            update: dict[str, Any] = {
                "messages": [*state["messages"], response.message],
                "answer": response.text,
            }
            if response.finish_reason == "refusal":
                update["stopped"] = "provider_refusal"
            return update

        def tools(state: GraphState) -> dict[str, Any]:
            calls = state["messages"][-1].tool_calls
            results = self._execute_tool_calls(calls, ctx.step, ctx.trace, ctx.records)
            return {"messages": [*state["messages"], *results]}

        def force_final(state: GraphState) -> dict[str, Any]:
            ctx.step += 1
            messages = [*state["messages"], Message.user(FORCE_FINAL_MESSAGE)]
            response = self._call_llm(ctx.system, messages, specs, ctx.step, ctx.trace, "none")
            return {
                "messages": [*messages, response.message],
                "answer": response.text,
                "stopped": "max_iterations",
            }

        def verify(state: GraphState) -> dict[str, Any]:
            answer = state["answer"]
            if is_refusal(answer):
                return {}
            evidence = [ctx.question] + [m.content for m in state["messages"] if m.role == "tool"]
            bad = [d.isoformat() for d in unsupported_dates(answer, evidence, self.config.today)]
            ctx.unsupported = bad
            if not bad or state["verify_retries"] >= MAX_VERIFY_RETRIES or ctx.step >= max_iter:
                return {}
            critique = Message.user(VERIFY_MESSAGE.format(dates=", ".join(bad)))
            return {
                "messages": [*state["messages"], critique],
                "verify_retries": state["verify_retries"] + 1,
            }

        def after_agent(state: GraphState) -> Literal["tools", "verify", "__end__"]:
            if state["stopped"] == "provider_refusal":
                return "__end__"
            return "tools" if state["messages"][-1].tool_calls else "verify"

        def after_tools(state: GraphState) -> Literal["agent", "force_final"]:
            return "force_final" if ctx.step >= max_iter else "agent"

        def after_verify(state: GraphState) -> Literal["agent", "__end__"]:
            # verify appended a critique (a user message) only when it wants another turn
            return "agent" if state["messages"][-1].role == "user" else "__end__"

        graph = StateGraph(GraphState)
        graph.add_node("agent", agent)
        graph.add_node("tools", tools)
        graph.add_node("force_final", force_final)
        graph.add_node("verify", verify)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", after_agent)
        graph.add_conditional_edges("tools", after_tools)
        graph.add_conditional_edges("verify", after_verify)
        graph.add_edge("force_final", END)
        return graph.compile()

    def run(self, question: str, **metadata: Any) -> AgentResult:
        start = time.perf_counter()
        trace = self.new_trace(question, **metadata)
        trace.start(self.llm.name, self.llm.model, max_iterations=self.config.max_iterations)
        ctx = _RunContext(question, build_system_prompt(self.config.today), trace)

        state: GraphState = {
            "messages": [Message.user(question)],
            "answer": "",
            "stopped": "final_answer",
            "verify_retries": 0,
        }
        error: str | None = None
        try:
            graph = self._build_graph(ctx)
            limit = 4 * self.config.max_iterations + 10  # node visits; the step cap binds first
            for snapshot in graph.stream(
                state, config={"recursion_limit": limit}, stream_mode="values"
            ):
                state = snapshot
        except BudgetExceeded as exc:
            state["stopped"], error = "budget", str(exc)
        except Exception as exc:
            state["stopped"], error = "error", f"{type(exc).__name__}: {exc}"
            log.warning("agent run failed", extra={"trace_id": trace.trace_id, "error": error})

        latency = time.perf_counter() - start
        answer = state["answer"] if state["stopped"] not in ("budget", "error") else ""
        return self._result(
            question,
            trace,
            answer,
            state["stopped"],
            ctx.step,
            ctx.records,
            latency,
            error,
            verify_retries=state["verify_retries"],
            unsupported_dates=ctx.unsupported,
        )
