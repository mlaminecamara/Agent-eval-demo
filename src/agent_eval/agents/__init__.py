"""Agents behind one interface (Agent.run -> AgentResult)."""

from __future__ import annotations

from pathlib import Path

from agent_eval.agents.base import Agent, AgentConfig, AgentResult
from agent_eval.config import Settings
from agent_eval.cost_guard import CostGuard
from agent_eval.llm import build_llm
from agent_eval.observability.cost import PricingTable
from agent_eval.observability.tracing import default_trace_path
from agent_eval.tools.registry import ToolRegistry, build_registry

__all__ = ["Agent", "AgentConfig", "AgentResult", "build_agent"]

AGENT_NAMES = ("react", "plan_execute", "langgraph")


def build_agent(
    name: str,
    settings: Settings,
    guard: CostGuard | None = None,
    tools: ToolRegistry | None = None,
    config: AgentConfig | None = None,
    trace_path: Path | None = None,
) -> Agent:
    llm = build_llm(settings.llm, guard)
    tools = tools or build_registry(settings)
    pricing = PricingTable.load(settings.resolve_path(settings.cost_guard.pricing_file))
    config = config or AgentConfig(
        max_iterations=settings.agent.max_iterations,
        max_tool_result_chars=settings.agent.max_tool_result_chars,
        today=settings.agent.today,
    )
    trace_path = trace_path or default_trace_path(settings.resolve_path(settings.logging.runs_dir))
    if name == "react":
        from agent_eval.agents.react_agent import ReActAgent

        return ReActAgent(llm, tools, pricing, config, trace_path)
    if name == "plan_execute":
        from agent_eval.agents.plan_execute_agent import PlanExecuteAgent

        return PlanExecuteAgent(llm, tools, pricing, config, trace_path)
    if name == "langgraph":
        from agent_eval.agents.langgraph_agent import LangGraphAgent

        return LangGraphAgent(llm, tools, pricing, config, trace_path)
    raise ValueError(f"Unknown agent {name!r}; available: {', '.join(AGENT_NAMES)}")
