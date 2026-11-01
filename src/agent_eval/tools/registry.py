"""Tool registry: the single entry point agents use to list and execute tools."""

from __future__ import annotations

from typing import Any

from agent_eval.config import Settings
from agent_eval.llm.base import ToolCall, ToolSpec
from agent_eval.retrieval.retriever import Retriever, build_retriever
from agent_eval.tools.base import Tool, ToolResult
from agent_eval.tools.calculate_date import CalculateDateTool
from agent_eval.tools.get_deadline import DeadlineTable, GetDeadlineTool
from agent_eval.tools.search_docs import SearchDocsTool

DEADLINES_PATH = "data/tools/deadlines.json"


class ToolRegistry:
    def __init__(self, tools: list[Tool[Any]]) -> None:
        names = [t.name for t in tools]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate tool names: {names}")
        self.tools = {t.name: t for t in tools}

    def specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self.tools.values()]

    def execute(self, call: ToolCall) -> ToolResult:
        tool = self.tools.get(call.name)
        if tool is None:
            return ToolResult(
                tool=call.name,
                ok=False,
                error_type="unknown_tool",
                content=f"Unknown tool '{call.name}'. Available tools: {', '.join(self.tools)}.",
            )
        if call.parse_error:
            return ToolResult(
                tool=call.name,
                ok=False,
                error_type="validation",
                content=f"Invalid arguments for {call.name}: {call.parse_error}. Retry with a "
                "JSON object matching the schema.",
            )
        return tool.run(call.arguments)


def build_registry(
    settings: Settings,
    retriever: Retriever | None = None,
    collections: list[str] | None = None,
) -> ToolRegistry:
    return ToolRegistry(
        [
            SearchDocsTool(retriever or build_retriever(settings), collections),
            GetDeadlineTool(DeadlineTable.load(settings.resolve_path(DEADLINES_PATH))),
            CalculateDateTool(),
        ]
    )
