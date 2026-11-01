"""Agent tools: search_docs, get_deadline, calculate_date."""

from agent_eval.tools.base import Tool, ToolResult
from agent_eval.tools.registry import ToolRegistry, build_registry

__all__ = ["Tool", "ToolRegistry", "ToolResult", "build_registry"]
