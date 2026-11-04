"""Wrapper that enforces the cost guard around any provider."""

from __future__ import annotations

from agent_eval.cost_guard import CostGuard
from agent_eval.llm.base import LLMProvider, LLMResponse, Message, ToolChoice, ToolSpec


class GuardedLLM(LLMProvider):
    """Checks the budget before each call and records usage after it."""

    def __init__(self, inner: LLMProvider, guard: CostGuard) -> None:
        super().__init__(inner.model, inner.temperature, inner.max_output_tokens, inner.timeout_s)
        self.inner = inner
        self.guard = guard
        self.name = inner.name

    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        max_output_tokens: int | None = None,
        tool_choice: ToolChoice = "auto",
    ) -> LLMResponse:
        self.guard.before_call()
        response = self.inner.complete(system, messages, tools, max_output_tokens, tool_choice)
        self.guard.after_call(response.usage.input_tokens, response.usage.output_tokens)
        return response

    def _complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        max_output_tokens: int,
        tool_choice: ToolChoice = "auto",
    ) -> LLMResponse:
        raise NotImplementedError  # complete() delegates to the inner provider
