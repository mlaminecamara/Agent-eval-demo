"""Wrapper that enforces the cost guard around any provider."""

from __future__ import annotations

from agent_eval.cost_guard import CostGuard
from agent_eval.llm.base import LLMProvider, LLMResponse, Message, ToolSpec


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
    ) -> LLMResponse:
        self.guard.before_call()
        response = self.inner.complete(system, messages, tools, max_output_tokens)
        self.guard.after_call(response.usage.input_tokens, response.usage.output_tokens)
        return response

    def _complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        max_output_tokens: int,
    ) -> LLMResponse:
        raise NotImplementedError  # complete() delegates to the inner provider
