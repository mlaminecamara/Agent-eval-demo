"""LLM providers behind one interface, selected by config."""

from __future__ import annotations

from typing import Any

from agent_eval.config import LLMConfig
from agent_eval.cost_guard import CostGuard
from agent_eval.llm.base import LLMProvider, LLMResponse, Message, ToolCall, ToolSpec, Usage
from agent_eval.llm.guarded import GuardedLLM

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "Message",
    "ToolCall",
    "ToolSpec",
    "Usage",
    "build_llm",
]


def build_llm(cfg: LLMConfig, guard: CostGuard | None = None) -> LLMProvider:
    common: dict[str, Any] = {
        "temperature": cfg.temperature,
        "max_output_tokens": cfg.max_output_tokens,
        "timeout_s": cfg.timeout_s,
    }
    provider: LLMProvider
    if cfg.provider == "mock":
        from agent_eval.llm.mock import MockLLM

        provider = MockLLM(cfg.resolved_model, **common)
    elif cfg.provider == "openai":
        from agent_eval.llm.openai_provider import OpenAIProvider

        provider = OpenAIProvider(cfg.resolved_model, effort=cfg.effort, **common)
    elif cfg.provider == "anthropic":
        from agent_eval.llm.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(cfg.resolved_model, effort=cfg.effort, **common)
    else:  # pragma: no cover - guarded by the Literal type in config
        raise ValueError(f"Unknown LLM provider {cfg.provider!r}")
    return GuardedLLM(provider, guard) if guard else provider
