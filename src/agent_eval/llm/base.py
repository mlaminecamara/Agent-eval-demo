"""Provider-neutral LLM interface.

Agents only speak this vocabulary (Message, ToolSpec, ToolCall, LLMResponse); each provider
adapter translates to and from its own SDK format. Assistant messages keep the provider's raw
content blocks (`provider_raw`) so that a tool loop can replay them unchanged — required by
Anthropic when responses contain thinking blocks.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["user", "assistant", "tool"]
FinishReason = Literal["stop", "tool_calls", "length", "refusal", "error"]
# "none": tools stay declared (a history with tool calls requires them on some APIs) but the
# model must answer in text.
ToolChoice = Literal["auto", "none"]


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema (type: object)
    strict: bool = True


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    # Set when the model produced arguments that are not valid JSON; the agent reports it back
    # to the model as a tool error instead of crashing.
    parse_error: str | None = None


@dataclass
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)  # assistant only
    tool_call_id: str | None = None  # tool only
    is_error: bool = False  # tool only
    provider: str | None = None  # assistant only: which adapter produced provider_raw
    provider_raw: Any = None  # assistant only: native content, replayed as-is

    @staticmethod
    def user(content: str) -> Message:
        return Message(role="user", content=content)

    @staticmethod
    def tool_result(tool_call_id: str, content: str, is_error: bool = False) -> Message:
        return Message(role="tool", content=content, tool_call_id=tool_call_id, is_error=is_error)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    finish_reason: FinishReason
    usage: Usage
    provider: str
    model: str
    latency_s: float
    message: Message  # the assistant turn to append to the conversation
    request_id: str | None = None


class LLMProvider(ABC):
    """One chat-completion call with optional tools. Implementations must be stateless."""

    name: str

    def __init__(
        self,
        model: str,
        temperature: float | None = None,
        max_output_tokens: int = 1024,
        timeout_s: float = 60,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout_s = timeout_s

    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        max_output_tokens: int | None = None,
        tool_choice: ToolChoice = "auto",
    ) -> LLMResponse:
        start = time.perf_counter()
        response = self._complete(
            system,
            messages,
            tools or [],
            max_output_tokens or self.max_output_tokens,
            tool_choice=tool_choice,
        )
        response.latency_s = time.perf_counter() - start
        return response

    @abstractmethod
    def _complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        max_output_tokens: int,
        tool_choice: ToolChoice = "auto",
    ) -> LLMResponse:
        """Provider call. latency_s is filled in by complete()."""
