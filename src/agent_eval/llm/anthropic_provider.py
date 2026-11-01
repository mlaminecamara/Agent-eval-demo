"""Anthropic adapter (Messages API with client tools).

Notes on the current API that shape this adapter:
- Sampling parameters (temperature/top_p/top_k) are rejected by current models, so `temperature`
  from config is not sent. Determinism comes from the eval design, not from sampling.
- Forced tool_choice ("any"/"tool") is rejected by current models; we use "auto" and steer
  from the system prompt, with `strict: true` on tool schemas.
- Responses may contain thinking blocks. They must be replayed unchanged on the next request, so
  assistant turns keep the raw content blocks and are never edited.
- All tool results answering one assistant turn go back in a single user message.
"""

from __future__ import annotations

from typing import Any

from agent_eval.llm.base import (
    FinishReason,
    LLMProvider,
    LLMResponse,
    Message,
    ToolCall,
    ToolSpec,
    Usage,
)

_FINISH_MAP: dict[str, FinishReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "refusal": "refusal",
}


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, model: str, effort: str | None = None, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        import anthropic  # imported lazily so mock mode needs no key

        self._client = anthropic.Anthropic(timeout=self.timeout_s)
        self.effort = effort

    @staticmethod
    def _to_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        pending_results: list[dict[str, Any]] = []

        def flush_results() -> None:
            if pending_results:
                out.append({"role": "user", "content": list(pending_results)})
                pending_results.clear()

        for m in messages:
            if m.role == "tool":
                block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id,
                    "content": m.content,
                }
                if m.is_error:
                    block["is_error"] = True
                pending_results.append(block)
                continue
            flush_results()
            if m.role == "user":
                out.append({"role": "user", "content": m.content})
            elif m.provider == "anthropic" and m.provider_raw is not None:
                out.append({"role": "assistant", "content": m.provider_raw})
            else:
                content: list[dict[str, Any]] = []
                if m.content:
                    content.append({"type": "text", "text": m.content})
                content.extend(
                    {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                    for tc in m.tool_calls
                )
                out.append({"role": "assistant", "content": content})
        flush_results()
        return out

    @staticmethod
    def _to_anthropic_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
                "strict": t.strict,
            }
            for t in tools
        ]

    def _complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        max_output_tokens: int,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "system": system,
            "messages": self._to_anthropic_messages(messages),
            "max_tokens": max_output_tokens,
        }
        if tools:
            kwargs["tools"] = self._to_anthropic_tools(tools)
            kwargs["tool_choice"] = {"type": "auto"}
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}

        response = self._client.messages.create(**kwargs)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                args = block.input if isinstance(block.input, dict) else {}
                parse_error = None if isinstance(block.input, dict) else "input is not an object"
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=args, parse_error=parse_error)
                )

        text = "".join(text_parts)
        finish = _FINISH_MAP.get(response.stop_reason or "end_turn", "stop")
        raw = [b.model_dump(mode="json", exclude_none=True) for b in response.content]
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            finish_reason=finish,
            usage=Usage(
                input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens
            ),
            provider=self.name,
            model=response.model or self.model,
            latency_s=0.0,
            message=Message(
                role="assistant",
                content=text,
                tool_calls=tool_calls,
                provider=self.name,
                provider_raw=raw,
            ),
            request_id=getattr(response, "_request_id", None),
        )
