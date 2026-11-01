"""OpenAI adapter (Chat Completions API with function tools)."""

from __future__ import annotations

import json
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
    "stop": "stop",
    "tool_calls": "tool_calls",
    "length": "length",
    "content_filter": "refusal",
}


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, model: str, effort: str | None = None, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self.effort = effort
        from openai import OpenAI  # imported lazily so mock mode needs no key

        self._client = OpenAI(timeout=self.timeout_s)

    @staticmethod
    def _to_openai_messages(system: str, messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for m in messages:
            if m.role == "user":
                out.append({"role": "user", "content": m.content})
            elif m.role == "tool":
                out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
            elif m.provider == "openai" and m.provider_raw is not None:
                out.append(m.provider_raw)
            else:
                msg: dict[str, Any] = {"role": "assistant", "content": m.content or None}
                if m.tool_calls:
                    msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                            },
                        }
                        for tc in m.tool_calls
                    ]
                out.append(msg)
        return out

    @staticmethod
    def _to_openai_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                    "strict": t.strict,
                },
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
            "messages": self._to_openai_messages(system, messages),
            # Includes reasoning tokens on reasoning models.
            "max_completion_tokens": max_output_tokens,
        }
        # Reasoning models (GPT-5 family) reject non-default temperature; only send it when set.
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.effort:
            kwargs["reasoning_effort"] = self.effort
        if tools:
            kwargs["tools"] = self._to_openai_tools(tools)
            kwargs["tool_choice"] = "auto"

        completion = self._client.chat.completions.create(**kwargs)
        choice = completion.choices[0]
        msg = choice.message

        tool_calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            fn = getattr(tc, "function", None)
            if fn is None:  # non-function tool call types are not used here
                continue
            try:
                args = json.loads(fn.arguments or "{}")
                parse_error = None if isinstance(args, dict) else "arguments must be a JSON object"
                if not isinstance(args, dict):
                    args = {}
            except json.JSONDecodeError as exc:
                args, parse_error = {}, f"invalid JSON arguments: {exc}"
            tool_calls.append(
                ToolCall(id=tc.id, name=fn.name, arguments=args, parse_error=parse_error)
            )

        finish = _FINISH_MAP.get(choice.finish_reason or "stop", "stop")
        if getattr(msg, "refusal", None):
            finish = "refusal"
        text = msg.content or (msg.refusal if getattr(msg, "refusal", None) else "") or ""
        usage = Usage(
            input_tokens=completion.usage.prompt_tokens if completion.usage else 0,
            output_tokens=completion.usage.completion_tokens if completion.usage else 0,
        )
        raw = msg.model_dump(mode="json", exclude_none=True)
        raw.pop("refusal", None)
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            finish_reason=finish,
            usage=usage,
            provider=self.name,
            model=completion.model or self.model,
            latency_s=0.0,
            message=Message(
                role="assistant",
                content=text,
                tool_calls=tool_calls,
                provider=self.name,
                provider_raw=raw,
            ),
            request_id=getattr(completion, "_request_id", None),
        )
