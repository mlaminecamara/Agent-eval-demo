"""Deterministic fake LLM for offline tests and CI.

It follows fixed rules so that the agent loop, tool calls, tracing and metrics can be exercised
end to end without an API key. Its answers say nothing about real model quality; every result
produced with it is labelled provider=mock.

Rules (first match wins), applied to the conversation so far:
1. Last message is a tool result -> final answer quoting the tool results.
2. User question has no domain keyword -> refusal.
3. Tools available -> call `search_docs` (or the first tool) with the question.
4. Otherwise -> echo answer.
"""

from __future__ import annotations

import hashlib
import math
import unicodedata

from agent_eval.llm.base import LLMProvider, LLMResponse, Message, ToolCall, ToolSpec, Usage

DOMAIN_KEYWORDS = (
    "factur",
    "invoice",
    "e-reporting",
    "ereporting",
    "pdp",
    "plateforme",
    "ppf",
    "tva",
    "ubl",
    "cii",
    "echeance",
    "obligation",
    "entreprise",
    "pme",
    "eti",
    "micro",
    "annuaire",
    "emission",
    "reception",
    "calendrier",
    "date",
    "agree",
    "immatricul",
    "dgfip",
)

REFUSAL_TEXT = (
    "Je ne peux pas répondre à cette question : elle sort du périmètre de la réforme de la "
    "facturation électronique couvert par mes sources."
)


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def estimate_tokens(text: str) -> int:
    """Crude, deterministic token estimate (~4 characters per token)."""
    return max(1, math.ceil(len(text) / 4))


def is_in_domain(text: str) -> bool:
    norm = _normalize(text)
    return any(k in norm for k in DOMAIN_KEYWORDS)


class MockLLM(LLMProvider):
    name = "mock"

    def _complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        max_output_tokens: int,
    ) -> LLMResponse:
        question = next((m.content for m in reversed(messages) if m.role == "user"), "")
        tool_calls: list[ToolCall] = []

        if messages and messages[-1].role == "tool":
            results = [m.content for m in messages if m.role == "tool"]
            text = "Réponse (mock) fondée sur les sources : " + " | ".join(r[:300] for r in results)
        elif not is_in_domain(question):
            text = REFUSAL_TEXT
        elif tools:
            tool = next((t for t in tools if t.name == "search_docs"), tools[0])
            call_id = (
                "call_" + hashlib.sha1(f"{len(messages)}:{question}".encode()).hexdigest()[:12]
            )
            tool_calls = [ToolCall(id=call_id, name=tool.name, arguments={"query": question})]
            text = ""
        else:
            text = f"Réponse (mock) : {question}"

        text = text[: max_output_tokens * 4]
        prompt_chars = len(system) + sum(len(m.content) for m in messages)
        usage = Usage(
            input_tokens=estimate_tokens("x" * prompt_chars),
            output_tokens=estimate_tokens(text) if text else 8 * len(tool_calls),
        )
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
            usage=usage,
            provider=self.name,
            model=self.model,
            latency_s=0.0,
            message=Message(
                role="assistant", content=text, tool_calls=tool_calls, provider=self.name
            ),
        )
