"""Deterministic fake LLM for offline tests and CI.

It follows fixed rules so that the agent loops, tool calls, tracing and metrics can be
exercised end to end without an API key. Its answers say nothing about real model quality;
every result produced with it is labelled provider=mock.

Rules, applied to the conversation so far (the question is the first user message):
1. No domain keyword in the question -> refusal starting with "HORS PÉRIMÈTRE".
2. Tools offered -> the next tool in this fixed chain that applies and was not called yet:
   a. search_docs(query=question)
   b. get_deadline, if the question names an entity size and an obligation
   c. calculate_date(diff), if the question contains an ISO date and asks for a number of
      days, using the date returned by get_deadline
3. Otherwise -> final answer quoting the tool results (dates and day counts first).
When `submit_plan` is offered (Plan-and-Execute planner), the chain of rule 2 is returned as
a plan. With tool_choice "none", rule 3 applies.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from typing import Any

from agent_eval.llm.base import (
    LLMProvider,
    LLMResponse,
    Message,
    ToolCall,
    ToolChoice,
    ToolSpec,
    Usage,
)

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
    "pme",
    "eti",
    "micro",
    "annuaire",
    "emission",
    "emettre",
    "reception",
    "recevoir",
    "calendrier",
    "agree",
    "immatricul",
    "dgfip",
)

# Must start with agents.prompts.REFUSAL_MARKER (kept literal here to avoid an import cycle;
# a test checks they match).
REFUSAL_TEXT = (
    "HORS PÉRIMÈTRE : je ne peux pas répondre à cette question, elle sort du périmètre de la "
    "réforme de la facturation électronique couvert par mes sources."
)

_SIZES = (
    ("GE", ("grande entreprise", "grandes entreprises")),
    ("ETI", ("taille intermediaire", " eti")),
    ("MICRO", ("micro", "auto-entrepreneur", "autoentrepreneur", "tpe")),
    ("PME", ("pme", "petite", "moyenne")),
)
_OBLIGATIONS = (
    ("e_reporting", ("e-reporting", "ereporting", "donnees de transaction")),
    ("emission", ("emettre", "emission", "emet")),
    ("reception", ("recevoir", "reception", "recoi")),
)
_ISO_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def estimate_tokens(text: str) -> int:
    """Crude, deterministic token estimate (~4 characters per token)."""
    return max(1, math.ceil(len(text) / 4))


def is_in_domain(text: str) -> bool:
    norm = _normalize(text)
    return any(k in norm for k in DOMAIN_KEYWORDS)


def _detect(norm: str, table: tuple[tuple[str, tuple[str, ...]], ...]) -> str | None:
    padded = f" {norm} "
    return next((key for key, words in table if any(w in padded for w in words)), None)


def _called(messages: list[Message]) -> dict[str, str]:
    """tool name -> content of its (last) result."""
    names = {tc.id: tc.name for m in messages for tc in m.tool_calls}
    return {names.get(m.tool_call_id or "", "?"): m.content for m in messages if m.role == "tool"}


def _final_answer(results: dict[str, str]) -> str:
    facts: list[str] = []
    for name, content in results.items():
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            for key in ("date", "result_date", "days_between"):
                if key in data:
                    facts.append(f"{name}: {key}={data[key]}")
        else:
            facts.append(f"{name}: {content[:300]}")
    return "Réponse (mock) fondée sur les sources [S1] : " + " | ".join(facts)


def _plan(question: str) -> list[dict[str, str]]:
    """The tool chain of rule 2, as a plan (for the Plan-and-Execute planner)."""
    norm = _normalize(question)
    steps = [{"tool": "search_docs", "goal": "chercher les passages pertinents"}]
    if _detect(norm, _SIZES) and _detect(norm, _OBLIGATIONS):
        steps.append({"tool": "get_deadline", "goal": "obtenir la date d'obligation"})
        if _ISO_DATE.search(question) and "jour" in norm:
            steps.append({"tool": "calculate_date", "goal": "compter les jours"})
    return steps


class MockLLM(LLMProvider):
    name = "mock"

    def _next_call(
        self, question: str, messages: list[Message], tools: list[ToolSpec]
    ) -> tuple[str, dict[str, Any]] | None:
        available = {t.name for t in tools}
        done = _called(messages)
        norm = _normalize(question)
        if "submit_plan" in available:
            return "submit_plan", {"steps": _plan(question)}
        if "search_docs" in available and "search_docs" not in done:
            return "search_docs", {"query": question[:500], "top_k": None}
        size, obligation = _detect(norm, _SIZES), _detect(norm, _OBLIGATIONS)
        if "get_deadline" in available and "get_deadline" not in done and size and obligation:
            return "get_deadline", {"entity_size": size, "obligation": obligation}
        iso = _ISO_DATE.search(question)
        if (
            "calculate_date" in available
            and "calculate_date" not in done
            and iso
            and "jour" in norm
            and "get_deadline" in done
        ):
            try:
                deadline = json.loads(done["get_deadline"]).get("date")
            except json.JSONDecodeError:
                deadline = None
            if deadline:
                return "calculate_date", {
                    "operation": "diff",
                    "start_date": iso.group(1),
                    "days": None,
                    "months": None,
                    "end_date": deadline,
                }
        return None

    def _complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        max_output_tokens: int,
        tool_choice: ToolChoice = "auto",
    ) -> LLMResponse:
        question = next((m.content for m in messages if m.role == "user"), "")
        tool_calls: list[ToolCall] = []

        if not is_in_domain(question):
            text = REFUSAL_TEXT
        else:
            nxt = (
                self._next_call(question, messages, tools)
                if tools and tool_choice == "auto"
                else None
            )
            if nxt is None:
                text = _final_answer(_called(messages))
            else:
                name, args = nxt
                call_id = (
                    "call_"
                    + hashlib.sha1(f"{len(messages)}:{name}:{question}".encode()).hexdigest()[:12]
                )
                tool_calls = [ToolCall(id=call_id, name=name, arguments=args)]
                text = ""

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
