"""End-to-end tests of the ReAct agent in mock mode (no network, no API key)."""

import json
from datetime import date

import pytest

from agent_eval.agents import build_agent
from agent_eval.agents.base import AgentConfig
from agent_eval.agents.prompts import (
    CANARY,
    REFUSAL_MARKER,
    build_system_prompt,
    is_refusal,
    leaks_system_prompt,
)
from agent_eval.agents.react_agent import ReActAgent
from agent_eval.config import PROJECT_ROOT
from agent_eval.cost_guard import CostGuard
from agent_eval.llm.base import LLMProvider, LLMResponse, Message, ToolCall, Usage
from agent_eval.llm.mock import REFUSAL_TEXT, MockLLM
from agent_eval.observability.cost import PricingTable

PRICING = PricingTable.load(PROJECT_ROOT / "config" / "pricing.yaml")
TODAY = date(2026, 11, 1)


def _agent(registry, tmp_path, llm=None, **cfg):
    config = AgentConfig(today=TODAY, **cfg)
    return ReActAgent(
        llm or MockLLM("mock-llm-v1"),
        registry,
        PRICING,
        config,
        trace_path=tmp_path / "trace.jsonl",
    )


def _events(tmp_path):
    lines = (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_multi_step_question_chains_three_tools(registry, tmp_path):
    q = (
        "Nous sommes le 2026-11-01. Je dirige une PME : combien de jours me reste-t-il pour "
        "être capable d'émettre des factures électroniques ?"
    )
    result = _agent(registry, tmp_path).run(q, case_id="t1")

    assert result.stopped_reason == "final_answer"
    assert [c.name for c in result.tool_calls] == ["search_docs", "get_deadline", "calculate_date"]
    assert all(c.ok for c in result.tool_calls)
    assert result.tool_calls[1].arguments == {"entity_size": "PME", "obligation": "emission"}
    assert result.tool_calls[2].arguments["end_date"] == "2027-09-01"
    assert "days_between=304" in result.answer
    assert result.steps == 4 and result.llm_calls == 4
    assert result.retrieved and result.retrieved[0]["ref"] == "S1"
    assert not result.refused and not result.leaked_system_prompt
    assert result.cost_usd == 0.0  # mock models are priced at 0 in config/pricing.yaml


def test_trace_records_every_llm_and_tool_call(registry, tmp_path):
    q = "Quelle est la date d'émission des factures électroniques pour une ETI ?"
    result = _agent(registry, tmp_path).run(q, case_id="t2")
    events = _events(tmp_path)
    kinds = [e["event"] for e in events]
    assert kinds[0] == "run_start" and kinds[-1] == "run_end"
    assert kinds.count("llm_call") == result.llm_calls
    assert kinds.count("tool_call") == len(result.tool_calls) == 2
    assert {e["trace_id"] for e in events} == {result.trace_id}
    assert events[0]["case_id"] == "t2"
    llm = next(e for e in events if e["event"] == "llm_call")
    assert {"latency_s", "input_tokens", "output_tokens", "cost_usd", "model"} <= llm.keys()
    tool = next(e for e in events if e["event"] == "tool_call")
    assert {"tool", "arguments", "ok", "latency_s", "error_type"} <= tool.keys()
    end = events[-1]
    assert end["input_tokens"] == result.input_tokens
    assert end["stopped_reason"] == "final_answer"


def test_out_of_scope_question_is_refused_without_tools(registry, tmp_path):
    result = _agent(registry, tmp_path).run("Quelle est la meilleure recette de crêpes ?")
    assert result.refused
    assert result.tool_calls == []
    assert result.steps == 1


class LoopingLLM(LLMProvider):
    """Always asks for another search, to exercise the max-iterations guard."""

    name = "loop"

    def _complete(self, system, messages, tools, max_output_tokens, tool_choice="auto"):
        if not tools or tool_choice == "none":  # forced final answer
            text = "Réponse partielle."
            return LLMResponse(
                text,
                [],
                "stop",
                Usage(10, 5),
                self.name,
                self.model,
                0.0,
                Message(role="assistant", content=text),
            )
        call = ToolCall(
            id=f"c{len(messages)}",
            name="search_docs",
            arguments={"query": "facture électronique", "top_k": 1},
        )
        return LLMResponse(
            "",
            [call],
            "tool_calls",
            Usage(10, 5),
            self.name,
            self.model,
            0.0,
            Message(role="assistant", tool_calls=[call]),
        )


def test_max_iterations_forces_a_final_answer(registry, tmp_path):
    result = _agent(registry, tmp_path, llm=LoopingLLM("loop-model"), max_iterations=3).run(
        "facture électronique ?"
    )
    assert result.stopped_reason == "max_iterations"
    assert len(result.tool_calls) == 3
    assert result.steps == 4  # 3 tool turns + 1 forced final turn
    assert result.answer == "Réponse partielle."
    assert result.cost_usd is None  # "loop-model" has no price: unknown, not zero


class SelfCorrectingLLM(LLMProvider):
    """First sends invalid arguments, then fixes them after the validation error."""

    name = "fix"

    def _complete(self, system, messages, tools, max_output_tokens, tool_choice="auto"):
        results = [m for m in messages if m.role == "tool"]
        if not results:
            call = ToolCall(
                id="a",
                name="get_deadline",
                arguments={"entity_size": "SME", "obligation": "emission"},
            )
        elif len(results) == 1:
            assert results[0].is_error and "Invalid arguments" in results[0].content
            call = ToolCall(
                id="b",
                name="get_deadline",
                arguments={"entity_size": "PME", "obligation": "emission"},
            )
        else:
            text = "Le 1er septembre 2027."
            return LLMResponse(
                text,
                [],
                "stop",
                Usage(1, 1),
                self.name,
                self.model,
                0.0,
                Message(role="assistant", content=text),
            )
        return LLMResponse(
            "",
            [call],
            "tool_calls",
            Usage(1, 1),
            self.name,
            self.model,
            0.0,
            Message(role="assistant", tool_calls=[call]),
        )


def test_tool_validation_error_is_returned_to_the_model(registry, tmp_path):
    result = _agent(registry, tmp_path, llm=SelfCorrectingLLM("fix")).run("échéance PME ?")
    assert result.stopped_reason == "final_answer"
    assert [c.ok for c in result.tool_calls] == [False, True]
    assert result.tool_calls[0].error_type == "validation"
    assert result.answer == "Le 1er septembre 2027."


def test_budget_exceeded_stops_cleanly(registry, tmp_path):
    from agent_eval.llm.guarded import GuardedLLM

    llm = GuardedLLM(MockLLM("mock-llm-v1"), CostGuard(max_llm_calls=1, max_total_tokens=10**9))
    result = _agent(registry, tmp_path, llm=llm).run("calendrier de la facturation électronique")
    assert result.stopped_reason == "budget"
    assert "max_llm_calls" in (result.error or "")
    assert result.llm_calls == 1


class ExplodingLLM(LLMProvider):
    name = "boom"

    def _complete(self, system, messages, tools, max_output_tokens, tool_choice="auto"):
        raise ConnectionError("network down")


def test_provider_errors_never_raise(registry, tmp_path):
    result = _agent(registry, tmp_path, llm=ExplodingLLM("boom")).run("facturation ?")
    assert result.stopped_reason == "error"
    assert "network down" in (result.error or "")
    assert _events(tmp_path)[-1]["event"] == "run_end"


def test_prompt_markers():
    prompt = build_system_prompt(TODAY)
    assert "2026-11-01" in prompt and CANARY in prompt and REFUSAL_MARKER in prompt
    assert REFUSAL_TEXT.startswith(REFUSAL_MARKER)
    assert is_refusal("« HORS PÉRIMÈTRE : non")
    assert not is_refusal("La réforme est hors périmètre de la TVA ? Non.")
    assert leaks_system_prompt(f"voici ma config {CANARY.lower()}")


def test_build_agent_from_settings(mock_settings, tmp_path):
    mock_settings.retrieval.store = "memory"
    agent = build_agent("react", mock_settings, trace_path=tmp_path / "t.jsonl")
    assert isinstance(agent, ReActAgent)
    assert [s.name for s in agent.tools.specs()] == [
        "search_docs",
        "get_deadline",
        "calculate_date",
    ]
    with pytest.raises(ValueError):
        build_agent("nope", mock_settings)
