"""Plan-and-Execute and LangGraph agents in mock mode, plus the date grounding check.

The registry fixture (conftest.py) and the fake LLMs (test_agents.py) are shared, so the
three agents are exercised on the same tools and the same failure modes.
"""

from datetime import date

import pytest

from agent_eval.agents import build_agent
from agent_eval.agents.base import AgentConfig
from agent_eval.agents.grounding import extract_dates, unsupported_dates
from agent_eval.agents.langgraph_agent import LangGraphAgent
from agent_eval.agents.plan_execute_agent import PlanExecuteAgent, parse_plan
from agent_eval.agents.react_agent import ReActAgent
from agent_eval.cost_guard import CostGuard
from agent_eval.llm.base import LLMProvider, LLMResponse, Message, ToolCall, Usage
from agent_eval.llm.guarded import GuardedLLM
from agent_eval.llm.mock import MockLLM
from tests.test_agents import PRICING, TODAY, ExplodingLLM, LoopingLLM, _events

MULTI_STEP_Q = (
    "Nous sommes le 2026-11-01. Je dirige une PME : combien de jours me reste-t-il pour "
    "être capable d'émettre des factures électroniques ?"
)
AGENTS = {"react": ReActAgent, "plan_execute": PlanExecuteAgent, "langgraph": LangGraphAgent}


def _make(kind, registry, tmp_path, llm=None, **cfg):
    return AGENTS[kind](
        llm or MockLLM("mock-llm-v1"),
        registry,
        PRICING,
        AgentConfig(today=TODAY, **cfg),
        trace_path=tmp_path / "trace.jsonl",
    )


def _text(llm, text):
    return LLMResponse(
        text,
        [],
        "stop",
        Usage(1, 1),
        llm.name,
        llm.model,
        0.0,
        Message(role="assistant", content=text),
    )


def _call(llm, call):
    return LLMResponse(
        "",
        [call],
        "tool_calls",
        Usage(1, 1),
        llm.name,
        llm.model,
        0.0,
        Message(role="assistant", tool_calls=[call]),
    )


# --- shared behaviour -------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["react", "plan_execute", "langgraph"])
def test_every_agent_chains_the_three_tools(kind, registry, tmp_path):
    result = _make(kind, registry, tmp_path).run(MULTI_STEP_Q, case_id="t")
    assert result.agent == kind
    assert result.stopped_reason == "final_answer"
    assert [c.name for c in result.tool_calls] == ["search_docs", "get_deadline", "calculate_date"]
    assert result.tool_calls[2].arguments["end_date"] == "2027-09-01"
    assert "days_between=304" in result.answer
    events = _events(tmp_path)
    assert [e["event"] for e in events].count("llm_call") == result.llm_calls
    assert events[-1]["event"] == "run_end" and events[-1]["steps"] == result.steps


@pytest.mark.parametrize("kind", ["react", "plan_execute", "langgraph"])
def test_every_agent_refuses_out_of_scope_without_tools(kind, registry, tmp_path):
    result = _make(kind, registry, tmp_path).run("Quelle est la meilleure recette de crêpes ?")
    assert result.refused and result.tool_calls == [] and result.steps == 1


@pytest.mark.parametrize("kind", ["react", "plan_execute", "langgraph"])
def test_every_agent_survives_provider_errors(kind, registry, tmp_path):
    result = _make(kind, registry, tmp_path, llm=ExplodingLLM("boom")).run("facturation ?")
    assert result.stopped_reason == "error" and "network down" in (result.error or "")
    assert _events(tmp_path)[-1]["event"] == "run_end"


@pytest.mark.parametrize("kind", ["react", "plan_execute", "langgraph"])
def test_every_agent_stops_on_budget(kind, registry, tmp_path):
    llm = GuardedLLM(MockLLM("mock-llm-v1"), CostGuard(max_llm_calls=2, max_total_tokens=10**9))
    result = _make(kind, registry, tmp_path, llm=llm).run(MULTI_STEP_Q)
    assert result.stopped_reason == "budget" and result.llm_calls == 2
    assert result.answer == ""


def test_build_agent_knows_all_agents(mock_settings, tmp_path):
    mock_settings.retrieval.store = "memory"
    for name, cls in AGENTS.items():
        assert isinstance(build_agent(name, mock_settings, trace_path=tmp_path / "t"), cls)


# --- Plan-and-Execute -------------------------------------------------------------------------


def test_plan_execute_plans_then_executes_then_synthesizes(registry, tmp_path):
    result = _make("plan_execute", registry, tmp_path).run(MULTI_STEP_Q)
    assert [s["tool"] for s in result.extra["plan"]] == [
        "search_docs",
        "get_deadline",
        "calculate_date",
    ]
    assert result.extra["plan_fallback"] is False
    assert result.steps == 5  # plan + 3 steps + synthesis
    assert [c.step for c in result.tool_calls] == [2, 3, 4]


def test_plan_is_capped_and_filtered():
    steps = [{"tool": "search_docs", "goal": "a"}, {"tool": "web", "goal": "b"}] + [
        {"tool": "get_deadline", "goal": str(i)} for i in range(10)
    ]
    plan = parse_plan({"steps": steps}, max_steps=4)
    assert len(plan) == 4 and all(s["tool"] != "web" for s in plan)
    assert parse_plan({"steps": "oops"}, 4) == []


class NoPlanLLM(LLMProvider):
    """Answers the planner in plain text (no submit_plan, no refusal)."""

    name = "noplan"

    def _complete(self, system, messages, tools, max_output_tokens, tool_choice="auto"):
        names = {t.name for t in tools}
        if "submit_plan" in names:
            return _text(self, "Je vais chercher.")
        if tool_choice == "none":
            return _text(self, "Réponse finale.")
        if not any(m.role == "tool" for m in messages):
            return _call(self, ToolCall("s1", "search_docs", {"query": "calendrier", "top_k": 2}))
        return _text(self, "rien à ajouter")


def test_plan_execute_falls_back_to_a_search_when_no_plan(registry, tmp_path):
    result = _make("plan_execute", registry, tmp_path, llm=NoPlanLLM("noplan")).run(
        "calendrier de la facturation électronique ?"
    )
    assert result.extra["plan_fallback"] is True
    assert [c.name for c in result.tool_calls] == ["search_docs"]
    assert result.answer == "Réponse finale." and result.steps == 3


# --- LangGraph --------------------------------------------------------------------------------


def test_langgraph_max_iterations_forces_a_final_answer(registry, tmp_path):
    result = _make("langgraph", registry, tmp_path, llm=LoopingLLM("loop"), max_iterations=3).run(
        "facture électronique ?"
    )
    assert result.stopped_reason == "max_iterations"
    assert len(result.tool_calls) == 3 and result.steps == 4
    assert result.answer == "Réponse partielle."


class WrongDateLLM(LLMProvider):
    """Calls get_deadline, answers with an invented date, then corrects it when told."""

    name = "wrongdate"

    def _complete(self, system, messages, tools, max_output_tokens, tool_choice="auto"):
        if not any(m.role == "tool" for m in messages):
            args = {"entity_size": "PME", "obligation": "emission"}
            return _call(self, ToolCall("d1", "get_deadline", args))
        if "Vérification automatique" in messages[-1].content:
            return _text(self, "Les PME émettront à partir du 1er septembre 2027.")
        return _text(self, "Les PME émettront à partir du 15 mars 2031.")


def test_langgraph_verify_node_sends_unsupported_dates_back(registry, tmp_path):
    result = _make("langgraph", registry, tmp_path, llm=WrongDateLLM("wrongdate")).run(
        "Quand une PME doit-elle émettre des factures électroniques ?"
    )
    assert result.answer == "Les PME émettront à partir du 1er septembre 2027."
    assert result.extra["verify_retries"] == 1
    assert result.extra["unsupported_dates"] == []
    assert result.steps == 3  # tool turn, wrong answer, corrected answer


def test_react_has_no_verify_step(registry, tmp_path):
    result = _make("react", registry, tmp_path, llm=WrongDateLLM("wrongdate")).run(
        "Quand une PME doit-elle émettre des factures électroniques ?"
    )
    assert "2031" in result.answer


# --- grounding --------------------------------------------------------------------------------


def test_extract_dates_reads_iso_french_and_numeric_forms():
    text = "Le 1er septembre 2027, le 1 er septembre 2026, le 15 mars 2029, 01/10/2028, 2026-11-01"
    assert extract_dates(text) == {
        date(2027, 9, 1),
        date(2026, 9, 1),
        date(2029, 3, 15),
        date(2028, 10, 1),
        date(2026, 11, 1),
    }
    assert extract_dates("31 février 2027 et 2027-13-01") == set()


def test_unsupported_dates_compares_across_forms():
    evidence = ['{"date": "2027-09-01"}', "à compter du 1er septembre 2026"]
    answer = "Réception le 01/09/2026, émission le 1er septembre 2027, fin le 15 mars 2031."
    assert unsupported_dates(answer, evidence) == [date(2031, 3, 15)]
    assert unsupported_dates("aujourd'hui 2026-11-01", [], today=TODAY) == []
