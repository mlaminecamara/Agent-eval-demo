import pytest

from agent_eval.cost_guard import BudgetExceeded, CostGuard
from agent_eval.llm import Message, ToolCall, ToolSpec, build_llm
from agent_eval.llm.anthropic_provider import AnthropicProvider
from agent_eval.llm.mock import REFUSAL_TEXT
from agent_eval.llm.openai_provider import OpenAIProvider

SEARCH = ToolSpec(
    name="search_docs",
    description="Search the corpus",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    },
)


def test_mock_calls_search_tool_for_in_domain_question(mock_settings):
    llm = build_llm(mock_settings.llm)
    q = "Quand les PME doivent-elles émettre des factures électroniques ?"
    r = llm.complete("sys", [Message.user(q)], tools=[SEARCH])
    assert r.finish_reason == "tool_calls"
    assert r.tool_calls[0].name == "search_docs"
    assert r.tool_calls[0].arguments == {"query": q}
    assert r.usage.input_tokens > 0


def test_mock_is_deterministic(mock_settings):
    llm = build_llm(mock_settings.llm)
    msgs = [Message.user("Qu'est-ce qu'une PDP pour la facturation ?")]
    a = llm.complete("sys", msgs, tools=[SEARCH])
    b = llm.complete("sys", msgs, tools=[SEARCH])
    assert a.tool_calls[0].id == b.tool_calls[0].id
    assert a.usage == b.usage


def test_mock_refuses_out_of_domain(mock_settings):
    llm = build_llm(mock_settings.llm)
    r = llm.complete(
        "sys", [Message.user("Quelle est la recette de la tarte tatin ?")], tools=[SEARCH]
    )
    assert r.finish_reason == "stop"
    assert r.text == REFUSAL_TEXT


def test_mock_answers_after_tool_result(mock_settings):
    llm = build_llm(mock_settings.llm)
    first = llm.complete(
        "sys", [Message.user("calendrier facturation électronique")], tools=[SEARCH]
    )
    msgs = [
        Message.user("calendrier facturation électronique"),
        first.message,
        Message.tool_result(
            first.tool_calls[0].id, "1er septembre 2026 pour les grandes entreprises"
        ),
    ]
    r = llm.complete("sys", msgs, tools=[SEARCH])
    assert r.finish_reason == "stop"
    assert "1er septembre 2026" in r.text


def test_guarded_llm_enforces_call_limit(mock_settings):
    guard = CostGuard(max_llm_calls=2, max_total_tokens=10**9)
    llm = build_llm(mock_settings.llm, guard=guard)
    for _ in range(2):
        llm.complete("sys", [Message.user("tva")])
    with pytest.raises(BudgetExceeded):
        llm.complete("sys", [Message.user("tva")])
    assert guard.calls == 2
    assert guard.total_tokens > 0


def _conversation() -> list[Message]:
    call_a = ToolCall(id="a", name="search_docs", arguments={"query": "x"})
    call_b = ToolCall(id="b", name="get_deadline", arguments={"entity_size": "PME"})
    return [
        Message.user("question"),
        Message(role="assistant", content="", tool_calls=[call_a, call_b]),
        Message.tool_result("a", "result a"),
        Message.tool_result("b", "bad args", is_error=True),
    ]


def test_anthropic_groups_tool_results_in_one_user_message():
    out = AnthropicProvider._to_anthropic_messages(_conversation())
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    results = out[2]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b"]
    assert results[1]["is_error"] is True
    assert [b["type"] for b in out[1]["content"]] == ["tool_use", "tool_use"]


def test_anthropic_replays_raw_assistant_blocks_unchanged():
    raw = [
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {"type": "tool_use", "id": "a", "name": "search_docs", "input": {"query": "x"}},
    ]
    msgs = [
        Message.user("q"),
        Message(
            role="assistant",
            provider="anthropic",
            provider_raw=raw,
            tool_calls=[ToolCall(id="a", name="search_docs", arguments={"query": "x"})],
        ),
        Message.tool_result("a", "r"),
    ]
    out = AnthropicProvider._to_anthropic_messages(msgs)
    assert out[1]["content"] is raw


def test_openai_message_conversion():
    out = OpenAIProvider._to_openai_messages("sys", _conversation())
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool", "tool"]
    assert out[2]["tool_calls"][1]["function"]["name"] == "get_deadline"
    assert out[3]["tool_call_id"] == "a"
