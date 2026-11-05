"""POST /ask and GET /health in mock mode (in-memory index, fake LLM, no network)."""

import pytest
from fastapi.testclient import TestClient

from agent_eval import api
from agent_eval.config import load_settings
from tests.test_agents import ExplodingLLM

MULTI_STEP_Q = (
    "Nous sommes le 2026-11-01. Je dirige une PME : combien de jours me reste-t-il pour "
    "être capable d'émettre des factures électroniques ?"
)


@pytest.fixture(scope="module")
def client():
    settings = load_settings(
        use_env=False,
        overrides={
            "llm.provider": "mock",
            "embeddings.provider": "mock",
            "retrieval.store": "memory",
        },
    )
    with TestClient(api.create_app(settings)) as c:  # the context runs the lifespan
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["llm"].startswith("mock:")
    assert body["agents"] == ["react", "plan_execute", "langgraph"]


@pytest.mark.parametrize("agent", ["react", "plan_execute", "langgraph"])
def test_ask_returns_answer_sources_and_trace(client, agent):
    r = client.post("/ask", json={"question": MULTI_STEP_Q, "agent": agent})
    assert r.status_code == 200
    body = r.json()
    assert body["agent"] == agent and body["refused"] is False
    assert "days_between=304" in body["answer"]
    assert [t["name"] for t in body["tool_calls"]] == [
        "search_docs",
        "get_deadline",
        "calculate_date",
    ]
    vias = {s["via"] for s in body["sources"]}
    assert vias == {"search_docs", "get_deadline"}
    assert any(s["ref"] == "S1" for s in body["sources"])
    assert body["trace_id"] and body["cost_usd"] == 0.0 and body["stopped_reason"] == "final_answer"


def test_out_of_scope_is_refused_without_sources(client):
    body = client.post("/ask", json={"question": "Une recette de crêpes ?"}).json()
    assert body["refused"] is True and body["sources"] == [] and body["tool_calls"] == []


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "x" * (api.MAX_QUESTION_CHARS + 1)},
        {"question": "facture ?", "agent": "unknown"},
        {},
    ],
)
def test_invalid_requests_are_rejected(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_api_key_is_enforced_when_set(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key-123")
    q = {"question": "facture électronique ?"}
    assert client.post("/ask", json=q).status_code == 401
    assert client.post("/ask", json=q, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/ask", json=q, headers={"X-API-Key": "test-key-123"}).status_code == 200


def test_provider_error_returns_502_with_trace_id(client, monkeypatch):
    original = api.build_agent

    def broken(*args, **kwargs):
        agent = original(*args, **kwargs)
        agent.llm = ExplodingLLM("boom")
        return agent

    monkeypatch.setattr(api, "build_agent", broken)
    r = client.post("/ask", json={"question": "facture électronique ?"})
    assert r.status_code == 502 and "trace_id" in r.json()["detail"]
    assert "network down" not in r.text  # internal error details are not exposed
