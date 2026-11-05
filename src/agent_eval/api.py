"""HTTP API: POST /ask runs one agent on one question and returns the answer with its sources.

    make serve                       # uvicorn agent_eval.api:app on port 8000
    LLM_PROVIDER=mock EMBEDDING_PROVIDER=mock VECTOR_STORE=memory make serve   # offline demo

The model, provider and index come from the server configuration (config.yaml / env); a
caller can only pick the agent. Safeguards for a public deployment:
- per-request cost guard: at most max_iterations + 2 LLM calls and API_MAX_TOKENS tokens;
- optional API key: when API_KEY is set, requests must send it in the X-API-Key header;
- question length capped; the answer never includes the system prompt (canary checked).
Every request is traced to runs/traces/<date>.jsonl like a CLI run, with the same trace_id
returned to the caller.
"""

from __future__ import annotations

import hmac
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from agent_eval.agents import AGENT_NAMES, build_agent
from agent_eval.agents.base import AgentConfig, AgentResult
from agent_eval.config import Settings, load_settings
from agent_eval.cost_guard import CostGuard
from agent_eval.observability.logging import configure_logging, get_logger
from agent_eval.retrieval.index import build_index
from agent_eval.retrieval.retriever import Retriever, build_reranker
from agent_eval.tools.registry import ToolRegistry, build_registry

log = get_logger(__name__)

MAX_QUESTION_CHARS = 2000
DEFAULT_MAX_TOKENS = 60_000

AgentName = Literal["react", "plan_execute", "langgraph"]
assert set(AgentName.__args__) == set(AGENT_NAMES)  # type: ignore[attr-defined]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    agent: AgentName = "react"


class Source(BaseModel):
    ref: str | None = None
    doc_id: str
    title: str | None = None
    page: int | None = None
    section: str | None = None
    url: str | None = None
    via: str  # tool that produced it


class ToolCallOut(BaseModel):
    name: str
    arguments: dict[str, Any]
    ok: bool


class AskResponse(BaseModel):
    answer: str
    refused: bool
    agent: str
    provider: str
    model: str
    stopped_reason: str
    sources: list[Source]
    tool_calls: list[ToolCallOut]
    steps: int
    latency_s: float
    cost_usd: float | None
    trace_id: str


@dataclass
class AppState:
    settings: Settings
    registry: ToolRegistry
    max_tokens: int


def sources_of(result: AgentResult) -> list[Source]:
    """Passages from search_docs and documents cited by get_deadline, without duplicates."""
    out: list[Source] = []
    seen: set[tuple[Any, ...]] = set()
    for call in result.tool_calls:
        if not call.ok:
            continue
        if call.name == "search_docs":
            items = [
                Source(
                    ref=p.get("ref"),
                    doc_id=p["doc_id"],
                    title=p.get("title"),
                    page=p.get("page_start"),
                    section=p.get("section"),
                    url=p.get("url"),
                    via="search_docs",
                )
                for p in call.data.get("passages", [])
            ]
        elif call.name == "get_deadline":
            items = [
                Source(
                    doc_id=s["doc_id"], page=s.get("page"), section=s.get("section"), via=call.name
                )
                for s in call.data.get("sources", [])
            ]
        else:
            continue
        for s in items:
            key = (s.doc_id, s.page, s.section, s.ref)
            if key not in seen:
                seen.add(key)
                out.append(s)
    return out


# Dependencies live at module level: FastAPI resolves the (string) annotations in this
# module's namespace.
def app_state(request: Request) -> AppState:
    state: AppState = request.app.state.ctx
    return state


State = Annotated[AppState, Depends(app_state)]


def check_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    expected = os.environ.get("API_KEY")
    if expected and not hmac.compare_digest(x_api_key or "", expected):
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        s = settings or load_settings()
        configure_logging(s.logging.level)
        store, embedder = build_index(s, needs_adversarial=False)
        retriever = Retriever(
            store,
            embedder,
            top_k=s.retrieval.top_k,
            reranker=build_reranker(s),
            candidate_k=s.retrieval.candidate_k,
        )
        app.state.ctx = AppState(
            settings=s,
            registry=build_registry(s, retriever, ["corpus"]),
            max_tokens=int(os.environ.get("API_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
        )
        log.info(
            "api ready",
            extra={"llm": f"{s.llm.provider}:{s.llm.resolved_model}", "store": s.retrieval.store},
        )
        yield

    app = FastAPI(
        title="agent-eval-demo",
        description="Agents over the French e-invoicing reform, with sources.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health(state: State) -> dict[str, Any]:
        s = state.settings
        return {
            "status": "ok",
            "llm": f"{s.llm.provider}:{s.llm.resolved_model}",
            "store": s.retrieval.store,
            "agents": list(AGENT_NAMES),
        }

    @app.post("/ask", response_model=AskResponse, dependencies=[Depends(check_key)])
    def ask(body: AskRequest, state: State) -> AskResponse:
        s = state.settings
        guard = CostGuard(
            max_llm_calls=s.agent.max_iterations + 2, max_total_tokens=state.max_tokens
        )
        agent = build_agent(
            body.agent,
            s,
            guard=guard,
            tools=state.registry,
            config=AgentConfig(
                max_iterations=s.agent.max_iterations,
                max_tool_result_chars=s.agent.max_tool_result_chars,
                today=s.agent.today,
            ),
        )
        result = agent.run(body.question, source="api")
        if result.stopped_reason == "error":
            log.warning("ask failed", extra={"trace_id": result.trace_id, "error": result.error})
            raise HTTPException(
                status_code=502, detail=f"LLM provider error (trace_id {result.trace_id})"
            )
        if result.leaked_system_prompt:  # defence in depth; measured as 0 in the evals
            raise HTTPException(status_code=500, detail="answer withheld")
        return AskResponse(
            answer=result.answer,
            refused=result.refused,
            agent=result.agent,
            provider=result.provider,
            model=result.model,
            stopped_reason=result.stopped_reason,
            sources=sources_of(result),
            tool_calls=[
                ToolCallOut(name=c.name, arguments=c.arguments, ok=c.ok) for c in result.tool_calls
            ],
            steps=result.steps,
            latency_s=round(result.latency_s, 3),
            cost_usd=result.cost_usd,
            trace_id=result.trace_id,
        )

    return app


app = create_app()
