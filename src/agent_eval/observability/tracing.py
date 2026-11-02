"""Tracing of every LLM call and tool call, as JSON lines.

One trace = one agent run on one question. Events share a trace_id and are appended to a
JSONL file (default runs/traces/<date>.jsonl; eval runs write to their own file):

  run_start  {question, agent, provider, model}
  llm_call   {step, provider, model, latency_s, input_tokens, output_tokens, cost_usd,
              finish_reason, n_tool_calls, request_id}
  tool_call  {step, tool, arguments, ok, error_type, latency_s, result_preview}
  run_end    {stopped_reason, steps, llm_calls, tool_calls, tokens, cost_usd, latency_s}

cost_usd is null when the model has no price in config/pricing.yaml (never guessed).
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_eval.llm.base import LLMResponse
from agent_eval.observability.cost import PricingTable
from agent_eval.observability.logging import get_logger
from agent_eval.tools.base import ToolResult

log = get_logger("agent_eval.trace")

PREVIEW_CHARS = 300


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


class TraceSink:
    """Thread-safe JSONL appender."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._lock = threading.Lock()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: dict[str, Any]) -> None:
        if self.path is None:
            return
        line = json.dumps(event, ensure_ascii=False, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def default_trace_path(runs_dir: Path) -> Path:
    return runs_dir / "traces" / f"{datetime.now(tz=UTC):%Y-%m-%d}.jsonl"


@dataclass
class Trace:
    """Accumulates one agent run and writes its events to the sink as they happen."""

    sink: TraceSink
    pricing: PricingTable
    agent: str
    question: str
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    llm_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = 0.0  # becomes None as soon as one call has an unknown price
    events: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def _emit(self, kind: str, **payload: Any) -> dict[str, Any]:
        event = {"ts": _now(), "trace_id": self.trace_id, "event": kind, **payload}
        self.events.append(event)
        self.sink.write(event)
        log.debug(kind, extra={k: v for k, v in event.items() if k != "event"})
        return event

    def start(self, provider: str, model: str, **extra: Any) -> None:
        self._emit(
            "run_start",
            agent=self.agent,
            question=self.question,
            provider=provider,
            model=model,
            **extra,
            **self.metadata,
        )

    def llm_call(self, step: int, response: LLMResponse, price_model: str) -> None:
        cost = self.pricing.cost(
            price_model, response.usage.input_tokens, response.usage.output_tokens
        )
        self.llm_calls += 1
        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens
        if cost is None or self.cost_usd is None:
            self.cost_usd = None
        else:
            self.cost_usd += cost
        self._emit(
            "llm_call",
            step=step,
            provider=response.provider,
            model=response.model,
            price_model=price_model,
            latency_s=round(response.latency_s, 4),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=cost,
            finish_reason=response.finish_reason,
            n_tool_calls=len(response.tool_calls),
            request_id=response.request_id,
        )

    def tool_call(self, step: int, arguments: dict[str, Any], result: ToolResult) -> None:
        self.tool_calls += 1
        self._emit(
            "tool_call",
            step=step,
            tool=result.tool,
            arguments=arguments,
            ok=result.ok,
            error_type=result.error_type,
            latency_s=round(result.latency_s, 4),
            result_preview=result.content[:PREVIEW_CHARS],
        )

    def end(self, stopped_reason: str, steps: int, latency_s: float, **extra: Any) -> None:
        self._emit(
            "run_end",
            stopped_reason=stopped_reason,
            steps=steps,
            llm_calls=self.llm_calls,
            tool_calls=self.tool_calls,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cost_usd=self.cost_usd,
            latency_s=round(latency_s, 4),
            **extra,
        )
