"""Common agent interface. The eval harness only depends on Agent.run() and AgentResult, so
the ReAct, Plan-and-Execute and LangGraph agents are measured identically."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

from agent_eval.agents.prompts import is_refusal, leaks_system_prompt
from agent_eval.llm.base import (
    LLMProvider,
    LLMResponse,
    Message,
    ToolCall,
    ToolChoice,
    ToolSpec,
)
from agent_eval.observability.cost import PricingTable
from agent_eval.observability.tracing import Trace, TraceSink
from agent_eval.tools.registry import ToolRegistry

StoppedReason = Literal[
    "final_answer",  # the model answered
    "max_iterations",  # step limit reached; a final answer was forced
    "budget",  # cost guard stopped the run
    "provider_refusal",  # the provider's safety system declined
    "error",  # unexpected exception (API error, etc.)
]


@dataclass
class ToolCallRecord:
    step: int
    name: str
    arguments: dict[str, Any]
    ok: bool
    error_type: str | None
    latency_s: float
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    agent: str
    question: str
    answer: str
    stopped_reason: StoppedReason
    steps: int  # LLM turns
    tool_calls: list[ToolCallRecord]
    llm_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    latency_s: float
    trace_id: str
    provider: str
    model: str
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)  # agent-specific (e.g. the plan)

    @property
    def refused(self) -> bool:
        return is_refusal(self.answer)

    @property
    def leaked_system_prompt(self) -> bool:
        return leaks_system_prompt(self.answer)

    @property
    def retrieved(self) -> list[dict[str, Any]]:
        """Passages returned by every successful search_docs call, in order (for retrieval
        metrics). Each has doc_id, page_start, page_end, section, chunk_id, score, ref."""
        out: list[dict[str, Any]] = []
        for call in self.tool_calls:
            if call.name == "search_docs" and call.ok:
                out.extend(call.data.get("passages", []))
        return out

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d |= {
            "refused": self.refused,
            "leaked_system_prompt": self.leaked_system_prompt,
            "total_tokens": self.input_tokens + self.output_tokens,
        }
        return d


@dataclass
class AgentConfig:
    max_iterations: int = 6
    max_tool_result_chars: int = 6000
    max_output_tokens: int | None = None
    today: date | None = None  # fixed in evals so expected answers do not drift


class Agent(ABC):
    name: str

    def __init__(
        self,
        llm: LLMProvider,
        tools: ToolRegistry,
        pricing: PricingTable,
        config: AgentConfig | None = None,
        trace_path: Path | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.pricing = pricing
        self.config = config or AgentConfig()
        self.sink = TraceSink(trace_path)

    def new_trace(self, question: str, **metadata: Any) -> Trace:
        return Trace(self.sink, self.pricing, self.name, question, metadata=metadata)

    # --- shared building blocks, so every agent is traced and measured the same way ---

    def _truncate(self, content: str) -> str:
        limit = self.config.max_tool_result_chars
        if len(content) <= limit:
            return content
        return content[:limit] + "\n[... résultat tronqué ...]"

    def _call_llm(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec] | None,
        step: int,
        trace: Trace,
        tool_choice: ToolChoice = "auto",
    ) -> LLMResponse:
        response = self.llm.complete(
            system,
            messages,
            tools=tools,
            max_output_tokens=self.config.max_output_tokens,
            tool_choice=tool_choice,
        )
        trace.llm_call(step, response, price_model=self.llm.model)
        return response

    def _execute_tool_calls(
        self, calls: list[ToolCall], step: int, trace: Trace, records: list[ToolCallRecord]
    ) -> list[Message]:
        """Run every tool call of one model turn; return the tool-result messages."""
        out: list[Message] = []
        for call in calls:
            result = self.tools.execute(call)
            trace.tool_call(step, call.arguments, result)
            records.append(
                ToolCallRecord(
                    step=step,
                    name=call.name,
                    arguments=call.arguments,
                    ok=result.ok,
                    error_type=result.error_type,
                    latency_s=result.latency_s,
                    data=result.data,
                )
            )
            out.append(
                Message.tool_result(call.id, self._truncate(result.content), is_error=not result.ok)
            )
        return out

    def _result(
        self,
        question: str,
        trace: Trace,
        answer: str,
        stopped: StoppedReason,
        steps: int,
        records: list[ToolCallRecord],
        latency: float,
        error: str | None,
        **extra: Any,
    ) -> AgentResult:
        trace.end(stopped, steps, latency, error=error, **extra)
        return AgentResult(
            agent=self.name,
            question=question,
            answer=answer,
            stopped_reason=stopped,
            steps=steps,
            tool_calls=records,
            llm_calls=trace.llm_calls,
            input_tokens=trace.input_tokens,
            output_tokens=trace.output_tokens,
            cost_usd=trace.cost_usd,
            latency_s=latency,
            trace_id=trace.trace_id,
            provider=self.llm.name,
            model=self.llm.model,
            error=error,
            extra=extra,
        )

    @abstractmethod
    def run(self, question: str, **metadata: Any) -> AgentResult:
        """Answer one question. Must never raise: failures end up in stopped_reason/error."""
