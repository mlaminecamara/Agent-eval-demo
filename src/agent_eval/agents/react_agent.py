"""Native ReAct loop (no framework): reason, call tools, observe, repeat.

With native tool calling the "thought" is whatever text the model writes next to its tool
calls; it is kept in the conversation and the trace. Guards:
- max_iterations: after N model turns that still request tools, one last call is made
  without tools and with an instruction to answer from what was gathered.
- tool errors (invalid arguments, unknown tool, execution failure) are returned to the model
  as error results so it can correct itself; they never abort the run.
- cost guard (BudgetExceeded), provider refusals and API errors end the run with a
  stopped_reason instead of raising.
"""

from __future__ import annotations

import time
from typing import Any

from agent_eval.agents.base import Agent, AgentResult, StoppedReason, ToolCallRecord
from agent_eval.agents.prompts import FORCE_FINAL_MESSAGE, build_system_prompt
from agent_eval.cost_guard import BudgetExceeded
from agent_eval.llm.base import Message
from agent_eval.observability.logging import get_logger

log = get_logger(__name__)


class ReActAgent(Agent):
    name = "react"

    def run(self, question: str, **metadata: Any) -> AgentResult:
        start = time.perf_counter()
        trace = self.new_trace(question, **metadata)
        trace.start(self.llm.name, self.llm.model, max_iterations=self.config.max_iterations)

        system = build_system_prompt(self.config.today)
        specs = self.tools.specs()
        messages: list[Message] = [Message.user(question)]
        records: list[ToolCallRecord] = []
        answer = ""
        stopped: StoppedReason = "final_answer"
        error: str | None = None
        step = 0

        try:
            while True:
                if step >= self.config.max_iterations:
                    stopped = "max_iterations"
                    messages.append(Message.user(FORCE_FINAL_MESSAGE))
                    step += 1
                    response = self._call_llm(system, messages, specs, step, trace, "none")
                    answer = response.text
                    break

                step += 1
                response = self._call_llm(system, messages, specs, step, trace)
                messages.append(response.message)

                if response.finish_reason == "refusal":
                    stopped = "provider_refusal"
                    answer = response.text
                    break
                if not response.tool_calls:
                    answer = response.text
                    break

                messages.extend(self._execute_tool_calls(response.tool_calls, step, trace, records))
        except BudgetExceeded as exc:
            stopped, error = "budget", str(exc)
        except Exception as exc:
            stopped, error = "error", f"{type(exc).__name__}: {exc}"
            log.warning("agent run failed", extra={"trace_id": trace.trace_id, "error": error})

        latency = time.perf_counter() - start
        return self._result(question, trace, answer, stopped, step, records, latency, error)
