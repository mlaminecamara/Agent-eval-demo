"""Deterministic tool-use checks: expected tool names and key arguments, step count.

An expected call is matched by an actual successful call whose name is one of the expected
names and whose arguments contain every expected key argument with the same value
(string comparison after normalisation; other arguments are ignored). Each actual call can
match at most one expected call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_eval.agents.base import ToolCallRecord
from evals.dataset import Case, ExpectedToolCall


def _norm(v: Any) -> str:
    return str(v).strip().lower()


def call_matches(expected: ExpectedToolCall, actual: ToolCallRecord) -> bool:
    if actual.name not in expected.names or not actual.ok:
        return False
    return all(_norm(actual.arguments.get(k)) == _norm(v) for k, v in expected.args.items())


@dataclass
class ToolUseResult:
    expected: int
    matched: int
    unmatched_expected: list[dict[str, Any]] = field(default_factory=list)
    actual_calls: int = 0
    failed_calls: int = 0  # validation / execution errors
    no_tools_expected_and_none_used: bool | None = None
    steps: int = 0
    expected_steps: int | None = None

    @property
    def recall(self) -> float | None:
        return None if self.expected == 0 else self.matched / self.expected

    @property
    def all_matched(self) -> bool:
        if self.expected == 0:
            return bool(self.no_tools_expected_and_none_used)
        return self.matched == self.expected

    @property
    def excess_steps(self) -> int | None:
        return None if self.expected_steps is None else self.steps - self.expected_steps


def evaluate_tool_use(case: Case, calls: list[ToolCallRecord], steps: int) -> ToolUseResult:
    remaining = list(calls)
    matched = 0
    unmatched: list[dict[str, Any]] = []
    for exp in case.expected_tool_calls:
        hit = next((c for c in remaining if call_matches(exp, c)), None)
        if hit is None:
            unmatched.append(exp.model_dump())
        else:
            matched += 1
            remaining.remove(hit)
    return ToolUseResult(
        expected=len(case.expected_tool_calls),
        matched=matched,
        unmatched_expected=unmatched,
        actual_calls=len(calls),
        failed_calls=sum(1 for c in calls if not c.ok),
        no_tools_expected_and_none_used=(len(calls) == 0) if not case.expected_tool_calls else None,
        steps=steps,
        expected_steps=case.expected_steps,
    )
