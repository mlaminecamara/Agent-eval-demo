"""Tool interface: a JSON Schema for the model, a pydantic model for strict validation.

The JSON Schema is written by hand so it stays compatible with strict tool use on both
providers (every property listed in `required`, optional values expressed as nullable types,
`additionalProperties: false`). The pydantic model enforces the same contract at run time,
plus checks a JSON Schema cannot express (cross-field rules, real calendar dates).

A tool never raises to the agent: invalid arguments and execution errors come back as a
ToolResult with ok=False and a message the model can act on.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

from agent_eval.llm.base import ToolSpec


class StrictArgs(BaseModel):
    """Base for tool arguments: unknown keys are rejected, no silent type coercion."""

    model_config = ConfigDict(extra="forbid", strict=True)


ArgsT = TypeVar("ArgsT", bound=StrictArgs)


@dataclass
class ToolResult:
    tool: str
    ok: bool
    content: str  # what the model sees
    data: dict[str, Any] = field(default_factory=dict)  # structured payload (traces, metrics)
    error_type: str | None = None  # "validation" | "execution" | "unknown_tool"
    latency_s: float = 0.0


def format_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err["loc"]) or "(arguments)"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)


class Tool(ABC, Generic[ArgsT]):
    name: ClassVar[str]
    description: ClassVar[str]
    parameters: ClassVar[dict[str, Any]]
    args_model: type[ArgsT]

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name, description=self.description, parameters=self.parameters, strict=True
        )

    def validate(self, arguments: dict[str, Any]) -> ArgsT:
        return self.args_model.model_validate(arguments)

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        start = time.perf_counter()
        try:
            args = self.validate(arguments)
        except ValidationError as exc:
            msg = format_validation_error(exc)
            return ToolResult(
                tool=self.name,
                ok=False,
                error_type="validation",
                content=f"Invalid arguments for {self.name}: {msg}. Fix the arguments and retry.",
                data={"arguments": arguments, "errors": msg},
                latency_s=time.perf_counter() - start,
            )
        try:
            result = self.execute(args)
        except ToolExecutionError as exc:
            result = ToolResult(
                tool=self.name,
                ok=False,
                error_type="execution",
                content=f"{self.name} failed: {exc}",
                data={"error": str(exc)},
            )
        except Exception as exc:
            result = ToolResult(
                tool=self.name,
                ok=False,
                error_type="execution",
                content=f"{self.name} failed with an internal error.",
                data={"error": f"{type(exc).__name__}: {exc}"},
            )
        result.latency_s = time.perf_counter() - start
        return result

    @abstractmethod
    def execute(self, args: ArgsT) -> ToolResult: ...


class ToolExecutionError(RuntimeError):
    """Expected failure (e.g. no data for this input), reported to the model as is."""


def dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=None, sort_keys=False)
