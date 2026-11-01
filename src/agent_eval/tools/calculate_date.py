"""calculate_date: deterministic date arithmetic (add/subtract days or months, difference)."""

from __future__ import annotations

import calendar
from datetime import date
from typing import Annotated, Any, ClassVar, Literal

from pydantic import Field, field_validator, model_validator

from agent_eval.tools.base import StrictArgs, Tool, ToolResult, dumps

MAX_DAYS = 36_500
MAX_MONTHS = 1_200


def _parse_iso(v: str) -> date:
    try:
        return date.fromisoformat(v)
    except ValueError as exc:
        raise ValueError(f"'{v}' is not a valid date in YYYY-MM-DD format") from exc


def add_months(d: date, months: int) -> date:
    """Calendar month arithmetic; the day is clamped to the end of the target month
    (2026-01-31 + 1 month = 2026-02-28)."""
    total = d.year * 12 + (d.month - 1) + months
    year, month = divmod(total, 12)
    month += 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


class CalculateDateArgs(StrictArgs):
    operation: Literal["add", "subtract", "diff"]
    start_date: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]
    days: Annotated[int, Field(ge=0, le=MAX_DAYS)] | None = None
    months: Annotated[int, Field(ge=0, le=MAX_MONTHS)] | None = None
    end_date: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")] | None = None

    @field_validator("start_date", "end_date")
    @classmethod
    def _real_date(cls, v: str | None) -> str | None:
        if v is not None:
            _parse_iso(v)
        return v

    @model_validator(mode="after")
    def _consistent(self) -> CalculateDateArgs:
        if self.operation in ("add", "subtract"):
            if self.days is None and self.months is None:
                raise ValueError(f"operation '{self.operation}' needs days and/or months")
            if self.end_date is not None:
                raise ValueError(f"end_date must be null for operation '{self.operation}'")
        else:
            if self.end_date is None:
                raise ValueError("operation 'diff' needs end_date")
            if self.days is not None or self.months is not None:
                raise ValueError("days and months must be null for operation 'diff'")
        return self


class CalculateDateTool(Tool[CalculateDateArgs]):
    name = "calculate_date"
    description = (
        "Calcul de dates exact. operation='add' ou 'subtract' : ajoute ou retire des jours "
        "et/ou des mois à start_date (end_date = null). operation='diff' : nombre de jours de "
        "start_date à end_date (days et months = null ; négatif si end_date est avant "
        "start_date). Dates au format YYYY-MM-DD. Les mois sont calendaires (fin de mois "
        "ramenée au dernier jour du mois cible)."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["add", "subtract", "diff"]},
            "start_date": {"type": "string", "description": "Date de départ, YYYY-MM-DD."},
            "days": {
                "type": ["integer", "null"],
                "description": f"Jours à ajouter/retirer (0 à {MAX_DAYS}), sinon null.",
            },
            "months": {
                "type": ["integer", "null"],
                "description": f"Mois à ajouter/retirer (0 à {MAX_MONTHS}), sinon null.",
            },
            "end_date": {
                "type": ["string", "null"],
                "description": "Date de fin pour 'diff', YYYY-MM-DD, sinon null.",
            },
        },
        "required": ["operation", "start_date", "days", "months", "end_date"],
        "additionalProperties": False,
    }
    args_model = CalculateDateArgs

    def execute(self, args: CalculateDateArgs) -> ToolResult:
        start = _parse_iso(args.start_date)
        payload: dict[str, Any] = {"operation": args.operation, "start_date": args.start_date}
        if args.operation == "diff":
            assert args.end_date is not None
            end = _parse_iso(args.end_date)
            payload |= {"end_date": args.end_date, "days_between": (end - start).days}
        else:
            sign = 1 if args.operation == "add" else -1
            months, days = args.months or 0, args.days or 0
            result = add_months(start, sign * months)
            result = date.fromordinal(result.toordinal() + sign * days)
            payload |= {"months": months, "days": days, "result_date": result.isoformat()}
        return ToolResult(tool=self.name, ok=True, content=dumps(payload), data=payload)
