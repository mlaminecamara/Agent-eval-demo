"""get_deadline: look up an obligation date by entity size in data/tools/deadlines.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Literal

from agent_eval.tools.base import StrictArgs, Tool, ToolExecutionError, ToolResult, dumps

EntitySize = Literal["GE", "ETI", "PME", "MICRO"]
Obligation = Literal["reception", "emission", "e_reporting"]


class GetDeadlineArgs(StrictArgs):
    entity_size: EntitySize
    obligation: Obligation


class DeadlineTable:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data
        self._index = {(d["entity_size"], d["obligation"]): d for d in data["deadlines"]}

    @classmethod
    def load(cls, path: Path | str) -> DeadlineTable:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def get(self, entity_size: str, obligation: str) -> dict[str, Any] | None:
        return self._index.get((entity_size, obligation))


class GetDeadlineTool(Tool[GetDeadlineArgs]):
    name = "get_deadline"
    description = (
        "Renvoie la date d'entrée en vigueur d'une obligation de la réforme pour une taille "
        "d'entreprise, avec ses sources. entity_size : GE (grande entreprise), ETI (entreprise "
        "de taille intermédiaire), PME (petite et moyenne entreprise), MICRO (microentreprise, "
        "y compris auto-entrepreneur). obligation : reception (recevoir des factures "
        "électroniques), emission (émettre des factures électroniques), e_reporting "
        "(transmettre les données de transaction)."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "entity_size": {
                "type": "string",
                "enum": ["GE", "ETI", "PME", "MICRO"],
                "description": "Taille de l'entreprise.",
            },
            "obligation": {
                "type": "string",
                "enum": ["reception", "emission", "e_reporting"],
                "description": "Obligation concernée.",
            },
        },
        "required": ["entity_size", "obligation"],
        "additionalProperties": False,
    }
    args_model = GetDeadlineArgs

    def __init__(self, table: DeadlineTable) -> None:
        self.table = table

    def execute(self, args: GetDeadlineArgs) -> ToolResult:
        entry = self.table.get(args.entity_size, args.obligation)
        if entry is None:
            raise ToolExecutionError(
                f"no deadline recorded for entity_size={args.entity_size}, "
                f"obligation={args.obligation}"
            )
        size = self.table.data["entity_sizes"][args.entity_size]
        payload = {
            "entity_size": args.entity_size,
            "entity_label": size["label"],
            "entity_definition": size["definition"],
            "obligation": args.obligation,
            "obligation_label": self.table.data["obligations"][args.obligation],
            "date": entry["date"],
            "sources": [
                {k: s[k] for k in ("doc_id", "page", "section") if k in s} for s in entry["sources"]
            ],
            "verified": entry["verified"],
        }
        return ToolResult(
            tool=self.name,
            ok=True,
            content=dumps(payload),
            data={**payload, "table_status": self.table.data.get("status")},
        )
