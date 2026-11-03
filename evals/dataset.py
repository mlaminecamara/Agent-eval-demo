"""Golden dataset schema (JSON lines, one case per line). See evals/datasets/SCHEMA.md."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Category = Literal["factual", "multi_step", "out_of_scope", "injection"]
Status = Literal["draft", "validated"]


class Fact(BaseModel):
    """A key fact the answer must contain, checked deterministically.

    - date: value is YYYY-MM-DD; matches common French spellings ("1er septembre 2027",
      "01/09/2027", "2027-09-01").
    - number: value is an integer; matched as a whole number.
    - text: any_of lists acceptable phrasings; matched case- and accent-insensitively.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["date", "number", "text"]
    value: str | int | None = None
    any_of: list[str] | None = None
    description: str = ""

    @model_validator(mode="after")
    def _check(self) -> Fact:
        if self.type == "date":
            date.fromisoformat(str(self.value))
        elif self.type == "number":
            if not isinstance(self.value, int):
                raise ValueError("number facts need an integer value")
        elif not self.any_of:
            raise ValueError("text facts need any_of")
        return self


class SourceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    page: int | None = None  # None: any page of the document counts
    section: str | None = None  # informative only (not used for matching)


class ExpectedToolCall(BaseModel):
    """name: one tool or a list of acceptable tools. args: key arguments that must match
    (subset match: other arguments are ignored)."""

    model_config = ConfigDict(extra="forbid")

    name: str | list[str]
    args: dict[str, Any] = Field(default_factory=dict)

    @property
    def names(self) -> list[str]:
        return [self.name] if isinstance(self.name, str) else list(self.name)


class Injection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vector: Literal["document", "user_message"]
    payload_markers: list[str] = Field(default_factory=list)  # must NOT appear in the answer
    adversarial_doc: str | None = None


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    category: Category
    status: Status = "draft"
    question: str
    reference_date: date = date(2026, 11, 1)
    expected_answer: str
    expected_facts: list[Fact] = Field(default_factory=list)
    expected_sources: list[SourceRef] = Field(default_factory=list)
    expected_tool_calls: list[ExpectedToolCall] = Field(default_factory=list)
    expected_steps: int | None = None  # minimal number of LLM turns for a direct solution
    must_refuse: bool = False
    collections: list[str] = Field(default_factory=lambda: ["corpus"])
    injection: Injection | None = None
    notes: str = ""

    @model_validator(mode="after")
    def _consistent(self) -> Case:
        if self.must_refuse and (self.expected_tool_calls or self.expected_facts):
            raise ValueError(f"{self.id}: refusal cases expect no tool calls and no facts")
        if self.category == "injection" and self.injection is None:
            raise ValueError(f"{self.id}: injection cases need an `injection` block")
        if self.category == "out_of_scope" and not self.must_refuse:
            raise ValueError(f"{self.id}: out_of_scope cases must have must_refuse=true")
        return self


def load_dataset(path: Path | str) -> list[Case]:
    cases: list[Case] = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(Case.model_validate(json.loads(line)))
        except Exception as exc:
            raise ValueError(f"{path}:{i}: {exc}") from exc
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate case ids in {path}")
    return cases


def dataset_fingerprint(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def status_counts(cases: list[Case]) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in cases:
        out[c.status] = out.get(c.status, 0) + 1
    return out
