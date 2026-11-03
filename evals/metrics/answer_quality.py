"""Deterministic answer checks: key facts (dates, numbers, phrases) must appear in the answer.

Matching is case- and accent-insensitive. Dates accept the usual French spellings; numbers
are matched as whole numbers (thin spaces and dots used as thousand separators are removed
first, so "15 000" contains 15000 but "2026" does not contain 26).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from evals.dataset import Fact

MONTHS = [
    "janvier",
    "fevrier",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "aout",
    "septembre",
    "octobre",
    "novembre",
    "decembre",
]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower().replace("’", "'"))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[  ]", " ", text)
    return re.sub(r"\s+", " ", text)


def date_patterns(d: date) -> list[re.Pattern[str]]:
    month = MONTHS[d.month - 1]
    day = r"(?:1\s?er|1|01|premier)" if d.day == 1 else rf"0?{d.day}"
    return [
        re.compile(rf"\b{d.isoformat()}\b"),
        re.compile(rf"\b{day}\s+{month}\s+{d.year}\b"),
        re.compile(rf"\b0?{d.day}[/.]0?{d.month}[/.]{d.year}\b"),
    ]


def _numbers(text: str) -> set[int]:
    # Join digit groups split by a space/dot thousand separator ("15 000", "15.000").
    joined = re.sub(r"(?<=\d)[ .](?=\d{3}\b)", "", text)
    return {int(n) for n in re.findall(r"(?<![\d-])\d+(?![\d-])", joined)}


def fact_present(fact: Fact, answer: str) -> bool:
    norm = normalize(answer)
    if fact.type == "date":
        return any(p.search(norm) for p in date_patterns(date.fromisoformat(str(fact.value))))
    if fact.type == "number":
        return int(fact.value) in _numbers(norm)  # type: ignore[arg-type]
    # Phrases must start at a word boundary ("ubl" must not match "publication"),
    # but may end mid-word so that stems like "exoner" or "renouvel" still match.
    return any(re.search(rf"(?<!\w){re.escape(normalize(alt))}", norm) for alt in fact.any_of or [])


@dataclass
class FactResult:
    expected: int
    found: int
    missing: list[dict[str, Any]] = field(default_factory=list)

    @property
    def recall(self) -> float | None:
        return None if self.expected == 0 else self.found / self.expected

    @property
    def all_found(self) -> bool | None:
        return None if self.expected == 0 else self.found == self.expected


def evaluate_facts(facts: list[Fact], answer: str) -> FactResult:
    missing = [f.model_dump() for f in facts if not fact_present(f, answer)]
    return FactResult(expected=len(facts), found=len(facts) - len(missing), missing=missing)
