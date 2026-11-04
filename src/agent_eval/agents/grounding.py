"""Deterministic grounding check on dates, used by the LangGraph agent's verify node.

A date written in the answer is "supported" when the same calendar date appears in the
question, in a tool result the agent received, or is today's date. Dates are recognised in
ISO form (2027-09-01), French long form (1er septembre 2027) and numeric form (01/09/2027),
so a date quoted from a passage in one form and written in another still matches.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date

MONTHS = (
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
)
_ISO = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_LONG = re.compile(r"\b(1\s?er|premier|\d{1,2})\s+(" + "|".join(MONTHS) + r")\s+(\d{4})\b")
_NUMERIC = re.compile(r"\b(\d{1,2})[/.](\d{1,2})[/.](\d{4})\b")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text.replace(" ", " ").replace(" ", " "))


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def extract_dates(text: str) -> set[date]:
    norm = _normalize(text)
    found: set[date | None] = set()
    for y, m, d in _ISO.findall(norm):
        found.add(_safe_date(int(y), int(m), int(d)))
    for d, month, y in _LONG.findall(norm):
        day = 1 if d.replace(" ", "") in ("1er", "premier") else int(d)
        found.add(_safe_date(int(y), MONTHS.index(month) + 1, day))
    for d, m, y in _NUMERIC.findall(norm):
        found.add(_safe_date(int(y), int(m), int(d)))
    return {d for d in found if d is not None}


def unsupported_dates(answer: str, evidence: list[str], today: date | None = None) -> list[date]:
    """Dates in `answer` that appear in none of the `evidence` texts (nor equal `today`)."""
    supported: set[date] = set()
    for text in evidence:
        supported |= extract_dates(text)
    if today:
        supported.add(today)
    return sorted(extract_dates(answer) - supported)
