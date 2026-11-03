"""Render golden.jsonl as a Markdown table for human review.

    uv run python -m evals.dataset_review   ->  evals/datasets/golden_review.md

Generated file: edit golden.jsonl, not the Markdown.
"""

from __future__ import annotations

import sys
from pathlib import Path

from evals.dataset import Case, load_dataset, status_counts

DATASET = Path(__file__).parent / "datasets" / "golden.jsonl"
OUT = Path(__file__).parent / "datasets" / "golden_review.md"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _facts(case: Case) -> str:
    out = []
    for f in case.expected_facts:
        value = f.value if f.value is not None else " / ".join(f.any_of or [])
        out.append(f"{f.type}: {value}")
    return "<br>".join(out) or "–"


def _sources(case: Case) -> str:
    return (
        "<br>".join(
            f"{s.doc_id}"
            + (f" p.{s.page}" if s.page else "")
            + (f" ({s.section})" if s.section else "")
            for s in case.expected_sources
        )
        or "–"
    )


def _tools(case: Case) -> str:
    out = []
    for t in case.expected_tool_calls:
        args = ", ".join(f"{k}={v}" for k, v in t.args.items())
        out.append(f"{' or '.join(t.names)}({args})")
    return "<br>".join(out) or "none"


def render(cases: list[Case]) -> str:
    lines = [
        "# Golden dataset: review sheet",
        "",
        "Generated from `golden.jsonl` by `python -m evals.dataset_review`. Do not edit by hand.",
        "",
        f"{len(cases)} cases. Status: {status_counts(cases)}.",
        "",
        "| ID | Status | Question | Expected answer | Key facts | Sources | Tool calls "
        "| Steps | Notes |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in cases:
        extra = []
        if c.must_refuse:
            extra.append("MUST REFUSE")
        if c.injection:
            extra.append(f"injection ({c.injection.vector}): markers {c.injection.payload_markers}")
        if c.reference_date.isoformat() != "2026-11-01":
            extra.append(f"reference date {c.reference_date}")
        notes = "; ".join([*extra, c.notes] if c.notes else extra)
        lines.append(
            f"| {c.id} | {c.status} | {_cell(c.question)} | {_cell(c.expected_answer)} "
            f"| {_cell(_facts(c))} | {_cell(_sources(c))} | {_cell(_tools(c))} "
            f"| {c.expected_steps if c.expected_steps is not None else '–'} | {_cell(notes)} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    OUT.write_text(render(load_dataset(DATASET)), encoding="utf-8")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
