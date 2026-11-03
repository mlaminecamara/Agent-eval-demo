"""Retrieval metrics against expected_sources: hit@k and MRR.

A retrieved passage matches an expected source when the doc_id is the same and, if the
expected source names a page, that page falls within the passage's page range. Sources
without a page (e.g. web pages) match on doc_id alone.

Two views are computed:
- direct: the retriever is queried with the case question itself. Independent of the agent,
  so it isolates chunking / embeddings / reranking.
- agent: the passages the agent actually retrieved through search_docs, in call order.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from evals.dataset import SourceRef


def passage_matches(passage: dict[str, Any], source: SourceRef) -> bool:
    if passage.get("doc_id") != source.doc_id:
        return False
    if source.page is None:
        return True
    start, end = passage.get("page_start"), passage.get("page_end")
    if start is None or end is None:
        return False
    return int(start) <= source.page <= int(end)


def first_relevant_rank(
    passages: Sequence[dict[str, Any]], sources: Sequence[SourceRef]
) -> int | None:
    for rank, passage in enumerate(passages, start=1):
        if any(passage_matches(passage, s) for s in sources):
            return rank
    return None


def hit_at_k(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def reciprocal_rank(rank: int | None, cutoff: int | None = None) -> float:
    if rank is None or (cutoff is not None and rank > cutoff):
        return 0.0
    return 1.0 / rank
