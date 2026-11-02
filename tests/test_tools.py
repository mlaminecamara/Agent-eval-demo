import json
import re

import pytest

from agent_eval.config import PROJECT_ROOT
from agent_eval.embeddings import build_embedder
from agent_eval.ingest.chunker import chunk_documents
from agent_eval.ingest.indexer import index_chunks
from agent_eval.ingest.loader import load_manifest
from agent_eval.llm.base import ToolCall
from agent_eval.retrieval.base import IndexSpec
from agent_eval.retrieval.memory_store import MemoryStore
from agent_eval.retrieval.retriever import Retriever
from agent_eval.tools.calculate_date import CalculateDateTool, add_months
from agent_eval.tools.get_deadline import DeadlineTable, GetDeadlineTool
from agent_eval.tools.registry import DEADLINES_PATH, ToolRegistry, build_registry
from agent_eval.tools.search_docs import SearchDocsTool

MANIFEST = PROJECT_ROOT / "data" / "corpus" / "manifest.yaml"
DEADLINES = PROJECT_ROOT / DEADLINES_PATH


@pytest.fixture(scope="module")
def retriever():
    from agent_eval.config import load_settings

    settings = load_settings(use_env=False, overrides={"embeddings.provider": "mock"})
    embedder = build_embedder(settings.embeddings)
    store = MemoryStore(IndexSpec(embedder.model, embedder.dimension, 500, 50))
    index_chunks(chunk_documents(load_manifest(MANIFEST), 500, 50), store, embedder)
    return Retriever(store, embedder, top_k=5)


@pytest.fixture(scope="module")
def registry(retriever):
    return ToolRegistry(
        [
            SearchDocsTool(retriever),
            GetDeadlineTool(DeadlineTable.load(DEADLINES)),
            CalculateDateTool(),
        ]
    )


# --- schemas ---------------------------------------------------------------------------------


def test_schemas_are_strict_and_match_validation_models(registry):
    for tool in registry.tools.values():
        schema = tool.parameters
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        # Strict tool use: every property must be required (optional = nullable type).
        assert set(schema["required"]) == set(schema["properties"])
        assert set(schema["properties"]) == set(tool.args_model.model_fields), tool.name
        spec = tool.spec()
        assert spec.strict and spec.name == tool.name and spec.description


# --- search_docs -----------------------------------------------------------------------------


def test_search_docs_returns_cited_passages(registry):
    call = ToolCall(
        id="1", name="search_docs", arguments={"query": "plateforme agréée", "top_k": 3}
    )
    result = registry.execute(call)
    assert result.ok
    assert len(result.data["passages"]) == 3
    assert [p["ref"] for p in result.data["passages"]] == ["S1", "S2", "S3"]
    assert result.content.count("<extrait ") == 3
    assert all(p["collection"] == "corpus" for p in result.data["passages"])


@pytest.mark.parametrize(
    "arguments",
    [
        {"query": "ab", "top_k": None},  # too short
        {"query": "   abc  ", "top_k": 0},  # top_k below range
        {"query": "plateforme", "top_k": 11},  # above range
        {"query": "plateforme", "top_k": "3"},  # strict: no str -> int coercion
        {"query": "plateforme", "top_k": None, "extra": 1},  # unknown key
        {"top_k": 3},  # missing query
    ],
)
def test_search_docs_rejects_invalid_arguments(registry, arguments):
    result = registry.execute(ToolCall(id="x", name="search_docs", arguments=arguments))
    assert not result.ok
    assert result.error_type == "validation"
    assert "Invalid arguments for search_docs" in result.content


# --- get_deadline ----------------------------------------------------------------------------


def test_get_deadline_lookup(registry):
    r = registry.execute(
        ToolCall(
            id="1", name="get_deadline", arguments={"entity_size": "PME", "obligation": "emission"}
        )
    )
    assert r.ok
    payload = json.loads(r.content)
    assert payload["date"] == "2027-09-01"
    assert isinstance(payload["verified"], bool)  # state comes from the reviewed file
    assert {s["doc_id"] for s in payload["sources"]} >= {"faq-fe"}


@pytest.mark.parametrize(
    "arguments",
    [
        {"entity_size": "SME", "obligation": "emission"},
        {"entity_size": "PME", "obligation": "invoicing"},
        {"entity_size": "pme", "obligation": "emission"},
        {"entity_size": "PME"},
    ],
)
def test_get_deadline_rejects_invalid_arguments(registry, arguments):
    r = registry.execute(ToolCall(id="1", name="get_deadline", arguments=arguments))
    assert not r.ok and r.error_type == "validation"


def test_deadline_table_is_complete_and_consistent():
    table = json.loads(DEADLINES.read_text(encoding="utf-8"))
    keys = {(d["entity_size"], d["obligation"]) for d in table["deadlines"]}
    assert keys == {(s, o) for s in table["entity_sizes"] for o in table["obligations"]}
    by_key = {(d["entity_size"], d["obligation"]): d["date"] for d in table["deadlines"]}
    # e-reporting follows the e-invoicing emission calendar (faq-fe 8.1).
    for size in table["entity_sizes"]:
        assert by_key[(size, "e_reporting")] == by_key[(size, "emission")]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def test_deadline_quotes_appear_verbatim_in_cited_documents():
    """Guards against invented sources: each quote (split on '...') must be found in the
    cited document's extracted text."""
    docs = {d.doc_id: _norm(d.text) for d in load_manifest(MANIFEST)}
    table = json.loads(DEADLINES.read_text(encoding="utf-8"))
    for entry in table["deadlines"]:
        for src in entry["sources"]:
            assert src["doc_id"] in docs, src
            for part in src["quote"].split("..."):
                part = _norm(part)
                if part:
                    assert part in docs[src["doc_id"]], (entry["entity_size"], src)


# --- calculate_date --------------------------------------------------------------------------


def _calc(registry, **arguments):
    full = {"days": None, "months": None, "end_date": None} | arguments
    return registry.execute(ToolCall(id="c", name="calculate_date", arguments=full))


def test_calculate_date_operations(registry):
    r = _calc(registry, operation="add", start_date="2026-09-01", months=12)
    assert json.loads(r.content)["result_date"] == "2027-09-01"
    r = _calc(registry, operation="subtract", start_date="2027-09-01", days=90)
    assert json.loads(r.content)["result_date"] == "2027-06-03"
    r = _calc(registry, operation="diff", start_date="2026-11-01", end_date="2027-09-01")
    assert json.loads(r.content)["days_between"] == 304
    r = _calc(registry, operation="diff", start_date="2027-09-01", end_date="2026-11-01")
    assert json.loads(r.content)["days_between"] == -304


def test_add_months_clamps_to_month_end():
    from datetime import date

    assert add_months(date(2026, 1, 31), 1) == date(2026, 2, 28)
    assert add_months(date(2028, 1, 31), 1) == date(2028, 2, 29)
    assert add_months(date(2026, 12, 15), 1) == date(2027, 1, 15)
    assert add_months(date(2026, 3, 31), -1) == date(2026, 2, 28)


@pytest.mark.parametrize(
    "arguments",
    [
        {"operation": "add", "start_date": "2026-09-01"},  # no amount
        {"operation": "add", "start_date": "2026-09-01", "days": 3, "end_date": "2026-10-01"},
        {"operation": "diff", "start_date": "2026-09-01"},  # no end_date
        {"operation": "diff", "start_date": "2026-09-01", "end_date": "2026-10-01", "days": 1},
        {"operation": "add", "start_date": "2026-02-30", "days": 1},  # not a real date
        {"operation": "add", "start_date": "01/09/2026", "days": 1},  # wrong format
        {"operation": "add", "start_date": "2026-09-01", "days": -5},  # negative amount
        {"operation": "multiply", "start_date": "2026-09-01", "days": 1},
    ],
)
def test_calculate_date_rejects_invalid_arguments(registry, arguments):
    r = _calc(registry, **arguments)
    assert not r.ok and r.error_type == "validation", r.content


# --- registry --------------------------------------------------------------------------------


def test_registry_unknown_tool_and_parse_error(registry):
    r = registry.execute(ToolCall(id="1", name="delete_everything", arguments={}))
    assert not r.ok and r.error_type == "unknown_tool"
    assert "search_docs" in r.content
    r = registry.execute(
        ToolCall(id="2", name="search_docs", arguments={}, parse_error="invalid JSON arguments")
    )
    assert not r.ok and r.error_type == "validation"


def test_registry_records_latency_and_never_raises(registry, monkeypatch):
    tool = registry.tools["calculate_date"]

    def boom(args):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(tool, "execute", boom)
    r = _calc(registry, operation="add", start_date="2026-09-01", days=1)
    assert not r.ok and r.error_type == "execution"
    assert "disk on fire" not in r.content  # internal details stay out of the model context
    assert "disk on fire" in r.data["error"]
    assert r.latency_s >= 0


def test_build_registry_from_settings(mock_settings):
    mock_settings.retrieval.store = "memory"
    reg = build_registry(mock_settings, collections=["corpus", "adversarial"])
    assert [s.name for s in reg.specs()] == ["search_docs", "get_deadline", "calculate_date"]
    assert reg.tools["search_docs"].collections == ["corpus", "adversarial"]
