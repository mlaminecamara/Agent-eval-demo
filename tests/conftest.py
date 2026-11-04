import pytest

from agent_eval.config import PROJECT_ROOT, Settings, load_settings
from agent_eval.tools.registry import ToolRegistry


@pytest.fixture
def mock_settings() -> Settings:
    """Settings from config.yaml with mock providers forced, ignoring the local environment."""
    return load_settings(
        use_env=False,
        overrides={"llm.provider": "mock", "embeddings.provider": "mock"},
    )


@pytest.fixture(scope="session")
def registry() -> ToolRegistry:
    """The three tools over the real corpus, indexed in memory with mock embeddings."""
    from agent_eval.embeddings import build_embedder
    from agent_eval.ingest.chunker import chunk_documents
    from agent_eval.ingest.indexer import index_chunks
    from agent_eval.ingest.loader import load_manifest
    from agent_eval.retrieval.base import IndexSpec
    from agent_eval.retrieval.memory_store import MemoryStore
    from agent_eval.retrieval.retriever import Retriever
    from agent_eval.tools.calculate_date import CalculateDateTool
    from agent_eval.tools.get_deadline import DeadlineTable, GetDeadlineTool
    from agent_eval.tools.registry import DEADLINES_PATH
    from agent_eval.tools.search_docs import SearchDocsTool

    settings = load_settings(use_env=False, overrides={"embeddings.provider": "mock"})
    embedder = build_embedder(settings.embeddings)
    store = MemoryStore(IndexSpec(embedder.model, embedder.dimension, 500, 50))
    manifest = PROJECT_ROOT / "data" / "corpus" / "manifest.yaml"
    index_chunks(chunk_documents(load_manifest(manifest), 500, 50), store, embedder)
    return ToolRegistry(
        [
            SearchDocsTool(Retriever(store, embedder, top_k=5)),
            GetDeadlineTool(DeadlineTable.load(PROJECT_ROOT / DEADLINES_PATH)),
            CalculateDateTool(),
        ]
    )
