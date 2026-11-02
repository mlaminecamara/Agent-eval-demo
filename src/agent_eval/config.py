"""Configuration: config.yaml, overridden by environment variables (and .env)."""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

LLMProviderName = Literal["mock", "openai", "anthropic"]
EmbeddingProviderName = Literal["mock", "openai"]


class LLMConfig(BaseModel):
    provider: LLMProviderName = "mock"
    model: str | None = None
    temperature: float | None = None  # not sent when None
    max_output_tokens: int = 4096
    timeout_s: float = 60
    effort: str | None = None  # OpenAI: reasoning_effort; Anthropic: output_config.effort
    defaults: dict[str, str] = Field(default_factory=dict)

    @property
    def resolved_model(self) -> str:
        if self.model:
            return self.model
        if self.provider not in self.defaults:
            raise ValueError(f"No model set and no default for provider {self.provider!r}")
        return self.defaults[self.provider]


class JudgeConfig(BaseModel):
    provider: LLMProviderName = "anthropic"
    model: str = "claude-haiku-5-5"


class EmbeddingConfig(BaseModel):
    provider: EmbeddingProviderName = "mock"
    model: str | None = None
    defaults: dict[str, str] = Field(default_factory=dict)
    dimensions: dict[str, int] = Field(default_factory=dict)

    @property
    def resolved_model(self) -> str:
        if self.model:
            return self.model
        if self.provider not in self.defaults:
            raise ValueError(f"No model set and no default for provider {self.provider!r}")
        return self.defaults[self.provider]

    @property
    def dimension(self) -> int:
        model = self.resolved_model
        if model not in self.dimensions:
            raise ValueError(f"Unknown embedding dimension for model {model!r}")
        return self.dimensions[model]


class DatabaseConfig(BaseModel):
    url: str = "postgresql://agent:agent@localhost:5432/agent_eval"


class CorpusConfig(BaseModel):
    dir: str = "data/corpus"
    manifest: str = "data/corpus/manifest.yaml"
    adversarial_dir: str = "data/adversarial"


class HNSWConfig(BaseModel):
    m: int = 16
    ef_construction: int = 64
    ef_search: int = 40


class RetrievalConfig(BaseModel):
    store: Literal["pgvector", "memory"] = "pgvector"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5
    candidate_k: int = 20
    rerank: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    hnsw: HNSWConfig = Field(default_factory=HNSWConfig)


class AgentSettings(BaseModel):
    max_iterations: int = 6
    max_tool_result_chars: int = 6000
    today: date | None = None  # reference date given to the agent; None = real today


class CostGuardConfig(BaseModel):
    max_llm_calls: int = 500
    max_total_tokens: int = 2_000_000
    pricing_file: str = "config/pricing.yaml"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    runs_dir: str = "runs"


class Settings(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    cost_guard: CostGuardConfig = Field(default_factory=CostGuardConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @property
    def is_mock(self) -> bool:
        return self.llm.provider == "mock" and self.embeddings.provider == "mock"

    def resolve_path(self, relative: str) -> Path:
        path = Path(relative)
        return path if path.is_absolute() else PROJECT_ROOT / path


# (env var, path in the config tree, caster)
_ENV_OVERRIDES: list[tuple[str, tuple[str, ...], Callable[[str], Any]]] = [
    ("LLM_PROVIDER", ("llm", "provider"), str),
    ("LLM_MODEL", ("llm", "model"), str),
    ("EMBEDDING_PROVIDER", ("embeddings", "provider"), str),
    ("EMBEDDING_MODEL", ("embeddings", "model"), str),
    ("JUDGE_PROVIDER", ("judge", "provider"), str),
    ("JUDGE_MODEL", ("judge", "model"), str),
    ("DATABASE_URL", ("database", "url"), str),
    ("VECTOR_STORE", ("retrieval", "store"), str),
    ("RERANK", ("retrieval", "rerank"), lambda v: v.lower() in {"1", "true", "yes"}),
    ("MAX_LLM_CALLS", ("cost_guard", "max_llm_calls"), int),
    ("MAX_TOTAL_TOKENS", ("cost_guard", "max_total_tokens"), int),
    ("LOG_LEVEL", ("logging", "level"), str),
]


def _set_path(tree: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    node = tree
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value


def load_settings(
    config_path: Path | str | None = None,
    overrides: dict[str, Any] | None = None,
    use_env: bool = True,
) -> Settings:
    """Load settings. Precedence: overrides > environment > config file > defaults.

    Empty environment variables are ignored, so a blank line in .env does not erase a value.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    tree: dict[str, Any] = {}
    if path.exists():
        tree = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    if use_env:
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        for env_name, cfg_path, cast in _ENV_OVERRIDES:
            raw = os.environ.get(env_name)
            if raw is not None and raw.strip() != "":
                _set_path(tree, cfg_path, cast(raw.strip()))

    for dotted, value in (overrides or {}).items():
        _set_path(tree, tuple(dotted.split(".")), value)

    return Settings.model_validate(tree)
