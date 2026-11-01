import pytest

from agent_eval.config import Settings, load_settings


@pytest.fixture
def mock_settings() -> Settings:
    """Settings from config.yaml with mock providers forced, ignoring the local environment."""
    return load_settings(
        use_env=False,
        overrides={"llm.provider": "mock", "embeddings.provider": "mock"},
    )
