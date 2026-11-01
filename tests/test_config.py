from agent_eval.config import load_settings


def test_defaults_from_config_file(mock_settings):
    assert mock_settings.is_mock
    assert mock_settings.llm.resolved_model == "mock-llm-v1"
    assert mock_settings.embeddings.dimension == 256
    assert mock_settings.llm.defaults["openai"] == "gpt-5.4-mini-2026-03-17"


def test_env_overrides_config(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("MAX_LLM_CALLS", "7")
    monkeypatch.setenv("LLM_MODEL", "")  # blank values are ignored
    s = load_settings()
    assert s.llm.provider == "anthropic"
    assert s.llm.resolved_model == "claude-sonnet-5-5"
    assert s.cost_guard.max_llm_calls == 7


def test_explicit_overrides_win(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    s = load_settings(overrides={"llm.provider": "openai", "llm.model": "gpt-x"})
    assert s.llm.provider == "openai"
    assert s.llm.resolved_model == "gpt-x"
