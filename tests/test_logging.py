import json
import logging

from agent_eval.observability.logging import JsonFormatter


def test_json_formatter_includes_extra_fields():
    record = logging.LogRecord("agent", logging.INFO, __file__, 1, "tool call", (), None)
    record.tool = "search_docs"
    record.latency_ms = 12.5
    payload = json.loads(JsonFormatter().format(record))
    assert payload["msg"] == "tool call"
    assert payload["tool"] == "search_docs"
    assert payload["latency_ms"] == 12.5
    assert payload["level"] == "INFO"
