"""Live integration test for the full graph (requires GEMINI_API_KEY)."""

from __future__ import annotations

import pytest

from evalcode.config import Settings, get_settings
from evalcode.graph import default_dependencies, run_task


@pytest.mark.live
def test_live_graph_basic(tmp_path):
    """Run a simple task through the full graph with real Gemini."""
    api_key = get_settings().gemini_api_key.get_secret_value()
    if not api_key:
        pytest.skip("GEMINI_API_KEY not configured (env or .env)")

    settings = Settings(max_retries=1, log_dir=str(tmp_path))
    deps = default_dependencies(settings, rag=False, observe=True)

    result = run_task(
        "write a function add(a, b) that returns the sum of two numbers",
        deps,
        auto_approve=True,
    )

    assert result["status"] in {"approved", "failed"}
    if result["status"] == "failed":
        assert result["failure_reason"]
    assert result["token_usage"]["llm_calls"] <= 4
    assert len(result["history"]) > 0

    # Observability assertions
    events = deps.logger.read_events()
    assert events and events[0]["node"] == "generate"
    assert events[-1]["cumulative_usage"]["llm_calls"] == result["token_usage"]["llm_calls"]
    assert (deps.logger.run_dir / "summary.json").exists()

    # API key redacted in events
    events_text = (deps.logger.events_path).read_text(encoding="utf-8")
    assert api_key not in events_text
