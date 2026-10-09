"""Live integration test for the full graph (requires GEMINI_API_KEY)."""

from __future__ import annotations

import pytest

from evalcode.config import Settings, get_settings
from evalcode.graph import default_dependencies, run_task


@pytest.mark.live
def test_live_graph_basic():
    """Run a simple task through the full graph with real Gemini."""
    api_key = get_settings().gemini_api_key.get_secret_value()
    if not api_key:
        pytest.skip("GEMINI_API_KEY not configured (env or .env)")

    settings = Settings(max_retries=1)
    deps = default_dependencies(settings, rag=False)

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
