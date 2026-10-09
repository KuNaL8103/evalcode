"""Live integration test for RAG graph (Task 10).

Marks: @pytest.mark.live AND @pytest.mark.slow
"""

from __future__ import annotations

import pytest

from evalcode.config import Settings, get_settings
from evalcode.graph import build_retriever, default_dependencies, run_task


@pytest.mark.live
@pytest.mark.slow
def test_live_rag_graph():
    """End-to-end live test with real Gemini and local Chroma index.

    - Skips with clear reason if GEMINI_API_KEY is not set.
    - Skips with clear reason if index is missing/empty.
    - Runs a json task (known to be in the index from Step 0.4).
    - Asserts history[0]["node"] == "retrieve", event n_docs > 0.
    - Asserts status in {"approved", "failed"} (if failed, failure_reason truthy).
    - Asserts token_usage llm_calls <= 4.
    - Closes retriever's store in finally.
    """
    settings = get_settings()

    # Check for API key
    if not settings.gemini_api_key or not settings.gemini_api_key.get_secret_value().strip():
        pytest.skip("GEMINI_API_KEY not set; skipping live test")

    # Build retriever
    retriever = build_retriever(settings)
    if retriever is None:
        pytest.skip("index missing/empty: run python -m evalcode.rag.ingest")

    try:
        # Use low retries to limit LLM calls
        deps = default_dependencies(Settings(max_retries=1, llm_model=settings.llm_model))

        # Task about JSON (known to be in the index)
        task = "write a function that parses a JSON string into a Python dict"
        result = run_task(task, deps, auto_approve=True)

        # History starts with retrieve
        history = result.get("history") or []
        assert len(history) > 0, "history should not be empty"
        first_node = history[0]["node"]
        assert first_node == "retrieve", f"first node is {first_node}, expected retrieve"

        # First retrieve event has n_docs > 0
        first_retrieve = history[0]["summary"]
        assert first_retrieve["n_docs"] > 0, f"expected n_docs > 0, got {first_retrieve['n_docs']}"

        # Status is approved or failed (with failure_reason)
        assert result["status"] in {"approved", "failed"}, f"unexpected status: {result['status']}"
        if result["status"] == "failed":
            assert result["failure_reason"], "failed status should have failure_reason"

        # Token usage: llm_calls <= 4 (retrieve rewrite + generate + analyze_error + revise max)
        token_usage = result.get("token_usage") or {}
        llm_calls = token_usage.get("llm_calls", 0)
        assert llm_calls <= 4, f"llm_calls {llm_calls} exceeds budget of 4"

    finally:
        # Close the retriever's store
        if retriever and hasattr(retriever, "store"):
            retriever.store.close()
