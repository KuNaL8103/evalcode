"""Unit tests for the LangGraph state schema (state.py)."""

from __future__ import annotations

from datetime import datetime
from typing import get_type_hints

from evalcode.state import AgentState, TokenUsage, merge_usage, utc_now_iso


def test_merge_usage_sums_and_tolerates_none_and_missing() -> None:
    # (a, b, expected) — None sides, missing keys, and None values are all tolerated.
    cases: list[tuple[TokenUsage | None, TokenUsage | None, dict[str, object]]] = [
        (None, None, {}),
        (None, {"llm_calls": 2, "wait_s": 1.0}, {"llm_calls": 2, "wait_s": 1.0}),
        ({"llm_calls": 1}, None, {"llm_calls": 1}),
        (
            {"llm_calls": 1, "wait_s": 0.5, "input_tokens": 3},
            {"llm_calls": 2, "wait_s": 1.5, "api_retries": 1},
            {"llm_calls": 3, "wait_s": 2.0, "input_tokens": 3, "api_retries": 1},
        ),
        ({"input_tokens": None}, {"input_tokens": 4}, {"input_tokens": 4}),
        ({"total_tokens": 7}, {"total_tokens": None}, {"total_tokens": 7}),
        ({"estimated_calls": None}, {"estimated_calls": None}, {}),
    ]
    for a, b, expected in cases:
        assert merge_usage(a, b) == expected, (a, b)

    # merge_usage is the reducer wired into AgentState; utc_now_iso is tz-aware ISO-8601.
    reducer = get_type_hints(AgentState, include_extras=True)["token_usage"].__metadata__[0]
    assert reducer is merge_usage
    parsed = datetime.fromisoformat(utc_now_iso())
    assert parsed.tzinfo is not None
