"""Unit tests for the Gemini LLM client (llm.py).

All sleeps/clocks/RNG are fakes: tests run instantly and assert the exact
sleep durations the backoff/throttle logic requested.
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import SecretStr

from evalcode.config import Settings
from evalcode.errors import (
    ConfigError,
    DailyQuotaExceeded,
    LLMAuthError,
    LLMBudgetExceeded,
    LLMModelError,
    LLMUnavailable,
)
from evalcode.llm import LLMClient, extract_usage, get_chat_model, strip_reasoning
from tests.fakes import (
    FakeChatModel,
    make_connection_error,
    make_rate_limit_error,
    make_status_error,
    make_timeout_error,
)

# Built at runtime so the secret-scan hygiene test never sees a literal key.
FAKE_KEY = "AIza" + "FAKE" * 8


def make_settings(**overrides: Any) -> Settings:
    defaults: dict[str, Any] = dict(
        gemini_api_key=SecretStr(FAKE_KEY),
        llm_min_interval_s=0.0,
        llm_backoff_base_s=2.0,
        llm_backoff_max_s=60.0,
        llm_max_wait_s=120.0,
        llm_max_api_retries=5,
        max_llm_calls_per_run=10,
    )
    defaults.update(overrides)
    return Settings(**defaults)


class FakeClock:
    """A monotonic-like clock the test can advance manually."""

    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class RecordingSleep:
    """Records requested sleeps and (optionally) advances a FakeClock by them."""

    def __init__(self, clock: FakeClock | None = None) -> None:
        self.calls: list[float] = []
        self._clock = clock

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        if self._clock is not None:
            self._clock.advance(seconds)


def make_client(
    script: list[Any],
    settings: Settings | None = None,
    rng: Any = None,
) -> tuple[LLMClient, FakeChatModel, RecordingSleep]:
    chat = FakeChatModel(script)
    clock = FakeClock()
    sleep = RecordingSleep(clock)
    client = LLMClient(
        chat,
        settings or make_settings(),
        sleep=sleep,
        clock=clock,
        rng=rng if rng is not None else (lambda: 1.0),
    )
    return client, chat, sleep


def ok_message(text: str = "ok", **kwargs: Any) -> AIMessage:
    return AIMessage(
        content=text,
        usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        **kwargs,
    )


def test_extract_usage_and_strip_reasoning() -> None:
    # usage_metadata present → exact provider numbers, not estimated
    with_meta = AIMessage(
        content="hi", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    )
    usage = extract_usage(with_meta, "prompt", "hi")
    assert usage["input_tokens"] == 10
    assert usage["output_tokens"] == 5
    assert usage["total_tokens"] == 15
    assert usage["llm_calls"] == 1
    assert usage["estimated_calls"] == 0

    # no metadata → estimate ceil(chars/4), flagged estimated
    plain = AIMessage(content="ok")
    usage2 = extract_usage(plain, "x" * 11, "y" * 5)
    assert usage2["input_tokens"] == 3
    assert usage2["output_tokens"] == 2
    assert usage2["total_tokens"] == 5
    assert usage2["estimated_calls"] == 1
    assert usage2["llm_calls"] == 1

    # tag literals built from parts (CLAUDE.md tag-literal safety)
    open_tag = "<" + "think" + ">"
    close_tag = "</" + "think" + ">"
    assert strip_reasoning(f"{open_tag}reasoning here{close_tag}final answer") == "final answer"
    assert strip_reasoning(f"prefix {open_tag}never closed") == "prefix"
    assert strip_reasoning(f"{open_tag}a{close_tag}mid{open_tag}b{close_tag}") == "mid"
    # stray closing tag with no opening tag → drop through the LAST closing tag
    assert strip_reasoning(f"prefix {close_tag}suffix kept") == "suffix kept"
    assert strip_reasoning(f"{close_tag}only close") == "only close"
    assert strip_reasoning(f"a {close_tag}b {close_tag}c") == "c"
    assert strip_reasoning("plain text") == "plain text"
    assert strip_reasoning("") == ""


def test_get_chat_model_configuration() -> None:
    settings = make_settings(
        llm_model="gemini-3.5-flash-lite",
        llm_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        llm_timeout_s=55.5,
    )
    model = get_chat_model(settings)
    assert model.model_name == "gemini-3.5-flash-lite"
    assert model.openai_api_base == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert model.max_retries == 0  # SDK retries disabled; LLMClient owns retries
    assert model.temperature == settings.llm_temperature
    assert model.max_tokens == settings.llm_max_tokens
    assert model.request_timeout == 55.5
    assert FAKE_KEY not in repr(model)  # SecretStr stays masked

    with pytest.raises(ConfigError):
        get_chat_model(Settings())  # no key at all


def test_invoke_success_returns_text_and_usage() -> None:
    client, chat, sleep = make_client(
        [ok_message("done", response_metadata={"model_name": "gemini-3.5-flash-lite"})]
    )
    messages = [HumanMessage(content="say done")]
    response = client.invoke_text(messages, purpose="generate")
    assert response.text == "done"
    assert response.model == "gemini-3.5-flash-lite"
    assert response.waited_s == 0.0
    assert response.usage["total_tokens"] == 5
    assert response.usage["estimated_calls"] == 0
    assert response.usage["llm_calls"] == 1
    assert response.usage["api_retries"] == 0
    assert chat.calls == [messages]
    assert sleep.calls == []
    assert client.stats == {"calls": 1, "api_retries": 0, "wait_s": 0.0}


def test_backoff_exponential_with_jitter_bounds() -> None:
    # base=2.0: delays are 2, 4, … scaled by jitter ∈ [0, 1]
    for jitter, expected_sleeeps in ((1.0, [2.0, 4.0]), (0.5, [1.0, 2.0]), (0.0, [0.0, 0.0])):
        client, chat, sleep = make_client(
            [make_rate_limit_error(), make_rate_limit_error(), ok_message()],
            rng=lambda j=jitter: j,
        )
        response = client.invoke_text([HumanMessage(content="hi")])
        assert response.text == "ok"
        assert len(chat.calls) == 3
        assert sleep.calls == expected_sleeeps
        assert client.stats["api_retries"] == 2
        assert response.usage["api_retries"] == 2
        assert response.waited_s == pytest.approx(sum(expected_sleeeps))


def test_rate_limit_headers_honored_as_minimum_wait() -> None:
    # Retry-After: 30 (seconds) beats the 2s base backoff
    client, chat, sleep = make_client(
        [make_rate_limit_error(headers={"retry-after": "30"}), ok_message()]
    )
    client.invoke_text([HumanMessage(content="hi")])
    assert sleep.calls == [30.0]
    assert len(chat.calls) == 2

    # X-RateLimit-Reset: epoch-millis ~20s from now
    reset_ms = int(time.time() * 1000) + 20_000
    client2, chat2, sleep2 = make_client(
        [make_rate_limit_error(headers={"x-ratelimit-reset": str(reset_ms)}), ok_message()]
    )
    client2.invoke_text([HumanMessage(content="hi")])
    assert 15.0 <= sleep2.calls[0] <= 25.0
    assert len(chat2.calls) == 2


def test_wait_above_max_wait_raises_daily_quota() -> None:
    settings = make_settings(llm_max_wait_s=10.0)
    client, chat, sleep = make_client(
        [make_rate_limit_error(headers={"retry-after": "30"})], settings=settings
    )
    with pytest.raises(DailyQuotaExceeded):
        client.invoke_text([HumanMessage(content="hi")])
    assert len(chat.calls) == 1  # failed fast, no retries
    assert sleep.calls == []


def test_daily_limit_429_fails_fast_without_retries() -> None:
    # Bodies loop: the legacy free-tier daily text plus the other markers.
    bodies = [
        # The legacy free-tier 429 text (implicit concatenation stays verbatim).
        "Rate limit exceeded: free-models-per-day. Add 10 credits to unlock "
        "1000 free model requests per day",
        "You have reached your per-day request limit",
        "Daily limit exceeded for this model",
        "Request limited per day; try again tomorrow",
    ]
    for body in bodies:
        error = make_rate_limit_error(
            body={"error": {"message": body}}, message=body, headers={"retry-after": "5"}
        )
        client, chat, sleep = make_client([error, ok_message()])
        with pytest.raises(DailyQuotaExceeded):
            client.invoke_text([HumanMessage(content="hi")])
        assert len(chat.calls) == 1  # ZERO retries
        assert sleep.calls == []

    # (a) Daily-quota body with exact free-tier marker → zero retries.
    quota_body = "GenerateRequestsPerDayPerProjectPerModel-FreeTier: limit"
    error = make_rate_limit_error(
        body={"error": {"message": quota_body}}, message=quota_body, headers={"retry-after": "5"}
    )
    client, chat, sleep = make_client([error, ok_message()])
    with pytest.raises(DailyQuotaExceeded):
        client.invoke_text([HumanMessage(content="hi")])
    assert len(chat.calls) == 1
    assert sleep.calls == []

    # A 200-with-error body (ValueError from langchain_openai) carrying the
    # real daily text → same fail-fast, zero retries.
    client, chat, sleep = make_client([ValueError(bodies[0]), ok_message()])
    with pytest.raises(DailyQuotaExceeded):
        client.invoke_text([HumanMessage(content="hi")])
    assert len(chat.calls) == 1
    assert sleep.calls == []

    # (b) Per-minute 429 with retryDelay 7s → retried, wait >=7.0.
    # Pass a larger llm_max_wait_s so the 7s wait doesn't trigger DailyQuotaExceeded.
    body = '{"error":{"message":"Rate limit: PerMinute exceeded","retryDelay":"7s"}}'
    error = make_rate_limit_error(body=json.loads(body), message="PerMinute", headers={})
    client, chat, sleep = make_client(
        [error, error, ok_message()],
        make_settings(llm_max_wait_s=15.0),
    )
    result = client.invoke_text([HumanMessage(content="hi")])
    assert result is not None
    assert len(chat.calls) == 3  # two retries + final
    assert any(w >= 7.0 for w in sleep.calls), f"waits={sleep.calls}"


def test_transient_5xx_and_connection_errors_are_retried() -> None:
    script = [
        make_status_error(503),
        make_status_error(408),
        make_status_error(502),
        make_connection_error(),
        make_timeout_error(),
        ok_message(),
    ]
    client, chat, sleep = make_client(script)
    response = client.invoke_text([HumanMessage(content="hi")])
    assert response.text == "ok"
    assert len(chat.calls) == 6
    assert sleep.calls == [2.0, 4.0, 8.0, 16.0, 32.0]
    assert client.stats["api_retries"] == 5
    assert response.usage["api_retries"] == 5


def test_auth_errors_fail_fast() -> None:
    # 401 → key hint; 403 → access may be model-restricted, so point at LLM_MODEL;
    # 402 → hint about credits
    cases = ((401, "GEMINI_API_KEY"), (403, "LLM_MODEL"), (402, "billing"))
    for status, hint in cases:
        client, chat, sleep = make_client([make_status_error(status)])
        with pytest.raises(LLMAuthError) as excinfo:
            client.invoke_text([HumanMessage(content="hi")])
        assert hint in str(excinfo.value)
        assert len(chat.calls) == 1  # no retry on 4xx
        assert sleep.calls == []


def test_missing_model_raises_model_error() -> None:
    client, chat, sleep = make_client(
        [
            make_status_error(
                404,
                body={"error": {"message": "No endpoints found for gemini/does-not-exist:free"}},
            ),
            ok_message(),
        ]
    )
    with pytest.raises(LLMModelError) as excinfo:
        client.invoke_text([HumanMessage(content="hi")])
    assert "LLM_MODEL" in str(excinfo.value)
    assert len(chat.calls) == 1
    assert sleep.calls == []


def test_empty_response_retried_then_succeeds() -> None:
    client, chat, sleep = make_client(
        [AIMessage(content=""), AIMessage(content="   "), ok_message("finally")]
    )
    response = client.invoke_text([HumanMessage(content="hi")])
    assert response.text == "finally"
    assert len(chat.calls) == 3
    assert sleep.calls == [2.0, 4.0]
    assert client.stats["api_retries"] == 2

    # Malformed HTTP-200 bodies raised by langchain_openai itself (chat_models/
    # base.py): ValueError for an "error" field, TypeError for null "choices".
    malformed = ValueError({"error": {"message": "upstream error"}})
    null_choices = TypeError("Received response with null value for 'choices'.")
    client, chat, sleep = make_client([malformed, null_choices, ok_message("ok")])
    response = client.invoke_text([HumanMessage(content="hi")])
    assert response.text == "ok"
    assert len(chat.calls) == 3
    assert sleep.calls == [2.0, 4.0]
    assert client.stats["api_retries"] == 2


def test_retries_exhausted_raise_unavailable() -> None:
    settings = make_settings(llm_max_api_retries=2)
    client, chat, sleep = make_client(
        [make_status_error(503), make_status_error(503), make_status_error(503)],
        settings=settings,
    )
    with pytest.raises(LLMUnavailable) as excinfo:
        client.invoke_text([HumanMessage(content="hi")])
    assert len(chat.calls) == 3  # 1 initial + 2 retries
    assert sleep.calls == [2.0, 4.0]
    assert client.stats["api_retries"] == 2
    assert excinfo.value.__cause__ is not None  # chains the last openai error
    assert "limit 2 per call" in str(excinfo.value)

    # The limit is PER logical call: two consecutive calls each get a fresh
    # retry budget (a cumulative counter would fail the second call instantly).
    client2, chat2, sleep2 = make_client(
        [
            make_status_error(503),
            make_status_error(503),
            ok_message("first"),
            make_status_error(503),
            make_status_error(503),
            ok_message("second"),
        ],
        settings=settings,
    )
    assert client2.invoke_text([HumanMessage(content="one")]).text == "first"
    assert client2.invoke_text([HumanMessage(content="two")]).text == "second"
    assert len(chat2.calls) == 6
    assert sleep2.calls == [2.0, 4.0, 2.0, 4.0]
    assert client2.stats["api_retries"] == 4


def test_budget_reset_and_throttle() -> None:
    settings = make_settings(max_llm_calls_per_run=2, llm_min_interval_s=5.0)
    chat = FakeChatModel([ok_message()])
    clock = FakeClock()
    sleep = RecordingSleep(clock)
    client = LLMClient(chat, settings, sleep=sleep, clock=clock)

    client.invoke_text([HumanMessage(content="a")])
    client.invoke_text([HumanMessage(content="b")])
    assert client.stats["calls"] == 2
    with pytest.raises(LLMBudgetExceeded):
        client.invoke_text([HumanMessage(content="c")])  # budget enforced BEFORE the call
    assert len(chat.calls) == 2

    client.reset_budget()
    client.invoke_text([HumanMessage(content="d")])
    assert len(chat.calls) == 3

    # throttle: only 2s elapsed since the last call, min interval is 5s → sleep 3s
    clock.advance(2.0)
    response = client.invoke_text([HumanMessage(content="e")])
    assert sleep.calls[-1] == 3.0
    assert response.waited_s == 3.0
