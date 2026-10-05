"""OpenRouter LLM access layer (ARCHITECTURE §5).

The only provider is OpenRouter (free models). ``get_chat_model`` builds a
``ChatOpenAI`` with SDK retries disabled (``max_retries=0``); ``LLMClient``
owns the whole retry policy: throttle, per-run call budget, exponential
backoff + jitter, ``Retry-After``/``X-RateLimit-Reset`` handling, daily-quota
fail-fast, an actionable error taxonomy, and usage extraction.

Sleep/clock/RNG are injectable so tests run instantly. Message contents are
never logged at INFO; retries/waits log at WARNING without secrets.
"""

from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Protocol, runtime_checkable

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APIStatusError, APITimeoutError

from evalcode.config import Settings, get_settings
from evalcode.errors import (
    DailyQuotaExceeded,
    EmptyResponseError,
    LLMAuthError,
    LLMBudgetExceeded,
    LLMModelError,
    LLMRequestError,
    LLMUnavailable,
)
from evalcode.state import TokenUsage

logger = logging.getLogger(__name__)

__all__ = [
    "LLMClient",
    "LLMResponse",
    "TextLLM",
    "build_llm_client",
    "extract_usage",
    "get_chat_model",
    "strip_reasoning",
]

# Tag-literal safety (CLAUDE.md): think tags are built from parts, never typed
# as literals, and the stripping regexes are compiled from these constants.
THINK_OPEN = "<" + "think" + ">"
THINK_CLOSE = "</" + "think" + ">"
_THINK_BLOCK_RE = re.compile(THINK_OPEN + r"\s*.*?" + THINK_CLOSE, re.DOTALL)
_THINK_TRAILING_RE = re.compile(THINK_OPEN + r"\s*.*\Z", re.DOTALL)

# Case-insensitive markers that a 429 refers to the *daily* free quota.
_DAILY_LIMIT_KEYWORDS = ("per-day", "per day", "daily", "free-models-per-day")


def get_chat_model(settings: Settings) -> ChatOpenAI:
    """Build the OpenRouter chat model.

    ``max_retries=0`` so the OpenAI SDK never retries on its own —
    ``LLMClient`` owns retry policy (backoff, headers, budget). Raises
    ``ConfigError`` (via ``require_api_key``) when the key is missing.
    """
    api_key = settings.require_api_key()
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.openrouter_base_url,
        api_key=api_key,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_s,
        max_retries=0,
        default_headers={"X-Title": "evalcode"},
    )


@dataclass
class LLMResponse:
    """One successful logical LLM call, post-processed."""

    text: str
    usage: TokenUsage
    model: str
    waited_s: float = 0.0


@runtime_checkable
class TextLLM(Protocol):
    """The LLM interface nodes program against (``ScriptedLLM`` in tests)."""

    def invoke_text(
        self, messages: list[BaseMessage], *, purpose: str = "generate"
    ) -> LLMResponse: ...


def strip_reasoning(text: str) -> str:
    """Remove reasoning-model ``<think>…</think>`` blocks (closed or unterminated)."""
    if not text:
        return text
    text = _THINK_BLOCK_RE.sub("", text)
    text = _THINK_TRAILING_RE.sub("", text)
    return text.strip()


def _normalize_content(content: Any) -> str:
    """Flatten message content (str, or list of str/dict content blocks) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(getattr(block, "text", None), str):
                parts.append(block.text)
        return "".join(parts)
    return str(content)


def extract_usage(ai_message: Any, prompt_text: str, completion_text: str) -> TokenUsage:
    """Build the per-call usage dict.

    Uses the provider's ``usage_metadata`` when present; otherwise estimates
    tokens as ceil(chars / 4) and marks the call ``estimated_calls=1``.
    """
    usage: TokenUsage = {"llm_calls": 1}
    metadata = getattr(ai_message, "usage_metadata", None)
    if metadata:
        input_tokens = int(metadata.get("input_tokens") or 0)
        output_tokens = int(metadata.get("output_tokens") or 0)
        usage["input_tokens"] = input_tokens
        usage["output_tokens"] = output_tokens
        usage["total_tokens"] = int(metadata.get("total_tokens") or input_tokens + output_tokens)
        usage["estimated_calls"] = 0
    else:
        input_tokens = -(-len(prompt_text) // 4)
        output_tokens = -(-len(completion_text) // 4)
        usage["input_tokens"] = input_tokens
        usage["output_tokens"] = output_tokens
        usage["total_tokens"] = input_tokens + output_tokens
        usage["estimated_calls"] = 1
    return usage


def _messages_to_text(messages: list[BaseMessage]) -> str:
    """Concatenated message text, used only for token estimation."""
    parts = [_normalize_content(getattr(m, "content", None)) for m in messages]
    return "\n".join(p for p in parts if p)


def _error_text(exc: APIStatusError) -> str:
    """Best-effort error text from the openai exception (no secrets in it)."""
    body = exc.body
    if isinstance(body, dict):
        body = str(body.get("error") or body)
    return f"{exc.message or ''} {body or ''}".strip()


class LLMClient:
    """OpenRouter client owning throttle, budget, backoff, and error taxonomy.

    Sleep, clock, and RNG are injectable so tests run instantly and can assert
    exact sleep durations.
    """

    def __init__(
        self,
        chat: ChatOpenAI,
        settings: Settings,
        *,
        sleep: Any = time.sleep,
        clock: Any = time.monotonic,
        rng: Any = random.random,
    ) -> None:
        self._chat = chat
        self._settings = settings
        self._sleep = sleep
        self._clock = clock
        self._rng = rng
        self._calls = 0
        self._api_retries = 0
        self._wait_s = 0.0
        self._last_call: float | None = None

    @property
    def stats(self) -> dict[str, Any]:
        """Cumulative stats for this client: logical calls, retries, wait time."""
        return {"calls": self._calls, "api_retries": self._api_retries, "wait_s": self._wait_s}

    def reset_budget(self) -> None:
        """Reset the per-run logical call budget (call at the start of each run)."""
        self._calls = 0

    def invoke_text(self, messages: list[BaseMessage], *, purpose: str = "generate") -> LLMResponse:
        """One logical LLM call: budget → throttle → retry loop with backoff."""
        settings = self._settings
        if self._calls >= settings.max_llm_calls_per_run:
            raise LLMBudgetExceeded(
                f"Per-run LLM call budget exhausted ({settings.max_llm_calls_per_run} "
                "logical calls made). Refusing to call OpenRouter; raise "
                "MAX_LLM_CALLS_PER_RUN only if the task really needs more."
            )
        self._calls += 1

        waited_s = self._throttle()
        prompt_text = _messages_to_text(messages)

        attempt = 0
        while True:
            last_error: Exception | None = None
            try:
                ai_message = self._chat.invoke(messages)
                text = strip_reasoning(_normalize_content(getattr(ai_message, "content", None)))
                if not text:
                    raise EmptyResponseError()
                usage = extract_usage(ai_message, prompt_text, text)
                usage["api_retries"] = attempt
                usage["wait_s"] = waited_s
                model_name = (getattr(ai_message, "response_metadata", None) or {}).get(
                    "model_name"
                )
                return LLMResponse(
                    text=text,
                    usage=usage,
                    model=model_name or settings.llm_model,
                    waited_s=waited_s,
                )
            except APIStatusError as exc:
                last_error = exc
                delay = self._handle_status_error(exc, attempt, purpose)
            except APIConnectionError as exc:  # incl. APITimeoutError
                last_error = exc
                reason = "timeout" if isinstance(exc, APITimeoutError) else "connection error"
                delay = self._retry_backoff(attempt, purpose, reason, exc)
            except EmptyResponseError as exc:
                last_error = exc
                delay = self._retry_backoff(attempt, purpose, "empty response", exc)

            attempt += 1
            self._api_retries += 1
            waited_s += delay
            self._wait_s += delay
            logger.warning(
                "OpenRouter call failed (%s); retry %d/%d in %.1fs (purpose=%s)",
                _short_reason(last_error),
                self._api_retries,
                settings.llm_max_api_retries,
                delay,
                purpose,
            )
            self._sleep(delay)

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #

    def _throttle(self) -> float:
        """Sleep so that at least ``llm_min_interval_s`` elapsed since the last call."""
        settings = self._settings
        waited = 0.0
        now = self._clock()
        if self._last_call is not None:
            gap = now - self._last_call
            if gap < settings.llm_min_interval_s:
                waited = settings.llm_min_interval_s - gap
                self._wait_s += waited
                self._sleep(waited)
        self._last_call = self._clock()
        return waited

    def _handle_status_error(self, exc: APIStatusError, attempt: int, purpose: str) -> float:
        """Map an HTTP status to a fatal LLMError, or return the backoff delay to wait."""
        settings = self._settings
        status = exc.status_code
        text = _error_text(exc)
        if status in (401, 403):
            raise LLMAuthError(
                f"OpenRouter authentication failed (HTTP {status}). Set a valid "
                "OPENROUTER_API_KEY (https://openrouter.ai/keys) in .env or the environment."
            ) from exc
        if status == 402:
            raise LLMAuthError(
                f"OpenRouter returned 402 Payment Required. The model '{settings.llm_model}' "
                "may not be free — pick a ':free' model for LLM_MODEL or add credits at "
                "https://openrouter.ai/credits."
            ) from exc
        if status == 404:
            raise LLMModelError(
                f"OpenRouter could not find model '{settings.llm_model}' (HTTP 404). "
                "Check LLM_MODEL; free models change — see https://openrouter.ai/models."
            ) from exc
        if 400 <= status < 500 and status not in (408, 429):
            raise LLMRequestError(
                f"OpenRouter rejected the request with HTTP {status}: {text[:200]}"
            ) from exc
        # 408 / 429 / 5xx: retryable
        reason = f"HTTP {status}"
        required_wait = 0.0
        if status == 429:
            if any(kw in text.lower() for kw in _DAILY_LIMIT_KEYWORDS):
                raise DailyQuotaExceeded(
                    f"OpenRouter reports the daily free quota is exhausted ({text[:160]}). "
                    "Wait for the quota to reset (or add credits) and try again later."
                ) from exc
            required_wait = self._header_wait(exc)
            if required_wait > settings.llm_max_wait_s:
                raise DailyQuotaExceeded(
                    f"OpenRouter asked us to wait {required_wait:.0f}s before retrying, which "
                    f"exceeds LLM_MAX_WAIT_S ({settings.llm_max_wait_s:.0f}s) — the free quota "
                    "is likely exhausted for today. Wait for the quota to reset (or add credits)."
                ) from exc
            reason = "rate limited (HTTP 429)"
        return self._retry_backoff(attempt, purpose, reason, exc, minimum_wait=required_wait)

    def _retry_backoff(
        self,
        attempt: int,
        purpose: str,
        reason: str,
        cause: Exception,
        *,
        minimum_wait: float = 0.0,
    ) -> float:
        """Exponential backoff + jitter, capped; raises when retries are exhausted."""
        settings = self._settings
        if self._api_retries >= settings.llm_max_api_retries:
            raise LLMUnavailable(
                f"OpenRouter request failed after {self._api_retries} retries "
                f"(last: {reason}). Model '{settings.llm_model}' may be temporarily "
                "unavailable — retry the run later."
            ) from cause
        base_delay = min(settings.llm_backoff_max_s, settings.llm_backoff_base_s * (2**attempt))
        delay = base_delay * self._rng()
        return max(delay, minimum_wait)

    def _header_wait(self, exc: APIStatusError) -> float:
        """Seconds to wait per Retry-After / X-RateLimit-Reset headers, if present."""
        response = exc.response
        headers = getattr(response, "headers", None) or {}
        required = 0.0
        retry_after = _get_header(headers, "retry-after")
        if retry_after:
            try:
                required = max(required, float(retry_after))
            except ValueError:
                try:  # HTTP-date form
                    when = parsedate_to_datetime(retry_after)
                    required = max(required, (when - datetime.now(UTC)).total_seconds())
                except (TypeError, ValueError):
                    pass
        reset = _get_header(headers, "x-ratelimit-reset")
        if reset:
            try:
                value = float(reset)
            except ValueError:
                value = None
            if value is not None:
                epoch_s = value / 1000.0 if value >= 1e10 else value  # ms vs s
                required = max(required, epoch_s - time.time())
        return max(required, 0.0)


def _get_header(headers: Any, name: str) -> str | None:
    """Case-insensitive-ish header lookup (stub headers in tests are lowercase)."""
    for candidate in (name, name.title(), name.upper()):
        try:
            value = headers.get(candidate)
        except AttributeError:
            return None
        if value is not None:
            return str(value)
    return None


def _short_reason(exc: Exception) -> str:
    """Short, secret-free description of the last error for log lines."""
    if isinstance(exc, APIStatusError):
        return f"HTTP {exc.status_code}"
    if isinstance(exc, EmptyResponseError):
        return "empty response"
    return type(exc).__name__


def build_llm_client(settings: Settings | None = None) -> LLMClient:
    """Convenience factory: real ``ChatOpenAI`` + settings into an ``LLMClient``."""
    if settings is None:
        settings = get_settings()
    return LLMClient(get_chat_model(settings), settings)
