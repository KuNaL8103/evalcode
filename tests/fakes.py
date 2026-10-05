"""In-memory fakes for unit tests: no network, no real LLM, no real key.

``FakeChatModel`` is a minimal duck-typed stand-in for
``langchain_openai.ChatOpenAI`` (``LLMClient`` only calls ``.invoke``),
driven by a script of items. The openai exception helpers build real
``openai`` exception objects with stub ``httpx2`` responses, so ``LLMClient``
sees the exact exception types the OpenAI SDK would raise.
"""

from __future__ import annotations

from typing import Any

import httpx2
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

# The OpenAI chat-completions endpoint, used only for stub httpx2.Request objects.
_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


class FakeChatModel:
    """Scripted chat model: ``invoke`` pops items from ``script`` in order.

    Each item is either a message-like object (e.g. an ``AIMessage`` with
    optional ``usage_metadata``/``response_metadata``) or an exception
    instance to raise. When the script is exhausted the LAST item repeats,
    so short scripts can stand in for "always fails" scenarios. Every
    call's messages are recorded in ``calls``.
    """

    def __init__(self, script: list[Any]) -> None:
        if not script:
            raise ValueError("FakeChatModel needs a non-empty script")
        self._script = list(script)
        self.calls: list[list[Any]] = []
        self._index = 0

    def invoke(self, messages: Any, **_kwargs: Any) -> Any:
        self.calls.append(list(messages))
        item = self._script[min(self._index, len(self._script) - 1)]
        self._index += 1
        if isinstance(item, Exception):
            raise item
        return item


def _request() -> httpx2.Request:
    return httpx2.Request("POST", _CHAT_URL)


def _response(
    status_code: int, headers: dict[str, str] | None = None, text: str | None = None
) -> httpx2.Response:
    return httpx2.Response(status_code, request=_request(), headers=headers, text=text)


def make_status_error(
    status: int,
    body: Any = None,
    message: str | None = None,
    headers: dict[str, str] | None = None,
) -> APIStatusError:
    """Build an ``openai.APIStatusError`` with a stub ``httpx2.Response``."""
    return APIStatusError(
        message or f"HTTP {status} error",
        response=_response(status, headers, str(body) if body is not None else None),
        body=body,
    )


def make_rate_limit_error(
    headers: dict[str, str] | None = None,
    body: Any = None,
    message: str = "Rate limit exceeded",
) -> RateLimitError:
    """Build an ``openai.RateLimitError`` (429) with optional rate-limit headers."""
    return RateLimitError(
        message,
        response=_response(429, headers, str(body) if body is not None else None),
        body=body,
    )


def make_connection_error() -> APIConnectionError:
    """Build an ``openai.APIConnectionError`` (network failure)."""
    return APIConnectionError(request=_request())


def make_timeout_error() -> APITimeoutError:
    """Build an ``openai.APITimeoutError`` (request timed out)."""
    return APITimeoutError(request=_request())
