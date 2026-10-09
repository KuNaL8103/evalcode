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
from langchain_core.messages import BaseMessage
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from evalcode.llm import LLMResponse
from evalcode.rag.types import RetrievedDoc
from evalcode.state import TokenUsage

# The OpenAI chat-completions endpoint, used only for stub httpx2.Request objects.
_CHAT_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"


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


class ScriptedLLM:
    """``TextLLM`` fake for node tests: scripted replies, no network.

    Each script item is either the response text (``str``) or an exception
    instance to raise. Every call's messages list is recorded in ``calls``
    (with the parallel ``purposes`` list). When the script is exhausted the
    LAST item repeats, so short scripts can stand in for "always fails".
    """

    def __init__(self, script: list[Any]) -> None:
        if not script:
            raise ValueError("ScriptedLLM needs a non-empty script")
        self._script = list(script)
        self._index = 0
        self.calls: list[list[Any]] = []
        self.purposes: list[str] = []

    def invoke_text(self, messages: list[BaseMessage], *, purpose: str = "generate") -> LLMResponse:
        self.calls.append(list(messages))
        self.purposes.append(purpose)
        item = self._script[min(self._index, len(self._script) - 1)]
        self._index += 1
        if isinstance(item, Exception):
            raise item
        usage: TokenUsage = {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
            "llm_calls": 1,
            "api_retries": 0,
            "wait_s": 0.0,
        }
        return LLMResponse(text=item, usage=usage, model="scripted-model", waited_s=0.0)


def bundle_text(
    code: str,
    tests: str | None = None,
    explanation: str = "test explanation",
    docs_used: list[str] | None = None,
) -> str:
    """Render the tagged protocol format for scripted responses.

    ``tests=None`` omits the ``<tests>`` section (code-only reply, as when
    the task carries provided tests).
    """
    parts = [f"<explanation>\n{explanation}\n</explanation>", f"<code>\n{code}\n</code>"]
    if tests is not None:
        parts.append(f"<tests>\n{tests}\n</tests>")
    if docs_used:
        parts.append(f"<docs_used>\n{', '.join(docs_used)}\n</docs_used>")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Retriever fake (Task 10)
# --------------------------------------------------------------------------- #


def make_doc(
    id: str,
    text: str = "doc text",
    score: float = 0.9,
    library: str = "json",
    qualname: str = "json.loads",
    import_path: str = "json",
) -> RetrievedDoc:
    """Create a RetrievedDoc for testing."""
    return RetrievedDoc(
        id=id,
        text=text,
        score=score,
        library=library,
        qualname=qualname,
        import_path=import_path,
    )


class FakeRetriever:
    """Fake retriever for unit tests.

    Args:
        docs_by_query: Mapping from query string to list of docs. When a query
            is not found, falls back to ``default``.
        default: Default docs to return for any query not in ``docs_by_query``.
        raises: If set, this exception is raised on every ``retrieve`` call.
    """

    def __init__(
        self,
        docs_by_query: dict[str, list[RetrievedDoc]] | None = None,
        default: list[RetrievedDoc] | None = None,
        raises: Exception | None = None,
    ) -> None:
        self.docs_by_query = docs_by_query or {}
        self.default = default or []
        self.raises = raises
        self.calls: list[list[str]] = []

    def retrieve(
        self, queries: list[str], k: int | None = None, library: str | None = None
    ) -> list[RetrievedDoc]:
        self.calls.append(list(queries))
        if self.raises:
            raise self.raises

        # Merge docs for all queries, keeping max score per id
        best: dict[str, float] = {}
        by_id: dict[str, RetrievedDoc] = {}
        for query in queries:
            docs = self.docs_by_query.get(query, self.default)
            for doc in docs:
                prev = best.get(doc["id"])
                if prev is None or doc["score"] > prev:
                    best[doc["id"]] = doc["score"]
                    by_id[doc["id"]] = doc

        # Sort by score descending
        result = list(by_id.values())
        result.sort(key=lambda d: d["score"], reverse=True)
        if k is not None:
            result = result[:k]
        return result
