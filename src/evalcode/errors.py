"""Shared exception types for evalcode.

LLM error messages are deliberately actionable (point at the env var, the
model catalog, or credits) and must never embed key material.
"""

from __future__ import annotations


class EvalcodeError(Exception):
    """Base class for all evalcode errors."""


class ConfigError(EvalcodeError):
    """Raised when configuration is missing or invalid."""


class LLMError(EvalcodeError):
    """Base class for all LLM (OpenRouter) access errors.

    Subclasses carry an actionable default message; raise with a more
    specific ``message`` when you have it.
    """

    _DEFAULT_MESSAGE = "OpenRouter LLM request failed."

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self._DEFAULT_MESSAGE)


class LLMAuthError(LLMError):
    """Authentication/authorization failure (401/403) or payment required (402)."""

    _DEFAULT_MESSAGE = (
        "OpenRouter authentication failed: set a valid OPENROUTER_API_KEY "
        "(get one at https://openrouter.ai/keys; put it in .env, never in code)."
    )


class LLMModelError(LLMError):
    """The configured model was not found on OpenRouter (404)."""

    _DEFAULT_MESSAGE = (
        "OpenRouter model not found. Check LLM_MODEL; free models change — "
        "see https://openrouter.ai/models for current free slugs."
    )


class LLMRequestError(LLMError):
    """OpenRouter rejected the request as a client error (other 4xx)."""

    _DEFAULT_MESSAGE = (
        "OpenRouter rejected the request (client error). Check the request "
        "parameters and LLM settings, then try again."
    )


class DailyQuotaExceeded(LLMError):
    """The free daily quota is exhausted; stop immediately (do not retry)."""

    _DEFAULT_MESSAGE = (
        "OpenRouter daily free quota exhausted. Wait for the quota to reset "
        "(or add credits at https://openrouter.ai/credits) and try again later."
    )


class LLMUnavailable(LLMError):
    """Transient failures persisted until the API retry budget was exhausted."""

    _DEFAULT_MESSAGE = "OpenRouter is temporarily unavailable. Retry the run later."


class LLMBudgetExceeded(LLMError):
    """The per-run logical LLM call budget (MAX_LLM_CALLS_PER_RUN) was reached."""

    _DEFAULT_MESSAGE = (
        "Per-run LLM call budget (MAX_LLM_CALLS_PER_RUN) exhausted; stopping "
        "to protect the free quota."
    )


class EmptyResponseError(LLMError):
    """Internal, retryable: the provider returned an empty/malformed body.

    Raised and caught only inside ``LLMClient`` — it should never reach a node.
    """

    _DEFAULT_MESSAGE = "OpenRouter returned an empty response body."
