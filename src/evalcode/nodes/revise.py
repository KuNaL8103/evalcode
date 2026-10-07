"""Revise node (Task 7): failure-aware code revision with optional human feedback.

``make_revise_node(llm, settings)`` returns a node ``(state) -> partial
state dict``. It builds a detailed prompt from the failure context and calls
the LLM once to produce a corrected solution. On parse failure it retries
exactly once with FORMAT_REMINDER (same pattern as generate.py).

A revision is "human-driven" iff ``human_feedback`` is non-empty. When
human-driven, the attempt counter increments but ``retries_used`` does NOT
increment (human feedback is not an automatic retry). The human_review node
(Task 9) MUST always set non-empty ``human_feedback`` on reject — this is
a contract assumption documented here.

The input state is never mutated.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from evalcode.config import Settings
from evalcode.errors import LLMError, ParseError
from evalcode.llm import TextLLM
from evalcode.parsing import parse_bundle
from evalcode.prompts import FORMAT_REMINDER, build_revise_messages
from evalcode.state import AgentState, TokenUsage, merge_usage, utc_now_iso


def _reply_head(text: str) -> str:
    """ASCII-safe, whitespace-collapsed first 200 chars of a model reply."""
    import re

    snippet = text[:200]
    safe = "".join(ch if ord(ch) < 128 else "?" for ch in snippet)
    collapsed = re.sub(r"\s+", " ", safe).strip()
    return collapsed


def _step(node: str, attempt: int, summary: dict[str, Any]) -> dict[str, Any]:
    """One ``StepEvent`` for this node."""
    return {"node": node, "attempt": attempt, "ts": utc_now_iso(), "summary": summary}


def make_revise_node(llm: TextLLM, settings: Settings) -> Callable[[AgentState], dict[str, Any]]:
    """Build the ``revise`` node around an injected ``TextLLM``."""

    def revise(state: AgentState) -> dict[str, Any]:
        human_feedback = state.get("human_feedback") or ""
        human_driven = bool(human_feedback.strip())

        attempt = state.get("attempt", 0) + 1
        retries_used = state.get("retries_used", 0) + (0 if human_driven else 1)

        provided_tests = state.get("provided_tests") or ""
        # When provided_tests is set, tests are fixed ground truth -> require them.
        # When provided_tests is empty, missing <tests> in reply is OK -> keep state["tests"].
        require_tests = bool(provided_tests.strip())

        messages = build_revise_messages(state, context_max_chars=settings.context_max_chars)
        usage: TokenUsage = {}
        reasks = 0
        bundle = None

        # First LLM call
        try:
            response = llm.invoke_text(messages, purpose="revise")
        except LLMError as exc:
            return _failed(exc, attempt, usage, reasks, human_driven)

        usage = merge_usage(usage, response.usage)
        try:
            bundle = parse_bundle(response.text, require_tests=require_tests)
        except ParseError:
            # Exactly one strict re-ask (only when require_tests=True)
            reasks = 1
            _reply_head(response.text)
            reask = [
                *messages,
                {"role": "assistant", "content": response.text},
                {"role": "user", "content": FORMAT_REMINDER},
            ]
            try:
                second = llm.invoke_text(reask, purpose="revise")
            except LLMError as exc2:
                return _failed(exc2, attempt, usage, reasks, human_driven)
            usage = merge_usage(usage, second.usage)
            try:
                bundle = parse_bundle(second.text, require_tests=require_tests)
            except ParseError as exc2:
                return _double_parse_failure(
                    attempt, usage, reasks, human_driven, second.text, str(exc2)
                )

        # Determine tests: 1) provided_tests (verbatim), 2) bundle.tests, 3) state["tests"]
        if provided_tests.strip():
            tests_result = provided_tests
        elif bundle and bundle.tests.strip():
            tests_result = bundle.tests
        else:
            tests_result = state.get("tests", "") or ""

        doc_ids = [doc["id"] for doc in (state.get("retrieved_docs") or [])]

        summary = {
            "code_chars": len(bundle.code) if bundle else 0,
            "tests_chars": len(tests_result),
            "human_driven": human_driven,
            "doc_ids": doc_ids,
            "docs_used": bundle.docs_used if bundle else [],
            "reasks": reasks,
            "retries_used": retries_used,
        }

        return {
            "code": bundle.code if bundle else "",
            "tests": tests_result,
            "explanation": bundle.explanation if bundle else "",
            "attempt": attempt,
            "retries_used": retries_used,
            "human_feedback": None,
            "status": "running",
            "token_usage": usage,
            "history": [_step("revise", attempt, summary)],
        }

    return revise


def _failed(
    exc: LLMError,
    attempt: int,
    usage: TokenUsage,
    reasks: int,
    human_driven: bool,
) -> dict[str, Any]:
    """Failed update for a mandatory LLM failure."""
    return {
        "status": "failed",
        "failure_reason": str(exc),
        "history": [
            _step(
                "revise",
                attempt,
                {"error": type(exc).__name__, "reasks": reasks, "human_driven": human_driven},
            )
        ],
        "token_usage": usage,
    }


def _double_parse_failure(
    attempt: int,
    usage: TokenUsage,
    reasks: int,
    human_driven: bool,
    reply_head: str,
    parse_reason: str,
) -> dict[str, Any]:
    """Failed update after two parse failures."""
    return {
        "status": "failed",
        "failure_reason": (
            "Revision failed: the model returned no parseable code even after a "
            "strict format re-ask. Retry the run, or pick a stronger model via "
            "LLM_MODEL."
        ),
        "history": [
            _step(
                "revise",
                attempt,
                {
                    "error": "ParseError",
                    "reasks": reasks,
                    "human_driven": human_driven,
                    "reply_head": _reply_head(reply_head),
                    "parse_reason": parse_reason,
                },
            )
        ],
        "token_usage": usage,
    }
