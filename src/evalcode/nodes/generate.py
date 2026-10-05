"""Generate node: the first LLM step of the loop (ARCHITECTURE §3.3).

``make_generate_node(llm, settings)`` returns a node ``(state) -> partial
state dict``. On success it writes code/tests/explanation, bumps
``attempt``, sets ``status="running"``, and records one history event with
a compact summary. An unparseable reply gets exactly ONE strict re-ask
(FORMAT_REMINDER); a second parse failure, or any ``LLMError`` from a
mandatory call, becomes a ``status="failed"`` update instead of a crash —
the graph's ``route_after_llm`` then sends the run to ``fail``.

The input state is never mutated.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from evalcode.config import Settings
from evalcode.errors import LLMError, ParseError
from evalcode.llm import TextLLM
from evalcode.parsing import parse_bundle
from evalcode.prompts import FORMAT_REMINDER, build_generate_messages
from evalcode.schemas import CodeBundle, extract_imports
from evalcode.state import AgentState, TokenUsage, merge_usage, utc_now_iso

__all__ = ["make_generate_node"]


def _reply_head(text: str) -> str:
    """ASCII-safe, whitespace-collapsed first 200 chars of a model reply."""
    snippet = text[:200]
    safe = "".join(ch if ord(ch) < 128 else "?" for ch in snippet)
    collapsed = re.sub(r"\s+", " ", safe).strip()
    return collapsed


def _step(attempt: int, summary: dict[str, Any]) -> dict[str, Any]:
    """One ``StepEvent`` for this node."""
    return {"node": "generate", "attempt": attempt, "ts": utc_now_iso(), "summary": summary}


def _failed(exc: LLMError, attempt: int, usage: TokenUsage) -> dict[str, Any]:
    """Failed update for a mandatory LLM failure (usage = successful calls only)."""
    return {
        "status": "failed",
        "failure_reason": str(exc),
        "history": [_step(attempt, {"error": type(exc).__name__})],
        "token_usage": usage,
    }


def make_generate_node(llm: TextLLM, settings: Settings) -> Callable[[AgentState], dict[str, Any]]:
    """Build the ``generate`` node around an injected ``TextLLM``."""

    def generate(state: AgentState) -> dict[str, Any]:
        provided_tests = state.get("provided_tests") or ""
        require_tests = not provided_tests.strip()
        attempt = state.get("attempt", 0) + 1

        messages = build_generate_messages(state, context_max_chars=settings.context_max_chars)
        usage: TokenUsage = {}
        reasks = 0
        bundle: CodeBundle | None = None
        parse_reason = ""

        try:
            response = llm.invoke_text(messages, purpose="generate")
        except LLMError as exc:
            return _failed(exc, attempt, usage)

        usage = merge_usage(usage, response.usage)
        try:
            bundle = parse_bundle(response.text, require_tests=require_tests)
        except ParseError as exc:
            # Exactly one strict re-ask; it counts against the client budget too.
            reasks = 1
            _reply_head(response.text)
            parse_reason = str(exc)
            reask = [
                *messages,
                AIMessage(content=response.text),
                HumanMessage(content=FORMAT_REMINDER),
            ]
            try:
                second = llm.invoke_text(reask, purpose="generate")
            except LLMError as exc:
                # The failed call yields no usage: keep only what succeeded.
                return _failed(exc, attempt, usage)
            usage = merge_usage(usage, second.usage)
            try:
                bundle = parse_bundle(second.text, require_tests=require_tests)
            except ParseError as exc2:
                return {
                    "status": "failed",
                    "failure_reason": (
                        "Generation failed: the model returned no parseable code even after "
                        "a strict format re-ask. Retry the run, or pick a stronger model via "
                        "LLM_MODEL."
                    ),
                    "history": [
                        _step(
                            attempt,
                            {
                                "error": "ParseError",
                                "reasks": reasks,
                                "reply_head": _reply_head(second.text),
                                "parse_reason": str(exc2),
                            },
                        )
                    ],
                    "token_usage": usage,
                }

        tests = provided_tests or bundle.tests
        doc_ids = [doc["id"] for doc in (state.get("retrieved_docs") or [])]
        summary = {
            "code_chars": len(bundle.code),
            "tests_chars": len(tests),
            "doc_ids": doc_ids,
            "docs_used": bundle.docs_used,
            "imports": extract_imports(bundle.code),
            "reasks": reasks,
            "reply_head": _reply_head(response.text) if reasks else "",
            "parse_reason": parse_reason if reasks else "",
        }
        return {
            "code": bundle.code,
            "tests": tests,
            "explanation": bundle.explanation,
            "attempt": attempt,
            "status": "running",
            "token_usage": usage,
            "history": [_step(attempt, summary)],
        }

    return generate
