"""Analyze-error node (Task 7): deterministic error analysis with optional LLM refinement.

``make_analyze_error_node(llm, settings)`` returns a node ``(state) -> partial
state dict``. It runs pure deterministic analysis first (via
``evalcode.analysis.analyze_run_result``), then optionally refines
``root_cause``/``fix_plan`` with one LLM call when ``settings.analyze_with_llm``
is true and an LLM is provided. The LLM call is best-effort: any failure falls
back to the deterministic result without crashing.

The input state is never mutated.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from evalcode.analysis import analyze_run_result
from evalcode.config import Settings
from evalcode.errors import LLMError
from evalcode.llm import TextLLM
from evalcode.prompts import build_analyze_messages
from evalcode.state import AgentState, merge_usage, utc_now_iso


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


def make_analyze_error_node(
    llm: TextLLM | None, settings: Settings
) -> Callable[[AgentState], dict[str, Any]]:
    """Build the ``analyze_error`` node around an optional injected ``TextLLM``."""

    def analyze_error(state: AgentState) -> dict[str, Any]:
        run_result = state.get("run_result")
        code = state.get("code", "") or ""
        tests = state.get("tests", "") or ""
        provided_tests = state.get("provided_tests") or ""
        attempt = state.get("attempt", 0)

        # Deterministic analysis (always runs)
        analysis = analyze_run_result(run_result, code, tests, provided_tests)

        # Prepare base result
        summary = {
            "category": analysis["category"],
            "fault": analysis["fault"],
            "needs_docs": analysis["needs_docs"],
            "root_cause": analysis["root_cause"][:200],
            "suspect_symbols": analysis["suspect_symbols"],
            "llm": False,
        }
        result: dict[str, Any] = {
            "error_analysis": analysis,
            "history": [_step("analyze_error", attempt, summary)],
        }

        # Optional LLM refinement
        if (
            settings.analyze_with_llm
            and llm is not None
            and not (run_result and run_result.get("passed"))
        ):
            messages = build_analyze_messages(state)
            try:
                response = llm.invoke_text(messages, purpose="analyze")
            except LLMError:
                # LLM failed: keep deterministic result, no token_usage
                return result

            # Merge usage
            result["token_usage"] = merge_usage(result.get("token_usage"), response.usage)

            # Lenient parse of <root_cause> and <fix_plan>
            from evalcode.parsing import parse_tagged

            text = response.text
            if not text:
                return result

            llm_root_cause = parse_tagged(text, "root_cause")
            llm_fix_plan = parse_tagged(text, "fix_plan")

            updated = False
            if llm_root_cause and llm_root_cause.strip():
                analysis["root_cause"] = llm_root_cause.strip()[:400]
                updated = True
            if llm_fix_plan and llm_fix_plan.strip():
                analysis["fix_plan"] = llm_fix_plan.strip()[:400]
                updated = True

            if updated:
                # Update the history summary with LLM-refined values
                summary["llm"] = True
                summary["root_cause"] = analysis["root_cause"][:200]
                result["history"] = [_step("analyze_error", attempt, summary)]

        return result

    return analyze_error
