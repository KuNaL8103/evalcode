"""Terminal nodes: finalize and fail (Task 8).

These nodes produce the final state updates when a run completes (either
successfully or after exhausting retries). They return only a single new
history event each (the history reducer is operator.add).
"""

from __future__ import annotations

from typing import Any

from evalcode.state import AgentState, StepEvent, utc_now_iso

__all__ = ["fail_node", "finalize_node"]


def _step(node: str, attempt: int, summary: dict[str, Any]) -> StepEvent:
    """One StepEvent for this node."""
    return {"node": node, "attempt": attempt, "ts": utc_now_iso(), "summary": summary}


def finalize_node(state: AgentState) -> dict[str, Any]:
    """Final update for an approved run."""
    attempt = state.get("attempt", 0)
    event = _step("finalize", attempt, {"final_code_chars": len(state.get("code", "") or "")})
    return {
        "final_code": state["code"],
        "status": "approved",
        "history": [event],
    }


def fail_node(state: AgentState) -> dict[str, Any]:
    """Final update for a failed run.

    - If failure_reason is already set (e.g., from an LLM error), preserve it.
    - Otherwise, compose a message from retries_used and the last run_result.
    - If no run_result exists, use a generic message.
    """
    attempt = state.get("attempt", 0)
    retries_used = state.get("retries_used", 0)
    failure_reason = state.get("failure_reason") or ""
    run_result = state.get("run_result")

    if failure_reason.strip():
        reason = failure_reason
    elif run_result:
        category = run_result.get("category", "unknown")
        reason = (
            f"Tests still failing after {retries_used} automatic revision(s) "
            f"(attempt {attempt}); last category: {category}."
        )
    else:
        reason = "Run ended without a test result."

    event = _step("fail", attempt, {"reason": reason[:200]})
    return {
        "status": "failed",
        "failure_reason": reason,
        "history": [event],
    }
