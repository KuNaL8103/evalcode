"""Human review node (Task 9): interrupt-based pause for approve/reject/edit.

``human_review_node`` calls ``langgraph.types.interrupt`` with a JSON-serializable
payload. On resume, LangGraph re-executes the node from the start, so **no
side effects may occur before the interrupt call**.

``apply_human_decision`` is a pure function that translates the resume
response into a partial state update. It is exported for testing and for the
node to call after ``interrupt`` returns.
"""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from evalcode.state import AgentState, StepEvent, utc_now_iso

__all__ = ["apply_human_decision", "human_review_node"]


def _step(node: str, attempt: int, summary: dict[str, Any]) -> StepEvent:
    """One ``StepEvent`` for this node."""
    return {"node": node, "attempt": attempt, "ts": utc_now_iso(), "summary": summary}


def apply_human_decision(state: AgentState, response: Any) -> dict[str, Any]:
    """Translate a resume response into a partial state update.

    ``response`` must be a dict with a ``"decision"`` key whose value is one of:
    - ``"approve"``: accept the current code/tests. Returns ``human_decision="approve"``,
      ``human_feedback=None``, and a history event.
    - ``"reject"``: reject with mandatory non-empty ``"feedback"`` string. Returns
      ``human_decision="reject"``, ``human_feedback=feedback``, increments
      ``human_rounds``, and adds a history event. If the new ``human_rounds`` exceeds
      ``max_human_rounds``, also returns a ``failure_reason``.
    - ``"edit"``: replace the code with a mandatory non-empty ``"code"`` string.
      Returns ``human_decision="edit"``, ``human_feedback=None``, the new ``code``,
      and a history event. ``human_rounds`` is unchanged (it counts rejections only).

    Raises ``ValueError`` for any malformed response (not a dict, unknown decision,
    missing/blank required fields). This indicates a caller bug, not an LLM failure.
    """
    if not isinstance(response, dict):
        raise ValueError(f"Resume response must be a dict, got {type(response).__name__}")

    decision = response.get("decision")
    if decision not in ("approve", "reject", "edit"):
        raise ValueError(f"Unknown decision: {decision!r}. Must be 'approve', 'reject', or 'edit'")

    attempt = state.get("attempt", 0)
    human_rounds = state.get("human_rounds", 0)
    max_human_rounds = state.get("max_human_rounds", 2)

    if decision == "approve":
        event = _step(
            "human_review", attempt, {"decision": "approve", "human_rounds": human_rounds}
        )
        return {
            "human_decision": "approve",
            "human_feedback": None,
            "history": [event],
        }

    if decision == "reject":
        feedback = response.get("feedback")
        if not isinstance(feedback, str) or not feedback.strip():
            raise ValueError("Decision 'reject' requires a non-empty 'feedback' string")

        new_human_rounds = human_rounds + 1
        feedback_truncated = feedback.strip()[:200]
        event = _step(
            "human_review",
            attempt,
            {
                "decision": "reject",
                "human_rounds": new_human_rounds,
                "feedback": feedback_truncated,
            },
        )

        update: dict[str, Any] = {
            "human_decision": "reject",
            "human_feedback": feedback.strip(),
            "human_rounds": new_human_rounds,
            "history": [event],
        }

        if new_human_rounds > max_human_rounds:
            update["failure_reason"] = (
                f"Reviewer rejected the solution {new_human_rounds} time(s); "
                f"max human rounds ({max_human_rounds}) exceeded. "
                f"Last feedback: {feedback_truncated}"
            )
        return update

    # decision == "edit"
    code = response.get("code")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("Decision 'edit' requires a non-empty 'code' string")

    event = _step("human_review", attempt, {"decision": "edit", "human_rounds": human_rounds})
    return {
        "human_decision": "edit",
        "human_feedback": None,
        "code": code.strip(),
        "history": [event],
    }


def human_review_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node that pauses execution for human review.

    Builds a JSON-serializable payload and calls ``interrupt()`` FIRST, with no
    side effects before it (the node re-executes on resume). The payload contains:
    - task_id, attempt, human_round, max_human_rounds
    - code, tests, explanation
    - run_summary with category, tests_total, tests_failed, duration_s

    Returns the result of ``apply_human_decision(state, response)``.
    """
    run_result = state.get("run_result") or {}
    payload = {
        "task_id": state.get("task_id"),
        "attempt": state.get("attempt", 0),
        "human_round": state.get("human_rounds", 0),
        "max_human_rounds": state.get("max_human_rounds", 2),
        "code": state.get("code", ""),
        "tests": state.get("tests", ""),
        "explanation": state.get("explanation", ""),
        "run_summary": {
            "category": run_result.get("category"),
            "tests_total": run_result.get("tests_total", 0),
            "tests_failed": run_result.get("tests_failed", 0),
            "duration_s": run_result.get("duration_s", 0.0),
        },
    }

    # interrupt() MUST be the first side-effecting operation; LangGraph re-runs
    # this node on resume, so nothing before it may mutate state or perform I/O.
    response = interrupt(payload)
    return apply_human_decision(state, response)
