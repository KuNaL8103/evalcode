"""Streamlit human-review session flow + dashboard smoke tests.

The flow tests exercise the same helpers the buttons call, with an injected
resume function so no LLM, network, or real checkpoint is needed.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app import streamlit_app as ui
from evalcode.core.config import Settings


def _run_result() -> dict[str, Any]:
    return {
        "passed": True,
        "category": "pass",
        "exit_code": 0,
        "timed_out": False,
        "duration_s": 0.2,
        "tests_total": 1,
        "tests_failed": 0,
        "failures": [],
        "stdout": "1 passed",
        "stderr": "",
    }


def _paused_state(*, task_id: str = "task-ui", attempt: int = 1) -> dict[str, Any]:
    payload = {
        "task_id": task_id,
        "attempt": attempt,
        "human_round": attempt - 1,
        "max_human_rounds": 2,
        "code": "def add(a, b):\n    return a + b\n",
        "tests": "from solution import add\n\ndef test_add(): assert add(1, 2) == 3\n",
        "explanation": "Implemented add.",
        "run_summary": {"category": "pass", "tests_total": 1, "tests_failed": 0},
    }
    return {
        "task_id": task_id,
        "task": "add two numbers",
        "status": "running",
        "attempt": attempt,
        "human_rounds": attempt - 1,
        "max_human_rounds": 2,
        "code": payload["code"],
        "tests": payload["tests"],
        "explanation": payload["explanation"],
        "run_result": _run_result(),
        "retrieved_docs": [],
        "retrieval_queries": ["add two numbers"],
        "token_usage": {},
        "__interrupt__": (SimpleNamespace(value=payload),),
    }


def test_make_run_record_normalizes_paused_state() -> None:
    record = ui.make_run_record(
        _paused_state(),
        elapsed_s=1.5,
        backend="subprocess",
        use_rag=True,
        route="Dense-only",
        task="add two numbers",
        auto_approve=False,
    )

    assert record["task_id"] == "task-ui"
    assert record["paused"] is True
    assert record["state"]["status"] == "awaiting_review"
    assert "__interrupt__" not in record["state"]
    assert record["pending_review"]["attempt"] == 1


def test_review_response_contract() -> None:
    assert ui.review_response("approve", "") == {"decision": "approve"}
    assert ui.review_response("revision", "add hints") == {
        "decision": "reject",
        "feedback": "add hints",
    }
    assert ui.review_response("reject", "") == {
        "decision": "reject_final",
        "feedback": "Rejected by reviewer.",
    }
    with pytest.raises(ValueError, match="feedback is required"):
        ui.review_response("revision", "   ")
    with pytest.raises(ValueError, match="Unknown review action"):
        ui.review_response("nope", "")


def _apply_with_fake(
    decision: str,
    feedback: str,
    new_state: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    record = ui.make_run_record(
        _paused_state(),
        elapsed_s=1.0,
        backend="subprocess",
        use_rag=True,
        route="Dense-only",
        task="add two numbers",
        auto_approve=False,
    )
    calls: list[dict[str, Any]] = []

    def fake_resume(
        task_id: str,
        response: dict[str, Any],
        settings: Settings,
        *,
        use_rag: bool,
    ) -> tuple[dict[str, Any], float]:
        calls.append(
            {
                "task_id": task_id,
                "response": response,
                "use_rag": use_rag,
            }
        )
        return new_state, 0.5

    updated = ui.perform_review(
        record,
        decision,
        feedback,
        Settings(),
        resume_fn=fake_resume,
    )
    return updated, calls


def test_approve_updates_session_to_approved() -> None:
    approved = {
        **_paused_state(),
        "status": "approved",
        "final_code": "def add(a, b):\n    return a + b\n",
        "human_decision": "approve",
    }
    approved.pop("__interrupt__", None)

    updated, calls = _apply_with_fake("approve", "", approved)

    assert calls == [{"task_id": "task-ui", "response": {"decision": "approve"}, "use_rag": True}]
    assert updated["paused"] is False
    assert updated["state"]["status"] == "approved"
    assert updated["task_id"] == "task-ui"


def test_request_revision_keeps_new_paused_attempt() -> None:
    revised = _paused_state(attempt=2)

    updated, calls = _apply_with_fake("revision", "add type hints", revised)

    assert calls[0]["response"] == {"decision": "reject", "feedback": "add type hints"}
    assert updated["paused"] is True
    assert updated["state"]["status"] == "awaiting_review"
    assert updated["state"]["attempt"] == 2
    assert updated["pending_review"]["attempt"] == 2


def test_reject_marks_session_rejected() -> None:
    rejected = {
        **_paused_state(),
        "status": "failed",
        "human_decision": "reject_final",
        "failure_reason": "Reviewer rejected the solution: not acceptable",
    }
    rejected.pop("__interrupt__", None)

    updated, calls = _apply_with_fake("reject", "not acceptable", rejected)

    assert calls[0]["response"] == {
        "decision": "reject_final",
        "feedback": "not acceptable",
    }
    assert updated["paused"] is False
    assert updated["state"]["status"] == "rejected"


def test_reject_final_core_decision_routes_to_fail() -> None:
    from evalcode.graph import route_after_human
    from evalcode.nodes import apply_human_decision

    state = {"attempt": 1, "human_rounds": 0, "max_human_rounds": 2}
    update = apply_human_decision(
        state,
        {"decision": "reject_final", "feedback": "does not meet requirements"},
    )

    assert update["human_decision"] == "reject_final"
    assert update["human_feedback"] == "does not meet requirements"
    assert "Reviewer rejected" in update["failure_reason"]
    assert route_after_human({**state, **update}) == "fail"


def test_streamlit_smoke_shows_review_panel_and_auto_approve_off() -> None:
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py"
    at = AppTest.from_file(str(app_path), default_timeout=90).run()

    assert not at.exception, [e.value for e in at.exception]
    toggle = next(t for t in at.toggle if t.label == "Auto-Approve Code")
    assert toggle.value is False

    record = ui.make_run_record(
        _paused_state(),
        elapsed_s=1.0,
        backend="subprocess",
        use_rag=True,
        route="Dense-only",
        task="add two numbers",
        auto_approve=False,
    )
    at.session_state["last_run"] = record
    at.run()

    assert not at.exception, [e.value for e in at.exception]
    labels = [button.label for button in at.button]
    assert any("Approve Code" in label for label in labels)
    assert any("Request Revision" in label for label in labels)
    assert any("Reject" in label for label in labels)
