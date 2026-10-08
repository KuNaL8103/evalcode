"""Terminal node tests."""

from __future__ import annotations

from evalcode.nodes.terminal import fail_node, finalize_node


def test_finalize_and_fail_nodes():
    # finalize sets final_code/status and one event
    state = {"code": "def add(a,b): return a+b", "attempt": 1}
    update = finalize_node(state)
    assert update["final_code"] == state["code"]
    assert update["status"] == "approved"
    assert len(update["history"]) == 1
    event = update["history"][0]
    assert event["node"] == "finalize"
    assert event["attempt"] == 1
    assert "summary" in event
    assert "final_code_chars" in event["summary"]

    # fail keeps an existing failure_reason verbatim
    state2 = {"attempt": 3, "retries_used": 2, "failure_reason": "LLM quota exhausted"}
    update2 = fail_node(state2)
    assert update2["status"] == "failed"
    assert update2["failure_reason"] == "LLM quota exhausted"
    assert len(update2["history"]) == 1
    assert update2["history"][0]["node"] == "fail"
    assert "quota exhausted" in update2["history"][0]["summary"]["reason"]

    # fail composes the retries/category message when none exists
    state3 = {"attempt": 4, "retries_used": 3, "run_result": {"category": "assertion_failure"}}
    update3 = fail_node(state3)
    assert update3["status"] == "failed"
    assert "3 automatic revision" in update3["failure_reason"]
    assert "attempt 4" in update3["failure_reason"]
    assert "assertion_failure" in update3["failure_reason"]

    # fail with no run_result gives the generic message
    state4 = {"attempt": 1, "retries_used": 0}
    update4 = fail_node(state4)
    assert update4["failure_reason"] == "Run ended without a test result."

    # inputs are not mutated
    assert state == {"code": "def add(a,b): return a+b", "attempt": 1}
    assert state2 == {
        "attempt": 3,
        "retries_used": 2,
        "failure_reason": "LLM quota exhausted",
    }
    assert state3 == {
        "attempt": 4,
        "retries_used": 3,
        "run_result": {"category": "assertion_failure"},
    }
    assert state4 == {"attempt": 1, "retries_used": 0}
