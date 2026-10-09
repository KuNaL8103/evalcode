"""Graph wiring tests."""

from __future__ import annotations

from typing import Any

import pytest

from evalcode.config import Settings
from evalcode.errors import DailyQuotaExceeded
from evalcode.graph import (
    Dependencies,
    build_graph,
    initial_state,
    pending_review,
    resume_task,
    route_after_analysis,
    route_after_human,
    route_after_llm,
    route_after_retrieve,
    route_after_tests,
    run_task,
    stream_task,
)
from evalcode.state import RunResult
from tests.fakes import FakeRetriever, ScriptedLLM, bundle_text, make_doc

CODE = 'import math\n\n\ndef add(a, b):\n    """Return a + b."""\n    return a + b\n'
TESTS = "from solution import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"


def make_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {"context_max_chars": 2000, "max_retries": 3, "max_human_rounds": 2}
    base.update(overrides)
    return Settings(**base)


def fake_sandbox_pass(code: str, tests: str, timeout_s: float, mem_mb: int) -> RunResult:
    return {
        "passed": True,
        "category": "pass",
        "exit_code": 0,
        "timed_out": False,
        "duration_s": 0.1,
        "tests_total": 1,
        "tests_failed": 0,
        "failures": [],
        "stdout": "",
        "stderr": "",
    }


def fake_sandbox_fail(code: str, tests: str, timeout_s: float, mem_mb: int) -> RunResult:
    return {
        "passed": False,
        "category": "assertion_failure",
        "exit_code": 1,
        "timed_out": False,
        "duration_s": 0.1,
        "tests_total": 1,
        "tests_failed": 1,
        "failures": [
            {
                "test_name": "test_add",
                "error_type": "AssertionError",
                "message": "assert 3 == 4",
                "traceback": "test_solution.py:5: AssertionError",
            }
        ],
        "stdout": "",
        "stderr": "",
    }


def test_routers():
    # route_after_tests
    # pass with auto_approve=True -> "finalize"
    state: dict[str, Any] = {
        "run_result": {"passed": True, "category": "pass"},
        "retries_used": 0,
        "max_retries": 3,
        "auto_approve": True,
    }
    assert route_after_tests(state) == "finalize"
    # pass with auto_approve=False -> "human_review"
    state["auto_approve"] = False
    assert route_after_tests(state) == "human_review"
    # fail with retries left
    state["run_result"] = {"passed": False, "category": "assertion_failure"}
    state["retries_used"] = 0
    state["max_retries"] = 3
    assert route_after_tests(state) == "analyze_error"
    # fail with retries_used == max_retries
    state["retries_used"] = 3
    assert route_after_tests(state) == "fail"
    # run_result None
    state["run_result"] = None
    assert route_after_tests(state) == "fail"
    # passed False with category "pass" must still NOT finalize (passed is the truth)
    state["run_result"] = {"passed": False, "category": "pass"}
    assert route_after_tests(state) == "fail"

    # route_after_llm
    assert route_after_llm({"status": "failed"}) == "fail"
    assert route_after_llm({"status": "running"}) == "run_tests"
    assert route_after_llm({"status": "approved"}) == "run_tests"

    # route_after_human
    base_human = {
        "human_decision": "approve",
        "human_rounds": 0,
        "max_human_rounds": 2,
    }
    assert route_after_human(base_human) == "finalize"
    base_human["human_decision"] = "edit"
    assert route_after_human(base_human) == "run_tests"
    base_human["human_decision"] = "reject"
    base_human["human_rounds"] = 0
    assert route_after_human(base_human) == "revise"
    base_human["human_rounds"] = 2
    assert route_after_human(base_human) == "revise"  # at limit, still revise
    base_human["human_rounds"] = 3
    assert route_after_human(base_human) == "fail"  # exceeds limit -> fail
    base_human["human_decision"] = "unknown"
    assert route_after_human(base_human) == "fail"


def test_graph_pass_first_try():
    # 1 scripted reply, sandbox PASS
    llm = ScriptedLLM([bundle_text(CODE, TESTS, explanation="sum")])
    calls = {"sandbox": 0}

    def sandbox(code, tests, timeout_s, mem_mb):
        calls["sandbox"] += 1
        return fake_sandbox_pass(code, tests, timeout_s, mem_mb)

    deps = Dependencies(llm=llm, settings=make_settings(max_retries=3), sandbox=sandbox)

    result = run_task("write a function add(a, b)", deps, auto_approve=True)

    assert result["status"] == "approved"
    assert result["final_code"] == CODE.strip()
    assert result["attempt"] == 1
    assert result["retries_used"] == 0
    assert len(llm.calls) == 1
    assert calls["sandbox"] == 1
    history_nodes = [e["node"] for e in result["history"]]
    assert history_nodes == ["generate", "run_tests", "finalize"]


def test_graph_fail_then_pass():
    # replies [v1, v2], sandbox [FAIL, PASS]
    CODE_V1 = "def add(a,b): return a - b\n"
    CODE_V2 = CODE

    llm = ScriptedLLM(
        [
            bundle_text(CODE_V1, TESTS, explanation="v1"),
            bundle_text(CODE_V2, TESTS, explanation="v2"),
        ]
    )
    sandbox_calls = {"count": 0}

    def sandbox(code, tests, timeout_s, mem_mb):
        sandbox_calls["count"] += 1
        if sandbox_calls["count"] == 1:
            return fake_sandbox_fail(code, tests, timeout_s, mem_mb)
        return fake_sandbox_pass(code, tests, timeout_s, mem_mb)

    deps = Dependencies(llm=llm, settings=make_settings(max_retries=3), sandbox=sandbox)

    result = run_task("write a function add(a, b)", deps, auto_approve=True)

    assert result["status"] == "approved"
    assert result["final_code"] == CODE.strip()  # v2
    assert result["attempt"] == 2
    assert result["retries_used"] == 1
    assert len(llm.calls) == 2
    assert sandbox_calls["count"] == 2
    history_nodes = [e["node"] for e in result["history"]]
    # No duplicated history - exactly this sequence
    assert history_nodes == [
        "generate",
        "run_tests",
        "analyze_error",
        "revise",
        "run_tests",
        "finalize",
    ]
    # revise LLM call's prompt contains "AssertionError"
    # The second call (index 1) is the revise call; its messages are in llm.calls[1]
    revise_messages = llm.calls[1]
    # Find the human message content (last message is human)
    human_msg = None
    for msg in revise_messages:
        if hasattr(msg, "content") and isinstance(msg.content, str):
            human_msg = msg.content
    assert human_msg is not None, "revise call should have a human message"
    assert "AssertionError" in human_msg, (
        f"revise prompt should contain AssertionError, got: {human_msg[:500]}"
    )
    assert result["token_usage"]["llm_calls"] == 2


def test_graph_exhausted_retries():
    # max_retries=3, sandbox always FAIL
    llm = ScriptedLLM([bundle_text(CODE, TESTS, explanation="same")] * 10)
    sandbox_calls = {"count": 0}

    def sandbox(code, tests, timeout_s, mem_mb):
        sandbox_calls["count"] += 1
        return fake_sandbox_fail(code, tests, timeout_s, mem_mb)

    deps = Dependencies(llm=llm, settings=make_settings(max_retries=3), sandbox=sandbox)

    result = run_task("write a function add(a, b)", deps, auto_approve=True)

    assert result["status"] == "failed"
    assert result["attempt"] == 4  # generation 1 + 3 retries
    assert result["retries_used"] == 3
    assert len(llm.calls) == 4  # 1 initial + 3 retries
    assert sandbox_calls["count"] == 4
    assert result["failure_reason"]
    assert "assertion_failure" in result["failure_reason"]
    assert result.get("final_code") is None


def test_graph_recursion_limit():
    # max_retries=10, sandbox always FAIL -> needs > 25 graph steps
    llm = ScriptedLLM([bundle_text(CODE, TESTS, explanation="same")] * 20)

    def sandbox(code, tests, timeout_s, mem_mb):
        return fake_sandbox_fail(code, tests, timeout_s, mem_mb)

    deps = Dependencies(llm=llm, settings=make_settings(max_retries=10), sandbox=sandbox)

    result = run_task("write a function add(a, b)", deps, auto_approve=True)

    assert result["status"] == "failed"
    assert result["attempt"] == 11  # 1 + 10 retries
    assert result["retries_used"] == 10
    assert len(llm.calls) == 11
    # recursion_limit=100 applied (default is 25)


def test_graph_llm_failures():
    # (a) DailyQuotaExceeded on the FIRST LLM call
    llm1 = ScriptedLLM([DailyQuotaExceeded("free-tier daily limit reached")])

    def sandbox(code, tests, timeout_s, mem_mb):
        pytest.fail("sandbox should not be called")

    deps1 = Dependencies(llm=llm1, settings=make_settings(max_retries=3), sandbox=sandbox)
    result1 = run_task("write a function add(a, b)", deps1, auto_approve=True)
    assert result1["status"] == "failed"
    assert result1["failure_reason"]
    assert "daily limit" in result1["failure_reason"].lower()
    assert result1["token_usage"] == {}  # no successful call

    # (b) first reply fine, sandbox FAIL, then DailyQuotaExceeded on the revise call
    llm2 = ScriptedLLM(
        [
            bundle_text(CODE, TESTS, explanation="first"),
            DailyQuotaExceeded("daily limit on revise"),
        ]
    )
    sandbox_calls = {"count": 0}

    def sandbox2(code, tests, timeout_s, mem_mb):
        sandbox_calls["count"] += 1
        return fake_sandbox_fail(code, tests, timeout_s, mem_mb)

    deps2 = Dependencies(llm=llm2, settings=make_settings(max_retries=3), sandbox=sandbox2)
    result2 = run_task("write a function add(a, b)", deps2, auto_approve=True)
    assert result2["status"] == "failed"
    assert sandbox_calls["count"] == 1
    assert result2["history"][-1]["node"] == "fail"
    assert "daily limit" in result2["failure_reason"].lower()


def test_graph_provided_tests():
    provided = "def test_given():\n    assert add(1, 1) == 2\n"
    llm = ScriptedLLM(
        [
            bundle_text(CODE, tests=None, explanation="first"),
            bundle_text(CODE, tests=None, explanation="second"),
        ]
    )
    received_tests = []

    def sandbox(code, tests, timeout_s, mem_mb):
        received_tests.append(tests)
        if len(received_tests) == 1:
            return fake_sandbox_fail(code, tests, timeout_s, mem_mb)
        return fake_sandbox_pass(code, tests, timeout_s, mem_mb)

    deps = Dependencies(llm=llm, settings=make_settings(max_retries=3), sandbox=sandbox)

    result = run_task(
        "write a function add(a, b)", deps, provided_tests=provided, auto_approve=True
    )

    assert result["status"] == "approved"
    assert result["tests"] == provided
    # Every sandbox call received exactly the provided_tests string
    assert received_tests == [provided, provided]


def test_run_and_stream_task():
    llm = ScriptedLLM([bundle_text(CODE, TESTS, explanation="sum")])

    def sandbox(code, tests, timeout_s, mem_mb):
        return fake_sandbox_pass(code, tests, timeout_s, mem_mb)

    deps = Dependencies(llm=llm, settings=make_settings(max_retries=3), sandbox=sandbox)

    # Run both with fresh fakes
    result = run_task("write add(a, b)", deps, auto_approve=True)
    history_nodes = [e["node"] for e in result["history"]]

    # Assertions on the result
    assert result["task"] == "write add(a, b)"
    assert result["max_retries"] == 3
    assert result["max_human_rounds"] == 2
    assert result["task_id"]  # non-empty string

    # Test initial_state keys and values for a given Settings
    settings = make_settings(max_retries=3, max_human_rounds=2)
    init = initial_state(
        task="write add(a, b)",
        settings=settings,
        task_id="test-task-123",
        provided_tests="def test_add(): pass",
        auto_approve=True,
    )
    assert init["task_id"] == "test-task-123"
    assert init["task"] == "write add(a, b)"
    assert init["provided_tests"] == "def test_add(): pass"
    assert init["auto_approve"] is True
    assert init["attempt"] == 0
    assert init["retries_used"] == 0
    assert init["max_retries"] == settings.max_retries
    assert init["max_human_rounds"] == settings.max_human_rounds
    assert init["human_rounds"] == 0
    assert init["status"] == "running"
    assert init["history"] == []
    assert init["token_usage"] == {}
    assert init["human_feedback"] is None
    assert init["human_decision"] is None
    assert init["retrieval_queries"] == []
    assert init["retrieved_docs"] == []
    assert init["code"] == ""
    assert init["tests"] == ""
    assert init["explanation"] == ""
    assert init["run_result"] is None
    assert init["error_analysis"] is None
    assert init["final_code"] is None
    assert init["failure_reason"] is None

    # Fresh deps for stream
    llm2 = ScriptedLLM([bundle_text(CODE, TESTS, explanation="sum")])

    def sandbox2(code, tests, timeout_s, mem_mb):
        return fake_sandbox_pass(code, tests, timeout_s, mem_mb)

    deps2 = Dependencies(llm=llm2, settings=make_settings(max_retries=3), sandbox=sandbox2)

    stream_nodes = []
    for node_name, _update in stream_task("write add(a, b)", deps2, auto_approve=True):
        stream_nodes.append(node_name)

    assert stream_nodes == history_nodes


def test_build_graph_topology():
    llm = ScriptedLLM([bundle_text(CODE, TESTS)])

    def sandbox(code, tests, timeout_s, mem_mb):
        return fake_sandbox_pass(code, tests, timeout_s, mem_mb)

    deps = Dependencies(llm=llm, settings=make_settings(), sandbox=sandbox)
    graph = build_graph(deps, checkpointer=None)
    nodes = set(graph.get_graph().nodes.keys())
    expected = {
        "generate",
        "run_tests",
        "analyze_error",
        "revise",
        "human_review",
        "finalize",
        "fail",
    }
    assert expected.issubset(nodes)
    assert "retrieve" not in nodes


# --- Task 9: Human-in-the-loop tests ---


def test_route_after_human_and_review_routing():
    # route_after_tests with pass + auto_approve True/False
    state_pass_auto = {
        "run_result": {"passed": True, "category": "pass"},
        "retries_used": 0,
        "max_retries": 3,
        "auto_approve": True,
    }
    assert route_after_tests(state_pass_auto) == "finalize"

    state_pass_human = {**state_pass_auto, "auto_approve": False}
    assert route_after_tests(state_pass_human) == "human_review"

    # route_after_human all branches
    base = {"human_rounds": 0, "max_human_rounds": 2}
    assert route_after_human({**base, "human_decision": "approve"}) == "finalize"
    assert route_after_human({**base, "human_decision": "edit"}) == "run_tests"
    assert route_after_human({**base, "human_decision": "reject"}) == "revise"
    # at limit -> revise
    assert route_after_human({**base, "human_decision": "reject", "human_rounds": 2}) == "revise"
    # exceeds limit -> fail
    assert route_after_human({**base, "human_decision": "reject", "human_rounds": 3}) == "fail"
    # unknown -> fail
    assert route_after_human({**base, "human_decision": "unknown"}) == "fail"


def test_apply_human_decision():
    from evalcode.nodes.human_review import apply_human_decision

    base_state = {
        "attempt": 1,
        "human_rounds": 0,
        "max_human_rounds": 2,
    }

    # approve
    result = apply_human_decision(base_state, {"decision": "approve"})
    assert result["human_decision"] == "approve"
    assert result["human_feedback"] is None
    assert "human_rounds" not in result  # unchanged
    assert len(result["history"]) == 1
    assert result["history"][0]["summary"]["decision"] == "approve"
    assert result["history"][0]["summary"]["human_rounds"] == 0

    # reject with feedback, rounds increments
    result = apply_human_decision(base_state, {"decision": "reject", "feedback": "add type hints"})
    assert result["human_decision"] == "reject"
    assert result["human_feedback"] == "add type hints"
    assert result["human_rounds"] == 1
    assert result["history"][0]["summary"]["decision"] == "reject"
    assert result["history"][0]["summary"]["human_rounds"] == 1
    assert result["history"][0]["summary"]["feedback"] == "add type hints"

    # reject exhausting max_human_rounds -> failure_reason
    state_at_limit = {**base_state, "human_rounds": 2, "max_human_rounds": 2}
    result = apply_human_decision(state_at_limit, {"decision": "reject", "feedback": "still wrong"})
    assert result["human_decision"] == "reject"
    assert result["human_rounds"] == 3
    assert "failure_reason" in result
    assert "max human rounds" in result["failure_reason"].lower()
    assert "3 time(s)" in result["failure_reason"]
    assert "still wrong" in result["failure_reason"]

    # edit keeps human_rounds unchanged, returns new code
    result = apply_human_decision(
        base_state, {"decision": "edit", "code": "def add(a,b): return a+b"}
    )
    assert result["human_decision"] == "edit"
    assert result["human_feedback"] is None
    assert "human_rounds" not in result  # unchanged
    assert result["code"] == "def add(a,b): return a+b"
    assert result["history"][0]["summary"]["decision"] == "edit"

    # malformed responses raise ValueError
    with pytest.raises(ValueError, match="must be a dict"):
        apply_human_decision(base_state, "not a dict")
    with pytest.raises(ValueError, match="Unknown decision"):
        apply_human_decision(base_state, {"decision": "invalid"})
    with pytest.raises(ValueError, match="non-empty.*feedback"):
        apply_human_decision(base_state, {"decision": "reject", "feedback": ""})
    with pytest.raises(ValueError, match="non-empty.*feedback"):
        apply_human_decision(base_state, {"decision": "reject"})
    with pytest.raises(ValueError, match="non-empty.*code"):
        apply_human_decision(base_state, {"decision": "edit", "code": ""})
    with pytest.raises(ValueError, match="non-empty.*code"):
        apply_human_decision(base_state, {"decision": "edit"})

    # input state not mutated
    original_human_rounds = base_state["human_rounds"]
    apply_human_decision(base_state, {"decision": "reject", "feedback": "x"})
    assert base_state["human_rounds"] == original_human_rounds


def test_graph_pauses_for_review():
    from langgraph.checkpoint.memory import MemorySaver

    llm = ScriptedLLM([bundle_text(CODE, TESTS, explanation="sum")])
    calls = {"sandbox": 0}

    def sandbox(code, tests, timeout_s, mem_mb):
        calls["sandbox"] += 1
        return fake_sandbox_pass(code, tests, timeout_s, mem_mb)

    deps = Dependencies(llm=llm, settings=make_settings(max_retries=3), sandbox=sandbox)
    checkpointer = MemorySaver()

    result = run_task(
        "write a function add(a, b)", deps, auto_approve=False, checkpointer=checkpointer
    )

    # Graph pauses at human_review
    assert "__interrupt__" in result
    assert result["status"] == "running"  # still running, awaiting review
    assert result["final_code"] is None
    assert calls["sandbox"] == 1  # sandbox called exactly once

    # pending_review extracts the payload
    payload = pending_review(result)
    assert payload is not None
    assert payload["task_id"] == result["task_id"]
    assert payload["attempt"] == 1
    assert payload["human_round"] == 0
    assert payload["max_human_rounds"] == 2
    assert payload["code"] == CODE.strip()
    assert payload["tests"] == TESTS.strip()
    assert payload["explanation"] == "sum"
    assert payload["run_summary"]["category"] == "pass"
    assert payload["run_summary"]["tests_total"] == 1
    assert payload["run_summary"]["tests_failed"] == 0

    # run_task without checkpointer + auto_approve=False raises ValueError
    with pytest.raises(ValueError, match="auto_approve=False requires a checkpointer"):
        run_task("write a function add(a, b)", deps, auto_approve=False, checkpointer=None)


def test_resume_approve():
    from langgraph.checkpoint.memory import MemorySaver

    llm = ScriptedLLM([bundle_text(CODE, TESTS, explanation="sum")])
    calls = {"sandbox": 0}

    def sandbox(code, tests, timeout_s, mem_mb):
        calls["sandbox"] += 1
        return fake_sandbox_pass(code, tests, timeout_s, mem_mb)

    deps = Dependencies(llm=llm, settings=make_settings(max_retries=3), sandbox=sandbox)
    checkpointer = MemorySaver()

    # First run: pauses at human_review
    result = run_task(
        "write a function add(a, b)", deps, auto_approve=False, checkpointer=checkpointer
    )
    assert "__interrupt__" in result
    task_id = result["task_id"]

    # Resume with approve
    result = resume_task(task_id, {"decision": "approve"}, deps, checkpointer=checkpointer)

    assert result["status"] == "approved"
    assert result["final_code"] == CODE.strip()
    history_nodes = [e["node"] for e in result["history"]]
    assert history_nodes == ["generate", "run_tests", "human_review", "finalize"]


def test_resume_reject_feedback_reaches_revise():
    from langgraph.checkpoint.memory import MemorySaver

    # LLM replies: 1st generate, 2nd revise (after reject)
    llm = ScriptedLLM(
        [
            bundle_text(CODE, TESTS, explanation="first"),
            bundle_text(CODE, TESTS, explanation="revised with type hints"),
        ]
    )
    sandbox_calls = {"count": 0}

    def sandbox(code, tests, timeout_s, mem_mb):
        sandbox_calls["count"] += 1
        return fake_sandbox_pass(code, tests, timeout_s, mem_mb)

    deps = Dependencies(
        llm=llm, settings=make_settings(max_retries=3, max_human_rounds=2), sandbox=sandbox
    )
    checkpointer = MemorySaver()

    # First run: generates, passes, pauses at human_review
    result = run_task(
        "write a function add(a, b)", deps, auto_approve=False, checkpointer=checkpointer
    )
    assert "__interrupt__" in result
    task_id = result["task_id"]
    assert result["human_rounds"] == 0

    # Resume with reject + feedback
    result = resume_task(
        task_id,
        {"decision": "reject", "feedback": "add type hints"},
        deps,
        checkpointer=checkpointer,
    )

    # Should have gone to revise (attempt 2), then run_tests, then paused again at human_review
    assert "__interrupt__" in result
    assert result["attempt"] == 2
    assert result["retries_used"] == 0  # reject doesn't consume retries_used
    assert result["human_rounds"] == 1
    # human_feedback is consumed by revise and cleared (set to None)
    assert result["human_feedback"] is None
    assert result["status"] == "running"

    # Verify the revise LLM call's prompt contained the feedback
    revise_messages = llm.calls[1]  # second call is revise
    human_msg = None
    for msg in revise_messages:
        if hasattr(msg, "content") and isinstance(msg.content, str):
            human_msg = msg.content
    assert human_msg is not None
    assert "add type hints" in human_msg, (
        f"revise prompt should contain human feedback, got: {human_msg[:500]}"
    )

    # Human rejected a PASSING solution: prompt should NOT contain
    # "Diagnosis:" and NOT contain "Category: pass",
    # and SHOULD contain "The tests currently PASS"
    assert "Diagnosis:" not in human_msg, (
        "revise prompt should NOT contain 'Diagnosis:' when human "
        f"rejected passing solution, got: {human_msg[:500]}"
    )
    assert "Category: pass" not in human_msg, (
        "revise prompt should NOT contain 'Category: pass' when human "
        f"rejected passing solution, got: {human_msg[:500]}"
    )
    assert "The tests currently PASS" in human_msg, (
        "revise prompt should contain 'The tests currently PASS' when "
        f"human rejected passing solution, got: {human_msg[:500]}"
    )

    # No analyze_error node in history (reject bypasses it)
    # After reject -> revise -> run_tests (pass) -> human_review (interrupt again).
    # The second human_review is paused at interrupt, so its history event
    # hasn't been added yet (added on next resume).
    history_nodes = [e["node"] for e in result["history"]]
    assert "analyze_error" not in history_nodes
    assert history_nodes == ["generate", "run_tests", "human_review", "revise", "run_tests"]

    # Second resume: approve
    result = resume_task(task_id, {"decision": "approve"}, deps, checkpointer=checkpointer)
    assert result["status"] == "approved"
    assert result["final_code"] == CODE.strip()
    history_nodes = [e["node"] for e in result["history"]]
    assert history_nodes == [
        "generate",
        "run_tests",
        "human_review",
        "revise",
        "run_tests",
        "human_review",
        "finalize",
    ]


def test_resume_edit_reruns_tests():
    from langgraph.checkpoint.memory import MemorySaver

    EDITED_CODE = "def add(a, b):\n    return a + b\n"

    llm = ScriptedLLM([bundle_text(CODE, TESTS, explanation="first")])
    sandbox_calls = {"count": 0, "last_code": None}

    def sandbox(code, tests, timeout_s, mem_mb):
        sandbox_calls["count"] += 1
        sandbox_calls["last_code"] = code
        return fake_sandbox_pass(code, tests, timeout_s, mem_mb)

    deps = Dependencies(llm=llm, settings=make_settings(max_retries=3), sandbox=sandbox)
    checkpointer = MemorySaver()

    # First run: generates, passes, pauses at human_review
    result = run_task(
        "write a function add(a, b)", deps, auto_approve=False, checkpointer=checkpointer
    )
    assert "__interrupt__" in result
    task_id = result["task_id"]

    # Resume with edit
    result = resume_task(
        task_id, {"decision": "edit", "code": EDITED_CODE}, deps, checkpointer=checkpointer
    )

    # Should have rerun tests with EDITED code, then paused again
    assert "__interrupt__" in result
    assert sandbox_calls["count"] == 2
    assert sandbox_calls["last_code"] == EDITED_CODE.strip()
    assert result["human_rounds"] == 0  # edit doesn't increment human_rounds
    assert result["code"] == EDITED_CODE.strip()

    # LLM calls unchanged (no revise call)
    assert len(llm.calls) == 1

    # Second resume: approve
    result = resume_task(task_id, {"decision": "approve"}, deps, checkpointer=checkpointer)
    assert result["status"] == "approved"
    assert result["final_code"] == EDITED_CODE.strip()
    history_nodes = [e["node"] for e in result["history"]]
    assert history_nodes == [
        "generate",
        "run_tests",
        "human_review",
        "run_tests",
        "human_review",
        "finalize",
    ]


def test_reject_exhausts_human_rounds():
    from langgraph.checkpoint.memory import MemorySaver

    # LLM replies: 1st generate, then 2 revise calls (for 2 rejects before exhaustion)
    llm = ScriptedLLM(
        [
            bundle_text(CODE, TESTS, explanation="first"),
            bundle_text(CODE, TESTS, explanation="rev1"),
            bundle_text(CODE, TESTS, explanation="rev2"),
        ]
    )
    sandbox_calls = {"count": 0}

    def sandbox(code, tests, timeout_s, mem_mb):
        sandbox_calls["count"] += 1
        return fake_sandbox_pass(code, tests, timeout_s, mem_mb)

    # max_human_rounds = 2
    deps = Dependencies(
        llm=llm, settings=make_settings(max_retries=3, max_human_rounds=2), sandbox=sandbox
    )
    checkpointer = MemorySaver()

    # First run: passes -> human_review
    result = run_task(
        "write a function add(a, b)", deps, auto_approve=False, checkpointer=checkpointer
    )
    assert "__interrupt__" in result
    task_id = result["task_id"]

    # Reject 1
    result = resume_task(
        task_id, {"decision": "reject", "feedback": "feedback 1"}, deps, checkpointer=checkpointer
    )
    assert "__interrupt__" in result
    assert result["human_rounds"] == 1

    # Reject 2
    result = resume_task(
        task_id, {"decision": "reject", "feedback": "feedback 2"}, deps, checkpointer=checkpointer
    )
    assert "__interrupt__" in result
    assert result["human_rounds"] == 2

    # Reject 3 -> exhausts (human_rounds becomes 3 > max_human_rounds=2)
    result = resume_task(
        task_id, {"decision": "reject", "feedback": "feedback 3"}, deps, checkpointer=checkpointer
    )
    assert result["status"] == "failed"
    assert "max human rounds" in result["failure_reason"].lower()
    assert result["history"][-1]["node"] == "fail"
    assert len(llm.calls) == 3  # generate + 2 revise calls


def test_sqlite_persistence_across_rebuilt_graphs(tmp_path):
    from evalcode.persistence import open_checkpointer

    llm = ScriptedLLM([bundle_text(CODE, TESTS, explanation="first")])
    sandbox_calls = {"count": 0}

    def sandbox(code, tests, timeout_s, mem_mb):
        sandbox_calls["count"] += 1
        return fake_sandbox_pass(code, tests, timeout_s, mem_mb)

    db_path = tmp_path / "checkpoints.db"

    # First graph: run to pause
    with open_checkpointer(db_path) as checkpointer1:
        deps1 = Dependencies(llm=llm, settings=make_settings(max_retries=3), sandbox=sandbox)
        result = run_task(
            "write a function add(a, b)", deps1, auto_approve=False, checkpointer=checkpointer1
        )
        assert "__interrupt__" in result
        task_id = result["task_id"]
        original_code = result["code"]
        assert original_code == CODE.strip()

    # Second graph: resume with new dependencies
    with open_checkpointer(db_path) as checkpointer2:
        # New LLM and sandbox (simulating process restart)
        llm2 = ScriptedLLM([bundle_text(CODE, TESTS, explanation="resumed")])
        sandbox_calls2 = {"count": 0}

        def sandbox2(code, tests, timeout_s, mem_mb):
            sandbox_calls2["count"] += 1
            return fake_sandbox_pass(code, tests, timeout_s, mem_mb)

        deps2 = Dependencies(llm=llm2, settings=make_settings(max_retries=3), sandbox=sandbox2)

        # Verify checkpoint state before resume
        snapshot = build_graph(deps2, checkpointer=checkpointer2).get_state(
            {"configurable": {"thread_id": task_id}}
        )
        assert snapshot.values["task_id"] == task_id
        assert snapshot.values["code"] == original_code
        assert snapshot.next == ("human_review",)

        result = resume_task(task_id, {"decision": "approve"}, deps2, checkpointer=checkpointer2)

        assert result["status"] == "approved"
        assert result["final_code"] == CODE.strip()
        # The resumed state should have the original task_id and code
        assert result["task_id"] == task_id
        assert result["code"] == original_code

    # After both contexts exit, verify the DB file is released
    # and deletable (Windows handle released)
    db_path.unlink()
    assert not db_path.exists()


# --- Task 10: RAG integration tests ---


def test_route_after_analysis_and_retrieve():
    """Test route_after_analysis and route_after_retrieve routers."""
    # needs_docs False -> revise
    state = {"error_analysis": {"needs_docs": False, "retrieval_queries": ["q1"]}}
    assert route_after_analysis(state) == "revise"

    # needs_docs True but empty queries -> revise
    state = {"error_analysis": {"needs_docs": True, "retrieval_queries": []}}
    assert route_after_analysis(state) == "revise"

    # Fresh queries -> retrieve
    state = {
        "error_analysis": {"needs_docs": True, "retrieval_queries": ["fresh query"]},
        "retrieval_queries": ["old query"],
    }
    assert route_after_analysis(state) == "retrieve"

    # All queries already used -> revise (dedupe)
    state = {
        "error_analysis": {"needs_docs": True, "retrieval_queries": ["q1", "q2"]},
        "retrieval_queries": ["q1", "q2"],
    }
    assert route_after_analysis(state) == "revise"

    # Some queries fresh -> retrieve
    state = {
        "error_analysis": {"needs_docs": True, "retrieval_queries": ["q1", "q3"]},
        "retrieval_queries": ["q1", "q2"],
    }
    assert route_after_analysis(state) == "retrieve"

    # route_after_retrieve: code empty -> generate
    assert route_after_retrieve({"code": ""}) == "generate"
    assert route_after_retrieve({"code": "   "}) == "generate"
    # code present -> revise
    assert route_after_retrieve({"code": "def foo(): pass"}) == "revise"


def test_topology_rag_on_and_off():
    """Test graph topology with and without retriever."""
    llm = ScriptedLLM([bundle_text(CODE, TESTS)])

    def sandbox(code, tests, timeout_s, mem_mb):
        return fake_sandbox_pass(code, tests, timeout_s, mem_mb)

    # RAG off (retriever=None)
    deps_off = Dependencies(llm=llm, settings=make_settings(), sandbox=sandbox)
    graph_off = build_graph(deps_off, checkpointer=None)
    nodes_off = set(graph_off.get_graph().nodes.keys())
    expected_base = {
        "generate",
        "run_tests",
        "analyze_error",
        "revise",
        "human_review",
        "finalize",
        "fail",
    }
    assert expected_base.issubset(nodes_off)
    assert "retrieve" not in nodes_off

    # RAG on (FakeRetriever)
    retriever = FakeRetriever(default=[make_doc("d1")])
    deps_on = Dependencies(llm=llm, settings=make_settings(), sandbox=sandbox, retriever=retriever)
    graph_on = build_graph(deps_on, checkpointer=None)
    nodes_on = set(graph_on.get_graph().nodes.keys())
    assert expected_base.issubset(nodes_on)
    assert "retrieve" in nodes_on
    # Graph also includes __start__ and __end__ nodes
    assert len(nodes_on) >= len(expected_base) + 1  # base + retrieve + internal nodes


def test_graph_rag_grounds_generate():
    """RAG on: retrieve runs first, generate prompt contains [doc:<id>],
    history starts with retrieve, result has retrieved_docs."""
    doc = make_doc("doc-json", "json.loads parses JSON", score=0.9)
    retriever = FakeRetriever(default=[doc])

    llm = ScriptedLLM([bundle_text(CODE, TESTS, explanation="with json")])
    sandbox_calls = {"count": 0}

    def sandbox(code, tests, timeout_s, mem_mb):
        sandbox_calls["count"] += 1
        return fake_sandbox_pass(code, tests, timeout_s, mem_mb)

    deps = Dependencies(llm=llm, settings=make_settings(), sandbox=sandbox, retriever=retriever)

    result = run_task("write a function to parse json", deps, auto_approve=True)

    # Status approved
    assert result["status"] == "approved"
    # History starts with retrieve -> generate -> run_tests -> finalize
    history_nodes = [e["node"] for e in result["history"]]
    assert history_nodes == ["retrieve", "generate", "run_tests", "finalize"]
    # retrieved_docs contains the doc
    assert result["retrieved_docs"][0]["id"] == "doc-json"
    # Generate LLM call's human message contains [doc:doc-json]
    gen_messages = llm.calls[0]
    human_content = ""
    for msg in gen_messages:
        if hasattr(msg, "content") and isinstance(msg.content, str):
            human_content = msg.content
    assert "[doc:doc-json]" in human_content


def test_graph_error_driven_reretrieval():
    """Error-driven re-retrieval: sandbox fails, analyze_error produces
    retrieval_queries, retrieve runs again (mode error), revise prompt
    contains the new doc. Second failure with same error -> only ONE
    error-mode retrieve (dedupe)."""
    # Doc for task query
    doc_task = make_doc("doc-task", "math module docs", score=0.8)
    # Doc for error query "math sqroot"
    doc_error = make_doc("doc-error", "math.sqrt square root", score=0.95)

    retriever = FakeRetriever(
        docs_by_query={
            "write a function using math sqroot": [doc_task],
            "math sqroot": [doc_error],
        }
    )

    # Sandbox: first fails with AttributeError (math.sqroot), then passes
    CODE_V1 = "import math\n\ndef f():\n    return math.sqroot(4)\n"
    CODE_V2 = "import math\n\ndef f():\n    return math.sqrt(4)\n"
    TESTS_MATH = "from solution import f\n\ndef test_f():\n    assert f() == 2.0\n"

    llm = ScriptedLLM(
        [
            bundle_text(CODE_V1, TESTS_MATH, explanation="v1"),
            bundle_text(CODE_V2, TESTS_MATH, explanation="v2"),
        ]
    )
    sandbox_calls = {"count": 0}

    def sandbox(code, tests, timeout_s, mem_mb):
        sandbox_calls["count"] += 1
        if sandbox_calls["count"] == 1:
            return {
                "passed": False,
                "category": "runtime_error",
                "exit_code": 1,
                "timed_out": False,
                "duration_s": 0.1,
                "tests_total": 1,
                "tests_failed": 1,
                "failures": [
                    {
                        "test_name": "test_f",
                        "error_type": "AttributeError",
                        "message": "module 'math' has no attribute 'sqroot'",
                        "traceback": (
                            "solution.py:3: AttributeError: module 'math' has no attribute 'sqroot'"
                        ),
                    }
                ],
                "stdout": "",
                "stderr": "",
            }
        return fake_sandbox_pass(code, tests, timeout_s, mem_mb)

    deps = Dependencies(
        llm=llm, settings=make_settings(max_retries=3), sandbox=sandbox, retriever=retriever
    )

    result = run_task("write a function using math sqroot", deps, auto_approve=True)

    assert result["status"] == "approved"
    # History: retrieve -> generate -> run_tests -> analyze_error
    # -> retrieve -> revise -> run_tests -> finalize
    history_nodes = [e["node"] for e in result["history"]]
    assert history_nodes == [
        "retrieve",
        "generate",
        "run_tests",
        "analyze_error",
        "retrieve",
        "revise",
        "run_tests",
        "finalize",
    ]

    # Second retrieve event mode == "error"
    retrieve_events = [e for e in result["history"] if e["node"] == "retrieve"]
    assert len(retrieve_events) == 2
    assert retrieve_events[0]["summary"]["mode"] == "task"
    assert retrieve_events[1]["summary"]["mode"] == "error"
    # Error queries include both suspect symbol and exception message
    assert "math sqroot" in retrieve_events[1]["summary"]["queries"]
    assert len(retrieve_events[1]["summary"]["queries"]) <= 3

    # Revise LLM call (index 1 because no rewrite LLM call with default settings)
    revise_messages = llm.calls[1]
    human_content = ""
    for msg in revise_messages:
        if hasattr(msg, "content") and isinstance(msg.content, str):
            human_content = msg.content
    assert "[doc:doc-error]" in human_content

    # Scenario 2: sandbox fails twice with SAME error -> only ONE error-mode retrieve
    retriever2 = FakeRetriever(
        docs_by_query={
            "task": [make_doc("dt")],
            "math sqroot": [make_doc("de")],
        }
    )
    llm2 = ScriptedLLM(
        [
            bundle_text(CODE_V1, TESTS_MATH, explanation="v1"),
            bundle_text(CODE_V1, TESTS_MATH, explanation="v1 again"),
            bundle_text(CODE_V2, TESTS_MATH, explanation="v2"),
        ]
    )
    sandbox_calls2 = {"count": 0}

    def sandbox2(code, tests, timeout_s, mem_mb):
        sandbox_calls2["count"] += 1
        if sandbox_calls2["count"] <= 2:
            return {
                "passed": False,
                "category": "runtime_error",
                "exit_code": 1,
                "timed_out": False,
                "duration_s": 0.1,
                "tests_total": 1,
                "tests_failed": 1,
                "failures": [
                    {
                        "test_name": "test_f",
                        "error_type": "AttributeError",
                        "message": "module 'math' has no attribute 'sqroot'",
                        "traceback": (
                            "solution.py:3: AttributeError: module 'math' has no attribute 'sqroot'"
                        ),
                    }
                ],
                "stdout": "",
                "stderr": "",
            }
        return fake_sandbox_pass(code, tests, timeout_s, mem_mb)

    deps2 = Dependencies(
        llm=llm2, settings=make_settings(max_retries=3), sandbox=sandbox2, retriever=retriever2
    )

    result2 = run_task("task", deps2, auto_approve=True)

    assert result2["status"] == "approved"
    # Should have only 2 retrieve events (one task, one error)
    retrieve_events2 = [e for e in result2["history"] if e["node"] == "retrieve"]
    assert len(retrieve_events2) == 2
    # The error-mode retrieve should only happen once despite two failures
    error_retrieves = [e for e in retrieve_events2 if e["summary"]["mode"] == "error"]
    assert len(error_retrieves) == 1


def test_graph_rag_off_has_no_retrieve():
    """RAG off: history has no retrieve; retrieval_queries == [] and retrieved_docs == []."""
    llm = ScriptedLLM([bundle_text(CODE, TESTS, explanation="sum")])

    def sandbox(code, tests, timeout_s, mem_mb):
        return fake_sandbox_pass(code, tests, timeout_s, mem_mb)

    # No retriever
    deps = Dependencies(llm=llm, settings=make_settings(), sandbox=sandbox)

    result = run_task("write a function add(a, b)", deps, auto_approve=True)

    assert result["status"] == "approved"
    history_nodes = [e["node"] for e in result["history"]]
    assert "retrieve" not in history_nodes
    assert result["retrieval_queries"] == []
    assert result["retrieved_docs"] == []


def test_build_retriever_fallbacks(tmp_path):
    """build_retriever returns None for missing dir, empty store, or
    returns Retriever for populated store. Always closes stores."""
    from evalcode.graph import build_retriever
    from evalcode.rag.embeddings import FakeEmbedder
    from evalcode.rag.store import VectorStore

    _ = make_settings(chroma_dir=str(tmp_path / "chroma"), collection_name="test")

    # Case 1: missing dir -> None, dir NOT created
    missing_dir = tmp_path / "missing"
    settings_missing = make_settings(chroma_dir=str(missing_dir), collection_name="test")
    retriever = build_retriever(settings_missing)
    assert retriever is None
    assert not missing_dir.exists()

    # Case 2: empty store -> None
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    settings_empty = make_settings(chroma_dir=str(empty_dir), collection_name="test")
    embedder = FakeEmbedder(dim=64)
    store = VectorStore(empty_dir, "test", embedder)
    store.close()  # empty store
    retriever = build_retriever(settings_empty, embedder=embedder)
    assert retriever is None
    # Dir should still be deletable (store closed)
    import shutil

    shutil.rmtree(empty_dir)

    # Case 3: store with 1 upserted DocChunk -> Retriever with top_k/min_score from settings
    populated_dir = tmp_path / "populated"
    populated_dir.mkdir()
    settings_pop = make_settings(
        chroma_dir=str(populated_dir),
        collection_name="test",
        retrieval_top_k=7,
        retrieval_min_score=0.3,
    )
    embedder2 = FakeEmbedder(dim=64)
    store2 = VectorStore(populated_dir, "test", embedder2)
    from evalcode.rag.types import DocChunk

    chunk = DocChunk(
        id="c1",
        text="test chunk",
        metadata={"library": "json", "qualname": "json.loads", "import_path": "json"},
    )
    store2.upsert([chunk])
    store2.close()
    retriever = build_retriever(settings_pop, embedder=embedder2)
    assert retriever is not None
    assert retriever.top_k == 7
    assert retriever.min_score == 0.3
    retriever.store.close()
    shutil.rmtree(populated_dir)
