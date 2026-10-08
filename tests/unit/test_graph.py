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
    route_after_llm,
    route_after_tests,
    run_task,
    stream_task,
)
from evalcode.state import RunResult
from tests.fakes import ScriptedLLM, bundle_text

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
    # pass
    state: dict[str, Any] = {
        "run_result": {"passed": True, "category": "pass"},
        "retries_used": 0,
        "max_retries": 3,
    }
    assert route_after_tests(state) == "finalize"
    # pass with auto_approve True/False -> both "finalize"
    state["auto_approve"] = True
    assert route_after_tests(state) == "finalize"
    state["auto_approve"] = False
    assert route_after_tests(state) == "finalize"
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


def test_graph_pass_first_try():
    # 1 scripted reply, sandbox PASS
    llm = ScriptedLLM([bundle_text(CODE, TESTS, explanation="sum")])
    calls = {"sandbox": 0}

    def sandbox(code, tests, timeout_s, mem_mb):
        calls["sandbox"] += 1
        return fake_sandbox_pass(code, tests, timeout_s, mem_mb)

    deps = Dependencies(llm=llm, settings=make_settings(max_retries=3), sandbox=sandbox)

    result = run_task("write a function add(a, b)", deps)

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

    result = run_task("write a function add(a, b)", deps)

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
    # Check that the second LLM call (revise) had the error in context
    # The ScriptedLLM records calls; verify second call had messages with error info
    # (We just verify token_usage and call count here)
    assert result["token_usage"]["llm_calls"] == 2


def test_graph_exhausted_retries():
    # max_retries=3, sandbox always FAIL
    llm = ScriptedLLM([bundle_text(CODE, TESTS, explanation="same")] * 10)
    sandbox_calls = {"count": 0}

    def sandbox(code, tests, timeout_s, mem_mb):
        sandbox_calls["count"] += 1
        return fake_sandbox_fail(code, tests, timeout_s, mem_mb)

    deps = Dependencies(llm=llm, settings=make_settings(max_retries=3), sandbox=sandbox)

    result = run_task("write a function add(a, b)", deps)

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

    result = run_task("write a function add(a, b)", deps)

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
    result1 = run_task("write a function add(a, b)", deps1)
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
    result2 = run_task("write a function add(a, b)", deps2)
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

    result = run_task("write a function add(a, b)", deps, provided_tests=provided)

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
    result = run_task("write add(a, b)", deps)
    history_nodes = [e["node"] for e in result["history"]]

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
    for node_name, _update in stream_task("write add(a, b)", deps2):
        stream_nodes.append(node_name)

    assert stream_nodes == history_nodes


def test_build_graph_topology():
    llm = ScriptedLLM([bundle_text(CODE, TESTS)])

    def sandbox(code, tests, timeout_s, mem_mb):
        return fake_sandbox_pass(code, tests, timeout_s, mem_mb)

    deps = Dependencies(llm=llm, settings=make_settings(), sandbox=sandbox)
    graph = build_graph(deps, checkpointer=None)
    nodes = set(graph.get_graph().nodes.keys())
    expected = {"generate", "run_tests", "analyze_error", "revise", "finalize", "fail"}
    assert expected.issubset(nodes)
    assert "human_review" not in nodes
    assert "retrieve" not in nodes
