"""Unit tests for the revise node (Task 7)."""

from __future__ import annotations

import copy
from typing import Any

from evalcode.config import Settings
from evalcode.errors import DailyQuotaExceeded
from evalcode.nodes.revise import make_revise_node
from tests.fakes import ScriptedLLM, bundle_text

CODE_V1 = "def add(a, b):\n    return a + b\n"
TESTS_V1 = "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"

CODE_V2 = "def add(a, b):\n    return a + b + 1\n"
TESTS_V2 = "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 4\n"

RUN_RESULT = {
    "passed": False,
    "category": "assertion_failure",
    "exit_code": 1,
    "timed_out": False,
    "duration_s": 2.0,
    "tests_total": 1,
    "tests_failed": 1,
    "failures": [
        {
            "test_name": "test_add",
            "error_type": "assertion_failure",
            "message": "assert 3 == 4",
            "traceback": (
                "def test_add():\n>       assert add(1, 2) == 4\n"
                "E       assert 3 == 4\n\n"
                "test_solution.py:3: AssertionError"
            ),
        }
    ],
    "stdout": "FAILED test_solution.py::test_add - assert 3 == 4",
    "stderr": "",
}

ERROR_ANALYSIS = {
    "category": "assertion_failure",
    "root_cause": "AssertionError: assert 3 == 4",
    "fault": "code",
    "fix_plan": "Fix the code: the logic does not satisfy the assertions.",
    "needs_docs": False,
    "retrieval_queries": [],
    "suspect_symbols": ["add"],
}

PROVIDED_TESTS = "from solution import add\n\ndef test_given():\n    assert add(1, 1) == 2\n"

RETRIEVED_DOCS = [
    {
        "id": "doc1",
        "text": "json.loads(s) -> object\nParse a JSON document.",
        "score": 0.9,
        "library": "json",
        "qualname": "json.loads",
        "import_path": "json",
    }
]


def make_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {"context_max_chars": 2000}
    base.update(overrides)
    return Settings(**base)


def test_revise_automatic_success_counters() -> None:
    """Automatic revision increments attempt and retries_used."""
    llm = ScriptedLLM(
        [bundle_text(CODE_V2, TESTS_V2, explanation="fixed off by one", docs_used=["doc1"])]
    )
    node = make_revise_node(llm, make_settings())

    state = {
        "task": "add(a, b)",
        "code": CODE_V1,
        "tests": TESTS_V1,
        "run_result": RUN_RESULT,
        "error_analysis": ERROR_ANALYSIS,
        "attempt": 1,
        "retries_used": 0,
        "human_feedback": None,
        "retrieved_docs": RETRIEVED_DOCS,
    }
    snapshot = copy.deepcopy(state)
    update = node(state)

    # Input state unchanged
    assert state == snapshot

    assert update["code"] == CODE_V2.strip()
    assert update["tests"] == TESTS_V2.strip()
    assert update["explanation"] == "fixed off by one"
    assert update["attempt"] == 2
    assert update["retries_used"] == 1
    assert "human_feedback" in update
    assert update["human_feedback"] is None
    assert update["status"] == "running"

    # Usage recorded
    assert update["token_usage"]["llm_calls"] == 1

    # History event
    event = update["history"][0]
    assert event["node"] == "revise"
    assert event["attempt"] == 2
    assert event["summary"]["human_driven"] is False
    assert event["summary"]["reasks"] == 0
    assert event["summary"]["retries_used"] == 1
    assert "doc_ids" in event["summary"]
    assert event["summary"]["docs_used"] == ["doc1"]


def test_revise_human_driven_does_not_spend_retry() -> None:
    """Human-driven revision increments attempt but NOT retries_used."""
    llm = ScriptedLLM([bundle_text(CODE_V2, TESTS_V2, explanation="human fix")])
    node = make_revise_node(llm, make_settings())

    state = {
        "task": "add(a, b)",
        "code": CODE_V1,
        "tests": TESTS_V1,
        "run_result": RUN_RESULT,
        "error_analysis": ERROR_ANALYSIS,
        "attempt": 1,
        "retries_used": 1,
        "human_feedback": "make it handle empty input",
        "retrieved_docs": RETRIEVED_DOCS,
    }
    update = node(state)

    assert update["attempt"] == 2
    assert update["retries_used"] == 1  # unchanged
    assert update["human_feedback"] is None
    assert update["status"] == "running"

    event = update["history"][0]
    assert event["summary"]["human_driven"] is True

    # Feedback text appears in the prompt sent to LLM
    assert len(llm.calls) == 1
    user_msg = llm.calls[0][1].content
    assert "make it handle empty input" in user_msg


def test_revise_respects_provided_tests_and_missing_tests() -> None:
    """Provided tests are forced; missing tests in reply keeps old tests."""
    # Case 1: provided_tests set, model returns different tests -> result uses provided
    llm1 = ScriptedLLM([bundle_text(CODE_V2, TESTS_V2, explanation="different tests")])
    node1 = make_revise_node(llm1, make_settings())

    state1 = {
        "task": "t",
        "code": CODE_V1,
        "tests": TESTS_V1,
        "run_result": RUN_RESULT,
        "error_analysis": ERROR_ANALYSIS,
        "attempt": 1,
        "retries_used": 0,
        "provided_tests": PROVIDED_TESTS,
    }
    update1 = node1(state1)
    assert update1["tests"] == PROVIDED_TESTS  # raw value from state (includes trailing newline)

    # Case 2: no provided_tests, model returns code but NO <tests> tag ->
    # keep state["tests"] unchanged, no re-ask (per FIX 5)
    llm2 = ScriptedLLM([bundle_text(CODE_V2, tests=None, explanation="no tests tag")])
    node2 = make_revise_node(llm2, make_settings())

    state2 = {
        "task": "t",
        "code": CODE_V1,
        "tests": TESTS_V1,
        "run_result": RUN_RESULT,
        "error_analysis": ERROR_ANALYSIS,
        "attempt": 1,
        "retries_used": 0,
        "provided_tests": "",
    }
    update2 = node2(state2)
    assert update2["status"] == "running"
    assert update2["tests"] == TESTS_V1  # state value, no strip
    assert update2["history"][0]["summary"]["reasks"] == 0
    assert len(llm2.calls) == 1  # exactly one LLM call, no re-ask


def test_revise_failure_paths() -> None:
    """Failure paths: LLMError, double parse failure, recoverable parse failure."""
    # (a) LLMError on first call -> failed, non-empty failure_reason, no attempt/retries_used
    llm1 = ScriptedLLM([DailyQuotaExceeded("daily limit reached")])
    node1 = make_revise_node(llm1, make_settings())

    state1 = {
        "task": "t",
        "code": CODE_V1,
        "tests": TESTS_V1,
        "run_result": RUN_RESULT,
        "error_analysis": ERROR_ANALYSIS,
        "attempt": 1,
        "retries_used": 0,
    }
    update1 = node1(state1)
    assert update1["status"] == "failed"
    assert update1["failure_reason"]
    assert "attempt" not in update1
    assert "retries_used" not in update1
    assert update1["history"][0]["summary"]["error"] == "DailyQuotaExceeded"

    # (b) Unparseable twice -> failed, re-ask happened (2 calls), reply_head recorded
    bad = "Sure! Here is the answer: 5"
    llm2 = ScriptedLLM([bad, "still prose"])
    node2 = make_revise_node(llm2, make_settings())

    state2 = {
        "task": "t",
        "code": CODE_V1,
        "tests": TESTS_V1,
        "run_result": RUN_RESULT,
        "error_analysis": ERROR_ANALYSIS,
        "attempt": 1,
        "retries_used": 0,
    }
    update2 = node2(state2)
    assert update2["status"] == "failed"
    assert "re-ask" in update2["failure_reason"]
    assert len(llm2.calls) == 2
    event = update2["history"][0]
    assert event["summary"]["reasks"] == 1
    assert "reply_head" in event["summary"]
    assert event["summary"]["reply_head"]
    assert "parse_reason" in event["summary"]

    # (c) Unparseable then good -> success, reasks == 1, usage llm_calls == 2
    llm3 = ScriptedLLM([bad, bundle_text(CODE_V2, TESTS_V2, explanation="fixed on retry")])
    node3 = make_revise_node(llm3, make_settings())

    state3 = {
        "task": "t",
        "code": CODE_V1,
        "tests": TESTS_V1,
        "run_result": RUN_RESULT,
        "error_analysis": ERROR_ANALYSIS,
        "attempt": 1,
        "retries_used": 0,
    }
    update3 = node3(state3)
    assert update3["status"] == "running"
    assert update3["code"] == CODE_V2.strip()
    assert update3["token_usage"]["llm_calls"] == 2
    assert update3["history"][0]["summary"]["reasks"] == 1


def test_revise_prompt_contents_and_bounds() -> None:
    """Build prompt with long content and verify bounds and sections."""
    from evalcode.nodes.analyze_error import make_analyze_error_node
    from evalcode.prompts import build_revise_messages

    long_traceback = "x" * 3000
    long_stdout = "y" * 1000
    long_stderr = "z" * 1000

    run_result = {
        "passed": False,
        "category": "assertion_failure",
        "exit_code": 1,
        "timed_out": False,
        "duration_s": 2.0,
        "tests_total": 1,
        "tests_failed": 1,
        "failures": [
            {
                "test_name": "test_add",
                "error_type": "assertion_failure",
                "message": "assert 3 == 4",
                "traceback": long_traceback,
            }
        ],
        "stdout": long_stdout,
        "stderr": long_stderr,
    }

    # Generate real analyze_error history events for attempts 1..5
    analyze_node = make_analyze_error_node(None, make_settings())
    history_events = []
    for i in range(1, 6):
        state_for_analyze = {
            "run_result": run_result,
            "code": CODE_V1,
            "tests": TESTS_V1,
            "attempt": i,
        }
        update = analyze_node(state_for_analyze)
        history_events.append(update["history"][0])

    state = {
        "task": "add(a, b)",
        "code": CODE_V1,
        "tests": TESTS_V1,
        "run_result": run_result,
        "error_analysis": {
            "category": "assertion_failure",
            "root_cause": "AssertionError: assert 3 == 4",
            "fault": "code",
            "fix_plan": "Fix the logic",
            "needs_docs": False,
            "retrieval_queries": [],
            "suspect_symbols": ["add"],
        },
        "attempt": 6,
        "retries_used": 2,
        "human_feedback": "please fix this",
        "retrieved_docs": RETRIEVED_DOCS,
        "history": history_events,
    }

    messages = build_revise_messages(state, context_max_chars=1500)
    assert len(messages) == 2
    human_content = messages[1].content

    # Contains key sections
    assert "test_add" in human_content
    assert "assert 3 == 4" in human_content
    assert "Fix the logic" in human_content
    assert "please fix this" in human_content
    assert "[doc:doc1]" in human_content
    assert "Previous failed attempts:" in human_content

    # Only attempts 3, 4, 5 appear (last 3 before current attempt 6)
    assert "attempt 3:" in human_content
    assert "attempt 4:" in human_content
    assert "attempt 5:" in human_content
    assert "attempt 1:" not in human_content
    assert "attempt 2:" not in human_content

    # Verify real history events have the keys revise reads
    for h in history_events:
        assert h["node"] == "analyze_error"
        assert "attempt" in h
        assert "summary" in h
        assert "category" in h["summary"]
        assert "root_cause" in h["summary"]

    # Traceback excerpt <= 1500 chars (text after "Traceback (tail):\n" up to next section)
    traceback_section = human_content.split("Traceback (tail):\n")[1].split("\n\n")[0]
    assert len(traceback_section) <= 1500

    # Total prompt < 10000 chars
    assert len(human_content) < 10000

    # FIX 7c: prompt contains exception type (AssertionError) and does NOT
    # contain "assertion_failure: " after "Exception: " (proves FIX 1)
    assert "AssertionError" in human_content
    exception_line = human_content.split("Exception: ")[1].split("\n")[0]
    assert not exception_line.startswith("assertion_failure: "), (
        f"Exception line should not start with 'assertion_failure: ', got: {exception_line}"
    )
