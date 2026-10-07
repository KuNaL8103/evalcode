"""Unit tests for the analyze_error node (Task 7)."""

from __future__ import annotations

import copy
import json
from typing import Any

from evalcode.config import Settings
from evalcode.errors import DailyQuotaExceeded
from evalcode.nodes.analyze_error import make_analyze_error_node
from evalcode.state import ErrorAnalysis
from tests.fakes import ScriptedLLM

# Real probe output constants - trimmed to essential parts
ASSERTION_RUN_RESULT = {
    "passed": False,
    "category": "assertion_failure",
    "exit_code": 1,
    "timed_out": False,
    "duration_s": 2.2,
    "tests_total": 1,
    "tests_failed": 1,
    "failures": [
        {
            "test_name": "test_x",
            "error_type": "assertion_failure",
            "message": "assert 3 == 99\n +  where 3 = add(1, 2)",
            "traceback": (
                "def test_x():\n>       assert add(1, 2) == 99\n"
                "E       assert 3 == 99\n"
                "E        +  where 3 = add(1, 2)\n\n"
                "test_solution.py:5: AssertionError"
            ),
        }
    ],
    "stdout": "F\nFAILED test_solution.py::test_x - assert 3 == 99",
    "stderr": "",
}

ATTRIBUTE_MISUSE_RUN_RESULT = {
    "passed": False,
    "category": "runtime_error",
    "exit_code": 1,
    "timed_out": False,
    "duration_s": 2.4,
    "tests_total": 1,
    "tests_failed": 1,
    "failures": [
        {
            "test_name": "test_x",
            "error_type": "assertion_failure",
            "message": (
                "AttributeError: module 'math' has no attribute 'sqroot'. Did you mean: 'sqrt'?"
            ),
            "traceback": (
                "def test_x():\n>       assert f() == 2\n               ^^^\n\n"
                "test_solution.py:5: \n"
                "_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ "
                "_ _ _ _ _\n\n"
                "    def f():\n>       return math.sqroot(4)\n"
                "               ^^^^^^^^^^^\n"
                "E       AttributeError: module 'math' has no attribute "
                "'sqroot'. Did you mean: 'sqrt'?\n\n"
                "solution.py:5: AttributeError"
            ),
        }
    ],
    "stdout": "FAILED test_solution.py::test_x - AttributeError",
    "stderr": "",
}

KWARG_MISUSE_RUN_RESULT = {
    "passed": False,
    "category": "runtime_error",
    "exit_code": 1,
    "timed_out": False,
    "duration_s": 2.1,
    "tests_total": 1,
    "tests_failed": 1,
    "failures": [
        {
            "test_name": "test_x",
            "error_type": "assertion_failure",
            "message": "TypeError: g() got an unexpected keyword argument 'b'",
            "traceback": (
                "def test_x():\n>       assert f() == 1\n               ^^^\n\n"
                "test_solution.py:5: \n"
                "_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ "
                "_ _ _ _ _\n\n"
                "    def f():\n>       return g(a=1, b=2)\n"
                "               ^^^^^^^^^^^\n"
                "E       TypeError: g() got an unexpected keyword argument 'b'\n\n"
                "solution.py:6: TypeError"
            ),
        }
    ],
    "stdout": "FAILED test_solution.py::test_x - TypeError",
    "stderr": "",
}

IMPORT_NAME_MISUSE_RUN_RESULT = {
    "passed": True,
    "category": "pass",
    "exit_code": 2,
    "timed_out": False,
    "duration_s": 2.3,
    "tests_total": 1,
    "tests_failed": 0,
    "failures": [],
    "stdout": (
        "test_solution.py:1: in <module>\n    from solution import f\n"
        "solution.py:1: in <module>\n    from collections import OrderedDictt\n"
        "E   ImportError: cannot import name 'OrderedDictt' from 'collections'"
    ),
    "stderr": "",
}

MISSING_MODULE_RUN_RESULT = {
    "passed": False,
    "category": "import_error",
    "exit_code": 1,
    "timed_out": False,
    "duration_s": 0.0,
    "tests_total": 0,
    "tests_failed": 0,
    "failures": [
        {
            "test_name": "preflight",
            "error_type": "ImportError",
            "message": "no module named numpyy_missing",
            "traceback": "",
        }
    ],
    "stdout": "",
    "stderr": "",
}

SYNTAX_ERROR_RUN_RESULT = {
    "passed": False,
    "category": "syntax_error",
    "exit_code": 1,
    "timed_out": False,
    "duration_s": 0.0,
    "tests_total": 0,
    "tests_failed": 0,
    "failures": [
        {
            "test_name": "preflight",
            "error_type": "SyntaxError",
            "message": "invalid syntax (<unknown>, line 1)",
            "traceback": "",
        }
    ],
    "stdout": "",
    "stderr": "",
}

NO_TESTS_RUN_RESULT = {
    "passed": False,
    "category": "no_tests",
    "exit_code": 1,
    "timed_out": False,
    "duration_s": 1.9,
    "tests_total": 0,
    "tests_failed": 0,
    "failures": [
        {
            "test_name": "collection",
            "error_type": "NoTests",
            "message": "no tests collected",
            "traceback": "",
        }
    ],
    "stdout": "no tests ran",
    "stderr": "",
}

SANDBOX_ERROR_RUN_RESULT = {
    "passed": False,
    "category": "sandbox_error",
    "exit_code": 1,
    "timed_out": False,
    "duration_s": 0.5,
    "tests_total": 0,
    "tests_failed": 0,
    "failures": [
        {
            "test_name": "setup",
            "error_type": "OSError",
            "message": "permission denied",
            "traceback": "",
        }
    ],
    "stdout": "",
    "stderr": "permission denied",
}

MISSING_DEFINITION_RUN_RESULT = {
    "passed": True,
    "category": "pass",
    "exit_code": 2,
    "timed_out": False,
    "duration_s": 2.3,
    "tests_total": 1,
    "tests_failed": 0,
    "failures": [],
    "stdout": (
        "test_solution.py:1: in <module>\n    from solution import nothere\n"
        "E   ImportError: cannot import name 'nothere' from 'solution'"
    ),
    "stderr": "",
}

NAME_ERROR_IN_TEST_RUN_RESULT = {
    "passed": False,
    "category": "runtime_error",
    "exit_code": 1,
    "timed_out": False,
    "duration_s": 2.1,
    "tests_total": 1,
    "tests_failed": 1,
    "failures": [
        {
            "test_name": "test_x",
            "error_type": "assertion_failure",
            "message": "NameError: name 'undefined_name' is not defined",
            "traceback": (
                "def test_x():\n>       assert f() == undefined_name\n"
                "                      ^^^^^^^^^^^^^^\n"
                "E       NameError: name 'undefined_name' is not defined\n\n"
                "test_solution.py:5: NameError"
            ),
        }
    ],
    "stdout": "FAILED test_solution.py::test_x - NameError",
    "stderr": "",
}

PASSED_RUN_RESULT = {
    "passed": True,
    "category": "pass",
    "exit_code": 0,
    "timed_out": False,
    "duration_s": 1.0,
    "tests_total": 2,
    "tests_failed": 0,
    "failures": [],
    "stdout": "2 passed",
    "stderr": "",
}

CODE = "def add(a, b):\n    return a + b\n"
TESTS = "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"


def make_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {"context_max_chars": 2000, "analyze_with_llm": False}
    base.update(overrides)
    return Settings(**base)


def test_analyze_assertion_failure_keeps_docs_off() -> None:
    node = make_analyze_error_node(None, make_settings())
    state = {
        "run_result": ASSERTION_RUN_RESULT,
        "code": CODE,
        "tests": TESTS,
        "provided_tests": "",
        "attempt": 1,
    }
    snapshot = copy.deepcopy(state)
    update = node(state)

    # Input state unchanged
    assert state == snapshot

    # Check error_analysis keys match ErrorAnalysis exactly
    analysis = update["error_analysis"]
    assert set(analysis.keys()) == set(ErrorAnalysis.__annotations__.keys())

    # JSON serializable
    json.dumps(analysis)

    # Assertion failure basics
    assert analysis["category"] == "assertion_failure"
    assert analysis["needs_docs"] is False
    assert analysis["retrieval_queries"] == []
    assert "assert" in analysis["root_cause"].lower()
    assert analysis["fault"] == "unknown"

    # With provided_tests -> fault "code"
    state2 = {**state, "provided_tests": "def test_given():\n    assert add(1, 1) == 2\n"}
    update2 = node(state2)
    assert update2["error_analysis"]["fault"] == "code"

    # History event
    event = update["history"][0]
    assert event["node"] == "analyze_error"
    assert event["attempt"] == 1
    assert event["summary"]["category"] == "assertion_failure"
    assert event["summary"]["fault"] == "unknown"
    assert event["summary"]["needs_docs"] is False
    assert "root_cause" in event["summary"]
    assert "suspect_symbols" in event["summary"]
    assert event["summary"]["llm"] is False


def test_analyze_api_misuse_yields_docs_and_queries() -> None:
    node = make_analyze_error_node(None, make_settings())
    test_cases = [
        (ATTRIBUTE_MISUSE_RUN_RESULT, "math.sqroot"),
        (KWARG_MISUSE_RUN_RESULT, "g"),
        (IMPORT_NAME_MISUSE_RUN_RESULT, "collections.OrderedDictt"),
    ]

    for run_result, expected_symbol in test_cases:
        state = {"run_result": run_result, "code": CODE, "tests": TESTS, "attempt": 1}
        update = node(state)
        analysis = update["error_analysis"]

        assert analysis["category"] == "api_misuse", f"Failed for {expected_symbol}"
        assert analysis["needs_docs"] is True, f"Failed for {expected_symbol}"
        assert len(analysis["retrieval_queries"]) >= 1, f"Failed for {expected_symbol}"
        assert len(analysis["retrieval_queries"]) <= 3, f"Failed for {expected_symbol}"
        for q in analysis["retrieval_queries"]:
            assert len(q) <= 80, f"Query too long: {q}"
            assert q, "Empty query"
        assert expected_symbol in analysis["suspect_symbols"], f"Missing {expected_symbol}"
        # Invariant: needs_docs implies non-empty queries
        assert analysis["needs_docs"] == (len(analysis["retrieval_queries"]) > 0)


def test_analyze_non_doc_categories_and_fault_attribution() -> None:
    node = make_analyze_error_node(None, make_settings())
    cases = [
        (
            {
                "passed": False,
                "category": "timeout",
                "exit_code": 1,
                "timed_out": True,
                "duration_s": 30.0,
                "tests_total": 1,
                "tests_failed": 0,
                "failures": [
                    {
                        "test_name": "timeout",
                        "error_type": "TimeoutExpired",
                        "message": "exceeded 30s",
                        "traceback": "",
                    }
                ],
                "stdout": "",
                "stderr": "",
            },
            "timeout",
            "code",
            False,
        ),
        (SYNTAX_ERROR_RUN_RESULT, "syntax_error", "code", False),
        (NO_TESTS_RUN_RESULT, "no_tests", "tests", False),
        (SANDBOX_ERROR_RUN_RESULT, "sandbox_error", "unknown", False),
        (MISSING_DEFINITION_RUN_RESULT, "pass", "code", False),
        (NAME_ERROR_IN_TEST_RUN_RESULT, "runtime_error", "unknown", False),
    ]

    for run_result, expected_category, expected_fault, expected_needs_docs in cases:
        state = {"run_result": run_result, "code": CODE, "tests": TESTS, "attempt": 1}
        if expected_fault == "code" and expected_category == "pass":
            # missing_definition: needs provided_tests to be empty to test fault="code"
            state["provided_tests"] = ""
        update = node(state)
        analysis = update["error_analysis"]

        assert analysis["category"] == expected_category, (
            f"Category mismatch for {expected_category}"
        )
        assert analysis["fault"] == expected_fault, f"Fault mismatch for {expected_category}"
        assert analysis["needs_docs"] == expected_needs_docs, (
            f"needs_docs mismatch for {expected_category}"
        )
        assert analysis["retrieval_queries"] == [], (
            f"Queries should be empty for {expected_category}"
        )


def test_analyze_handles_missing_or_passing_run_result() -> None:
    node = make_analyze_error_node(None, make_settings())

    # None run_result
    state = {"run_result": None, "code": CODE, "tests": TESTS, "attempt": 1}
    update = node(state)
    analysis = update["error_analysis"]
    assert analysis["category"] == "unknown"
    assert analysis["needs_docs"] is False

    # Empty run_result
    state = {"run_result": {}, "code": CODE, "tests": TESTS, "attempt": 1}
    update = node(state)
    analysis = update["error_analysis"]
    assert analysis["category"] == "unknown"
    assert analysis["needs_docs"] is False

    # Passing run_result
    state = {"run_result": PASSED_RUN_RESULT, "code": CODE, "tests": TESTS, "attempt": 1}
    update = node(state)
    analysis = update["error_analysis"]
    assert analysis["category"] == "pass"
    assert analysis["needs_docs"] is False

    # Exception text only in stderr (preflight import error)
    run_result = {
        "passed": False,
        "category": "import_error",
        "exit_code": 1,
        "timed_out": False,
        "duration_s": 0.0,
        "tests_total": 0,
        "tests_failed": 0,
        "failures": [],
        "stdout": "",
        "stderr": "ModuleNotFoundError: No module named 'xyz'",
    }
    state = {"run_result": run_result, "code": CODE, "tests": TESTS, "attempt": 1}
    update = node(state)
    analysis = update["error_analysis"]
    assert analysis["category"] == "import_error"
    assert "ModuleNotFoundError" in analysis["root_cause"]

    # Exception text only in stdout (collection error like import_name_misuse)
    run_result2 = {
        "passed": True,
        "category": "pass",
        "exit_code": 2,
        "timed_out": False,
        "duration_s": 2.0,
        "tests_total": 1,
        "tests_failed": 0,
        "failures": [],
        "stdout": "E   ImportError: cannot import name 'X' from 'Y'",
        "stderr": "",
    }
    state2 = {"run_result": run_result2, "code": CODE, "tests": TESTS, "attempt": 1}
    update2 = node(state2)
    analysis2 = update2["error_analysis"]
    # The exception is in stdout, so it should be picked up
    assert analysis2["category"] in ("pass", "api_misuse")
    assert "ImportError" in analysis2["root_cause"]


def test_analyze_llm_refinement_and_fallback() -> None:
    # Tagged LLM reply for refinement
    tagged_reply = (
        "<root_cause>\nLLM says: AttributeError on math.sqroot\n</root_cause>\n"
        "<fix_plan>\nLLM says: use math.sqrt instead\n</fix_plan>"
    )

    # LLM returns tagged reply
    llm = ScriptedLLM([tagged_reply])
    node = make_analyze_error_node(llm, make_settings(analyze_with_llm=True))
    state = {"run_result": ATTRIBUTE_MISUSE_RUN_RESULT, "code": CODE, "tests": TESTS, "attempt": 1}
    update = node(state)

    assert update["error_analysis"]["root_cause"] == "LLM says: AttributeError on math.sqroot"
    assert update["error_analysis"]["fix_plan"] == "LLM says: use math.sqrt instead"
    assert update["token_usage"]["llm_calls"] == 1
    # Deterministic fields unchanged
    assert update["error_analysis"]["needs_docs"] is True
    assert update["error_analysis"]["fault"] == "code"
    assert update["error_analysis"]["suspect_symbols"] == ["math.sqroot"]
    assert update["history"][0]["summary"]["llm"] is True

    # LLMError subclass (DailyQuotaExceeded) -> deterministic result, no token_usage
    llm2 = ScriptedLLM([DailyQuotaExceeded("free-tier daily limit reached")])
    node2 = make_analyze_error_node(llm2, make_settings(analyze_with_llm=True))
    state2 = {"run_result": ATTRIBUTE_MISUSE_RUN_RESULT, "code": CODE, "tests": TESTS, "attempt": 1}
    update2 = node2(state2)
    assert "token_usage" not in update2 or update2.get("token_usage", {}).get("llm_calls", 0) == 0
    assert update2["error_analysis"]["category"] == "api_misuse"
    assert update2["error_analysis"]["needs_docs"] is True

    # Untagged reply -> deterministic text kept
    llm3 = ScriptedLLM(["just some text without tags"])
    node3 = make_analyze_error_node(llm3, make_settings(analyze_with_llm=True))
    state3 = {"run_result": ATTRIBUTE_MISUSE_RUN_RESULT, "code": CODE, "tests": TESTS, "attempt": 1}
    update3 = node3(state3)
    # Should keep deterministic root_cause/fix_plan
    # root_cause uses exception message which doesn't have the dot, but suspect_symbols does
    assert "math.sqroot" in update3["error_analysis"]["suspect_symbols"]
    assert "token_usage" in update3  # call succeeded, so usage recorded

    # analyze_with_llm False -> LLM never called
    llm4 = ScriptedLLM([tagged_reply])
    node4 = make_analyze_error_node(llm4, make_settings(analyze_with_llm=False))
    state4 = {"run_result": ATTRIBUTE_MISUSE_RUN_RESULT, "code": CODE, "tests": TESTS, "attempt": 1}
    update4 = node4(state4)
    assert len(llm4.calls) == 0
    assert update4["error_analysis"]["category"] == "api_misuse"
