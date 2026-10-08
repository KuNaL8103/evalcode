"""run_tests node tests."""

from __future__ import annotations

from evalcode.config import get_settings
from evalcode.nodes.run_tests import make_run_tests_node
from evalcode.sandbox.errors import C_SANDBOX_ERROR


def test_run_tests_node_success():
    node = make_run_tests_node(get_settings())
    result = node(
        {
            "code": "def add(a,b): return a+b",
            "tests": "from solution import add\n\ndef test_add(): assert add(1,2)==3",
        }
    )
    assert "run_result" in result
    assert isinstance(result["run_result"], dict)
    assert result["run_result"]["category"] == "pass" or result["run_result"]["tests_total"] >= 1
    # Returns exactly ONE new history event (not cumulative)
    assert len(result["history"]) == 1
    event = result["history"][0]
    assert set(event.keys()) == {"node", "attempt", "ts", "summary"}
    assert event["node"] == "run_tests"

    # Passing a state that already holds an unrelated history list does NOT
    # make the returned history longer than 1 (second call, same node)
    result2 = node(
        {
            "code": "def add(a,b): return a+b",
            "tests": "from solution import add\n\ndef test_add(): assert add(1,2)==3",
            "history": [{"node": "generate", "attempt": 1, "ts": "x", "summary": {}}],
        }
    )
    assert len(result2["history"]) == 1


def test_run_tests_node_missing_code():
    node = make_run_tests_node(get_settings())
    result = node({"code": "", "tests": "def test_x(): pass"})
    assert result["run_result"]["category"] == C_SANDBOX_ERROR
    assert result["run_result"]["passed"] is False
    assert len(result["history"]) == 1
    event = result["history"][0]
    assert event["node"] == "run_tests"
    assert event["summary"]["category"] == C_SANDBOX_ERROR
    assert event["summary"]["error"] == "missing code or tests"


def test_run_tests_node_missing_tests():
    node = make_run_tests_node(get_settings())
    result = node({"code": "def f(): pass", "tests": ""})
    assert isinstance(result["run_result"], dict)
    assert result["run_result"]["category"] == C_SANDBOX_ERROR
    assert result["run_result"]["passed"] is False
    assert len(result["history"]) == 1
