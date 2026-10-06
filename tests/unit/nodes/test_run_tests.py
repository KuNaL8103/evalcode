"""run_tests node tests."""

from __future__ import annotations

from evalcode.config import get_settings
from evalcode.nodes.run_tests import make_run_tests_node
from evalcode.sandbox.errors import C_SANDBOX_ERROR


def test_run_tests_node_success():
    node = make_run_tests_node(get_settings())
    result = node(
        {"code": "def add(a,b): return a+b", "tests": "def test_add(): assert add(1,2)==3"}
    )
    assert "run_result" in result
    assert isinstance(result["run_result"], dict)
    assert result["run_result"]["category"] == "pass" or result["run_result"]["tests_total"] >= 1
    assert len(result.get("history", [])) >= 1


def test_run_tests_node_missing_code():
    node = make_run_tests_node(get_settings())
    result = node({"code": "", "tests": "def test_x(): pass"})
    assert result["run_result"]["category"] == C_SANDBOX_ERROR or "failed" in (
        result.get("history") or [{}]
    )[0].get("status", "")


def test_run_tests_node_missing_tests():
    node = make_run_tests_node(get_settings())
    result = node({"code": "def f(): pass", "tests": ""})
    assert isinstance(result["run_result"], dict)
