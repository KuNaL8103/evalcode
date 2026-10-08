"""Run-tests node (§6 / ARCHITECTURE.md)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from evalcode.config import Settings
from evalcode.sandbox.errors import C_SANDBOX_ERROR
from evalcode.sandbox.runner import run_in_sandbox
from evalcode.state import AgentState, utc_now_iso


def make_run_tests_node(
    settings: Settings,
    sandbox: Callable[[str, str, float, int], dict] = run_in_sandbox,
) -> Callable[[AgentState], dict[str, Any]]:
    def node(state: AgentState) -> dict[str, Any]:
        code = state.get("code", "") or ""
        tests = state.get("tests", "") or ""
        attempt = state.get("attempt", 0)
        if not code.strip() or not tests.strip():
            event = {
                "node": "run_tests",
                "attempt": attempt,
                "ts": utc_now_iso(),
                "summary": {"category": C_SANDBOX_ERROR, "error": "missing code or tests"},
            }
            return {
                "run_result": {
                    "passed": False,
                    "category": C_SANDBOX_ERROR,
                    "exit_code": 1,
                    "timed_out": False,
                    "duration_s": 0.0,
                    "tests_total": 0,
                    "tests_failed": 0,
                    "failures": [
                        {
                            "test_name": "setup",
                            "error_type": "ValueError",
                            "message": "missing code or tests",
                            "traceback": "",
                        }
                    ],
                    "stdout": "",
                    "stderr": "",
                },
                "history": [event],
            }
        result = sandbox(
            code,
            tests,
            timeout_s=float(settings.sandbox_timeout_s),
            mem_mb=int(settings.sandbox_mem_mb),
        )
        summary = {
            "category": result["category"],
            "tests_total": result["tests_total"],
            "tests_failed": result["tests_failed"],
            "duration_s": round(result["duration_s"], 2),
        }
        event = {
            "node": "run_tests",
            "attempt": attempt,
            "ts": utc_now_iso(),
            "summary": summary,
        }
        return {
            "run_result": result,
            "history": [event],
        }

    return node
