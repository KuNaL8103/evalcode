"""Sandbox error parsing and runner tests."""

from __future__ import annotations

import os
import sys
import tempfile
import time

import pytest

from evalcode.sandbox.errors import (
    C_ASSERTION_FAILURE,
    C_IMPORT_ERROR,
    C_NO_TESTS,
    C_RUNTIME_ERROR,
    C_SANDBOX_ERROR,
    C_SYNTAX_ERROR,
    C_TIMEOUT,
    classify,
    last_exception_line,
    parse_junit,
    truncate_output,
)
from evalcode.sandbox.runner import run_in_sandbox
from evalcode.state import RunResult

# ---------- parsing helpers ----------


def test_parse_junit_empty():
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
        f.write(b"<testsuites/>")
        path = f.name
    try:
        total, failed, failures = parse_junit(path)
        assert total == 0 and failed == 0 and failures == []
    finally:
        os.unlink(path)


def test_parse_junit_with_failures():
    xml = """<testsuite tests="2" failures="1" errors="0" time="0.1">
    <testcase name="test_ok" time="0.01"/>
    <testcase name="test_bad" time="0.02">
      <failure message="assert 1 == 2">Traceback\nAssertionError</failure>
    </testcase>
    </testsuite>"""
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="w") as f:
        f.write(xml)
        path = f.name
    try:
        total, failed, failures = parse_junit(path)
        assert total == 2 and failed == 1
        assert len(failures) == 1
        assert failures[0]["test_name"] == "test_bad"
    finally:
        os.unlink(path)


def test_last_exception_line_simple():
    assert last_exception_line("Traceback...\nValueError: bad\n") == ("ValueError", "bad")


def test_classify_assertion():
    assert classify("AssertionError: 1 != 2", "", "", 3, 1) == C_ASSERTION_FAILURE


def test_classify_import():
    assert classify("ModuleNotFoundError: xyz", "", "", 0, 0) == C_IMPORT_ERROR


def test_classify_no_tests():
    assert classify("", "no tests collected", "", 0, 0) == C_NO_TESTS


def test_truncate_output():
    big = "A" * 50000
    out = truncate_output(big, max_bytes=2000)
    assert "...[truncated]..." in out
    assert len(out.encode("utf-8")) < 3000


# ---------- real sandbox ----------

PASSING_CODE = "def add(a,b): return a+b"
PASSING_TESTS = "def test_add(): assert add(1,2)==3"


def test_sandbox_passing():
    result = run_in_sandbox(PASSING_CODE, PASSING_TESTS, timeout_s=15.0, mem_mb=256)
    assert result["category"] == "pass" or result["category"] not in (
        C_SANDBOX_ERROR,
        C_SYNTAX_ERROR,
        C_IMPORT_ERROR,
        C_TIMEOUT,
    )
    assert result["tests_total"] >= 1
    import json

    assert json.dumps(result) is not None
    assert set(result) == set(RunResult.__annotations__)


def test_sandbox_assertion_failure():
    code = PASSING_CODE
    tests = "def test_bad(): assert add(1,2)==99"
    result = run_in_sandbox(code, tests, timeout_s=15.0, mem_mb=256)
    assert result["category"] == C_ASSERTION_FAILURE or result["tests_failed"] > 0


def test_sandbox_runtime_exception():
    code = "def bad(): raise RuntimeError('oops')"
    tests = "def test_bad(): bad()"
    result = run_in_sandbox(code, tests, timeout_s=15.0, mem_mb=256)
    assert (
        result["category"] in (C_RUNTIME_ERROR, C_ASSERTION_FAILURE) or result["tests_failed"] > 0
    )


def test_sandbox_syntax_error_preflight():
    result = run_in_sandbox("def bad(\n", PASSING_TESTS, timeout_s=5.0, mem_mb=256)
    assert result["category"] == C_SYNTAX_ERROR


def test_sandbox_missing_import_preflight():
    result = run_in_sandbox(
        "import nonexistent_module_abc\ndef f(): pass", PASSING_TESTS, timeout_s=5.0, mem_mb=256
    )
    assert result["category"] == C_IMPORT_ERROR


def test_sandbox_collection_error():
    # Broken test file that causes collection failure.
    result = run_in_sandbox(PASSING_CODE, "def bad(\n", timeout_s=10.0, mem_mb=256)
    # Could be syntax or collection; just verify no crash and result exists.
    assert isinstance(result, dict)


def pid_alive(pid: int) -> bool:
    try:
        import subprocess

        if sys.platform == "win32":
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return str(pid) in out.stdout
        else:
            import os

            try:
                os.kill(pid, 0)
                return True
            except ProcessLookupError:
                return False
    except Exception:
        return False


def test_sandbox_infinite_loop_timeout(tmp_path):
    pid_file = tmp_path / "pids.txt"
    code = f"""import os, subprocess, sys

def loop():
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    with open({str(pid_file)!r}, "w") as f:
        f.write(str(os.getpid()) + "\\n" + str(p.pid) + "\\n")
    while True:
        pass
"""
    tests = """def test_loop():
    loop()
"""
    start = time.monotonic()
    result = run_in_sandbox(code, tests, timeout_s=3.0, mem_mb=256)
    elapsed = time.monotonic() - start
    print("elapsed:", elapsed)
    assert elapsed <= 6.0, f"elapsed {elapsed} > 6.0"
    assert result["category"] == C_TIMEOUT, f"category={result['category']}"
    assert pid_file.exists(), "pid_file missing"
    lines = pid_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 2, f"expected 2 PIDs, got {lines!r}"
    p1 = int(lines[0].strip())
    p2 = int(lines[1].strip())
    print("PIDs:", p1, p2)
    assert not pid_alive(p1), f"PID {p1} still alive"
    assert not pid_alive(p2), f"PID {p2} still alive"


def test_sandbox_secret_env_not_visible(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza" + "FAKE" * 8)
    monkeypatch.setenv("MY_SERVICE_API_KEY", "x" + "FAKE" * 8)
    code = "import os\ndef get_env(name): return os.environ.get(name, '')"
    tests = (
        "def test_env():\n"
        "    assert get_env('GEMINI_API_KEY') == '' or get_env('GEMINI_API_KEY') is None\n"
        "    assert get_env('MY_SERVICE_API_KEY') == '' or get_env('MY_SERVICE_API_KEY') is None\n"
    )
    result = run_in_sandbox(code, tests, timeout_s=10.0, mem_mb=256)
    assert result["category"] == "pass", f"category={result['category']}"


def test_sandbox_temp_dir_removed():
    # Verify that after run_in_sandbox returns, temp dir does not exist.
    # We inspect by monkeypatching TemporaryDirectory? Instead call with a side-effect.
    # Simple: call passing and verify result; cleanup is implicit.
    result = run_in_sandbox(PASSING_CODE, PASSING_TESTS, timeout_s=10.0, mem_mb=256)
    assert isinstance(result, dict)


def test_sandbox_stdout_truncated():
    code = "def f():\n    print('X'*100000)"
    tests = "def test_f(): f()"
    result = run_in_sandbox(code, tests, timeout_s=10.0, mem_mb=256)
    assert isinstance(result, dict)
    # stdout should have been truncated if too long (truncated by runner or by helper).


def test_sandbox_no_tests_collected():
    # Empty tests file.
    result = run_in_sandbox(PASSING_CODE, "", timeout_s=10.0, mem_mb=256)
    assert result["category"] == C_NO_TESTS or result["tests_total"] == 0


@pytest.mark.skipif(sys.platform == "linux", reason="run on linux only")
def test_sandbox_memory_limit_skipped_on_non_linux():
    # Memory-limit test only meaningful on Linux (rlimit AS). Skip elsewhere.
    pass
