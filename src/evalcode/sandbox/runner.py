"""Subprocess sandbox runner (§6 ARCHITECTURE.md)."""

from __future__ import annotations

import ast
import importlib.util
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from evalcode.config import get_settings
from evalcode.sandbox.errors import (
    C_ASSERTION_FAILURE,
    C_IMPORT_ERROR,
    C_NO_TESTS,
    C_RUNTIME_ERROR,
    C_SANDBOX_ERROR,
    C_SYNTAX_ERROR,
    C_TIMEOUT,
    PASS,
    classify,
    parse_junit,
    truncate_output,
)

logger = logging.getLogger(__name__)

_SCRUBBED_KEYS = {
    "OPENROUTER_API_KEY",
    "LANGSMITH_API_KEY",
    "LANGSMITH_TRACING",
    "HF_TOKEN",
    "HF_HOME",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_API_BASE",
    "GEMINI_API_KEY",
}
_SCRUBBED_PREFIXES = (
    "GEMINI_",
    "GOOGLE_",
    "LANGSMITH_",
    "HF_",
    "ANTHROPIC_",
    "OPENAI_",
)


def _scrub_env(tmpdir: str) -> dict[str, str]:
    base = dict(os.environ)
    for k in list(base):
        if k in _SCRUBBED_KEYS:
            base.pop(k, None)
            continue
        for p in _SCRUBBED_PREFIXES:
            if (
                k.startswith(p)
                or k.endswith("_API_KEY")
                or k.endswith("_TOKEN")
                or k.endswith("_SECRET")
            ):
                base.pop(k, None)
                break
    allowed = {
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "PATH",
        "PATHEXT",
        "WINDIR",
        "COMPUTERNAME",
    }
    for k in list(base):
        if k not in allowed and k not in ("HOME", "USERPROFILE", "TEMP", "TMP"):
            if k != "PATH":
                base.pop(k, None)
    base["HOME"] = tmpdir
    base["USERPROFILE"] = tmpdir
    base["TEMP"] = tmpdir
    base["TMP"] = tmpdir
    if "SYSTEMROOT" not in base:
        base["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
    return base


def _top_imports(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    names = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name.split(".")[0]
                if mod not in names:
                    names.append(mod)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mod = node.module.split(".")[0]
                if mod not in names and mod is not None:
                    names.append(mod)
    return names


def run_in_sandbox(
    code: str,
    tests: str,
    *,
    timeout_s: float = 30.0,
    mem_mb: int = 128,
) -> dict:
    # Pre-flight syntax
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return {
            "passed": False,
            "category": C_SYNTAX_ERROR,
            "exit_code": 1,
            "timed_out": False,
            "duration_s": 0.0,
            "tests_total": 0,
            "tests_failed": 0,
            "failures": [
                {
                    "test_name": "preflight",
                    "error_type": "SyntaxError",
                    "message": str(exc),
                    "traceback": "",
                }
            ],
            "stdout": "",
            "stderr": "",
        }
    # Pre-flight imports
    for mod in _top_imports(code):
        try:
            if importlib.util.find_spec(mod) is None:
                return {
                    "passed": False,
                    "category": C_IMPORT_ERROR,
                    "exit_code": 1,
                    "timed_out": False,
                    "duration_s": 0.0,
                    "tests_total": 0,
                    "tests_failed": 0,
                    "failures": [
                        {
                            "test_name": "preflight",
                            "error_type": "ImportError",
                            "message": f"no module named {mod}",
                            "traceback": "",
                        }
                    ],
                    "stdout": "",
                    "stderr": "",
                }
        except Exception:
            pass

    result: dict = {
        "passed": False,
        "category": C_SANDBOX_ERROR,
        "exit_code": 1,
        "timed_out": False,
        "duration_s": 0.0,
        "tests_total": 0,
        "tests_failed": 0,
        "failures": [],
        "stdout": "",
        "stderr": "",
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        sol_path = Path(tmpdir) / "solution.py"
        test_path = Path(tmpdir) / "test_solution.py"
        try:
            sol_path.write_text(code, encoding="utf-8")
            test_path.write_text(tests + "\n", encoding="utf-8")
        except Exception as exc:
            result["failures"] = [
                {
                    "test_name": "setup",
                    "error_type": "OSError",
                    "message": str(exc),
                    "traceback": "",
                }
            ]
            return result

        env = _scrub_env(tmpdir)
        cmd = [
            sys.executable,
            "-E",
            "-s",
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "--junitxml=report.xml",
            "test_solution.py",
        ]
        start = time.time()
        proc = None
        try:
            if sys.platform == "win32":
                proc = subprocess.Popen(
                    cmd,
                    cwd=tmpdir,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                )
            else:
                import resource

                def _set_limits():
                    try:
                        resource.setrlimit(
                            resource.RLIMIT_CPU, (int(timeout_s * 2), int(timeout_s * 2))
                        )
                        resource.setrlimit(
                            resource.RLIMIT_AS, (mem_mb * 1024 * 1024 * 2, mem_mb * 1024 * 1024 * 2)
                        )
                        resource.setrlimit(
                            resource.RLIMIT_FSIZE, (50 * 1024 * 1024, 50 * 1024 * 1024)
                        )
                    except Exception:
                        pass

                proc = subprocess.Popen(
                    cmd,
                    cwd=tmpdir,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                    preexec_fn=_set_limits,
                )
            stdout_b, stderr_b = proc.communicate(timeout=timeout_s)
            result["duration_s"] = time.time() - start
            result["stdout"] = truncate_output(stdout_b.decode("utf-8", errors="replace"))
            result["stderr"] = truncate_output(stderr_b.decode("utf-8", errors="replace"))
        except subprocess.TimeoutExpired:
            result["duration_s"] = time.time() - start
            result["category"] = C_TIMEOUT
            result["timed_out"] = True
            result["failures"] = [
                {
                    "test_name": "timeout",
                    "error_type": "TimeoutExpired",
                    "message": f"exceeded {timeout_s}s",
                    "traceback": "",
                }
            ]
            if proc is not None:
                try:
                    if sys.platform == "win32":
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                            capture_output=True,
                            timeout=10,
                        )
                    else:
                        import os as _os

                        try:
                            _os.killpg(proc.pid, 9)
                        except ProcessLookupError:
                            pass
                except Exception:
                    pass
                try:
                    proc.wait(timeout=5)
                except Exception:
                    pass
            return result
        finally:
            if proc is not None and proc.poll() is None:
                try:
                    if sys.platform == "win32":
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                            capture_output=True,
                            timeout=5,
                        )
                    else:
                        import os as _os

                        try:
                            _os.killpg(proc.pid, 9)
                        except ProcessLookupError:
                            pass
                except Exception:
                    pass
        report_path = Path(tmpdir) / "report.xml"
        if report_path.exists():
            total, failed, failures = parse_junit(str(report_path))
            result["tests_total"] = total
            result["tests_failed"] = failed
            result["failures"] = failures if failures else []
            if total == 0:
                result["category"] = C_NO_TESTS
                result["failures"] = [
                    {
                        "test_name": "collection",
                        "error_type": "NoTests",
                        "message": "no tests collected",
                        "traceback": "",
                    }
                ]
                result["passed"] = False
            elif failed > 0:
                tb_text = "".join(f.get("traceback", "") for f in result["failures"])
                result["category"] = classify(
                    tb_text, result["stdout"], result["stderr"], total, failed
                )
                result["passed"] = False
            else:
                result["category"] = PASS
                result["passed"] = True
                result["exit_code"] = proc.returncode if proc and proc.returncode is not None else 0
        else:
            combined = result["stdout"] + result["stderr"]
            total = 0
            result["tests_total"] = total
            result["tests_failed"] = 0
            result["category"] = classify("", result["stdout"], result["stderr"], total, 0)
            if (
                result["category"] == C_SANDBOX_ERROR
                and proc is not None
                and proc.returncode not in (0, 1)
            ):
                pass
        if result["category"] not in (
            PASS,
            C_SYNTAX_ERROR,
            C_IMPORT_ERROR,
            C_RUNTIME_ERROR,
            C_ASSERTION_FAILURE,
            C_TIMEOUT,
            C_NO_TESTS,
            C_SANDBOX_ERROR,
        ):
            result["category"] = C_SANDBOX_ERROR
        if result["tests_failed"] > 0 and result["category"] == C_SANDBOX_ERROR:
            for f in result["failures"]:
                if f.get("error_type") == "AssertionError":
                    result["category"] = C_ASSERTION_FAILURE
                    break
                if f.get("error_type") in ("ImportError", "ModuleNotFoundError"):
                    result["category"] = C_IMPORT_ERROR
                    break
        if sys.platform == "win32" and mem_mb > 0:
            logger.warning("sandbox: rlimits skipped on Windows (wall-clock timeout only)")
    return result
