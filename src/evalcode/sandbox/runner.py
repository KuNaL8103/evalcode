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
    C_COLLECTION_ERROR,
    C_IMPORT_ERROR,
    C_NO_TESTS,
    C_RUNTIME_EXCEPTION,
    C_SANDBOX_ERROR,
    C_SYNTAX_ERROR,
    C_TIMEOUT,
    RunFailure,
    RunResult,
    classify,
    parse_junit,
    truncate_output,
)

logger = logging.getLogger(__name__)

# Scrubbed env vars (do not pass to sandbox).
_SCRUBBED_PREFIXES = (
    "OPENROUTER_",
    "LANGSMITH_",
    "HF_",
    "ANTHROPIC_",
    "OPENAI_",
)
_SCRUBBED_KEYS = {
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    "LANGSMITH_API_KEY",
    "LANGSMITH_TRACING",
    "HF_TOKEN",
    "HF_HOME",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_API_BASE",
}


def _scrub_env(tmpdir: str) -> dict[str, str]:
    base = dict(os.environ)
    # Remove any key that looks like an API secret.
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
    # Keep minimal Windows vars; point HOME/TEMP at tmpdir.
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
    # Keep SYSTEMROOT if present.
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
) -> RunResult:
    # Pre-flight: syntax.
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return RunResult(
            category=C_SYNTAX_ERROR,
            tests_total=0,
            tests_failed=0,
            failures=[
                RunFailure(test_name="preflight", error_type="SyntaxError", message=str(exc))
            ],
        )
    # Pre-flight: top-level imports.
    for mod in _top_imports(code):
        try:
            if importlib.util.find_spec(mod) is None:
                return RunResult(
                    category=C_IMPORT_ERROR,
                    tests_total=0,
                    tests_failed=0,
                    failures=[
                        RunFailure(
                            test_name="preflight",
                            error_type="ImportError",
                            message=f"no module named {mod}",
                        )
                    ],
                )
        except Exception:
            pass

    # Write temp files.
    result = RunResult()
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write files.
        sol_path = Path(tmpdir) / "solution.py"
        test_path = Path(tmpdir) / "test_solution.py"
        try:
            sol_path.write_text(code, encoding="utf-8")
            # Combine user tests with a minimal pytest wrapper if needed.
            test_path.write_text(tests + "\n", encoding="utf-8")
        except Exception as exc:
            result.category = C_SANDBOX_ERROR
            result.failures = [
                RunFailure(test_name="setup", error_type="OSError", message=str(exc))
            ]
            return result

        env = _scrub_env(tmpdir)
        # Build pytest command.
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
                # POSIX branch: start_new_session + preexec rlimits.
                import resource

                def _set_limits():
                    try:
                        # CPU time ~ timeout_s * 2 (soft guard)
                        resource.setrlimit(
                            resource.RLIMIT_CPU, (int(timeout_s * 2), int(timeout_s * 2))
                        )
                        # Address space ~ mem_mb * 2 (MB -> bytes ~ * 1024*1024)
                        resource.setrlimit(
                            resource.RLIMIT_AS, (mem_mb * 1024 * 1024 * 2, mem_mb * 1024 * 1024 * 2)
                        )
                        # File size
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
            result.duration_s = time.time() - start
            result.stdout = truncate_output(stdout_b.decode("utf-8", errors="replace"))
            result.stderr = truncate_output(stderr_b.decode("utf-8", errors="replace"))
        except subprocess.TimeoutExpired:
            result.duration_s = time.time() - start
            result.category = C_TIMEOUT
            result.failures = [
                RunFailure(
                    test_name="timeout",
                    error_type="TimeoutExpired",
                    message=f"exceeded {timeout_s}s",
                )
            ]
            # Kill process group / tree.
            if proc is not None:
                try:
                    if sys.platform == "win32":
                        # taskkill /F /T /PID <pid>
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
            # Wait for cleanup.
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
        # After completed run.
        if proc is not None and proc.returncode != 0:
            # Could be collection error or test failures.
            pass
        # Parse junit XML if present.
        report_path = Path(tmpdir) / "report.xml"
        if report_path.exists():
            total, failed, failures = parse_junit(str(report_path))
            result.tests_total = total
            result.tests_failed = failed
            result.failures = failures if failures else []
            # If no tests collected but exit non-zero and failures empty.
            if total == 0:
                result.category = C_NO_TESTS
                result.failures = [
                    RunFailure(
                        test_name="collection", error_type="NoTests", message="no tests collected"
                    )
                ]
            elif failed > 0:
                # Classify based on first failure traceback / stdout.
                tb_text = ""
                for f in result.failures:
                    tb_text += f.traceback_tail
                result.category = classify(tb_text, result.stdout, result.stderr, total, failed)
            else:
                result.category = C_SANDBOX_ERROR if (proc.returncode not in (0, 1)) else "passing"
                if result.category not in (
                    C_SANDBOX_ERROR,
                    C_SYNTAX_ERROR,
                    C_IMPORT_ERROR,
                    C_TIMEOUT,
                    C_NO_TESTS,
                    C_COLLECTION_ERROR,
                ):
                    result.category = "passing"
            if result.category == "passing":
                result.category = "passing"
        else:
            # No junit report produced — likely a crash/collection failure.
            combined = result.stdout + result.stderr
            total = 0
            failed = 0
            # Try to detect collection errors from stderr.
            if "test session" in combined.lower():
                total = 0
            result.tests_total = total
            result.tests_failed = failed
            result.category = classify("", result.stdout, result.stderr, total, failed)
            if result.category == C_SANDBOX_ERROR and (
                proc.returncode is not None and proc.returncode not in (0, 1)
            ):
                # Keep sandbox_error.
                pass
        # Ensure category is a valid literal if unclassified.
        valid = {
            C_SYNTAX_ERROR,
            C_IMPORT_ERROR,
            C_ASSERTION_FAILURE,
            C_RUNTIME_EXCEPTION,
            C_TIMEOUT,
            C_COLLECTION_ERROR,
            C_SANDBOX_ERROR,
            C_NO_TESTS,
            "passing",
        }
        if result.category not in valid:
            result.category = C_SANDBOX_ERROR
        # If tests failed but category stayed sandbox_error, refine.
        if result.tests_failed > 0 and result.category == C_SANDBOX_ERROR:
            # Try to infer from failure messages.
            for f in result.failures:
                if f.error_type == "AssertionError":
                    result.category = C_ASSERTION_FAILURE
                    break
                if f.error_type == "ImportError" or f.error_type == "ModuleNotFoundError":
                    result.category = C_IMPORT_ERROR
                    break
        # Clean up temp dir with retry on Windows file-lock errors.
        # TemporaryDirectory already cleaned; retry once if needed (log only).
        try:
            # Already removed by TemporaryDirectory exit; if leftover, try.
            for p in Path(tmpdir).rglob("*"):
                try:
                    if p.is_file():
                        p.unlink()
                    elif p.is_dir():
                        p.rmdir()
                except PermissionError:
                    # Windows file-lock: retry once after short delay.
                    import time as _t

                    _t.sleep(0.2)
                    try:
                        if p.is_file():
                            p.unlink()
                        elif p.is_dir():
                            p.rmdir()
                    except Exception:
                        pass
            Path(tmpdir).rmdir()
        except Exception:
            logger.warning("sandbox temp-dir cleanup failed for %s", tmpdir)
    if sys.platform != "win32" and mem_mb > 0:
        # POSIX-only memory limit already applied; skip rlimit warning.
        pass
    else:
        if mem_mb > 0 and sys.platform == "win32":
            logger.warning("sandbox: rlimits skipped on Windows (wall-clock timeout only)")
    return result
