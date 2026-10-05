"""Sandbox error parsing and classification (pure; no subprocess)."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

# Category literals used by classify() and RunResult.
C_SYNTAX_ERROR = "syntax_error"
C_IMPORT_ERROR = "import_error"
C_ASSERTION_FAILURE = "assertion_failure"
C_RUNTIME_EXCEPTION = "runtime_exception"
C_TIMEOUT = "timeout"
C_COLLECTION_ERROR = "collection_error"
C_SANDBOX_ERROR = "sandbox_error"
C_NO_TESTS = "no_tests"


@dataclass
class RunFailure:
    test_name: str
    error_type: str
    message: str = ""
    traceback_tail: str = ""


@dataclass
class RunResult:
    category: str = C_SANDBOX_ERROR
    tests_total: int = 0
    tests_failed: int = 0
    failures: list[RunFailure] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0


def parse_junit(xml_path: str) -> tuple[int, int, list[RunFailure]]:
    """Parse a pytest --junitxml report."""
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return 0, 0, []
    total = 0
    failed = 0
    failures: list[RunFailure] = []
    for suite in root.iter("testsuite"):
        # pytest writes a single testsuite usually.
        for case in suite.iter("testcase"):
            total += 1
            failure = case.find("failure")
            if failure is not None:
                failed += 1
                msg = (failure.get("message") or "").strip()
                tb = (failure.text or "").strip()
                failures.append(
                    RunFailure(
                        test_name=case.get("name") or "unknown",
                        error_type="assertion_failure",
                        message=msg,
                        traceback_tail=tb[-2000:] if tb else "",
                    )
                )
    return total, failed, failures


def last_exception_line(traceback_text: str) -> tuple[str, str]:
    """Return (error_type, message) from the last exception line."""
    lines = traceback_text.splitlines()
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("Traceback (most recent call last):"):
            break
        if line.startswith("File "):
            continue
        # Match 'ExceptionType: message' lines (often after File lines).
        m = re.search(r"([A-Za-z_][A-Za-z0-9_\.]*)(?::\s*(.*))?$", line)
        if m:
            return m.group(1), (m.group(2) or "").strip()
    # Fallback: last non-empty line.
    for line in reversed(lines):
        s = line.strip()
        if s and not s.startswith("File ") and not s.startswith("Traceback"):
            m = re.search(r"([A-Za-z_][A-Za-z0-9_\.]*)(?::\s*(.*))?$", s)
            if m:
                return m.group(1), (m.group(2) or "").strip()
            return s[:120], ""
    return "Exception", ""


def classify(
    traceback_text: str,
    stdout_text: str,
    stderr_text: str,
    tests_total: int,
    tests_failed: int,
) -> str:
    """Classify sandbox result into a category literal."""
    combined = (stdout_text or "") + (stderr_text or "") + (traceback_text or "")
    lower = combined.lower()
    # Pre-flight / syntax handled by runner; here map exceptions.
    if "syntaxerror" in lower or (lower.startswith("  file ") and "syntax error" in lower):
        return C_SYNTAX_ERROR
    if "modulenotfounderror" in lower or "importerror" in lower:
        return C_IMPORT_ERROR
    if "assertionerror" in lower:
        return C_ASSERTION_FAILURE
    if "timeout" in lower or "timed out" in lower:
        return C_TIMEOUT
    if tests_total == 0 and (
        "no tests" in lower or "collected 0" in lower or "test session" in lower
    ):
        return C_NO_TESTS
    if "collection" in lower or "collected" in lower and "error" in lower:
        return C_COLLECTION_ERROR
    if "runtime" in lower or "exception" in lower or "error" in lower:
        return C_RUNTIME_EXCEPTION
    return C_SANDBOX_ERROR


def truncate_output(text: str, max_bytes: int = 8000) -> str:
    """Keep head + tail, truncated to max_bytes."""
    if len(text.encode("utf-8")) <= max_bytes:
        return text
    # Rough byte budget: reserve half for head, half for tail, plus separator.
    half = max_bytes // 2 - 20
    head = text[:half]
    tail = text[-half:]

    # Ensure we don't cut multi-byte chars at boundary.
    def safe(s: str) -> str:
        b = s.encode("utf-8")
        while len(b) > half and b:
            s = s[:-1]
            b = s.encode("utf-8")
        return s

    head = safe(head)
    tail = safe(tail)
    return head + "\n...[truncated]...\n" + tail
