"""Sandbox error parsing and classification (pure; no subprocess)."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

PASS = "pass"
C_SYNTAX_ERROR = "syntax_error"
C_IMPORT_ERROR = "import_error"
C_RUNTIME_ERROR = "runtime_error"
C_ASSERTION_FAILURE = "assertion_failure"
C_TIMEOUT = "timeout"
C_NO_TESTS = "no_tests"
C_SANDBOX_ERROR = "sandbox_error"


def parse_junit(xml_path: str) -> tuple[int, int, list[dict]]:
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return 0, 0, []
    total = 0
    failed = 0
    failures: list[dict] = []
    for suite in root.iter("testsuite"):
        for case in suite.iter("testcase"):
            total += 1
            failure = case.find("failure")
            if failure is not None:
                failed += 1
                msg = (failure.get("message") or "").strip()
                tb = (failure.text or "").strip()
                failures.append(
                    {
                        "test_name": case.get("name") or "unknown",
                        "error_type": "assertion_failure",
                        "message": msg,
                        "traceback": tb[-2000:] if tb else "",
                    }
                )
    return total, failed, failures


def last_exception_line(traceback_text: str) -> tuple[str, str]:
    lines = traceback_text.splitlines()
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("Traceback (most recent call last):"):
            break
        if line.startswith("File "):
            continue
        m = re.search(r"([A-Za-z_][A-Za-z0-9_\.]*)(?::\s*(.*))?$", line)
        if m:
            return m.group(1), (m.group(2) or "").strip()
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
    combined = (stdout_text or "") + (stderr_text or "") + (traceback_text or "")
    lower = combined.lower()
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
    if "collection" in lower or ("collected" in lower and "error" in lower):
        return C_RUNTIME_ERROR
    if "runtime" in lower or "exception" in lower or "error" in lower:
        return C_RUNTIME_ERROR
    return C_SANDBOX_ERROR


def truncate_output(text: str, max_bytes: int = 8000) -> str:
    if len(text.encode("utf-8")) <= max_bytes:
        return text
    half = max_bytes // 2 - 20

    def safe(s: str) -> str:
        b = s.encode("utf-8")
        while len(b) > half and b:
            s = s[:-1]
            b = s.encode("utf-8")
        return s

    return safe(text[:half]) + "\n...[truncated]...\n" + safe(text[-half:])
