"""Pure deterministic error analysis (Task 7).

No I/O, no LLM calls. All logic is synchronous and side-effect free.
"""

from __future__ import annotations

import re
from typing import Any

from evalcode.state import ErrorAnalysis

# Regex patterns derived from the probe output
_EXCEPTION_LINE_RE = re.compile(r"(?:test_solution|solution)\.py:\d+:\s*([A-Za-z_][A-Za-z0-9_\.]*)")
# Match "module 'M' has no attribute 'A'"
_ATTR_ERROR_RE = re.compile(r"module\s+'([^']+)'\s+has\s+no\s+attribute\s+'([^']+)'")
# Match "'T' object has no attribute 'A'" or "type object 'T' has no attribute 'A'"
_OBJ_ATTR_ERROR_RE = re.compile(
    r"(?:'([^']+)'\s+object|type\s+object\s+'([^']+)')\s+has\s+no\s+attribute\s+'([^']+)'"
)
# Match "cannot import name 'N' from 'M'"
_IMPORT_NAME_ERROR_RE = re.compile(r"cannot\s+import\s+name\s+'([^']+)'\s+from\s+'([^']+)'")
# Match "No module named 'M'"
_MODULE_NOT_FOUND_RE = re.compile(r"No\s+module\s+named\s+'([^']+)'")
# TypeError call signature patterns
_TYPE_ERROR_KWARG_RE = re.compile(r"unexpected\s+keyword\s+argument\s+'([^']+)'")
_TYPE_ERROR_REQUIRED_RE = re.compile(r"required\s+(?:positional\s+)?argument")
_TYPE_ERROR_POSITIONAL_RE = re.compile(r"positional\s+argument")
_TYPE_ERROR_TAKES_RE = re.compile(r"takes\s+\d+\s+argument")
_TYPE_ERROR_MULTIPLE_RE = re.compile(r"got\s+multiple\s+values\s+for\s+argument\s+'([^']+)'")
_TYPE_ERROR_INVALID_KWARG_RE = re.compile(r"invalid\s+keyword\s+argument\s+'([^']+'" r")")


def _extract_exception_text(run_result: dict[str, Any] | None) -> str:
    """Extract the best exception text from run_result.

    Priority: failures traceback + message -> stderr -> stdout.
    """
    if not run_result:
        return ""
    # Try failures first
    failures = run_result.get("failures") or []
    for f in failures:
        tb = f.get("traceback") or ""
        msg = f.get("message") or ""
        if tb.strip() or msg.strip():
            return (tb + "\n" + msg).strip()
    # Fall back to stderr
    stderr = run_result.get("stderr") or ""
    if stderr.strip():
        return stderr.strip()
    # Fall back to stdout
    stdout = run_result.get("stdout") or ""
    if stdout.strip():
        return stdout.strip()
    return ""


def _extract_exception_type_and_message(exception_text: str) -> tuple[str, str]:
    """Return (exception_type, message) from exception text.

    Uses the LAST 'file.py:NNN: ExceptionType' or 'ExceptionType: message' line.
    """
    if not exception_text:
        return "Exception", ""

    # Pattern 1: file.py:NNN: ExceptionType[: message]
    # e.g., "solution.py:5: AttributeError" or "test_solution.py:5: AssertionError: message"
    # Must be a valid Python exception type (ends with Error or Exception or is a builtin)
    # Exclude "in <module>" and similar
    file_line_exc_re = re.compile(
        r"(?:test_solution|solution)\.py:\d+:\s*([A-Za-z_][A-Za-z0-9_\.]*?(?:Error|Exception))\s*(?::\s*(.*))?$"
    )
    matches = list(file_line_exc_re.finditer(exception_text))
    if matches:
        last = matches[-1]
        exc_type = last.group(1)
        msg = (last.group(2) or "").strip()
        return exc_type, msg

    # Pattern 2: bare ExceptionType: message (fallback)
    lines = exception_text.splitlines()
    for line in reversed(lines):
        line = line.strip()
        if not line or line.startswith("File ") or line.startswith("Traceback"):
            continue
        m = re.match(r"([A-Za-z_][A-Za-z0-9_\.]*?(?:Error|Exception))\s*:\s*(.*)", line)
        if m:
            return m.group(1), m.group(2).strip()

    # Pattern 3: "E   ExceptionType: message" format (pytest short format)
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("E   "):
            m = re.match(r"E\s+([A-Za-z_][A-Za-z0-9_\.]*?(?:Error|Exception))\s*:\s*(.*)", line)
            if m:
                return m.group(1), m.group(2).strip()

    # Pattern 4: any non-empty line that's not a file reference
    for line in reversed(lines):
        s = line.strip()
        if s and not s.startswith("File ") and not s.startswith("Traceback"):
            # Try to extract exception type
            m = re.match(r"([A-Za-z_][A-Za-z0-9_\.]*?(?:Error|Exception))\s*:\s*(.*)", s)
            if m:
                return m.group(1), m.group(2).strip()
            # Last resort: return "Exception" as type, truncated line as message
            return "Exception", s[:120]

    return "Exception", ""


def _find_deepest_frame_file(exception_text: str) -> str | None:
    """Find the deepest frame file (solution.py or test_solution.py) from traceback.

    Returns the basename of the last file:line occurrence, or None.
    """
    if not exception_text:
        return None
    matches = list(_EXCEPTION_LINE_RE.finditer(exception_text))
    if not matches:
        return None
    last_match = matches[-1]
    # Extract the filename from the match
    line = last_match.group(0)
    # line looks like "solution.py:5: AttributeError" or "test_solution.py:5: NameError"
    if line.startswith("solution.py:"):
        return "solution.py"
    if line.startswith("test_solution.py:"):
        return "test_solution.py"
    return None


def _extract_suspect_symbols(exception_text: str, exception_type: str) -> list[str]:
    """Extract suspect symbols from exception text, max 5, unique, discovery order."""
    if not exception_text:
        return []
    symbols: list[str] = []
    seen: set[str] = set()

    def _add(sym: str) -> None:
        if sym and sym not in seen:
            seen.add(sym)
            symbols.append(sym)

    # "module 'M' has no attribute 'A'" -> "M.A"
    for m in _ATTR_ERROR_RE.finditer(exception_text):
        _add(f"{m.group(1)}.{m.group(2)}")

    # "'T' object has no attribute 'A'" or "type object 'T' has no attribute 'A'" -> "T.A"
    for m in _OBJ_ATTR_ERROR_RE.finditer(exception_text):
        t = m.group(1) or m.group(2)
        if t:
            _add(f"{t}.{m.group(3)}")

    # "cannot import name 'N' from 'M'" -> "M.N"
    for m in _IMPORT_NAME_ERROR_RE.finditer(exception_text):
        _add(f"{m.group(2)}.{m.group(1)}")

    # "No module named 'M'" -> "M"
    for m in _MODULE_NOT_FOUND_RE.finditer(exception_text):
        _add(m.group(1))

    # TypeError patterns - extract function name before "()" plus quoted keyword
    # Look for patterns like "g() got an unexpected keyword argument 'b'"
    for m in _TYPE_ERROR_KWARG_RE.finditer(exception_text):
        # Find the function name before the parenthesis
        before = exception_text[: m.start()]
        func_match = re.search(r"(\w+)\s*\(\)\s*got", before)
        if func_match:
            func_name = func_match.group(1)
            _add(func_name)  # bare function name
            _add(f"{func_name}() kwarg:{m.group(1)}")
        else:
            _add(f"kwarg:{m.group(1)}")

    for m in _TYPE_ERROR_MULTIPLE_RE.finditer(exception_text):
        before = exception_text[: m.start()]
        func_match = re.search(r"(\w+)\s*\(\)\s*got", before)
        if func_match:
            func_name = func_match.group(1)
            _add(func_name)
            _add(f"{func_name}() kwarg:{m.group(1)}")
        else:
            _add(f"kwarg:{m.group(1)}")

    for m in _TYPE_ERROR_INVALID_KWARG_RE.finditer(exception_text):
        before = exception_text[: m.start()]
        func_match = re.search(r"(\w+)\s*\(\)\s*got", before)
        if func_match:
            func_name = func_match.group(1)
            _add(func_name)
            _add(f"{func_name}() kwarg:{m.group(1)}")
        else:
            _add(f"kwarg:{m.group(1)}")

    # Other TypeError patterns (required args, positional, takes N args) - add generic
    if any(
        p.search(exception_text)
        for p in [
            _TYPE_ERROR_REQUIRED_RE,
            _TYPE_ERROR_POSITIONAL_RE,
            _TYPE_ERROR_TAKES_RE,
        ]
    ):
        # Try to find function name
        func_match = re.search(r"(\w+)\s*\(\)", exception_text)
        if func_match:
            func_name = func_match.group(1)
            _add(func_name)
            _add(f"{func_name}() signature")

    return symbols[:5]


def extract_exception(run_result: dict[str, Any] | None) -> tuple[str, str]:
    """Extract (exception_type, message) from a run_result.

    Priority:
    1. failures[0].traceback (if present and yields a real exception type)
    2. failures[0].message (if present)
    3. run_result["stdout"] (collection errors)
    4. run_result["stderr"]

    Delegates to _extract_exception_type_and_message for parsing.
    Never reads failures[].error_type (it is hardcoded in parse_junit).
    """
    if not run_result:
        return "Exception", ""

    # Try failures first - prefer traceback for exception type
    failures = run_result.get("failures") or []
    for f in failures:
        tb = f.get("traceback") or ""
        msg = f.get("message") or ""
        if tb.strip():
            exc_type, exc_msg = _extract_exception_type_and_message(tb)
            # If traceback yields a real exception type (not generic "Exception"), use it
            if exc_type != "Exception" or exc_msg:
                return exc_type, exc_msg
        if msg.strip():
            return _extract_exception_type_and_message(msg.strip())

    # Collection errors: exception only in stdout/stderr
    stdout = run_result.get("stdout") or ""
    if stdout.strip():
        return _extract_exception_type_and_message(stdout.strip())

    stderr = run_result.get("stderr") or ""
    if stderr.strip():
        return _extract_exception_type_and_message(stderr.strip())

    return "Exception", ""


def _build_retrieval_queries(
    exception_text: str, suspect_symbols: list[str], category: str
) -> list[str]:
    """Build retrieval queries (max 3, each <= 80 chars, unique, non-empty)."""
    if category != "api_misuse":
        return []

    queries: list[str] = []
    seen: set[str] = set()

    # First: symbols rewritten as space-separated words
    for sym in suspect_symbols:
        # Convert "math.sqroot" -> "math sqroot"
        words = sym.replace(".", " ").replace("()", "").replace("kwarg:", "").strip()
        if words and words not in seen:
            seen.add(words)
            queries.append(words)

    # Second: exception message truncated to 80 chars
    exc_type, exc_msg = _extract_exception_type_and_message(exception_text)
    if exc_msg:
        msg_query = exc_msg[:80]
        if msg_query not in seen:
            seen.add(msg_query)
            queries.append(msg_query)

    return queries[:3]


def _determine_category(
    run_result: dict[str, Any] | None, exception_text: str, exception_type: str
) -> str:
    """Determine the error category from exception text (not sandbox category)."""
    if not run_result:
        return "unknown"

    passed = run_result.get("passed")
    base_category = run_result.get("category") or "unknown"

    # If truly passed with no exception text, return "pass"
    if passed and not exception_text.strip():
        return "pass"

    # Collection error: sandbox reports category "pass" with exit_code 2,
    # empty failures, but exception text in stdout/stderr.
    # Map from the extracted exception type. Trigger on the sandbox's
    # collection-error signature: category="pass", exit_code=2, empty failures.
    is_collection_error = (
        base_category == "pass"
        and exception_text.strip()
        and run_result.get("exit_code") == 2
        and not (run_result.get("failures") or [])
    )
    if is_collection_error:
        # Map from exception type - never return "pass" for collection errors
        if exception_type == "SyntaxError":
            return "syntax_error"
        if exception_type == "AssertionError":
            return "assertion_failure"
        if exception_type in ("ImportError", "ModuleNotFoundError"):
            # For ImportError, check if it's api_misuse (from non-solution module)
            lower_text = exception_text.lower()
            if "cannot import name" in lower_text:
                m = _IMPORT_NAME_ERROR_RE.search(exception_text)
                if m:
                    from_module = m.group(2)
                    if from_module != "solution":
                        return "api_misuse"
            # "No module named" or from 'solution' -> import_error
            return "import_error"
        # Anything else -> runtime_error (will be checked for api_misuse below)
        base_category = "runtime_error"

    # Derive category from exception text when available
    if exception_text:
        lower_text = exception_text.lower()

        # "No module named X" -> import_error (missing module, not API misuse)
        if "no module named" in lower_text:
            return "import_error"

        # api_misuse overrides base ONLY when ALL conditions met:
        # - AttributeError matching "module 'M' has no attribute 'A'" with M != "solution"
        # - ImportError matching "cannot import name 'N' from 'M'" with M != "solution"
        # - TypeError with call-signature patterns
        # Missing definition from 'solution' (module name "solution") is NOT api_misuse

        # AttributeError on module -> api_misuse (unless module is "solution")
        if exception_type == "AttributeError":
            m = _ATTR_ERROR_RE.search(exception_text)
            if m:
                module_name = m.group(1)
                if module_name != "solution":
                    return "api_misuse"
            # If no match or module is "solution", fall through to base_category

        # ImportError "cannot import name N from M" -> api_misuse (unless M is "solution")
        if exception_type == "ImportError":
            if "cannot import name" in lower_text and "from 'solution'" in lower_text:
                return run_result.get("category") or "import_error"
            if "cannot import name" in lower_text:
                m = _IMPORT_NAME_ERROR_RE.search(exception_text)
                if m:
                    from_module = m.group(2)
                    if from_module != "solution":
                        return "api_misuse"
                # If from 'solution' or no match, fall through
            # Other ImportError -> import_error
            return "import_error"

        # ModuleNotFoundError -> import_error (handled by "no module named" above)
        if exception_type == "ModuleNotFoundError":
            return "import_error"

        # TypeError with call-signature patterns -> api_misuse
        if exception_type == "TypeError":
            signature_patterns = [
                "unexpected keyword argument",
                "required positional argument",
                "positional argument",
                "takes ",
                "got multiple values",
                "invalid keyword argument",
            ]
            if any(p in lower_text for p in signature_patterns):
                return "api_misuse"

        # AssertionError -> assertion_failure
        if exception_type == "AssertionError":
            return "assertion_failure"

        # SyntaxError -> syntax_error
        if exception_type == "SyntaxError":
            return "syntax_error"

        # Timeout comes ONLY from run_result["category"] (sandbox), not from free text
        if exception_type == "TimeoutExpired":
            return "timeout"

    # Return the (possibly mapped) base category
    # Valid categories: 8 sandbox literals + "api_misuse" + "unknown"
    valid_categories = {
        "pass",
        "syntax_error",
        "import_error",
        "runtime_error",
        "assertion_failure",
        "timeout",
        "no_tests",
        "sandbox_error",
        "api_misuse",
        "unknown",
    }
    if base_category in valid_categories:
        return base_category
    return "unknown"


def _determine_fault(
    run_result: dict[str, Any] | None,
    category: str,
    deepest_file: str | None,
    provided_tests: str,
    exception_text: str,
) -> str:
    """Determine fault: 'code', 'tests', or 'unknown'."""
    if not run_result:
        return "unknown"

    # provided_tests non-empty -> "code"
    if provided_tests and provided_tests.strip():
        return "code"

    # category no_tests -> "tests"
    if category == "no_tests":
        return "tests"

    # category timeout -> "code"
    if category == "timeout":
        return "code"

    # syntax_error/import_error with no test_solution.py location -> "code"
    if category in ("syntax_error", "import_error") and deepest_file != "test_solution.py":
        return "code"

    # By deepest frame file
    if deepest_file == "solution.py":
        return "code"
    if deepest_file == "test_solution.py":
        # But if it's a missing definition from solution.py, fault is code
        if "cannot import name" in exception_text and "from 'solution'" in exception_text:
            return "code"
        return "unknown"

    # Missing definition from 'solution' -> "code"
    # Check even when category is "pass" (collection error case)
    if "cannot import name" in exception_text and "from 'solution'" in exception_text:
        return "code"

    return "unknown"


def _make_root_cause(
    category: str, exception_text: str, exception_type: str, suspect_symbols: list[str]
) -> str:
    """Generate root_cause string (<= 200 chars).

    For module-attribute errors (AttributeError "module 'M' has no attribute 'A'"),
    include the dotted name "M.A" (e.g., "math.sqroot") in addition to the
    exception type.
    """
    if not exception_text:
        if category == "timeout":
            return "Execution exceeded the sandbox time limit (possible infinite loop)."
        if category == "no_tests":
            return "No tests were collected or provided."
        if category == "syntax_error":
            return "Syntax error in generated code."
        if category == "import_error":
            return "Import error in generated code."
        return f"{category}: no exception details available"

    # Prefer the concise exception message over the full traceback
    _, exc_msg = _extract_exception_type_and_message(exception_text)

    # For api_misuse with module-attribute errors, include dotted symbol in root_cause
    if category == "api_misuse" and exception_type == "AttributeError":
        # Find the first suspect_symbol that looks like "M.A"
        for sym in suspect_symbols:
            if "." in sym and not sym.endswith("()"):
                # This is likely a module.attribute pair
                if exc_msg:
                    cause = f"{exception_type}: {exc_msg} (symbol: {sym})"
                else:
                    cause = f"{exception_type}: {sym}"
                return cause[:200]

    if exc_msg:
        cause = f"{exception_type}: {exc_msg}"
    else:
        # Fall back to traceback, but truncate intelligently
        cause = f"{exception_type}: {exception_text}"

    return cause[:200]


def _make_fix_plan(
    category: str,
    fault: str,
    suspect_symbols: list[str],
    provided_tests: str,
) -> str:
    """Generate fix_plan string (<= 240 chars)."""
    symbols_str = ", ".join(suspect_symbols) if suspect_symbols else "the failing code"

    if category == "pass":
        return "No fix needed; code passed."

    if category == "api_misuse":
        if fault == "tests":
            return f"Fix the tests: they misuse {symbols_str}."
        return f"Fix the code: correct the API usage for {symbols_str}."

    if category == "assertion_failure":
        if fault == "tests":
            return "Fix the tests: the assertions do not match the intended behavior."
        return f"Fix the code: the logic does not satisfy the assertions (suspect: {symbols_str})."

    if category == "runtime_error":
        if fault == "tests":
            return "Fix the tests: they trigger a runtime error."
        return f"Fix the code: handle the runtime error (suspect: {symbols_str})."

    if category == "timeout":
        return "Fix the code: remove infinite loops or optimize slow operations."

    if category == "no_tests":
        return "Provide valid tests or fix the test collection issue."

    if category == "syntax_error":
        return "Fix the code: correct the syntax error."

    if category == "import_error":
        return f"Fix the code: resolve the import error (missing: {symbols_str})."

    if category == "sandbox_error":
        return "Fix the code: address the sandbox execution error."

    if fault == "tests":
        return "Fix the tests as they are the source of failure."

    return f"Fix the code: address the {category} (suspect: {symbols_str})."


def analyze_run_result(
    run_result: dict[str, Any] | None,
    code: str,
    tests: str,
    provided_tests: str = "",
) -> ErrorAnalysis:
    """Analyze a sandbox run result and return an ErrorAnalysis dict.

    This is the pure deterministic analysis function. It never calls an LLM.
    """
    # Extract exception type and message
    exception_type, exception_message = extract_exception(run_result)
    # Reconstruct exception_text for downstream functions that need full text
    exception_text = _extract_exception_text(run_result)

    # Determine category
    category = _determine_category(run_result, exception_text, exception_type)

    # Find deepest frame file
    deepest_file = _find_deepest_frame_file(exception_text)

    # Extract suspect symbols
    suspect_symbols = _extract_suspect_symbols(exception_text, exception_type)

    # Build retrieval queries
    retrieval_queries = _build_retrieval_queries(exception_text, suspect_symbols, category)

    # Determine needs_docs
    needs_docs = category == "api_misuse" and len(retrieval_queries) > 0

    # Determine fault
    fault = _determine_fault(run_result, category, deepest_file, provided_tests, exception_text)

    # Build root_cause and fix_plan
    root_cause = _make_root_cause(category, exception_text, exception_type, suspect_symbols)
    fix_plan = _make_fix_plan(category, fault, suspect_symbols, provided_tests)

    return ErrorAnalysis(
        category=category,
        root_cause=root_cause,
        fault=fault,
        fix_plan=fix_plan,
        needs_docs=needs_docs,
        retrieval_queries=retrieval_queries,
        suspect_symbols=suspect_symbols,
    )
