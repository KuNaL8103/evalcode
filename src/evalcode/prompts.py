"""Prompts for the generate node and analyze_error node (tagged plain-text protocol).

Kept compact on purpose: these run on small free-tier quotas. The system
prompt pins the exact output shape; ``build_generate_messages`` assembles
system + one human message (task, doc context, optional provided tests);
``FORMAT_REMINDER`` is the short strict re-ask sent when a reply fails to
parse.
"""

from __future__ import annotations

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from evalcode.analysis import extract_exception
from evalcode.rag.types import RetrievedDoc
from evalcode.state import AgentState

__all__ = [
    "FORMAT_REMINDER",
    "GENERATE_SYSTEM",
    "build_generate_messages",
    "format_context",
    "ANALYZE_SYSTEM",
    "build_analyze_messages",
    "REVISE_SYSTEM",
    "REVISE_INSTRUCTION",
    "build_revise_messages",
    "REWRITE_SYSTEM",
    "build_rewrite_messages",
]

GENERATE_SYSTEM = """\
You are a Python code-generation agent. You write one module, solution.py,
and, unless told otherwise, pytest tests in test_solution.py.

REPLY WITH EXACTLY THESE TAGGED BLOCKS AND NOTHING ELSE:
<explanation>
1-3 sentences: what you implemented and which documented APIs you used.
</explanation>
<code>
complete solution.py source
</code>
<tests>
complete test_solution.py source (pytest, using "from solution import ...")
</tests>
<docs_used>
comma-separated ids of the documentation blocks you actually used, if any
</docs_used>

Rules:
- If the task section includes "PROVIDED TESTS", output ONLY <explanation>
  and <code> (no <tests> section): your code must pass those tests.
- Use only the Python standard library and already-installed packages.
- Prefer the APIs shown in the documentation context. Never invent APIs,
  modules, or parameters; if a needed API is not documented, say so in
  <explanation>.
- No network access, no input(), no writing files outside a temporary dir.
- Tests must be deterministic and finish in under 5 seconds.
- Do not wrap the code or tests in markdown fences inside the tags.
"""

FORMAT_REMINDER = (
    "Your last reply did not follow the required format. Reply again with EXACTLY:\n"
    "<explanation>...</explanation>\n"
    "<code>... complete solution.py source ...</code>\n"
    "<tests>... complete test_solution.py source ...</tests>\n"
    "Omit the <tests> block only if the task included PROVIDED TESTS. "
    "Nothing outside these tags; no markdown fences inside them; no apologies."
)


def format_context(docs: list[RetrievedDoc], max_chars: int) -> str:
    """Render doc blocks as ``[doc:<id>] <import_path>`` + text, joined on a
    blank line.

    Truncates by whole block (never a partial block) so the total length is
    at most ``max_chars``. Returns "" when there are no docs or none fit.
    """
    if not docs:
        return ""
    blocks: list[str] = []
    total = 0
    for doc in docs:
        block = f"[doc:{doc['id']}] {doc['import_path']}\n{doc['text']}"
        cost = len(block) + (2 if blocks else 0)  # the blank-line separator
        if total + cost > max_chars:
            break
        blocks.append(block)
        total += cost
    return "\n\n".join(blocks)


def build_generate_messages(
    state: AgentState, *, context_max_chars: int = 6000
) -> list[BaseMessage]:
    """System prompt + one human message: task, capped doc context, and
    (when set) the provided-tests instruction."""
    task = state.get("task") or ""
    docs = state.get("retrieved_docs") or []
    provided = (state.get("provided_tests") or "").strip()

    parts = [f"Task:\n{task}"]
    context = format_context(docs, context_max_chars)
    if context:
        parts.append(f"Documentation context (prefer these documented APIs):\n{context}")
    if provided:
        parts.append(
            "PROVIDED TESTS — these tests are fixed. Do NOT output a <tests> section; "
            "output only <explanation> and <code>, and make the code pass them:\n" + provided
        )
    return [SystemMessage(content=GENERATE_SYSTEM), HumanMessage(content="\n\n".join(parts))]


ANALYZE_SYSTEM = """\
You are a Python error-analysis agent. Given a failed code execution, you
produce a concise root cause and a concrete fix plan.

REPLY WITH EXACTLY THESE TAGGED BLOCKS AND NOTHING ELSE:
<root_cause>
One sentence: the fundamental reason the code failed (exception type + key detail).
</root_cause>
<fix_plan>
One sentence: the minimal change to fix the failure. If tests are provided
ground truth, say so explicitly.
</fix_plan>
"""


def build_analyze_messages(
    state: AgentState, *, failure_summary_chars: int = 1500
) -> list[BaseMessage]:
    """Build messages for the analyze_error LLM call.

    The human message contains a compact failure summary (category, exception
    line, traceback tail, stdout/stderr tails) capped at ``failure_summary_chars``.
    """
    run_result = state.get("run_result") or {}
    error_analysis = state.get("error_analysis") or {}
    provided = (state.get("provided_tests") or "").strip()

    category = error_analysis.get("category") or run_result.get("category") or "unknown"
    failures = run_result.get("failures") or []
    stdout = run_result.get("stdout") or ""
    stderr = run_result.get("stderr") or ""

    # Build compact failure summary
    parts: list[str] = [f"Category: {category}"]

    if failures:
        # Up to 5 failing test names
        test_names = [f.get("test_name", "unknown") for f in failures[:5]]
        parts.append(f"Failing tests: {', '.join(test_names)}")

    # Exception line from extract_exception (never uses failures[].error_type)
    exc_type, exc_msg = extract_exception(run_result)
    if exc_type or exc_msg:
        parts.append(f"Exception: {exc_type}: {exc_msg}"[:300])

    # Traceback tail (from first failure if available)
    if failures:
        tb = failures[0].get("traceback", "")
        if tb:
            tail = tb[-1500:]
            parts.append(f"Traceback (tail):\n{tail}")

    if stdout:
        parts.append(f"Stdout (tail):\n{stdout[-500:]}")
    if stderr:
        parts.append(f"Stderr (tail):\n{stderr[-500:]}")

    # Deterministic diagnosis from error_analysis
    root_cause = error_analysis.get("root_cause", "")
    fix_plan = error_analysis.get("fix_plan", "")
    fault = error_analysis.get("fault", "unknown")
    suspects = error_analysis.get("suspect_symbols", [])

    parts.append(f"Deterministic diagnosis:\n  root_cause: {root_cause}")
    parts.append(f"  fix_plan: {fix_plan}")
    parts.append(f"  fault: {fault}")
    if suspects:
        parts.append(f"  suspect_symbols: {', '.join(suspects)}")

    if provided:
        parts.append("PROVIDED TESTS are fixed ground truth — do not modify them.")

    human_content = "\n\n".join(parts)
    # Cap total length
    if len(human_content) > failure_summary_chars:
        human_content = human_content[:failure_summary_chars] + "\n...[truncated]"

    return [SystemMessage(content=ANALYZE_SYSTEM), HumanMessage(content=human_content)]


REVISE_SYSTEM = GENERATE_SYSTEM  # Reuse the same system prompt

REVISE_INSTRUCTION = """\
Return the COMPLETE corrected solution.py and test_solution.py in the exact
tagged format. Use <explanation>, <code>, <tests>, and <docs_used> blocks.
If PROVIDED TESTS were given, omit the <tests> block (they are fixed ground truth).
"""


def build_revise_messages(state: AgentState, *, context_max_chars: int = 6000) -> list[BaseMessage]:
    """Build messages for the revise LLM call.

    The user message contains:
    - Task
    - Current solution.py (uncapped)
    - Current test_solution.py (uncapped, labeled as FIXED ground truth if provided_tests)
    - Failure report (category, up to 5 failing test names, exception line <= 300 chars,
      traceback tail <= 1500, stdout/stderr tails <= 500)
    - Diagnosis (root_cause, fix_plan, fault, suspect_symbols)
    - Human feedback if present (<= 1000 chars)
    - Reference docs via format_context (only when docs exist)
    - Previous failed attempts: up to 3 history events with node == "analyze_error"
      and attempt < current attempt
    - Instruction to return complete corrected code/tests in tagged format
    """
    run_result = state.get("run_result") or {}
    error_analysis = state.get("error_analysis") or {}
    code = state.get("code") or ""
    tests = state.get("tests") or ""
    provided = (state.get("provided_tests") or "").strip()
    human_feedback = state.get("human_feedback") or ""
    attempt = state.get("attempt", 0)
    retrieved_docs = state.get("retrieved_docs") or []
    history = state.get("history") or []

    category = error_analysis.get("category") or run_result.get("category") or "unknown"
    failures = run_result.get("failures") or []
    stdout = run_result.get("stdout") or ""
    stderr = run_result.get("stderr") or ""

    parts: list[str] = [f"Task:\n{state.get('task', '')}"]

    # Current solution.py (uncapped)
    parts.append(f"Current solution.py:\n{code}")

    # Current test_solution.py (labeled if provided_tests)
    if provided:
        parts.append("Current test_solution.py (FIXED ground truth — do not modify):\n" + provided)
    else:
        parts.append(f"Current test_solution.py:\n{tests}")

    # Human feedback present AND run_result.passed is True -> human rejected a passing solution
    human_rejected_passing = human_feedback.strip() and run_result.get("passed") is True

    if not human_rejected_passing:
        # Failure report (omitted when human rejected a passing solution)
        parts.append(f"Category: {category}")
        if failures:
            test_names = [f.get("test_name", "unknown") for f in failures[:5]]
            parts.append(f"Failing tests: {', '.join(test_names)}")

        # Exception line from extract_exception (never uses failures[].error_type)
        exc_type, exc_msg = extract_exception(run_result)
        if exc_type or exc_msg:
            parts.append(f"Exception: {exc_type}: {exc_msg}"[:300])

        # Traceback tail (from first failure if available)
        if failures:
            tb = failures[0].get("traceback", "")
            if tb:
                parts.append(f"Traceback (tail):\n{tb[-1500:]}")
        if stdout:
            parts.append(f"Stdout (tail):\n{stdout[-500:]}")
        if stderr:
            parts.append(f"Stderr (tail):\n{stderr[-500:]}")

        # Diagnosis (omitted when human rejected a passing solution)
        root_cause = error_analysis.get("root_cause", "")
        fix_plan = error_analysis.get("fix_plan", "")
        fault = error_analysis.get("fault", "unknown")
        suspects = error_analysis.get("suspect_symbols", [])

        parts.append(
            f"Diagnosis:\n  root_cause: {root_cause}\n  fix_plan: {fix_plan}\n  fault: {fault}"
        )
        if suspects:
            parts.append(f"  suspect_symbols: {', '.join(suspects)}")
    else:
        # Human rejected a PASSING solution: omit failure report and diagnosis
        parts.append(
            "The tests currently PASS; a human reviewer rejected the solution. "
            "Address the feedback below."
        )

    # Human feedback
    if human_feedback.strip():
        parts.append(f"Human feedback:\n{human_feedback.strip()[:1000]}")

    # Reference docs
    if retrieved_docs:
        doc_context = format_context(retrieved_docs, context_max_chars)
        if doc_context:
            parts.append(f"Reference documentation:\n{doc_context}")

    # Previous failed attempts (up to 3 analyze_error events with attempt < current)
    prev_attempts = [
        h for h in history if h.get("node") == "analyze_error" and h.get("attempt", 0) < attempt
    ][-3:]
    if prev_attempts:
        lines = []
        for h in prev_attempts:
            a = h.get("attempt", 0)
            cat = h.get("summary", {}).get("category", "unknown")
            rc = h.get("summary", {}).get("root_cause", "")[:150]
            lines.append(f"  attempt {a}: {cat} - {rc}")
        parts.append("Previous failed attempts:\n" + "\n".join(lines))

    # Instruction
    parts.append(REVISE_INSTRUCTION)

    return [SystemMessage(content=REVISE_SYSTEM), HumanMessage(content="\n\n".join(parts))]


# --------------------------------------------------------------------------- #
# Query rewrite prompt (Task 10)
# --------------------------------------------------------------------------- #

REWRITE_SYSTEM = """\
You are a query rewriter for a Python documentation search engine. Given a
coding task, produce 2-4 short, API-oriented search queries that will find
the most relevant standard-library documentation.

REPLY WITH EXACTLY THIS TAGGED BLOCK AND NOTHING ELSE:
<queries>
one query per line
</queries>

Rules:
- Each query <= 80 characters.
- Focus on module names, function names, class names, and method names.
- No natural language questions; use keyword phrases like "json loads" or
  "pathlib Path mkdir".
- Maximum 4 queries.
- Do NOT include markdown fences or any text outside the <queries> block.
"""


def build_rewrite_messages(task: str) -> list[BaseMessage]:
    """Build messages for the query rewrite LLM call."""
    return [
        SystemMessage(content=REWRITE_SYSTEM),
        HumanMessage(content=f"Task:\n{task}"),
    ]
