"""Prompts for the generate node and analyze_error node (tagged plain-text protocol).

Kept compact on purpose: these run on small free-tier quotas. The system
prompt pins the exact output shape; ``build_generate_messages`` assembles
system + one human message (task, doc context, optional provided tests);
``FORMAT_REMINDER`` is the short strict re-ask sent when a reply fails to
parse.
"""

from __future__ import annotations

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from evalcode.rag.types import RetrievedDoc
from evalcode.state import AgentState

__all__ = [
    "FORMAT_REMINDER",
    "GENERATE_SYSTEM",
    "build_generate_messages",
    "format_context",
    "ANALYZE_SYSTEM",
    "build_analyze_messages",
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

        # Exception line from first failure
        first = failures[0]
        exc_type = first.get("error_type", "")
        msg = first.get("message", "")
        if exc_type or msg:
            parts.append(f"Exception: {exc_type}: {msg}"[:300])

        # Traceback tail
        tb = first.get("traceback", "")
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
