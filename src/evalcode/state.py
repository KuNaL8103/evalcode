"""LangGraph state schema (ARCHITECTURE §3.2).

All nested values are plain ``TypedDict`` / JSON-serializable so LangGraph
checkpoints serialize cleanly. Do not name any class ``Test*`` here: pytest
would try to collect it.
"""

from __future__ import annotations

import operator
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, TypedDict

from evalcode.config import Settings
from evalcode.rag.types import RetrievedDoc

__all__ = [
    "AgentState",
    "ErrorAnalysis",
    "RunFailure",
    "RunResult",
    "initial_state",
    "StepEvent",
    "TokenUsage",
    "merge_usage",
    "utc_now_iso",
]


class TokenUsage(TypedDict, total=False):
    """Token/cost bookkeeping, accumulated via the ``merge_usage`` reducer."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    llm_calls: int
    api_retries: int
    estimated_calls: int  # calls where the provider gave no usage metadata
    wait_s: float  # time spent in backoff/throttle


class RunFailure(TypedDict):
    test_name: str
    error_type: str
    message: str
    traceback: str


class RunResult(TypedDict):
    passed: bool
    category: Literal[
        "pass",
        "syntax_error",
        "import_error",
        "runtime_error",
        "assertion_failure",
        "timeout",
        "no_tests",
        "sandbox_error",
    ]
    exit_code: int
    timed_out: bool
    duration_s: float
    tests_total: int
    tests_failed: int
    failures: list[RunFailure]
    stdout: str  # truncated
    stderr: str  # truncated


class ErrorAnalysis(TypedDict):
    category: str
    root_cause: str
    fault: Literal["code", "tests", "unknown"]
    fix_plan: str
    needs_docs: bool
    retrieval_queries: list[str]
    suspect_symbols: list[str]


class StepEvent(TypedDict):
    node: str
    attempt: int
    ts: str
    summary: dict[str, Any]


class AgentState(TypedDict, total=False):
    # input
    task_id: str
    task: str
    provided_tests: str | None  # if set, tests are immutable ground truth
    auto_approve: bool  # skip human_review (CI / eval)
    # budgets
    attempt: int  # total generations so far (monotonic)
    retries_used: int
    max_retries: int  # automatic revisions in the current round
    human_rounds: int
    max_human_rounds: int
    # RAG
    retrieval_queries: list[str]
    retrieved_docs: list[RetrievedDoc]
    # artifacts
    code: str
    tests: str
    explanation: str
    # feedback
    run_result: RunResult | None
    error_analysis: ErrorAnalysis | None
    human_decision: Literal["approve", "reject", "edit"] | None
    human_feedback: str | None
    # outcome
    status: Literal["running", "awaiting_review", "approved", "failed"]
    final_code: str | None
    failure_reason: str | None
    # bookkeeping (reducers)
    history: Annotated[list[StepEvent], operator.add]
    token_usage: Annotated[TokenUsage, merge_usage]


def merge_usage(a: TokenUsage | None, b: TokenUsage | None) -> TokenUsage:
    """Reducer that sums all numeric usage fields from two partial usages.

    Tolerates ``None`` and missing keys: a ``None`` side contributes nothing,
    and a key whose value is ``None`` on both sides is dropped. Used as the
    LangGraph reducer for ``AgentState.token_usage``.
    """
    if a is None:
        return dict(b) if b else {}
    if b is None:
        return dict(a)
    merged: dict[str, Any] = {}
    for key in set(a) | set(b):
        value_a = a.get(key)
        value_b = b.get(key)
        if value_a is None and value_b is None:
            continue
        merged[key] = (value_a or 0) + (value_b or 0)
    return merged


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string (for ``StepEvent.ts``)."""
    return datetime.now(UTC).isoformat()


def initial_state(
    task: str,
    settings: Settings,
    *,
    task_id: str | None = None,
    provided_tests: str | None = None,
    auto_approve: bool = False,
) -> AgentState:
    """Build the initial AgentState for a new run."""
    import uuid

    return {
        "task_id": task_id or uuid.uuid4().hex,
        "task": task,
        "provided_tests": provided_tests,
        "auto_approve": auto_approve,
        "attempt": 0,
        "retries_used": 0,
        "max_retries": settings.max_retries,
        "max_human_rounds": settings.max_human_rounds,
        "human_rounds": 0,
        "retrieval_queries": [],
        "retrieved_docs": [],
        "code": "",
        "tests": "",
        "explanation": "",
        "run_result": None,
        "error_analysis": None,
        "human_decision": None,
        "human_feedback": None,
        "status": "running",
        "final_code": None,
        "failure_reason": None,
        "history": [],
        "token_usage": {},
    }
