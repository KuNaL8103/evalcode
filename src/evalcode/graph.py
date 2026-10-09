"""LangGraph wiring for the evalcode agent (Task 9).

This module compiles the StateGraph with nodes: generate, run_tests,
analyze_error, revise, human_review, finalize, fail. Routers use
run_result["passed"] exclusively. No retrieve node yet (Task 10).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from evalcode.config import Settings, get_settings
from evalcode.llm import TextLLM, build_llm_client
from evalcode.nodes.analyze_error import make_analyze_error_node
from evalcode.nodes.generate import make_generate_node
from evalcode.nodes.human_review import human_review_node
from evalcode.nodes.revise import make_revise_node
from evalcode.nodes.run_tests import make_run_tests_node
from evalcode.nodes.terminal import fail_node, finalize_node
from evalcode.sandbox.runner import run_in_sandbox
from evalcode.state import AgentState

__all__ = [
    "Dependencies",
    "default_dependencies",
    "route_after_tests",
    "route_after_llm",
    "route_after_human",
    "build_graph",
    "run_task",
    "stream_task",
    "initial_state",
    "pending_review",
    "resume_task",
    "RECURSION_LIMIT",
]

RECURSION_LIMIT = 100


@dataclass(frozen=True)
class Dependencies:
    """All external dependencies for the graph."""

    llm: TextLLM
    settings: Settings
    sandbox: Callable[[str, str, float, int], dict] = run_in_sandbox


def default_dependencies(settings: Settings | None = None) -> Dependencies:
    """Build real dependencies from settings (the only place that builds LLMClient)."""
    if settings is None:
        settings = get_settings()
    llm = build_llm_client(settings)
    return Dependencies(llm=llm, settings=settings, sandbox=run_in_sandbox)


# --------------------------------------------------------------------------- #
# Routers (pure, module-level, exported for testing)
# --------------------------------------------------------------------------- #


def route_after_tests(state: AgentState) -> str:
    """Route after run_tests: human_review | finalize | analyze_error | fail.

    If tests passed and auto_approve is False, route to human_review.
    If tests passed and auto_approve is True, route to finalize.
    If tests failed and retries remain, route to analyze_error.
    Otherwise route to fail.
    """
    run_result = state.get("run_result")
    if not run_result:
        return "fail"
    if run_result.get("passed") is True:
        if state.get("auto_approve") is True:
            return "finalize"
        return "human_review"
    retries_used = state.get("retries_used", 0)
    max_retries = state.get("max_retries", 0)
    if retries_used < max_retries:
        return "analyze_error"
    return "fail"


def route_after_llm(state: AgentState) -> str:
    """Route after generate/revise: fail | run_tests."""
    if state.get("status") == "failed":
        return "fail"
    return "run_tests"


def route_after_human(state: AgentState) -> str:
    """Route after human_review: finalize | run_tests | revise | fail.

    - approve -> finalize
    - edit   -> run_tests (re-validate edited code; retry budget unchanged)
    - reject -> revise if human_rounds <= max_human_rounds, else fail
    - anything else -> fail (caller bug)
    """
    decision = state.get("human_decision")
    if decision == "approve":
        return "finalize"
    if decision == "edit":
        return "run_tests"
    if decision == "reject":
        human_rounds = state.get("human_rounds", 0)
        max_human_rounds = state.get("max_human_rounds", 2)
        if human_rounds <= max_human_rounds:
            return "revise"
        return "fail"
    return "fail"


# --------------------------------------------------------------------------- #
# Graph builder
# --------------------------------------------------------------------------- #


def build_graph(
    deps: Dependencies,
    *,
    checkpointer: Any = None,
) -> Any:
    """Compile the agent graph with the given dependencies.

    Returns the compiled graph (type varies with LangGraph version; annotated
    as ``Any`` to avoid version-specific imports).
    """
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("generate", make_generate_node(deps.llm, deps.settings))
    graph.add_node("run_tests", make_run_tests_node(deps.settings, deps.sandbox))
    graph.add_node("analyze_error", make_analyze_error_node(deps.llm, deps.settings))
    graph.add_node("revise", make_revise_node(deps.llm, deps.settings))
    graph.add_node("human_review", human_review_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("fail", fail_node)

    # Edges
    graph.add_edge(START, "generate")
    graph.add_conditional_edges(
        "generate", route_after_llm, {"fail": "fail", "run_tests": "run_tests"}
    )
    graph.add_conditional_edges(
        "revise", route_after_llm, {"fail": "fail", "run_tests": "run_tests"}
    )
    graph.add_conditional_edges(
        "run_tests",
        route_after_tests,
        {
            "human_review": "human_review",
            "finalize": "finalize",
            "analyze_error": "analyze_error",
            "fail": "fail",
        },
    )
    graph.add_edge("analyze_error", "revise")
    graph.add_conditional_edges(
        "human_review",
        route_after_human,
        {"finalize": "finalize", "run_tests": "run_tests", "revise": "revise", "fail": "fail"},
    )
    graph.add_edge("finalize", END)
    graph.add_edge("fail", END)

    # Task 10 will insert retrieve before generate and after analyze_error

    return graph.compile(checkpointer=checkpointer)


# --------------------------------------------------------------------------- #
# Initial state builder
# --------------------------------------------------------------------------- #


def initial_state(
    task: str,
    settings: Settings,
    *,
    task_id: str | None = None,
    provided_tests: str | None = None,
    auto_approve: bool = False,
) -> AgentState:
    """Build the initial AgentState for a new run."""
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


# --------------------------------------------------------------------------- #
# High-level run/stream helpers
# --------------------------------------------------------------------------- #


def run_task(
    task: str,
    deps: Dependencies,
    *,
    task_id: str | None = None,
    provided_tests: str | None = None,
    auto_approve: bool = False,
    checkpointer: Any = None,
) -> AgentState:
    """Run the agent to completion and return the final state.

    When ``auto_approve=False`` (human review enabled), a ``checkpointer``
    **must** be provided; otherwise a ``ValueError`` is raised before any
    graph execution. The returned state may contain an ``"__interrupt__"`` key
    if the graph paused for human review.
    """
    if auto_approve is False and checkpointer is None:
        raise ValueError("auto_approve=False requires a checkpointer")
    graph = build_graph(deps, checkpointer=checkpointer)
    init = initial_state(
        task=task,
        settings=deps.settings,
        task_id=task_id,
        provided_tests=provided_tests,
        auto_approve=auto_approve,
    )
    config = {"recursion_limit": RECURSION_LIMIT, "configurable": {"thread_id": init["task_id"]}}
    result = graph.invoke(init, config=config)
    return result


def stream_task(
    task: str,
    deps: Dependencies,
    *,
    task_id: str | None = None,
    provided_tests: str | None = None,
    auto_approve: bool = False,
    checkpointer: Any = None,
) -> Iterator[tuple[str, Any]]:
    """Stream the agent execution, yielding (node_name, update_dict) pairs.

    When ``auto_approve=False``, a ``checkpointer`` **must** be provided.
    The stream may yield a ``("__interrupt__", payload)`` chunk when the graph
    pauses for human review.
    """
    if auto_approve is False and checkpointer is None:
        raise ValueError("auto_approve=False requires a checkpointer")
    graph = build_graph(deps, checkpointer=checkpointer)
    init = initial_state(
        task=task,
        settings=deps.settings,
        task_id=task_id,
        provided_tests=provided_tests,
        auto_approve=auto_approve,
    )
    config = {"recursion_limit": RECURSION_LIMIT, "configurable": {"thread_id": init["task_id"]}}
    for chunk in graph.stream(init, config=config, stream_mode="updates"):
        yield from chunk.items()


def pending_review(result: AgentState) -> dict[str, Any] | None:
    """Extract the interrupt payload from an interrupted run result.

    Returns the first ``Interrupt.value`` dict if ``result`` contains an
    ``"__interrupt__"`` key (a tuple of ``Interrupt`` objects), otherwise
    ``None``.
    """
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    # interrupts is a tuple of Interrupt objects; take the first one's value
    first = interrupts[0]
    return first.value if hasattr(first, "value") else None


def resume_task(
    task_id: str,
    response: dict[str, Any],
    deps: Dependencies,
    *,
    checkpointer: Any,
) -> AgentState:
    """Resume a paused task after human review.

    Builds a new graph with the same ``checkpointer`` and ``thread_id``,
    then invokes it with ``Command(resume=response)``. Returns the resulting
    state (which may again contain ``"__interrupt__"`` if another review is
    needed).
    """
    graph = build_graph(deps, checkpointer=checkpointer)
    config = {"recursion_limit": RECURSION_LIMIT, "configurable": {"thread_id": task_id}}
    result = graph.invoke(Command(resume=response), config=config)
    return result
