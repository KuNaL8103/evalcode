"""LangGraph wiring for the evalcode agent (Task 8).

This module compiles the StateGraph with nodes: generate, run_tests,
analyze_error, revise, finalize, fail. Routers use run_result["passed"]
exclusively. No human_review or retrieve nodes yet (added in Tasks 9/10).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph

from evalcode.config import Settings, get_settings
from evalcode.llm import TextLLM, build_llm_client
from evalcode.nodes.analyze_error import make_analyze_error_node
from evalcode.nodes.generate import make_generate_node
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
    "build_graph",
    "run_task",
    "stream_task",
    "initial_state",
    "RECURSION_LIMIT",
]

RECURSION_LIMIT = 100


@dataclass(frozen=True)
class Dependencies:
    """All external dependencies for the graph (no retriever/human_review yet)."""

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
    """Route after run_tests: finalize | analyze_error | fail."""
    run_result = state.get("run_result")
    if not run_result:
        return "fail"
    if run_result.get("passed") is True:
        return "finalize"
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


# --------------------------------------------------------------------------- #
# Graph builder
# --------------------------------------------------------------------------- #


def build_graph(
    deps: Dependencies,
    *,
    checkpointer: Any = None,
) -> StateGraph:
    """Compile the agent graph with the given dependencies."""
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("generate", make_generate_node(deps.llm, deps.settings))
    graph.add_node("run_tests", make_run_tests_node(deps.settings, deps.sandbox))
    graph.add_node("analyze_error", make_analyze_error_node(deps.llm, deps.settings))
    graph.add_node("revise", make_revise_node(deps.llm, deps.settings))
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
        {"finalize": "finalize", "analyze_error": "analyze_error", "fail": "fail"},
    )
    graph.add_edge("analyze_error", "revise")
    graph.add_edge("finalize", END)
    graph.add_edge("fail", END)

    # Task 9 will insert human_review between finalize and END (or similar)
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
    """Run the agent to completion and return the final state."""
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
) -> Iterator[tuple[str, dict]]:
    """Stream the agent execution, yielding (node_name, update_dict) pairs."""
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
