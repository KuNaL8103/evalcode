"""LangGraph wiring for the evalcode agent (Task 11).

This module compiles the StateGraph with nodes: generate, run_tests,
analyze_error, revise, human_review, finalize, fail, and retrieve (RAG).
Routers use run_result["passed"] exclusively.

RAG is enabled iff Dependencies.retriever is set. default_dependencies(rag=True)
builds a retriever from the local index and falls back to None with a warning
when the index is missing/empty. A checkpointed thread must be resumed with the
same RAG topology (retriever present/absent).

Observability: when Dependencies.logger is set (via default_dependencies(observe=True)),
every node is wrapped with traced_node for structured event logging, and
configure_langsmith is applied. run_task/resume_task write summary.json;
stream_task does not.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from evalcode.config import Settings, get_settings
from evalcode.llm import TextLLM, build_llm_client
from evalcode.nodes.analyze_error import make_analyze_error_node
from evalcode.nodes.generate import make_generate_node
from evalcode.nodes.human_review import human_review_node
from evalcode.nodes.retrieve import make_retrieve_node
from evalcode.nodes.revise import make_revise_node
from evalcode.nodes.run_tests import make_run_tests_node
from evalcode.nodes.terminal import fail_node, finalize_node
from evalcode.observability import (
    build_logger,
    build_run_config,
    configure_langsmith,
    traced_node,
)
from evalcode.sandbox.runner import run_in_sandbox
from evalcode.state import AgentState

__all__ = [
    "Dependencies",
    "default_dependencies",
    "build_retriever",
    "route_after_tests",
    "route_after_llm",
    "route_after_human",
    "route_after_analysis",
    "route_after_retrieve",
    "build_graph",
    "run_task",
    "stream_task",
    "initial_state",
    "pending_review",
    "resume_task",
    "get_task_state",
    "stream_resume_task",
    "close_dependencies",
    "rag_enabled_in",
    "RECURSION_LIMIT",
]

RECURSION_LIMIT = 100

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Dependencies:
    """All external dependencies for the graph.

    When ``retriever`` is None (default), the graph runs in RAG-off mode:
    START -> generate -> run_tests -> ... (no retrieve node).
    When set, the graph includes the retrieve node before generate and
    after analyze_error (when needs_docs with fresh queries).

    When ``logger`` is set (RunLogger), structured event logging is enabled
    for all nodes via traced_node. default_dependencies(observe=True) also
    configures LangSmith if settings.langsmith_tracing and a key are set.
    """

    llm: TextLLM
    settings: Settings
    sandbox: Callable[[str, str, float, int], dict] = run_in_sandbox
    retriever: Any = None  # RetrieverLike | None; RAG on iff not None
    logger: Any = None  # RunLogger | None; observability on iff not None


def default_dependencies(
    settings: Settings | None = None,
    *,
    rag: bool = True,
    observe: bool = False,
    run_id: str | None = None,
) -> Dependencies:
    """Build real dependencies from settings.

    If ``rag=True`` (default), builds a retriever from the local index.
    If the index is missing or empty, logs a warning and returns None for
    retriever (RAG-off mode). If ``rag=False``, never builds a retriever.

    If ``observe=True``, configures LangSmith (if tracing requested and key exists)
    and creates a RunLogger for structured event logging. If ``observe=False``,
    neither LangSmith nor the filesystem is touched.
    """
    if settings is None:
        settings = get_settings()
    llm = build_llm_client(settings)
    retriever = build_retriever(settings) if rag else None
    logger = None
    if observe:
        configure_langsmith(settings)
        logger = build_logger(settings, run_id)
    return Dependencies(
        llm=llm,
        settings=settings,
        sandbox=run_in_sandbox,
        retriever=retriever,
        logger=logger,
    )


def build_retriever(settings: Settings, *, embedder: Any = None) -> Any | None:
    """Build a Retriever from settings, or return None with a warning.

    - If the chroma_dir does not exist, logs a warning and returns None.
    - If the store exists but has 0 chunks, closes it, logs a warning, and returns None.
    - Otherwise returns a Retriever(store, settings.retrieval_top_k, settings.retrieval_min_score).

    All heavy imports (chromadb, sentence-transformers) happen inside this function.
    """
    from evalcode.rag.embeddings import get_embedder
    from evalcode.rag.retriever import Retriever
    from evalcode.rag.store import VectorStore

    chroma_dir = Path(settings.chroma_dir)
    if not chroma_dir.exists():
        logger.warning("Chroma directory %s does not exist; RAG disabled", chroma_dir)
        return None

    # Use provided embedder or create one from settings
    if embedder is None:
        embedder = get_embedder(settings)

    store = VectorStore(chroma_dir, settings.collection_name, embedder)
    try:
        if store.count() == 0:
            logger.warning("Chroma collection %s is empty; RAG disabled", settings.collection_name)
            store.close()
            return None
        return Retriever(store, settings.retrieval_top_k, settings.retrieval_min_score)
    except Exception:
        store.close()
        raise


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


def route_after_analysis(state: AgentState) -> str:
    """Route after analyze_error: retrieve | revise.

    - If error_analysis.needs_docs is False -> revise (no docs needed).
    - If error_analysis.retrieval_queries is empty -> revise (nothing to retrieve).
    - If every query in retrieval_queries is already in state.retrieval_queries
      (dedupe memory) -> revise (avoid repeat-retrieval loops).
    - Otherwise -> retrieve.
    """
    error_analysis = state.get("error_analysis") or {}
    needs_docs = error_analysis.get("needs_docs", False)
    if not needs_docs:
        return "revise"

    queries = error_analysis.get("retrieval_queries") or []
    if not queries:
        return "revise"

    prior_queries = state.get("retrieval_queries") or []
    # Dedupe: if all queries have been used before, skip retrieval
    if all(q in prior_queries for q in queries):
        return "revise"

    return "retrieve"


def route_after_retrieve(state: AgentState) -> str:
    """Route after retrieve: generate | revise.

    - If state.code is empty/blank -> generate (first pass).
    - Otherwise -> revise (error-driven re-retrieval).
    """
    code = state.get("code") or ""
    if not code.strip():
        return "generate"
    return "revise"


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

    When deps.retriever is None, the topology is:
      START -> generate -> run_tests -> (human_review|finalize|analyze_error|fail)
      analyze_error -> revise -> run_tests ...

    When deps.retriever is set, the topology adds a "retrieve" node:
      START -> retrieve -> route_after_retrieve {generate, revise}
      analyze_error -> route_after_analysis {retrieve, revise}

    A checkpointed thread must be resumed with the same RAG on/off topology.

    When deps.logger is set, every node is wrapped with traced_node for
    structured event logging (per-node usage, cumulative usage, LLM stats deltas).
    """
    graph = StateGraph(AgentState)

    # Optional node wrapper for observability
    def _wrap(name: str, fn: Any) -> Any:
        if deps.logger is not None:
            return traced_node(name, fn, deps.logger, llm_stats=deps.llm)
        return fn

    # Nodes (always present)
    graph.add_node("generate", _wrap("generate", make_generate_node(deps.llm, deps.settings)))
    graph.add_node(
        "run_tests", _wrap("run_tests", make_run_tests_node(deps.settings, deps.sandbox))
    )
    graph.add_node(
        "analyze_error",
        _wrap("analyze_error", make_analyze_error_node(deps.llm, deps.settings)),
    )
    graph.add_node("revise", _wrap("revise", make_revise_node(deps.llm, deps.settings)))
    graph.add_node("human_review", _wrap("human_review", human_review_node))
    graph.add_node("finalize", _wrap("finalize", finalize_node))
    graph.add_node("fail", _wrap("fail", fail_node))

    # Conditionally add retrieve node
    has_retriever = deps.retriever is not None
    if has_retriever:
        graph.add_node(
            "retrieve",
            _wrap("retrieve", make_retrieve_node(deps.retriever, deps.llm, deps.settings)),
        )

    # Edges
    if has_retriever:
        graph.add_edge(START, "retrieve")
        graph.add_conditional_edges(
            "retrieve",
            route_after_retrieve,
            {"generate": "generate", "revise": "revise"},
        )
        graph.add_conditional_edges(
            "analyze_error",
            route_after_analysis,
            {"retrieve": "retrieve", "revise": "revise"},
        )
    else:
        graph.add_edge(START, "generate")
        graph.add_edge("analyze_error", "revise")

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
    graph.add_conditional_edges(
        "human_review",
        route_after_human,
        {"finalize": "finalize", "run_tests": "run_tests", "revise": "revise", "fail": "fail"},
    )
    graph.add_edge("finalize", END)
    graph.add_edge("fail", END)

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

    If deps.logger is set, writes summary.json after completion (including
    when paused at human_review with status "awaiting_review").
    """
    if auto_approve is False and checkpointer is None:
        raise ValueError("auto_approve=False requires a checkpointer")

    # Reset per-run LLM call budget for new runs (not resumes)
    reset = getattr(deps.llm, "reset_budget", None)
    if callable(reset):
        reset()

    graph = build_graph(deps, checkpointer=checkpointer)
    init = initial_state(
        task=task,
        settings=deps.settings,
        task_id=task_id,
        provided_tests=provided_tests,
        auto_approve=auto_approve,
    )
    run_id = deps.logger.run_id if deps.logger else None
    config = build_run_config(init["task_id"], run_id=run_id, recursion_limit=RECURSION_LIMIT)
    result = graph.invoke(init, config=config)

    if deps.logger is not None:
        deps.logger.write_summary(result)
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

    Events are logged via traced_node if deps.logger is set, but this function
    does NOT write summary.json (the Task 12 CLI will call write_summary).
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
    run_id = deps.logger.run_id if deps.logger else None
    config = build_run_config(init["task_id"], run_id=run_id, recursion_limit=RECURSION_LIMIT)
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

    Does NOT reset the LLM per-run call budget (a resume continues the same run).
    If deps.logger is set, writes summary.json after completion.
    """
    graph = build_graph(deps, checkpointer=checkpointer)
    run_id = deps.logger.run_id if deps.logger else None
    config = build_run_config(task_id, run_id=run_id, recursion_limit=RECURSION_LIMIT)
    result = graph.invoke(Command(resume=response), config=config)

    if deps.logger is not None:
        deps.logger.write_summary(result)
    return result


def get_task_state(
    task_id: str,
    settings: Settings,
    *,
    checkpointer: Any,
) -> dict[str, Any] | None:
    """Read the checkpointed state for a task without executing the graph.

    Compiles a graph with a private _NoLLM (raises on any LLM call) and the
    given checkpointer, then calls .get_state({"configurable": {"thread_id": task_id}}).
    If the snapshot has no values, returns None. Otherwise returns a dict copy
    of snapshot.values with ``__interrupt__`` populated from the snapshot's
    tasks' interrupts (using the PregelTask.interrupts field) so that
    pending_review(state) and build_summary work unchanged.

    Never mutates the checkpoint.
    """
    from langgraph.types import StateSnapshot

    # Private no-op LLM that raises on any call
    class _NoLLM:
        def invoke_text(self, *args: Any, **kwargs: Any) -> str:
            raise RuntimeError("No LLM available in get_task_state (read-only)")

        def reset_budget(self) -> None:
            pass

    deps = Dependencies(
        llm=_NoLLM(),
        settings=settings,
        sandbox=run_in_sandbox,
        retriever=None,
        logger=None,
    )
    graph = build_graph(deps, checkpointer=checkpointer)
    snapshot: StateSnapshot = graph.get_state({"configurable": {"thread_id": task_id}})

    if not snapshot.values:
        return None

    state = dict(snapshot.values)
    # Collect interrupts from snapshot.tasks (PregelTask has 'interrupts' field)
    all_interrupts = tuple(
        i for t in snapshot.tasks if hasattr(t, "interrupts") for i in t.interrupts
    )
    if all_interrupts:
        state["__interrupt__"] = all_interrupts
    return state


def stream_resume_task(
    task_id: str,
    response: dict[str, Any],
    deps: Dependencies,
    *,
    checkpointer: Any,
) -> Iterator[tuple[str, Any]]:
    """Stream a resume after human review, yielding (node, update) pairs.

    Builds the graph exactly like resume_task, but uses graph.stream with
    Command(resume=response) and stream_mode="updates". Does NOT reset the
    LLM budget and does NOT write a summary (the CLI writes it).

    Args:
        task_id: The thread_id to resume.
        response: The human decision response dict.
        deps: Dependencies (with fresh LLM, logger, etc.).
        checkpointer: The same checkpointer used for the original run.

    Yields:
        (node_name, update_dict) pairs from the stream.
    """
    graph = build_graph(deps, checkpointer=checkpointer)
    run_id = deps.logger.run_id if deps.logger else None
    config = build_run_config(task_id, run_id=run_id, recursion_limit=RECURSION_LIMIT)
    for chunk in graph.stream(Command(resume=response), config=config, stream_mode="updates"):
        yield from chunk.items()


def close_dependencies(deps: Dependencies) -> None:
    """Close the retriever's Chroma store if present.

    Catches and logs any exception at WARNING level; never raises.
    Safe to call multiple times and with retriever=None.
    """
    store = getattr(getattr(deps, "retriever", None), "store", None)
    close_fn = getattr(store, "close", None)
    if callable(close_fn):
        try:
            close_fn()
        except Exception as e:
            logger.warning("Failed to close retriever store: %s", e)


def rag_enabled_in(state: AgentState) -> bool:
    """Return True iff the state indicates RAG was enabled (history[0].node == 'retrieve').

    The retrieve node always runs first when RAG is on. Pure function, no side effects.
    """
    history = state.get("history") or []
    if not history:
        return False
    first = history[0]
    if isinstance(first, dict):
        return first.get("node") == "retrieve"
    return False
