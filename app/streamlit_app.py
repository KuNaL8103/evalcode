"""Interactive Streamlit dashboard for evalcode.

Launch:

    streamlit run app/streamlit_app.py

Tab 1 runs the real LangGraph workflow (generation, sandboxed tests, revision).
Tab 2 exercises the same hybrid retriever directly, so you can show the
BM25 + dense routing without spending LLM quota.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import streamlit as st

from evalcode.core.config import Settings, get_settings

ResumeFn = Callable[..., tuple[dict[str, Any], float]]

_STATUS_LABELS = {
    "approved": "Approved",
    "failed": "Failed",
    "awaiting_review": "Awaiting Human Review",
    "rejected": "Rejected",
    "running": "Running",
}

DEFAULT_INGEST_LIBS = "json,re,pathlib,collections,itertools,datetime"

SAMPLE_TASKS: dict[str, str] = {
    "JSON: parse and list keys": (
        "Write a function `json_keys(text)` that parses a JSON object string and "
        "returns its top-level keys as a sorted list. Raise ValueError on invalid JSON."
    ),
    "Regex: extract log fields": (
        "Write a function `parse_log_line(line)` that parses a line like "
        "'2024-01-01T12:00:00Z INFO user=alice action=login' and returns a dict "
        "with timestamp, level, user, and action."
    ),
    "Pathlib: find files by suffix": (
        "Write a function `find_files(root, suffix)` that recursively returns a "
        "sorted list of file paths under `root` ending with `suffix`, using pathlib."
    ),
    "Collections: count words": (
        "Write a function `word_counts(text)` that returns a dict mapping each "
        "lowercase word to its frequency, using collections.Counter. Ignore punctuation."
    ),
}


# --------------------------------------------------------------------------- #
# Lazy helpers (heavy imports stay out of module import time)
# --------------------------------------------------------------------------- #


@st.cache_data(ttl=10, show_spinner=False)
def chroma_chunk_count(chroma_dir: str, collection_name: str) -> int:
    """Count indexed chunks without loading the embedding model."""
    if not Path(chroma_dir).exists():
        return 0
    import chromadb

    try:
        client = chromadb.PersistentClient(path=chroma_dir)
    except Exception:
        return 0
    try:
        try:
            collection = client.get_collection(collection_name)
        except Exception:
            return 0
        return int(collection.count())
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


@st.cache_data(ttl=15, show_spinner=False)
def docker_available(image: str) -> bool:
    """Probe the Docker daemon (cached so the sidebar stays responsive)."""
    try:
        from evalcode.sandbox.docker import DockerSandbox

        return DockerSandbox(image=image).is_available()
    except Exception:
        return False


def resolve_backend(choice: str, settings: Settings) -> str:
    """Map the UI choice (auto/docker/subprocess) to a concrete backend."""
    if choice == "subprocess":
        return "subprocess"
    ready = docker_available(settings.docker_image)
    return "docker" if ready else "subprocess"


def settings_with_overrides(*, top_k: int, adaptive: bool, backend: str) -> Settings:
    base = get_settings()
    return base.model_copy(
        update={
            "retrieval_top_k": int(top_k),
            # The retrieve node caps merged docs; honor top-k end to end.
            "retrieval_max_docs": max(int(base.retrieval_max_docs), int(top_k)),
            "hybrid_adaptive": bool(adaptive),
            "sandbox_backend": backend,
        }
    )


def ingest_libraries(libraries: list[str], settings: Settings) -> dict[str, Any]:
    from evalcode.rag.embeddings import get_embedder
    from evalcode.rag.ingest import ingest
    from evalcode.rag.store import VectorStore

    docs_dir = Path(settings.docs_dir)
    store = VectorStore(
        settings.chroma_dir,
        settings.collection_name,
        get_embedder(settings),
    )
    try:
        return ingest(
            libraries,
            store,
            docs_dir=docs_dir if docs_dir.exists() else None,
            reset=False,
        )
    finally:
        store.close()


def run_agent(
    task: str,
    provided_tests: str | None,
    settings: Settings,
    *,
    use_rag: bool,
    auto_approve: bool,
) -> tuple[dict[str, Any], float]:
    """Run the graph with a persistent checkpointer.

    With ``auto_approve=False`` the run pauses at ``human_review`` and the
    returned state contains ``__interrupt__``; the checkpoint can then be
    resumed by ``resume_agent`` using the same ``task_id``.
    """
    from evalcode.core.persistence import open_checkpointer
    from evalcode.graph import close_dependencies, default_dependencies, run_task

    started = time.perf_counter()
    with open_checkpointer(settings.checkpoint_db) as checkpointer:
        deps = default_dependencies(settings, rag=use_rag)
        try:
            state = run_task(
                task,
                deps,
                provided_tests=provided_tests,
                auto_approve=auto_approve,
                checkpointer=checkpointer,
            )
        finally:
            close_dependencies(deps)
    return state, time.perf_counter() - started


def resume_agent(
    task_id: str,
    response: dict[str, Any],
    settings: Settings,
    *,
    use_rag: bool,
) -> tuple[dict[str, Any], float]:
    """Resume a paused checkpoint with an approve/reject/reject_final decision."""
    from evalcode.core.persistence import open_checkpointer
    from evalcode.graph import close_dependencies, default_dependencies, resume_task

    started = time.perf_counter()
    with open_checkpointer(settings.checkpoint_db) as checkpointer:
        deps = default_dependencies(settings, rag=use_rag)
        try:
            if use_rag and deps.retriever is None:
                raise RuntimeError(
                    "This task was started with RAG, but the documentation index is "
                    "unavailable. Re-ingest the libraries before resuming so the graph "
                    "topology matches the checkpoint."
                )
            state = resume_task(task_id, response, deps, checkpointer=checkpointer)
        finally:
            close_dependencies(deps)
    return state, time.perf_counter() - started


def review_response(decision: str, feedback: str) -> dict[str, Any]:
    """Translate a Streamlit review action into the existing resume payload."""
    text = (feedback or "").strip()
    if decision == "approve":
        return {"decision": "approve"}
    if decision == "revision":
        if not text:
            raise ValueError("Reviewer feedback is required to request a revision.")
        return {"decision": "reject", "feedback": text}
    if decision == "reject":
        return {"decision": "reject_final", "feedback": text or "Rejected by reviewer."}
    raise ValueError(f"Unknown review action: {decision!r}")


def _pending(state: dict[str, Any]) -> dict[str, Any] | None:
    """Extract the interrupt payload without keeping LangGraph objects in session."""
    from evalcode.graph import pending_review

    return pending_review(state)


def _display_state(state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    pending = _pending(state)
    display = dict(state)
    display.pop("__interrupt__", None)
    if pending is not None and display.get("status") not in ("approved", "failed"):
        display["status"] = "awaiting_review"
    if state.get("human_decision") == "reject_final":
        display["status"] = "rejected"
    return display, pending


def make_run_record(
    state: dict[str, Any],
    *,
    elapsed_s: float,
    backend: str,
    use_rag: bool,
    route: str,
    task: str,
    auto_approve: bool,
) -> dict[str, Any]:
    """Build the session record for a fresh run (paused or complete)."""
    display, pending = _display_state(state)
    return {
        "state": display,
        "task_id": str(state.get("task_id") or ""),
        "pending_review": pending,
        "paused": pending is not None,
        "elapsed_s": elapsed_s,
        "backend": backend,
        "use_rag": use_rag,
        "route": route,
        "task": task,
        "auto_approve": auto_approve,
    }


def apply_resume_result(
    record: dict[str, Any],
    state: dict[str, Any],
    elapsed_s: float,
    *,
    decision: str,
) -> dict[str, Any]:
    """Update a session record after a review decision (shared by UI + tests)."""
    display, pending = _display_state(state)
    updated = dict(record)
    updated.update(
        {
            "state": display,
            "task_id": str(state.get("task_id") or record.get("task_id") or ""),
            "pending_review": pending,
            "paused": pending is not None,
            "elapsed_s": elapsed_s,
            "last_decision": decision,
        }
    )
    return updated


def perform_review(
    record: dict[str, Any],
    decision: str,
    feedback: str,
    settings: Settings,
    *,
    resume_fn: ResumeFn | None = None,
) -> dict[str, Any]:
    """Apply a review action and return the updated session record."""
    resume = resume_fn or resume_agent
    response = review_response(decision, feedback)
    state, elapsed = resume(
        record["task_id"],
        response,
        settings,
        use_rag=bool(record.get("use_rag", True)),
    )
    return apply_resume_result(record, state, elapsed, decision=decision)


def search_docs(
    query: str,
    settings: Settings,
    *,
    top_k: int,
    use_rag: bool,
    adaptive: bool,
) -> tuple[list[dict[str, Any]], str]:
    """Return (docs, routing label) for the RAG explorer."""
    if not use_rag:
        return [], "RAG disabled"

    from evalcode.rag.embeddings import get_embedder
    from evalcode.rag.retriever import HybridRetriever, Retriever, is_symbol_query
    from evalcode.rag.store import VectorStore

    hybrid = settings.hybrid_retrieval
    if hybrid:
        lexical = (not adaptive) or is_symbol_query(query)
        route = "Hybrid (BM25 + Dense)" if lexical else "Dense-only"
    else:
        route = "Dense-only"

    store = VectorStore(
        settings.chroma_dir,
        settings.collection_name,
        get_embedder(settings),
    )
    try:
        if hybrid:
            retriever = HybridRetriever(
                store,
                top_k,
                settings.retrieval_min_score,
                rrf_k=settings.rrf_k,
                adaptive=adaptive,
            )
        else:
            retriever = Retriever(store, top_k, settings.retrieval_min_score)
        docs = retriever.retrieve([query], k=top_k)
    finally:
        store.close()
    return [dict(doc) for doc in docs], route


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #


def _snippet(text: str, limit: int = 600) -> str:
    collapsed = " ".join((text or "").split())
    return collapsed if len(collapsed) <= limit else collapsed[:limit] + "..."


def _score_label(route: str) -> str:
    return "RRF score" if route.startswith("Hybrid") else "dense cosine"


def render_docs(docs: list[dict[str, Any]], *, route: str, empty_message: str) -> None:
    if not docs:
        st.info(empty_message)
        return
    for i, doc in enumerate(docs, start=1):
        title = doc.get("qualname") or doc.get("id") or f"Chunk {i}"
        score = float(doc.get("score") or 0.0)
        library = doc.get("library") or "?"
        # Containers (not nested expanders): Streamlit forbids an expander
        # inside another expander, and the generator tab already wraps these
        # docs in one collapsible section.
        with st.container(border=True):
            st.markdown(
                f"**{i}. `{title}`**  \n"
                f"{_score_label(route)}: **{score:.4f}**  •  library: `{library}`  "
                f"•  id: `{doc.get('id', '?')}`"
            )
            st.write(_snippet(str(doc.get("text") or "")))


def render_agent_result(record: dict[str, Any], task: str) -> None:
    state = record["state"]
    status = str(state.get("status") or "unknown")
    if status == "approved":
        st.success("Approved — generated tests passed.")
    elif status == "awaiting_review":
        st.warning("Awaiting Human Review — tests passed; approve or request changes below.")
    elif status == "rejected":
        reason = str(state.get("failure_reason") or "").strip()
        st.error(f"Rejected by reviewer. {reason}".strip())
    elif status == "failed":
        st.error(f"Failed — {state.get('failure_reason') or 'see details below.'}")
    else:
        st.info(f"Status: {status}")

    usage = state.get("token_usage") or {}
    total_tokens = int(usage.get("total_tokens") or 0)
    llm_calls = int(usage.get("llm_calls") or 0)
    run_result = state.get("run_result") or {}

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Status", _STATUS_LABELS.get(status, status.title()))
    c2.metric("Attempts", int(state.get("attempt") or 0))
    c3.metric("Execution time", f"{record['elapsed_s']:.1f}s")
    c4.metric("Sandbox", record["backend"])
    c5.metric("Tokens", f"{total_tokens:,}" if total_tokens else "n/a")
    st.caption(
        f"task_id: {state.get('task_id', '?')}  •  LLM calls: {llm_calls}  "
        f"•  provider input/output tokens: "
        f"{int(usage.get('input_tokens') or 0):,}/"
        f"{int(usage.get('output_tokens') or 0):,}  "
        f"•  RAG: {'on' if record['use_rag'] else 'off'}  "
        f"•  retrieval route: {record['route']}"
    )

    docs = state.get("retrieved_docs") or []
    with st.expander(
        f"Retrieved RAG documentation ({len(docs)} chunk{'s' if len(docs) != 1 else ''})"
    ):
        render_docs(
            docs,
            route=record["route"],
            empty_message=(
                "No documentation retrieved. Build the index in the sidebar, or turn RAG on."
            ),
        )

    code = state.get("final_code") or state.get("code") or ""
    tests = state.get("tests") or ""
    code_col, tests_col = st.columns(2, gap="large")
    with code_col:
        st.subheader("Final Generated Code")
        st.caption("solution.py")
        if code:
            st.code(code, language="python", line_numbers=True)
        else:
            st.info("No code was produced.")
    with tests_col:
        st.subheader("Generated Pytest Suite")
        st.caption("test_solution.py")
        if tests:
            st.code(tests, language="python", line_numbers=True)
        else:
            st.info("No generated tests.")

    st.subheader("Sandbox Result")
    if not run_result:
        st.warning("The workflow did not reach the sandbox.")
    else:
        passed = bool(run_result.get("passed"))
        category = run_result.get("category", "unknown")
        if passed:
            st.success(f"pytest passed — category: {category}")
        else:
            st.error(f"pytest failed — category: {category}")
        st.caption(
            f"exit_code: {run_result.get('exit_code')}  •  "
            f"timed_out: {run_result.get('timed_out')}  •  "
            f"tests: {run_result.get('tests_failed', 0)} failed / "
            f"{run_result.get('tests_total', 0)} total  •  "
            f"duration: {float(run_result.get('duration_s') or 0.0):.2f}s"
        )
        with st.expander("pytest stdout"):
            st.code(str(run_result.get("stdout") or "(empty)"), language="text")
        with st.expander("pytest stderr"):
            st.code(str(run_result.get("stderr") or "(empty)"), language="text")
        failures = run_result.get("failures") or []
        if failures:
            with st.expander(f"Failure details ({len(failures)})", expanded=True):
                for failure in failures:
                    st.markdown(
                        f"**{failure.get('test_name', '?')}** — `{failure.get('error_type', '?')}`"
                    )
                    st.code(
                        str(failure.get("message") or failure.get("traceback") or "(no message)"),
                        language="text",
                    )

    analysis = state.get("error_analysis") or {}
    if analysis:
        with st.expander("Deterministic error analysis"):
            st.markdown(
                f"**Category:** `{analysis.get('category', '?')}`  \n"
                f"**Fault:** {analysis.get('fault', '?')}  \n"
                f"**Root cause:** {analysis.get('root_cause', '?')}  \n"
                f"**Fix plan:** {analysis.get('fix_plan', '?')}"
            )


def _handle_review_action(
    record: dict[str, Any],
    decision: str,
    feedback: str,
    settings: Settings,
) -> None:
    label = {
        "approve": "Approving and finalizing...",
        "revision": "Applying reviewer feedback and re-running...",
        "reject": "Rejecting the solution...",
    }[decision]
    with st.spinner(label):
        try:
            updated = perform_review(record, decision, feedback, settings)
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            st.error(f"Review action failed: {exc}")
            st.exception(exc)
            return
    updated["backend"] = settings.sandbox_backend
    st.session_state["last_run"] = updated
    st.rerun()


def render_review_panel(record: dict[str, Any], settings: Settings) -> None:
    """Human-in-the-loop checkpoint controls for a paused run."""
    if not record.get("paused"):
        return

    pending = record.get("pending_review") or {}
    task_id = record.get("task_id") or pending.get("task_id") or "?"
    attempt = int(pending.get("attempt") or record["state"].get("attempt") or 0)
    human_round = int(pending.get("human_round") or 0)
    max_rounds = int(pending.get("max_human_rounds") or 2)

    st.divider()
    st.subheader("Human-in-the-Loop Review Checkpoint")
    st.caption(
        f"task_id: `{task_id}`  •  attempt: {attempt}  •  "
        f"human round: {human_round}/{max_rounds}  •  "
        "the workflow is paused on a persisted SQLite checkpoint."
    )
    if pending.get("explanation"):
        st.info(str(pending["explanation"]))

    feedback_key = f"review_feedback_{task_id}_{attempt}"
    feedback = st.text_area(
        "Reviewer feedback / revision instructions",
        key=feedback_key,
        height=100,
        placeholder=(
            "Optional for approval. Required for Request Revision — e.g. "
            "'add type hints and handle empty input'."
        ),
    )

    approve_col, revise_col, reject_col = st.columns(3)
    if approve_col.button(
        "✅ Approve Code",
        key=f"approve_{task_id}_{attempt}",
        type="primary",
        use_container_width=True,
    ):
        _handle_review_action(record, "approve", feedback, settings)

    if revise_col.button(
        "🔄 Request Revision (Re-run with Feedback)",
        key=f"revise_{task_id}_{attempt}",
        use_container_width=True,
    ):
        if not feedback.strip():
            st.warning("Add reviewer feedback before requesting a revision.")
        else:
            _handle_review_action(record, "revision", feedback, settings)

    if reject_col.button(
        "❌ Reject",
        key=f"reject_{task_id}_{attempt}",
        use_container_width=True,
    ):
        _handle_review_action(record, "reject", feedback, settings)


def render_search_result(record: dict[str, Any]) -> None:
    route = record["route"]
    if route == "RAG disabled":
        st.info("RAG is off. Enable it in the sidebar to search the index.")
        return
    if route.startswith("Hybrid"):
        st.success(f"Routing: {route}")
    else:
        st.info(f"Routing: {route}")
    render_docs(
        record["docs"],
        route=route,
        empty_message="No chunks matched. Build the index in the sidebar or raise top-k.",
    )


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #


def main() -> None:
    st.set_page_config(page_title="evalcode", layout="wide")
    st.title("evalcode")
    st.caption("Iterative RAG + LangGraph code-generation agent")

    settings = get_settings()

    # --- Sidebar ----------------------------------------------------------
    with st.sidebar:
        st.header("Controls")
        use_rag = st.toggle("Enable RAG", value=True)
        auto_approve = st.toggle(
            "Auto-Approve Code",
            value=False,
            help=(
                "Off: pause at the human-review checkpoint after tests pass. "
                "On: finalize automatically (demo/CI mode)."
            ),
        )
        if auto_approve:
            st.caption("Human review skipped — the run finalizes itself.")
        else:
            st.caption("Human review active — runs pause for approve/reject.")
        adaptive = st.toggle(
            "Hybrid adaptive gating",
            value=settings.hybrid_adaptive,
            disabled=not use_rag,
            help=(
                "On: BM25 is fused only for symbol-like queries. "
                "Off: BM25 is fused into every query."
            ),
        )
        backend_choice = st.selectbox(
            "Sandbox backend",
            options=["auto", "docker", "subprocess"],
            index=0,
            help="auto uses Docker when the daemon is available, else subprocess.",
        )
        top_k = st.slider(
            "Top-k retrieval",
            min_value=1,
            max_value=20,
            value=min(max(int(settings.retrieval_top_k), 1), 20),
        )

        backend = resolve_backend(backend_choice, settings)
        docker_ready = docker_available(settings.docker_image)
        st.caption(
            f"Resolved sandbox: **{backend}**  \n"
            f"Docker daemon: {'available' if docker_ready else 'unavailable (fallback active)'}"
        )

        st.divider()
        st.subheader("Documentation index")
        libs_text = st.text_input(
            "Libraries (comma-separated)",
            value=DEFAULT_INGEST_LIBS,
            help="Semantic search is only as good as this local doc index.",
        )
        if st.button("Ingest / refresh libraries", use_container_width=True):
            libraries = [part.strip() for part in libs_text.split(",") if part.strip()]
            if not libraries:
                st.warning("Enter at least one library name.")
            else:
                with st.spinner("Chunking, embedding, and upserting into Chroma..."):
                    try:
                        report = ingest_libraries(libraries, settings)
                    except Exception as exc:  # noqa: BLE001 - surfaced in the UI
                        st.error(f"Ingestion failed: {exc}")
                    else:
                        chroma_chunk_count.clear()
                        st.success(
                            f"Ingested {report['total']} chunks in {report['seconds']:.1f}s."
                        )
                        if report.get("per_library"):
                            st.caption(
                                "  •  ".join(
                                    f"{lib}: {count}"
                                    for lib, count in sorted(report["per_library"].items())
                                )
                            )
                        if report.get("skipped"):
                            st.caption(f"Skipped: {', '.join(report['skipped'])}")

        count = chroma_chunk_count(settings.chroma_dir, settings.collection_name)
        st.metric("Indexed chunks", f"{count:,}")
        st.caption(f"Chroma: `{settings.chroma_dir}`")

        st.divider()
        st.caption(f"LLM: `{settings.llm_model}`")
        key_state = (
            "configured"
            if settings.gemini_api_key is not None
            and bool(settings.gemini_api_key.get_secret_value().strip())
            else "missing (set GEMINI_API_KEY)"
        )
        st.caption(f"Gemini API key: {key_state}")
        st.caption(f"Hybrid retrieval: {'on' if settings.hybrid_retrieval else 'off'}")

    # Build the effective settings for this rerun.
    effective = settings_with_overrides(
        top_k=top_k,
        adaptive=adaptive,
        backend=backend,
    )

    if not use_rag:
        route = "RAG off"
    elif not effective.hybrid_retrieval:
        route = "Dense-only"
    elif not adaptive:
        route = "Hybrid (BM25 + Dense)"
    else:
        route = "adaptive (symbol → Hybrid, prose → Dense-only)"

    tab_agent, tab_rag = st.tabs(["Code Generator Agent", "Hybrid RAG Explorer"])

    # --- Tab 1: agent -----------------------------------------------------
    with tab_agent:
        st.subheader("Describe a coding task")
        st.selectbox(
            "Load a sample task",
            options=["— choose a sample —", *SAMPLE_TASKS],
            key="sample_task",
            on_change=_load_sample,
        )
        task = st.text_area(
            "Coding task",
            key="task_input",
            height=130,
            placeholder="Write a function that ...",
        )
        provided_tests = st.text_area(
            "Optional custom pytest tests",
            key="provided_tests",
            height=110,
            placeholder=(
                "Leave blank to let the agent write tests. If provided, these tests "
                "are treated as ground truth."
            ),
        )

        generate = st.button(
            "Generate & Verify Code",
            type="primary",
            use_container_width=True,
        )

        if generate:
            if not task.strip():
                st.warning("Enter a coding task first.")
            else:
                with st.spinner(
                    "Running retrieve → generate → sandbox → revise. "
                    "This can take a minute on the free tier..."
                ):
                    try:
                        state, elapsed = run_agent(
                            task.strip(),
                            provided_tests.strip() or None,
                            effective,
                            use_rag=use_rag,
                            auto_approve=auto_approve,
                        )
                    except Exception as exc:  # noqa: BLE001 - surfaced in the UI
                        st.error(f"Run failed: {exc}")
                        st.exception(exc)
                    else:
                        run_route = route
                        if use_rag and effective.hybrid_retrieval and adaptive:
                            from evalcode.rag.retriever import is_symbol_query

                            run_route = (
                                "Hybrid (BM25 + Dense)" if is_symbol_query(task) else "Dense-only"
                            )
                        elif not use_rag:
                            run_route = "RAG off"
                        elif not effective.hybrid_retrieval:
                            run_route = "Dense-only"
                        st.session_state["last_run"] = make_run_record(
                            state,
                            elapsed_s=elapsed,
                            backend=backend,
                            use_rag=use_rag,
                            route=run_route,
                            task=task.strip(),
                            auto_approve=auto_approve,
                        )

        if st.session_state.get("last_run"):
            record = st.session_state["last_run"]
            st.divider()
            render_agent_result(record, record["task"])
            render_review_panel(record, effective)
        else:
            st.info(
                "Run a task to see the generated code, tests, retrieved docs, and sandbox logs."
            )

    # --- Tab 2: RAG explorer ---------------------------------------------
    with tab_rag:
        st.subheader("Search the documentation index")
        st.caption(
            f"Current routing: **{route}**. Scores are RRF for hybrid queries "
            "and cosine similarity for dense-only queries."
        )
        with st.form("rag_search_form"):
            query = st.text_input(
                "Query",
                value="collections.deque",
                placeholder="json.loads, move_to_end, parse a JSON string...",
            )
            submitted = st.form_submit_button("Search documentation")

        if submitted:
            if not query.strip():
                st.warning("Enter a query.")
            elif not use_rag:
                st.info("RAG is off. Enable it in the sidebar first.")
            elif chroma_chunk_count(settings.chroma_dir, settings.collection_name) == 0:
                st.warning("The index is empty. Ingest libraries in the sidebar first.")
            else:
                with st.spinner("Searching Chroma + BM25..."):
                    try:
                        docs, search_route = search_docs(
                            query.strip(),
                            effective,
                            top_k=top_k,
                            use_rag=use_rag,
                            adaptive=adaptive,
                        )
                    except Exception as exc:  # noqa: BLE001 - surfaced in the UI
                        st.error(f"Search failed: {exc}")
                    else:
                        st.session_state["last_search"] = {
                            "query": query.strip(),
                            "docs": docs,
                            "route": search_route,
                        }

        if st.session_state.get("last_search"):
            record = st.session_state["last_search"]
            st.markdown(f"**Query:** `{record['query']}`")
            render_search_result(record)


def _load_sample() -> None:
    """Selectbox callback: copy the chosen sample into the task text area."""
    choice = st.session_state.get("sample_task")
    if choice in SAMPLE_TASKS:
        st.session_state["task_input"] = SAMPLE_TASKS[choice]


if __name__ == "__main__":
    main()
