"""Typer CLI for evalcode: ingest, search, run, resume."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.syntax import Syntax

from evalcode.config import Settings
from evalcode.errors import ConfigError
from evalcode.graph import close_dependencies, pending_review
from evalcode.state import AgentState

# Exit codes
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_QUOTA = 3
EXIT_CONFIG = 4
EXIT_PAUSED = 5

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="evalcode: iterative code generation agent with RAG + sandbox",
)


def _make_stdio_tolerant() -> None:
    """Make stdout/stderr tolerant of cp1252 consoles (Windows)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except Exception:
                pass


# Run once at import time
_make_stdio_tolerant()


def _safe(text: str) -> str:
    """Replace characters the stdout encoding cannot represent."""
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        return text.encode(enc, "replace").decode(enc)
    except Exception:
        return text.encode("utf-8", "replace").decode("utf-8")


def _console() -> Console:
    """Create a Console with safe defaults."""
    return Console(highlight=False, markup=False, soft_wrap=True)


# Seam functions (monkeypatched in tests)
def _get_settings() -> Settings:
    from evalcode.config import get_settings

    return get_settings()


def _build_deps(settings: Settings, *, rag: bool, run_id: str) -> Any:
    from evalcode.graph import default_dependencies

    return default_dependencies(settings, rag=rag, observe=True, run_id=run_id)


def _open_checkpointer(path: str | Path) -> Any:
    from evalcode.persistence import open_checkpointer

    return open_checkpointer(path)


def _build_search_retriever(settings: Settings) -> Any:
    from evalcode.rag.embeddings import get_embedder
    from evalcode.rag.retriever import Retriever
    from evalcode.rag.store import VectorStore

    embedder = get_embedder(settings)
    store = VectorStore(settings.chroma_dir, settings.collection_name, embedder)
    return Retriever(store, settings.retrieval_top_k, settings.retrieval_min_score)


def _build_ingest_store(settings: Settings) -> Any:
    from evalcode.rag.embeddings import get_embedder
    from evalcode.rag.store import VectorStore

    embedder = get_embedder(settings)
    return VectorStore(settings.chroma_dir, settings.collection_name, embedder)


# Pure helpers (unit-tested)
def format_event(node: str, update: Any) -> str:
    """Format one streamed (node, update) pair as a single ASCII line (max 160 chars)."""
    summary = {}
    attempt = 0
    # update may be a dict, or an interrupt value (tuple/Interrupt) when graph pauses
    if isinstance(update, dict):
        if update.get("history"):
            first_hist = update["history"][0]
            if isinstance(first_hist, dict):
                summary = first_hist.get("summary", {})
                attempt = first_hist.get("attempt", 0)

        # Handle interrupt in update dict
        if update.get("__interrupt__") is not None:
            return "[human_review] PAUSED: awaiting human review"
    elif node == "__interrupt__":
        # Stream yielded an interrupt chunk: node="__interrupt__", update=interrupt value
        return "[human_review] PAUSED: awaiting human review"

    # Route based on node
    if node == "retrieve":
        mode = summary.get("mode", "")
        queries = summary.get("queries", [])
        docs = summary.get("docs", [])
        top_score = summary.get("top_score")
        error = summary.get("error")
        parts = [f"[retrieve] mode={mode} queries={len(queries)} docs={len(docs)}"]
        if top_score is not None:
            parts.append(f"top={top_score:.3f}")
        else:
            parts.append("top=-")
        if error:
            parts.append(f"error={error}")
        return " ".join(parts)

    if node in ("generate", "revise"):
        code = summary.get("code", "") or ""
        tests = summary.get("tests", "") or ""
        error = summary.get("error")
        parts = [f"[{node}] attempt={attempt} code={len(code)} chars tests={len(tests)} chars"]
        if error:
            parts.append(f"FAILED error={error}")
        return " ".join(parts)

    if node == "run_tests":
        run_result = update.get("run_result") or {}
        passed = run_result.get("passed", False)
        category = run_result.get("category", "")
        total = run_result.get("tests_total", 0)
        failed = run_result.get("tests_failed", 0)
        duration = run_result.get("duration_s", 0.0)
        status = "PASS" if passed else "FAIL"
        return (
            f"[run_tests] attempt={attempt} {status} category={category} "
            f"tests={total} failed={failed} time={duration:.1f}s"
        )

    if node == "analyze_error":
        category = summary.get("category", "")
        fault = summary.get("fault", "")
        needs_docs = summary.get("needs_docs", False)
        return f"[analyze_error] category={category} fault={fault} needs_docs={needs_docs}"

    if node == "human_review":
        decision = summary.get("decision", "")
        rounds = summary.get("human_rounds", 0)
        return f"[human_review] decision={decision} round={rounds}"

    if node == "finalize":
        final_code = update.get("final_code") or ""
        return f"[finalize] approved final_code={len(final_code)} chars"

    if node == "fail":
        reason = update.get("failure_reason") or ""
        return f"[fail] {reason[:120]}"

    return f"[{node}] done"


def classify_exit(state: AgentState) -> int:
    """Map final AgentState to an exit code."""
    # Paused for review
    interrupt = state.get("__interrupt__")
    if interrupt:
        return EXIT_PAUSED

    status = state.get("status")
    if status == "approved":
        return EXIT_OK
    if status == "failed":
        # Check for quota exhaustion in history
        for event in state.get("history", []):
            summary = event.get("summary", {})
            if summary.get("error") == "DailyQuotaExceeded":
                return EXIT_QUOTA
        return EXIT_FAILED
    return EXIT_FAILED


# Commands: search, ingest
@app.command()
def search(
    query: str,
    k: int = typer.Option(5, "-k", "--top-k", help="Max results to return"),
) -> None:
    """Search the indexed Python documentation."""
    settings = _get_settings()
    retriever = _build_search_retriever(settings)
    try:
        results = retriever.retrieve([query], k=k)
    finally:
        store = getattr(retriever, "store", None)
        close_fn = getattr(store, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass

    if not results:
        console = _console()
        console.print(_safe(f"no results for: {query} (run `evalcode ingest` first?)"))
        raise typer.Exit(1)

    console = _console()
    for i, doc in enumerate(results, start=1):
        console.print(_safe(f"[{i}] score={doc['score']:.3f}  {doc['qualname']}"))
        text = doc["text"].replace("\n", " ")[:300]
        console.print(_safe(text))
        console.print()


@app.command()
def ingest(
    libs: Annotated[
        str | None,
        typer.Option("--libs", help="Comma-separated libraries (default: DOC_LIBRARIES)"),
    ] = None,
    docs_dir: Annotated[
        Path | None,
        typer.Option("--docs-dir", help="Directory of .md/.rst/.txt docs"),
    ] = None,
    reset: Annotated[
        bool,
        typer.Option("--reset", help="Clear the store before ingesting"),
    ] = False,
) -> None:
    """Ingest libraries and/or docs into the vector store."""
    settings = _get_settings()

    if libs:
        libraries = [item.strip() for item in libs.split(",") if item.strip()]
    else:
        libraries = list(settings.doc_libraries)

    if docs_dir:
        docs_path = docs_dir
    else:
        default_docs = Path(settings.docs_dir)
        docs_path = default_docs if default_docs.exists() else None

    store = _build_ingest_store(settings)
    try:
        from evalcode.rag.ingest import ingest as do_ingest

        report = do_ingest(libraries, store, docs_dir=docs_path, reset=reset)
    finally:
        store.close()

    console = _console()
    for library, count in report["per_library"].items():
        console.print(_safe(f"{library}: {count} chunks"))
    for library in report["skipped"]:
        console.print(_safe(f"{library}: skipped (no chunks)"))
    console.print(
        _safe(
            f"total: {report['total']} chunks in {report['seconds']:.1f}s -> {settings.chroma_dir}"
        )
    )


# Helper functions for run/resume
def _prompt_decision(
    console: Console, payload: dict[str, Any], *, prompt: Any = None
) -> dict[str, Any]:
    """Interactive prompt for human review decision."""
    if prompt is None:
        prompt = typer.prompt

    # Show the review
    _render_review(console, payload)

    while True:
        choice = (
            prompt(
                _safe("Decision (a/approve, r/reject, e/edit)"),
                type=str,
            )
            .strip()
            .lower()
        )

        if choice in ("a", "approve"):
            return {"decision": "approve"}

        if choice in ("r", "reject"):
            while True:
                feedback = prompt(_safe("Feedback"), type=str).strip()
                if feedback:
                    return {"decision": "reject", "feedback": feedback}
                console.print(_safe("Feedback cannot be empty. Please provide feedback."))

        if choice in ("e", "edit"):
            while True:
                edit_path = prompt(_safe("Path to edited code file"), type=str).strip()
                if not edit_path:
                    console.print(_safe("Path cannot be empty."))
                    continue
                try:
                    code = Path(edit_path).read_text(encoding="utf-8")
                    if code.strip():
                        return {"decision": "edit", "code": code}
                    console.print(_safe("File is empty."))
                except FileNotFoundError:
                    console.print(_safe(f"File not found: {edit_path}"))
                except Exception as e:
                    console.print(_safe(f"Error reading file: {e}"))

        console.print(_safe("Invalid choice. Enter a, r, or e."))


def _render_stream(console: Console, node: str, update: dict[str, Any]) -> None:
    """Render a single streamed event."""
    line = format_event(node, update)
    console.print(_safe(line))


def _render_review(console: Console, payload: dict[str, Any]) -> None:
    """Render the human review payload."""
    console.print()
    console.print(_safe("=" * 60))
    console.print(_safe("HUMAN REVIEW REQUIRED"))
    console.print(_safe("=" * 60))
    console.print(_safe(f"Task ID: {payload.get('task_id', '')}"))
    console.print(_safe(f"Attempt: {payload.get('attempt', 0)}"))
    hr = payload.get("human_round", 0)
    mhr = payload.get("max_human_rounds", 2)
    console.print(_safe(f"Human round: {hr} / {mhr}"))
    console.print()
    console.print(_safe("Explanation:"))
    console.print(_safe(payload.get("explanation", "")))
    console.print()
    run_summary = payload.get("run_summary", {})
    cat = run_summary.get("category", "?")
    tot = run_summary.get("tests_total", 0)
    fail = run_summary.get("tests_failed", 0)
    dur = run_summary.get("duration_s", 0.0)
    console.print(_safe(f"Run summary: category={cat} tests={tot} failed={fail} time={dur:.1f}s"))
    console.print()
    console.print(_safe("Code:"))
    code = payload.get("code", "")
    if code:
        console.print(Syntax(_safe(code), "python", theme="monokai", word_wrap=True))
    else:
        console.print(_safe("(no code)"))
    console.print(_safe("=" * 60))
    console.print()


def _render_final(
    console: Console, state: AgentState, *, show_code: bool, output_path: Path | None
) -> None:
    """Render final status and optionally write code to file."""
    status = state.get("status", "?")
    attempt = state.get("attempt", 0)
    retries = state.get("retries_used", 0)
    human_rounds = state.get("human_rounds", 0)
    token_usage = state.get("token_usage", {})
    llm_calls = token_usage.get("llm_calls", 0)
    total_tokens = token_usage.get("total_tokens", 0)

    # Find run_dir from logger (logs/<task_id>)
    task_id = state.get("task_id", "")
    run_dir = f"logs/{task_id}" if task_id else "logs/unknown"

    console.print()
    console.print(_safe("=" * 60))
    console.print(
        _safe(
            f"STATUS: {status} | attempts={attempt} retries_used={retries} "
            f"human_rounds={human_rounds} | llm_calls={llm_calls} "
            f"total_tokens={total_tokens} | log: {run_dir}"
        )
    )

    failure_reason = state.get("failure_reason")
    if status == "failed" and failure_reason:
        console.print(_safe(f"FAILED: {failure_reason}"))

    if status == "approved" and show_code:
        final_code = state.get("final_code") or ""
        console.print()
        console.print(_safe("Final code:"))
        console.print(Syntax(_safe(final_code), "python", theme="monokai", word_wrap=True))

    if output_path and status == "approved":
        final_code = state.get("final_code") or ""
        if final_code:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(final_code, encoding="utf-8")
            console.print(_safe(f"Written to {output_path}"))


def _drive(
    task: str,
    deps: Any,
    task_id: str,
    provided_tests: str | None,
    auto_approve: bool,
    checkpointer: Any,
    *,
    interactive: bool,
    output_path: Path | None,
    show_code: bool,
    resume_mode: bool = False,
    resume_response: dict[str, Any] | None = None,
) -> int:
    """Shared driver for run and resume."""
    from evalcode.graph import (
        close_dependencies,
        get_task_state,
        pending_review,
        stream_resume_task,
        stream_task,
    )

    console = _console()

    try:
        if resume_mode:
            # For resume mode, we stream the resume with the provided response
            if resume_response is None:
                console.print(_safe("Error: resume_mode requires resume_response"))
                return EXIT_FAILED
            for node, update in stream_resume_task(
                task_id, resume_response, deps, checkpointer=checkpointer
            ):
                _render_stream(console, node, update)
        else:
            # Initial stream for run mode
            for node, update in stream_task(
                task,
                deps,
                task_id=task_id,
                provided_tests=provided_tests,
                auto_approve=auto_approve,
                checkpointer=checkpointer,
            ):
                _render_stream(console, node, update)

        # Read final state from checkpoint
        state = get_task_state(task_id, deps.settings, checkpointer=checkpointer)
        if state is None:
            console.print(_safe("Error: could not read final state from checkpoint"))
            return EXIT_FAILED

        # Write summary
        if deps.logger is not None:
            deps.logger.write_summary(state)

        # Handle pause / human review
        while state.get("__interrupt__"):
            payload = pending_review(state)
            if payload is None:
                console.print(_safe("Error: interrupted but no review payload"))
                return EXIT_FAILED

            # Always render the review
            _render_review(console, payload)

            if not interactive:
                console.print(_safe("AWAITING REVIEW. Resume with:"))
                resume_cmd = (
                    f"  evalcode resume {task_id} "
                    f'--approve | --reject "feedback" | --edit-file PATH'
                )
                console.print(_safe(resume_cmd))
                return EXIT_PAUSED

            # Interactive prompt
            response = _prompt_decision(console, payload)

            # Stream the resume
            for node, update in stream_resume_task(
                task_id, response, deps, checkpointer=checkpointer
            ):
                _render_stream(console, node, update)

            # Read updated state
            state = get_task_state(task_id, deps.settings, checkpointer=checkpointer)
            if state is None:
                console.print(_safe("Error: could not read state after resume"))
                return EXIT_FAILED

            if deps.logger is not None:
                deps.logger.write_summary(state)

        # Not paused - render final output
        _render_final(console, state, show_code=show_code, output_path=output_path)
        return classify_exit(state)

    except typer.Exit as e:
        return e.exit_code
    except Exception as e:
        # Redact error message
        settings = _get_settings()
        secrets = []
        if settings.gemini_api_key:
            secrets.append(settings.gemini_api_key.get_secret_value())
        if settings.langsmith_api_key:
            secrets.append(settings.langsmith_api_key.get_secret_value())
        from evalcode.observability import redact

        msg = redact(str(e), tuple(secrets))
        console.print(_safe(f"Unexpected error: {type(e).__name__}: {msg}"))
        return EXIT_FAILED
    finally:
        close_dependencies(deps)


# Command: run
_TASK_FILE_DEFAULT = None
_TESTS_FILE_DEFAULT = None
_AUTO_APPROVE_DEFAULT = False
_NO_RAG_DEFAULT = False
_INTERACTIVE_DEFAULT = None  # None means auto-detect (tty)
_OUTPUT_DEFAULT = None
_SHOW_CODE_DEFAULT = True
_NO_SHOW_CODE_DEFAULT = False


@app.command()
def run(
    task: Annotated[
        str | None,
        typer.Argument(help="Task description (or use --task-file)"),
    ] = None,
    task_file: Annotated[
        Path | None,
        typer.Option("--task-file", help="Read task from file (utf-8)"),
    ] = _TASK_FILE_DEFAULT,
    tests_file: Annotated[
        Path | None,
        typer.Option("--tests", help="Read provided tests from file (utf-8)"),
    ] = _TESTS_FILE_DEFAULT,
    auto_approve: Annotated[
        bool,
        typer.Option("--auto-approve", help="Skip human review (CI/eval mode)"),
    ] = _AUTO_APPROVE_DEFAULT,
    no_rag: Annotated[
        bool,
        typer.Option("--no-rag", help="Disable RAG retrieval"),
    ] = _NO_RAG_DEFAULT,
    interactive: Annotated[
        bool | None,
        typer.Option(
            "--interactive/--no-interactive",
            help="Force interactive mode on/off (default: auto from stdin.isatty())",
        ),
    ] = _INTERACTIVE_DEFAULT,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Write final code to file (utf-8)"),
    ] = _OUTPUT_DEFAULT,
    show_code: Annotated[
        bool,
        typer.Option("--show-code/--no-show-code", help="Show final code in output"),
    ] = _SHOW_CODE_DEFAULT,
) -> None:
    """Run a code generation task."""
    # Validate mutually exclusive task sources
    if task and task_file:
        raise typer.BadParameter("Provide either TASK argument or --task-file, not both")
    if not task and not task_file:
        raise typer.BadParameter("Provide either TASK argument or --task-file")

    # Read task from file if provided
    if task_file:
        try:
            task = task_file.read_text(encoding="utf-8").strip()
        except Exception as e:
            raise typer.BadParameter(f"Cannot read --task-file: {e}") from None

    # Read tests from file if provided
    provided_tests = None
    if tests_file:
        try:
            provided_tests = tests_file.read_text(encoding="utf-8")
        except Exception as e:
            raise typer.BadParameter(f"Cannot read --tests file: {e}") from None

    # Determine interactive mode
    if interactive is None:
        interactive = sys.stdin.isatty()

    settings = _get_settings()
    task_id = uuid.uuid4().hex
    console = _console()
    console.print(_safe(f"TASK_ID: {task_id}"))

    # Build deps
    try:
        deps = _build_deps(settings, rag=not no_rag, run_id=task_id)
    except ConfigError as e:
        console.print(_safe(str(e)))
        raise typer.Exit(EXIT_CONFIG) from None
    except Exception as e:
        # Redact error message
        secrets = []
        if settings.gemini_api_key:
            secrets.append(settings.gemini_api_key.get_secret_value())
        if settings.langsmith_api_key:
            secrets.append(settings.langsmith_api_key.get_secret_value())
        from evalcode.observability import redact

        msg = redact(str(e), tuple(secrets))
        console.print(_safe(f"Unexpected error: {type(e).__name__}: {msg}"))
        raise typer.Exit(EXIT_FAILED) from None

    if not no_rag and deps.retriever is None:
        console.print(_safe("RAG disabled: no usable index (run `evalcode ingest`)"))

    # Open checkpointer (always, even with --auto-approve)
    with _open_checkpointer(settings.checkpoint_db) as checkpointer:
        # Drive
        exit_code = _drive(
            task=task,
            deps=deps,
            task_id=task_id,
            provided_tests=provided_tests,
            auto_approve=auto_approve,
            checkpointer=checkpointer,
            interactive=interactive,
            output_path=output,
            show_code=show_code,
            resume_mode=False,
        )
    raise typer.Exit(exit_code)


# Command: resume
_RESUME_APPROVE_DEFAULT = False
_RESUME_REJECT_DEFAULT = None
_RESUME_EDIT_FILE_DEFAULT = None


@app.command()
def resume(
    task_id: Annotated[str, typer.Argument(help="Task ID to resume")],
    approve: Annotated[
        bool,
        typer.Option("--approve", help="Approve the current solution"),
    ] = _RESUME_APPROVE_DEFAULT,
    reject: Annotated[
        str | None,
        typer.Option("--reject", help="Reject with feedback text"),
    ] = _RESUME_REJECT_DEFAULT,
    edit_file: Annotated[
        Path | None,
        typer.Option("--edit-file", help="Path to file with edited code"),
    ] = _RESUME_EDIT_FILE_DEFAULT,
    interactive: Annotated[
        bool | None,
        typer.Option(
            "--interactive/--no-interactive",
            help="Force interactive mode on/off (default: auto from stdin.isatty())",
        ),
    ] = _INTERACTIVE_DEFAULT,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Write final code to file (utf-8)"),
    ] = _OUTPUT_DEFAULT,
    show_code: Annotated[
        bool,
        typer.Option("--show-code/--no-show-code", help="Show final code in output"),
    ] = _SHOW_CODE_DEFAULT,
) -> None:
    """Resume a paused task after human review."""
    # Validate decision options (at most one)
    decision_count = sum(1 for x in (approve, reject is not None, edit_file is not None) if x)
    if decision_count > 1:
        raise typer.BadParameter("Provide at most one of --approve, --reject, --edit-file")

    settings = _get_settings()
    console = _console()

    # Open checkpointer and read state
    with _open_checkpointer(settings.checkpoint_db) as checkpointer:
        from evalcode.graph import get_task_state, rag_enabled_in

        state = get_task_state(task_id, settings, checkpointer=checkpointer)
        if state is None:
            console.print(_safe(f"unknown task id: {task_id}"))
            raise typer.Exit(EXIT_FAILED)

        interrupt = state.get("__interrupt__")
        if not interrupt:
            status = state.get("status", "?")
            console.print(_safe(f"task {task_id} is not awaiting review (status: {status})"))
            raise typer.Exit(EXIT_FAILED)

        # Determine if RAG was enabled in the original run
        rag = rag_enabled_in(state)

        # Build deps with same RAG topology
        deps = _build_deps(settings, rag=rag, run_id=task_id)
        if rag and deps.retriever is None:
            console.print(
                _safe(
                    "this task was started with RAG but the index is unavailable; "
                    "refusing to resume with a different graph topology"
                )
            )
            close_dependencies(deps)
            raise typer.Exit(EXIT_CONFIG) from None

        # Determine interactive mode
        if interactive is None:
            interactive = sys.stdin.isatty()

        # Build response
        response = None
        if approve:
            response = {"decision": "approve"}
        elif reject is not None:
            response = {"decision": "reject", "feedback": reject}
        elif edit_file is not None:
            try:
                code = edit_file.read_text(encoding="utf-8")
                if not code.strip():
                    console.print(_safe("Edit file is empty"))
                    raise typer.Exit(EXIT_FAILED) from None
                response = {"decision": "edit", "code": code}
            except Exception as e:
                console.print(_safe(f"Cannot read --edit-file: {e}"))
                raise typer.Exit(EXIT_FAILED) from None
        elif not interactive:
            msg = "No decision option provided. Use --approve, --reject, or --edit-file."
            console.print(_safe(msg))
            raise typer.Exit(EXIT_USAGE) from None

        # If no decision yet and interactive, prompt
        if response is None and interactive:
            payload = pending_review(state)
            if payload:
                response = _prompt_decision(console, payload)

        if response is None:
            console.print(_safe("No decision provided"))
            raise typer.Exit(EXIT_USAGE)

        # Drive resume
        exit_code = _drive(
            task="",  # not used in resume mode
            deps=deps,
            task_id=task_id,
            provided_tests=None,
            auto_approve=False,
            checkpointer=checkpointer,
            interactive=interactive,
            output_path=output,
            show_code=show_code,
            resume_mode=True,
            resume_response=response,
        )
    raise typer.Exit(exit_code)
