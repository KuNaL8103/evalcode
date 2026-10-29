"""Evaluation harness for evalcode (Task 13).

Provides quota-aware RAG on/off evaluation across demo tasks with resumability.
"""

from __future__ import annotations

import json
import os
import statistics
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

from evalcode.core.config import Settings
from evalcode.core.errors import ConfigError
from evalcode.core.observability import redact
from evalcode.graph import Dependencies, build_logger, run_task
from evalcode.state import AgentState

__all__ = [
    "MODES",
    "SCHEMA_VERSION",
    "EvalTask",
    "EvalOutcome",
    "load_tasks",
    "stop_reason",
    "result_path",
    "is_complete",
    "load_results",
    "build_record",
    "run_eval",
    "verify_references",
    "build_report",
    "write_report",
    "format_eval_line",
]

MODES = ("rag", "norag")
SCHEMA_VERSION = 1


class EvalTask(BaseModel):
    """A single evaluation task with task description, tests, and reference solution."""

    model_config = {"extra": "forbid"}

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    title: str
    libraries: list[str] = Field(min_length=1)
    difficulty: int = Field(ge=1, le=3)
    task: str
    tests: str
    reference: str

    @field_validator("task", "tests", "reference")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must be non-blank after strip")
        return v


def load_tasks(tasks_dir: Path, *, only: list[str] | None = None) -> list[EvalTask]:
    """Load evaluation tasks from YAML files in tasks_dir.

    Args:
        tasks_dir: Directory containing *.yaml task files.
        only: Optional list of task ids to filter. Unknown ids raise ConfigError.

    Returns:
        List of EvalTask objects sorted by id.

    Raises:
        ConfigError: If directory missing, empty, YAML invalid, validation fails,
            id != file stem, duplicate id, or unknown id in only.
    """
    if not tasks_dir.exists():
        raise ConfigError(f"tasks directory not found: {tasks_dir}")

    yaml_files = sorted(tasks_dir.glob("*.yaml"))
    if not yaml_files:
        raise ConfigError(f"no task files found in {tasks_dir}")

    tasks: dict[str, EvalTask] = {}
    for file_path in yaml_files:
        try:
            with open(file_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"invalid YAML in {file_path.name}: {e}") from e

        if data is None:
            raise ConfigError(f"empty YAML in {file_path.name}")

        try:
            task = EvalTask.model_validate(data)
        except Exception as e:
            raise ConfigError(f"validation error in {file_path.name}: {e}") from e

        if task.id != file_path.stem:
            raise ConfigError(
                f"id mismatch in {file_path.name}: id={task.id} != stem={file_path.stem}"
            )

        tasks[task.id] = task

    if only is not None:
        unknown = set(only) - set(tasks.keys())
        if unknown:
            known = sorted(tasks.keys())
            raise ConfigError(f"unknown task ids in --only: {sorted(unknown)}. Known ids: {known}")
        return [tasks[tid] for tid in only if tid in tasks]

    return sorted(tasks.values(), key=lambda t: t.id)


def stop_reason(state: AgentState) -> str | None:
    """Determine if run should stop due to quota/auth.

    Args:
        state: Final agent state.

    Returns:
        "quota" if DailyQuotaExceeded in history, "auth" if LLMAuthError or LLMModelError,
        None otherwise.
    """
    for event in state.get("history", []):
        summary = event.get("summary", {})
        error = summary.get("error")
        if error == "DailyQuotaExceeded":
            return "quota"
        if error in ("LLMAuthError", "LLMModelError"):
            return "auth"
    return None


def result_path(out_dir: Path, task_id: str, mode: str) -> Path:
    """Get path for raw result file: <out_dir>/raw/<task_id>__<mode>.json"""
    return out_dir / "raw" / f"{task_id}__{mode}.json"


def is_complete(record: dict) -> bool:
    """Check if a result record represents a completed run.

    Args:
        record: Parsed JSON record.

    Returns:
        True if schema_version==1 and status in {"approved","failed"}.
    """
    return record.get("schema_version") == SCHEMA_VERSION and record.get("status") in (
        "approved",
        "failed",
    )


def load_results(out_dir: Path) -> list[dict]:
    """Load all complete result files from out_dir/raw/*.json.

    Args:
        out_dir: Output directory containing raw/ subdirectory.

    Returns:
        List of parsed records sorted by (task_id, mode). Corrupt files are skipped.
    """
    raw_dir = out_dir / "raw"
    if not raw_dir.exists():
        return []

    records = []
    for file_path in sorted(raw_dir.glob("*.json")):
        try:
            with open(file_path, encoding="utf-8") as f:
                record = json.load(f)
            if record.get("schema_version") == SCHEMA_VERSION:
                records.append(record)
        except (json.JSONDecodeError, OSError):
            continue

    records.sort(key=lambda r: (r.get("task_id", ""), r.get("mode", "")))
    return records


def build_record(
    task: EvalTask,
    mode: str,
    state: AgentState,
    *,
    wall_s: float,
    model: str,
    run_id: str,
    secrets: tuple[str, ...],
) -> dict:
    """Build a result record from final agent state.

    Args:
        task: The evaluation task.
        mode: "rag" or "norag".
        state: Final agent state from run_task.
        wall_s: Wall-clock time in seconds.
        model: Model name used.
        run_id: Unique run identifier.
        secrets: Tuple of secret strings for redaction.

    Returns:
        Record dict with all required fields.
    """
    status = state.get("status", "failed")
    passed = status == "approved"
    attempt = state.get("attempt", 0)
    retries_used = state.get("retries_used", 0)
    human_rounds = state.get("human_rounds", 0)
    token_usage = state.get("token_usage", {})
    run_result = state.get("run_result") or {}

    # Count re-retrievals (retrieve events with mode == "error")
    re_retrievals = 0
    for event in state.get("history", []):
        if event.get("node") == "retrieve":
            summary = event.get("summary", {})
            if summary.get("mode") == "error":
                re_retrievals += 1

    retrieved_docs = state.get("retrieved_docs") or []
    doc_ids = [doc["id"] for doc in retrieved_docs]

    failure_reason = state.get("failure_reason")
    if failure_reason:
        failure_reason = redact(failure_reason, secrets, max_str=300)

    from evalcode.core.observability import utc_now_iso

    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": task.id,
        "mode": mode,
        "status": status,
        "passed": passed,
        "attempts": attempt,
        "retries_used": retries_used,
        "human_rounds": human_rounds,
        "first_try": passed and attempt == 1,
        "llm_calls": token_usage.get("llm_calls", 0),
        "api_retries": token_usage.get("api_retries", 0),
        "input_tokens": token_usage.get("input_tokens", 0),
        "output_tokens": token_usage.get("output_tokens", 0),
        "total_tokens": token_usage.get("total_tokens", 0),
        "wait_s": token_usage.get("wait_s", 0.0),
        "wall_s": round(wall_s, 2),
        "final_category": run_result.get("category"),
        "failure_reason": failure_reason,
        "n_docs": len(doc_ids),
        "doc_ids": doc_ids,
        "re_retrievals": re_retrievals,
        "final_code": state.get("final_code"),
        "model": model,
        "run_id": run_id,
        "finished_ts": utc_now_iso(),
    }


@dataclass
class EvalOutcome:
    """Outcome of run_eval."""

    executed: int
    skipped: int
    pending: int
    errors: int
    stopped: str | None  # None | "quota" | "auth" | "limit"


def run_eval(
    tasks: list[EvalTask],
    modes: list[str],
    deps: Dependencies,
    out_dir: Path,
    *,
    limit: int | None = None,
    force: bool = False,
    on_line: Callable[[str], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> EvalOutcome:
    """Run evaluation across tasks and modes.

    Args:
        tasks: List of EvalTask objects (sorted by id).
        modes: List of modes ("rag" and/or "norag").
        deps: Base Dependencies (shared llm; retriever may be None).
        out_dir: Output directory for raw results.
        limit: Max number of executed runs (not counting skipped).
        force: Re-run even if result file exists and is complete.
        on_line: Optional callback for per-run output line.
        clock: Time function (default: time.monotonic).

    Returns:
        EvalOutcome with counts and stop reason.
    """
    # Validate modes
    for mode in modes:
        if mode not in MODES:
            raise ConfigError(f"invalid mode: {mode}. Must be one of {MODES}")

    # RAG mode requires retriever
    need_rag = "rag" in modes
    if need_rag and deps.retriever is None:
        raise ConfigError("RAG mode requested but no usable index")

    # Get secrets for redaction (API keys)
    secrets = []
    if deps.settings.gemini_api_key:
        secrets.append(deps.settings.gemini_api_key.get_secret_value())
    if deps.settings.langsmith_api_key:
        secrets.append(deps.settings.langsmith_api_key.get_secret_value())
    secrets_tuple = tuple(secrets)

    executed = 0
    skipped = 0
    errors = 0
    stopped: str | None = None

    # Determine all runs to execute
    all_runs = [(task, mode) for task in tasks for mode in modes]
    pending_runs = len(all_runs)

    for task, mode in all_runs:
        run_id = f"eval-{task.id}-{mode}-{uuid.uuid4().hex[:8]}"
        result_file = result_path(out_dir, task.id, mode)

        # Check if we should skip
        if not force and result_file.exists():
            try:
                with open(result_file, encoding="utf-8") as f:
                    existing = json.load(f)
                if is_complete(existing):
                    skipped += 1
                    pending_runs -= 1
                    if on_line:
                        on_line(f"eval {task.id} {mode}: skip (done)")
                    continue
            except (json.JSONDecodeError, OSError):
                pass  # Corrupt file, will re-run

        # Check limit
        if limit is not None and executed >= limit:
            stopped = "limit"
            break

        # Build deps for this run
        run_deps = replace(
            deps,
            retriever=deps.retriever if mode == "rag" else None,
            logger=build_logger(deps.settings, run_id),
        )

        start = clock()
        try:
            state = run_task(
                task=task.task,
                deps=run_deps,
                task_id=run_id,
                provided_tests=task.tests,
                auto_approve=True,
                checkpointer=None,
            )
        except Exception as exc:
            # Record error but continue
            elapsed = clock() - start
            from evalcode.core.observability import utc_now_iso

            record = {
                "schema_version": SCHEMA_VERSION,
                "task_id": task.id,
                "mode": mode,
                "status": "error",
                "failure_reason": f"{type(exc).__name__}: "
                f"{redact(str(exc), secrets_tuple, max_str=300)}",
                "run_id": run_id,
                "model": deps.settings.llm_model,
                "finished_ts": utc_now_iso(),
            }
            # Write error record
            result_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_file = result_file.with_suffix(".tmp")
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            os.replace(tmp_file, result_file)

            errors += 1
            pending_runs -= 1
            executed += 1
            if on_line:
                on_line(f"eval {task.id} {mode}: ERROR {type(exc).__name__}")
            continue

        elapsed = clock() - start

        # Check for quota/auth stop
        r = stop_reason(state)
        if r:
            stopped = r
            # Do NOT count the quota/auth run as executed (no result file written)
            # Do NOT decrement pending_runs for this run (it remains pending)
            if on_line:
                on_line(f"eval {task.id} {mode}: STOPPED {r}")
            break

        # Build and write record
        record = build_record(
            task=task,
            mode=mode,
            state=state,
            wall_s=elapsed,
            model=deps.settings.llm_model,
            run_id=run_id,
            secrets=secrets_tuple,
        )

        result_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = result_file.with_suffix(".tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        os.replace(tmp_file, result_file)

        executed += 1
        pending_runs -= 1

        if on_line:
            if record["status"] == "approved":
                on_line(
                    f"eval {task.id} {mode}: PASS attempts={record['attempts']} "
                    f"calls={record['llm_calls']} tokens={record['total_tokens']} "
                    f"wall={record['wall_s']}s"
                )
            else:
                on_line(
                    f"eval {task.id} {mode}: FAIL category={record['final_category']} "
                    f"attempts={record['attempts']} calls={record['llm_calls']} "
                    f"tokens={record['total_tokens']} wall={record['wall_s']}s"
                )

    return EvalOutcome(
        executed=executed,
        skipped=skipped,
        pending=pending_runs,
        errors=errors,
        stopped=stopped,
    )


def verify_references(
    tasks: list[EvalTask],
    sandbox: Callable[[str, str, float, int], dict],
    settings: Settings,
) -> list[dict]:
    """Verify reference solutions against their tests.

    Args:
        tasks: List of evaluation tasks.
        sandbox: Sandbox function to run code.
        settings: Settings for timeout and memory limits.

    Returns:
        List of result dicts with id, passed, category, tests_total, tests_failed, duration_s.
    """
    results = []
    for task in tasks:
        try:
            run_result = sandbox(
                task.reference,
                task.tests,
                timeout_s=float(settings.sandbox_timeout_s),
                mem_mb=int(settings.sandbox_mem_mb),
            )
            results.append(
                {
                    "id": task.id,
                    "passed": run_result.get("passed", False),
                    "category": run_result.get("category", "sandbox_error"),
                    "tests_total": run_result.get("tests_total", 0),
                    "tests_failed": run_result.get("tests_failed", 0),
                    "duration_s": run_result.get("duration_s", 0.0),
                }
            )
        except Exception:
            results.append(
                {
                    "id": task.id,
                    "passed": False,
                    "category": "sandbox_error",
                    "tests_total": 0,
                    "tests_failed": 0,
                    "duration_s": 0.0,
                }
            )
    return results


def format_eval_line(record: dict) -> str:
    """Format a single-line summary for a completed run.

    Args:
        record: Result record from build_record.

    Returns:
        Formatted line string.
    """
    if record.get("status") == "approved":
        return (
            f"eval {record['task_id']} {record['mode']}: "
            f"PASS attempts={record['attempts']} calls={record['llm_calls']} "
            f"tokens={record['total_tokens']} wall={record['wall_s']}s"
        )
    elif record.get("status") == "failed":
        return (
            f"eval {record['task_id']} {record['mode']}: "
            f"FAIL category={record['final_category']} attempts={record['attempts']} "
            f"calls={record['llm_calls']} tokens={record['total_tokens']} "
            f"wall={record['wall_s']}s"
        )
    else:
        return (
            f"eval {record['task_id']} {record['mode']}: "
            f"ERROR {record.get('failure_reason', 'unknown')}"
        )


def build_report(results: list[dict], tasks: list[EvalTask], *, model: str | None = None) -> str:
    """Build markdown evaluation report.

    Args:
        results: List of result records (from load_results or build_record).
        tasks: List of all tasks (for pairing and task order).
        model: Model name for header.

    Returns:
        Markdown report string.
    """
    # Filter complete results
    complete = [r for r in results if is_complete(r)]
    error_results = [r for r in results if r.get("status") == "error"]

    # Group by mode
    rag_results = [r for r in complete if r["mode"] == "rag"]
    norag_results = [r for r in complete if r["mode"] == "norag"]

    # Count tasks without both modes complete
    rag_complete_tasks = {r["task_id"] for r in rag_results}
    norag_complete_tasks = {r["task_id"] for r in norag_results}
    incomplete_pairs = sum(
        1
        for task in tasks
        if task.id not in rag_complete_tasks or task.id not in norag_complete_tasks
    )

    lines = []
    lines.append("# evalcode evaluation report")
    model_str = model or "unknown"
    lines.append(
        f"{model_str} | tasks {len(tasks)} | complete runs {len(complete)} | "
        f"error runs {len(error_results)} | incomplete pairs {incomplete_pairs}"
    )
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| mode | runs | passed | pass rate | first-try | mean attempts | "
        "mean LLM calls | mean tokens | median wall s |"
    )
    lines.append(
        "|------|------|--------|-----------|-----------|---------------|----------------|-------------|---------------|"
    )

    def mode_summary(mode_results: list[dict], mode_name: str) -> None:
        if not mode_results:
            return
        runs = len(mode_results)
        passed = sum(1 for r in mode_results if r["passed"])
        pass_rate = passed / runs if runs else 0
        first_try = sum(1 for r in mode_results if r.get("first_try", False))
        mean_attempts = sum(r["attempts"] for r in mode_results) / runs
        mean_llm_calls = sum(r["llm_calls"] for r in mode_results) / runs
        mean_tokens = sum(r["total_tokens"] for r in mode_results) / runs
        walls = [r["wall_s"] for r in mode_results]
        median_wall = statistics.median(walls) if walls else 0.0

        lines.append(
            f"| {mode_name} | {runs} | {passed} | {pass_rate:.1%} | {first_try} | "
            f"{mean_attempts:.2f} | {mean_llm_calls:.2f} | {mean_tokens:.0f} | {median_wall:.1f} |"
        )

    mode_summary(rag_results, "rag")
    mode_summary(norag_results, "norag")
    lines.append("")

    # Paired comparison
    lines.append("## Paired comparison (K tasks with both modes complete)")
    lines.append("")

    # Find tasks with both modes complete
    rag_by_task = {r["task_id"]: r for r in rag_results}
    norag_by_task = {r["task_id"]: r for r in norag_results}
    paired_task_ids = sorted(set(rag_by_task.keys()) & set(norag_by_task.keys()))

    if not paired_task_ids:
        lines.append("no complete pairs yet")
        lines.append("")
    else:
        both_pass = 0
        rag_only = 0
        norag_only = 0
        neither = 0
        llm_calls_diffs = []
        attempts_diffs = []

        for tid in paired_task_ids:
            r_r = rag_by_task[tid]
            r_n = norag_by_task[tid]
            rp = r_r["passed"]
            np = r_n["passed"]
            if rp and np:
                both_pass += 1
            elif rp and not np:
                rag_only += 1
            elif not rp and np:
                norag_only += 1
            else:
                neither += 1
            llm_calls_diffs.append(r_r["llm_calls"] - r_n["llm_calls"])
            attempts_diffs.append(r_r["attempts"] - r_n["attempts"])

        lines.append(
            f"both pass: {both_pass}, rag only: {rag_only}, "
            f"norag only: {norag_only}, neither: {neither}"
        )
        lines.append("")
        mean_llm_diff = sum(llm_calls_diffs) / len(llm_calls_diffs)
        mean_attempts_diff = sum(attempts_diffs) / len(attempts_diffs)
        lines.append(f"mean LLM calls (rag - norag): {mean_llm_diff:+.2f}")
        lines.append(f"mean attempts (rag - norag): {mean_attempts_diff:+.2f}")
        lines.append("")

        if len(paired_task_ids) < 10:
            lines.append(
                "K < 10 pairs: differences are indicative only, "
                f"not statistically meaningful (K={len(paired_task_ids)})"
            )
            lines.append("")

    # Per task table - include error results in cell display
    lines.append("## Per task")
    lines.append("")
    lines.append("| id | libs | rag | norag |")
    lines.append("|----|------|-----|-------|")

    # Build mappings that include error results for per-task display
    rag_by_task_all = {r["task_id"]: r for r in complete if r["mode"] == "rag"}
    norag_by_task_all = {r["task_id"]: r for r in complete if r["mode"] == "norag"}
    # Add error results (they don't appear in complete)
    for r in error_results:
        if r["mode"] == "rag":
            rag_by_task_all[r["task_id"]] = r
        elif r["mode"] == "norag":
            norag_by_task_all[r["task_id"]] = r

    for task in tasks:
        r_r = rag_by_task_all.get(task.id)
        r_n = norag_by_task_all.get(task.id)

        def cell(r: dict | None) -> str:
            if r is None:
                return "-"
            if r.get("status") == "error":
                return "error"
            if r["passed"]:
                return f"PASS a={r['attempts']} c={r['llm_calls']}"
            return f"FAIL a={r['attempts']} c={r['llm_calls']}"

        libs = ", ".join(task.libraries)
        lines.append(f"| {task.id} | {libs} | {cell(r_r)} | {cell(r_n)} |")

    lines.append("")

    # Caveats
    lines.append("## Caveats")
    lines.append("")
    lines.append("- Sampling temperature > 0; results may vary between runs.")
    lines.append(
        '- The model sees the acceptance tests; pass rate measures "can satisfy the visible spec".'
    )
    lines.append("- RAG's effect shows mainly in attempts / LLM calls / first-try rate.")
    if len(paired_task_ids) < 10 and paired_task_ids:
        lines.append(
            f"- K < 10 pairs ({len(paired_task_ids)}): differences are indicative only, "
            "not statistically meaningful."
        )
    if error_results:
        error_ids = [r["task_id"] for r in error_results]
        lines.append(f"- Tasks with errors: {', '.join(sorted(set(error_ids)))}")
    lines.append("")

    return "\n".join(lines)


def write_report(out_dir: Path, tasks: list[EvalTask], *, model: str | None = None) -> Path:
    """Write evaluation report to <out_dir>/report.md.

    Args:
        out_dir: Output directory.
        tasks: List of all tasks.
        model: Model name for header.

    Returns:
        Path to written report.md.
    """
    results = load_results(out_dir)
    report = build_report(results, tasks, model=model)
    report_path = out_dir / "report.md"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    return report_path
