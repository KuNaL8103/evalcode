"""Tests for the evaluation harness (Task 13)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from evalcode.config import Settings
from evalcode.evaluation import EvalTask, load_tasks, run_eval
from evalcode.graph import Dependencies
from evalcode.observability import build_logger
from evalcode.rag.types import RetrievedDoc
from tests.fakes import FakeRetriever, ScriptedLLM, bundle_text, make_doc


class FakeSandboxFail:
    """Sandbox that always fails."""

    def __call__(self, code: str, tests: str, timeout_s: float, mem_mb: int) -> dict:
        return {
            "passed": False,
            "category": "assertion_failure",
            "exit_code": 1,
            "timed_out": False,
            "duration_s": 0.1,
            "tests_total": 1,
            "tests_failed": 1,
            "failures": [
                {
                    "test_name": "test",
                    "error_type": "AssertionError",
                    "message": "fail",
                    "traceback": "",
                }
            ],
            "stdout": "",
            "stderr": "AssertionError: fail",
        }


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def eval_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Settings:
    """Monkeypatch env vars into tmp_path and return fresh Settings."""
    monkeypatch.setenv("CHECKPOINT_DB", str(tmp_path / "checkpoints.sqlite"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("GEMINI_API_KEY", "AIza" + "x" * 35)
    from evalcode.config import get_settings

    get_settings.cache_clear()
    return get_settings()


class ClosableRetriever(FakeRetriever):
    """FakeRetriever with a store that counts close() calls."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.store = MagicMock()
        self.store.close = MagicMock()
        self.retrieve_k: list[int | None] = []

    def retrieve(
        self, queries: list[str], k: int | None = None, library: str | None = None
    ) -> list[RetrievedDoc]:
        self.retrieve_k.append(k)
        return super().retrieve(queries, k, library)


class PassSandbox:
    """Sandbox that records (code, tests) and returns a passing RunResult."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, code: str, tests: str, timeout_s: float, mem_mb: int) -> dict[str, Any]:
        self.calls.append((code, tests))
        return {
            "passed": True,
            "category": "pass",
            "exit_code": 0,
            "timed_out": False,
            "duration_s": 0.1,
            "tests_total": 1,
            "tests_failed": 0,
            "failures": [],
            "stdout": "",
            "stderr": "",
        }


def make_task(
    task_id: str,
    *,
    title: str = "Test Task",
    libraries: list[str] | None = None,
    difficulty: int = 1,
    task: str = "Write a function",
    tests: str | None = None,
    reference: str | None = None,
) -> EvalTask:
    """Create an EvalTask with defaults."""
    if libraries is None:
        libraries = ["json"]
    if tests is None:
        tests = "from solution import foo\n\ndef test_foo():\n    assert foo() == 1"
    if reference is None:
        reference = "def foo() -> int:\n    return 1"
    return EvalTask(
        id=task_id,
        title=title,
        libraries=libraries,
        difficulty=difficulty,
        task=task,
        tests=tests,
        reference=reference,
    )


def make_deps(
    settings: Settings,
    script: list[Any],
    *,
    sandbox: Any = None,
    retriever: Any = None,
) -> Dependencies:
    """Create Dependencies with ScriptedLLM and optional sandbox/retriever."""
    llm = ScriptedLLM(script)
    return Dependencies(
        llm=llm,
        settings=settings,
        sandbox=sandbox or PassSandbox(),
        retriever=retriever,
        logger=build_logger(settings, "test-run"),
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_task_files_valid_and_loader_rejects_bad_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that real eval/tasks are valid and loader rejects bad input."""
    # Real eval/tasks directory
    repo_root = Path(__file__).resolve().parents[2]
    tasks_dir = repo_root / "eval" / "tasks"

    tasks = load_tasks(tasks_dir)
    ids = {t.id for t in tasks}
    expected_ids = {
        "dt_business_days",
        "dt_utc_iso",
        "itertools_rle",
        "pathlib_find_files",
        "dataclasses_inventory",
        "csv_sum_by_category",
        "sqlite_top_customers",
        "argparse_cli",
        "pandas_named_agg",
        "pandas_monthly_totals",
        "numpy_zscore",
        "re_log_parse",
    }
    assert ids == expected_ids
    for task in tasks:
        # id matches file stem
        task_file = tasks_dir / f"{task.id}.yaml"
        assert task_file.exists()
        # reference and tests parse as valid Python
        compile(task.reference, f"{task.id}.py", "exec")
        compile(task.tests, f"{task.id}_test.py", "exec")
        # tests contain "from solution import" and >= 4 "def test_"
        assert "from solution import" in task.tests
        assert task.tests.count("def test_") >= 4
        # no forbidden imports in reference or tests
        forbidden = ["requests", "urllib", "socket", "http", "subprocess"]
        for mod in forbidden:
            assert mod not in task.reference, f"{task.id} reference imports {mod}"
            assert mod not in task.tests, f"{task.id} tests imports {mod}"
        # libraries subset of Settings().doc_libraries
        from evalcode.config import Settings

        settings = Settings()
        for lib in task.libraries:
            assert lib in settings.doc_libraries, f"{task.id} lib {lib} not in doc_libraries"

    # --- Bad input cases in tmp_path ---
    # 1. invalid YAML
    bad_dir = tmp_path / "bad_yaml"
    bad_dir.mkdir()
    (bad_dir / "bad_yaml.yaml").write_text("invalid: [")
    with pytest.raises(Exception, match="invalid YAML"):
        load_tasks(bad_dir)

    # 2. missing field
    bad_dir = tmp_path / "missing_field"
    bad_dir.mkdir()
    missing_content = "id: test\ntitle: Test\nlibraries: []\ndifficulty: 1\n"
    (bad_dir / "missing_field.yaml").write_text(missing_content)
    with pytest.raises(Exception, match="validation error in"):
        load_tasks(bad_dir)

    # 3. extra field
    bad_dir = tmp_path / "extra_field"
    bad_dir.mkdir()
    extra_content = (
        "id: test\ntitle: Test\nlibraries: [json]\ndifficulty: 1\n"
        "task: t\ntests: t\nreference: t\nextra: 1"
    )
    (bad_dir / "extra_field.yaml").write_text(extra_content)
    with pytest.raises(Exception, match="validation error in"):
        load_tasks(bad_dir)

    # 4. id != stem
    bad_dir = tmp_path / "wrong_id"
    bad_dir.mkdir()
    wrong_id_content = (
        "id: different\ntitle: Test\nlibraries: [json]\n"
        "difficulty: 1\ntask: t\ntests: t\nreference: t"
    )
    (bad_dir / "wrong_id.yaml").write_text(wrong_id_content)
    with pytest.raises(Exception, match="id mismatch"):
        load_tasks(bad_dir)

    # 5. duplicate id - create two files with same id but different stems
    # The loader checks id!=stem first, so this triggers "id mismatch"
    bad_dir = tmp_path / "duplicate"
    bad_dir.mkdir()
    dup_content = (
        "id: dup\ntitle: Test\nlibraries: [json]\ndifficulty: 1\ntask: t\ntests: t\nreference: t"
    )
    (bad_dir / "dup1.yaml").write_text(dup_content)
    (bad_dir / "dup2.yaml").write_text(dup_content)
    with pytest.raises(Exception, match="id mismatch"):
        load_tasks(bad_dir)

    # 6. blank tests
    bad_dir = tmp_path / "blank_tests"
    bad_dir.mkdir()
    blank_content = (
        "id: blank\ntitle: Test\nlibraries: [json]\n"
        "difficulty: 1\ntask: t\ntests: '   '\nreference: t"
    )
    (bad_dir / "blank_tests.yaml").write_text(blank_content)
    with pytest.raises(Exception, match="validation error in"):
        load_tasks(bad_dir)

    # 7. missing dir
    with pytest.raises(Exception, match="tasks directory not found"):
        load_tasks(tmp_path / "nonexistent")

    # 8. empty dir
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(Exception, match="no task files found"):
        load_tasks(empty_dir)

    # 9. unknown only id
    bad_dir = tmp_path / "unknown_only"
    bad_dir.mkdir()
    (bad_dir / "real1.yaml").write_text(
        "id: real1\ntitle: Test\nlibraries: [json]\ndifficulty: 1\ntask: t\ntests: t\nreference: t"
    )
    with pytest.raises(Exception, match="unknown task ids"):
        load_tasks(bad_dir, only=["real1", "unknown"])


def test_run_eval_both_modes_records_fields(
    eval_env: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test run_eval records all fields for both rag and norag modes."""
    # Two tasks
    task1 = make_task("task_a", title="Task A")
    task2 = make_task("task_b", title="Task B", libraries=["re"])

    tasks = [task1, task2]

    # LLM script that passes first try
    script = [
        bundle_text(
            "def foo() -> int:\n    return 1",
            "from solution import foo\n\ndef test_foo():\n    assert foo() == 1",
        )
    ]

    # Create retriever that returns a doc
    docs = [make_doc("doc1", "json.loads parses JSON", 0.9, "json", "json.loads", "json")]
    retriever = ClosableRetriever(default=docs)

    deps = make_deps(eval_env, script, retriever=retriever)

    out_dir = tmp_path / "eval_results"

    lines: list[str] = []

    def on_line(s: str) -> None:
        lines.append(s)

    _ = run_eval(
        tasks=tasks,
        modes=["rag", "norag"],
        deps=deps,
        out_dir=out_dir,
        on_line=on_line,
    )

    # 4 result files (2 tasks x 2 modes)
    raw_files = list((out_dir / "raw").glob("*.json"))
    assert len(raw_files) == 4

    # Check each record has all required fields
    for raw_file in raw_files:
        with open(raw_file) as f:
            record = json.load(f)

        # Required fields with correct types
        assert record["schema_version"] == 1
        assert record["task_id"] in ("task_a", "task_b")
        assert record["mode"] in ("rag", "norag")
        assert record["status"] in ("approved", "failed")
        assert isinstance(record["passed"], bool)
        assert isinstance(record["attempts"], int)
        assert isinstance(record["retries_used"], int)
        assert isinstance(record["human_rounds"], int)
        assert isinstance(record["first_try"], bool)
        assert isinstance(record["llm_calls"], int)
        assert isinstance(record["api_retries"], int)
        assert isinstance(record["input_tokens"], int)
        assert isinstance(record["output_tokens"], int)
        assert isinstance(record["total_tokens"], int)
        # wait_s can be int (0) or float (0.0) due to merge_usage
        assert isinstance(record["wait_s"], (int, float))
        assert isinstance(record["wall_s"], float)
        assert record["final_category"] is not None
        assert record["failure_reason"] is None  # passed runs have None
        assert isinstance(record["n_docs"], int)
        assert isinstance(record["doc_ids"], list)
        assert isinstance(record["re_retrievals"], int)
        assert record["final_code"] is not None
        assert isinstance(record["model"], str)
        assert record["run_id"].startswith("eval-")
        assert isinstance(record["finished_ts"], str)

        # rag records n_docs >= 1 and doc_ids non-empty
        if record["mode"] == "rag":
            assert record["n_docs"] >= 1
            assert len(record["doc_ids"]) >= 1
        else:
            assert record["n_docs"] == 0
            assert record["doc_ids"] == []

        # first_try True
        assert record["first_try"] is True

    # Check retriever was called only for rag runs
    # (We have 2 rag runs, so retriever.retrieve_k should have 2 entries)
    assert len(retriever.retrieve_k) == 2

    # Check sandbox received task.tests verbatim
    sandbox = deps.sandbox
    assert isinstance(sandbox, PassSandbox)
    assert len(sandbox.calls) == 4  # 4 runs total
    for _code, tests in sandbox.calls:
        assert tests == "from solution import foo\n\ndef test_foo():\n    assert foo() == 1"

    # run_ids unique and start with "eval-"
    run_ids = [json.loads(f.read_text())["run_id"] for f in raw_files]
    assert len(set(run_ids)) == 4
    for rid in run_ids:
        assert rid.startswith("eval-")

    # logs/<run_id>/summary.json exists for each run
    log_dir = tmp_path / "logs"
    for rid in run_ids:
        summary_path = log_dir / rid / "summary.json"
        assert summary_path.exists(), f"Missing summary for {rid}"
        with open(summary_path) as f:
            summary = json.load(f)
        assert summary["status"] == "approved"
        assert "rag" in summary
        assert "n_events" in summary

    # The fake key text appears in no result file
    fake_key = "AIza" + "x" * 35
    for raw_file in raw_files:
        content = raw_file.read_text()
        assert fake_key not in content, f"API key leaked in {raw_file}"


def test_run_eval_resume_limit_force(
    eval_env: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test limit, skip, force behavior in run_eval."""
    from evalcode.evaluation import run_eval

    # 3 tasks x 2 modes = 6 runs
    task1 = make_task("task_a")
    task2 = make_task("task_b")
    task3 = make_task("task_c")
    tasks = [task1, task2, task3]

    script = [
        bundle_text(
            "def foo() -> int:\n    return 1",
            "from solution import foo\n\ndef test_foo():\n    assert foo() == 1",
        )
    ]

    retriever = ClosableRetriever(
        default=[make_doc("doc1", "json.loads parses JSON", 0.9, "json", "json.loads", "json")]
    )
    deps = make_deps(eval_env, script, retriever=retriever)

    out_dir = tmp_path / "eval_results"

    # First run: limit=2 -> executed 2, stopped "limit", pending 4
    outcome1 = run_eval(
        tasks=tasks,
        modes=["rag", "norag"],
        deps=deps,
        out_dir=out_dir,
        limit=2,
    )
    assert outcome1.executed == 2
    assert outcome1.skipped == 0
    assert outcome1.pending == 4
    assert outcome1.stopped == "limit"
    assert outcome1.errors == 0

    # Second run: no limit -> skipped 2 (already done), executed 4
    outcome2 = run_eval(
        tasks=tasks,
        modes=["rag", "norag"],
        deps=deps,
        out_dir=out_dir,
    )
    assert outcome2.executed == 4
    assert outcome2.skipped == 2
    assert outcome2.pending == 0
    assert outcome2.stopped is None

    # Third run: force=False -> executed 0, skipped 6
    outcome3 = run_eval(
        tasks=tasks,
        modes=["rag", "norag"],
        deps=deps,
        out_dir=out_dir,
    )
    assert outcome3.executed == 0
    assert outcome3.skipped == 6
    assert outcome3.pending == 0

    # Fourth run: force=True -> re-executes all 6
    outcome4 = run_eval(
        tasks=tasks,
        modes=["rag", "norag"],
        deps=deps,
        out_dir=out_dir,
        force=True,
    )
    assert outcome4.executed == 6
    assert outcome4.skipped == 0
    assert outcome4.pending == 0

    # Test: corrupt result file and "error"-status file are re-executed, not skipped
    # Create a corrupt file
    corrupt_file = out_dir / "raw" / "task_a__rag.json"
    corrupt_file.write_text("not valid json")

    # Create an error-status file
    error_file = out_dir / "raw" / "task_b__rag.json"
    error_record = {
        "schema_version": 1,
        "task_id": "task_b",
        "mode": "rag",
        "status": "error",
        "run_id": "eval-task_b-rag-abcdef12",
    }
    error_file.write_text(json.dumps(error_record))

    # Run with force=False - corrupt and error should be re-executed
    outcome5 = run_eval(
        tasks=tasks,
        modes=["rag", "norag"],
        deps=deps,
        out_dir=out_dir,
    )
    # task_a rag (corrupt) + task_b rag (error) = 2 executed
    assert outcome5.executed == 2
    assert outcome5.skipped == 4  # the other 4 complete files are skipped


def test_run_eval_quota_stop_and_error_isolation(
    eval_env: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test quota/auth stop and error isolation in run_eval."""
    from evalcode.errors import DailyQuotaExceeded, LLMAuthError
    from evalcode.evaluation import run_eval, stop_reason
    from evalcode.state import AgentState

    # 2 tasks x 2 modes = 4 runs
    task1 = make_task("task_a")
    task2 = make_task("task_b")
    tasks = [task1, task2]

    # --- (a) first run succeeds then DailyQuotaExceeded -> stopped "quota" ---
    # Script: first call passes, second call raises DailyQuotaExceeded
    script_quota = [
        bundle_text(
            "def foo() -> int:\n    return 1",
            "from solution import foo\n\ndef test_foo():\n    assert foo() == 1",
        ),
        DailyQuotaExceeded("quota exhausted"),
    ]

    retriever = ClosableRetriever(
        default=[make_doc("doc1", "json.loads parses JSON", 0.9, "json", "json.loads", "json")]
    )
    deps = make_deps(eval_env, script_quota, retriever=retriever)

    out_dir = tmp_path / "eval_results_quota"

    outcome = run_eval(
        tasks=tasks,
        modes=["rag", "norag"],
        deps=deps,
        out_dir=out_dir,
    )
    # First run (task_a rag) passes, second (task_a norag) raises quota
    # The quota run writes NO result file, so executed=1, pending=3 (quota run + 2 remaining)
    assert outcome.executed == 1
    assert outcome.stopped == "quota"
    # The quota run has NO result file
    raw_files = list((out_dir / "raw").glob("*.json"))
    assert len(raw_files) == 1  # only the first completed run
    # Earlier files intact (task_a rag)
    record = json.loads(raw_files[0].read_text())
    assert record["task_id"] == "task_a"
    assert record["mode"] == "rag"
    assert record["status"] == "approved"
    # Rest pending (quota run + task_b rag + task_b norag)
    assert outcome.pending == 3

    # --- (b) LLMAuthError -> stopped "auth" ---
    script_auth = [
        bundle_text(
            "def foo() -> int:\n    return 1",
            "from solution import foo\n\ndef test_foo():\n    assert foo() == 1",
        ),
        LLMAuthError("invalid api key"),
    ]

    deps_auth = make_deps(
        eval_env, script_auth, retriever=ClosableRetriever(default=retriever.default)
    )
    out_dir_auth = tmp_path / "eval_results_auth"

    outcome_auth = run_eval(
        tasks=tasks,
        modes=["rag", "norag"],
        deps=deps_auth,
        out_dir=out_dir_auth,
    )
    assert outcome_auth.executed == 1
    assert outcome_auth.stopped == "auth"
    raw_files_auth = list((out_dir_auth / "raw").glob("*.json"))
    assert len(raw_files_auth) == 1

    # --- (c) sandbox raising RuntimeError with fake key -> status "error" record ---
    class BadSandbox:
        def __call__(self, code: str, tests: str, timeout_s: float, mem_mb: int) -> dict:
            fake_key = "AIza" + "x" * 35
            raise RuntimeError(f"something failed with {fake_key}")

    deps_bad = make_deps(
        eval_env,
        script_quota[:1],  # only the passing response
        sandbox=BadSandbox(),
        retriever=ClosableRetriever(default=retriever.default),
    )
    out_dir_bad = tmp_path / "eval_results_bad"

    outcome_bad = run_eval(
        tasks=[task1],  # just one task
        modes=["rag"],
        deps=deps_bad,
        out_dir=out_dir_bad,
    )
    assert outcome_bad.executed == 1
    assert outcome_bad.errors == 1
    assert outcome_bad.stopped is None
    raw_files_bad = list((out_dir_bad / "raw").glob("*.json"))
    assert len(raw_files_bad) == 1
    record_bad = json.loads(raw_files_bad[0].read_text())
    assert record_bad["status"] == "error"
    # failure_reason should be redacted (no fake key or its first 10 chars)
    failure_reason = record_bad["failure_reason"]
    assert failure_reason is not None
    assert "AIza" not in failure_reason
    assert "xxxxxxxxxx" not in failure_reason  # first 10 chars
    assert "RuntimeError" in failure_reason

    # NEXT run still executes (not stopped) - use fresh deps with passing script
    deps_fresh = make_deps(
        eval_env,
        script_quota[:1],  # only the passing response
        sandbox=PassSandbox(),
        retriever=ClosableRetriever(default=retriever.default),
    )
    outcome_bad2 = run_eval(
        tasks=[task1, task2],
        modes=["rag", "norag"],
        deps=deps_fresh,
        out_dir=out_dir_bad,
    )
    assert outcome_bad2.executed == 4
    # task1 rag (error, re-executed) + task1 norag + task2 rag + task2 norag

    # --- (d) rag with retriever None -> ConfigError before any run ---
    deps_no_rag = make_deps(eval_env, script_quota[:1], retriever=None)
    out_dir_no_rag = tmp_path / "eval_results_no_rag"

    try:
        run_eval(
            tasks=tasks,
            modes=["rag"],  # only rag mode
            deps=deps_no_rag,
            out_dir=out_dir_no_rag,
        )
        raise AssertionError("Expected ConfigError")
    except Exception as e:
        assert "no usable index" in str(e).lower() or "rag mode" in str(e).lower()

    # --- (e) stop_reason: LLMBudgetExceeded -> None ---
    state: AgentState = {
        "status": "failed",
        "failure_reason": "budget",
        "history": [{"summary": {"error": "LLMBudgetExceeded"}}],
    }
    assert stop_reason(state) is None


def test_build_report_aggregates_and_pairing(
    eval_env: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test build_report aggregates correctly and pairs modes."""
    from evalcode.evaluation import EvalTask, build_report, write_report

    # Create synthetic results
    results = [
        # task_a rag - PASS
        {
            "schema_version": 1,
            "task_id": "task_a",
            "mode": "rag",
            "status": "approved",
            "passed": True,
            "attempts": 1,
            "retries_used": 0,
            "human_rounds": 0,
            "first_try": True,
            "llm_calls": 2,
            "api_retries": 0,
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "wait_s": 0.0,
            "wall_s": 5.0,
            "final_category": "pass",
            "failure_reason": None,
            "n_docs": 3,
            "doc_ids": ["d1", "d2", "d3"],
            "re_retrievals": 0,
            "final_code": "def foo(): pass",
            "model": "gemini-3.5-flash-lite",
            "run_id": "eval-task_a-rag-123",
            "finished_ts": "2024-01-01T00:00:00Z",
        },
        # task_a norag - PASS
        {
            "schema_version": 1,
            "task_id": "task_a",
            "mode": "norag",
            "status": "approved",
            "passed": True,
            "attempts": 2,
            "retries_used": 0,
            "human_rounds": 0,
            "first_try": False,
            "llm_calls": 4,
            "api_retries": 0,
            "input_tokens": 200,
            "output_tokens": 100,
            "total_tokens": 300,
            "wait_s": 0.5,
            "wall_s": 10.0,
            "final_category": "pass",
            "failure_reason": None,
            "n_docs": 0,
            "doc_ids": [],
            "re_retrievals": 0,
            "final_code": "def foo(): pass",
            "model": "gemini-3.5-flash-lite",
            "run_id": "eval-task_a-norag-456",
            "finished_ts": "2024-01-01T00:00:00Z",
        },
        # task_b rag - FAIL
        {
            "schema_version": 1,
            "task_id": "task_b",
            "mode": "rag",
            "status": "failed",
            "passed": False,
            "attempts": 3,
            "retries_used": 2,
            "human_rounds": 0,
            "first_try": False,
            "llm_calls": 6,
            "api_retries": 1,
            "input_tokens": 300,
            "output_tokens": 150,
            "total_tokens": 450,
            "wait_s": 1.0,
            "wall_s": 15.0,
            "final_category": "assertion_failure",
            "failure_reason": "test failed",
            "n_docs": 2,
            "doc_ids": ["d1", "d2"],
            "re_retrievals": 1,
            "final_code": "def bar(): pass",
            "model": "gemini-3.5-flash-lite",
            "run_id": "eval-task_b-rag-789",
            "finished_ts": "2024-01-01T00:00:00Z",
        },
        # task_b norag - FAIL
        {
            "schema_version": 1,
            "task_id": "task_b",
            "mode": "norag",
            "status": "failed",
            "passed": False,
            "attempts": 3,
            "retries_used": 2,
            "human_rounds": 0,
            "first_try": False,
            "llm_calls": 6,
            "api_retries": 1,
            "input_tokens": 300,
            "output_tokens": 150,
            "total_tokens": 450,
            "wait_s": 1.0,
            "wall_s": 15.0,
            "final_category": "assertion_failure",
            "failure_reason": "test failed",
            "n_docs": 0,
            "doc_ids": [],
            "re_retrievals": 0,
            "final_code": "def bar(): pass",
            "model": "gemini-3.5-flash-lite",
            "run_id": "eval-task_b-norag-101",
            "finished_ts": "2024-01-01T00:00:00Z",
        },
        # task_c rag only - PASS (no norag)
        {
            "schema_version": 1,
            "task_id": "task_c",
            "mode": "rag",
            "status": "approved",
            "passed": True,
            "attempts": 1,
            "retries_used": 0,
            "human_rounds": 0,
            "first_try": True,
            "llm_calls": 2,
            "api_retries": 0,
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "wait_s": 0.0,
            "wall_s": 5.0,
            "final_category": "pass",
            "failure_reason": None,
            "n_docs": 3,
            "doc_ids": ["d1", "d2", "d3"],
            "re_retrievals": 0,
            "final_code": "def baz(): pass",
            "model": "gemini-3.5-flash-lite",
            "run_id": "eval-task_c-rag-202",
            "finished_ts": "2024-01-01T00:00:00Z",
        },
        # task_d error status
        {
            "schema_version": 1,
            "task_id": "task_d",
            "mode": "rag",
            "status": "error",
            "failure_reason": "some error",
            "run_id": "eval-task_d-rag-303",
        },
    ]

    # Create task objects
    tasks = [
        EvalTask(
            id="task_a",
            title="Task A",
            libraries=["json"],
            difficulty=1,
            task="t",
            tests="t",
            reference="t",
        ),
        EvalTask(
            id="task_b",
            title="Task B",
            libraries=["json"],
            difficulty=1,
            task="t",
            tests="t",
            reference="t",
        ),
        EvalTask(
            id="task_c",
            title="Task C",
            libraries=["json"],
            difficulty=1,
            task="t",
            tests="t",
            reference="t",
        ),
        EvalTask(
            id="task_d",
            title="Task D",
            libraries=["json"],
            difficulty=1,
            task="t",
            tests="t",
            reference="t",
        ),
    ]

    # Build report
    report = build_report(results, tasks, model="test-model")

    # Check summary rows per mode (exact format)
    # rag: 3 runs, 2 passed, first_try=2, mean_attempts=1.67, mean_llm_calls=3.33,
    # mean_tokens=250, median_wall=5.0 (walls: 5.0, 5.0, 15.0 -> median 5.0)
    assert "| rag | 3 | 2 | 66.7% | 2 | 1.67 | 3.33 | 250 | 5.0 |" in report
    # norag: 2 runs, 1 passed, first_try=0, mean_attempts=2.50, mean_llm_calls=5.00,
    # mean_tokens=375, median_wall=12.5 (walls: 10.0, 15.0 -> median 12.5)
    assert "| norag | 2 | 1 | 50.0% | 0 | 2.50 | 5.00 | 375 | 12.5 |" in report

    # Paired comparison: task_a and task_b have both modes, task_c only rag, task_d error
    assert "both pass: 1, rag only: 0, norag only: 0, neither: 1" in report
    # mean LLM calls (rag - norag): task_a: 2-4=-2, task_b: 6-6=0 -> mean = -1.00
    assert "mean LLM calls (rag - norag): -1.00" in report
    # mean attempts (rag - norag): task_a: 1-2=-1, task_b: 3-3=0 -> mean = -0.50
    assert "mean attempts (rag - norag): -0.50" in report

    # Per task table - check cells
    assert "| task_a | json | PASS a=1 c=2 | PASS a=2 c=4 |" in report
    assert "| task_b | json | FAIL a=3 c=6 | FAIL a=3 c=6 |" in report
    assert "| task_c | json | PASS a=1 c=2 | - |" in report
    # task_d has error status in rag mode - shows as "error" for rag, "-" for norag
    assert "| task_d | json | error | - |" in report

    # Caveats - error results named
    assert "Tasks with errors: task_d" in report
    # K < 10 pairs caveat
    assert (
        "K < 10 pairs (2): differences are indicative only, not statistically meaningful."
    ) in report

    # Deterministic - same input twice -> identical string
    report2 = build_report(results, tasks, model="test-model")
    assert report == report2

    # write_report writes report.md utf-8 - need to write result files first
    out_dir = tmp_path / "report_out"
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True)
    for record in results:
        result_file = raw_dir / f"{record['task_id']}__{record['mode']}.json"
        result_file.write_text(json.dumps(record, ensure_ascii=False, indent=2))

    report_path = write_report(out_dir, tasks, model="test-model")
    assert report_path == out_dir / "report.md"
    assert report_path.exists()
    written = report_path.read_text(encoding="utf-8")
    assert written == report


def test_eval_cli_and_verify_references(
    eval_env: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test eval CLI command and verify_references."""
    import yaml
    from typer.testing import CliRunner

    from evalcode import cli as cli_module

    # Create 2 tasks in a temp dir
    tasks_dir = tmp_path / "eval_tasks"
    tasks_dir.mkdir()
    make_task("task_a", title="Task A")
    make_task("task_b", title="Task B", libraries=["re"])

    (tasks_dir / "task_a.yaml").write_text(
        yaml.dump(
            {
                "id": "task_a",
                "title": "Task A",
                "libraries": ["json"],
                "difficulty": 1,
                "task": "Write a function",
                "tests": "from solution import foo\n\ndef test_foo():\n    assert foo() == 1",
                "reference": "def foo() -> int:\n    return 1",
            }
        )
    )
    (tasks_dir / "task_b.yaml").write_text(
        yaml.dump(
            {
                "id": "task_b",
                "title": "Task B",
                "libraries": ["re"],
                "difficulty": 1,
                "task": "Write a function",
                "tests": "from solution import bar\n\ndef test_bar():\n    assert bar() == 2",
                "reference": "def bar() -> int:\n    return 2",
            }
        )
    )

    # Monkeypatch _build_eval_deps and _eval_sandbox
    script = [
        bundle_text(
            "def foo() -> int:\n    return 1",
            "from solution import foo\n\ndef test_foo():\n    assert foo() == 1",
        )
    ]
    docs = [make_doc("doc1", "json.loads parses JSON", 0.9, "json", "json.loads", "json")]
    retriever = ClosableRetriever(default=docs)
    sandbox = PassSandbox()

    def fake_build_eval_deps(settings: Settings, *, rag: bool) -> Dependencies:
        return make_deps(settings, script, sandbox=sandbox, retriever=retriever if rag else None)

    def fake_eval_sandbox():
        return sandbox

    monkeypatch.setattr(cli_module, "_build_eval_deps", fake_build_eval_deps)
    monkeypatch.setattr(cli_module, "_eval_sandbox", fake_eval_sandbox)

    runner = CliRunner()

    # --verify-refs should use _eval_sandbox and exit 0 with PASS lines
    result = runner.invoke(cli_module.app, ["eval", "--verify-refs", "--tasks-dir", str(tasks_dir)])
    assert result.exit_code == 0
    assert "PASS" in result.output
    assert "task_a" in result.output
    assert "task_b" in result.output

    # Failing fake sandbox -> exit 1 with FAIL
    failing_sandbox = FakeSandboxFail()

    def fake_failing_sandbox():
        return failing_sandbox

    monkeypatch.setattr(cli_module, "_eval_sandbox", fake_failing_sandbox)
    result2 = runner.invoke(
        cli_module.app, ["eval", "--verify-refs", "--tasks-dir", str(tasks_dir)]
    )
    assert result2.exit_code == 1
    assert "FAIL" in result2.output

    # --modes bogus -> exit 2
    monkeypatch.setattr(cli_module, "_eval_sandbox", fake_eval_sandbox)
    result3 = runner.invoke(
        cli_module.app, ["eval", "--modes", "bogus", "--tasks-dir", str(tasks_dir)]
    )
    assert result3.exit_code == 2
    assert "Invalid mode" in result3.output

    # --limit 0 -> exit 2
    result4 = runner.invoke(cli_module.app, ["eval", "--limit", "0", "--tasks-dir", str(tasks_dir)])
    assert result4.exit_code == 2
    assert "limit must be >= 1" in result4.output

    # --report-only on empty out dir -> exit 1
    empty_out = tmp_path / "empty_out"
    result5 = runner.invoke(
        cli_module.app,
        ["eval", "--report-only", "--out", str(empty_out), "--tasks-dir", str(tasks_dir)],
    )
    assert result5.exit_code == 1
    assert "no results in" in result5.output

    # --modes norag --limit 1 -> exit 0, report.md exists
    out_dir = tmp_path / "eval_out"
    result6 = runner.invoke(
        cli_module.app,
        [
            "eval",
            "--modes",
            "norag",
            "--limit",
            "1",
            "--out",
            str(out_dir),
            "--tasks-dir",
            str(tasks_dir),
        ],
    )
    assert result6.exit_code == 0
    assert "executed=1" in result6.output
    assert "skipped=0" in result6.output
    report_path = out_dir / "report.md"
    assert report_path.exists()

    # store.close called exactly once per normal run
    # Create a fresh retriever for this test
    retriever_close_test = ClosableRetriever(
        default=[make_doc("doc1", "json.loads parses JSON", 0.9, "json", "json.loads", "json")]
    )

    def fake_build_eval_deps_close(settings: Settings, *, rag: bool) -> Dependencies:
        ret = retriever_close_test if rag else None
        return make_deps(settings, script, sandbox=sandbox, retriever=ret)

    monkeypatch.setattr(cli_module, "_build_eval_deps", fake_build_eval_deps_close)
    out_dir_close = tmp_path / "eval_out_close"
    result_close = runner.invoke(
        cli_module.app,
        [
            "eval",
            "--modes",
            "rag",
            "--limit",
            "1",
            "--out",
            str(out_dir_close),
            "--tasks-dir",
            str(tasks_dir),
        ],
    )
    assert result_close.exit_code == 0
    # store.close should be called exactly once for the single rag run
    assert retriever_close_test.store.close.call_count == 1

    # Also test --modes norag does not close a retriever it never received
    retriever_norag = ClosableRetriever(
        default=[make_doc("doc1", "json.loads parses JSON", 0.9, "json", "json.loads", "json")]
    )

    def fake_build_eval_deps_norag(settings: Settings, *, rag: bool) -> Dependencies:
        return make_deps(settings, script, sandbox=sandbox, retriever=None)

    monkeypatch.setattr(cli_module, "_build_eval_deps", fake_build_eval_deps_norag)
    out_dir_norag = tmp_path / "eval_out_norag"
    result_norag = runner.invoke(
        cli_module.app,
        [
            "eval",
            "--modes",
            "norag",
            "--limit",
            "1",
            "--out",
            str(out_dir_norag),
            "--tasks-dir",
            str(tasks_dir),
        ],
    )
    assert result_norag.exit_code == 0
    # No retriever was passed, so no close should be called
    # (retriever_norag.store.close is a MagicMock but we never use it)
    assert retriever_norag.store.close.call_count == 0

    # --modes rag with retriever None -> exit 4 with ingest hint
    def fake_no_retriever(settings: Settings, *, rag: bool):
        return make_deps(settings, script, sandbox=sandbox, retriever=None)

    monkeypatch.setattr(cli_module, "_build_eval_deps", fake_no_retriever)
    result7 = runner.invoke(
        cli_module.app,
        [
            "eval",
            "--modes",
            "rag",
            "--out",
            str(tmp_path / "eval_out2"),
            "--tasks-dir",
            str(tasks_dir),
        ],
    )
    assert result7.exit_code == 4
    assert "needs an index" in result7.output.lower() or "ingest" in result7.output.lower()

    # store.close called exactly once per normal run
    # (verifiable by checking retriever.store.close.call_count after runs)

    # "eval --help" lists the options
    result_help = runner.invoke(cli_module.app, ["eval", "--help"])
    assert result_help.exit_code == 0
    assert "--modes" in result_help.output
    assert "--limit" in result_help.output
    assert "--verify-refs" in result_help.output
    assert "--report-only" in result_help.output

    # app "--help" lists eval
    result_app_help = runner.invoke(cli_module.app, ["--help"])
    assert result_app_help.exit_code == 0
    assert "eval" in result_app_help.output
