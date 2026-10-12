# evalcode

evalcode is an iterative code-generation agent: RAG over Python docs + LangGraph loop (retrieve → generate → sandboxed tests → analyze → revise), with human approval, observability, Typer CLI, and an evaluation harness.

[![CI](https://github.com/KuNaL8103/evalcode/actions/workflows/ci.yml/badge.svg)](https://github.com/KuNaL8103/evalcode/actions/workflows/ci.yml)

## How it works

The graph executes these nodes in a loop until the solution passes, hits the retry budget, or pauses for human review:

```mermaid
flowchart TD
    START_RAG[START (RAG on)] --> RETRIEVE
    START_NORAG[START (RAG off)] --> GENERATE
    RETRIEVE -->|first pass| GENERATE
    RETRIEVE -->|re-retrieval| REVISE
    GENERATE -->|llm ok| RUN_TESTS
    GENERATE -->|llm error| FAIL
    REVISE -->|llm ok| RUN_TESTS
    REVISE -->|llm error| FAIL
    RUN_TESTS -->|pass, auto_approve| FINALIZE
    RUN_TESTS -->|pass, !auto_approve| HUMAN_REVIEW
    RUN_TESTS -->|fail, retries left| ANALYZE_ERROR
    RUN_TESTS -->|fail, no retries| FAIL
    ANALYZE_ERROR -->|needs_docs, fresh queries| RETRIEVE
    ANALYZE_ERROR -->|no docs needed| REVISE
    HUMAN_REVIEW -->|approve| FINALIZE
    HUMAN_REVIEW -->|edit| RUN_TESTS
    HUMAN_REVIEW -->|reject, rounds left| REVISE
    HUMAN_REVIEW -->|reject, no rounds| FAIL
    FINALIZE --> END
    FAIL --> END
```

Key properties:
- Tagged-text output protocol: LLM emits `<code>...</code>` and `<tests>...</tests>` blocks; parser is fault-tolerant.
- Bounded retries: MAX_RETRIES (default 3) bounds generate→test→revise cycles; exponential backoff and the per-run call budget (MAX_LLM_CALLS_PER_RUN) are separate mechanisms.
- Error-driven re-retrieval: on failure, analyze_error may trigger a second retrieve with error context.
- Human review interrupt: approve / reject (with feedback) / edit (provide edited file); state checkpointed in SQLite.
- SQLite checkpointer enables resume across processes; logs append under `logs/<task_id>/`.
- RAG-off mode: `--no-rag` or `eval --modes norag` skips retrieval entirely.

## Requirements and install

- Python 3.11+
- Create and activate a virtual environment:
  - Linux/macOS:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```
  - Windows PowerShell:
    ```powershell
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    ```
- Install dependencies:
  ```bash
  pip install -e ".[dev,corpus]"
  ```
  Tip: install CPU-only torch first to avoid multi-GB CUDA wheels on Linux:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```
- Copy `.env.example` to `.env` and set `GEMINI_API_KEY`:
  - Linux/macOS: `cp .env.example .env`
  - Windows PowerShell: `copy .env.example .env`
- First run downloads the MiniLM embedding model (~90 MB).

## Gemini setup and free-tier guidance

1. Get an API key from https://aistudio.google.com/apikey
2. Copy `.env.example` to `.env` and set `GEMINI_API_KEY`:
   ```bash
   copy .env.example .env
   # edit .env and add your key
   ```
   `.env` is git-ignored; never commit it.
3. Default model: `gemini-3.5-flash-lite` via the OpenAI-compatible endpoint (`https://generativelanguage.googleapis.com/v1beta/openai/`).
   Note: "inkling" models return 403 for plain API-key clients; do not use them as `LLM_MODEL`.

Free-tier design:
- Throttle: `LLM_MIN_INTERVAL_S` (default 3.0 s) between calls.
- Exponential backoff with `Retry-After` header support.
- `DailyQuotaExceeded` fails fast (CLI exit code 3).
- `MAX_LLM_CALLS_PER_RUN` (default 10) bounds per-run cost.
- Optional LLM calls (`ANALYZE_WITH_LLM`, `QUERY_REWRITE_WITH_LLM`) are off by default.
- Evaluation is resumable with `--limit`.

Default values from `.env.example`:
- `LLM_MODEL=gemini-3.5-flash-lite`
- `LLM_MIN_INTERVAL_S=3.0`
- `MAX_LLM_CALLS_PER_RUN=10`
- `MAX_RETRIES=3`
- `ANALYZE_WITH_LLM=false`
- `QUERY_REWRITE_WITH_LLM=false`

## Quickstart

```bash
# Ingest Python stdlib docs (or a subset)
evalcode ingest --libs json,re,pathlib

# Search the index
evalcode search "json.loads" -k 3

# Run a task with auto-approve (non-interactive)
evalcode run "write a function that parses JSON and returns the keys" --auto-approve

# Run interactively (pauses for human review)
evalcode run "write a function that parses JSON and returns the keys"

# Resume a paused task (task_id printed as TASK_ID)
evalcode resume <TASK_ID> --approve
evalcode resume <TASK_ID> --reject "needs type hints"
evalcode resume <TASK_ID> --edit-file edited_solution.py
```

Exit codes:
| Code | Meaning |
|------|---------|
| 0 | Approved / success |
| 1 | Failed / other error |
| 2 | Usage error |
| 3 | Free quota exhausted |
| 4 | Configuration error |
| 5 | Paused awaiting review |

## Human review

Three decisions:
- **approve** — accept the current solution and finalize.
- **reject** — provide feedback text; the agent revises and re-runs tests.
- **edit** — supply a path to an edited code file; the agent treats it as the new candidate.

Resume works across processes: the task id is printed as `TASK_ID`, state lives in the SQLite checkpoint, and logs append under `logs/<task_id>/`.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | (required) | Gemini API key from AI Studio |
| `LLM_MODEL` | `gemini-3.5-flash-lite` | Model name |
| `LLM_MIN_INTERVAL_S` | `3.0` | Minimum seconds between LLM calls |
| `MAX_LLM_CALLS_PER_RUN` | `10` | Per-run call budget |
| `MAX_RETRIES` | `3` | Max generate→test→revise cycles |
| `ANALYZE_WITH_LLM` | `false` | Use LLM for error analysis |
| `QUERY_REWRITE_WITH_LLM` | `false` | Use LLM to rewrite retrieval queries |
| `SANDBOX_TIMEOUT_S` | `10` | Subprocess timeout |
| `SANDBOX_MEM_MB` | `1024` | Memory limit passed to the sandbox (applied as an address-space rlimit on POSIX; ignored on Windows) |
| `RETRIEVAL_TOP_K` | `5` | Docs per query |
| `RETRIEVAL_MIN_SCORE` | `0.2` | Minimum similarity score |
| `CHECKPOINT_DB` | `data/checkpoints.sqlite` | SQLite checkpoint path |
| `LOG_DIR` | `logs` | Log directory |
| `CHROMA_DIR` | `data/chroma` | Vector store path |
| `DOC_LIBRARIES` | `json,re,datetime,collections,itertools,pathlib,dataclasses,csv,sqlite3,argparse,pandas,numpy` | Default libraries to ingest |

See `.env.example` for the full list. No secrets in the repo.

## Observability

- `logs/<run_id>/events.jsonl` — one JSON line per node event: node, attempt, duration, usage, doc ids.
- `logs/<run_id>/summary.json` — final status, rag flag, node counts, token totals.
- Code, tests, and prompts are **never** logged; API keys are redacted.
- Optional LangSmith: set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY`.

## Sandbox

Subprocess with timeout, scrubbed environment, rlimits on POSIX only (skipped on Windows).
**Threat model:** protects against accidental damage and runaway code; it is NOT a security boundary (network not blocked, filesystem not jailed).

## Evaluation

```bash
# Verify reference solutions (real sandbox, no LLM)
evalcode eval --verify-refs

# Run evaluation with limit (resumable)
evalcode eval --limit 5

# Report only from existing results
evalcode eval --report-only
```

- Resumability via per-(task,mode) result files in `eval/results/raw/`.
- Quota/auth stop writes no result for the interrupted run.
- Exit code 3 on quota exhaustion.
- RAG mode requires an index; run `evalcode ingest` first, or use `--modes norag` to skip retrieval.

### Results (copied from `eval/results/report.md`)

```
gemini-3.5-flash-lite | tasks 12 | complete runs 24 | error runs 0 | incomplete pairs 0
```

| mode | runs | passed | pass rate | first-try | mean attempts | mean LLM calls | mean tokens | median wall s |
|------|------|--------|-----------|-----------|---------------|----------------|-------------|---------------|
| rag | 12 | 11 | 91.7% | 11 | 1.25 | 1.25 | 2248 | 4.3 |
| norag | 12 | 12 | 100.0% | 12 | 1.00 | 1.00 | 943 | 4.1 |

**Paired comparison (K tasks with both modes complete)**
both pass: 11, rag only: 0, norag only: 1, neither: 0
mean LLM calls (rag - norag): +0.25
mean attempts (rag - norag): +0.25

| id | libs | rag | norag |
|----|------|-----|-------|
| argparse_cli | argparse | PASS a=1 c=1 | PASS a=1 c=1 |
| csv_sum_by_category | csv | PASS a=1 c=1 | PASS a=1 c=1 |
| dataclasses_inventory | dataclasses | FAIL a=4 c=4 | PASS a=1 c=1 |
| dt_business_days | datetime | PASS a=1 c=1 | PASS a=1 c=1 |
| dt_utc_iso | datetime | PASS a=1 c=1 | PASS a=1 c=1 |
| itertools_rle | itertools | PASS a=1 c=1 | PASS a=1 c=1 |
| numpy_zscore | numpy | PASS a=1 c=1 | PASS a=1 c=1 |
| pandas_monthly_totals | pandas | PASS a=1 c=1 | PASS a=1 c=1 |
| pandas_named_agg | pandas | PASS a=1 c=1 | PASS a=1 c=1 |
| pathlib_find_files | pathlib | PASS a=1 c=1 | PASS a=1 c=1 |
| re_log_parse | re | PASS a=1 c=1 | PASS a=1 c=1 |
| sqlite_top_customers | sqlite3 | PASS a=1 c=1 | PASS a=1 c=1 |

**Honest reading:**
- (a) 12 tasks × 2 modes, one sample per cell, model `gemini-3.5-flash-lite`, run on 2026-10-11 (finished_ts of raw results).
- (b) Both modes are near the ceiling; RAG did not improve pass rate on this benchmark (11/12 vs 12/12) and used more tokens (2248 vs 943 mean).
- (c) The single RAG failure is `dataclasses_inventory`: four attempts, each classified `syntax_error`; the cause was NOT diagnosed (failed runs do not store their code) and the task text contains a literal placeholder `<default empty list>` which is a plausible confound; with one sample it cannot be attributed to RAG.
- (d) Limitations: acceptance tests are visible to the model, tasks are easy, n = 12, temperature > 0. Do not claim RAG helps or hurts in general.

See `eval/results/report.md` for the full report.

## Testing and development

```bash
# Unit tests (excludes live and slow)
pytest -q
# 139 tests passed, 4 deselected

# Real embedding model (downloads MiniLM weights once)
pytest -m slow -q

# Real Gemini calls (needs GEMINI_API_KEY; uses free quota — run sparingly)
pytest -m live -q

# Lint and format
ruff check .
ruff format --check .
```

CI runs on ubuntu (3.11, 3.13) and windows (3.13); the Linux sandbox path is verified only by CI.

Project layout:
```
evalcode/
├── CLAUDE.md  docs/ARCHITECTURE.md  docs/PLAN.md
├── pyproject.toml  .gitignore  .env.example
├── LICENSE  README.md
├── src/evalcode/
│   ├── __init__.py
│   ├── config.py  errors.py
│   ├── rag/types.py  chunking.py  loaders.py
│   ├── rag/embeddings.py  store.py  retriever.py  ingest.py
│   ├── state.py  llm.py
│   ├── prompts.py  parsing.py  schemas.py  nodes/generate.py
│   ├── sandbox/runner.py  sandbox/errors.py  nodes/run_tests.py
│   ├── nodes/analyze_error.py  nodes/revise.py
│   ├── graph.py  nodes/terminal.py  nodes/human_review.py  persistence.py
│   ├── nodes/retrieve.py
│   ├── observability.py
│   └── cli.py
├── tests/{unit,integration}/  tests/fakes.py
├── eval/{tasks/*.yaml,results/}  src/evalcode/evaluation.py
├── .github/workflows/ci.yml
└── data/ logs/  (gitignored runtime dirs)
```

## Limitations and future work

From `docs/ARCHITECTURE.md` section 12 plus:
- Harder tasks and hidden tests needed to measure RAG effect reliably.
- Repeated samples (n > 1) for statistical significance.
- Linux sandbox verified only through CI (Windows dev environment).
- No network sandboxing; filesystem not jailed.
- Single-sample evaluation cannot attribute failures to RAG vs. model variance.

## License

MIT — see [LICENSE](LICENSE).