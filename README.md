# evalcode — Iterative code-generation agent (RAG + LangGraph)

`evalcode` is an iterative code-generation agent that behaves like a developer with a scratchpad: it retrieves version-matched Python API docs, writes a `solution.py` plus pytest tests with a Gemini free-tier model, runs the tests in a subprocess sandbox, analyzes failures, and revises in a bounded loop. The only required input is a task string; the only external requirement is a free `GEMINI_API_KEY`. Its notable property is that **every reliability concern — free-tier rate limits, retry budgets, output-format drift, sandbox isolation, and resumable human review — is designed into the loop rather than bolted on**.

```text
  INPUT                          PROCESSING (LangGraph loop)                     OUTPUT
  task: str
     │
     ▼
 ┌─────────────┐   docs   ┌─────────────┐  code,tests  ┌─────────────┐  PASS   ┌──────────────┐
 │  retrieve   │────────▶ │  generate   │────────────▶ │  run_tests  │────────▶│ human_review │
 │  (RAG)      │◀──────── │  (TextLLM)  │              │  (sandbox)  │         │ (interrupt)  │
 └─────────────┘ needs_   └─────────────┘              └──────┬──────┘         └──────┬───────┘
                  docs          ▲                          FAIL│               approve│
                                │                             ▼                      ▼
                                │                     ┌───────────────┐      ┌──────────────┐
                                └───── revise ◀───────│ analyze_error │      │   finalize   │
                                       (TextLLM)      └───────────────┘      └──────┬───────┘
                                                                                    ▼
                                                                        final_code + logs/<task_id>/
```

When it works, you get a runnable `solution.py` whose generated tests pass, printed with a per-run JSONL log of every node, attempt, token count, and retry.

---

## Table of Contents

- [What This Is](#what-this-is)
- [How It Works](#how-it-works)
  - [Core Logic](#core-logic)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
  - [Language-level dependencies](#language-level-dependencies)
  - [System-level dependencies](#system-level-dependencies)
- [Setup](#setup)
- [Running](#running)
  - [Running Components Individually (for debugging)](#running-components-individually-for-debugging)
- [Cleaning Up / Restarting](#cleaning-up--restarting)
- [Configuration](#configuration)
  - [Gemini / LLM](#gemini--llm)
  - [Embeddings & vector store](#embeddings--vector-store)
  - [Agent budgets & persistence](#agent-budgets--persistence)
  - [Sandbox](#sandbox)
  - [Retrieval](#retrieval)
  - [LangSmith](#langsmith)
- [Interface Reference](#interface-reference)
- [Running Tests](#running-tests)
- [Troubleshooting](#troubleshooting)
- [Known Limitations](#known-limitations)
- [Possible Extensions](#possible-extensions)
- [License](#license)

---

## What This Is

A single-process Python CLI plus a library that turns a natural-language coding task into tested Python code, grounded in documentation and validated by actually running the tests. The LLM is treated as an unreliable component: the agent assumes the model will ignore the output format, hit rate limits, and write wrong tests, and it is built to survive all three.

Major stages:

1. **Ingest** — introspect installed libraries (`json`, `re`, `pandas`, …) and local text docs into bounded chunks, embed them locally with MiniLM, and persist them in a Chroma collection with deterministic ids.
2. **Retrieve** — embed the task (or error-derived queries) and pull the top-k chunks by cosine similarity, dropped below a score floor and injected into the prompt as `[doc:<id>]` blocks.
3. **Generate / revise** — the model writes `solution.py` and `test_solution.py` as tagged plain text (`<explanation>`, `<code>`, `<tests>`, `<docs_used>`).
4. **Run tests in a sandbox** — a subprocess pytest run with a wall-clock timeout, scrubbed environment, and POSIX rlimits produces a structured `RunResult`.
5. **Analyze failures** — a deterministic classifier maps the failure to a category, extracts suspect symbols, and (for API misuse) builds re-retrieval queries; an opt-in LLM pass can refine the root cause.
6. **Human review** — a LangGraph `interrupt` pauses for approve / reject-with-feedback / edit, checkpointed in SQLite so it can resume in a new process.
7. **Finalize or fail** — three independent budgets (API retries, per-run LLM calls, graph retries and human rounds) bound the loop.
8. **Observe and evaluate** — one JSONL event per node plus a `summary.json`, and `evalcode eval` runs the same task set with RAG on and off for comparison.

The project demonstrates retrieval-augmented generation worn end-to-end (chunking, local embeddings, vector search, prompt grounding), LangGraph state machines with conditional edges and interrupts, subprocess isolation with platform-specific process control, and defensive free-tier LLM engineering (throttling, exponential backoff with jitter, `Retry-After` handling, daily-quota fail-fast, and a per-run call budget). It also handles real cross-platform friction: Windows has no `resource`/`os.killpg`, so rlimits are skipped and process trees are killed with `taskkill /F /T`; consoles default to cp1252, so the CLI reconfigures stdio and ASCII-sanitizes output.

---

## How It Works

```text
                         ┌──────────────────────────────────────────────────────┐
   task: str ──────────▶│ retrieve            in: task, error_analysis          │
                         │ (RetrieverLike)     out: retrieved_docs              │
                         │                     list[RetrievedDoc]               │
                         └──────────────────────────────┬───────────────────────┘
                                                        │ prompt(task + docs)
                                                        ▼
                         ┌──────────────────────────────────────────────────────┐
                         │ generate            in: task, retrieved_docs,        │
                         │ (TextLLM.invoke_text)   provided_tests               │
                         │                     out: code: str, tests: str       │
                         └──────────────────────────────┬───────────────────────┘
                                                        │ code, tests
                                                        ▼
                         ┌──────────────────────────────────────────────────────┐
                         │ run_tests           in: code: str, tests: str        │
                         │ (run_in_sandbox)    out: run_result: RunResult       │
                         └───────────┬────────────────────────────────┬─────────┘
                          passed=False│                      passed=True│
                                      ▼                                 ▼
              ┌───────────────────────────────┐      ┌───────────────────────────────┐
              │ analyze_error                 │      │ human_review                  │
              │ (deterministic + opt. TextLLM)│      │ (interrupt / Command resume)  │
              └───────────────┬───────────────┘      └───────────────┬───────────────┘
                needs_docs ───┤                        approve ──────┤ reject/edit
                     │        │                                      ▼          │
                     │        ▼                                finalize ─▶ END  │
                     │  ┌───────────────────────────────┐                       │
                     └─▶│ revise (TextLLM.invoke_text)  │◀─────────────────────┘
                        └───────────────┬───────────────┘
                                        │ code, tests  (loop back to run_tests)
                                        └────────────────────────────────────▶
```

- **`retrieve`** (`src/evalcode/nodes/retrieve.py`) — runs first when RAG is on, and again after `analyze_error` when the failure looks like API misuse. Consumes the task or `error_analysis.retrieval_queries`; produces `retrieved_docs: list[RetrievedDoc]` and appends to `retrieval_queries`. It takes a `RetrieverLike` (`retrieve(queries, k, library)`) and an optional `TextLLM`. Retrieval is **best-effort**: any retriever exception is logged and treated as zero docs, never a failed run. See [Troubleshooting](#troubleshooting) for the "RAG disabled" path.
- **`generate`** (`src/evalcode/nodes/generate.py`) — the mandatory first LLM call. Consumes `task`, `retrieved_docs`, and optional `provided_tests`; produces `code`, `tests`, `explanation`, bumps `attempt`, and increments `status`. An unparseable reply gets exactly **one** strict format re-ask; a second failure, or any `LLMError`, becomes `status="failed"` with a `failure_reason` instead of raising.
- **`run_tests`** (`src/evalcode/nodes/run_tests.py`) — calls the injected `sandbox(code, tests, timeout_s, mem_mb)` (default `run_in_sandbox`) and writes `run_result: RunResult`. Invariant: `passed is True` iff `category == "pass"` and `exit_code == 0`.
- **`analyze_error`** (`src/evalcode/nodes/analyze_error.py`) — deterministic analysis always runs (`src/evalcode/analysis.py`, no I/O); one optional LLM call refines `root_cause`/`fix_plan` only when `ANALYZE_WITH_LLM=true`, and any `LLMError` silently falls back to the deterministic result.
- **`revise`** (`src/evalcode/nodes/revise.py`) — a failure-aware prompt that includes the current code/tests, the failure report, the diagnosis, optional human feedback, and up to three prior failed attempts. It increments `attempt` always but `retries_used` only for automatic (test-failure-driven) revisions.
- **`human_review`** (`src/evalcode/nodes/human_review.py`) — calls `langgraph.types.interrupt(payload)` with **no side effects before it** (LangGraph re-executes the node on resume) and translates the resume response via the pure `apply_human_decision`.
- **`finalize` / `fail`** (`src/evalcode/nodes/terminal.py`) — set `status="approved"` with `final_code`, or `status="failed"` with a composed `failure_reason`.
- **Graph wiring** (`src/evalcode/graph.py`) — `build_graph` compiles the `StateGraph`; the routers `route_after_tests`, `route_after_llm`, `route_after_analysis`, `route_after_retrieve`, and `route_after_human` are pure module-level functions. RAG is enabled **iff** `Dependencies.retriever is not None`, and a checkpointed thread must be resumed with the same RAG topology.
- **`LLMClient`** (`src/evalcode/llm.py`) — the only thing that talks to Gemini. Nodes depend on the `TextLLM` protocol (`invoke_text(messages, *, purpose) -> LLMResponse`), never on `ChatOpenAI`. See [Troubleshooting](#troubleshooting) for 401/402/403/404/429 handling.
- **`RunLogger` / `traced_node`** (`src/evalcode/observability.py`) — wraps every node when logging is on and writes one redacted JSONL event per execution plus a rebuilt `summary.json`. Code, tests, and prompts are never logged; keys are redacted.

### Core Logic

- **Test result invariant**: `passed` is `True` iff `category == "pass"` and the sandbox `exit_code == 0`. Collection errors (pytest exit code 2) are `passed=False` with a category mapped from the exception type, before any keyword heuristics.
- **Error categories**: `pass`, `syntax_error`, `import_error`, `runtime_error`, `assertion_failure`, `timeout`, `no_tests`, `sandbox_error`; error analysis additionally yields `api_misuse` and `unknown`.
- **Retrieval scoring**: Chroma cosine space returns `score = 1 - distance` in `[-1, 1]` (1.0 = identical). Chunks below `RETRIEVAL_MIN_SCORE` (default `0.2`) are dropped; multiple queries keep the max score per chunk id; the list is sorted descending and cut to `RETRIEVAL_TOP_K` (default `5`). Merged new+prior docs are capped at `RETRIEVAL_MAX_DOCS` (default `8`).
- **Re-retrieval rule**: `needs_docs = (category == "api_misuse") and len(retrieval_queries) > 0`. After a failure, `route_after_analysis` goes to `retrieve` only if `needs_docs` and **not all** queries are already in `retrieval_queries` — that accumulated list is the dedupe memory that prevents repeat-retrieval loops.
- **Backoff formula**: `delay = min(LLM_BACKOFF_MAX_S, LLM_BACKOFF_BASE_S * 2**attempt) * rng()`, then `max(delay, required_wait)` where `required_wait` comes from `Retry-After` / `X-RateLimit-Reset` / Gemini's `retryDelay`. `attempt` is per-call; the lifetime retry counter never gates retries.
- **Fail-fast rules**: a 429 whose body matches daily-limit keywords, or a required wait above `LLM_MAX_WAIT_S` (default `120`), raises `DailyQuotaExceeded` immediately instead of burning retries. HTTP 401/403/402 → `LLMAuthError`, 404 → `LLMModelError`, other 4xx → `LLMRequestError`, exhausted retries → `LLMUnavailable`.
- **Throttle**: at least `LLM_MIN_INTERVAL_S` (default `3.0`) between logical calls, enforced from the last call's monotonic clock.
- **Three budgets**: API level `LLM_MAX_API_RETRIES` (default `5`) per call; run level `MAX_LLM_CALLS_PER_RUN` (default `10`) logical calls → `LLMBudgetExceeded` → run ends `failed`; graph level `MAX_RETRIES` (default `3`) automatic revisions and `MAX_HUMAN_ROUNDS` (default `2`) human rejections.
- **Parsing safety**: the parser accepts case-insensitive tags, stray whitespace, markdown fences inside tags, and a missing closing tag; if no `<code>` tag exists at all it falls back to the first fenced block as code and the second as tests. `ParseError` carries only a short reason, never the model's raw text.
- **No side effects before `interrupt()`**: the `human_review` node builds its payload and calls `interrupt` first, because LangGraph re-runs the node from the start on resume.
- **Failure containment**: mandatory-node `LLMError`s become `status="failed"` (not exceptions); optional LLM calls (`ANALYZE_WITH_LLM`, `QUERY_REWRITE_WITH_LLM`) are off by default and degrade to deterministic behavior.

---

## Project Structure

```text
evalcode/
├── pyproject.toml                 # hatchling build, deps, extras [dev,corpus], ruff + pytest config
├── .env.example                   # every setting with its default; copy to .env
├── .gitignore                     # ignores .env, data/, logs/, eval/results/raw/
├── CLAUDE.md                      # engineering conventions + per-task status log
├── LICENSE                        # MIT
├── README.md                      # this file
├── docs/
│   ├── ARCHITECTURE.md            # design of record: graph, state schema, sandbox, risks
│   └── PLAN.md                    # the 15-task implementation plan
├── src/evalcode/
│   ├── __init__.py                # __version__
│   ├── __main__.py                # `python -m evalcode`
│   ├── cli.py                     # Typer app: ingest | search | run | resume | eval
│   ├── config.py                  # pydantic-settings Settings; get_settings()
│   ├── errors.py                  # ConfigError, ParseError, LLMError hierarchy
│   ├── state.py                   # AgentState + nested TypedDicts, merge_usage reducer
│   ├── llm.py                     # LLMClient: throttle, backoff, budget, usage
│   ├── prompts.py                 # system prompts + message builders
│   ├── parsing.py                 # tolerant tagged-text parser
│   ├── schemas.py                 # CodeBundle, extract_imports
│   ├── analysis.py                # deterministic error classification
│   ├── graph.py                   # build_graph, routers, run/resume helpers, Dependencies
│   ├── persistence.py             # SQLite checkpointer context manager
│   ├── evaluation.py              # resumable RAG on/off eval harness
│   ├── observability.py           # RunLogger, traced_node, redact, LangSmith config
│   ├── nodes/                     # generate, run_tests, analyze_error, revise, retrieve,
│   │                              #   human_review, terminal
│   ├── sandbox/
│   │   ├── runner.py              # run_in_sandbox: subprocess + timeout + rlimits + env scrub
│   │   └── errors.py              # junit parsing and category classification
│   └── rag/
│       ├── types.py               # DocChunk, RetrievedDoc
│       ├── chunking.py            # stable_id, truncate_text, split_markdown
│       ├── loaders.py             # introspection + text-file chunk generators
│       ├── embeddings.py          # Embedder protocol, HFEmbedder, FakeEmbedder
│       ├── store.py               # Chroma PersistentClient wrapper
│       ├── retriever.py           # multi-query top-k merge (also a `python -m` CLI)
│       └── ingest.py              # ingest() (also a `python -m` CLI)
├── tests/
│   ├── conftest.py                # autouse env-isolation fixture
│   ├── fakes.py                   # FakeChatModel, ScriptedLLM, FakeEmbedder, FakeRetriever
│   ├── unit/                      # config, llm, state, nodes, rag, sandbox, graph, cli, eval
│   └── integration/               # live Gemini tests (marker: live / slow)
├── eval/
│   ├── tasks/*.yaml               # 12 demo tasks: id, task, tests, reference solution
│   └── results/                   # committed report.md; raw/ is git-ignored
├── scripts/probe_retrieval.py     # dev-only retrieval-quality probe (hit@k, MRR)
└── data/  logs/                   # git-ignored runtime dirs (Chroma, SQLite, JSONL logs)
```

> The layout separates the RAG layer (`rag/`, pure + offline), the execution boundary (`sandbox/`), and the orchestration layer (`graph.py` + `nodes/`) so that only `graph.default_dependencies` constructs a real LLM client and only `sandbox/runner.py` spawns processes. Library modules import torch, sentence-transformers, `langchain_huggingface`, and `chromadb` **lazily** (inside functions), which keeps the default test suite fast and lets unit tests inject fakes without loading gigabytes of weights.

---

## Prerequisites

- **OS**: Windows is the primary, tested dev environment (Windows 11, Python 3.13.3). Linux should work and is exercised by CI (ubuntu, Python 3.11 and 3.13). macOS shares the POSIX code path but is untested.
- **Runtime**: Python `>=3.11` (`requires-python` in `pyproject.toml`). The dev venv and CI use 3.13.
- **Language**: Python only. Non-Python targets and multi-file projects are explicit non-goals.
- **Hardware**: CPU only — no GPU required. A few hundred MB of disk for the venv, the embedding-model cache, the Chroma index, and the SQLite checkpoint.
- **Network**: required for the first embedding-model download and for every Gemini call.
- **Credentials**: one free `GEMINI_API_KEY` from <https://aistudio.google.com/apikey>.

### Language-level dependencies

```bash
pip install -e ".[dev,corpus]"
```

- `langgraph`, `langgraph-checkpoint-sqlite` — the state machine, conditional edges, `interrupt`, and the SQLite checkpointer that makes resume work across processes.
- `langchain-core`, `langchain-openai`, `openai` — message types and the `ChatOpenAI` client pointed at Gemini's OpenAI-compatible endpoint.
- `langchain-huggingface`, `sentence-transformers`, `chromadb` — local embeddings (MiniLM) and the persistent vector store.
- `pydantic`, `pydantic-settings`, `python-dotenv` — typed settings, `.env` loading, and validation.
- `typer`, `rich` — the CLI and its live, syntax-highlighted review display.
- `pyyaml` — reads the evaluation task files.
- `pytest`, `pytest-cov`, `ruff` (`dev` extra) — test runner and lint/format.
- `pandas`, `numpy`, `requests` (`corpus` extra) — installed so their APIs can be introspected into the index and used by eval tasks.

### System-level dependencies

```bash
# Optional, Linux/Windows: install the CPU-only torch wheel first to skip multi-GB CUDA wheels.
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

- `torch` is pulled in transitively by `sentence-transformers`; installing the CPU wheel first avoids large CUDA downloads on Linux. No other system packages are required. The `all-MiniLM-L6-v2` weights are downloaded on first use (first `evalcode ingest`, the `slow` test, or any RAG run).

---

## Setup

1. **Create the virtual environment** (one-time).

   ```bash
   python -m venv .venv
   ```

2. **Activate it** (per session). On Windows PowerShell:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

   On Linux/macOS:

   ```bash
   source .venv/bin/activate
   ```

   > If you skip activation, call the interpreter directly (`.venv\Scripts\python` on Windows) for the rest of these commands.

3. **Install the package** (one-time; re-run after dependency changes).

   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional CPU torch, do this first
   pip install -e ".[dev,corpus]"
   ```

   `-e` installs in editable mode so `src/evalcode` changes take effect without reinstalling; the extras pull in the dev tools and the corpus libraries.

4. **Create the environment file and set your key** (one-time).

   ```powershell
   copy .env.example .env      # Windows
   ```

   ```bash
   cp .env.example .env        # Linux/macOS
   ```

   Edit `.env` and set `GEMINI_API_KEY=<your key>`. `.env` is git-ignored; blank values fall back to the defaults in the next section.

5. **Build the documentation index** (one-time per corpus; re-run with `--reset` if you change loaders, the embedding text, or the embedding model).

   ```bash
   evalcode ingest --libs json,re,pathlib
   ```

   Omitting `--libs` ingests everything in `DOC_LIBRARIES`. The first run downloads the MiniLM weights. `--reset` clears the collection before ingesting.

6. **Run a task** — see [Running](#running).

---

## Running

Simplest full run (non-interactive, auto-approves the result; runs on a throwaway thread id):

```bash
evalcode run "write a function that parses a JSON string and returns its keys" --auto-approve
```

Interactive run (pauses for human review and exits code 5 if stdin is not a terminal):

```bash
evalcode run "write a function that parses a JSON string and returns its keys"
evalcode resume <TASK_ID> --approve
evalcode resume <TASK_ID> --reject "needs type hints"
evalcode resume <TASK_ID> --edit-file edited_solution.py
```

The run prints `TASK_ID: <hex>` at the start; use that id with `resume`. Add `--no-rag` to skip retrieval, `--tests path.py` to supply fixed ground-truth tests, `--output out.py` to write the final code, and `--show-code/--no-show-code` to control the final print.

### Running Components Individually (for debugging)

```bash
# Search the index without running the agent.
evalcode search "json.loads" -k 3

# Equivalent module entry points (no console script needed).
python -m evalcode.rag.ingest --libs json,re --reset
python -m evalcode.rag.retriever "parse a JSON string" -k 5

# Verify the eval reference solutions in the real sandbox, with no LLM calls.
evalcode eval --verify-refs

# Run the RAG-on/RAG-off comparison (resumable, quota-aware).
evalcode eval --limit 5
evalcode eval --modes norag        # RAG off only
evalcode eval --report-only        # rebuild report from existing raw results
```

> **Note**: `evalcode run` without `--auto-approve` always opens the SQLite checkpointer and will pause at `human_review`; in a non-interactive shell it prints `AWAITING REVIEW` and exits with code 5 instead of prompting. `evalcode resume` infers whether the original run used RAG from the checkpointed history and **refuses to resume** if the index is now missing, because the graph topology would differ. `--interactive`/`--no-interactive` override the automatic `stdin.isatty()` detection.

Sanity checks:

```bash
python -m evalcode --help                  # lists search | ingest | run | resume | eval
python -m evalcode run --help              # every run flag
evalcode eval --verify-refs                # expect PASS for all 12 reference solutions
```

---

## Cleaning Up / Restarting

State lives in three git-ignored places: `data/chroma/` (the vector index), `data/checkpoints.sqlite` (paused task state), and `logs/<task_id>/` (JSONL events + summary). To reset, stop any running CLI process first, then remove the directories you want to clear.

```powershell
# Windows: clear the index and checkpoints, keep nothing
Remove-Item -Recurse -Force data, logs
# then rebuild
evalcode ingest --libs json,re,pathlib
```

```bash
# Linux/macOS
rm -rf data logs
evalcode ingest --libs json,re,pathlib
```

> Failing to close a run cleanly (for example killing the process) can leave the SQLite file or the Chroma directory locked on Windows; the CLI normally closes both via the checkpointer context manager and `close_dependencies`. If a delete fails, wait for the process to exit and retry. Deleting `data/chroma/` without re-ingesting forces every subsequent run into RAG-off mode with a warning.

Optional alias for a clean re-ingest:

```powershell
Set-Alias reindex { Remove-Item -Recurse -Force data/chroma; evalcode ingest }
```

---

## Configuration

All settings come from environment variables or `.env` and are declared in `src/evalcode/config.py` (the full list and comments are in `.env.example`; see `docs/ARCHITECTURE.md` §10). Real environment variables always win over `.env`, blank values are treated as unset, and the key is held as a `SecretStr`.

### Gemini / LLM

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GEMINI_API_KEY` | *(required)* | Free API key from <https://aistudio.google.com/apikey>. |
| `LLM_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai/` | OpenAI-compatible endpoint for Gemini. |
| `LLM_MODEL` | `gemini-3.5-flash-lite` | Model id; blank in `.env` falls back to the default. **Do not** set it to an "inkling" model — those return HTTP 403 for plain API-key clients. |
| `LLM_TEMPERATURE` | `0.2` | Sampling temperature (`>= 0`). |
| `LLM_MAX_TOKENS` | `8192` | Max output tokens per completion; caps reasoning-model token burn. |
| `LLM_TIMEOUT_S` | `120` | Per-request timeout, seconds. |
| `LLM_MAX_API_RETRIES` | `5` | Retries per logical call on transient errors (429/408/5xx/timeout/empty). |
| `LLM_BACKOFF_BASE_S` | `2.0` | Exponential backoff base, seconds. |
| `LLM_BACKOFF_MAX_S` | `60.0` | Backoff cap, seconds. |
| `LLM_MAX_WAIT_S` | `120.0` | Give up with `DailyQuotaExceeded` if a required wait exceeds this many seconds. |
| `LLM_MIN_INTERVAL_S` | `3.0` | Minimum seconds between LLM calls (free-tier per-minute throttling). |
| `MAX_LLM_CALLS_PER_RUN` | `10` | Hard cap on logical LLM calls per run. **Do not raise** this without reason — every extra call spends free-tier quota. |

### Embeddings & vector store

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model, downloaded once, CPU-only. Changing it requires a re-ingest (`--reset`). |
| `CHROMA_DIR` | `data/chroma` | Chroma persistent-client directory. |
| `COLLECTION_NAME` | `python_docs` | Chroma collection name. |
| `DOC_LIBRARIES` | `json,re,datetime,collections,itertools,pathlib,dataclasses,csv,sqlite3,argparse,pandas,numpy` | Comma-separated libraries introspected at ingest time. |
| `DOCS_DIR` | `data/docs` | Directory scanned for optional local `.md`/`.rst`/`.txt` docs. |

### Agent budgets & persistence

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHECKPOINT_DB` | `data/checkpoints.sqlite` | LangGraph SQLite checkpointer; enables resume across CLI invocations. |
| `LOG_DIR` | `logs` | Run-log directory; each run writes `logs/<task_id>/`. |
| `MAX_RETRIES` | `3` | Automatic (test-failure-driven) revisions per round. |
| `MAX_HUMAN_ROUNDS` | `2` | Human-requested rejections before the run fails. |

### Sandbox

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SANDBOX_TIMEOUT_S` | `10` | Wall-clock timeout, seconds, for a sandboxed test run. |
| `SANDBOX_MEM_MB` | `1024` | Address-space rlimit, MB. **Ignored on Windows** (no `resource` module); only the wall-clock timeout enforces limits there. |

### Retrieval

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RETRIEVAL_TOP_K` | `5` | Chroma results per query. |
| `RETRIEVAL_MAX_DOCS` | `8` | Max documents kept in state after merging and dedup. |
| `RETRIEVAL_MIN_SCORE` | `0.2` | Drop documents scoring below this cosine similarity (`0.0`–`1.0`). |
| `CONTEXT_MAX_CHARS` | `6000` | Cap on injected document context in prompts, characters. |

### LangSmith

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LANGSMITH_TRACING` | `false` | Enable LangSmith tracing. |
| `LANGSMITH_API_KEY` | *(unset)* | LangSmith key; tracing stays off if this is missing. |
| `LANGSMITH_PROJECT` | `evalcode` | Project name for traces. |

> **Debug/optional parameters** (off by default; the deterministic paths always run): `ANALYZE_WITH_LLM` (`false`) adds one LLM call to refine error diagnosis; `QUERY_REWRITE_WITH_LLM` (`false`) uses an LLM to rewrite the first-pass retrieval query; `LANGSMITH_TRACING` + `LANGSMITH_API_KEY` add remote tracing. All three consume extra quota or network and are not needed for a normal run.

---

## Interface Reference

| Name | Type | Producer → Consumer |
|------|------|---------------------|
| `evalcode run` | CLI command | user → graph (`stream_task`) |
| `evalcode resume` | CLI command | user → graph (`stream_resume_task` via `Command(resume=...)`) |
| `evalcode ingest` | CLI command | user → `rag.ingest` → Chroma |
| `evalcode search` | CLI command | user → `Retriever` → scored docs |
| `evalcode eval` | CLI command | user → `evaluation.run_eval` |
| `TextLLM.invoke_text` | protocol method | nodes → Gemini via `LLMClient` |
| `RetrieverLike.retrieve` | protocol method | `nodes/retrieve` → `Retriever` |
| `run_in_sandbox` | callable | `nodes/run_tests` → subprocess pytest |
| `interrupt` / `Command` | LangGraph | `nodes/human_review` ↔ resuming process |
| `events.jsonl` / `summary.json` | files | `RunLogger` → `logs/<task_id>/` |

The LLM output contract (parsed by `src/evalcode/parsing.py`):

```text
<explanation>
1-3 sentences: what was implemented and which documented APIs were used.
</explanation>
<code>
complete solution.py source
</code>
<tests>
complete test_solution.py source (pytest, uses "from solution import ...")
</tests>
<docs_used>
comma-separated ids of documentation blocks actually used   # optional
</docs_used>
```

The `run_tests` result stored in state (`RunResult` in `src/evalcode/state.py`):

```text
{
    "passed": bool,                    # True iff category == "pass" and exit_code == 0
    "category": "pass" | "syntax_error" | "import_error" | "runtime_error"
                | "assertion_failure" | "timeout" | "no_tests" | "sandbox_error",
    "exit_code": int,
    "timed_out": bool,
    "duration_s": float,
    "tests_total": int,
    "tests_failed": int,
    "failures": [{"test_name": str, "error_type": str, "message": str, "traceback": str}],
    "stdout": str,                     # truncated
    "stderr": str,                     # truncated
}
```

The human-review resume payload (`src/evalcode/nodes/human_review.py`):

```text
{"decision": "approve"}
{"decision": "reject", "feedback": "add type hints"}   # feedback must be non-empty
{"decision": "edit",   "code": "<full solution.py text>"}  # code must be non-empty
```

Full contracts: `docs/ARCHITECTURE.md` §2 (output protocol), §3.2 (state schema), §7 (human-in-the-loop).

---

## Running Tests

```bash
pytest -q                 # default suite: excludes live and slow; 139 passed, 4 deselected
pytest -m slow -q         # real embedding model (downloads MiniLM weights once)
pytest -m live -q         # real Gemini calls (needs GEMINI_API_KEY; uses free quota)
ruff check .
ruff format --check .
```

- **Mocked**: `FakeChatModel` (scripted message/exception scripts for `LLMClient`), `ScriptedLLM` (implements `TextLLM` with scripted tagged replies and usage), `FakeEmbedder` (deterministic hashing-based vectors), and `FakeRetriever`. An autouse fixture in `tests/conftest.py` strips every settings env var and points `find_dotenv` at a non-existent path, so the default suite never reads your shell environment or your real `.env`.
- **Real**: sandbox tests run the actual subprocess on tiny snippets; backoff tests inject fake `sleep`/`clock` so they run instantly; graph tests use an in-memory saver.
- **Verified**: `pytest -q` reports `139 passed, 4 deselected` (the 4 are the `live`/`slow` opt-ins). Default tests cover config and secret hygiene, the LLM retry/backoff/quota taxonomy, parsing and the generate/revise nodes, sandbox classification (including pytest exit-code-2 collection errors), graph routing and human-in-the-loop pause/resume, observability event fields, the CLI, and the eval harness. CI additionally runs `ruff check` and `evalcode eval --verify-refs`.

---

## Troubleshooting

Highlights, from the failures this project actually handles (`docs/ARCHITECTURE.md` §6 and §11 are the fuller reference):

- **`GEMINI_API_KEY is not set` and exit code 4** — the key is missing or blank. Cause: `require_api_key()` raises `ConfigError`. Fix: copy `.env.example` to `.env`, set `GEMINI_API_KEY`, and re-run. The message never contains key material.
- **Every call fails with HTTP 403** — the model is restricted to specific clients. Cause: some Gemini models (the "inkling" family) reject plain API-key clients. Fix: set `LLM_MODEL` to a model that allows plain API-key access; the error text is truncated into the message. Do **not** keep retrying a 403 — `LLMClient` raises `LLMAuthError` without retrying.
- **HTTP 404 model not found** — the free model slug was renamed or retired. Fix: update `LLM_MODEL` (free slugs change over time). `LLMModelError` raises immediately with this hint.
- **Exit code 3 / `DailyQuotaExceeded`** — the free daily quota is gone. Cause: a daily-limit 429, or a required wait longer than `LLM_MAX_WAIT_S`. Fix: stop and wait for the quota to reset; do **not** immediately retry. `evalcode eval` is resumable, so re-run the same command later and completed cells are skipped.
- **Exit code 5 / `AWAITING REVIEW`** — a non-interactive run paused at human review. Fix: `evalcode resume <TASK_ID> --approve` (or `--reject "..."` / `--edit-file PATH`). The task id is printed as `TASK_ID:` at the start of the run.
- **`RAG disabled: no usable index (run \`evalcode ingest\`)`** — the Chroma directory is missing or empty. Fix: `evalcode ingest --libs ...`. Retrieval failures never fail a run; they just degrade to no docs.
- **`refusing to resume with a different graph topology`** — the task started with RAG but the index no longer exists. Fix: re-ingest before resuming, so the graph shape matches the checkpoint.
- **Sandbox timeouts or memory not enforced on Windows** — expected. Cause: `resource`/`os.killpg` are POSIX-only; on Windows rlimits are skipped (a warning is logged) and the process tree is killed with `taskkill /F /T /PID <pid>`. The wall-clock timeout is the only resource guard there. Negative finding: memory-limit behavior cannot be tested on Windows; the POSIX memory test is skipped outside Linux/CI.
- **A generated test that fails to collect reports `passed=False`** — pytest exits with code 2 on collection errors, which the sandbox maps to a category from the exception type and never counts as a pass. Diagnostic: `evalcode eval --verify-refs` runs the reference solutions in the real sandbox with no LLM and should print `PASS` for every task.
- **Garbled or `?` characters in Windows console output** — the console defaults to cp1252. The CLI reconfigures stdout/stderr with `errors="replace"` and sanitizes rendered text; library modules still must not print non-ASCII, which is why only the `rag` `python -m` entry points are allowed to print.
- **The first ingest or `slow` test appears to hang** — it is downloading the MiniLM weights once. Subsequent runs use the cache.

---

## Known Limitations

- **RAG did not improve pass rate on the bundled benchmark.** On the committed results (`eval/results/report.md`), RAG passed 11/12 versus 12/12 without, and used more tokens (2248 vs 943 mean). This is one sample per (task, mode) at `temperature=0.2`, so it is indicative only — do **not** claim RAG helps or hurts in general from it.
- **Acceptance tests are visible to the model.** The eval provides `tests` in the prompt, so pass rate measures "can satisfy the visible spec", not generalization to hidden tests.
- **Single sample, small n.** Twelve tasks and one run per cell mean the reported differences are not statistically meaningful; the report itself flags `K < 10 pairs`.
- **The sandbox is not a security boundary.** It protects against accidental damage and runaway code (timeout, scrubbed env, POSIX rlimits) but does not block network access or jail the filesystem; adversarial code is out of scope.
- **Windows skips resource limits.** Only the wall-clock timeout applies on Windows, and the Linux sandbox path is verified only through CI (the dev machine is Windows).
- **Single provider, single file, Python only.** Gemini is the only LLM; multi-file projects and non-Python languages are explicit non-goals; the model output is a `solution.py` plus one test file.
- **Quota-bounded task size.** `MAX_LLM_CALLS_PER_RUN` and the graph retry caps limit how large or iterative a task can be; hard tasks will end `failed` rather than run forever.
- **Free model slugs drift.** `gemini-3.5-flash-lite` is only the default; if Google retires it, runs fail with a clear `LLMModelError` until `LLM_MODEL` is updated.

---

## Possible Extensions

- **Hybrid retrieval (BM25 + dense, RRF)** — add a lexical scorer alongside Chroma and fuse the rankings to catch exact-symbol queries that dense embeddings miss.
- **Docker sandbox backend** — run the test subprocess in a container to get real network and filesystem isolation without the POSIX/Windows branching.
- **Repeated samples (n > 1)** — loop the eval harness per cell and report variance so RAG's effect becomes measurable.
- **Hidden acceptance tests** — hold back a second test set from the prompt to measure generalization instead of spec-following.
- **Multi-file output** — let the tagged protocol emit a file manifest and have the sandbox write a small package tree.
- **Streaming token display** — surface model tokens live in the Rich UI instead of per-node summaries.

---

## License

MIT — see [`LICENSE`](LICENSE).

*Built as a portfolio project that is nonetheless real end to end: a real local RAG pipeline, a real LangGraph state machine with interrupts, a real subprocess sandbox with platform-specific process control, real free-tier quota engineering, and a real — if humbling — evaluation showing RAG at the ceiling on easy tasks.*
