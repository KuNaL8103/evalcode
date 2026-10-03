# evalcode — Architecture

Design of record. If an implementation decision changes this design, update this file in the same commit and note it in CLAUDE.md "Current status".

## 1. Goals and non-goals

**Goals**
- Iterative code-generation agent that mimics a developer: read docs → write code (+ tests) → run them → learn from failures → revise.
- Ground generation in real, version-matched Python API documentation (RAG) to cut hallucinated imports and APIs.
- Bounded, observable retry loop that respects **free-tier LLM quotas**; human approval before code is finalized.
- Everything testable offline: unit tests never call a real LLM or the network.

**Hard constraints**
- The only LLM provider is **OpenRouter free models**, accessed through LangChain `ChatOpenAI` with `base_url="https://openrouter.ai/api/v1"`. The key comes only from the `OPENROUTER_API_KEY` environment variable (optionally loaded from a git-ignored `.env`). No hardcoded keys, no other providers.
- Embeddings are local (no API, no quota).

**Non-goals**: multi-file projects, non-Python languages, adversarial-grade sandboxing, web UI, multi-user serving.

## 2. Technology choices

| Concern | Choice | Rationale | Alternatives considered |
|---|---|---|---|
| Language | Python 3.11+ | Required by the LangGraph ecosystem; modern typing | — |
| Orchestration | LangGraph `StateGraph` | Explicit state, conditional edges, checkpointing, native `interrupt` for human-in-the-loop | Hand-rolled loop (no persistence/interrupts) |
| LLM provider | OpenRouter via `langchain_openai.ChatOpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)`; default model `qwen/qwen3.8-27b:free`, set by `LLM_MODEL` | Zero cost, OpenAI-compatible API, one variable swaps between free models (e.g. `qwen/qwen3-coder:free`). Free slugs change often, so the model is never hardcoded outside config defaults | Paid providers (excluded by constraint) |
| LLM output protocol | Tagged plain text: `<explanation>`, `<code>`, `<tests>`, optional `<docs_used>`; parsed locally | Free models/providers differ in tool-calling and JSON-mode support; plain text works everywhere. Reasoning models' `<think>…</think>` blocks are stripped before parsing | `with_structured_output`/tool calling (unreliable on free endpoints) |
| LLM reliability layer | `LLMClient` wrapping the chat model (see §5) | Free tiers have low per-minute and daily limits; the agent loops, so backoff, throttling, and a per-run call budget are mandatory | Relying on the SDK's built-in retries (too blunt, no budget) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` via `langchain_huggingface.HuggingFaceEmbeddings` (local, CPU, normalized, 384-d) | No key, no quota, small download. **Max sequence ≈256 word-pieces**, so chunks are capped around 1000 chars with the signature first | `bge-small`, `e5` (need query prefixes) — swappable through the `Embedder` protocol and `EMBEDDING_MODEL` |
| Vector store | ChromaDB `PersistentClient` (embedded, local) | No server, persistent, metadata filters, cosine similarity, idempotent upserts by id. Used directly (embeddings supplied explicitly) | FAISS (weak metadata), Qdrant/pgvector (overkill) |
| Doc corpus | Introspection of *installed* packages (signature + docstring per public API object) plus optional local .md/.rst/.txt files | Version-matched to the interpreter that runs the code → targets hallucinated APIs; offline and reproducible | Scraping web docs (brittle) |
| Sandbox | `subprocess` of the current interpreter in a temp dir: timeout, rlimits (POSIX), scrubbed env, process-group kill | Real isolation of crashes/hangs/memory from the agent; enforceable timeouts. In-process `exec()` can't be timed out, can hang/crash the agent, and pollutes `sys.modules` | In-process `exec` (rejected); Docker backend (future) |
| Test runner | `pytest` inside the sandbox, `--junitxml` | Structured results; collection errors distinguishable from failures | Plain asserts |
| Checkpointing | `langgraph-checkpoint-sqlite` (`SqliteSaver`); `MemorySaver` in tests | Persistent pause/resume across CLI invocations | Postgres (overkill) |
| Config | `python-dotenv` + `pydantic-settings` | `.env` for local dev, real env vars always win; typed settings; blank values treated as unset | — |
| CLI | Typer + Rich | Typed commands, live display, syntax-highlighted review | argparse |
| Tooling | `hatchling`, venv + pip, `pytest`, `ruff` | Boring and reliable | — |

## 3. Graph design

### 3.1 Flow

```
START ─► retrieve ─► generate ─► run_tests ─┬─ PASS ───────────────► human_review ─► (see 3.1b)
(RAG off: START ─► generate)        ▲       │
                                    │       └─ FAIL ─► retries_used < max_retries ?
                                    │                     │ yes                    │ no
                                    │                     ▼                        ▼
                                    │               analyze_error                fail ─► END
                                    │                     │
                                    │       needs_docs?   ├── yes ─► retrieve ─┐
                                    │       (and RAG on)  └── no  ──────────────┤
                                    │                                           ▼
                                    └───────────────────────────────────────── revise

generate / revise ── LLM failure (daily quota, auth, call budget, unparseable output) ──► fail ─► END
```

3.1b Human review sub-flow:

```
human_review ─┬─ approve ─► finalize ─► END
              ├─ edit   (human supplies code) ─► run_tests        (re-validated; retry budget unchanged)
              └─ reject (+ feedback) ─► human_rounds += 1 ─┬─ human_rounds <= max_human_rounds ─► revise ─► run_tests
                                                           └─ otherwise ─► fail ─► END
```

`retrieve` is used twice: before the first generation (queries from the task) and after `analyze_error` when the failure looks like API misuse (queries from the error). `route_after_retrieve` sends to `generate` if no code exists yet, else to `revise`.

### 3.2 State schema (`state.py`)

All nested values are plain `TypedDict`/JSON-serializable so checkpoints serialize cleanly. Do not name classes `Test*` (pytest collection).

```python
class RetrievedDoc(TypedDict):
    id: str; text: str; score: float
    library: str; qualname: str; import_path: str

class TokenUsage(TypedDict, total=False):
    input_tokens: int; output_tokens: int; total_tokens: int
    llm_calls: int; api_retries: int; estimated_calls: int   # estimated = provider gave no usage
    wait_s: float                                            # time spent in backoff/throttle

class RunFailure(TypedDict):
    test_name: str; error_type: str; message: str; traceback: str

class RunResult(TypedDict):
    passed: bool
    category: Literal["pass", "syntax_error", "import_error", "runtime_error",
                      "assertion_failure", "timeout", "no_tests", "sandbox_error"]
    exit_code: int; timed_out: bool; duration_s: float
    tests_total: int; tests_failed: int
    failures: list[RunFailure]
    stdout: str; stderr: str            # truncated

class ErrorAnalysis(TypedDict):
    category: str
    root_cause: str
    fault: Literal["code", "tests", "unknown"]
    fix_plan: str
    needs_docs: bool
    retrieval_queries: list[str]
    suspect_symbols: list[str]

class StepEvent(TypedDict):
    node: str; attempt: int; ts: str; summary: dict[str, Any]

class AgentState(TypedDict, total=False):
    # input
    task_id: str; task: str
    provided_tests: str | None          # if set, tests are immutable ground truth
    auto_approve: bool                  # skip human_review (CI / eval)
    # budgets
    attempt: int                        # total generations so far (monotonic)
    retries_used: int; max_retries: int # automatic revisions in the current round
    human_rounds: int; max_human_rounds: int
    # RAG
    retrieval_queries: list[str]
    retrieved_docs: list[RetrievedDoc]
    # artifacts
    code: str; tests: str; explanation: str
    # feedback
    run_result: RunResult | None
    error_analysis: ErrorAnalysis | None
    human_decision: Literal["approve", "reject", "edit"] | None
    human_feedback: str | None
    # outcome
    status: Literal["running", "awaiting_review", "approved", "failed"]
    final_code: str | None; failure_reason: str | None
    # bookkeeping (reducers)
    history: Annotated[list[StepEvent], operator.add]
    token_usage: Annotated[TokenUsage, merge_usage]
```

### 3.3 Nodes

| Node | Reads | Writes | LLM call? |
|---|---|---|---|
| `retrieve` | task, error_analysis, retrieved_docs | retrieval_queries, retrieved_docs, history | Only if `QUERY_REWRITE_WITH_LLM=true` (default off) |
| `generate` | task, retrieved_docs, provided_tests | code, tests, explanation, attempt, status, token_usage, history (or status=failed + failure_reason on LLM failure) | Yes (1, +1 if the format must be re-requested) |
| `run_tests` | code, tests | run_result, history | No |
| `analyze_error` | run_result, code, tests, history | error_analysis, token_usage, history | Only if `ANALYZE_WITH_LLM=true` (default off); deterministic parsing always runs |
| `revise` | code, tests, run_result, error_analysis, human_feedback, retrieved_docs, history | code, tests, explanation, attempt, retries_used, human_feedback→None, token_usage, history (or failed) | Yes (1) |
| `human_review` | code, tests, run_result | human_decision, human_feedback, human_rounds, code (on edit), history | No (`interrupt`) |
| `finalize` | code | final_code, status=approved | No |
| `fail` | run_result, attempt, failure_reason | status=failed, failure_reason | No |

Node factories take their dependencies (`make_generate_node(llm)`, `make_retrieve_node(retriever, llm, settings)`) where `llm` is a `TextLLM` (anything with `invoke_text`), so tests inject `ScriptedLLM`. Optional LLM calls fall back to deterministic behavior on any `LLMError`; mandatory calls (generate/revise) convert `LLMError` into `status="failed"` with a clear `failure_reason` instead of raising.

### 3.4 Conditional edges

- `route_after_tests`: `passed` → `human_review` (or `finalize` if `auto_approve`); failed and `retries_used < max_retries` → `analyze_error`; else → `fail`.
- `route_after_llm` (after `generate` and `revise`): `status == "failed"` → `fail`; else → `run_tests`.
- `route_after_analysis`: `error_analysis.needs_docs` and RAG enabled → `retrieve`; else → `revise`.
- `route_after_retrieve`: `code` empty → `generate`; else → `revise`.
- `route_after_review`: approve → `finalize`; edit → `run_tests`; reject → `revise` if `human_rounds <= max_human_rounds` else `fail`.
- Compile/run with `recursion_limit=100` (default 25 is too low for long loops).

### 3.5 Budgets (three layers, all protecting the free quota)

1. **API level** (inside `LLMClient`): exponential backoff on transient errors, `LLM_MAX_API_RETRIES` per call (default 5).
2. **Run level**: `MAX_LLM_CALLS_PER_RUN` (default 10) hard cap on logical LLM calls → `LLMBudgetExceeded` → run ends as `failed`. With default flags a full 4-generation round uses at most 4 calls (plus format re-asks).
3. **Graph level**: `max_retries=3` automatic revisions per round (≤4 generations); `max_human_rounds=2` human-requested revisions.

`revise` always increments `attempt`; it increments `retries_used` only for automatic (test-failure-driven) revisions. Each revise prompt includes summaries of the last 3 failed attempts. `fault` attribution (code vs tests) guards against loops caused by wrong LLM-written tests; `revise` may fix tests unless `provided_tests` is set.

## 4. RAG layer

- **Ingestion**: (a) *introspection loader*: for each configured library walk the public namespace (`__all__` or non-underscore names) and emit one chunk per function/class/method: `import_path + signature + docstring (truncated)`; class chunks include a method index. (b) *text loader*: .md/.rst/.txt split by headings then paragraphs.
- **Chunk size**: MiniLM truncates at ~256 word-pieces, so introspection chunks ≤ 1000 chars (signature first), text chunks ≤ 800 chars with 100 overlap.
- **Metadata**: library, version, qualname, kind, import_path, source_type. IDs are deterministic hashes → re-ingest is idempotent.
- **Retrieval**: embed query → Chroma cosine top-k → drop below `RETRIEVAL_MIN_SCORE` → dedupe → merge across queries by max score. `retrieved_docs` is merged with prior docs, capped at `RETRIEVAL_MAX_DOCS`, newest first.
- **Query formation**: first pass = raw task text, or (opt-in) an LLM rewrite into 2–4 API-oriented queries. Error pass = `error_analysis.retrieval_queries` built deterministically from suspect symbols (e.g. "pandas DataFrame groupby agg").
- **Prompt grounding**: docs are injected as `[doc:<id>] import_path signature + summary` blocks, capped at `CONTEXT_MAX_CHARS` to keep prompts small; the system prompt tells the model to prefer documented APIs and not invent others.

## 5. LLM access layer (`llm.py`)

`get_chat_model(settings)` builds `ChatOpenAI(model=LLM_MODEL, base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY, temperature, max_tokens, timeout, max_retries=0)` — SDK retries are disabled because `LLMClient` owns retry policy. `settings.require_api_key()` raises `ConfigError` with setup instructions if the key is missing.

`LLMClient.invoke_text(messages, purpose=...) -> LLMResponse(text, usage, model, waited_s)`:
- **Throttle**: ensure at least `LLM_MIN_INTERVAL_S` between calls (free tiers limit requests per minute).
- **Budget**: refuse the call (`LLMBudgetExceeded`) once `MAX_LLM_CALLS_PER_RUN` is reached; `reset_budget()` per run.
- **Retry with exponential backoff + jitter** (`LLM_BACKOFF_BASE_S` × 2^n, capped at `LLM_BACKOFF_MAX_S`) for: HTTP 429, 408, 5xx, connection errors, timeouts, and empty/None responses (free providers sometimes return HTTP 200 with an empty or malformed body).
- **Rate-limit awareness**: on 429 honor `Retry-After` / `X-RateLimit-Reset` when present (waiting at least that long); if the required wait exceeds `LLM_MAX_WAIT_S`, or the error text indicates a *daily* limit, raise `DailyQuotaExceeded` immediately instead of burning retries.
- **No retry**: 401/403 → `LLMAuthError`; 404 / model unavailable → `LLMModelError` (hint to check `LLM_MODEL`); other 4xx → `LLMRequestError`. Exhausted retries → `LLMUnavailable`.
- **Usage**: read `usage_metadata`; if absent, estimate from character counts and mark `estimated_calls`.
- **Post-processing**: normalize content (string or list of blocks) and strip `<think>…</think>`.
- Sleep/clock/RNG are injectable so tests run instantly.

All errors derive from `LLMError`. Nodes talk only to `TextLLM`; nothing else imports `ChatOpenAI`.

## 6. Sandbox

`run_in_sandbox(code, tests, timeout_s, mem_mb) -> RunResult`:
1. Pre-flight (no process): `ast.parse` (→ `syntax_error`); `importlib.util.find_spec` on top-level imports (→ `import_error`).
2. Write `solution.py` and `test_solution.py` into a fresh temp dir.
3. Run `[sys.executable, "-E", "-s", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--junitxml=report.xml", "test_solution.py"]` with `cwd=tmpdir`, `start_new_session=True`, scrubbed environment (no `OPENROUTER_*`, `*_API_KEY`, `*_TOKEN`, `HF_*`, `LANGSMITH_*`; `HOME=tmpdir`), and on POSIX `resource.setrlimit` for CPU time, address space, file size.
4. Wall-clock timeout → kill the process group → `timeout`.
5. Parse junit XML and traceback tails into structured `RunFailure`s; truncate stdout/stderr; classify (`ImportError`/`ModuleNotFoundError` → `import_error`; `AssertionError` → `assertion_failure`; collection errors map by exception type).
6. Always clean up the temp dir.

**Threat model**: protects against accidental damage and runaway code from LLM output. It is **not** a security boundary against a determined adversary (network not blocked; filesystem not jailed). Windows lacks `resource`; rlimits are skipped there with a logged warning. A Docker backend is the documented upgrade path.

## 7. Human-in-the-loop

`human_review` calls `langgraph.types.interrupt(payload)` with `{task, code, tests, run_summary, attempt, retrieved_doc_ids, human_round}`. Execution pauses at the checkpoint (needs a checkpointer and `thread_id`). Resume with `Command(resume={"action": "approve"|"reject"|"edit", "feedback"?: str, "code"?: str})`. On resume LangGraph re-executes the node from its start, so everything before `interrupt()` must be side-effect free. Streaming surfaces the pause as an `__interrupt__` update.

## 8. Observability

- Structured JSONL per run: `logs/<run_id>/events.jsonl`; one event per node execution: `run_id, thread_id, task_id, node, attempt, ts, duration_ms`, retrieved doc ids + scores, run_result summary, decision, LLM usage for the node (tokens, calls, `api_retries`, `wait_s`, estimated flag) and cumulative totals; plus `summary.json` at the end.
- Optional LangSmith tracing (off by default; needs its own key, enabled only when `LANGSMITH_TRACING=true` and a key exists). Runs are tagged with `run_id`/`task_id`.
- Rich console rendering for the CLI. API keys are redacted from all logs.

## 9. Testing strategy

- Unit tests: `FakeEmbedder` (deterministic hashing), `FakeChatModel` (scripted outputs and scripted openai-style exceptions, for `LLMClient`), `ScriptedLLM` (implements `TextLLM`, scripted tagged responses with usage), real sandbox on tiny snippets, `MemorySaver` for graph tests. Backoff tests inject fake sleep/clock.
- Markers: `live` (real OpenRouter calls; needs `OPENROUTER_API_KEY`; consumes free quota) and `slow` (downloads the real embedding model); both deselected by default.
- Eval harness (`eval/`): fixed tasks with reference solutions and acceptance tests; compares RAG on vs off; resumable and quota-aware.

## 10. Configuration (env or `.env`; real env vars win over `.env`; blank = unset)

`OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, `LLM_MODEL`, `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`, `LLM_TIMEOUT_S`, `LLM_MAX_API_RETRIES`, `LLM_BACKOFF_BASE_S`, `LLM_BACKOFF_MAX_S`, `LLM_MAX_WAIT_S`, `LLM_MIN_INTERVAL_S`, `MAX_LLM_CALLS_PER_RUN`, `EMBEDDING_MODEL`, `CHROMA_DIR`, `COLLECTION_NAME`, `DOC_LIBRARIES`, `DOCS_DIR`, `CHECKPOINT_DB`, `LOG_DIR`, `MAX_RETRIES`, `MAX_HUMAN_ROUNDS`, `SANDBOX_TIMEOUT_S`, `SANDBOX_MEM_MB`, `RETRIEVAL_TOP_K`, `RETRIEVAL_MAX_DOCS`, `RETRIEVAL_MIN_SCORE`, `CONTEXT_MAX_CHARS`, `ANALYZE_WITH_LLM`, `QUERY_REWRITE_WITH_LLM`, `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`.

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Free-tier per-minute/daily limits | Throttle, backoff, `Retry-After`, `DailyQuotaExceeded` fail-fast, per-run call budget, optional LLM calls off by default, small prompts, resumable eval |
| Free model slug removed/renamed | `LLM_MODEL` env var; `LLMModelError` with a clear hint; verify slug against `https://openrouter.ai/api/v1/models` |
| Free models ignore the output format | Tagged-text protocol with tolerant parser, fenced-block fallback, one format re-ask, then graceful fail |
| Reasoning models burn tokens on `think` | `LLM_MAX_TOKENS` cap, `think` stripping |
| Missing usage metadata from provider | Estimate from char counts, flagged `estimated_calls` |
| LLM writes wrong tests, loop chases a phantom bug | `fault` attribution; `provided_tests` mode; human review |
| Irrelevant retrieval | min-score threshold, metadata, `search` command, eval RAG on/off |
| Generated code harms host | Subprocess + rlimits + env scrubbing; documented non-goal of adversarial safety |
| Key leakage | Env-only key, `SecretStr`, redaction, scrubbed sandbox env, `.env` git-ignored, secret-scan test |
| LangGraph/LangChain API drift | Verify `interrupt`/`Command`/`ChatOpenAI` signatures against installed versions first |
| Large torch install | Optional CPU-only wheel install; embedding model download is opt-in `slow` in tests |

## 12. Future work

Hybrid BM25 + dense retrieval (RRF), Docker sandbox backend, multi-file output, streaming token display, web UI.
