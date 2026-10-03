# evalcode — Implementation Plan

Each task = one Claude Code session, ending with: tests written and passing, `ruff` clean, conventional commit, push to `origin main`, and CLAUDE.md "Current status" updated. Do tasks in order; don't skip.

LLM: OpenRouter free models only (default `qwen/qwen3.8-27b:free`, swap via `LLM_MODEL`). Embeddings: local `sentence-transformers/all-MiniLM-L6-v2`. Vector store: Chroma. Free-tier quota is a design constraint: backoff, throttling, call budgets, and small prompts are built in.

## Overview

| # | Task | Key deliverables | New unit tests (cumulative) | Commit |
|---|---|---|---|---|
| 0 | Repo init & scaffold | docs, pyproject, `.env.example`, `.gitignore` covering `.env` | 2 (2) | `chore: initialize project scaffold, docs, and tooling` |
| 1 | Env & config | python-dotenv loading, `Settings`, `ConfigError`, secret hygiene tests | 8 (10) | `feat(config): add dotenv-based settings for OpenRouter and RAG` |
| 2 | Doc loaders & chunking | `rag/types,chunking,loaders` | 12 (22) | `feat(rag): add documentation loaders and chunking` |
| 3 | Embeddings, Chroma, retriever, ingest | `rag/embeddings,store,retriever,ingest` | 10 (32) | `feat(rag): add HuggingFace embeddings, Chroma vector store, and retriever` |
| 4 | State schema & OpenRouter LLM client | `state.py`, `llm.py` (backoff, rate-limit, budget), `FakeChatModel` | 14 (46) | `feat(llm): add state schema and OpenRouter client with backoff and call budget` |
| 5 | Prompts, parser, generate node | `prompts,parsing,schemas,nodes/generate`, `ScriptedLLM` | 10 (56) | `feat(agent): add prompts, response parser, and generate node` |
| 6 | Sandbox & run_tests | `sandbox/*`, `nodes/run_tests` | 16 (72) | `feat(sandbox): add subprocess sandbox and run_tests node with structured error capture` |
| 7 | Error analysis & revise | `nodes/analyze_error,revise` | 10 (82) | `feat(agent): add analyze_error and revise nodes` |
| 8 | Graph wiring & retry edges | `graph.py`, `nodes/terminal` | 10 (92) | `feat(graph): wire LangGraph state machine with conditional retry edges` |
| 9 | Human-in-the-loop | `nodes/human_review`, `persistence.py` | 8 (100) | `feat(hitl): add human_review interrupt with approve/reject/edit` |
| 10 | RAG integration | `nodes/retrieve`, graph edges, prompt grounding | 10 (110) | `feat(rag): integrate retrieval into the graph with error-driven re-retrieval` |
| 11 | Observability | `observability.py`, LangSmith | 8 (118) | `feat(obs): add structured run logging, usage accounting, and LangSmith tracing` |
| 12 | CLI | `cli.py` (ingest/search/run/resume) | 9 (127) | `feat(cli): add Typer CLI with live loop display and resume` |
| 13 | Eval harness & demo tasks | `eval/` (resumable, quota-aware) | 6 (133) | `feat(eval): add demo tasks and quota-aware RAG on/off evaluation harness` |
| 14 | README, CI, polish | README, CI, LICENSE, cleanup | 0 (133) | `docs: add README, CI, and final polish` |

Opt-in tests (deselected by default): 1 `slow` (Task 3), `live` tests in Tasks 5, 8, 10 (the last also `slow`) → 4 total.

## Tasks

**Task 0 — Repo init & scaffold.** Goal: working repo with docs and tooling. Deliver: git repo with remote, three docs, `pyproject.toml` (hatchling, src layout, extras `dev`/`corpus`, ruff/pytest config, markers `live`/`slow`), `.env.example` (`OPENROUTER_API_KEY=`, `LLM_MODEL=` blank placeholders + comments), `.gitignore` covering `.env`, empty package dirs. Accept: install works, `pytest -q` 2 passed, `git check-ignore .env` succeeds, pushed.

**Task 1 — Env & config.** Goal: safe, typed configuration. Deliver: `config.py` (python-dotenv + pydantic-settings; blank env values = unset; real env beats `.env`), all settings from ARCHITECTURE §10, `require_api_key()`, `ConfigError`, secret-hygiene tests. Accept: 8 new tests; key never appears in repr/logs; `.env.example` matches `Settings`. The secret-scan hygiene test skips the tests/ directory and treats obvious placeholders (your-…, <…>, ...) as blank, so later fake keys in tests and README examples don't trip it.

**Task 2 — Doc loaders & chunking.** Goal: turn installed libraries and local text docs into `DocChunk`s sized for MiniLM. Deliver: `DocChunk`, `stable_id`, `split_markdown`, `iter_introspection_chunks`, `iter_text_file_chunks`. Accept: chunks for `json` look right; ids stable; limits respected; 12 new tests.

**Task 3 — Embeddings, Chroma, retriever, ingest.** Goal: searchable doc index with local embeddings. Deliver: `Embedder` protocol, `HFEmbedder` (langchain-huggingface), `FakeEmbedder`, `VectorStore`, `Retriever`, `ingest()` with `python -m` entry. Accept: ingest `json,re,collections`; "parse a JSON string" surfaces `json.loads`; 10 new tests + 1 slow.

**Task 4 — State schema & OpenRouter LLM client.** Goal: reliable free-tier LLM access. Deliver: full `AgentState` + nested TypedDicts + `merge_usage`; `get_chat_model`, `LLMClient`/`TextLLM` with throttle, exponential backoff, `Retry-After`, daily-quota detection, error taxonomy, call budget, usage extraction, `think` stripping; `FakeChatModel`. Accept: 14 new tests with injected fake sleep; real model slug verified against OpenRouter's public models list. Include a test using the real OpenRouter 429 text 'Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000 free model requests per day' → DailyQuotaExceeded with zero retries.

**Task 5 — Prompts, parser, generate node.** Goal: first LLM node. Deliver: tagged-text protocol, tolerant parser, `CodeBundle`, `format_context`, `make_generate_node` (graceful failure on `LLMError`/unparseable output), `ScriptedLLM`. Accept: 10 new tests + 1 live (≤2 calls).

**Task 6 — Sandbox & run_tests.** Goal: safe execution with structured errors. Deliver: `run_in_sandbox`, error parsing/classification, `run_tests_node`. Accept: syntax/import/assertion/runtime/timeout/no-tests classified; env (including OpenRouter key) scrubbed; 16 new tests. Windows-aware: on Windows kill the process tree with taskkill /T and skip rlimits (see ARCHITECTURE §6).

**Task 7 — Error analysis & revise.** Goal: learn from failures. Deliver: deterministic classifier + symbol extraction (+ optional LLM diagnosis, off by default); `revise` node with failure-aware prompt and budget bookkeeping. Accept: API-misuse errors yield `needs_docs` + queries; counters correct; 10 new tests.

**Task 8 — Graph wiring.** Goal: runnable loop. Deliver: `Dependencies`, `build_graph`, routers incl. `route_after_llm`, `finalize`/`fail`, `run_task`/`stream_task`. Accept: pass-first-try, fail-then-pass, exhausted-budget, quota-failure scenarios with fakes; 10 new tests + 1 live.

**Task 9 — Human-in-the-loop.** Goal: pause/approve/reject/edit. Deliver: `human_review` with `interrupt`, routing, SQLite checkpointer helper. Accept: pause/resume via `Command`, reject feedback reaches revise, persistence across rebuilt graphs; 8 new tests.

**Task 10 — RAG integration.** Goal: ground generation. Deliver: `retrieve` node (raw-task queries by default, opt-in LLM rewrite), graph edges, doc-grounded prompts, RAG-off mode. Accept: topologies correct; error-driven re-retrieval triggers; 10 new tests + 1 live/slow.

**Task 11 — Observability.** Goal: see every step. Deliver: `RunLogger` (JSONL + summary), `traced_node`, LangSmith config, run config builder. Accept: events contain attempt/docs/test results/usage incl. retries and wait time; 8 new tests.

**Task 12 — CLI.** Goal: use it. Deliver: Typer app `evalcode` with `ingest`, `search`, `run`, `resume`; live display; interactive review; exit codes (3 = free quota exhausted). Accept: CliRunner tests; 9 new tests.

**Task 13 — Eval harness & demo tasks.** Goal: measure whether RAG helps without blowing the free quota. Deliver: ≥10 YAML tasks with references, `evalcode eval` (resumable, stops cleanly on daily quota, `--limit`), comparison report. Accept: references pass; harness works offline with fakes; 6 new tests; live run in batches only with approval.

**Task 14 — README, CI, polish.** Goal: shippable repo. Deliver: README (incl. OpenRouter setup and free-tier guidance), CI workflow, LICENSE, doc reconciliation, cleanup, fresh-clone verification. Accept: fresh clone → setup → `pytest -q` green.
