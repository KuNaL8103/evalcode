# CLAUDE.md — evalcode

## Project overview
evalcode is an iterative **Code Generation Agent** built with **RAG + LangGraph**: it retrieves Python API docs (local MiniLM embeddings + Chroma), generates code and tests with an **OpenRouter free model**, runs them in a subprocess sandbox, analyzes failures, and revises in a bounded loop, with a human-approval interrupt, structured logging, optional LangSmith tracing, and a Typer CLI.

- Repo: https://github.com/KuNaL8103/evalcode.git (branch `main`)
- Design of record: `docs/ARCHITECTURE.md`. Task plan: `docs/PLAN.md`.
- Python 3.11+. Package: `src/evalcode`.
- **LLM constraint**: OpenRouter free models only, via `langchain_openai.ChatOpenAI(base_url="https://openrouter.ai/api/v1")`. Key only from env `OPENROUTER_API_KEY`; model from env `LLM_MODEL` (default `qwen/qwen3.8-27b:free`). No other providers, no hardcoded keys.
- Dev environment: Windows, Python 3.13.3, venv at .venv (use `.venv/Scripts/python`). Bash-style commands in this file need Windows equivalents (e.g. `copy .env.example .env`, `.venv\Scripts\activate`). Guard POSIX-only APIs (`os.killpg`, `preexec_fn`, `resource`) behind `sys.platform` checks.

## Directory layout (target; `(Task N)` = task that creates it)
```
evalcode/
├── CLAUDE.md  docs/ARCHITECTURE.md  docs/PLAN.md          (Task 0)
├── pyproject.toml  .gitignore  .env.example               (Task 0)
├── src/evalcode/
│   ├── __init__.py                                        (Task 0)
│   ├── config.py  errors.py                               (Task 1)
│   ├── rag/types.py  chunking.py  loaders.py              (Task 2)
│   ├── rag/embeddings.py  store.py  retriever.py  ingest.py   (Task 3)
│   ├── state.py  llm.py                                   (Task 4)
│   ├── prompts.py  parsing.py  schemas.py  nodes/generate.py  (Task 5)
│   ├── sandbox/runner.py  sandbox/errors.py  nodes/run_tests.py  (Task 6)
│   ├── nodes/analyze_error.py  nodes/revise.py            (Task 7)
│   ├── graph.py  nodes/terminal.py                        (Task 8)
│   ├── nodes/human_review.py  persistence.py              (Task 9)
│   ├── nodes/retrieve.py                                  (Task 10)
│   ├── observability.py                                   (Task 11)
│   └── cli.py                                             (Task 12)
├── tests/{unit,integration}/  tests/fakes.py              (fakes: Task 4 FakeChatModel, Task 5 ScriptedLLM)
├── eval/{tasks/*.yaml,run_eval.py,results/}  examples/    (Task 13)
├── .github/workflows/ci.yml  README.md  LICENSE           (Task 14)
└── data/ logs/                                            (gitignored runtime dirs)
```

## Commands
```
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
# optional, saves GBs (Linux/Windows): pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,corpus]"
cp .env.example .env     # then set OPENROUTER_API_KEY in .env (never commit .env)

pytest -q                # unit tests (excludes `live` and `slow`)
pytest -m slow -q        # real embedding model (downloads MiniLM weights once)
pytest -m live -q        # real OpenRouter calls (needs OPENROUTER_API_KEY; uses free quota — run sparingly)
ruff check . && ruff format --check .

python -m evalcode.rag.ingest --help   # ingestion (CLI wrapper arrives in Task 12)
evalcode run "write a function that ..."   # from Task 12
```

On Windows: `.venv\Scripts\activate`, `copy .env.example .env`, and use `.venv/Scripts/python -m pytest` if the venv isn't activated.

## Conventions
- **One task per session.** Read `docs/PLAN.md` for the current task, do only that task, never start the next one.
- **Tests are first-class**: every task ships tests; `pytest -q` and `ruff check .` must be green before committing. Unit tests must never hit the network or a real LLM — use `tests/fakes.py` (`FakeChatModel`, `ScriptedLLM`) and `FakeEmbedder`.
- **LLM access only through `evalcode.llm.LLMClient` / the `TextLLM` protocol.** Nodes must never import or call `ChatOpenAI` directly. Backoff, throttling, and the per-run call budget live there.
- **Protect the free quota**: don't add LLM calls casually; optional LLM calls (`ANALYZE_WITH_LLM`, `QUERY_REWRITE_WITH_LLM`) stay off by default; `live` tests must use the fewest calls possible; never loop on LLM calls without a budget.
- **Secrets**: the key comes only from the environment (`.env` via python-dotenv). Never hardcode, print, log, or commit it. Use `SecretStr`; redact in logs; the sandbox env is scrubbed. `.env` must stay git-ignored.
- **Secret-scan hygiene test** (added in Task 1) skips the tests/ directory and treats obvious placeholders (your-…, <…>, ...) as blank; fake keys in tests should still be built at runtime (e.g. `'sk-or-v1-' + 'FAKE' * 6`) rather than written as literals.
- **Typing**: full type hints; `from __future__ import annotations`; pydantic v2 for validated models; plain `TypedDict` for LangGraph state values (checkpoint-safe).
- **Nodes** are pure-ish functions `(state) -> partial state dict`; dependencies injected via factories (`make_*_node`). Never mutate the incoming state. LLM failures in mandatory nodes become `status="failed"` + `failure_reason`, not crashes.
- **No side effects before `interrupt()`** (the node re-runs on resume).
- **Config** only through `evalcode.config` (`get_settings()`); no hard-coded model names, URLs, paths, or limits outside config defaults.
- **Logging** via the `logging` module / `RunLogger`; no stray `print` outside the CLI.
- **Dependencies**: only those in `docs/ARCHITECTURE.md` unless the task says otherwise; ask before adding more.
- **LangGraph/LangChain/OpenAI SDK APIs change**: check the installed version's docs/signatures before relying on memory.
- **Git**: Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`, `refactor:`), one commit per task, push to `origin main`. Never commit `.env`, `data/`, `logs/`, `.venv/`.
- **Session end checklist**: tests + lint green → commit → push → update "Current status" below with: task completed, exact passing test count (and count of deselected live/slow tests), key files/APIs added, what is NOT started.

## Current status
Task 0 complete (2026-10-03): repo initialized on `main` with remote, docs + tooling scaffolded.
- `pytest -q` → 2 passed (0 live/slow tests exist yet; both markers configured and deselected by default).
- Added: `docs/ARCHITECTURE.md`, `docs/PLAN.md`, `pyproject.toml` (hatchling, src layout, extras `dev`/`corpus`; ruff `line-length 100`, select `E,F,I,UP,B`, `docs/` excluded from formatting to keep the architecture doc's code blocks verbatim), `.gitignore` (`.env` ignored, `.env.example` negated), `.env.example`, `src/evalcode` package (empty `rag/`, `nodes/`, `sandbox/` subpackages), `tests/{unit,integration}/` + `conftest.py`, 2 unit tests.
- Installed in `.venv`: langgraph 1.2.12, langchain-openai 1.6.7, langchain-huggingface 1.2.2, sentence-transformers 6.1.0, chromadb 1.5.9, openai 3.24.0, torch 2.14.1+cpu (CPU-only wheel; see Commands).
- NOT started: everything from Task 1 onward (no `config.py`, `errors.py`, or node code yet; `config.py` intentionally left out of this task).
- Housekeeping commit (2026-10-03): fixed PLAN.md Task 8 aside + 3 task-entry additions, replaced ARCHITECTURE §6's one-line Windows note with full Windows sandbox guidance (`CREATE_NEW_PROCESS_GROUP`, `taskkill /F /T`, env/cleanup/skip details) and annotated sandbox steps 3–4, recorded the Windows dev environment in CLAUDE.md; tag-integrity grep confirmed all `<...>` protocol tags intact. No source or test changes.
