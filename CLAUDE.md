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
- **Quota- and CPU-friendly tooling**: every tool call costs free-model quota. Run pytest in the foreground as ONE process (timeout up to 10 minutes) — never in the background, never two heavy python/pytest processes at once, never poll with sleep loops. The default (non-slow) suite must stay fast (target < 60 s) because default tests never import torch, sentence-transformers, or langchain-huggingface; real heavy imports live only in `slow` tests. If any command runs longer than ~3 minutes, stop and ask instead of waiting. Never kill processes you didn't start — ask me.
- **Lazy heavy imports**: never import torch, sentence_transformers, transformers, or langchain_huggingface at module top level; import them inside the function/method that needs them (and resolve the class through a tiny factory function that tests can monkeypatch) so unit tests using fakes stay fast. Import chromadb inside `VectorStore.__init__` for the same reason.
- **Tests are first-class**: every task ships tests; `pytest -q` and `ruff check .` must be green before committing. Unit tests must never hit the network or a real LLM — use `tests/fakes.py` (`FakeChatModel`, `ScriptedLLM`) and `FakeEmbedder`.
- **LLM access only through `evalcode.llm.LLMClient` / the `TextLLM` protocol.** Nodes must never import or call `ChatOpenAI` directly. Backoff, throttling, and the per-run call budget live there.
- **Protect the free quota**: don't add LLM calls casually; optional LLM calls (`ANALYZE_WITH_LLM`, `QUERY_REWRITE_WITH_LLM`) stay off by default; `live` tests must use the fewest calls possible; never loop on LLM calls without a budget.
- **Secrets**: the key comes only from the environment (`.env` via python-dotenv). Never hardcode, print, log, or commit it. Use `SecretStr`; redact in logs; the sandbox env is scrubbed. `.env` must stay git-ignored.
- **Secret-scan hygiene test** (added in Task 1) skips the tests/ directory and treats obvious placeholders (your-…, <…>, ...) as blank; fake keys in tests should still be built at runtime (e.g. `'sk-or-v1-' + 'FAKE' * 6`) rather than written as literals.
- **Windows text I/O**: always pass `encoding='utf-8'` to `open()`/`read_text()`/`write_text()`; read files CRLF-safely (`splitlines()`); don't print non-ASCII to the console without handling cp1252 (Rich output in Task 12 must cope).
- **Test env isolation**: unit tests never read the real shell environment or the developer's `.env`; the autouse fixture in `tests/conftest.py` enforces this. New tests that need env vars must set them with `monkeypatch`.
- **Tag-literal safety**: in this tooling a literal think-tag typed inside a tool call can be silently stripped. In source and tests NEVER type the think open/close tags as literals; build them from parts (e.g. `THINK_OPEN = "<" + "think" + ">"`, `THINK_CLOSE = "</" + "think" + ">"`) and build regexes from those constants. After writing such files, verify with a short Python snippet that the constants contain the real characters and that `strip_reasoning` actually removes a block. Likewise, don't put a think tag in an edit tool's old_str/new_str — anchor on other unique text or use a small Python script.
- **Typing**: full type hints; `from __future__ import annotations`; pydantic v2 for validated models; plain `TypedDict` for LangGraph state values (checkpoint-safe).
- **Nodes** are pure-ish functions `(state) -> partial state dict`; dependencies injected via factories (`make_*_node`). Never mutate the incoming state. LLM failures in mandatory nodes become `status="failed"` + `failure_reason`, not crashes.
- **No side effects before `interrupt()`** (the node re-runs on resume).
- **Config** only through `evalcode.config` (`get_settings()`); no hard-coded model names, URLs, paths, or limits outside config defaults.
- **Logging** via the `logging` module / `RunLogger`; no stray `print` outside the CLI.
- **print in `python -m` wrappers**: `rag/ingest.py` and `rag/retriever.py` `main()` and `scripts/*.py` may print; all other library modules must not (Task 14 cleanup must allow these exceptions).
- **Re-ingest rule**: changing loaders, `embed_text`, or the embedding model requires `python -m evalcode.rag.ingest --reset` (the stored vectors embed the old text otherwise).
- **Dependencies**: only those in `docs/ARCHITECTURE.md` unless the task says otherwise; ask before adding more.
- **LangGraph/LangChain/OpenAI SDK APIs change**: check the installed version's docs/signatures before relying on memory.
- **Git**: Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`, `refactor:`), one commit per task, push to `origin main`. Never commit `.env`, `data/`, `logs/`, `.venv/`.
- **Session end checklist**: tests + lint green → commit → push → update "Current status" below with: task completed, exact passing test count (and count of deselected live/slow tests), key files/APIs added, what is NOT started. Keep 'Current status' compact: one line per completed task plus a block of at most 12 lines for the latest milestone.

## Current status

### Completed tasks
- Task 0 (e2c588e): repo init & scaffold — docs, pyproject, `.env.example`, `.gitignore`, package + tests layout; 2 unit tests.
- Docs housekeeping (1c382ae): PLAN.md fixes, full Windows sandbox guidance in ARCHITECTURE §6, dev-environment note; no source/test changes.
- Task 1 (bdbba70): `.env` + python-dotenv setup and the config module (`config.py`, `errors.py`, full `.env.example`, 8 config tests).
- Housekeeping (27782b7): autouse env-isolation fixture in `tests/conftest.py`, test-name fix, compact status format, tag-literal-safety conventions.
- Task 2 (97c1f2b): doc loaders & chunking — `rag/types.py`, `rag/chunking.py`, `rag/loaders.py` (`DocChunk`, `stable_id`, `truncate_text`, `split_markdown`, `iter_introspection_chunks`, `iter_text_file_chunks`); 12 new tests.
- Task 2 fix-up (b951075): skip non-callable constants, root-relative text-file paths, BOM-safe (`utf-8-sig`) decoding, fence-aware headings; test count unchanged (22).
- Task 2 fix-up 2 (6fe6d3f): index C-implemented class methods (method/builtin/classmethod descriptors); default test suite no longer imports heavy packages (install check + `find_spec`); test count unchanged (22).
- Task 3 pre-work (8f4a18d): `_public_methods` walks `dir(cls)`/MRO, so methods inherited from Python base classes are indexed (e.g. `pathlib.Path.with_suffix`), while members inherited from builtins bases (`BaseException.add_note`) stay skipped; json still yields exactly 12 chunks; test count unchanged (22).
- Task 3 (f03e7ea): embeddings, Chroma store, retriever, ingest (`rag/embeddings,store,retriever,ingest` + `RetrievedDoc`); 10 new tests + 1 slow.
- Task 3b (8a09984): retrieval-quality tuning — `_clean_signature`, `DocChunk.embed_text` (compact embedding input), probe script; test count unchanged (32).
- Task 4 (4c54c9e): state schema & OpenRouter LLM client — `state.py`, `llm.py` (backoff, rate-limit, budget), LLM error hierarchy, `tests/fakes.py`; 14 new tests (46).
- Task 4 fix-up (39e2543): per-call API retry limit, close-only reasoning strip, malformed-response retry; test count unchanged (46).
- Task 5 fix-up (ca12cdc): tolerant missing-closing-tag parsing (next-tag termination), reply_head / parse_reason diagnostics in generate node and live test; live: slug 404, diagnostics code verified; 56 passed.
- Task 5 (ca12cdc): prompts, tagged-text parser, and generate node — `prompts.py`, `parsing.py`, `schemas.py`, `nodes/generate.py`, `ScriptedLLM`; 10 new unit tests + 1 live (56).

### Latest milestone (Task 5 — prompts, response parser, generate node)
- `prompts.py`: `GENERATE_SYSTEM` (strict tagged-format rules, compact for free-tier quotas), `FORMAT_REMINDER` (strict re-ask), `build_generate_messages` (task + capped doc context + PROVIDED TESTS instruction), `format_context` (whole-block truncation that accounts for the join separator, total ≤ `context_max_chars`).
- `parsing.py`: `parse_bundle` (case-insensitive tags, whitespace, fences inside tags, missing closing tag at end, surrounding prose, defensive re-`strip_reasoning`; fenced-block fallback when `<code>` is absent; `ParseError` only when code — or required tests — is truly unrecoverable) + reusable `parse_tagged`.
- `schemas.py`: pydantic `CodeBundle {explanation, code, tests, docs_used}`, `extract_imports` (ast top-level module names, first-seen order, syntax errors → []).
- `nodes/generate.py`: `make_generate_node(llm, settings)` — one re-ask on `ParseError`; any `LLMError` on a mandatory call → `status="failed"` + actionable `failure_reason` (never a crash; failed updates carry only successful-call usage); history `StepEvent` summary `{code_chars, tests_chars, doc_ids, docs_used, imports, reasks}`; input state never mutated.
- `tests/fakes.py`: `ScriptedLLM` (`TextLLM` fake: scripted texts/exceptions, records `.calls`/`.purposes`, fake usage) + `bundle_text` (renders the tagged format).
- Housekeeping: 403 now says access denied / key-lacks-access / model-restricted-to-specific-clients (points at `LLM_MODEL`, includes first 160 body chars); 401 keeps the key message; 403 assertion adjusted inside the existing `test_auth_errors_fail_fast`.
- `pytest -q` → 56 passed (46 + 10 new), 2 deselected (1 live, 1 slow); ruff check + format clean; tag-literal + round-trip snippet verified.
- Live (fix-up, 1 run): `qwen/qwen3.8-27b:free` now returns HTTP 404 from OpenRouter ("unavailable for free" — slug removed/renamed upstream, paid version suggested); no reply received, so first reply parse is N/A, reply_head diagnostic did not trigger (call blocked at `LLMModelError`), tokens 0. The diagnostics code (`reply_head` + `parse_reason` in summary, print before re-ask) is verified by unit tests; the slug needs a replacement (check `https://openrouter.ai/api/v1/models`).
- Not started: Task 6 onward (sandbox & run_tests).

### Not started
Task 6 onward.
