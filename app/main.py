"""FastAPI wrapper around the evalcode LangGraph workflow.

Run locally::

    uvicorn app.main:app --reload --port 8000

Endpoints
---------
``GET /health``   cheap liveness/config check (never calls Gemini).
``POST /generate`` runs the full retrieve -> generate -> test -> revise loop
                   with auto-approval and returns the final state.

The app is built with :func:`create_app` so tests (or another service) can
inject a workflow function without a Gemini key or a real vector index.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from evalcode import __version__
from evalcode.core.config import Settings, get_settings
from evalcode.core.errors import ConfigError, DailyQuotaExceeded, EvalcodeError, LLMError

__all__ = ["GenerateRequest", "GenerateResponse", "HealthResponse", "app", "create_app"]

logger = logging.getLogger(__name__)


class GenerateRequest(BaseModel):
    """Input for one code-generation run."""

    task: str = Field(
        ...,
        min_length=1,
        max_length=8_000,
        description="Natural-language coding task, e.g. 'parse a JSON string'.",
    )
    provided_tests: str | None = Field(
        default=None,
        max_length=100_000,
        description="Optional fixed pytest source; skips model-written tests.",
    )
    use_rag: bool = Field(
        default=True,
        description="Retrieve Python API docs before generating (hybrid BM25 + dense).",
    )


class GenerateResponse(BaseModel):
    """Final workflow state exposed over HTTP."""

    task_id: str
    status: Literal["running", "approved", "failed", "awaiting_review"]
    final_code: str | None = None
    code: str = ""
    tests: str = ""
    explanation: str = ""
    attempts: int = 0
    retries_used: int = 0
    retrieval_queries: list[str] = Field(default_factory=list)
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    duration_s: float = 0.0
    failure_reason: str | None = None


class HealthResponse(BaseModel):
    """Liveness and static configuration (safe to expose publicly)."""

    status: Literal["ok"] = "ok"
    version: str
    timestamp: str
    llm_configured: bool
    rag_index_present: bool
    sandbox_backend: Literal["subprocess", "docker"]


Workflow = Callable[[GenerateRequest, Settings], dict[str, Any]]


def _state_to_response(state: dict[str, Any], *, duration_s: float) -> GenerateResponse:
    docs = state.get("retrieved_docs") or []
    return GenerateResponse(
        task_id=str(state.get("task_id") or uuid.uuid4().hex),
        status=state.get("status", "running"),
        final_code=state.get("final_code"),
        code=state.get("code") or "",
        tests=state.get("tests") or "",
        explanation=state.get("explanation") or "",
        attempts=int(state.get("attempt") or 0),
        retries_used=int(state.get("retries_used") or 0),
        retrieval_queries=list(state.get("retrieval_queries") or []),
        retrieved_doc_ids=[str(doc.get("id", "")) for doc in docs if isinstance(doc, dict)],
        duration_s=round(duration_s, 3),
        failure_reason=state.get("failure_reason"),
    )


def _default_run_workflow(payload: GenerateRequest, settings: Settings) -> dict[str, Any]:
    """Build real dependencies and run the LangGraph to completion."""
    settings.require_api_key()

    # Imported lazily: importing the graph pulls LangGraph/LangChain, and the
    # health endpoint should stay fast and dependency-light.
    from evalcode.graph import close_dependencies, default_dependencies, run_task

    deps = default_dependencies(settings, rag=payload.use_rag)
    try:
        return run_task(
            payload.task,
            deps,
            provided_tests=payload.provided_tests,
            auto_approve=True,
        )
    finally:
        close_dependencies(deps)


def create_app(
    *,
    run_workflow: Workflow | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    """Create the FastAPI app with an injectable workflow for tests."""
    workflow = run_workflow or _default_run_workflow

    application = FastAPI(
        title="evalcode",
        version=__version__,
        description="Iterative RAG + LangGraph code-generation agent.",
    )

    def _settings() -> Settings:
        return settings if settings is not None else get_settings()

    @application.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        """Send browser visits straight to the interactive Swagger UI."""
        return RedirectResponse(url="/docs")

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        current = _settings()
        return HealthResponse(
            version=__version__,
            timestamp=datetime.now(UTC).isoformat(),
            llm_configured=current.gemini_api_key is not None
            and bool(current.gemini_api_key.get_secret_value().strip()),
            rag_index_present=Path(current.chroma_dir).is_dir(),
            sandbox_backend=current.sandbox_backend,
        )

    @application.post("/generate", response_model=GenerateResponse)
    def generate(payload: GenerateRequest) -> GenerateResponse:
        start = time.perf_counter()
        try:
            state = workflow(payload, _settings())
        except DailyQuotaExceeded as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except ConfigError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except LLMError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except EvalcodeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - never leak internals to clients
            logger.exception("generate workflow failed")
            raise HTTPException(status_code=500, detail="workflow failed") from exc
        return _state_to_response(state, duration_s=time.perf_counter() - start)

    return application


app = create_app()
