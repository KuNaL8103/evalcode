"""FastAPI service tests: health, /generate, validation, and error mapping."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import GenerateRequest, create_app
from evalcode.core.config import Settings
from evalcode.core.errors import DailyQuotaExceeded


def _settings(tmp_path: Path, **overrides) -> Settings:
    base = {
        "chroma_dir": str(tmp_path / "chroma"),
        "sandbox_backend": "subprocess",
    }
    base.update(overrides)
    return Settings(**base)


def _state() -> dict:
    return {
        "task_id": "task-123",
        "status": "approved",
        "final_code": "def add(a, b):\n    return a + b\n",
        "code": "def add(a, b):\n    return a + b\n",
        "tests": "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        "explanation": "added two numbers",
        "attempt": 1,
        "retries_used": 0,
        "retrieval_queries": ["add two numbers"],
        "retrieved_docs": [{"id": "doc-1", "text": "docs", "score": 0.5}],
        "failure_reason": None,
    }


def test_root_redirects_to_swagger_ui(tmp_path: Path) -> None:
    client = TestClient(create_app(settings=_settings(tmp_path)))

    response = client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_health_is_cheap_and_redacts_nothing(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    client = TestClient(create_app(settings=settings))

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"]
    assert body["llm_configured"] is False
    assert body["rag_index_present"] is False
    assert body["sandbox_backend"] == "subprocess"
    assert "GEMINI" not in response.text


def test_health_reports_present_index_and_docker(tmp_path: Path) -> None:
    chroma = tmp_path / "chroma"
    chroma.mkdir()
    settings = _settings(tmp_path, chroma_dir=str(chroma), sandbox_backend="docker")
    client = TestClient(create_app(settings=settings))

    body = client.get("/health").json()
    assert body["rag_index_present"] is True
    assert body["sandbox_backend"] == "docker"


def test_generate_runs_injected_workflow(tmp_path: Path) -> None:
    seen: list[tuple[GenerateRequest, Settings]] = []

    def fake_workflow(payload: GenerateRequest, settings: Settings) -> dict:
        seen.append((payload, settings))
        return _state()

    client = TestClient(create_app(run_workflow=fake_workflow, settings=_settings(tmp_path)))
    response = client.post(
        "/generate",
        json={"task": "add two numbers", "use_rag": False, "provided_tests": "fixed"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["task_id"] == "task-123"
    assert body["status"] == "approved"
    assert body["final_code"].startswith("def add")
    assert body["retrieved_doc_ids"] == ["doc-1"]
    assert body["attempts"] == 1
    assert body["duration_s"] >= 0

    assert len(seen) == 1
    assert seen[0][0].task == "add two numbers"
    assert seen[0][0].use_rag is False
    assert seen[0][0].provided_tests == "fixed"


def test_generate_rejects_empty_task(tmp_path: Path) -> None:
    def fake_workflow(payload: GenerateRequest, settings: Settings) -> dict:  # pragma: no cover
        return _state()

    client = TestClient(create_app(run_workflow=fake_workflow, settings=_settings(tmp_path)))
    response = client.post("/generate", json={"task": ""})
    assert response.status_code == 422


def test_generate_maps_quota_error_to_429(tmp_path: Path) -> None:
    def exhausted(payload: GenerateRequest, settings: Settings) -> dict:
        raise DailyQuotaExceeded()

    client = TestClient(create_app(run_workflow=exhausted, settings=_settings(tmp_path)))
    response = client.post("/generate", json={"task": "x"})
    assert response.status_code == 429
    assert "quota" in response.json()["detail"].lower()


def test_generate_hides_unexpected_exception_details(tmp_path: Path) -> None:
    def boom(payload: GenerateRequest, settings: Settings) -> dict:
        raise RuntimeError("secret internal detail")

    client = TestClient(create_app(run_workflow=boom, settings=_settings(tmp_path)))
    response = client.post("/generate", json={"task": "x"})
    assert response.status_code == 500
    assert response.json()["detail"] == "workflow failed"
    assert "secret internal detail" not in response.text


def test_default_workflow_requires_api_key(tmp_path: Path) -> None:
    client = TestClient(create_app(settings=_settings(tmp_path)))
    response = client.post("/generate", json={"task": "x"})
    assert response.status_code == 503
    assert "GEMINI_API_KEY" in response.json()["detail"]
