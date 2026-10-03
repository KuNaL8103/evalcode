"""Task 1 config tests: defaults, env/.env precedence, CSV lists, secret hygiene."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from evalcode.config import Settings, get_settings, load_settings
from evalcode.errors import ConfigError

# Fake key built at runtime so the secret-scan test never sees a literal.
FAKE_KEY = "sk-or-v1-" + "FAKE" * 6

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_VAR_NAMES = [name.upper() for name in Settings.model_fields]


def _is_placeholder(value: str) -> bool:
    """Obvious placeholders (your-…, <…>, ...) count as blank in the scan."""
    v = value.strip().strip('"').strip("'")
    if not v:
        return True
    low = v.lower()
    if low.startswith(("your", "<")) or "…" in low or "..." in low:
        return True
    return len(set(low)) == 1  # e.g. "xxxxxxxx"


def _unset(vars: list[str]) -> None:
    """Undo load_dotenv's writes to os.environ (monkeypatch can't see them)."""
    for var in vars:
        os.environ.pop(var, None)


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts with no settings env vars and a cold settings cache."""
    for var in ENV_VAR_NAMES:
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_defaults() -> None:
    s = Settings()
    assert s.openrouter_base_url == "https://openrouter.ai/api/v1"
    assert s.llm_model == "qwen/qwen3.8-27b:free"
    assert s.openrouter_api_key is None
    assert s.langsmith_api_key is None
    assert s.llm_temperature == 0.2
    assert s.llm_max_tokens == 8192
    assert s.llm_min_interval_s == 3.0
    assert s.max_llm_calls_per_run == 10
    assert s.max_retries == 3
    assert s.max_human_rounds == 2
    assert s.sandbox_timeout_s == 10
    assert s.sandbox_mem_mb == 1024
    assert s.retrieval_top_k == 5
    assert s.retrieval_max_docs == 8
    assert s.retrieval_min_score == 0.2
    assert s.analyze_with_llm is False
    assert s.query_rewrite_with_llm is False
    assert s.langsmith_tracing is False
    assert s.langsmith_project == "evalcode"
    assert s.doc_libraries == [
        "json",
        "re",
        "datetime",
        "collections",
        "itertools",
        "pathlib",
        "dataclasses",
        "csv",
        "sqlite3",
        "argparse",
        "pandas",
        "numpy",
    ]


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "other/model:free")
    monkeypatch.setenv("MAX_RETRIES", "7")
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    s = Settings()
    assert s.llm_model == "other/model:free"
    assert s.max_retries == 7
    assert s.openrouter_api_key is not None
    assert s.openrouter_api_key.get_secret_value() == FAKE_KEY


def test_blank_env_values_fall_back_to_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    s = Settings()
    assert s.llm_model == "qwen/qwen3.8-27b:free"
    assert s.openrouter_api_key is None


def test_dotenv_file_loading_and_real_env_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(f"LLM_MODEL=file-model\nOPENROUTER_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    try:
        # .env values are picked up by load_settings(env_file=...)
        s = load_settings(env_file=env_file)
        assert s.llm_model == "file-model"
        assert s.openrouter_api_key is not None
        assert s.openrouter_api_key.get_secret_value() == FAKE_KEY
    finally:
        _unset(["LLM_MODEL", "OPENROUTER_API_KEY"])

    # A real environment variable beats the .env value
    monkeypatch.setenv("LLM_MODEL", "real-env-model")
    s2 = load_settings(env_file=env_file)
    assert s2.llm_model == "real-env-model"
    _unset(["OPENROUTER_API_KEY"])  # re-loaded from the file above


def test_doc_libraries_comma_separated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOC_LIBRARIES", "json, re ,datetime")
    assert Settings().doc_libraries == ["json", "re", "datetime"]

    monkeypatch.setenv("DOC_LIBRARIES", "pandas")
    assert Settings().doc_libraries == ["pandas"]

    # Direct construction still accepts a real list
    assert Settings(doc_libraries=["json"]).doc_libraries == ["json"]


def test_require_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ConfigError) as excinfo:
        Settings().require_api_key()
    msg = str(excinfo.value)
    assert "OPENROUTER_API_KEY" in msg
    assert "openrouter.ai" in msg
    assert FAKE_KEY not in msg  # the message never contains key material

    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    key = Settings().require_api_key()
    assert key.get_secret_value() == FAKE_KEY


def test_secrets_never_revealed(monkeypatch: pytest.MonkeyPatch) -> None:
    # Unset secrets are shown as "unset", never as a leak
    s0 = Settings()
    assert s0.safe_dump()["openrouter_api_key"] == "unset"
    assert s0.safe_dump()["langsmith_api_key"] == "unset"

    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    s = Settings()
    assert FAKE_KEY not in repr(s)
    assert FAKE_KEY not in str(s)
    dumped = s.model_dump()
    assert FAKE_KEY not in repr(dumped)
    # model_dump() keeps the SecretStr, which self-masks on str/repr
    assert str(dumped["openrouter_api_key"]) == "**********"

    safe = s.safe_dump()
    assert safe["openrouter_api_key"] == "***"
    assert FAKE_KEY not in repr(safe)


def test_env_example_and_secret_hygiene() -> None:
    example_lines = (REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()

    # Every Settings field has a .env.example line, and vice versa
    defined = {
        m.group(1) for line in example_lines if (m := re.match(r"^([A-Z][A-Z0-9_]*)=", line))
    }
    assert defined == {name.upper() for name in Settings.model_fields}
    # Secrets stay blank in the template
    assert "OPENROUTER_API_KEY=" in example_lines
    assert "LLM_MODEL=" in example_lines

    # .env must stay git-ignored
    try:
        proc = subprocess.run(
            ["git", "check-ignore", ".env"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except OSError:
        pytest.skip("git unavailable; cannot verify .env is git-ignored")
    assert proc.returncode == 0, f".env is not git-ignored (rc={proc.returncode})"

    # No real-looking key or non-blank key assignment in any git-tracked file
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git unavailable; cannot scan tracked files")
    key_re = re.compile(r"sk-or-v1-[A-Za-z0-9]{16,}")
    key_line_re = re.compile(r"^\s*OPENROUTER_API_KEY\s*=\s*(\S.*?)\s*$")
    for rel in (p for p in out.split("\0") if p):
        if rel.startswith("tests/") or rel.startswith("tests\\"):
            continue
        text = (REPO_ROOT / rel).read_text(encoding="utf-8", errors="replace")
        for match in key_re.finditer(text):
            assert _is_placeholder(match.group(0)), f"possible OpenRouter key in {rel}"
        for line in text.splitlines():  # CRLF-safe
            km = key_line_re.match(line)
            if km and not _is_placeholder(km.group(1)):
                pytest.fail(f"non-blank OPENROUTER_API_KEY= in {rel}: {line!r}")
