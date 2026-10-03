"""Configuration: python-dotenv + pydantic-settings.

Env loading is explicit (``load_settings``), not via pydantic-settings'
``env_file``: real environment variables always beat ``.env`` values
(``load_dotenv(override=False)``), and blank env values are treated as
unset (``env_ignore_empty=True``) so they fall back to defaults.

The OpenRouter API key comes only from the ``OPENROUTER_API_KEY``
environment variable (optionally loaded from a git-ignored ``.env``) and
is held as a ``SecretStr`` so it never appears in reprs, dumps, or logs.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from dotenv import find_dotenv, load_dotenv
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from evalcode.errors import ConfigError

__all__ = ["Settings", "get_settings", "load_settings"]

# Fields holding secrets; safe_dump() masks them.
_SECRET_FIELDS = ("openrouter_api_key", "langsmith_api_key")


class Settings(BaseSettings):
    """Typed configuration. Env names are the upper-case field names."""

    model_config = SettingsConfigDict(extra="ignore", env_ignore_empty=True)

    # --- OpenRouter / LLM ---
    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "qwen/qwen3.8-27b:free"
    llm_temperature: float = Field(default=0.2, ge=0)
    llm_max_tokens: int = Field(default=8192, gt=0)
    llm_timeout_s: float = Field(default=120, gt=0)
    llm_max_api_retries: int = Field(default=5, ge=0)
    llm_backoff_base_s: float = Field(default=2.0, gt=0)
    llm_backoff_max_s: float = Field(default=60.0, gt=0)
    llm_max_wait_s: float = Field(default=120.0, ge=0)
    llm_min_interval_s: float = Field(default=3.0, ge=0)
    max_llm_calls_per_run: int = Field(default=10, gt=0)

    # --- Embeddings & vector store ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_dir: str = "data/chroma"
    collection_name: str = "python_docs"
    # NoDecode: skip JSON decoding of the env value so the before-validator
    # can split the comma-separated string (verified against pydantic-settings 2.15).
    doc_libraries: Annotated[list[str], NoDecode] = [
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
    docs_dir: str = "data/docs"

    # --- Agent budgets / persistence ---
    checkpoint_db: str = "data/checkpoints.sqlite"
    log_dir: str = "logs"
    max_retries: int = Field(default=3, ge=0)
    max_human_rounds: int = Field(default=2, ge=0)

    # --- Sandbox ---
    sandbox_timeout_s: int = Field(default=10, gt=0)
    sandbox_mem_mb: int = Field(default=1024, gt=0)

    # --- Retrieval ---
    retrieval_top_k: int = Field(default=5, gt=0)
    retrieval_max_docs: int = Field(default=8, gt=0)
    retrieval_min_score: float = Field(default=0.2, ge=0, le=1)
    context_max_chars: int = Field(default=6000, gt=0)
    analyze_with_llm: bool = False
    query_rewrite_with_llm: bool = False

    # --- LangSmith (off by default) ---
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "evalcode"

    @field_validator("doc_libraries", mode="before")
    @classmethod
    def _split_doc_libraries(cls, value: object) -> object:
        """Accept a comma-separated env string as well as a real list."""
        if isinstance(value, str):
            value = [item.strip() for item in value.split(",")]
        if isinstance(value, list):
            cleaned = [item for item in (str(v).strip() for v in value) if item]
            if not cleaned:
                raise ValueError("doc_libraries must name at least one library")
            return cleaned
        return value

    def require_api_key(self) -> SecretStr:
        """Return the OpenRouter API key, or raise ``ConfigError`` with setup help.

        The raised message never includes any key material.
        """
        key = self.openrouter_api_key
        if key is None or not key.get_secret_value().strip():
            raise ConfigError(
                "OPENROUTER_API_KEY is not set. Get a free API key at "
                "https://openrouter.ai/keys, then export OPENROUTER_API_KEY in your "
                "environment or set it in a git-ignored .env file (copy "
                ".env.example to .env and fill it in). Never commit your key."
            )
        return key

    def safe_dump(self) -> dict[str, Any]:
        """All settings as a dict with secret values masked, for logging."""
        data: dict[str, Any] = {name: getattr(self, name) for name in type(self).model_fields}
        for name in _SECRET_FIELDS:
            data[name] = "***" if data[name] is not None else "unset"
        return data


def load_settings(env_file: Path | None = None) -> Settings:
    """Load ``.env`` (without overriding real env vars), then build ``Settings``.

    ``env_file`` points at an explicit .env; otherwise the nearest ``.env``
    found from the working directory is used. Real environment variables
    always win over ``.env`` values.
    """
    load_dotenv(dotenv_path=env_file or find_dotenv(usecwd=True), override=False)
    return Settings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings. Call ``get_settings.cache_clear()`` to reload (tests)."""
    return load_settings()
