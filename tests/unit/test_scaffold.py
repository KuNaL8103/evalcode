"""Task 0 scaffold tests: package metadata and dependency availability."""

from __future__ import annotations

import importlib


def test_package_exposes_version() -> None:
    import evalcode

    assert evalcode.__version__ == "0.1.0"


def test_core_dependencies_import() -> None:
    """All runtime dependencies from pyproject.toml are importable."""
    for module_name in [
        "langgraph",
        "langchain_openai",
        "langchain_huggingface",
        "sentence_transformers",
        "openai",
        "chromadb",
        "dotenv",
        "typer",
        "rich",
    ]:
        assert importlib.import_module(module_name) is not None
