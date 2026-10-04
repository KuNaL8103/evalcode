"""Task 0 scaffold tests: package metadata and dependency availability."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util


def test_package_exposes_version() -> None:
    import evalcode

    assert evalcode.__version__ == "0.1.0"


def test_core_dependencies_import() -> None:
    """All runtime dependencies from pyproject.toml are installed and located.

    This deliberately does NOT import the heavy packages (torch,
    sentence-transformers, langchain-huggingface): real imports of those are
    exercised by the ``slow`` tests that start in Task 3. Here we only check
    that each distribution is installed (``importlib.metadata.version``) and
    that its import target can be found (``importlib.util.find_spec``) without
    running any of it.
    """
    distributions = [
        "langgraph",
        "langgraph-checkpoint-sqlite",
        "langchain-core",
        "langchain-openai",
        "langchain-huggingface",
        "sentence-transformers",
        "openai",
        "chromadb",
        "python-dotenv",
        "pydantic",
        "pydantic-settings",
        "typer",
        "rich",
        "pyyaml",
        "torch",
    ]
    for dist in distributions:
        importlib.metadata.version(dist)  # raises PackageNotFoundError if missing

    import_names = [
        "langgraph",
        "langchain_openai",
        "langchain_huggingface",
        "sentence_transformers",
        "openai",
        "chromadb",
        "dotenv",
        "typer",
        "rich",
    ]
    for name in import_names:
        assert importlib.util.find_spec(name) is not None, f"cannot locate {name!r}"
