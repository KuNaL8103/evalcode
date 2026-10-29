"""Validated response shapes and small code-introspection helpers.

``CodeBundle`` is the validated result of parsing one model response
(tagged plain-text protocol, ARCHITECTURE §2). ``extract_imports`` gives
the graph's bookkeeping a cheap view of which modules the generated code
touches — used for summaries and later error attribution.
"""

from __future__ import annotations

import ast

from pydantic import BaseModel, Field

__all__ = ["CodeBundle", "extract_imports"]


class CodeBundle(BaseModel):
    """One parsed model response.

    ``code`` is the required artifact (the ``solution.py`` body);
    ``tests`` is empty when the model was told not to write tests
    (``provided_tests`` mode) or when none were requested.
    """

    explanation: str = ""
    code: str
    tests: str = ""
    docs_used: list[str] = Field(default_factory=list)


def extract_imports(code: str) -> list[str]:
    """Top-level imported module names, in first-seen order.

    ``import json`` → ``["json"]``; ``import os.path`` → ``["os"]``;
    ``from collections import deque`` → ``["collections"]``. Syntax
    errors (or non-str input) are tolerated and yield ``[]`` — this is
    a summary aid, never a gate.
    """
    if not code:
        return []
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return []
    seen: set[str] = set()
    names: list[str] = []

    def _add(module: str | None) -> None:
        if not module:
            return
        top = module.split(".")[0]
        if top not in seen:
            seen.add(top)
            names.append(top)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            # node.module is None for bare relative imports; only the
            # absolute ones (level == 0) name a real top-level module.
            if node.level == 0:
                _add(node.module)
    return names
