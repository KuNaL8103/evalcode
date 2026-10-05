"""Tolerant parser for the tagged plain-text LLM output protocol (§2).

Free models ignore strict formats, so the parser accepts: case-insensitive
tags, stray whitespace, markdown code fences *inside* the ``<code>``/
``<tests>`` tags, a missing closing tag at the end of the text, and prose
before/after. When no ``<code>`` tag exists at all, the first fenced block
is treated as code and the second as tests. ``ParseError`` (with a short
reason) is raised only when no usable code — or, when requested, no usable
tests — can be recovered.
"""

from __future__ import annotations

import re

from evalcode.errors import ParseError
from evalcode.llm import strip_reasoning
from evalcode.schemas import CodeBundle

__all__ = ["parse_bundle", "parse_tagged"]


def parse_tagged(text: str, tag: str) -> str | None:
    """Return the content between ``<tag>`` and ``</tag>`` (case-insensitive).

    Tolerates surrounding whitespace and a missing closing tag (the content
    then runs to the end of the text). Returns ``None`` when no opening tag
    is present. Fence stripping is *not* done here — tag extraction only.
    """
    if not text:
        return None
    pattern = re.compile(
        r"<\s*" + re.escape(tag) + r"\s*>\s*(.*?)\s*(?:</\s*" + re.escape(tag) + r"\s*>|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text)
    if match is None:
        return None
    return match.group(1)


_FENCE_CLOSE_RE = re.compile(r"^[ \t]*```")


def _strip_code_fence(text: str) -> str:
    """Strip an optional ```` ```python …``` ```` fence wrapping the text."""
    stripped = text.strip()
    lines = stripped.splitlines()
    if not lines or not lines[0].lstrip().startswith("```"):
        return stripped
    end = len(lines)
    for i in range(1, len(lines)):
        if _FENCE_CLOSE_RE.match(lines[i]):
            end = i
            break
    return "\n".join(lines[1:end]).strip()


def _fenced_blocks(text: str) -> list[str]:
    """All fenced code blocks in order (contents, fences removed)."""
    pattern = re.compile(r"```[A-Za-z0-9_+.\-]*[ \t]*\n(.*?)```", re.DOTALL)
    return [block.strip() for block in pattern.findall(text)]


def _parse_doc_ids(raw: str | None) -> list[str]:
    """Split a ``docs_used`` body ("id1, id2" or one per line) into ids."""
    if not raw:
        return []
    ids = [part.strip() for part in re.split(r"[,\n;]", raw)]
    return [part for part in ids if part]


def parse_bundle(text: str, *, require_tests: bool = False) -> CodeBundle:
    """Parse one model response into a ``CodeBundle``.

    Raises ``ParseError`` with a short reason when no code is recoverable,
    or when ``require_tests`` is set and no tests are recoverable.
    """
    if not text or not text.strip():
        raise ParseError("empty model response")
    # The client already strips reasoning blocks; do it again defensively
    # in case the text reached us by another route.
    text = strip_reasoning(text)

    explanation = parse_tagged(text, "explanation") or ""
    code_raw = parse_tagged(text, "code")
    tests_raw = parse_tagged(text, "tests")
    docs_used = _parse_doc_ids(parse_tagged(text, "docs_used"))

    code = _strip_code_fence(code_raw) if code_raw is not None else ""
    tests = _strip_code_fence(tests_raw) if tests_raw is not None else ""

    if not code.strip():
        # Fallback (§2 risk table): the model ignored the tag protocol
        # entirely — first fenced block is code, second is tests.
        blocks = _fenced_blocks(text)
        if blocks:
            code = blocks[0]
            if not tests.strip() and len(blocks) > 1:
                tests = blocks[1]

    if not code.strip():
        raise ParseError("no code found: no <code> tag or fenced Python block in response")
    if require_tests and not tests.strip():
        raise ParseError("no tests found: no <tests> tag or second fenced block in response")

    return CodeBundle(
        explanation=explanation.strip(),
        code=code.strip(),
        tests=tests.strip(),
        docs_used=docs_used,
    )
