"""Task 2 tests: chunking helpers and documentation loaders.

No LLM or network access; the introspection loader runs against stdlib ``json``
and a throwaway package created in ``tmp_path``.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from evalcode.rag.chunking import split_markdown, stable_id, truncate_text
from evalcode.rag.loaders import iter_introspection_chunks, iter_text_file_chunks
from evalcode.rag.types import DocChunk

# Throwaway package exercising __all__, privacy, classes/methods, a signature
# fallback, a re-export dedupe, and a deliberately broken attribute.
_PACKAGE_SRC = '''
__all__ = ["public_func", "MyClass", "alias_of_public", "nosig", "broken"]

import builtins as _builtins


def _hidden():
    """Private helper, must be skipped."""
    return 0


def public_func(x):
    """Public function doc."""
    return x * 2


alias_of_public = public_func


def other_public():
    """Not listed in __all__, must be excluded."""
    return 1


nosig = _builtins.vars


class MyClass:
    """A sample class."""

    def __init__(self, value):
        self.value = value

    def greet(self, name):
        """Greet a person."""
        return "hello " + name

    def _secret(self):
        return 1


def __getattr__(name):
    if name == "broken":
        raise RuntimeError("intentional failure")
    raise AttributeError(name)
'''


@pytest.fixture
def mymod(tmp_path: Path):
    """Create, import, and clean up a throwaway package named ``mymod``."""
    pkg = tmp_path / "mymod"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(_PACKAGE_SRC, encoding="utf-8")
    sys.path.insert(0, str(tmp_path))
    try:
        module = importlib.import_module("mymod")
        yield module
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("mymod", None)


def _qualnames(chunks: list[DocChunk]) -> set[str]:
    return {c.metadata["qualname"] for c in chunks}


# --- chunking: stable_id ---------------------------------------------------


def test_stable_id_deterministic() -> None:
    a = stable_id("json", "1.0", "function", "json.loads")
    b = stable_id("json", "1.0", "function", "json.loads")
    assert a == b
    assert len(a) == 16
    assert int(a, 16) >= 0  # hex
    # Different parts → different id; order of parts matters.
    assert a != stable_id("json", "1.0", "function", "json.dumps")
    # The separator keeps ("ab","c") distinct from ("a","bc").
    assert stable_id("ab", "c") != stable_id("a", "bc")


# --- chunking: truncate_text ----------------------------------------------


def test_truncate_text_boundaries() -> None:
    # Within the limit (or exactly at it) → returned unchanged.
    assert truncate_text("abc", 10) == "abc"
    assert truncate_text("abcdef", 6) == "abcdef"
    # max_chars <= 0 → returned unchanged.
    assert truncate_text("abcdef", 0) == "abcdef"

    # A cut appends the ellipsis and never exceeds the limit.
    out = truncate_text("The quick brown fox jumps over the lazy dog. More.", 20)
    assert len(out) <= 20
    assert out.endswith("…")
    # Cuts at a word boundary, not mid-word.
    assert out.startswith("The quick brown")

    # Prefers a sentence boundary when one is available in the window.
    assert truncate_text("Hello world. Next part.", 15) == "Hello world.…"


# --- chunking: split_markdown ----------------------------------------------


def test_split_markdown_by_headings() -> None:
    md = (
        "# Title\n\nIntro paragraph text.\n\n"
        "## Section A\n\nAlpha content.\n\n"
        "## Section B\n\nBeta content.\n"
    )
    pieces = split_markdown(md, max_chars=800, overlap=100)
    assert all(p.strip() for p in pieces)
    assert all(len(p) <= 800 for p in pieces)
    # Each section keeps its heading at the top of the piece.
    assert any("Section A" in p and "Alpha content." in p for p in pieces)
    assert any("Section B" in p and "Beta content." in p for p in pieces)
    assert any("Title" in p and "Intro paragraph text." in p for p in pieces)


def test_split_markdown_overlap() -> None:
    body = "word " * 200  # one long line of repeated tokens
    md = "# H\n\n" + body
    pieces = split_markdown(md, max_chars=120, overlap=40)
    assert len(pieces) >= 2
    # Strip the shared heading prefix, then the tail of one piece reappears
    # at the head of the next (the overlap region).
    prefix = "# H\n\n"
    bodies = [p[len(prefix) :] for p in pieces]
    for i in range(len(bodies) - 1):
        tail = bodies[i][-12:]
        assert tail in bodies[i + 1], f"missing overlap between pieces {i} and {i + 1}"


def test_split_markdown_empty_and_whitespace() -> None:
    assert split_markdown("") == []
    assert split_markdown("   ") == []
    assert split_markdown("  \n\t  \n") == []
    # A heading with no body still yields a piece, not an empty string.
    pieces = split_markdown("# Only A Heading", max_chars=100, overlap=0)
    assert pieces == ["# Only A Heading"]


def test_split_markdown_respects_max_chars() -> None:
    md = "# H\n\n" + ("lorem ipsum " * 100)  # one long line
    pieces = split_markdown(md, max_chars=100, overlap=20)
    assert pieces
    for piece in pieces:
        assert len(piece) <= 100, f"piece too long: {len(piece)}"
    assert all(p.strip() for p in pieces)

    # An unsplittable single line longer than max_chars is kept whole.
    long_line = "x" * 500
    pieces2 = split_markdown("# H\n\n" + long_line, max_chars=100, overlap=20)
    assert len(pieces2) == 1
    assert len(pieces2[0]) > 100  # allowed to exceed for a single line
    assert pieces2[0].endswith(long_line)


# --- loaders: introspection ------------------------------------------------


def test_introspection_json_loads() -> None:
    chunks = list(iter_introspection_chunks("json"))
    assert chunks
    by_name = {c.metadata["qualname"]: c for c in chunks}
    assert "json.loads" in by_name
    loads = by_name["json.loads"]
    # Signature line comes first and mentions the dotted name.
    assert loads.text.startswith("json.loads")
    assert "loads" in loads.metadata["qualname"]
    assert loads.metadata["kind"] == "function"
    assert loads.metadata["library"] == "json"
    assert loads.metadata["source_type"] == "introspection"
    # A missing library logs a warning and yields nothing, no exception.
    assert list(iter_introspection_chunks("no_such_module_xyz")) == []


def test_introspection_thrownaway_package(mymod) -> None:
    chunks = list(iter_introspection_chunks("mymod"))
    names = _qualnames(chunks)

    # __all__ is respected: other_public is NOT in __all__, so it is excluded.
    assert "mymod.other_public" not in names
    # Private names are skipped.
    assert "mymod._hidden" not in names
    assert "mymod.MyClass._secret" not in names
    # Class and its public method are both present.
    assert "mymod.MyClass" in names
    assert "mymod.MyClass.greet" in names
    cls = next(c for c in chunks if c.metadata["qualname"] == "mymod.MyClass")
    assert cls.metadata["kind"] == "class"
    assert "greet" in cls.text  # the class chunk lists method names
    # Re-export deduped: public_func indexed once, alias contributes no chunk.
    assert len([c for c in chunks if c.metadata["qualname"] == "mymod.public_func"]) == 1
    assert "mymod.alias_of_public" not in names
    # Signature fallback "(...)" for the builtin with no introspectable sig.
    nosig = next(c for c in chunks if c.metadata["qualname"] == "mymod.nosig")
    assert "(...)" in nosig.text


def test_introspection_broken_attribute_does_not_abort(mymod) -> None:
    # "broken" is in __all__ but its access raises; the walk must survive and
    # still yield the healthy chunks.
    chunks = list(iter_introspection_chunks("mymod"))
    names = _qualnames(chunks)
    assert "mymod.broken" not in names
    # The other, valid objects were still produced.
    assert {"mymod.public_func", "mymod.MyClass", "mymod.nosig"} <= names


def test_introspection_ids_stable_across_runs() -> None:
    run1 = {c.metadata["qualname"]: c.id for c in iter_introspection_chunks("json")}
    run2 = {c.metadata["qualname"]: c.id for c in iter_introspection_chunks("json")}
    assert run1 == run2
    assert run1  # sanity: we actually compared ids
    for cid in run1.values():
        assert len(cid) == 16


# --- loaders: text files ---------------------------------------------------


def test_text_file_loader_nested(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    (root / "sub").mkdir(parents=True)
    (root / "a.md").write_bytes(b"# Alpha\n\nAlpha body text here. Some words.\n")
    # A CRLF-encoded file, written explicitly as bytes.
    (root / "sub" / "b.md").write_bytes(b"# Beta\r\n\r\nBeta body with CRLF.\r\n")
    # A non-ASCII file, written explicitly as UTF-8 bytes.
    (root / "c.txt").write_bytes("Café notes with émoji.\n".encode("utf-8"))  # noqa: UP012

    chunks = list(iter_text_file_chunks(root))
    assert chunks
    # Every chunk has forward-slash source_path and no backslashes.
    for c in chunks:
        assert "\\" not in c.metadata["source_path"]
        assert c.metadata["source_type"] == "text_file"
        assert c.metadata["kind"] == "text"

    # The CRLF file and the non-ASCII file each contributed content.
    texts = " ".join(c.text for c in chunks)
    assert "Beta body with CRLF." in texts
    assert "Café" in texts

    # ids are stable across two runs over the same tree.
    ids_a = [c.id for c in iter_text_file_chunks(root)]
    ids_b = [c.id for c in iter_text_file_chunks(root)]
    assert ids_a == ids_b
    assert len(set(ids_a)) == len(ids_a)


def test_chunk_metadata_and_size_bounds(tmp_path: Path) -> None:
    required = {"library", "version", "qualname", "kind", "import_path", "source_type"}
    allowed_kinds = {"module", "function", "class", "method", "text"}

    # Introspection chunks: required keys present, valid kind, bounded size.
    for c in iter_introspection_chunks("json", max_chars=1000):
        assert required <= set(c.metadata)
        assert c.metadata["kind"] in allowed_kinds
        assert c.metadata["source_type"] == "introspection"
        assert len(c.text) <= 1000, f"introspection chunk too long: {len(c.text)}"

    # Text chunks: required keys + source_path present, bounded size.
    doc = tmp_path / "d.md"
    doc.write_text("# T\n\n" + ("body words " * 200), encoding="utf-8")
    for c in iter_text_file_chunks(doc, max_chars=200, overlap=20):
        assert required <= set(c.metadata)
        assert c.metadata["kind"] == "text"
        assert c.metadata["source_type"] == "text_file"
        assert c.metadata["source_path"]  # forward-slash path present
        assert len(c.text) <= 200, f"text chunk too long: {len(c.text)}"
