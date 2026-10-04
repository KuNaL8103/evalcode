"""Task 2 tests: chunking helpers and documentation loaders.

No LLM or network access; the introspection loader runs against stdlib ``json``
and a throwaway package created in ``tmp_path``.
"""

from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path

import pytest

from evalcode.rag.chunking import split_markdown, stable_id, truncate_text
from evalcode.rag.loaders import iter_introspection_chunks, iter_text_file_chunks
from evalcode.rag.types import DocChunk

# Throwaway package exercising __all__, privacy, classes/methods, inherited
# methods, a signature fallback, a re-export dedupe, a broken attribute, and
# plain constants.
_PACKAGE_SRC = '''
__all__ = [
    "public_func",
    "MyClass",
    "alias_of_public",
    "nosig",
    "broken",
    "with_obj",
    "PI",
    "LIMIT",
    "NOTHING",
    "Base",
    "Child",
    "Err",
]

import builtins as _builtins


def _hidden():
    """Private helper, must be skipped."""
    return 0


def public_func(x):
    """Public function doc."""
    return x * 2


def with_obj(x, _w=object()):
    """Doc."""
    return x


alias_of_public = public_func


def other_public():
    """Not listed in __all__, must be excluded."""
    return 1


nosig = _builtins.vars


PI = 3.14159
LIMIT = 10
NOTHING = None


class MyClass:
    """A sample class."""

    def __init__(self, value):
        self.value = value

    def greet(self, name):
        """Greet a person."""
        return "hello " + name

    def _secret(self):
        return 1


class Base:
    def inherited(self):
        """Inherited doc."""
        return 1


class Child(Base):
    def own(self):
        """Own doc."""
        return 2


class Err(ValueError):
    """Custom error."""


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
        "```python\n# not a heading\nx = 1\n```\n\n"
        "## Section B\n\nBeta content.\n"
    )
    pieces = split_markdown(md, max_chars=800, overlap=100)
    assert all(p.strip() for p in pieces)
    assert all(len(p) <= 800 for p in pieces)
    # Each section keeps its heading at the top of the piece.
    assert any("Section A" in p and "Alpha content." in p for p in pieces)
    assert any("Section B" in p and "Beta content." in p for p in pieces)
    assert any("Title" in p and "Intro paragraph text." in p for p in pieces)

    # A "#" line inside a fenced code block is NOT a heading: the piece that
    # holds it also carries its section heading and the following code line,
    # and no piece starts with that in-fence line.
    fenced = [p for p in pieces if "# not a heading" in p]
    assert fenced, "expected a piece containing the in-fence comment"
    for piece in fenced:
        assert "Section A" in piece
        assert "x = 1" in piece
    assert not any(p.startswith("# not a heading") for p in pieces)


def test_split_markdown_overlap() -> None:
    # Distinct tokens (so a shared token is only possible via real overlap),
    # enough of them to force several pieces.
    body = " ".join(f"tok{i:03d}" for i in range(200))
    md = "# H\n\n" + body
    prefix = "# H\n\n"

    def tokens_of(piece: str) -> set[str]:
        return set(piece[len(prefix) :].split())

    # overlap=40: consecutive pieces share at least one full token.
    pieces = split_markdown(md, max_chars=120, overlap=40)
    assert len(pieces) >= 3
    for i in range(len(pieces) - 1):
        shared = tokens_of(pieces[i]) & tokens_of(pieces[i + 1])
        assert shared, f"no shared token between pieces {i} and {i + 1}"

    # overlap=0: consecutive pieces share NO tokens.
    pieces0 = split_markdown(md, max_chars=120, overlap=0)
    assert len(pieces0) >= 3
    for i in range(len(pieces0) - 1):
        shared = tokens_of(pieces0[i]) & tokens_of(pieces0[i + 1])
        assert not shared, f"unexpected shared token at {i} with overlap=0"


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
    assert len(chunks) == 12  # json still yields exactly 12 chunks
    by_name = {c.metadata["qualname"]: c for c in chunks}
    assert "json.loads" in by_name
    loads = by_name["json.loads"]
    # Signature line comes first and mentions the dotted name.
    assert loads.text.startswith("json.loads")
    assert "loads" in loads.metadata["qualname"]
    assert loads.metadata["kind"] == "function"
    assert loads.metadata["library"] == "json"
    assert loads.metadata["source_type"] == "introspection"
    # Compact embedding text: "dotted.name: first sentence", short, while the
    # full chunk text keeps the signature line for display/prompts.
    assert loads.embed_text is not None
    assert loads.embed_text.startswith("json.loads")
    assert len(loads.embed_text) < 300
    assert loads.text.startswith("json.loads")

    # C-implemented class methods are indexed as method chunks whose text
    # starts with their dotted name.
    dt = {c.metadata["qualname"]: c for c in iter_introspection_chunks("datetime")}
    for qual in ("datetime.datetime.strptime", "datetime.timedelta.total_seconds"):
        assert qual in dt, qual
        assert dt[qual].metadata["kind"] == "method"
        assert dt[qual].text.startswith(qual)

    col = {c.metadata["qualname"]: c for c in iter_introspection_chunks("collections")}
    assert "collections.deque.appendleft" in col
    assert col["collections.deque.appendleft"].metadata["kind"] == "method"

    # Methods inherited from a Python base class are indexed under the
    # concrete class (PurePath.with_suffix on pathlib.Path).
    pl = {c.metadata["qualname"]: c for c in iter_introspection_chunks("pathlib")}
    assert "pathlib.Path.with_suffix" in pl
    assert pl["pathlib.Path.with_suffix"].metadata["library"] == "pathlib"
    assert pl["pathlib.Path.with_suffix"].metadata["kind"] == "method"

    # Every introspection chunk (json, datetime, collections) stays bounded,
    # and no runtime object-repr pointer leaks into any chunk text.
    for c in list(chunks) + list(dt.values()) + list(col.values()):
        assert len(c.text) <= 1000, f"chunk too long: {c.metadata['qualname']}"
        assert " at 0x" not in c.text, c.metadata["qualname"]

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
    # Methods inherited from a Python base are indexed on the subclass,
    # alongside methods defined on the subclass itself.
    assert "mymod.Child.own" in names
    assert "mymod.Child.inherited" in names
    # Members inherited from builtins bases are NOT re-indexed.
    assert "mymod.Err.add_note" not in names
    assert "mymod.Err.with_traceback" not in names
    # Signature fallback "(...)" for the builtin with no introspectable sig.
    nosig = next(c for c in chunks if c.metadata["qualname"] == "mymod.nosig")
    assert "(...)" in nosig.text
    # A default value that renders as a runtime object repr is sanitized to
    # "..." so the chunk text carries no pointer and stays deterministic.
    with_obj = next(c for c in chunks if c.metadata["qualname"] == "mymod.with_obj")
    assert "_w=..." in with_obj.text
    assert "0x" not in with_obj.text
    # Plain constants are never turned into chunks.
    assert "mymod.PI" not in names
    assert "mymod.LIMIT" not in names
    assert "mymod.NOTHING" not in names


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
    # A BOM-prefixed file, written explicitly as bytes.
    (root / "sub" / "gamma.md").write_bytes(b"\xef\xbb\xbf# Gamma\n\nGamma body.\n")

    chunks = list(iter_text_file_chunks(root))
    assert chunks
    # source_path is root-relative with forward slashes (no drive, no leading /).
    paths = {c.metadata["source_path"] for c in chunks}
    assert {"a.md", "sub/b.md", "c.txt", "sub/gamma.md"} <= paths
    for c in chunks:
        sp = c.metadata["source_path"]
        assert not sp.startswith("/")
        assert "\\" not in sp
        assert c.metadata["source_type"] == "text_file"
        assert c.metadata["kind"] == "text"

    # The CRLF, non-ASCII, and BOM files each contributed content.
    texts = " ".join(c.text for c in chunks)
    assert "Beta body with CRLF." in texts
    assert "Café" in texts
    # The BOM is dropped: no U+FEFF survives in any chunk text.
    assert not any("\ufeff" in c.text for c in chunks)
    assert any(c.text.startswith("# Gamma") for c in chunks)

    # ids depend on the root-relative path, so copying the tree to a new
    # location yields identical ids.
    dest = tmp_path / "docs_copy"
    shutil.copytree(root, dest)
    ids_a = [c.id for c in iter_text_file_chunks(root)]
    ids_b = [c.id for c in iter_text_file_chunks(dest)]
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
