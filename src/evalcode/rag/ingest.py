"""Ingestion, chunking, and documentation loaders.

This module is the offline corpus-building half of the RAG layer:

- deterministic chunk ids and text truncation;
- markdown-aware splitting for local docs;
- introspection of installed Python libraries;
- text-file loading;
- embedding/upsert orchestration (:func:`ingest`) and its CLI.

Splitting it further adds files without adding a boundary: all of these share
the same ``DocChunk`` shape and embedding-size constraints.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import logging
import re
import sys
import time
import types
from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict

from evalcode.rag.store import DocChunk, MetadataValue, VectorStore

logger = logging.getLogger(__name__)


class IngestReport(TypedDict):
    """Outcome of one :func:`ingest` call (plain data, JSON-serializable)."""

    per_library: dict[str, int]  # chunk counts grouped by the library metadata
    skipped: list[str]  # requested libraries that yielded no chunks
    total: int
    seconds: float


def ingest(
    libraries: list[str],
    store: VectorStore,
    *,
    docs_dir: Path | None = None,
    reset: bool = False,
) -> IngestReport:
    """Chunk ``libraries`` (and, if given, text docs in ``docs_dir``) into ``store``.

    With ``reset`` the store is emptied first. Each library is embedded and
    upserted in one batch with progress logging; a library that yields no
    chunks (e.g. it cannot be imported) is recorded in ``skipped``.
    """
    started = time.monotonic()
    if reset:
        store.reset()
        logger.info("ingest: store reset (%s)", store.persist_dir)

    per_library: dict[str, int] = {}
    skipped: list[str] = []
    total = 0

    for library in libraries:
        chunks: list[DocChunk] = list(iter_introspection_chunks(library))
        if not chunks:
            logger.warning("ingest: no chunks for library %r (skipped)", library)
            skipped.append(library)
            continue
        written = store.upsert(chunks)
        _add_counts(per_library, chunks)
        total += written
        logger.info("ingest: %s -> %d chunks (store total %d)", library, written, store.count())

    if docs_dir is not None:
        doc_chunks = list(iter_text_file_chunks(docs_dir))
        if doc_chunks:
            written = store.upsert(doc_chunks)
            total += written
            _add_counts(per_library, doc_chunks)
            logger.info("ingest: docs dir %s -> %d chunks", docs_dir, written)
        else:
            logger.warning("ingest: no text docs found in %s", docs_dir)

    seconds = time.monotonic() - started
    logger.info("ingest: done in %.1fs (total %d chunks)", seconds, total)
    return IngestReport(per_library=per_library, skipped=skipped, total=total, seconds=seconds)


def _add_counts(per_library: dict[str, int], chunks: list[DocChunk]) -> None:
    """Accumulate per-library counts for a batch of chunks (text docs)."""
    for c in chunks:
        library = str(c.metadata.get("library", ""))
        per_library[library] = per_library.get(library, 0) + 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest libraries/docs into the evalcode vector store "
        "(python -m evalcode.rag.ingest)."
    )
    parser.add_argument(
        "--libs",
        default=None,
        help="comma-separated libraries (default: DOC_LIBRARIES setting)",
    )
    parser.add_argument("--docs-dir", default=None, help="directory of .md/.rst/.txt docs")
    parser.add_argument("--reset", action="store_true", help="clear the store before ingesting")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # Imported here (not at module level) so importing this module in tests
    # never loads settings or env.
    from evalcode.core.config import get_settings
    from evalcode.rag.embeddings import get_embedder

    settings = get_settings()
    if args.libs:
        libraries = [item.strip() for item in args.libs.split(",") if item.strip()]
    else:
        libraries = list(settings.doc_libraries)
    if args.docs_dir:
        docs_dir = Path(args.docs_dir)
    else:
        default_docs = Path(settings.docs_dir)
        docs_dir = default_docs if default_docs.exists() else None

    embedder = get_embedder(settings)
    store = VectorStore(settings.chroma_dir, settings.collection_name, embedder)
    try:
        report = ingest(libraries, store, docs_dir=docs_dir, reset=args.reset)
    finally:
        store.close()

    for library, count in report["per_library"].items():
        print(f"{library}: {count} chunks")
    for library in report["skipped"]:
        print(f"{library}: skipped (no chunks)")
    print(f"total: {report['total']} chunks in {report['seconds']:.1f}s -> {settings.chroma_dir}")
    return 0


# --- Chunking helpers -----------------------------------------------------

_ELLIPSIS = "…"


def stable_id(*parts: str) -> str:
    """Return a stable 16-char hex id derived from the given parts.

    A separator is inserted between parts so that ``("ab", "c")`` and
    ``("a", "bc")`` do not collide.
    """
    joined = "\x00".join(parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


def truncate_text(text: str, max_chars: int) -> str:
    """Cut ``text`` to at most ``max_chars``, preferring a clean boundary.

    Preference order: sentence end, newline, then word boundary. When the text
    is actually cut, a single ``…`` (U+2026) is appended; the result is always
    no longer than ``max_chars``. Text already within the limit is returned
    unchanged.
    """
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text

    window = text[: max_chars - 1]  # reserve one slot for the ellipsis
    cut = _pick_truncation(window)
    cut = max(cut, 1)
    return window[:cut].rstrip() + _ELLIPSIS


def _pick_truncation(window: str) -> int:
    """Choose a truncation index inside ``window``."""
    last_sentence: re.Match[str] | None = None
    for match in re.finditer(r"[.!?](?:\s|$)", window):
        last_sentence = match
    if last_sentence is not None:
        return last_sentence.end()

    nl = window.rfind("\n")
    if nl > 0:
        return nl + 1
    space = window.rfind(" ")
    if space > 0:
        return space + 1
    return len(window)


# --- Markdown splitting ----------------------------------------------------

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s")


def split_markdown(
    text: str,
    max_chars: int = 800,
    overlap: int = 100,
) -> list[str]:
    """Split markdown/text into bounded pieces, keeping heading context.

    Strategy: first split on headings (each piece keeps the current heading at
    its top), then pack content up to ``max_chars`` preferring paragraph,
    line, and word boundaries. Consecutive pieces of an over-long section share
    an ``overlap``-character tail so context is not lost at the boundary. No
    piece is empty; a single line longer than ``max_chars`` is returned whole
    rather than broken mid-line. CRLF line endings are treated as LF.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    out: list[str] = []
    for heading, body in _split_into_sections(text):
        out.extend(_split_section(heading, body, max_chars, overlap))
    return [piece for piece in out if piece.strip()]


_FENCE_RE = re.compile(r"^\s*(```|~~~)")


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split into ``(heading, body)`` sections on heading lines.

    Lines inside a fenced code block (delimited by a line starting with three
    backticks or three tildes) are never treated as headings, so a ``#`` comment
    inside e.g. a Python block does not start a new section.
    """
    sections: list[tuple[str, str]] = []
    heading = ""
    buffer: list[str] = []
    in_fence = False
    for line in text.split("\n"):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            buffer.append(line)
            continue
        if not in_fence and _HEADING_RE.match(line):
            if heading or "".join(buffer).strip():
                sections.append((heading, "\n".join(buffer)))
            heading = line.strip()
            buffer = []
        else:
            buffer.append(line)
    if heading or "".join(buffer).strip():
        sections.append((heading, "\n".join(buffer)))
    return [(h, b) for h, b in sections if h.strip() or b.strip()]


def _split_section(heading: str, body: str, max_chars: int, overlap: int) -> list[str]:
    """Split one section, re-attaching the heading to every piece."""
    body = body.strip()
    if not body:
        return [heading] if heading and len(heading) <= max_chars else []

    prefix = heading + "\n\n" if heading else ""
    # Fast path: the whole section fits.
    if len(prefix) + len(body) <= max_chars:
        return [prefix + body]

    budget = max(max_chars - len(prefix), 1)
    pieces = _split_into_pieces(body, budget, overlap)
    return [prefix + piece for piece in pieces]


def _split_into_pieces(body: str, max_chars: int, overlap: int) -> list[str]:
    """Slice ``body`` into pieces of at most ``max_chars`` with an overlap."""
    n = len(body)
    pieces: list[str] = []
    pos = 0
    while pos < n:
        end = pos + max_chars
        if end >= n:
            pieces.append(body[pos:].strip())
            break

        cut = _find_boundary(body, pos + 1, end)
        extended = False
        if cut == end and _is_unsplittable(body, pos, end):
            # A single line/word runs past the window: keep it whole even if
            # it exceeds the limit (the documented exception).
            nl = body.find("\n", end)
            cut = nl + 1 if nl != -1 else n
            extended = True
        if cut <= pos:
            cut = end

        pieces.append(body[pos:cut].strip())
        if cut >= n:
            break
        # Overlap into the next piece, unless we just emitted a long line.
        next_pos = cut if extended else cut - overlap
        pos = next_pos if next_pos > pos else cut
    return [piece for piece in pieces if piece]


def _find_boundary(text: str, lo: int, hi: int) -> int:
    """Find the last clean boundary in ``text[lo:hi]`` (else ``hi``)."""
    segment = text[lo:hi]
    for marker, step in (("\n\n", 2), ("\n", 1), (" ", 1)):
        idx = segment.rfind(marker)
        if idx > 0:
            return lo + idx + step
    return hi


def _is_unsplittable(text: str, lo: int, hi: int) -> bool:
    """True when ``text[lo:hi]`` is one unbroken token with no line break."""
    segment = text[lo:hi]
    return "\n" not in segment and " " not in segment


# --- Documentation loaders -----------------------------------------------


logger = logging.getLogger(__name__)

_TEXT_SUFFIXES = {".md", ".rst", ".txt"}

# Member types that count as a class method, including C-implemented ones.
# Properties, nested classes, and plain data attributes are intentionally
# excluded because their static value is none of these types.
_METHOD_TYPES = (
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodDescriptorType,
    types.ClassMethodDescriptorType,
)

# Default values that render as a runtime object repr, e.g.
# ``_w=<built-in method match of re.Pattern object at 0x000001CD4B82E810>``.
# They are pure noise in embeddings and prompts, and the pointer makes chunk
# text non-deterministic across processes; replace the whole ``<...>`` token.
_OBJ_REPR_RE = re.compile(r"<[^<>]*at 0x[0-9A-Fa-f]+>")


def _clean_signature(signature: str) -> str:
    """Strip runtime object reprs from a rendered signature."""
    return _OBJ_REPR_RE.sub("...", signature)


# --- Introspection loader --------------------------------------------------


def iter_introspection_chunks(
    library: str,
    *,
    max_chars: int = 1000,
    include_private: bool = False,
) -> Iterator[DocChunk]:
    """Yield a :class:`DocChunk` for each public API object in ``library``.

    The library is imported; on ``ImportError`` a warning is logged and nothing
    is yielded. Public names are taken from ``__all__`` when present, else the
    non-underscore names (or, with ``include_private``, all but dunder names).
    A single broken object is logged and skipped.
    """
    try:
        module = importlib.import_module(library)
    except Exception as exc:  # ImportError and friends
        logger.warning("introspection: cannot import %r: %s", library, exc)
        return

    seen: set[int] = set()

    for name in _public_names(module, include_private):
        try:
            obj = getattr(module, name)
        except Exception as exc:  # a broken attribute must not abort the walk
            logger.warning("introspection: skipping %s.%s: %s", library, name, exc)
            continue

        if inspect.ismodule(obj):
            # Submodules are not walked (and not already top-level attrs of
            # interest); they are skipped, not recursed into.
            continue
        oid = id(obj)
        if oid in seen:
            # Re-exported under another public name: dedupe by identity.
            continue
        seen.add(oid)

        try:
            if inspect.isclass(obj):
                yield from _class_chunks(library, name, obj, max_chars, include_private)
            else:
                yield from _callable_chunks(library, name, obj, max_chars)
        except Exception as exc:  # never let one object kill the iteration
            logger.warning("introspection: error chunking %s.%s: %s", library, name, exc)


def _class_chunks(
    library: str,
    name: str,
    cls,
    max_chars: int,
    include_private: bool,
) -> Iterator[DocChunk]:
    qualname = f"{library}.{name}"
    doc = inspect.getdoc(cls) or ""
    methods = _public_methods(cls, include_private)

    # Class chunk: signature line first, then docstring, then method index.
    head, _ok = _signature_string(cls, name)
    lines = [f"{library}.{head}"]
    if doc:
        lines.extend(["", doc])
    if methods:
        lines.extend(["", "Methods: " + ", ".join(methods)])
    text = truncate_text("\n".join(lines), max_chars)
    yield _chunk(
        library,
        version=_version_of(library),
        qualname=qualname,
        kind="class",
        text=text,
        embed_text=_embed_text(qualname, doc),
    )

    # One chunk per public method, named ``Class.method``.
    for mname in methods:
        # Prefer the live value (bound / C method); fall back to the static
        # member if live access fails.
        try:
            mobj = getattr(cls, mname)
        except Exception:
            mobj = _unwrap_method(cls, mname)
        if mobj is None:
            continue
        mdoc = inspect.getdoc(mobj) or ""
        mdisplay = f"{name}.{mname}"
        mhead, _mok = _signature_string(mobj, mdisplay)
        if not mdoc and not _has_signature_info(mobj):
            continue
        mtext = truncate_text(
            f"{library}.{mhead}\n\n{mdoc}".rstrip(),
            max_chars,
        )
        yield _chunk(
            library,
            version=_version_of(library),
            qualname=f"{library}.{mdisplay}",
            kind="method",
            text=mtext,
            embed_text=_embed_text(f"{library}.{mdisplay}", mdoc),
        )


def _callable_chunks(
    library: str,
    name: str,
    obj,
    max_chars: int,
) -> Iterator[DocChunk]:
    """One chunk for a public function or callable object.

    Plain constants (numbers, strings, ``None``, module-level instances such as
    ``datetime.MINYEAR`` / ``datetime.UTC``) are skipped: ``inspect.getdoc`` on
    an instance returns its *class* docstring, which would produce a bogus chunk.
    """
    if not (callable(obj) or inspect.isroutine(obj)):
        return
    doc = inspect.getdoc(obj) or ""
    head, ok = _signature_string(obj, name)
    # Skip objects that are neither documented nor introspectable.
    if not doc and not _has_signature_info(obj):
        return
    text = truncate_text(f"{library}.{head}\n\n{doc}".rstrip(), max_chars)
    qualname = f"{library}.{name}"
    yield _chunk(
        library,
        version=_version_of(library),
        qualname=qualname,
        kind="function",
        text=text,
        embed_text=_embed_text(qualname, doc),
    )


def _signature_string(obj, display_name: str) -> tuple[str, bool]:
    """Return ``(name(params), ok)`` where ``ok`` is False on the fallback."""
    try:
        params = str(inspect.signature(obj))
    except (ValueError, TypeError):
        return f"{display_name}(...)", False
    return _clean_signature(f"{display_name}{params}"), True


def _has_signature_info(obj) -> bool:
    try:
        inspect.signature(obj)
        return True
    except (ValueError, TypeError):
        return inspect.isroutine(obj)


def _unwrap_method(cls, mname: str):
    val = inspect.getattr_static(cls, mname)
    if isinstance(val, (classmethod, staticmethod)):
        return val.__func__
    return val


def _public_methods(cls, include_private: bool) -> list[str]:
    """Public method names on ``cls`` or inherited from a Python base class.

    For each non-dunder name in ``dir(cls)``, the *owner* is the first class
    in ``cls.__mro__`` whose own namespace holds the name. The member counts
    as a method when the owner's raw value (``vars(owner)[name]``) is a Python
    function, a built-in function, a method descriptor, a class-method
    descriptor, or a ``classmethod`` / ``staticmethod`` wrapper. Members whose
    owner lives in the ``builtins`` module (``object``, ``BaseException``,
    ``dict``, ...) are skipped unless defined on ``cls`` itself, so an error
    subclass does not re-index ``add_note`` / ``with_traceback``; methods
    inherited from ordinary Python bases (``PurePath.with_suffix`` on
    ``Path``) are kept.
    """
    methods: list[str] = []
    try:
        names = dir(cls)
        mro = cls.__mro__
    except Exception:
        return methods
    for mname in names:
        if mname.startswith("__") and mname.endswith("__"):
            continue
        if mname.startswith("_") and not include_private:
            continue
        owner = _mro_owner(mro, mname)
        if owner is None:
            continue
        if owner is not cls and getattr(owner, "__module__", None) == "builtins":
            continue
        try:
            raw = vars(owner)[mname]
        except Exception:
            continue
        if isinstance(raw, (classmethod, staticmethod)):
            raw = raw.__func__
        if isinstance(raw, _METHOD_TYPES):
            methods.append(mname)
    return methods


def _mro_owner(mro, mname: str):
    """First class in ``mro`` whose own namespace contains ``mname``."""
    for base in mro:
        if mname in vars(base):
            return base
    return None


def _public_names(module, include_private: bool) -> list[str]:
    all_ = getattr(module, "__all__", None)
    if all_ is not None:
        return [str(n) for n in all_]
    if include_private:
        return [n for n in dir(module) if not (n.startswith("__") and n.endswith("__"))]
    return [n for n in dir(module) if not n.startswith("_")]


def _resolve_version(library: str) -> str:
    try:
        return importlib.metadata.version(library)
    except Exception:
        return "stdlib"


_version_cache: dict[str, str] = {}


def _version_of(library: str) -> str:
    if library not in _version_cache:
        _version_cache[library] = _resolve_version(library)
    return _version_cache[library]


def _embed_text(dotted_name: str, doc: str) -> str:
    """Compact embedding input: ``dotted.name: first sentence of the docstring``.

    MiniLM's ~256-token window is mostly consumed by long signatures, so
    embeddings use a short, meaning-first text instead of the full chunk.
    Falls back to the dotted name alone when there is no docstring.
    """
    piece = re.split(r"\. |\n\s*\n", doc, maxsplit=1)[0]
    first_sentence = " ".join(piece.split())[:200]
    if not first_sentence:
        return dotted_name
    return f"{dotted_name}: {first_sentence}"


def _chunk(
    library: str,
    *,
    version: str,
    qualname: str,
    kind: str,
    text: str,
    embed_text: str | None = None,
) -> DocChunk:
    metadata: dict[str, MetadataValue] = {
        "library": library,
        "version": version,
        "qualname": qualname,
        "kind": kind,
        "import_path": library,
        "source_type": "introspection",
    }
    chunk_id = stable_id("introspection", library, version, kind, qualname)
    return DocChunk(id=chunk_id, text=text, metadata=metadata, embed_text=embed_text)


# --- Text-file loader ------------------------------------------------------


def iter_text_file_chunks(
    path: Path | str,
    *,
    library: str | None = None,
    max_chars: int = 800,
    overlap: int = 100,
) -> Iterator[DocChunk]:
    """Yield :class:`DocChunk` items from ``.md`` / ``.rst`` / ``.txt`` files.

    ``path`` may be a single file or a directory (recursed for the supported
    extensions). Files are decoded as ``utf-8-sig`` (a leading BOM is dropped);
    undecodable bytes fall back to replacement with a warning logged. CRLF is
    handled by :func:`split_markdown`. ``source_path`` is the forward-slash path
    RELATIVE to the given root (just the file name for a single-file input), so
    ids never embed an absolute path and stay stable across locations.
    """
    root = Path(path)
    files = _collect_files(root)
    if not files:
        if not root.exists():
            logger.warning("text loader: path does not exist: %s", root)
        return

    default_library = root.name if root.is_dir() else root.stem
    lib = library or default_library
    root_is_file = root.is_file()

    for f in files:
        try:
            data = f.read_bytes()
        except OSError as exc:
            logger.warning("text loader: cannot read %s: %s", f, exc)
            continue
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            logger.warning("text loader: undecodable bytes in %s; replacing", f)
            text = data.decode("utf-8-sig", errors="replace")

        # Root-relative forward-slash path (just the file name for a single
        # file) so ids never embed an absolute path.
        src = f.name if root_is_file else f.relative_to(root).as_posix()
        for i, piece in enumerate(split_markdown(text, max_chars=max_chars, overlap=overlap)):
            if not piece.strip():
                continue
            piece_hash = hashlib.sha1(piece.encode("utf-8")).hexdigest()[:8]
            metadata: dict[str, MetadataValue] = {
                "library": lib,
                "version": "",
                "qualname": src,
                "kind": "text",
                "import_path": "",
                "source_type": "text_file",
                "source_path": src,
            }
            chunk_id = stable_id("text_file", src, str(i), piece_hash)
            yield DocChunk(id=chunk_id, text=piece, metadata=metadata)


def _collect_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in _TEXT_SUFFIXES else []
    if not root.is_dir():
        return []
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _TEXT_SUFFIXES]
    return sorted(files, key=lambda p: p.as_posix())


__all__ = [
    "IngestReport",
    "ingest",
    "iter_introspection_chunks",
    "iter_text_file_chunks",
    "main",
    "split_markdown",
    "stable_id",
    "truncate_text",
]


if __name__ == "__main__":
    sys.exit(main())
