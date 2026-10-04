"""Documentation loaders: turn installed libraries and text files into chunks.

Two generators feed the RAG corpus:

- ``iter_introspection_chunks`` — walk an installed library's public namespace
  and emit one :class:`~evalcode.rag.types.DocChunk` per function / class /
  method, sized for a MiniLM-class embedder (signature first, ≤ ``max_chars``).
- ``iter_text_file_chunks`` — read local ``.md`` / ``.rst`` / ``.txt`` files
  (or a directory of them) and split them with :func:`split_markdown`.

Both are pure-ish, offline, and defensive: a single broken object or an
undecodable byte sequence is logged and skipped, never fatal.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import logging
from collections.abc import Iterator
from pathlib import Path

from evalcode.rag.chunking import split_markdown, stable_id, truncate_text
from evalcode.rag.types import DocChunk, MetadataValue

__all__ = ["iter_introspection_chunks", "iter_text_file_chunks"]

logger = logging.getLogger(__name__)

_TEXT_SUFFIXES = {".md", ".rst", ".txt"}


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
    )

    # One chunk per public method, named ``Class.method``.
    for mname in methods:
        try:
            mobj = _unwrap_method(cls, mname)
        except Exception:
            continue
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
    yield _chunk(
        library,
        version=_version_of(library),
        qualname=f"{library}.{name}",
        kind="function",
        text=text,
    )


def _signature_string(obj, display_name: str) -> tuple[str, bool]:
    """Return ``(name(params), ok)`` where ``ok`` is False on the fallback."""
    try:
        params = str(inspect.signature(obj))
    except (ValueError, TypeError):
        return f"{display_name}(...)", False
    return f"{display_name}{params}", True


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
    methods: list[str] = []
    for mname in dir(cls):
        if mname.startswith("__") and mname.endswith("__"):
            continue
        if mname.startswith("_") and not include_private:
            continue
        val = _unwrap_method(cls, mname)
        if inspect.isfunction(val):
            methods.append(mname)
    return methods


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


def _chunk(
    library: str,
    *,
    version: str,
    qualname: str,
    kind: str,
    text: str,
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
    return DocChunk(id=chunk_id, text=text, metadata=metadata)


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
