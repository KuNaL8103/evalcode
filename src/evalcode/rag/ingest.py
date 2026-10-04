"""Ingestion: turn configured libraries (and an optional docs dir) into a
persisted, searchable :class:`VectorStore` index.

CLI: ``python -m evalcode.rag.ingest --libs json,re --docs-dir PATH --reset``
(libs and docs dir default to the configured settings).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import TypedDict

from evalcode.rag.loaders import iter_introspection_chunks, iter_text_file_chunks
from evalcode.rag.store import VectorStore
from evalcode.rag.types import DocChunk

__all__ = ["IngestReport", "ingest", "main"]

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
    from evalcode.config import get_settings
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


if __name__ == "__main__":
    sys.exit(main())
