"""Retriever: multi-query top-k search over a :class:`VectorStore`.

Each query is embedded and searched independently; results are merged by
chunk id keeping the max score, filtered by ``min_score``, sorted by score
(descending), and truncated to ``k``.

CLI: ``python -m evalcode.rag.retriever "query" [-k 5]`` prints the results
of a search against the configured store (ASCII-safe on Windows).
"""

from __future__ import annotations

import argparse
import sys

from evalcode.rag.store import VectorStore
from evalcode.rag.types import RetrievedDoc

__all__ = ["Retriever", "main"]


def _safe(text: str) -> str:
    """ASCII-safe rendering for Windows consoles (cp1252 defaults)."""
    return text.encode("ascii", "replace").decode("ascii")


class Retriever:
    """Searches a :class:`VectorStore` with score filtering and deduplication."""

    def __init__(self, store: VectorStore, top_k: int, min_score: float) -> None:
        self.store = store
        self.top_k = top_k
        self.min_score = min_score

    def retrieve(
        self,
        queries: list[str],
        k: int | None = None,
        library: str | None = None,
    ) -> list[RetrievedDoc]:
        """Merge top-k results for each query into a single ranked list.

        A chunk reached by several queries is kept once, with the max score.
        Chunks below ``min_score`` are dropped before the final ``k`` cut.
        """
        k = k if k is not None else self.top_k
        where = {"library": library} if library is not None else None
        best: dict[str, float] = {}
        by_id: dict[str, RetrievedDoc] = {}
        for query in queries:
            for chunk, score in self.store.query(query, k=k, where=where):
                meta = chunk.metadata
                doc = RetrievedDoc(
                    id=chunk.id,
                    text=chunk.text,
                    score=score,
                    library=str(meta.get("library", "")),
                    qualname=str(meta.get("qualname", "")),
                    import_path=str(meta.get("import_path", "")),
                )
                prev = best.get(chunk.id)
                if prev is None or score > prev:
                    best[chunk.id] = score
                    by_id[chunk.id] = doc
        docs = [by_id[cid] for cid, score in best.items() if score >= self.min_score]
        docs.sort(key=lambda doc: doc["score"], reverse=True)
        return docs[:k]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Search the evalcode doc index (python -m evalcode.rag.retriever)."
    )
    parser.add_argument("query", help="search query text")
    parser.add_argument("-k", type=int, default=None, help="max results (default from settings)")
    args = parser.parse_args(argv)

    # Imported here (not at module level) so tests importing this module
    # never pay for settings/env loading.
    from evalcode.config import get_settings
    from evalcode.rag.embeddings import get_embedder

    settings = get_settings()
    embedder = get_embedder(settings)
    store = VectorStore(settings.chroma_dir, settings.collection_name, embedder)
    try:
        retriever = Retriever(store, settings.retrieval_top_k, settings.retrieval_min_score)
        results = retriever.retrieve([args.query], k=args.k)
    finally:
        store.close()

    if not results:
        print(_safe(f"no results for: {args.query} (run evalcode.rag.ingest first?)"))
        return 1
    for i, doc in enumerate(results, start=1):
        print(_safe(f"[{i}] score={doc['score']:.3f}  {doc['qualname']}"))
        print(_safe(doc["text"].replace("\n", " ")[:300]))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
