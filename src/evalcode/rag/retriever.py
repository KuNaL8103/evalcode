"""Retrievers over a :class:`VectorStore`.

``Retriever`` is the original dense-only path: each query is embedded and
searched independently; results are merged by chunk id keeping the max score,
filtered by ``min_score``, sorted by score (descending), and cut to ``k``.

``HybridRetriever`` adds a lazily built BM25 index and fuses both rankings
with Reciprocal Rank Fusion (query-adaptively for symbol-like queries).

CLI: ``python -m evalcode.rag.retriever "query" [-k 5]`` prints the results
of a search against the configured store (ASCII-safe on Windows).
"""

from __future__ import annotations

import argparse
import logging
import re
import sys

from evalcode.rag.bm25 import BM25Index
from evalcode.rag.store import DocChunk, RetrievedDoc, VectorStore

__all__ = ["HybridRetriever", "Retriever", "is_symbol_query", "main"]

logger = logging.getLogger(__name__)

# A dotted/snake_case identifier, e.g. ``json.loads`` or ``move_to_end``.
_SYMBOL_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+|[A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]+"
)
_CAMEL_RE = re.compile(r"[A-Z][A-Za-z0-9]*")


def is_symbol_query(text: str) -> bool:
    """True when a query contains an explicit code symbol.

    Dense embeddings handle natural-language intent better; BM25 earns its
    keep on identifiers. Gating lexical fusion this way avoids the
    natural-language top-1 regression measured in the benchmark while keeping
    the exact-symbol gain.
    """
    stripped = text.strip()
    if _SYMBOL_RE.search(stripped):
        return True
    return len(stripped.split()) == 1 and bool(_CAMEL_RE.fullmatch(stripped))


def _retrieved_doc(chunk: DocChunk, score: float) -> RetrievedDoc:
    meta = chunk.metadata
    return RetrievedDoc(
        id=chunk.id,
        text=chunk.text,
        score=score,
        library=str(meta.get("library", "")),
        qualname=str(meta.get("qualname", "")),
        import_path=str(meta.get("import_path", "")),
    )


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
                doc = _retrieved_doc(chunk, score)
                prev = best.get(chunk.id)
                if prev is None or score > prev:
                    best[chunk.id] = score
                    by_id[chunk.id] = doc
        docs = [by_id[cid] for cid, score in best.items() if score >= self.min_score]
        docs.sort(key=lambda doc: doc["score"], reverse=True)
        return docs[:k]


class HybridRetriever(Retriever):
    """Dense MiniLM + BM25 retrieval fused with Reciprocal Rank Fusion.

    RRF is rank-based, so the two scorers never need comparable magnitudes::

        fused(doc) = sum(1 / (rrf_k + rank_dense)) + sum(1 / (rrf_k + rank_bm25))

    The public ``score`` field is the fused RRF score (a small number around
    ``1/60``); it is used only for ordering and observability. ``min_score``
    still applies to the dense cosine scores, while BM25-only hits are kept —
    that is what makes exact symbols retrievable when embeddings blur them.

    With ``adaptive=True`` (default), the lexical ranking is only fused when
    the query contains an explicit symbol (``json.loads``, ``move_to_end``,
    ``Counter``). The benchmark showed that fusing BM25 into natural-language
    queries can demote the correct doc, so the dense path is left untouched
    for prose queries.

    The BM25 index is built lazily once from the store's chunks (not per
    query), so steady-state latency stays in-memory. If ``rank_bm25`` is
    missing, retrieval degrades cleanly to the dense parent implementation.
    """

    def __init__(
        self,
        store: VectorStore,
        top_k: int,
        min_score: float,
        *,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
        adaptive: bool = True,
        index_factory: type[BM25Index] = BM25Index,
    ) -> None:
        super().__init__(store, top_k, min_score)
        self.rrf_k = max(1, int(rrf_k))
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.adaptive = adaptive
        self._index_factory = index_factory
        self._index: BM25Index | None = None
        self._lexical_enabled = True

    def _lexical_index(self) -> BM25Index:
        if self._index is None:
            self._index = self._index_factory(self.store.all_chunks())
        return self._index

    def refresh(self) -> None:
        """Drop the cached BM25 index (call after re-ingesting)."""
        self._index = None

    def retrieve(
        self,
        queries: list[str],
        k: int | None = None,
        library: str | None = None,
    ) -> list[RetrievedDoc]:
        if not self._lexical_enabled:
            return super().retrieve(queries, k=k, library=library)

        k = k if k is not None else self.top_k
        where = {"library": library} if library is not None else None
        try:
            index = self._lexical_index()
        except ImportError:
            logger.warning("rank_bm25 is not installed; hybrid retrieval degraded to dense")
            self._lexical_enabled = False
            return super().retrieve(queries, k=k, library=library)

        fused: dict[str, float] = {}
        dense_best: dict[str, float] = {}
        by_id: dict[str, RetrievedDoc] = {}
        for query in queries:
            dense_hits = [
                (chunk, score)
                for chunk, score in self.store.query(query, k=k, where=where)
                if score >= self.min_score
            ]
            use_lexical = not self.adaptive or is_symbol_query(query)
            lexical_hits = index.search(query, k=k, where=where) if use_lexical else []
            for rank, (chunk, _score) in enumerate(dense_hits, start=1):
                fused[chunk.id] = fused.get(chunk.id, 0.0) + self.dense_weight / (self.rrf_k + rank)
                by_id.setdefault(chunk.id, _retrieved_doc(chunk, 0.0))
            for rank, (chunk, _score) in enumerate(lexical_hits, start=1):
                fused[chunk.id] = fused.get(chunk.id, 0.0) + self.bm25_weight / (self.rrf_k + rank)
                by_id.setdefault(chunk.id, _retrieved_doc(chunk, 0.0))
            for chunk, score in dense_hits:
                dense_best[chunk.id] = max(dense_best.get(chunk.id, score), score)

        ranked_ids = sorted(
            fused,
            key=lambda cid: (fused[cid], dense_best.get(cid, 0.0)),
            reverse=True,
        )
        docs: list[RetrievedDoc] = []
        for cid in ranked_ids[:k]:
            doc = by_id[cid]
            doc["score"] = fused[cid]
            docs.append(doc)
        return docs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Search the evalcode doc index (python -m evalcode.rag.retriever)."
    )
    parser.add_argument("query", help="search query text")
    parser.add_argument("-k", type=int, default=None, help="max results (default from settings)")
    args = parser.parse_args(argv)

    # Imported here (not at module level) so tests importing this module
    # never pay for settings/env loading.
    from evalcode.core.config import get_settings
    from evalcode.rag.embeddings import get_embedder

    settings = get_settings()
    embedder = get_embedder(settings)
    store = VectorStore(settings.chroma_dir, settings.collection_name, embedder)
    try:
        if settings.hybrid_retrieval:
            retriever = HybridRetriever(
                store,
                settings.retrieval_top_k,
                settings.retrieval_min_score,
                rrf_k=settings.rrf_k,
                adaptive=settings.hybrid_adaptive,
            )
        else:
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
