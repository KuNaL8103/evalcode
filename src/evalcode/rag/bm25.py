"""Lightweight lexical index over documentation chunks.

Dense MiniLM embeddings are strong at intent ("parse a JSON string") but can
blur exact API symbols (``json.loads`` vs ``json.dump``). BM25 is the mirror
image: it nails rare identifier tokens and ignores paraphrase. We keep both
rankings and fuse them with Reciprocal Rank Fusion in
:mod:`evalcode.rag.retriever`.

``rank_bm25`` is imported lazily so importing this module stays cheap.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable

from evalcode.rag.store import DocChunk

__all__ = ["BM25Index", "tokenize"]

logger = logging.getLogger(__name__)

# Keep dots/underscores inside a token so ``json.loads`` and ``move_to_end``
# survive tokenization; also emit their parts so "loads" still matches.
_TOKEN_RE = re.compile(r"[a-z0-9_]+(?:\.[a-z0-9_]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase code-aware tokens (full dotted names plus their parts)."""
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text.lower()):
        token = match.group(0)
        tokens.append(token)
        if "." in token:
            tokens.extend(part for part in token.split(".") if part)
    return tokens


def _document(chunk: DocChunk) -> str:
    """Index the symbol name as well as the prose/signature."""
    qualname = str(chunk.metadata.get("qualname", ""))
    import_path = str(chunk.metadata.get("import_path", ""))
    return f"{qualname} {import_path} {chunk.text}"


class BM25Index:
    """In-memory BM25 index over a snapshot of chunks.

    Uses ``BM25Plus`` (a non-negative BM25 variant) so tiny corpora and very
    common terms still produce stable, comparable scores.
    """

    def __init__(self, chunks: Iterable[DocChunk]) -> None:
        from rank_bm25 import BM25Plus  # lazy: keeps module import light

        self._chunks: list[DocChunk] = list(chunks)
        self._corpus: list[list[str]] = [tokenize(_document(c)) for c in self._chunks]
        self._token_sets: list[set[str]] = [set(tokens) for tokens in self._corpus]
        self._bm25 = BM25Plus(self._corpus) if self._corpus else None

    def __len__(self) -> int:
        return len(self._chunks)

    def search(
        self,
        query: str,
        k: int,
        where: dict | None = None,
    ) -> list[tuple[DocChunk, float]]:
        """Top-``k`` chunks by BM25 score, honoring a metadata filter."""
        if self._bm25 is None or k <= 0:
            return []
        terms = tokenize(query)
        if not terms:
            return []
        scores = self._bm25.get_scores(terms)
        term_set = set(terms)
        ranked: list[tuple[DocChunk, float]] = []
        for chunk, token_set, score in zip(self._chunks, self._token_sets, scores, strict=True):
            # BM25Plus adds a delta floor even with zero term overlap; require
            # at least one real query token so lexical retrieval never returns
            # unrelated chunks on a positive baseline.
            if not term_set & token_set:
                continue
            if where is not None and any(
                chunk.metadata.get(key) != value for key, value in where.items()
            ):
                continue
            if score > 0.0:
                ranked.append((chunk, float(score)))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:k]
