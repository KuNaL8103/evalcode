"""Chroma-backed persistent vector store for :class:`DocChunk` items.

``chromadb`` is imported inside :meth:`VectorStore.__init__` (never at module
level) so that merely importing this module stays cheap; unit tests that
exercise the store use the real embedded client on a tmp dir.

Notes for the installed chromadb 1.5.x:

- telemetry was removed in 1.x, so there is no ``anonymized_telemetry``
  keyword to disable (nothing is sent anyway);
- the embedding function is intentionally omitted (``embedding_function=None``)
  because all vectors are supplied explicitly by our :class:`Embedder`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from evalcode.rag.embeddings import Embedder
from evalcode.rag.types import DocChunk

__all__ = ["VectorStore"]

logger = logging.getLogger(__name__)

# Fallback upsert batch size when the client does not expose a limit.
_DEFAULT_MAX_BATCH_SIZE = 500


class VectorStore:
    """Embed and persist chunks in a Chroma ``PersistentClient`` collection.

    The collection uses cosine space and ids double as the idempotency key:
    re-upserting a chunk id replaces that entry and leaves the count
    unchanged.
    """

    def __init__(self, persist_dir: Path | str, collection_name: str, embedder: Embedder) -> None:
        import chromadb  # lazy heavy import (see module docstring)

        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self._embedder = embedder
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,  # vectors are always supplied explicitly
        )
        self._max_batch_size = self._max_batch_size_of(self._client)

    @staticmethod
    def _max_batch_size_of(client) -> int:
        try:
            return int(client.get_max_batch_size())
        except Exception:
            return _DEFAULT_MAX_BATCH_SIZE

    # --- write -------------------------------------------------------------

    def upsert(self, chunks: list[DocChunk]) -> int:
        """Embed and upsert ``chunks``; returns the number of distinct ids written.

        Ids are deduplicated within the batch (last occurrence wins) and the
        batch is split at the client's max batch size. Upserting ids that are
        already in the store replaces them without growing the collection.
        """
        by_id: dict[str, DocChunk] = {}
        for chunk in chunks:
            by_id[chunk.id] = chunk  # last one wins
        ids = list(by_id)
        if not ids:
            return 0

        vectors = self._embedder.embed_documents([by_id[i].text for i in ids])
        for i in range(0, len(ids), self._max_batch_size):
            batch = ids[i : i + self._max_batch_size]
            self._collection.upsert(
                ids=batch,
                embeddings=vectors[i : i + self._max_batch_size],
                documents=[by_id[id].text for id in batch],
                metadatas=[by_id[id].metadata for id in batch],
            )
        return len(ids)

    # --- read --------------------------------------------------------------

    def query(
        self,
        text: str,
        k: int,
        where: dict | None = None,
    ) -> list[tuple[DocChunk, float]]:
        """Top-``k`` chunks by cosine similarity for one embedded query.

        ``where`` is a Chroma metadata filter (e.g. ``{"library": "json"}``).
        Scores are ``1 - cosine_distance`` in ``[-1, 1]``; the exact-text
        chunk (identical embedding) scores ~1.0 and comes first.
        """
        total = self.count()
        if total == 0 or k <= 0:
            return []
        n = min(k, total)
        embedding = self._embedder.embed_query(text)
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=n,
            where=where,
            include=["metadatas", "documents", "distances"],
        )
        out: list[tuple[DocChunk, float]] = []
        ids = result["ids"][0]
        for j, cid in enumerate(ids):
            chunk = DocChunk(
                id=cid,
                text=result["documents"][0][j],
                metadata=result["metadatas"][0][j] or {},
            )
            out.append((chunk, 1.0 - result["distances"][0][j]))
        return out

    def count(self) -> int:
        """Number of chunks in the collection."""
        return self._collection.count()

    def libraries(self) -> dict[str, int]:
        """Chunk counts grouped by the ``library`` metadata value."""
        result = self._collection.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for meta in result["metadatas"] or []:
            library = str((meta or {}).get("library", ""))
            counts[library] = counts.get(library, 0) + 1
        return counts

    # --- lifecycle ---------------------------------------------------------

    def reset(self) -> None:
        """Drop the collection and start empty (the upsert path recreates it)."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    def close(self) -> None:
        """Release the Chroma client (required on Windows before deleting the dir)."""
        self._client.close()
