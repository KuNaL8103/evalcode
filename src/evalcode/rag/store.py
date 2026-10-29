"""RAG data types and the Chroma-backed persistent vector store.

``DocChunk`` is the unit of retrieval; ids are deterministic (see
:func:`evalcode.rag.ingest.stable_id`) so re-ingestion is idempotent.
``RetrievedDoc`` is the JSON-safe shape stored in graph state.

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
from typing import TypedDict

from pydantic import BaseModel, Field

from evalcode.rag.embeddings import Embedder

__all__ = ["DocChunk", "MetadataValue", "RetrievedDoc", "VectorStore"]

logger = logging.getLogger(__name__)

# The metadata values we ever store. Kept as a narrow union so the vector
# store can pass them through as plain JSON scalars.
MetadataValue = str | int | float | bool


class RetrievedDoc(TypedDict):
    """One retrieved document, as it is stored in graph state.

    Plain ``TypedDict`` (JSON-serializable) so it checkpoints cleanly.
    """

    id: str
    text: str
    score: float
    library: str
    qualname: str
    import_path: str


class DocChunk(BaseModel):
    """A single chunk of documentation, sized for a MiniLM-class embedder.

    Required metadata keys (set by the loaders):

    - ``library``: top-level package/module name (e.g. ``"json"``)
    - ``version``: resolved version, or ``"stdlib"`` for standard-library
      modules
    - ``qualname``: dotted qualified name (e.g. ``"json.loads"``)
    - ``kind``: one of ``module | function | class | method | text``
    - ``import_path``: the import path a reader would use
      (e.g. ``"json"`` for ``json.loads``)
    - ``source_type``: ``"introspection"`` or ``"text_file"``
    - ``source_path``: present only for ``text_file`` chunks (forward slashes)
    """

    id: str
    text: str
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)
    # Optional shortened text used for EMBEDDING only (not stored in Chroma,
    # not part of metadata). ``text`` stays the full chunk for display/prompts.
    embed_text: str | None = None

    def is_text_file(self) -> bool:
        """True when this chunk came from a local text document."""
        return self.metadata.get("source_type") == "text_file"


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

        # Embed the compact text when present; the FULL text stays the stored
        # document (display/prompts) and the text of chunks returned by query.
        vectors = self._embedder.embed_documents(
            [(by_id[i].embed_text or by_id[i].text) for i in ids]
        )
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

    def all_chunks(self, where: dict | None = None) -> list[DocChunk]:
        """Return every stored chunk (optionally metadata-filtered).

        Used to build the in-memory BM25 index once per retriever. The dense
        path is untouched; this is a single metadata/document read.
        """
        result = self._collection.get(
            where=where,
            include=["metadatas", "documents"],
        )
        return [
            DocChunk(id=cid, text=text or "", metadata=meta or {})
            for cid, text, meta in zip(
                result["ids"],
                result["documents"] or [],
                result["metadatas"] or [],
                strict=False,
            )
        ]

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
