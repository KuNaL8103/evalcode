"""Core RAG types.

A ``DocChunk`` is the unit of retrieval: a bounded piece of text plus
structured metadata. IDs are deterministic (see ``chunking.stable_id``) so
re-ingestion is idempotent against the vector store.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# The metadata values we ever store. Kept as a narrow union so the vector
# store can pass them through as plain JSON scalars.
MetadataValue = str | int | float | bool


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

    def is_text_file(self) -> bool:
        """True when this chunk came from a local text document."""
        return self.metadata.get("source_type") == "text_file"
