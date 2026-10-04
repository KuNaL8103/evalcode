"""Embedders: the ``Embedder`` protocol, the local HuggingFace embedder, and
a deterministic fake for unit tests.

Heavy imports (``langchain_huggingface`` → ``sentence_transformers`` →
``torch``) never happen at module import time: ``HFEmbedder`` resolves the
embedding class through :func:`_import_hf_embeddings_cls` (which tests can
monkeypatch) and builds the underlying model lazily on first use.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, runtime_checkable

from evalcode.config import Settings

__all__ = ["Embedder", "FakeEmbedder", "HFEmbedder", "get_embedder"]

# Documents are embedded in batches this large (bounded memory on CPU).
_EMBED_BATCH_SIZE = 64

# Word-like tokens (incl. dots/underscores so API names like "json.loads"
# stay one token) used by the deterministic fake embedder.
_TOKEN_RE = re.compile(r"[a-z0-9_.]+")


@runtime_checkable
class Embedder(Protocol):
    """Anything that can turn texts into fixed-dimension vectors."""

    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts; one ``dim``-length vector per text, in order."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        ...


def _import_hf_embeddings_cls():
    """Lazily import the langchain-huggingface embeddings class.

    Module-level indirection so unit tests can monkeypatch this factory and
    exercise ``HFEmbedder`` without the real package (or the model download).
    """
    import langchain_huggingface

    return langchain_huggingface.HuggingFaceEmbeddings


class HFEmbedder:
    """Local sentence-transformers embedder (MiniLM default: 384-d, L2-normalized).

    Construction is cheap and side-effect free; the model is loaded only on
    the first embed call. No query prefix is applied (MiniLM needs none;
    e5/bge-style prefixes would be configured here if the model changes).
    """

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._client = None  # HuggingFaceEmbeddings, built lazily
        self._dim: int | None = None

    def _ensure_client(self):
        if self._client is None:
            cls = _import_hf_embeddings_cls()
            self._client = cls(
                model_name=self.model_name,
                model_kwargs={"device": self.device},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._client

    @property
    def dim(self) -> int:
        """Vector dimensionality, from the loaded model (or a probe embedding)."""
        if self._dim is None:
            client = self._ensure_client()
            dim = _resolve_dimension(client)
            if dim is None:
                dim = len(client.embed_query(""))
            self._dim = dim
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        client = self._ensure_client()
        out: list[list[float]] = []
        for i in range(0, len(texts), _EMBED_BATCH_SIZE):
            out.extend(client.embed_documents(texts[i : i + _EMBED_BATCH_SIZE]))
        return out

    def embed_query(self, text: str) -> list[float]:
        client = self._ensure_client()
        return client.embed_query(text)


def _resolve_dimension(client) -> int | None:
    """Ask the underlying sentence-transformers model for its dimension.

    Falls back to ``None`` (caller probes with an embedding) when the client
    does not expose the accessor (e.g. a test double).
    """
    model = getattr(client, "_client", None)
    # Newer sentence-transformers renamed get_sentence_embedding_dimension ->
    # get_embedding_dimension (the old name now warns). Try the new name first.
    for name in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        getter = getattr(model, name, None)
        if callable(getter):
            try:
                return int(getter())
            except Exception:
                return None
    return None


class FakeEmbedder:
    """Deterministic, dependency-free embedder for unit tests.

    A bag-of-tokens vector: each token (lowercased word, keeping dots so
    ``json.loads`` is one token) adds +1 to the cell ``sha256(token) % dim``;
    the vector is L2-normalized. ``hashlib`` (never Python's salted ``hash``)
    makes it stable across processes, so identical texts yield identical
    vectors and texts sharing tokens yield higher cosine similarity.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            vec[int.from_bytes(digest[:8], "big") % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def get_embedder(settings: Settings) -> Embedder:
    """Production embedder factory: local HuggingFace model from settings."""
    return HFEmbedder(settings.embedding_model)
