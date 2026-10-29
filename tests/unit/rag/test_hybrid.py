"""Hybrid retrieval: tokenizer, BM25 ranking, RRF fusion, and wiring."""

from __future__ import annotations

import shutil
from pathlib import Path

from evalcode.core.config import Settings
from evalcode.rag.bm25 import BM25Index, tokenize
from evalcode.rag.embeddings import FakeEmbedder
from evalcode.rag.retriever import HybridRetriever, Retriever, is_symbol_query
from evalcode.rag.store import DocChunk, VectorStore


def _chunk(
    id: str,
    text: str,
    *,
    qualname: str,
    library: str = "json",
) -> DocChunk:
    return DocChunk(
        id=id,
        text=text,
        metadata={
            "library": library,
            "version": "1",
            "qualname": qualname,
            "kind": "function",
            "import_path": library,
            "source_type": "introspection",
        },
    )


class _RankedStore:
    """Tiny dense store double: fixed ranking per query + all_chunks()."""

    def __init__(self, chunks: list[DocChunk], scores: dict[str, float]) -> None:
        self._chunks = {c.id: c for c in chunks}
        self._scores = scores

    def query(self, text: str, k: int, where: dict | None = None):
        hits = []
        for cid, score in sorted(self._scores.items(), key=lambda item: item[1], reverse=True):
            chunk = self._chunks[cid]
            if where and any(chunk.metadata.get(key) != value for key, value in where.items()):
                continue
            hits.append((chunk, score))
        return hits[:k]

    def all_chunks(self) -> list[DocChunk]:
        return list(self._chunks.values())


def test_symbol_query_detection() -> None:
    assert is_symbol_query("json.loads")
    assert is_symbol_query("collections.OrderedDict.move_to_end")
    assert is_symbol_query("move_to_end")
    assert is_symbol_query("Counter")
    assert not is_symbol_query("replace all matches of a pattern")
    assert not is_symbol_query("parse a JSON string")


def test_natural_language_query_stays_dense_only() -> None:
    chunks = [
        _chunk("dense", "parse a json string", qualname="json.parse"),
        _chunk("lexical", "json.loads", qualname="json.loads"),
    ]
    # Dense puts json.parse first; BM25 would love json.loads for the token
    # "json". Adaptive fusion must ignore BM25 on a prose query.
    store = _RankedStore(chunks, {"dense": 0.9, "lexical": 0.05})
    retriever = HybridRetriever(store, top_k=3, min_score=0.2)

    results = retriever.retrieve(["parse a JSON string from text"])
    assert [doc["id"] for doc in results] == ["dense"]


def test_tokenize_keeps_code_symbols_and_parts() -> None:
    tokens = tokenize("json.loads and OrderedDict.move_to_end")
    assert "json.loads" in tokens
    assert {"json", "loads"} <= set(tokens)
    assert "ordereddict.move_to_end" in tokens
    assert "move_to_end" in tokens


def test_bm25_exact_symbol_wins() -> None:
    chunks = [
        _chunk("loads", "Deserialize a JSON string into an object.", qualname="json.loads"),
        _chunk("dump", "Serialize an object to a JSON string.", qualname="json.dumps"),
        _chunk("other", "Completely unrelated zebra text.", qualname="csv.reader", library="csv"),
    ]
    index = BM25Index(chunks)
    hits = index.search("json.loads", k=3)
    assert hits, "expected lexical hits"
    assert hits[0][0].id == "loads"
    assert all(score > 0 for _, score in hits)


def test_bm25_respects_metadata_filter() -> None:
    chunks = [
        _chunk("j", "parse json with loads", qualname="json.loads", library="json"),
        _chunk("r", "parse json with loads", qualname="re.loads", library="re"),
    ]
    index = BM25Index(chunks)
    hits = index.search("json.loads", k=5, where={"library": "re"})
    assert [chunk.id for chunk, _ in hits] == ["r"]


def test_rrf_prefers_documents_found_by_both_rankers() -> None:
    chunks = [
        _chunk("both", "json.loads parse a json string", qualname="json.loads"),
        _chunk("dense_only", "parse a json string", qualname="json.parse"),
        _chunk("lex_only", "json loads", qualname="json.loads_alias"),
        _chunk("low", "unrelated text", qualname="csv.unrelated", library="csv"),
    ]
    store = _RankedStore(chunks, {"dense_only": 0.9, "both": 0.8, "lex_only": 0.02, "low": 0.01})
    retriever = HybridRetriever(store, top_k=4, min_score=0.2, rrf_k=60)

    results = retriever.retrieve(["json.loads"])

    assert results[0]["id"] == "both"
    # A BM25-only hit is retained even though its dense score is below min_score.
    assert "lex_only" in [doc["id"] for doc in results]
    # Dense floor still removes a document that neither ranker wants.
    assert "low" not in [doc["id"] for doc in results]
    # The reported score is the fused RRF value, not a cosine similarity.
    assert 0 < results[0]["score"] <= 2 / 60


def test_rrf_honors_library_filter_and_k() -> None:
    chunks = [
        _chunk("a", "json.loads parse json", qualname="json.loads", library="json"),
        _chunk("b", "json.loads parse json", qualname="re.loads", library="re"),
    ]
    store = _RankedStore(chunks, {"a": 0.9, "b": 0.8})
    retriever = HybridRetriever(store, top_k=5, min_score=0.0)

    assert [d["id"] for d in retriever.retrieve(["json.loads"], library="re")] == ["b"]
    assert len(retriever.retrieve(["json.loads"], k=1)) == 1


def test_hybrid_degrades_to_dense_without_rank_bm25() -> None:
    class _MissingIndex:
        def __init__(self, _chunks) -> None:
            raise ImportError("rank_bm25 missing")

    chunks = [_chunk("a", "parse json", qualname="json.loads")]
    store = _RankedStore(chunks, {"a": 0.9})
    retriever = HybridRetriever(store, top_k=3, min_score=0.1, index_factory=_MissingIndex)

    results = retriever.retrieve(["parse json"])
    assert [d["id"] for d in results] == ["a"]
    assert retriever._lexical_enabled is False


def test_build_retriever_selects_hybrid_when_enabled(tmp_path: Path) -> None:
    from evalcode.graph import build_retriever

    chunk = _chunk("c1", "parse json", qualname="json.loads")

    def _populate(name: str) -> Path:
        directory = tmp_path / name
        directory.mkdir()
        embedder = FakeEmbedder(dim=64)
        store = VectorStore(directory, "test", embedder)
        store.upsert([chunk])
        store.close()
        return directory

    hybrid_dir = _populate("hybrid")
    hybrid = build_retriever(
        Settings(chroma_dir=str(hybrid_dir), collection_name="test", hybrid_retrieval=True),
        embedder=FakeEmbedder(dim=64),
    )
    try:
        assert isinstance(hybrid, HybridRetriever)
        assert hybrid.adaptive is True
    finally:
        hybrid.store.close()
        shutil.rmtree(hybrid_dir)

    dense_dir = _populate("dense")
    dense = build_retriever(
        Settings(chroma_dir=str(dense_dir), collection_name="test", hybrid_retrieval=False),
        embedder=FakeEmbedder(dim=64),
    )
    try:
        assert type(dense) is Retriever
    finally:
        dense.store.close()
        shutil.rmtree(dense_dir)
