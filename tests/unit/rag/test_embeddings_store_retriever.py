"""Task 3 tests: embedders, Chroma VectorStore, Retriever, and ingest.

Unit tests use ``FakeEmbedder`` plus the real embedded Chroma client on a
tmp dir (chromadb is a required runtime dep; only the torch-family stack is
excluded here, and only the ``slow`` test loads the real embedding model).
Every store is closed so tmp dirs can be cleaned on Windows.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

import evalcode.rag.embeddings as embeddings_mod
from evalcode.config import get_settings
from evalcode.rag.embeddings import FakeEmbedder, HFEmbedder, get_embedder
from evalcode.rag.ingest import ingest
from evalcode.rag.retriever import Retriever
from evalcode.rag.store import VectorStore
from evalcode.rag.types import DocChunk


def _chunk(id: str, text: str, library: str) -> DocChunk:
    return DocChunk(
        id=id,
        text=text,
        metadata={
            "library": library,
            "version": "1",
            "qualname": f"{library}.{id}",
            "kind": "function",
            "import_path": library,
            "source_type": "introspection",
        },
    )


def _make_store(tmp_path: Path, dim: int = 64) -> VectorStore:
    return VectorStore(tmp_path / "chroma", "python_docs", FakeEmbedder(dim=dim))


# --- FakeEmbedder ----------------------------------------------------------


def test_fake_embedder_deterministic_dim_and_normalized() -> None:
    e = FakeEmbedder(dim=32)
    text = "parse json string into python object"

    a = e.embed_query(text)
    b = e.embed_documents([text])[0]
    c = FakeEmbedder(dim=32).embed_query(text)  # a different instance
    assert a == b == c  # deterministic, including across processes (hashlib)
    assert len(a) == 32

    # L2-normalized (non-empty input).
    assert math.isclose(math.sqrt(sum(v * v for v in a)), 1.0, rel_tol=1e-9)

    # Shared tokens → higher cosine similarity than disjoint tokens.
    def sim(u: list[float], v: list[float]) -> float:
        return sum(x * y for x, y in zip(u, v, strict=True))

    base = e.embed_query("parse json string into object")
    close = e.embed_query("parse json string into a python object")
    far = e.embed_query("zebra quartet mangrove saxophone")
    assert sim(base, close) > sim(base, far)

    # Empty text → zero vector (still the right dimension).
    assert e.embed_query("") == [0.0] * 32


# --- VectorStore -----------------------------------------------------------


def test_store_upsert_idempotent(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    try:
        chunks = [_chunk("a", "alpha body text", "lib"), _chunk("b", "beta body text", "lib")]
        assert store.upsert(chunks) == 2
        assert store.count() == 2
        # Re-upserting the same ids replaces them: count unchanged.
        assert store.upsert(chunks) == 2
        assert store.count() == 2
        # Duplicate ids inside ONE batch: last one wins, ids deduplicated.
        dup = [
            _chunk("a", "alpha old text", "lib"),
            _chunk("b", "beta body text", "lib"),
            _chunk("a", "alpha new text", "lib"),
        ]
        assert store.upsert(dup) == 2
        assert store.count() == 2
        hits = store.query("alpha new text", k=1)
        assert hits and hits[0][0].id == "a"
        assert hits[0][0].text == "alpha new text"
    finally:
        store.close()


def test_store_query_returns_exact_text_first(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    target = _chunk("target", "parse json string into a python object", "json")
    others = [
        _chunk("o1", "completely unrelated zebra quartet", "json"),
        _chunk("o2", "another different mangrove document", "json"),
    ]
    try:
        store.upsert([target, *others])
        hits = store.query("parse json string into a python object", k=3)
        assert hits[0][0].id == "target"
        assert hits[0][1] > 0.99  # identical embedding → cosine distance ~0
        # Results are sorted by score, descending.
        assert [h[1] for h in hits] == sorted([h[1] for h in hits], reverse=True)
        # Every hit carries the chunk text and metadata back.
        assert hits[0][0].metadata["library"] == "json"
    finally:
        store.close()


def test_store_query_where_library_filter(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    lib_a = [_chunk(f"a{i}", f"alpha document number {i} parse json", "lib_a") for i in range(3)]
    lib_b = [
        _chunk(f"b{i}", f"beta document number {i} replace pattern", "lib_b") for i in range(2)
    ]
    try:
        store.upsert(lib_a + lib_b)
        assert store.count() == 5
        hits = store.query("parse json document", k=3, where={"library": "lib_a"})
        assert len(hits) == 3
        assert {c.metadata["library"] for c, _ in hits} == {"lib_a"}
        assert {c.id for c, _ in hits} == {"a0", "a1", "a2"}
    finally:
        store.close()


def test_store_reset_empties(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    try:
        store.upsert(
            [
                _chunk("a", "alpha body text", "lib"),
                _chunk("b", "beta body text", "lib"),
                _chunk("c", "gamma body text", "lib"),
            ]
        )
        assert store.count() == 3
        store.reset()
        assert store.count() == 0
        # The recreated collection is usable immediately.
        assert store.upsert([_chunk("d", "delta body text", "lib")]) == 1
        assert store.count() == 1
        assert store.query("delta body text", k=1)[0][0].id == "d"
    finally:
        store.close()


def test_store_persists_across_instances(tmp_path: Path) -> None:
    dir_ = tmp_path / "chroma"
    chunks = [
        _chunk("a", "alpha body text parse json", "lib"),
        _chunk("b", "beta body text replace pattern", "lib"),
    ]
    s1 = VectorStore(dir_, "python_docs", FakeEmbedder(dim=64))
    s1.upsert(chunks)
    s1.close()  # release the SQLite file handle before re-opening (Windows)

    s2 = VectorStore(dir_, "python_docs", FakeEmbedder(dim=64))
    try:
        assert s2.count() == 2
        hits = s2.query("alpha body text parse json", k=1)
        assert hits[0][0].id == "a"
        assert s2.libraries() == {"lib": 2}
    finally:
        s2.close()


# --- Retriever -------------------------------------------------------------


def test_retriever_min_score_dedupe_merge_and_k(tmp_path: Path) -> None:
    # High dim: FakeEmbedder is a bag-of-tokens hash, so at dim=64 disjoint
    # texts collide into shared cells and get spurious positive scores.
    store = _make_store(tmp_path, dim=2048)
    alpha = _chunk("alpha", "parse json string into python object json loads", "json")
    # beta shares several tokens with q2 (survives min_score); gamma shares
    # none with either query (score ~0, dropped).
    beta = _chunk("beta", "parse python object from json module with re", "re")
    gamma = _chunk("gamma", "completely unrelated zebra quartet mangrove", "csv")
    try:
        store.upsert([alpha, beta, gamma])
        retriever = Retriever(store, top_k=3, min_score=0.1)

        q1 = "parse json string into object"
        q2 = "parse json string into python object"
        results = retriever.retrieve([q1, q2])

        # Dedup: alpha is reachable from BOTH queries but appears once, with
        # the max of the two scores.
        alphas = [r for r in results if r["id"] == "alpha"]
        assert len(alphas) == 1
        s1 = {c.id: s for c, s in store.query(q1, k=3)}["alpha"]
        s2 = {c.id: s for c, s in store.query(q2, k=3)}["alpha"]
        assert math.isclose(alphas[0]["score"], max(s1, s2), abs_tol=1e-9)
        assert alphas[0]["qualname"] == "json.alpha"
        assert alphas[0]["import_path"] == "json"

        # min_score drops the disjoint chunk (cosine sim 0 → score ~0).
        assert {r["id"] for r in results} == {"alpha", "beta"}

        # k truncation applies after the merge.
        assert len(retriever.retrieve([q1, q2], k=1)) == 1
        # library filter is honored.
        assert {r["id"] for r in retriever.retrieve([q1, q2], library="json")} == {"alpha"}
        # Default k is the constructor top_k.
        assert len(retriever.retrieve([q1])) <= 3
    finally:
        store.close()


# --- ingest ----------------------------------------------------------------


def test_ingest_report_on_tmp_package(tmp_path: Path) -> None:
    pkg = tmp_path / "rngpkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        'def sample(x):\n    """Sample doc."""\n    return x\n\n\n'
        'def other(y):\n    """Other doc."""\n    return y\n',
        encoding="utf-8",
    )
    sys.path.insert(0, str(tmp_path))
    store = _make_store(tmp_path)
    try:
        report = ingest(["rngpkg", "no_such_library_xyz"], store, docs_dir=None, reset=True)
        assert report["per_library"]["rngpkg"] == 2
        assert report["total"] == 2
        assert report["skipped"] == ["no_such_library_xyz"]
        assert report["seconds"] >= 0
        assert store.count() == report["total"]
        # Re-ingest is idempotent (deterministic ids).
        report2 = ingest(["rngpkg"], store, docs_dir=None, reset=False)
        assert report2["total"] == 2
        assert store.count() == 2
    finally:
        store.close()
        sys.path.remove(str(tmp_path))
        sys.modules.pop("rngpkg", None)


def test_ingest_docs_dir(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# Alpha\n\nAlpha body text about parsing json.\n", encoding="utf-8")
    (docs / "b.md").write_text("# Beta\n\nBeta body text about patterns.\n", encoding="utf-8")
    store = _make_store(tmp_path)
    try:
        report = ingest([], store, docs_dir=docs, reset=False)
        # A docs DIRECTORY is keyed under its directory name.
        assert report["per_library"]["docs"] == 2
        assert report["total"] == 2
        assert report["skipped"] == []
        assert store.count() == 2
        hits = store.query("Alpha body text about parsing json", k=1)
        assert hits[0][0].metadata["source_type"] == "text_file"
    finally:
        store.close()


# --- HFEmbedder wiring (no real model, no real langchain_huggingface) ------


def test_hf_embedder_wiring(monkeypatch: pytest.MonkeyPatch) -> None:
    init_calls: list[dict[str, object]] = []
    batch_sizes: list[int] = []

    class _FakeHF:
        def __init__(self, model_name=None, model_kwargs=None, encode_kwargs=None, **_kw):
            init_calls.append(
                {
                    "model_name": model_name,
                    "model_kwargs": model_kwargs,
                    "encode_kwargs": encode_kwargs,
                }
            )

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            batch_sizes.append(len(texts))
            return [[0.5, 0.5] for _ in texts]

        def embed_query(self, text: str) -> list[float]:
            return [0.5, 0.5]

    monkeypatch.setattr(embeddings_mod, "_import_hf_embeddings_cls", lambda: _FakeHF)

    emb = HFEmbedder("test-model")
    out = emb.embed_documents([f"text {i}" for i in range(65)])

    assert len(out) == 65
    assert batch_sizes == [64, 1]  # batched at 64
    assert init_calls == [
        {
            "model_name": "test-model",
            "model_kwargs": {"device": "cpu"},
            "encode_kwargs": {"normalize_embeddings": True},
        }
    ]
    assert init_calls and emb.dim == 2  # probe embedding (no _client accessor)
    assert get_embedder(get_settings()).model_name == get_settings().embedding_model


# --- slow: real embedding model -------------------------------------------


@pytest.mark.slow
def test_real_embedder_ingest_and_retrieve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end with the real MiniLM model (downloads on first run).

    Three libraries are ingested so retrieval competes across libraries.
    With all-MiniLM-L6-v2 the observed ranks on a json+re+collections corpus
    are json.loads #7 and re.sub #8 for the queries below (queries are NOT
    tuned), so the assertions are failing-safe at top-10: any serious
    degradation (wrong embeddings, lost metadata, empty store) drops the
    targets out of the top 10.
    """
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    settings = get_settings()
    embedder = get_embedder(settings)
    assert embedder.dim == 384

    store = VectorStore(settings.chroma_dir, settings.collection_name, embedder)
    try:
        report = ingest(["json", "re", "collections"], store, docs_dir=None, reset=True)
        assert set(report["per_library"]) == {"json", "re", "collections"}
        assert report["skipped"] == []
        assert report["total"] == sum(report["per_library"].values())

        retriever = Retriever(store, settings.retrieval_top_k, min_score=0.0)

        results = retriever.retrieve(["parse a JSON string into a Python object"], k=10)
        assert results, "expected at least one result"
        assert "json.loads" in [r["qualname"] for r in results], [r["qualname"] for r in results]
        # The best match for a JSON-parse query is a json chunk, comfortably
        # above the configured retrieval floor.
        assert results[0]["library"] == "json"
        assert results[0]["score"] >= settings.retrieval_min_score

        results2 = retriever.retrieve(["replace all matches of a pattern"], k=10)
        assert "re.sub" in [r["qualname"] for r in results2], [r["qualname"] for r in results2]
    finally:
        store.close()
