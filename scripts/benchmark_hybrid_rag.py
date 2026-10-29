"""Dense-only vs Hybrid (BM25 + RRF) retrieval benchmark.

Uses the real local MiniLM embedder and a throwaway Chroma index. For every
fixed query it records the rank of the expected API symbol, then reports
hit@1 / hit@5 / hit@10, MRR, and per-query latency (mean + p95). Queries are
split into natural-language intent and exact-symbol lookups so the trade-off
is visible instead of averaged away.

Usage:
    python scripts/benchmark_hybrid_rag.py
    python scripts/benchmark_hybrid_rag.py --libs json,re,collections --repeat 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

# (kind, query, expected qualname). Fixed BEFORE the run; do not tune to results.
QUERIES: list[tuple[str, str, str]] = [
    # Natural-language intent
    ("natural", "parse a JSON string into a Python object", "json.loads"),
    ("natural", "serialize a Python object to a JSON string", "json.dumps"),
    ("natural", "write an object as JSON to a file", "json.dump"),
    ("natural", "replace all matches of a pattern", "re.sub"),
    ("natural", "find all non-overlapping matches of a regex in a string", "re.findall"),
    ("natural", "split a string by a regular expression", "re.split"),
    ("natural", "count occurrences of items in an iterable", "collections.Counter"),
    ("natural", "double-ended queue with fast appends and pops", "collections.deque"),
    (
        "natural",
        "move a key to the end of an ordered dictionary",
        "collections.OrderedDict.move_to_end",
    ),
    (
        "natural",
        "dictionary with a default value factory for missing keys",
        "collections.defaultdict",
    ),
    # Exact-symbol lookups (the case dense embeddings blur)
    ("exact", "json.loads", "json.loads"),
    ("exact", "json.dumps", "json.dumps"),
    ("exact", "re.findall", "re.findall"),
    ("exact", "re.sub", "re.sub"),
    ("exact", "collections.Counter", "collections.Counter"),
    ("exact", "collections.defaultdict", "collections.defaultdict"),
    ("exact", "collections.OrderedDict.move_to_end", "collections.OrderedDict.move_to_end"),
    ("exact", "collections.deque", "collections.deque"),
    ("exact", "datetime.datetime.strptime", "datetime.datetime.strptime"),
    ("exact", "pathlib.Path.with_suffix", "pathlib.Path.with_suffix"),
]


def _ascii(text: str) -> str:
    return text.encode("ascii", "replace").decode("ascii")


def _metrics(rows: list[dict], label: str, kind: str | None = None) -> dict:
    selected = [r for r in rows if kind is None or r["kind"] == kind]
    ranks = [r[f"{label}_rank"] for r in selected]
    found = [r for r in ranks if r is not None]
    return {
        "n": len(selected),
        "hit_at_1": round(sum(1 for r in found if r == 1) / len(selected), 3),
        "hit_at_5": round(sum(1 for r in found if r <= 5) / len(selected), 3),
        "hit_at_10": round(sum(1 for r in found if r <= 10) / len(selected), 3),
        "mrr": round(sum(1.0 / r for r in found) / len(selected), 3),
        "mean_rank_found": round(statistics.mean(found), 2) if found else None,
    }


def _latency(samples: list[float]) -> dict:
    ordered = sorted(samples)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return {
        "samples": len(samples),
        "mean_ms": round(statistics.mean(samples), 2),
        "p95_ms": round(p95, 2),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dense vs hybrid retrieval benchmark.")
    parser.add_argument("--libs", default="json,re,collections,datetime,pathlib")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=3, help="timed retrievals per query")
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="reuse the existing index instead of re-ingesting (fast iteration)",
    )
    parser.add_argument(
        "--no-adaptive",
        action="store_true",
        help="fuse BM25 into every query (reproduces the natural-language regression)",
    )
    parser.add_argument("--persist-dir", default="data/benchmark_chroma")
    parser.add_argument("--out-dir", default="eval/results")
    args = parser.parse_args(argv)

    from evalcode.core.config import get_settings
    from evalcode.rag.embeddings import get_embedder
    from evalcode.rag.ingest import ingest
    from evalcode.rag.retriever import HybridRetriever, Retriever
    from evalcode.rag.store import VectorStore

    settings = get_settings()
    embedder = get_embedder(settings)
    libraries = [lib.strip() for lib in args.libs.split(",") if lib.strip()]
    store = VectorStore(args.persist_dir, "benchmark_hybrid", embedder)
    total = 0
    if args.reuse:
        total = store.count()
        print(f"reusing index: {total} chunks")
    else:
        report = ingest(libraries, store, docs_dir=None, reset=True)
        total = report["total"]
        print(f"ingested {total} chunks from {', '.join(libraries)} in {report['seconds']:.1f}s")
        # Close and reopen: production retrieval starts against the persisted
        # index, not the writer's in-memory HNSW segment.
        store.close()
        store = VectorStore(args.persist_dir, "benchmark_hybrid", embedder)
    try:
        dense = Retriever(store, args.k, min_score=0.0)
        hybrid = HybridRetriever(
            store,
            args.k,
            min_score=0.0,
            rrf_k=settings.rrf_k,
            adaptive=not args.no_adaptive,
        )

        # Build the BM25 index once, outside per-query timing.
        index_start = time.perf_counter()
        hybrid._lexical_index()
        index_build_ms = (time.perf_counter() - index_start) * 1000.0

        rows: list[dict] = []
        latencies: dict[str, list[float]] = {"dense": [], "hybrid": []}
        for kind, query, expected in QUERIES:
            row: dict = {"kind": kind, "query": query, "expected": expected}
            for label, retriever in (("dense", dense), ("hybrid", hybrid)):
                results = retriever.retrieve([query], k=args.k)
                rank = next(
                    (i for i, doc in enumerate(results, start=1) if doc["qualname"] == expected),
                    None,
                )
                row[f"{label}_rank"] = rank
                # Warm + timed repeats. Rank is recorded from the first run.
                for _ in range(args.repeat):
                    start = time.perf_counter()
                    retriever.retrieve([query], k=args.k)
                    latencies[label].append((time.perf_counter() - start) * 1000.0)
            rows.append(row)

        summary = {
            "libs": libraries,
            "chunks": total,
            "k": args.k,
            "rrf_k": settings.rrf_k,
            "adaptive": not args.no_adaptive,
            "bm25_index_build_ms": round(index_build_ms, 2),
            "dense": {
                "overall": _metrics(rows, "dense"),
                "natural": _metrics(rows, "dense", "natural"),
                "exact": _metrics(rows, "dense", "exact"),
                "latency": _latency(latencies["dense"]),
            },
            "hybrid": {
                "overall": _metrics(rows, "hybrid"),
                "natural": _metrics(rows, "hybrid", "natural"),
                "exact": _metrics(rows, "hybrid", "exact"),
                "latency": _latency(latencies["hybrid"]),
            },
            "rows": rows,
        }

        print()
        print(f"{'query':<52}{'dense':>8}{'hybrid':>8}  expected")
        print("-" * 100)
        for row in rows:
            dense_rank = row["dense_rank"] if row["dense_rank"] is not None else f">{args.k}"
            hybrid_rank = row["hybrid_rank"] if row["hybrid_rank"] is not None else f">{args.k}"
            print(
                _ascii(f"{row['query'][:50]:<52}{str(dense_rank):>8}{str(hybrid_rank):>8}  ")
                + row["expected"]
            )
        print("-" * 100)

        for label in ("dense", "hybrid"):
            block = summary[label]
            for kind in ("overall", "natural", "exact"):
                m = block[kind]
                print(
                    f"{label:<7} {kind:<8} hit@1={m['hit_at_1']:.2f} "
                    f"hit@5={m['hit_at_5']:.2f} hit@10={m['hit_at_10']:.2f} "
                    f"MRR={m['mrr']:.3f} mean_rank={m['mean_rank_found']}"
                )
            lat = block["latency"]
            print(f"{label:<7} latency  mean={lat['mean_ms']:.2f}ms  p95={lat['p95_ms']:.2f}ms")
        print(f"BM25 index build (one-time): {index_build_ms:.1f}ms")

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "retrieval_benchmark.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        _write_markdown(summary, out_dir / "retrieval_benchmark.md")
        print(f"wrote {_ascii(str(out_dir / 'retrieval_benchmark.json'))}")
    finally:
        store.close()
    return 0


def _write_markdown(summary: dict, path: Path) -> None:
    def table(label: str) -> list[str]:
        block = summary[label]
        lines = [
            f"### {label.title()}",
            "",
            "| Slice | n | hit@1 | hit@5 | hit@10 | MRR | Mean rank (found) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for kind in ("overall", "natural", "exact"):
            m = block[kind]
            lines.append(
                f"| {kind} | {m['n']} | {m['hit_at_1']:.2f} | {m['hit_at_5']:.2f} | "
                f"{m['hit_at_10']:.2f} | {m['mrr']:.3f} | {m['mean_rank_found']} |"
            )
        lat = block["latency"]
        lines += [
            "",
            f"Latency: **{lat['mean_ms']:.2f} ms mean**, "
            f"**{lat['p95_ms']:.2f} ms p95** ({lat['samples']} timed retrievals).",
            "",
        ]
        return lines

    lines = [
        "# Retrieval benchmark: Dense-only vs Hybrid (BM25 + RRF)",
        "",
        f"- Corpus: {summary['chunks']} chunks from `{', '.join(summary['libs'])}`",
        "- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (local, CPU)",
        f"- Queries: {len(summary['rows'])} fixed queries "
        f"({summary['dense']['natural']['n']} natural-language, "
        f"{summary['dense']['exact']['n']} exact-symbol)",
        f"- Retrieval depth: top-{summary['k']}; RRF k={summary['rrf_k']}; "
        f"adaptive={'on' if summary['adaptive'] else 'off'}",
        f"- One-time BM25 index build: {summary['bm25_index_build_ms']:.1f} ms",
        "",
        *table("dense"),
        *table("hybrid"),
        "## Per-query ranks",
        "",
        "| Kind | Query | Expected | Dense rank | Hybrid rank |",
        "|---|---|---|---:|---:|",
    ]
    for row in summary["rows"]:
        dense_rank = row["dense_rank"] if row["dense_rank"] is not None else f">{summary['k']}"
        hybrid_rank = row["hybrid_rank"] if row["hybrid_rank"] is not None else f">{summary['k']}"
        lines.append(
            f"| {row['kind']} | {row['query']} | `{row['expected']}` "
            f"| {dense_rank} | {hybrid_rank} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
