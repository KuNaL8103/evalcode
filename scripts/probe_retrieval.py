"""Retrieval-quality probe (dev tool, printing allowed).

Ingests a fixed library set with the REAL local embedder into a throwaway
Chroma dir, runs a fixed query list, and reports per-query ranks plus
summary metrics (hit@5, hit@10, MRR, mean rank over found queries).

Usage:
    python scripts/probe_retrieval.py --label baseline
    python scripts/probe_retrieval.py --label compact --persist-dir data/probe_chroma

Results are written to data/probe_<label>.json. The store is always closed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Fixed probe queries: (query, expected qualname). DO NOT EDIT after the
# baseline run — changing them would invalidate comparisons across labels.
# Rank = position of the first retrieved chunk whose metadata qualname equals
# the expected one; None means "not in the top k".
PROBE_QUERIES: list[tuple[str, str]] = [
    ("parse a JSON string into a Python object", "json.loads"),
    ("serialize a Python object to a JSON string", "json.dumps"),
    ("write an object as JSON to a file", "json.dump"),
    ("replace all matches of a pattern", "re.sub"),
    ("find all non-overlapping matches of a regex in a string", "re.findall"),
    ("split a string by a regular expression", "re.split"),
    ("count occurrences of items in an iterable", "collections.Counter"),
    ("double-ended queue with fast appends and pops", "collections.deque"),
    ("move a key to the end of an ordered dictionary", "collections.OrderedDict.move_to_end"),
    ("dictionary with a default value factory for missing keys", "collections.defaultdict"),
]


def _ascii(text: str) -> str:
    """ASCII-safe rendering for Windows consoles."""
    return text.encode("ascii", "replace").decode("ascii")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retrieval quality probe.")
    parser.add_argument("--persist-dir", default="data/probe_chroma")
    parser.add_argument("--label", required=True, help="result label (probe_<label>.json)")
    parser.add_argument("--libs", default="json,re,collections")
    parser.add_argument("--k", type=int, default=20, help="retrieval depth for rank lookup")
    args = parser.parse_args(argv)

    from evalcode.config import get_settings
    from evalcode.rag.embeddings import get_embedder
    from evalcode.rag.ingest import ingest
    from evalcode.rag.retriever import Retriever
    from evalcode.rag.store import VectorStore

    settings = get_settings()
    embedder = get_embedder(settings)
    store = VectorStore(args.persist_dir, f"probe_{args.label}", embedder)
    try:
        libraries = [x.strip() for x in args.libs.split(",") if x.strip()]
        report = ingest(libraries, store, docs_dir=None, reset=True)
        print(
            f"ingested {report['total']} chunks "
            f"({', '.join(f'{k}={v}' for k, v in sorted(report['per_library'].items()))}) "
            f"in {report['seconds']:.1f}s"
        )

        retriever = Retriever(store, args.k, min_score=0.0)
        rows: list[dict[str, object]] = []
        for query, expected in PROBE_QUERIES:
            results = retriever.retrieve([query], k=args.k)
            rank = next((i for i, r in enumerate(results, 1) if r["qualname"] == expected), None)
            rows.append({"query": query, "expected": expected, "rank": rank})

        print()
        print(f"probe [{args.label}]  k={args.k}")
        print("-" * 88)
        print(f"{'rank':<6}{'expected':<36}query")
        for row in rows:
            shown = str(row["rank"]) if row["rank"] is not None else f"> {args.k}"
            print(f"{shown:<6}{row['expected']:<36}{row['query']}")
        print("-" * 88)

        found = [r["rank"] for r in rows if r["rank"] is not None]
        hit5 = sum(1 for r in found if r <= 5)
        hit10 = sum(1 for r in found if r <= 10)
        mrr = sum(1.0 / r for r in found) / len(rows)
        mean_rank = (sum(found) / len(found)) if found else 0.0
        summary = {
            "hit_at_5": round(hit5 / len(rows), 4),
            "hit_at_10": round(hit10 / len(rows), 4),
            "mrr": round(mrr, 4),
            "mean_rank_found": round(mean_rank, 2),
        }
        print(
            f"hit@5={summary['hit_at_5']:.2f}  hit@10={summary['hit_at_10']:.2f}  "
            f"MRR={summary['mrr']:.3f}  mean_rank(found)={summary['mean_rank_found']:.2f}"
        )

        out = Path("data") / f"probe_{args.label}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "label": args.label,
            "libs": libraries,
            "k": args.k,
            "total_chunks": report["total"],
            "rows": rows,
            **summary,
        }
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"results -> {_ascii(str(out))}")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
