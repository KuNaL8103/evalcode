"""Retrieve node: RAG retrieval before generate and after analyze_error (Task 10).

This node runs in two modes:
- "task" mode: when state.code is empty/blank (first pass). Uses the raw task
  text as query, optionally rewritten via LLM (opt-in).
- "error" mode: when state.code is present and analyze_error produced
  retrieval_queries. Uses those deterministic queries.

Retrieval is best-effort: any exception from the retriever is caught, logged,
and treated as zero docs. The run is NEVER failed by retrieval.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from evalcode.config import Settings
from evalcode.llm import TextLLM
from evalcode.parsing import parse_tagged
from evalcode.rag.types import RetrievedDoc
from evalcode.state import AgentState

__all__ = ["RetrieverLike", "make_retrieve_node", "REWRITE_SYSTEM", "build_rewrite_messages"]

logger = logging.getLogger(__name__)


class RetrieverLike(Protocol):
    """Protocol for the retriever used by the retrieve node."""

    def retrieve(
        self, queries: list[str], k: int | None = None, library: str | None = None
    ) -> list[RetrievedDoc]: ...


# --------------------------------------------------------------------------- #
# Query rewrite prompt (opt-in, off by default)
# --------------------------------------------------------------------------- #

REWRITE_SYSTEM = """\
You are a query rewriter for a Python documentation search engine. Given a
coding task, produce 2-4 short, API-oriented search queries that will find
the most relevant standard-library documentation.

REPLY WITH EXACTLY THIS TAGGED BLOCK AND NOTHING ELSE:
<queries>
one query per line
</queries>

Rules:
- Each query <= 80 characters.
- Focus on module names, function names, class names, and method names.
- No natural language questions; use keyword phrases like "json loads" or
  "pathlib Path mkdir".
- Maximum 4 queries.
- Do NOT include markdown fences or any text outside the <queries> block.
"""


def build_rewrite_messages(task: str) -> list[BaseMessage]:
    """Build messages for the query rewrite LLM call."""
    return [
        SystemMessage(content=REWRITE_SYSTEM),
        HumanMessage(content=f"Task:\n{task}"),
    ]


# --------------------------------------------------------------------------- #
# Retrieve node factory
# --------------------------------------------------------------------------- #


def _truncate_query(text: str, max_len: int = 500) -> str:
    """Collapse whitespace and truncate to max_len."""
    collapsed = " ".join(text.split())
    if len(collapsed) > max_len:
        return collapsed[:max_len]
    return collapsed


def _process_llm_queries(raw: str, task: str) -> list[str]:
    """Parse <queries> tag, clean lines, deduplicate, cap at 4, fallback to task."""
    queries_tag = parse_tagged(raw, "queries")
    if not queries_tag:
        return [_truncate_query(task)]

    lines = queries_tag.splitlines()
    out: list[str] = []
    seen: set[str] = set()
    for line in lines:
        # Strip leading bullets, dashes, spaces
        cleaned = line.lstrip("-*• ").strip()
        if not cleaned:
            continue
        truncated = cleaned[:80]
        if truncated not in seen:
            seen.add(truncated)
            out.append(truncated)
        if len(out) >= 4:
            break

    if not out:
        return [_truncate_query(task)]
    return out


def make_retrieve_node(retriever: RetrieverLike, llm: TextLLM | None, settings: Settings):
    """Factory for the retrieve graph node.

    Returns a node function `(state) -> partial_state_dict`.
    """

    def retrieve_node(state: AgentState) -> dict[str, Any]:
        # Determine mode
        code = state.get("code") or ""
        is_task_mode = not code.strip()

        # Get prior retrieval queries and docs (for merging)
        prior_queries = state.get("retrieval_queries") or []
        prior_docs = state.get("retrieved_docs") or []

        new_queries: list[str] = []
        rewrite_used = False
        token_usage: dict[str, Any] = {}

        if is_task_mode:
            # Task mode: raw task query, optionally rewritten
            raw_task = state.get("task") or ""
            base_query = _truncate_query(raw_task)
            new_queries = [base_query]

            if settings.query_rewrite_with_llm and llm is not None:
                try:
                    response = llm.invoke_text(build_rewrite_messages(raw_task), purpose="rewrite")
                    rewrite_used = True
                    # Merge usage
                    if response.usage:
                        token_usage = dict(response.usage)
                    new_queries = _process_llm_queries(response.text, raw_task)
                except Exception:
                    # On any LLM error, fall back to raw task query
                    new_queries = [base_query]
                    rewrite_used = False
        else:
            # Error mode: use deterministic queries from error_analysis
            error_analysis = state.get("error_analysis") or {}
            new_queries = list(error_analysis.get("retrieval_queries") or [])

        # If no queries (error mode with empty list), skip retriever call
        n_docs_returned = 0
        doc_ids: list[str] = []
        scores: list[float] = []
        retriever_error: str | None = None

        if new_queries:
            try:
                new_docs = retriever.retrieve(new_queries)
                n_docs_returned = len(new_docs)
                doc_ids = [d["id"] for d in new_docs]
                scores = [round(d["score"], 3) for d in new_docs]

                # Merge: new docs first (score order), then prior docs not duplicated
                seen_ids = set(doc_ids)
                merged_docs = list(new_docs)
                for doc in prior_docs:
                    if doc["id"] not in seen_ids:
                        merged_docs.append(doc)
                        seen_ids.add(doc["id"])

                # Cap at retrieval_max_docs
                if len(merged_docs) > settings.retrieval_max_docs:
                    merged_docs = merged_docs[: settings.retrieval_max_docs]

                new_retrieved_docs = merged_docs
            except Exception as e:  # noqa: BLE001
                # Best-effort retrieval: log and treat as zero docs
                logger.warning("Retriever error: %s", e, exc_info=True)
                retriever_error = type(e).__name__
                new_retrieved_docs = prior_docs
        else:
            # No queries: return prior docs unchanged
            new_retrieved_docs = prior_docs

        # Accumulate retrieval queries (unique, order preserved)
        all_queries = list(prior_queries)
        for q in new_queries:
            if q not in all_queries:
                all_queries.append(q)

        # Build history event
        attempt = state.get("attempt", 0)
        event = {
            "node": "retrieve",
            "attempt": attempt,
            "summary": {
                "mode": "task" if is_task_mode else "error",
                "queries": new_queries,
                "rewrite_used": rewrite_used,
                "n_docs": n_docs_returned,
                "doc_ids": doc_ids,
                "scores": scores,
                "error": retriever_error,
            },
        }

        result: dict[str, Any] = {
            "retrieval_queries": all_queries,
            "retrieved_docs": new_retrieved_docs,
            "history": [event],
        }
        if token_usage:
            result["token_usage"] = token_usage
        return result

    return retrieve_node
