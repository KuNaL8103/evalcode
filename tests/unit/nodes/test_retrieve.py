"""Tests for the retrieve node (Task 10)."""

from __future__ import annotations

from typing import Any

from evalcode.config import Settings
from evalcode.errors import DailyQuotaExceeded
from evalcode.graph import initial_state
from evalcode.nodes.retrieve import make_retrieve_node
from tests.fakes import FakeRetriever, ScriptedLLM, make_doc


def make_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "context_max_chars": 2000,
        "max_retries": 3,
        "max_human_rounds": 2,
        "retrieval_max_docs": 8,
        "query_rewrite_with_llm": False,
    }
    base.update(overrides)
    return Settings(**base)


def test_retrieve_task_mode_raw_query():
    """Task mode (code empty): queries == [task]; docs stored; retrieval_queries set;
    event keys/values incl. mode 'task', n_docs, doc_ids, scores, error None;
    retriever called once; no llm call; returned keys are only the allowed ones;
    input state not mutated."""
    # Two docs for the task query
    doc1 = make_doc("doc1", "json loads doc", score=0.9)
    doc2 = make_doc("doc2", "json dumps doc", score=0.8)
    retriever = FakeRetriever(docs_by_query={"write a function to parse json": [doc1, doc2]})
    llm = ScriptedLLM(["<queries>\njson loads\njson dumps\n</queries>"])

    node = make_retrieve_node(retriever, llm, make_settings(query_rewrite_with_llm=False))

    init_state = initial_state(
        task="write a function to parse json",
        settings=make_settings(),
        auto_approve=True,
    )
    # Keep a copy to verify no mutation
    original_state = dict(init_state)

    result = node(init_state)

    # Returned keys only (token_usage only present when LLM call happened)
    allowed_keys = {"retrieval_queries", "retrieved_docs", "history"}
    assert set(result.keys()) == allowed_keys

    # Input state not mutated
    assert init_state == original_state

    # Queries
    assert result["retrieval_queries"] == ["write a function to parse json"]

    # Docs
    docs = result["retrieved_docs"]
    assert len(docs) == 2
    assert docs[0]["id"] == "doc1"
    assert docs[1]["id"] == "doc2"

    # History event
    history = result["history"]
    assert len(history) == 1
    event = history[0]
    assert event["node"] == "retrieve"
    assert event["attempt"] == 0
    assert set(event) == {"node", "attempt", "ts", "summary"}
    assert isinstance(event["ts"], str) and event["ts"]
    summary = event["summary"]
    assert summary["mode"] == "task"
    assert summary["queries"] == ["write a function to parse json"]
    assert summary["rewrite_used"] is False
    assert summary["n_docs"] == 2
    assert summary["doc_ids"] == ["doc1", "doc2"]
    assert summary["scores"] == [0.9, 0.8]
    assert summary["error"] is None

    # Retriever called once with the task query
    assert retriever.calls == [["write a function to parse json"]]

    # LLM NOT called (query_rewrite_with_llm=False)
    assert llm.calls == []

    # No token_usage (no LLM call)
    assert "token_usage" not in result


def test_retrieve_error_mode_merge_and_cap():
    """Error mode (code present + error_analysis queries): new docs first,
    prior docs kept if not duplicated, duplicate id appears once, capped to
    retrieval_max_docs; retrieval_queries accumulate prior + new without
    duplicates; mode 'error'."""
    # Prior docs
    prior_doc1 = make_doc("p1", "prior 1", score=0.7)
    prior_doc2 = make_doc("p2", "prior 2", score=0.6)
    prior_doc3 = make_doc("p3", "prior 3", score=0.5)
    # New docs for error queries (n1 has same id as prior_doc1 but new score/text)
    n1_new = make_doc("p1", "new p1", score=0.99, import_path="math.sqrt")
    n1 = make_doc("n1", "new 1", score=0.95)
    n3 = make_doc("n3", "new 3", score=0.85)

    # retrieval_max_docs=4 to test cap
    retriever = FakeRetriever(
        docs_by_query={
            "math sqroot": [n1, n1_new],
            "math sqrt fix": [n3],
        }
    )
    llm = ScriptedLLM(["dummy"])

    node = make_retrieve_node(retriever, llm, make_settings(retrieval_max_docs=4))

    # State with code present (error mode) and prior docs/queries
    state = initial_state(
        task="write a function using math",
        settings=make_settings(retrieval_max_docs=4),
        auto_approve=True,
    )
    state["code"] = "import math\nmath.sqroot(4)"
    state["retrieval_queries"] = ["write a function using math"]
    state["retrieved_docs"] = [prior_doc1, prior_doc2, prior_doc3]
    state["error_analysis"] = {
        "retrieval_queries": ["math sqroot", "math sqrt fix"],
        "needs_docs": True,
    }
    state["attempt"] = 1

    result = node(state)

    # Queries accumulated (prior + new, unique, order preserved)
    assert result["retrieval_queries"] == [
        "write a function using math",
        "math sqroot",
        "math sqrt fix",
    ]

    # Docs: new docs first (score order), then prior docs not duplicated, capped at 4.
    # New docs: n1 (0.95), p1/new (0.99), n3 (0.85) -> sorted by score:
    # p1/new (0.99), n1 (0.95), n3 (0.85)
    # Then prior p2 (0.6) not duplicated, p3 (0.5) dropped by cap
    docs = result["retrieved_docs"]
    assert len(docs) == 4  # capped at 4
    assert docs[0]["id"] == "p1"  # new p1 with score 0.99
    assert docs[1]["id"] == "n1"
    assert docs[2]["id"] == "n3"
    assert docs[3]["id"] == "p2"

    # History event mode error
    event = result["history"][0]
    summary = event["summary"]
    assert summary["mode"] == "error"
    assert summary["queries"] == ["math sqroot", "math sqrt fix"]
    assert summary["n_docs"] == 3  # new docs returned by retriever
    assert summary["doc_ids"] == ["p1", "n1", "n3"]
    assert summary["error"] is None

    # Retriever called with both queries
    assert retriever.calls == [["math sqroot", "math sqrt fix"]]


def test_retrieve_no_queries_and_failure_tolerance():
    """Error mode with empty retrieval_queries -> retriever.calls == [] and
    docs unchanged; retriever raising RuntimeError -> no exception, docs
    unchanged, summary['error'] == 'RuntimeError', history event present."""
    retriever = FakeRetriever()
    llm = ScriptedLLM(["dummy"])
    node = make_retrieve_node(retriever, llm, make_settings())

    # Case 1: empty retrieval_queries
    state = initial_state(
        task="some task",
        settings=make_settings(),
        auto_approve=True,
    )
    state["code"] = "code"
    state["retrieval_queries"] = ["prior"]
    state["retrieved_docs"] = [make_doc("d1")]
    state["error_analysis"] = {"retrieval_queries": [], "needs_docs": True}
    state["attempt"] = 1

    result = node(state)

    assert retriever.calls == []  # no retriever call
    assert result["retrieved_docs"] == [make_doc("d1")]  # unchanged
    assert result["retrieval_queries"] == ["prior"]  # unchanged
    event = result["history"][0]
    assert event["summary"]["mode"] == "error"
    assert event["summary"]["queries"] == []
    assert event["summary"]["n_docs"] == 0
    assert event["summary"]["error"] is None

    # Case 2: retriever raises
    retriever2 = FakeRetriever(raises=RuntimeError("chroma down"))
    node2 = make_retrieve_node(retriever2, llm, make_settings())

    state2 = initial_state(
        task="some task",
        settings=make_settings(),
        auto_approve=True,
    )
    state2["code"] = "code"
    state2["retrieval_queries"] = ["prior"]
    state2["retrieved_docs"] = [make_doc("d1")]
    state2["error_analysis"] = {"retrieval_queries": ["new query"], "needs_docs": True}
    state2["attempt"] = 1

    result2 = node2(state2)

    # No exception raised
    assert result2["retrieved_docs"] == [make_doc("d1")]  # unchanged
    assert result2["retrieval_queries"] == ["prior", "new query"]  # accumulated
    event2 = result2["history"][0]
    assert event2["summary"]["mode"] == "error"
    assert event2["summary"]["error"] == "RuntimeError"
    assert event2["summary"]["n_docs"] == 0


def test_retrieve_llm_rewrite():
    """query_rewrite_with_llm=True + ScriptedLLM replying with queries ->
    queries == those two, llm.purposes == ['rewrite'], token_usage llm_calls 1,
    rewrite_used True; DailyQuotaExceeded script -> falls back to [task],
    no raise, no token_usage; flag False with an llm supplied -> zero llm calls."""
    doc1 = make_doc("d1", "json loads", score=0.9)
    doc2 = make_doc("d2", "json dumps", score=0.85)

    # Case 1: rewrite works
    retriever = FakeRetriever(docs_by_query={"json loads": [doc1], "parse json string": [doc2]})
    llm = ScriptedLLM(["<queries>\n- json loads\nparse json string\n</queries>"])
    settings = make_settings(query_rewrite_with_llm=True)
    node = make_retrieve_node(retriever, llm, settings)

    state = initial_state(
        task="write a function to parse a JSON string into a dict",
        settings=settings,
        auto_approve=True,
    )
    state["attempt"] = 0

    result = node(state)

    assert result["retrieval_queries"] == ["json loads", "parse json string"]
    assert llm.purposes == ["rewrite"]
    assert result["token_usage"]["llm_calls"] == 1
    event = result["history"][0]
    assert event["summary"]["rewrite_used"] is True
    assert event["summary"]["queries"] == ["json loads", "parse json string"]

    # Case 2: DailyQuotaExceeded -> fallback to raw task, no raise, no token_usage
    llm2 = ScriptedLLM([DailyQuotaExceeded("daily limit")])
    query = "write a function to parse a JSON string into a dict"
    retriever2 = FakeRetriever(docs_by_query={query: [doc1]})
    node2 = make_retrieve_node(retriever2, llm2, settings)

    state2 = initial_state(
        task="write a function to parse a JSON string into a dict",
        settings=settings,
        auto_approve=True,
    )
    state2["attempt"] = 0

    result2 = node2(state2)

    assert result2["retrieval_queries"] == ["write a function to parse a JSON string into a dict"]
    assert "token_usage" not in result2
    event2 = result2["history"][0]
    assert event2["summary"]["rewrite_used"] is False

    # Case 3: query_rewrite_with_llm=False but llm supplied -> zero llm calls
    llm3 = ScriptedLLM(["<queries>\nshould not be called\n</queries>"])
    retriever3 = FakeRetriever(docs_by_query={"task query": [doc1]})
    node3 = make_retrieve_node(retriever3, llm3, make_settings(query_rewrite_with_llm=False))

    state3 = initial_state(
        task="task query",
        settings=make_settings(query_rewrite_with_llm=False),
        auto_approve=True,
    )
    state3["attempt"] = 0

    result3 = node3(state3)

    assert result3["retrieval_queries"] == ["task query"]
    assert llm3.calls == []
    assert "token_usage" not in result3
