"""Unit tests for prompts, the tagged-text parser, schemas, and the generate node.

All LLM interaction goes through ``ScriptedLLM`` — no network, no real key.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from evalcode.config import Settings
from evalcode.errors import DailyQuotaExceeded, ParseError
from evalcode.nodes.generate import make_generate_node
from evalcode.parsing import parse_bundle, parse_tagged
from evalcode.prompts import (
    FORMAT_REMINDER,
    GENERATE_SYSTEM,
    build_generate_messages,
    format_context,
)
from evalcode.schemas import CodeBundle, extract_imports
from tests.fakes import ScriptedLLM, bundle_text

CODE = 'import math\n\n\ndef add(a, b):\n    """Return a + b."""\n    return a + b\n'
TESTS = "from solution import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"

DOC_ID = "json-loads-a1"
DOCS = [
    {
        "id": DOC_ID,
        "text": "json.loads(s) -> object\nParse a JSON document.",
        "score": 0.9,
        "library": "json",
        "qualname": "json.loads",
        "import_path": "json",
    }
]


def make_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {"context_max_chars": 2000}
    base.update(overrides)
    return Settings(**base)


def test_parse_bundle_clean() -> None:
    raw = bundle_text(CODE, TESTS, explanation="sums two numbers", docs_used=[DOC_ID])
    bundle = parse_bundle(raw, require_tests=True)
    assert isinstance(bundle, CodeBundle)
    assert bundle.code == CODE.strip()
    assert bundle.tests == TESTS.strip()
    assert bundle.explanation == "sums two numbers"
    assert bundle.docs_used == [DOC_ID]

    # docs_used is optional
    assert parse_bundle(bundle_text(CODE, TESTS)).docs_used == []


def test_parse_bundle_tolerant_variants() -> None:
    # Tag-literal safety: think tags built from parts, never typed as literals.
    open_think = "<" + "think" + ">"
    close_think = "</" + "think" + ">"
    base = bundle_text(CODE, TESTS, docs_used=[DOC_ID])
    fenced_code = f"<code>\n```python\n{CODE}```\n</code>"
    fenced_tests = f"<tests>\n```\n{TESTS}```\n</tests>"

    cases = [
        # all-uppercase tags (content uppercased too — parsing must be case-insensitive)
        (
            base.upper(),
            CODE.upper().strip(),
            TESTS.upper().strip(),
        ),
        # stray prose before and after the tagged blocks
        (f"Sure! Here you go:\n\n{base}\n\nHope that helps!", CODE.strip(), TESTS.strip()),
        # missing </tests> closing tag at the end of the text
        (
            f"<explanation>x</explanation>\n<code>{CODE}</code>\n<tests>\n{TESTS}",
            CODE.strip(),
            TESTS.strip(),
        ),
        # reasoning block in front of the response (defensive re-strip)
        (f"{open_think}planning...{close_think}{base}", CODE.strip(), TESTS.strip()),
        # markdown fences INSIDE the tags
        (
            f"<explanation>x</explanation>\n{fenced_code}\n{fenced_tests}",
            CODE.strip(),
            TESTS.strip(),
        ),
        # missing </code> closing tag followed by <tests>
        (
            f"<explanation>x</explanation>\n<code>\n{CODE}\n<tests>\n{TESTS}\n</tests>",
            CODE.strip(),
            TESTS.strip(),
        ),
    ]
    for raw, expected_code, expected_tests in cases:
        bundle = parse_bundle(raw, require_tests=True)
        assert bundle.code == expected_code, raw[:80]
        assert bundle.tests == expected_tests, raw[:80]


def test_parse_bundle_fallback_fenced_blocks() -> None:
    # No <code>/<tests> tags at all: first fence = code, second = tests.
    raw = f"Here you go:\n\n```python\n{CODE}```\n\n```python\n{TESTS}```"
    bundle = parse_bundle(raw, require_tests=True)
    assert bundle.code == CODE.strip()
    assert bundle.tests == TESTS.strip()
    assert bundle.explanation == ""
    assert bundle.docs_used == []

    # A single fenced block: code is fine, but required tests are not.
    one = f"```python\n{CODE}```"
    assert parse_bundle(one).tests == ""
    with pytest.raises(ParseError, match="no tests"):
        parse_bundle(one, require_tests=True)


def test_parse_bundle_raises_parse_error() -> None:
    with pytest.raises(ParseError, match="empty"):
        parse_bundle("   ")
    # Empty <code> tag and no fenced block anywhere → no code at all.
    with pytest.raises(ParseError, match="no code"):
        parse_bundle(
            "<explanation>oops</explanation>\n<code></code>\n<tests>x</tests>",
            require_tests=True,
        )
    # Code present but tests missing when they are required.
    with pytest.raises(ParseError, match="no tests"):
        parse_bundle(bundle_text(CODE, tests=None), require_tests=True)


def test_parse_tagged_helper() -> None:
    assert parse_tagged("intro <CODE>  spaced code  </code> tail", "code") == "spaced code"
    assert parse_tagged("no tags here", "code") is None
    assert parse_tagged("", "code") is None
    # Missing closing tag: content runs to the end of the text.
    # Missing closing tag followed by a different protocol tag -> stops at that tag.
    assert parse_tagged("<code>\ncode body\n<tests>\ntests</tests>", "code") == "code body"
    assert parse_tagged("<tests>\nunterminated\nmore", "tests") == "unterminated\nmore"
    assert parse_tagged("", "code") is None


def test_extract_imports() -> None:
    assert extract_imports("import json\nimport os.path\nfrom collections import deque\n") == [
        "json",
        "os",
        "collections",
    ]
    assert extract_imports("import json") == ["json"]
    assert extract_imports("def f():\n    import math\n    return math") == ["math"]
    assert extract_imports("from . import x") == []  # relative import names no top-level module
    assert extract_imports("def f( -> :") == []  # syntax errors tolerated
    assert extract_imports("") == []


def test_format_context() -> None:
    assert format_context([], 1000) == ""

    def doc(doc_id: str, import_path: str, text: str) -> dict[str, Any]:
        return {
            "id": doc_id,
            "text": text,
            "score": 0.5,
            "library": "json",
            "qualname": doc_id,
            "import_path": import_path,
        }

    docs = [doc("a", "json", "A" * 20), doc("b", "re", "B" * 20), doc("c", "datetime", "C" * 20)]
    full = format_context(docs, 10_000)
    # "[doc:a] json\n" + 20 chars = 33 per block; blocks joined on a blank line.
    blocks = full.split("\n\n")
    assert len(blocks) == 3
    assert blocks[0] == "[doc:a] json\n" + "A" * 20
    assert len(blocks[0]) == 33
    # Relevance order preserved.
    assert full.index("[doc:b]") > full.index("[doc:a]")

    two = "\n\n".join(blocks[:2])
    assert format_context(docs, len(two)) == two  # exactly two blocks fit
    assert format_context(docs, len(two) - 1) == blocks[0]  # third dropped whole
    assert len(format_context(docs, 31)) == 0  # not even one block fits


def test_build_generate_messages() -> None:
    state = {"task": "Write add(a, b).", "retrieved_docs": DOCS}
    messages = build_generate_messages(state, context_max_chars=10_000)
    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == GENERATE_SYSTEM
    human = messages[1]
    assert isinstance(human, HumanMessage)
    content = human.content
    assert "Write add(a, b)." in content
    assert f"[doc:{DOC_ID}] json" in content

    # Provided tests: explicit instruction, no doc context when there are no docs.
    provided = "def test_given():\n    assert add(1, 1) == 2"
    content2 = build_generate_messages({"task": "t", "provided_tests": provided})[1].content
    assert "PROVIDED TESTS" in content2
    assert provided in content2
    assert "Documentation context" not in content2


def test_generate_node_success() -> None:
    llm = ScriptedLLM([bundle_text(CODE, TESTS, explanation="sum", docs_used=[DOC_ID])])
    node = make_generate_node(llm, make_settings())
    state = {"task": "add(a, b)", "retrieved_docs": DOCS}
    snapshot = copy.deepcopy(state)
    update = node(state)
    assert state == snapshot  # input state is never mutated

    assert update["code"] == CODE.strip()
    assert update["tests"] == TESTS.strip()
    assert update["explanation"] == "sum"
    assert update["attempt"] == 1  # absent attempt defaults to 0 -> 1
    assert update["status"] == "running"
    assert update["token_usage"] == {
        "input_tokens": 10,
        "output_tokens": 20,
        "total_tokens": 30,
        "llm_calls": 1,
        "api_retries": 0,
        "wait_s": 0.0,
    }
    event = update["history"][0]
    assert event["node"] == "generate"
    assert event["attempt"] == 1
    assert event["ts"]
    assert event["summary"]["code_chars"] == len(CODE.strip())
    assert event["summary"]["tests_chars"] == len(TESTS.strip())
    assert event["summary"]["doc_ids"] == [DOC_ID]
    assert event["summary"]["docs_used"] == [DOC_ID]
    assert event["summary"]["imports"] == ["math"]
    assert event["summary"]["reasks"] == 0
    assert llm.purposes == ["generate"]

    # provided_tests are forced: no <tests> section required, tests come from state.
    provided = "def test_given():\n    assert add(1, 1) == 2\n"
    llm2 = ScriptedLLM([bundle_text(CODE, tests=None)])
    state2 = {"task": "t", "attempt": 2, "provided_tests": provided}
    update2 = make_generate_node(llm2, make_settings())(state2)
    assert update2["tests"] == provided
    assert update2["attempt"] == 3
    assert update2["status"] == "running"
    assert update2["history"][0]["summary"]["tests_chars"] == len(provided)


def test_generate_node_reask_and_failures() -> None:
    bad = "Sure! The answer is: 5"  # no tags, no fences -> unparseable

    # (a) bad reply, then one strict re-ask -> success on the second call.
    llm = ScriptedLLM([bad, bundle_text(CODE, TESTS)])
    update = make_generate_node(llm, make_settings())({"task": "t"})
    assert update["status"] == "running"
    assert len(llm.calls) == 2
    second = llm.calls[1]
    assert second[:2] == llm.calls[0]  # same system + task messages
    assert isinstance(second[2], AIMessage) and second[2].content == bad
    assert isinstance(second[3], HumanMessage) and second[3].content == FORMAT_REMINDER
    assert update["history"][0]["summary"]["reasks"] == 1
    assert "reply_head" in update["history"][0]["summary"]
    assert update["history"][0]["summary"]["reply_head"]
    assert "parse_reason" in update["history"][0]["summary"]
    assert update["token_usage"]["llm_calls"] == 2  # the re-ask counts too

    # (b) bad twice -> failed update, never an exception.
    llm2 = ScriptedLLM([bad, "still prose"])
    update2 = make_generate_node(llm2, make_settings())({"task": "t"})
    assert update2["status"] == "failed"
    assert "parseable" in update2["failure_reason"]
    assert "reply_head" in update2["history"][0]["summary"]
    assert "parse_reason" in update2["history"][0]["summary"]
    assert update2["history"][0]["summary"]["error"] == "ParseError"
    assert update2["history"][0]["summary"]["reasks"] == 1
    assert "reply_head" in update2["history"][0]["summary"]
    assert "parse_reason" in update2["history"][0]["summary"]
    assert len(llm2.calls) == 2

    # (c) daily quota on the first call -> failed update with the quota message.
    llm3 = ScriptedLLM([DailyQuotaExceeded("free-tier daily limit reached")])
    update3 = make_generate_node(llm3, make_settings())({"task": "t"})
    assert update3["status"] == "failed"
    assert "daily limit" in update3["failure_reason"]
    assert update3["history"][0]["summary"] == {"error": "DailyQuotaExceeded"}
    assert update3["token_usage"] == {}  # no successful call, no usage

    # (d) quota on the re-ask -> failed, but keeps the FIRST call's usage only.
    llm4 = ScriptedLLM([bad, DailyQuotaExceeded("daily limit")])
    update4 = make_generate_node(llm4, make_settings())({"task": "t"})
    assert update4["status"] == "failed"
    assert update4["token_usage"]["total_tokens"] == 30
    assert update4["token_usage"]["llm_calls"] == 1
    assert len(llm4.calls) == 2
