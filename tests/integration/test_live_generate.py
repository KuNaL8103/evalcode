"""Opt-in live test: the generate path against real Gemini (``-m live``).

Makes at most 2 calls (one strict re-ask only if the first reply is
unparseable) on the DEFAULT free model. Skips with a clear reason when
``OPENROUTER_API_KEY`` is unavailable.
"""

from __future__ import annotations

import os
import re

import pytest
from dotenv import dotenv_values, find_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import SecretStr

from evalcode.config import Settings
from evalcode.errors import ParseError
from evalcode.llm import build_llm_client
from evalcode.parsing import parse_bundle
from evalcode.prompts import FORMAT_REMINDER, build_generate_messages

DEFAULT_MODEL = "gemini-3.5-flash-lite"


def _read_key() -> str:
    """Real key from the shell env, else the developer's .env (live opt-in only).

    The autouse env-isolation fixture strips Settings vars from ``os.environ``
    and hides the real .env from ``evalcode.config``; a live test deliberately
    opts back in through the raw dotenv helpers.
    """
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    env_file = find_dotenv(usecwd=True)
    if env_file:
        values = dotenv_values(env_file)
        return str(values.get("OPENROUTER_API_KEY") or "").strip()
    return ""


@pytest.mark.live
def test_live_generate_add_function() -> None:
    key = _read_key()
    if not key:
        pytest.skip("OPENROUTER_API_KEY not set (env or .env); live test needs it")
    settings = Settings(
        openrouter_api_key=SecretStr(key),
        llm_model=DEFAULT_MODEL,
        llm_min_interval_s=0.0,
    )
    client = build_llm_client(settings)
    state = {"task": "Write a Python function add(a, b) that returns a + b."}
    messages = build_generate_messages(state)
    response = client.invoke_text(messages, purpose="generate")
    try:
        bundle = parse_bundle(response.text, require_tests=True)
        first_reply_ok = True
    except ParseError as exc:
        # Diagnostics before the strict re-ask — never log the full reply or key.
        raw = response.text or ""
        head = "".join(ch if ord(ch) < 128 else "?" for ch in raw[:300])
        head = re.sub(r"\s+", " ", head).strip()
        print(f"live parse-fail: reason={exc}, reply_len={len(raw)}, reply_head={head!r}")
        # Exactly one strict re-ask: the test still makes at most 2 calls.
        reask = [
            *messages,
            AIMessage(content=response.text),
            HumanMessage(content=FORMAT_REMINDER),
        ]
        response = client.invoke_text(reask, purpose="generate")
        bundle = parse_bundle(response.text, require_tests=True)
        first_reply_ok = False
    assert bundle.code.strip(), "parsed code is empty"
    assert bundle.tests.strip(), "parsed tests is empty"
    print(f"live generate: model={response.model} first_reply_followed_format={first_reply_ok}")
