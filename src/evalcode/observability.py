"""Observability: structured run logging, node tracing, and LangSmith config (Task 11).

Provides:
- RunLogger: JSONL event log + summary.json per run (lazy directory creation,
  file handle per write so deletable on Windows, never raises into the run).
- traced_node: wraps any graph node to log structured events after execution,
  with per-node usage, cumulative usage, and LLMClient stats deltas.
- redact: deep redaction of secrets (API keys, patterns) and string truncation.
- configure_langsmith: enables/disables LangSmith via environment variables.
- build_run_config: LangGraph config dict with run_id tags and metadata.
- build_summary: pure function building the final summary from events + state.
- new_run_id: generates a deterministic run identifier.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evalcode.config import Settings
from evalcode.state import AgentState, merge_usage, utc_now_iso

try:
    from langgraph.errors import GraphInterrupt
except Exception:  # pragma: no cover - langgraph may not be installed in all envs

    class GraphInterrupt(Exception):
        pass


__all__ = [
    "RunLogger",
    "traced_node",
    "redact",
    "configure_langsmith",
    "build_run_config",
    "build_summary",
    "build_logger",
    "new_run_id",
]

# Regex patterns for secret detection (built from parts to avoid literal keys in source)
_SECRETS_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_-]{30,}"),
    re.compile(r"lsv2_[A-Za-z0-9_]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
]


def new_run_id() -> str:
    """Generate a run identifier: UTC timestamp + short UUID."""
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return f"{ts}-{short_uuid}"


def redact(obj: Any, secrets: tuple[str, ...] = (), *, max_str: int = 2000) -> Any:
    """Return a deep copy with secrets redacted and long strings truncated.

    Args:
        obj: The object to redact (dict, list, tuple, str, or other).
        secrets: Tuple of exact secret strings to replace with "***".
        max_str: Maximum string length; longer strings get "...[truncated]".

    Returns:
        A new object with the same structure, never mutating the input.
    """
    if isinstance(obj, str):
        result = obj
        # Replace exact secret strings (length >= 8)
        for secret in secrets:
            if secret and len(secret) >= 8:
                result = result.replace(secret, "***")
        # Replace pattern-matched secrets
        for pattern in _SECRETS_PATTERNS:
            result = pattern.sub("***", result)
        # Truncate long strings
        if len(result) > max_str:
            result = result[:max_str] + "...[truncated]"
        return result

    if isinstance(obj, dict):
        return {
            redact(k, secrets, max_str=max_str): redact(v, secrets, max_str=max_str)
            for k, v in obj.items()
        }

    if isinstance(obj, list):
        return [redact(item, secrets, max_str=max_str) for item in obj]

    if isinstance(obj, tuple):
        return tuple(redact(item, secrets, max_str=max_str) for item in obj)

    # For non-string scalars (int, float, bool, None), return as-is
    if obj is None or isinstance(obj, (int, float, bool)):
        return obj

    # For other objects (SecretStr, etc.), stringify and redact
    return redact(str(obj), secrets, max_str=max_str)


class RunLogger:
    """Structured run logger writing JSONL events and a final summary.json.

    Directory is created lazily on first write. Each event opens the file for
    append and closes it immediately so the run directory remains deletable
    on Windows. Any I/O error is caught and logged (never raised into the run).
    """

    def __init__(
        self, log_dir: str | Path, run_id: str | None = None, *, secrets: tuple[str, ...] = ()
    ):
        self.run_id = run_id or new_run_id()
        self.run_dir = Path(log_dir) / self.run_id
        self.events_path = self.run_dir / "events.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self._secrets = secrets

    def log_event(self, event: dict[str, Any]) -> None:
        """Append one JSON line to events.jsonl (redacted, with run_id)."""
        # Add run_id if absent
        event_with_id = dict(event)
        if "run_id" not in event_with_id:
            event_with_id["run_id"] = self.run_id

        # Redact
        event_redacted = redact(event_with_id, self._secrets)

        # Ensure directory exists and append line
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            with open(self.events_path, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(event_redacted, ensure_ascii=False) + "\n")
        except (OSError, TypeError, ValueError) as e:
            logging.getLogger(__name__).warning("RunLogger log_event failed: %s", e)

    def read_events(self) -> list[dict[str, Any]]:
        """Parse events.jsonl line by line, skipping blank/malformed lines."""
        if not self.events_path.exists():
            return []
        events: list[dict[str, Any]] = []
        try:
            with open(self.events_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        # Skip malformed lines
                        continue
        except OSError:
            return []
        return events

    def _clear_events(self) -> None:
        """Clear the events file (for test isolation)."""
        try:
            if self.events_path.exists():
                self.events_path.unlink()
        except OSError:
            pass

    def write_summary(
        self, state: AgentState, *, extra: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Build and write summary.json; return the summary dict."""
        events = self.read_events()
        summary = build_summary(self.run_id, events, state)
        if extra:
            summary.update(extra)
        summary = redact(summary, self._secrets)
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            with open(self.summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        except (OSError, TypeError, ValueError) as e:
            logging.getLogger(__name__).warning("RunLogger write_summary failed: %s", e)
        return summary


def build_summary(run_id: str, events: list[dict[str, Any]], state: AgentState) -> dict[str, Any]:
    """Pure function: build the final run summary from events and state."""
    if not events:
        started_ts = None
        finished_ts = None
        duration_ms_total = 0
        node_counts: dict[str, int] = {}
        outcomes: dict[str, int] = {}
        llm_client_totals = {"calls": 0, "api_retries": 0, "wait_s": 0.0}
    else:
        started_ts = events[0].get("ts")
        finished_ts = events[-1].get("ts")
        duration_ms_total = round(sum(e.get("duration_ms", 0) for e in events), 1)

        node_counts = {}
        outcomes = {}
        llm_calls = 0
        api_retries = 0
        wait_s = 0.0

        for e in events:
            node = e.get("node")
            if node:
                node_counts[node] = node_counts.get(node, 0) + 1
            outcome = e.get("outcome")
            if outcome:
                outcomes[outcome] = outcomes.get(outcome, 0) + 1

            llm_client = e.get("llm_client") or {}
            llm_calls += llm_client.get("calls", 0)
            api_retries += llm_client.get("api_retries", 0)
            wait_s += llm_client.get("wait_s", 0.0)

        llm_client_totals = {
            "calls": llm_calls,
            "api_retries": api_retries,
            "wait_s": round(wait_s, 3),
        }

    # RAG flag: any event with node == "retrieve"
    rag = any(e.get("node") == "retrieve" for e in events)

    # Retrieved doc IDs from state
    retrieved_docs = state.get("retrieved_docs") or []
    retrieved_doc_ids = [d.get("id") for d in retrieved_docs if isinstance(d, dict) and d.get("id")]

    # Status: "awaiting_review" if __interrupt__ is truthy (non-empty), else state.status
    interrupt = state.get("__interrupt__")
    status = "awaiting_review" if interrupt else state.get("status")

    # Failure reason truncated to 300 chars
    failure_reason = state.get("failure_reason")
    if isinstance(failure_reason, str) and len(failure_reason) > 300:
        failure_reason = failure_reason[:300]

    # Final category from run_result
    run_result = state.get("run_result") or {}
    final_category = run_result.get("category")

    return {
        "run_id": run_id,
        "task_id": state.get("task_id"),
        "status": status,
        "failure_reason": failure_reason,
        "attempt": state.get("attempt", 0),
        "retries_used": state.get("retries_used", 0),
        "human_rounds": state.get("human_rounds", 0),
        "n_events": len(events),
        "node_counts": node_counts,
        "outcomes": outcomes,
        "duration_ms_total": duration_ms_total,
        "token_usage": state.get("token_usage") or {},
        "llm_client_totals": llm_client_totals,
        "rag": rag,
        "retrieved_doc_ids": retrieved_doc_ids,
        "final_category": final_category,
        "started_ts": started_ts,
        "finished_ts": finished_ts,
    }


def configure_langsmith(settings: Settings, environ: dict[str, str] | None = None) -> bool:
    """Enable or disable LangSmith tracing via environment variables.

    Args:
        settings: Settings with langsmith_tracing and langsmith_api_key.
        environ: Optional environment dict to modify (defaults to os.environ).

    Returns:
        True if tracing was enabled (key present and tracing requested),
        False otherwise (tracing disabled or key missing).
    """
    if environ is None:
        environ = os.environ

    tracing_requested = settings.langsmith_tracing
    api_key = settings.langsmith_api_key.get_secret_value() if settings.langsmith_api_key else ""

    if tracing_requested and api_key:
        environ["LANGSMITH_TRACING"] = "true"
        environ["LANGSMITH_API_KEY"] = api_key
        environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        return True

    # Disable tracing
    environ["LANGSMITH_TRACING"] = "false"
    if tracing_requested and not api_key:
        logging.getLogger(__name__).warning("LangSmith tracing requested but no API key provided")
    return False


def build_run_config(
    task_id: str, *, run_id: str | None = None, recursion_limit: int = 100
) -> dict[str, Any]:
    """Build LangGraph config dict with recursion_limit and run/task tags."""
    config: dict[str, Any] = {
        "recursion_limit": recursion_limit,
        "configurable": {"thread_id": task_id},
    }
    if run_id is not None:
        config["tags"] = [f"run_id:{run_id}", f"task_id:{task_id}"]
        config["metadata"] = {"run_id": run_id, "task_id": task_id}
    return config


def build_logger(settings: Settings, run_id: str | None = None) -> RunLogger:
    """Create a RunLogger with secrets from settings for automatic redaction."""
    secrets: list[str] = []
    if settings.gemini_api_key:
        secrets.append(settings.gemini_api_key.get_secret_value())
    if settings.langsmith_api_key:
        secrets.append(settings.langsmith_api_key.get_secret_value())
    return RunLogger(settings.log_dir, run_id, secrets=tuple(secrets))


def traced_node(name: str, fn: Any, logger: RunLogger | None, *, llm_stats: Any = None) -> Any:
    """Wrap a graph node to log structured events after execution.

    Args:
        name: Node name for the event.
        fn: The node function to wrap.
        logger: RunLogger instance, or None to disable logging (return fn unchanged).
        llm_stats: Optional object with a `stats` attribute (dict with calls, api_retries, wait_s).

    Returns:
        A wrapped node function that logs after the inner function returns/raises.
    """
    if logger is None:
        return fn

    def traced(state: AgentState) -> dict[str, Any]:
        # Snapshot LLM stats before
        stats_before = None
        if llm_stats is not None and hasattr(llm_stats, "stats"):
            try:
                stats_before = dict(llm_stats.stats)
            except Exception:
                stats_before = None

        t0 = time.perf_counter()
        try:
            update = fn(state)
            outcome = "failed" if update.get("status") == "failed" else "ok"
            # Log after successful return
            _log_node_event(
                logger=logger,
                state=state,
                update=update,
                name=name,
                t0=t0,
                outcome=outcome,
                stats_before=stats_before,
                llm_stats=llm_stats,
                error=None,
            )
            return update
        except GraphInterrupt:
            # Log interrupt and re-raise the SAME exception
            _log_node_event(
                logger=logger,
                state=state,
                update={},
                name=name,
                t0=t0,
                outcome="interrupted",
                stats_before=stats_before,
                llm_stats=llm_stats,
                error=None,
            )
            raise
        except Exception as e:
            # Log error and re-raise the SAME exception
            _log_node_event(
                logger=logger,
                state=state,
                update={},
                name=name,
                t0=t0,
                outcome="error",
                stats_before=stats_before,
                llm_stats=llm_stats,
                error=e,
            )
            raise

    return traced


def _log_node_event(
    logger: RunLogger,
    state: AgentState,
    update: dict[str, Any],
    name: str,
    t0: float,
    outcome: str,
    stats_before: dict[str, Any] | None,
    llm_stats: Any,
    error: Exception | None,
) -> None:
    """Build and log a node event (after node returns/raises)."""
    duration_ms = round((time.perf_counter() - t0) * 1000, 1)
    ts = utc_now_iso()

    # Attempt from the node's own history event, or state fallback
    attempt = state.get("attempt", 0)
    if update and update.get("history"):
        first_hist = update["history"][0]
        if isinstance(first_hist, dict) and "attempt" in first_hist:
            attempt = int(first_hist["attempt"])

    # Detail: the node's own first history event summary
    detail: dict[str, Any] = {}
    if update and update.get("history"):
        first_hist = update["history"][0]
        if isinstance(first_hist, dict):
            detail = first_hist.get("summary", {})

    # Usage from update
    usage = update.get("token_usage") or {}

    # Cumulative usage: merge state + update
    state_usage = state.get("token_usage") or {}
    cumulative_usage = merge_usage(state_usage, usage) if outcome == "ok" else state_usage

    # LLM client stats delta
    llm_client: dict[str, Any] = {}
    if stats_before is not None and llm_stats is not None and hasattr(llm_stats, "stats"):
        try:
            stats_after = dict(llm_stats.stats)
            calls = stats_after.get("calls", 0) - stats_before.get("calls", 0)
            api_retries = stats_after.get("api_retries", 0) - stats_before.get("api_retries", 0)
            wait_s = stats_after.get("wait_s", 0.0) - stats_before.get("wait_s", 0.0)
            llm_client = {
                "calls": calls,
                "api_retries": api_retries,
                "wait_s": round(wait_s, 3),
            }
        except Exception:
            llm_client = {}

    event = {
        "run_id": logger.run_id,
        "thread_id": state.get("task_id"),
        "task_id": state.get("task_id"),
        "node": name,
        "attempt": attempt,
        "ts": ts,
        "duration_ms": duration_ms,
        "outcome": outcome,
        "detail": detail,
        "usage": usage,
        "cumulative_usage": cumulative_usage,
        "llm_client": llm_client,
    }

    if outcome == "failed":
        failure_reason = update.get("failure_reason")
        if isinstance(failure_reason, str) and len(failure_reason) > 300:
            failure_reason = failure_reason[:300]
        event["failure_reason"] = failure_reason
    elif outcome == "error" and error is not None:
        event["error"] = type(error).__name__
        error_msg = str(error)
        if len(error_msg) > 300:
            error_msg = error_msg[:300]
        event["error_message"] = error_msg

    logger.log_event(event)
