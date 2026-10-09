"""SQLite checkpointer helper (Task 9).

Provides a context manager that yields a ``SqliteSaver`` backed by a file or
``:memory:``. The connection is closed on exit, and on Windows the database
file is deletable afterwards.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

__all__ = ["open_checkpointer"]


@contextmanager
def open_checkpointer(db_path: str | Path) -> Iterator[SqliteSaver]:
    """Context manager yielding a ``SqliteSaver`` for the given path.

    ``db_path`` may be a file path (str or Path) or ``":memory:"`` for an
    in-memory database. For file paths, parent directories are created if
    they don't exist. The SQLite connection is closed on exit, ensuring the
    file is not locked (important on Windows for subsequent deletion).
    """
    path_str = str(db_path)
    if path_str != ":memory:":
        Path(path_str).parent.mkdir(parents=True, exist_ok=True)

    # SqliteSaver.from_conn_string is itself a context manager that yields the saver.
    with SqliteSaver.from_conn_string(path_str) as saver:
        yield saver
