"""Shared pytest fixtures for evalcode."""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

import pytest

import evalcode.config as config_module
from evalcode.config import Settings, get_settings

# Every env var that can feed a Settings field (upper-cased field names).
_SETTINGS_ENV_VARS = [name.upper() for name in Settings.model_fields]


@pytest.fixture(autouse=True)
def _env_isolation(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Generator[None, None, None]:
    """Seal unit tests off from the real shell environment and the dev .env.

    - Snapshots ``os.environ`` and restores it exactly afterwards, so
      ``load_dotenv`` side effects can never leak between tests.
    - Removes every Settings-field env var (incl. GEMINI_API_KEY,
      LANGSMITH_*) for the duration of the test.
    - Clears the settings cache before and after the test.
    - Points ``find_dotenv`` (as ``evalcode.config`` imports it) at a path
      that does not exist, so the developer's real ``.env`` is never read;
      ``load_dotenv`` tolerates that value (returns False, sets nothing).

    Tests marked with ``@pytest.mark.live`` skip this isolation entirely so
    they can read the real environment and ``.env`` file.
    """
    if request.node.get_closest_marker("live") is not None:
        get_settings.cache_clear()
        yield
        get_settings.cache_clear()
        return

    snapshot = dict(os.environ)
    for var in _SETTINGS_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(
        config_module,
        "find_dotenv",
        lambda **_kwargs: str(tmp_path / "no-dotenv-here.env"),
    )
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
    os.environ.clear()
    os.environ.update(snapshot)
