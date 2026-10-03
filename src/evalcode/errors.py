"""Shared exception types for evalcode."""

from __future__ import annotations


class EvalcodeError(Exception):
    """Base class for all evalcode errors."""


class ConfigError(EvalcodeError):
    """Raised when configuration is missing or invalid."""
