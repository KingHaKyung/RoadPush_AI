"""Thresholds for direction-local safe extension retention."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SafePrefixPolicy:
    """Trim an unsafe suffix only after a persistent consecutive run."""

    failure_persistence_rows: int = 7
    minimum_retained_extension_rows: int = 3
