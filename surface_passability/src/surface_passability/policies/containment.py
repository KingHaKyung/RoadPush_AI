"""Thresholds for obstacle-aware boundary containment validation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundaryContainmentPolicy:
    """Accept an extension unless persistent signed conflicts remain."""

    minimum_extension_row_fill_p10: float = 0.50
    minimum_in_range_clean_rows: int = 4
    persistent_contradiction_minimum_fraction: float = 0.60
    persistent_contradiction_minimum_run_rows: int = 7
    s2_minimum_outside_clean_rows: int = 6
