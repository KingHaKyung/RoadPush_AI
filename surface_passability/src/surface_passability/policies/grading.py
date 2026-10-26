"""Thresholds for boundary-extension quality grading and routing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundaryQualityPolicy:
    """Route accepted geometry without changing its candidate mask."""

    ambiguous_containment_fraction: float = 0.50
    ambiguous_containment_run_rows: int = 4
    ambiguous_retained_fraction: float = 0.15
    retained_fraction_requires_clean_rows_below: int = 4
    maximum_top_bottom_width_ratio: float = 1.05
    partial_direction_classes: tuple[str, ...] = (
        "coherent_turn",
        "one_sided_detour",
    )
