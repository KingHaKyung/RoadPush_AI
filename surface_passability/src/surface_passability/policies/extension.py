"""Thresholds for bounded straight-boundary extension generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundaryExtensionPolicy:
    """Bound projection without searching for new bands or refitting lines."""

    maximum_extension_span_ratio: float = 2.0
    structural_zero_overlap_persistence_rows: int = 7
    stop_before_side_frame_contact: bool = False
    minimum_resolvable_width_px: int | None = None
    minimum_width_uncertainty_multiplier: float = 2.0
