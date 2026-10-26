"""Typed immutable input for boundary extension decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class BoundaryObservationSnapshot:
    """Freeze only the accepted straight-boundary evidence used downstream."""

    sample_id: str
    selected_surface: str | None
    straight_disposition: str
    straight_reasons: tuple[str, ...]
    image_shape: tuple[int, int]
    component_y_range: tuple[int, int] | None
    support_y_range: tuple[int, int] | None
    selected_band_centers: tuple[int, ...]
    left_slope: float | None
    left_intercept: float | None
    right_slope: float | None
    right_intercept: float | None
    left_seed_p90_residual_px: float | None
    right_seed_p90_residual_px: float | None
    row_evidence: tuple[Mapping[str, Any], ...]
    direction_classification: str | None
    hard_veto_applied: bool
    blocker_sha256: str | None
