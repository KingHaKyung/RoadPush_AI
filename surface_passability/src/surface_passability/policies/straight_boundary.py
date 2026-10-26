"""Final straight-boundary candidate thresholds."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StraightBoundaryPolicy:
    """Accepted thresholds for the base straight-boundary candidate."""

    minimum_span_px: int = 12
    minimum_span_fraction: float = 0.02
    clean_fill_ratio: float = 0.80
    band_rows: int = 7
    minimum_passing_rows_per_band: int = 5
    endpoint_contact_radius_px: int = 6
    minimum_clean_cap_bands: int = 2
    minimum_anchor_rows: int = 12
    minimum_segment_median_span_fraction: float = 0.10
    fit_inlier_tolerance_fraction: float = 0.02
    minimum_inlier_fraction: float = 0.80
    maximum_median_residual_fraction: float = 0.015
    maximum_p90_residual_fraction: float = 0.03
    minimum_median_envelope_evidence: float = 0.70
    minimum_corridor_support_rows: int = 5
    measurement_band_rows: int = 3
    blocker_suspect_added_area_fraction: float = 0.05
