"""Accepted Phase-6 surface-selection policy values."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SurfaceSelectionPolicy:
    """Conservative routing policy over already-computed surface evidence."""

    anchor_height_fraction: float = 0.18
    anchor_width_fraction: float = 0.36
    central_width_fraction: float = 0.60
    minimum_source_fraction: float = 0.001
    minimum_structural_fraction: float = 0.004
    minimum_anchor_structural_fraction: float = 0.06
    minimum_vertical_coverage: float = 0.20
    minimum_route_progress: float = 0.15
    image_upload_anchor_margin: float = 0.05
    bilateral_curb_warning_fraction: float = 0.18
    curb_search_width_fraction: float = 0.08
    sidewalk_fill_warning_fraction: float = 0.20
    dominant_sidewalk_source_warning_fraction: float = 0.55
    weak_opposing_road_source_fraction: float = 0.05
    cctv_minimum_sidewalk_component_fraction: float = 0.0025
    cctv_minimum_road_component_fraction: float = 0.0025
    cctv_minimum_component_vertical_span: float = 0.10
    parking_only_is_weak: bool = True
    uncertain_fallback: str = "explicit_mode_required_return_both"
