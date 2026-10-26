"""Thresholds for runwise straight-boundary direction consistency."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunwiseDirectionPolicy:
    """Classify coherent turns and unstable straight-boundary runs."""

    material_triplet_turn_degrees: float = 5.0
    coherent_net_turn_degrees: float = 22.0
    coherent_smoothed_spread_degrees: float = 22.0
    minimum_dominant_turn_fraction: float = 0.70
    severe_detour_smoothed_spread_degrees: float = 45.0
    weak_detour_return_excess_degrees: float = 30.0
    bilateral_instability_smoothed_spread_degrees: float = 25.0
    cross_gap_alignment_degrees: float = 12.0
    smoothing_window_bands: int = 5
    minimum_strong_run_bands: int = 6
