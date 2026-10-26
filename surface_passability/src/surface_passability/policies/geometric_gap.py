"""Thresholds for conservative one-sided geometric-gap demotion."""

from __future__ import annotations

from dataclasses import dataclass

from surface_passability.policies.straight_boundary import StraightBoundaryPolicy


@dataclass(frozen=True)
class GeometricGapPolicy(StraightBoundaryPolicy):
    """Require stable pre/post geometry before demoting a support gap."""

    minimum_global_candidate_bands: int = 3
    deviation_tolerance_fraction: float = 0.02
    opposite_stability_tolerance_fraction: float = 0.02
    terminal_severe_deviation_multiplier: float = 4.0
    minimum_interior_context_bands_per_side: int = 3
    preferred_context_bands_per_side: int = 5
    minimum_terminal_core_bands: int = 5
    maximum_context_angle_delta_degrees: float = 12.0
    maximum_context_prediction_delta_fraction: float = 0.03
