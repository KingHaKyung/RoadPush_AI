"""Composition policy for the complete surface-boundary analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

from surface_passability.policies.boundary_decision import (
    StraightBoundaryDecisionPolicy,
)
from surface_passability.policies.containment import BoundaryContainmentPolicy
from surface_passability.policies.extension import BoundaryExtensionPolicy
from surface_passability.policies.grading import BoundaryQualityPolicy
from surface_passability.policies.safe_prefix import SafePrefixPolicy


def _final_extension_policy() -> BoundaryExtensionPolicy:
    return BoundaryExtensionPolicy(
        stop_before_side_frame_contact=True,
        minimum_resolvable_width_px=12,
        minimum_width_uncertainty_multiplier=2.0,
    )


@dataclass(frozen=True)
class BoundaryAnalysisPolicy:
    """Own every independently verified policy used by Phase 7A."""

    straight: StraightBoundaryDecisionPolicy = field(
        default_factory=StraightBoundaryDecisionPolicy
    )
    extension: BoundaryExtensionPolicy = field(
        default_factory=_final_extension_policy
    )
    containment: BoundaryContainmentPolicy = field(
        default_factory=BoundaryContainmentPolicy
    )
    safe_prefix: SafePrefixPolicy = field(default_factory=SafePrefixPolicy)
    grading: BoundaryQualityPolicy = field(
        default_factory=BoundaryQualityPolicy
    )
