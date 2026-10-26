"""Composition policy for the final straight-boundary decision."""

from __future__ import annotations

from dataclasses import dataclass, field

from surface_passability.policies.core_fit import ContinuousCorePolicy
from surface_passability.policies.direction import RunwiseDirectionPolicy
from surface_passability.policies.geometric_gap import GeometricGapPolicy
from surface_passability.policies.vegetation import (
    GroundedVegetationCapPolicy,
)


@dataclass(frozen=True)
class StraightBoundaryDecisionPolicy:
    """Own the four independently validated boundary decision policies."""

    geometric_gap: GeometricGapPolicy = field(
        default_factory=GeometricGapPolicy
    )
    direction: RunwiseDirectionPolicy = field(
        default_factory=RunwiseDirectionPolicy
    )
    continuous_core: ContinuousCorePolicy = field(
        default_factory=ContinuousCorePolicy
    )
    vegetation_cap: GroundedVegetationCapPolicy = field(
        default_factory=GroundedVegetationCapPolicy
    )
