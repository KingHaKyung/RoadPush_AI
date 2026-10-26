"""Replay an approved transverse drain into an owned Road context."""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from surface_passability.contracts.road import (
    RoadContext,
    RoadDrainRestorationResult,
    RoadOwnershipPolicy,
)
from surface_passability.contracts.surfaces import SurfaceContext
from surface_passability.policies.road import (
    DEFAULT_ROAD_DRAIN_RESTORATION_POLICY,
    DEFAULT_ROAD_OWNERSHIP_POLICY,
    ROAD_UNION_SEAM_RADIUS_PX,
)
from surface_passability.surfaces.routes import (
    select_routed_surface_component,
    solve_surface_route,
)
from surface_passability.surfaces.structure import build_surface_structure


def restore_approved_linear_drain(
    owned_road: RoadContext,
    approved_drain_mask: np.ndarray,
    *,
    ownership_policy: RoadOwnershipPolicy = DEFAULT_ROAD_OWNERSHIP_POLICY,
) -> RoadDrainRestorationResult:
    """Replay prior drain approval without reevaluating its eligibility."""

    approved = approved_drain_mask & ~owned_road.surface_exclusion_mask
    repaired_union = owned_road.surface.repaired_mask | approved
    seam_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (ROAD_UNION_SEAM_RADIUS_PX * 2 + 1,) * 2,
    )
    repaired_union = cv2.morphologyEx(
        repaired_union.astype(np.uint8),
        cv2.MORPH_CLOSE,
        seam_kernel,
    ).astype(bool)
    repaired_union &= ~owned_road.surface_exclusion_mask
    structure_result = build_surface_structure(repaired_union)
    structural_mask = structure_result.structural_mask
    structural_addition_mask = structure_result.addition_mask
    structural_mask &= ~owned_road.surface_exclusion_mask
    structural_addition_mask &= ~owned_road.surface_exclusion_mask
    route = solve_surface_route(repaired_union)
    observed_component = select_routed_surface_component(
        structural_mask,
        route.main_path,
    )
    if not observed_component.any():
        observed_component = structural_mask.copy()
    source_component = select_routed_surface_component(
        repaired_union,
        route.main_path,
    )
    if not source_component.any():
        source_component = repaired_union.copy()
    surface = SurfaceContext(
        profile=owned_road.surface.profile,
        source_mask=source_component,
        repaired_mask=repaired_union,
        structural_mask=structural_mask,
        forbidden_mask=owned_road.surface.forbidden_mask,
        observed_component=observed_component,
        route=route,
        route_path=route.main_path,
        structural_addition_mask=structural_addition_mask,
    )
    road = replace(
        owned_road,
        surface=surface,
        repaired_road_mask=owned_road.repaired_road_mask | approved,
        accepted_repair_mask=owned_road.accepted_repair_mask | approved,
    )
    return RoadDrainRestorationResult(
        road=road,
        ownership_policy=ownership_policy,
        policy=DEFAULT_ROAD_DRAIN_RESTORATION_POLICY,
    )
