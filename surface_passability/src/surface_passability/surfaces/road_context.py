"""Typed pre-obstacle Road surface context construction."""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from surface_passability.contracts.road_analysis import RoadSurfacePreparation
from surface_passability.contracts.road import (
    OwnedRoadContextResult,
    RoadContext,
    RoadOwnershipPolicy,
    SurfaceOwnershipResult,
)
from surface_passability.contracts.surfaces import SurfaceContext
from surface_passability.policies.road import (
    BIKE_LANE_CLASS_ID,
    CURB_CLASS_ID,
    CURB_CUT_CLASS_ID,
    DEFAULT_ROAD_CONTEXT_POLICY,
    DEFAULT_ROAD_OWNERSHIP_POLICY,
    PARKING_CLASS_ID,
    ROAD_CLASS_ID,
    ROAD_SURFACE_PROFILE,
    ROAD_UNION_SEAM_RADIUS_PX,
    SERVICE_LANE_CLASS_ID,
)
from surface_passability.surfaces.road_repair import repair_road_class_locally
from surface_passability.surfaces.linear_drain_analysis import (
    analyze_linear_transverse_drains,
)
from surface_passability.surfaces.road_drain_restoration import (
    restore_approved_linear_drain,
)
from surface_passability.surfaces.road_ownership import evaluate_road_ownership
from surface_passability.surfaces.routes import (
    select_routed_surface_component,
    solve_surface_route,
)
from surface_passability.surfaces.structure import build_surface_structure


def build_road_context(
    class_map: np.ndarray,
    sidewalk: SurfaceContext,
) -> RoadContext:
    """Build the frozen pre-obstacle Road state in production order."""

    trusted_sidewalk_guard = (
        sidewalk.repaired_mask & ~sidewalk.forbidden_mask
    )
    road_forbidden = (
        trusted_sidewalk_guard
        | (class_map == CURB_CLASS_ID)
        | (class_map == CURB_CUT_CLASS_ID)
    )
    bike_lane = class_map == BIKE_LANE_CLASS_ID
    surface_exclusion = road_forbidden | bike_lane

    raw_road = (class_map == ROAD_CLASS_ID) & ~road_forbidden
    raw_service_lane = (
        (class_map == SERVICE_LANE_CLASS_ID) & ~road_forbidden
    )
    raw_parking = (class_map == PARKING_CLASS_ID) & ~road_forbidden
    repaired_road, added_road, road_components = repair_road_class_locally(
        class_map,
        raw_road,
        ROAD_CLASS_ID,
        road_forbidden,
    )
    (
        repaired_service_lane,
        added_service_lane,
        service_lane_components,
    ) = repair_road_class_locally(
        class_map,
        raw_service_lane,
        SERVICE_LANE_CLASS_ID,
        road_forbidden,
    )
    (
        repaired_parking,
        added_parking,
        parking_components,
    ) = repair_road_class_locally(
        class_map,
        raw_parking,
        PARKING_CLASS_ID,
        road_forbidden,
    )

    repaired_union = (
        repaired_road | repaired_service_lane | repaired_parking
    )
    seam_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (ROAD_UNION_SEAM_RADIUS_PX * 2 + 1,) * 2,
    )
    repaired_union = cv2.morphologyEx(
        repaired_union.astype(np.uint8),
        cv2.MORPH_CLOSE,
        seam_kernel,
    ).astype(bool)
    repaired_union &= ~surface_exclusion
    accepted_repair = (
        added_road | added_service_lane | added_parking
    ) & ~surface_exclusion

    structure = build_surface_structure(repaired_union)
    structural_mask = structure.structural_mask
    structural_addition_mask = structure.addition_mask
    structural_mask &= ~surface_exclusion
    structural_addition_mask &= ~surface_exclusion
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
        profile=ROAD_SURFACE_PROFILE,
        source_mask=source_component,
        repaired_mask=repaired_union,
        structural_mask=structural_mask,
        forbidden_mask=surface_exclusion,
        observed_component=observed_component,
        route=route,
        route_path=route.main_path,
        structural_addition_mask=structural_addition_mask,
    )
    return RoadContext(
        surface=surface,
        raw_road_mask=raw_road,
        raw_service_lane_mask=raw_service_lane,
        raw_parking_mask=raw_parking,
        repaired_road_mask=repaired_road,
        repaired_service_lane_mask=repaired_service_lane,
        repaired_parking_mask=repaired_parking,
        accepted_repair_mask=accepted_repair,
        trusted_sidewalk_guard_mask=trusted_sidewalk_guard,
        road_forbidden_mask=road_forbidden,
        surface_exclusion_mask=surface_exclusion,
        bike_lane_mask=bike_lane,
        repair_components=tuple(
            road_components
            + service_lane_components
            + parking_components
        ),
        policy=DEFAULT_ROAD_CONTEXT_POLICY,
    )


def rebuild_owned_road_context(
    class_map: np.ndarray,
    base: RoadContext,
    ownership: SurfaceOwnershipResult,
) -> OwnedRoadContextResult:
    """Rebuild Raw Road after positive ownership, preserving family masks."""

    owned_raw_road = base.raw_road_mask & ownership.ownership_mask
    repaired_road, added_road, road_components = repair_road_class_locally(
        class_map,
        owned_raw_road,
        ROAD_CLASS_ID,
        base.road_forbidden_mask,
    )
    repaired_road &= ownership.ownership_mask
    added_road &= ownership.ownership_mask

    repaired_service_lane = base.repaired_service_lane_mask.copy()
    repaired_parking = base.repaired_parking_mask.copy()
    preserved_family = repaired_service_lane | repaired_parking
    repaired_union = repaired_road | preserved_family
    seam_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (ROAD_UNION_SEAM_RADIUS_PX * 2 + 1,) * 2,
    )
    closed_union = cv2.morphologyEx(
        repaired_union.astype(np.uint8),
        cv2.MORPH_CLOSE,
        seam_kernel,
    ).astype(bool)
    repaired_union = (
        (closed_union & ownership.ownership_mask) | preserved_family
    ) & ~base.surface_exclusion_mask
    accepted_repair = (
        added_road
        | (
            base.repaired_service_lane_mask
            & ~base.raw_service_lane_mask
        )
        | (base.repaired_parking_mask & ~base.raw_parking_mask)
    ) & ~base.surface_exclusion_mask

    structure = build_surface_structure(repaired_union)
    structural_mask = structure.structural_mask
    structural_addition_mask = structure.addition_mask
    structural_allow = ownership.ownership_mask | preserved_family
    structural_mask &= structural_allow & ~base.surface_exclusion_mask
    structural_addition_mask &= (
        structural_allow & ~base.surface_exclusion_mask
    )
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
        profile=ROAD_SURFACE_PROFILE,
        source_mask=source_component,
        repaired_mask=repaired_union,
        structural_mask=structural_mask,
        forbidden_mask=base.surface_exclusion_mask,
        observed_component=observed_component,
        route=route,
        route_path=route.main_path,
        structural_addition_mask=structural_addition_mask,
    )
    ownership_policy = RoadOwnershipPolicy(
        positive_ownership_policy=ownership.profile.policy_name,
        positive_ownership_before_repair=(
            DEFAULT_ROAD_OWNERSHIP_POLICY.positive_ownership_before_repair
        ),
        raw_road_ownership_enabled=(
            DEFAULT_ROAD_OWNERSHIP_POLICY.raw_road_ownership_enabled
        ),
        service_lane_ownership_policy=(
            DEFAULT_ROAD_OWNERSHIP_POLICY.service_lane_ownership_policy
        ),
        parking_ownership_policy=(
            DEFAULT_ROAD_OWNERSHIP_POLICY.parking_ownership_policy
        ),
        obstacle_or_footprint_used_for_ownership=(
            DEFAULT_ROAD_OWNERSHIP_POLICY.obstacle_or_footprint_used_for_ownership
        ),
    )
    road = replace(
        base,
        surface=surface,
        raw_road_mask=owned_raw_road,
        repaired_road_mask=repaired_road,
        repaired_service_lane_mask=repaired_service_lane,
        repaired_parking_mask=repaired_parking,
        accepted_repair_mask=accepted_repair,
        repair_components=tuple(road_components)
        + tuple(
            component
            for component in base.repair_components
            if component.source_class_id != ROAD_CLASS_ID
        ),
    )
    return OwnedRoadContextResult(
        road=road,
        policy=ownership_policy,
    )


def prepare_road_surface_context(
    class_map: np.ndarray,
    sidewalk: SurfaceContext,
) -> RoadSurfacePreparation:
    """Prepare final Road evidence in the accepted one-way policy order."""

    base_road = build_road_context(class_map, sidewalk)
    linear_drain = analyze_linear_transverse_drains(
        class_map,
        base_road,
        restore_whole_accepted_component=True,
        production_policy=True,
    )
    ownership_evidence = replace(
        base_road,
        surface=linear_drain.road.surface,
    )
    ownership = evaluate_road_ownership(class_map, ownership_evidence)
    owned_road = rebuild_owned_road_context(
        class_map,
        base_road,
        ownership,
    )
    restoration = restore_approved_linear_drain(
        owned_road.road,
        linear_drain.approved_linear_drain_mask,
        ownership_policy=owned_road.policy,
    )
    return RoadSurfacePreparation(
        restoration=restoration,
        base_road=base_road,
        linear_drain=linear_drain,
        ownership=ownership,
    )
