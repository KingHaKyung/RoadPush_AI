"""Aggregate linear transverse-drain analysis for the Road surface."""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from surface_passability.contracts.drainage import (
    LinearDrainAnalysisResult,
    LinearDrainComponentAnalysis,
    LinearDrainComponentEvidence,
    LinearDrainConditions,
    LinearDrainRouteDirectionEvidence,
)
from surface_passability.contracts.road import RoadContext
from surface_passability.contracts.surfaces import SurfaceContext
from surface_passability.policies.drainage import (
    CATCH_BASIN_CLASS_ID,
    LINEAR_DRAIN_MAXIMUM_COMPONENT_FRACTION,
    LINEAR_DRAIN_MAXIMUM_MINOR_SPAN_FRACTION,
    LINEAR_DRAIN_MINIMUM_COMPONENT_PIXELS,
    LINEAR_DRAIN_MINIMUM_CORRIDOR_APPROVED_PIXELS,
    LINEAR_DRAIN_MINIMUM_CORRIDOR_COMPONENT_FRACTION,
    LINEAR_DRAIN_MINIMUM_PCA_ELONGATION,
    LINEAR_DRAIN_MINIMUM_SIDE_SUPPORT_FRACTION,
    LINEAR_DRAIN_MINIMUM_SIDE_SUPPORT_PIXELS,
    LINEAR_DRAIN_MINIMUM_TRANSVERSE_ANGLE_DEGREES,
    LINEAR_DRAIN_STRONG_MINIMUM_COMPONENT_FRACTION,
    LINEAR_DRAIN_STRONG_MINIMUM_PCA_ELONGATION,
    LINEAR_DRAIN_STRONG_MINIMUM_SIDE_SUPPORT_FRACTION,
    LINEAR_DRAIN_STRONG_MINIMUM_TRANSVERSE_ANGLE_DEGREES,
    LINEAR_DRAIN_SUPPORT_DEPTH_FRACTION,
    build_linear_drain_policy,
)
from surface_passability.policies.road import ROAD_UNION_SEAM_RADIUS_PX
from surface_passability.surfaces.drainage import (
    estimate_drain_corridor,
    estimate_drain_route_vector,
    measure_acute_angle_degrees,
    measure_drain_component_geometry,
    measure_drain_frame_contact,
    measure_drain_side_support,
)
from surface_passability.surfaces.routes import (
    select_routed_surface_component,
    solve_surface_route,
)
from surface_passability.surfaces.structure import build_surface_structure


def _analyze_drain_component(
    labels: np.ndarray,
    road: RoadContext,
    road_family: np.ndarray,
    component_index: int,
    components: np.ndarray,
    stats: np.ndarray,
    support_depth: int,
    *,
    restore_whole_accepted_component: bool,
) -> LinearDrainComponentAnalysis | None:
    """Evaluate one connected Catch Basin component in frozen order."""

    component = components == component_index
    pixels = int(component.sum())
    if not pixels:
        return None
    x, y, width, height, _ = [int(value) for value in stats[component_index]]
    geometry = measure_drain_component_geometry(component)
    geometry_record = geometry.to_legacy_record()
    side_support, _ = measure_drain_side_support(
        component,
        road_family,
        geometry_record["minor"],
        support_depth,
    )
    support = [item.to_legacy_record() for item in side_support]
    corridor, corridor_evidence = estimate_drain_corridor(
        labels.shape,
        road_family,
        component,
        float(geometry_record["center"][0]),
    )
    corridor_record = corridor_evidence.to_legacy_record()
    path_route_vector, route_direction_base = estimate_drain_route_vector(
        list(road.surface.route_path),
        float(geometry_record["center"][1]),
        height,
    )
    corridor_route_vector = np.asarray(
        corridor_record.get("route_vector_xy", [0.0, 1.0]),
        np.float64,
    )
    left_slope = corridor_record.get("left_fit_x_per_y", [None])[0]
    right_slope = corridor_record.get("right_fit_x_per_y", [None])[0]
    converging_edges = bool(
        left_slope is not None
        and right_slope is not None
        and float(left_slope) * float(right_slope) <= 0.0
    )
    route_vector = (
        corridor_route_vector if converging_edges else path_route_vector
    )
    route_direction = LinearDrainRouteDirectionEvidence(
        base=route_direction_base,
        selected_estimator=(
            "corridor_center_from_converging_edges"
            if converging_edges
            else "existing_road_route_path"
        ),
        path_route_vector_xy=tuple(path_route_vector.tolist()),
        corridor_route_vector_xy=tuple(corridor_route_vector.tolist()),
        corridor_edges_converge=converging_edges,
    )
    transverse_angle = measure_acute_angle_degrees(
        geometry_record["major"],
        route_vector,
    )
    unclipped_candidate = component & corridor
    forbidden_overlap = component & road.surface_exclusion_mask
    corridor_clipped_candidate = (
        unclipped_candidate & ~road.surface_exclusion_mask
    )
    corridor_clipped_pixels = int(corridor_clipped_candidate.sum())
    component_fraction = float(pixels / labels.size)
    corridor_fraction = float(corridor_clipped_pixels / max(1, pixels))
    frame_contact = measure_drain_frame_contact(component)
    strong_corridor_crossing_exception = bool(
        transverse_angle
        >= LINEAR_DRAIN_STRONG_MINIMUM_TRANSVERSE_ANGLE_DEGREES
        and geometry_record["pca_elongation"]
        >= LINEAR_DRAIN_STRONG_MINIMUM_PCA_ELONGATION
        and support[0]["road_support_fraction"]
        >= LINEAR_DRAIN_STRONG_MINIMUM_SIDE_SUPPORT_FRACTION
        and support[1]["road_support_fraction"]
        >= LINEAR_DRAIN_STRONG_MINIMUM_SIDE_SUPPORT_FRACTION
        and corridor_fraction
        >= LINEAR_DRAIN_STRONG_MINIMUM_COMPONENT_FRACTION
    )

    conditions_record = {
        "minimum_pixels": pixels >= LINEAR_DRAIN_MINIMUM_COMPONENT_PIXELS,
        "long_and_thin": (
            geometry_record["pca_elongation"]
            >= LINEAR_DRAIN_MINIMUM_PCA_ELONGATION
            and geometry_record["minor_span_px"] / min(labels.shape)
            <= LINEAR_DRAIN_MAXIMUM_MINOR_SPAN_FRACTION
            and component_fraction
            <= LINEAR_DRAIN_MAXIMUM_COMPONENT_FRACTION
        ),
        "transverse_to_route": (
            transverse_angle
            >= LINEAR_DRAIN_MINIMUM_TRANSVERSE_ANGLE_DEGREES
            or strong_corridor_crossing_exception
        ),
        "independent_road_support_before": (
            support[0]["road_support_pixels"]
            >= LINEAR_DRAIN_MINIMUM_SIDE_SUPPORT_PIXELS
            and support[0]["road_support_fraction"]
            >= LINEAR_DRAIN_MINIMUM_SIDE_SUPPORT_FRACTION
        ),
        "independent_road_support_after": (
            support[1]["road_support_pixels"]
            >= LINEAR_DRAIN_MINIMUM_SIDE_SUPPORT_PIXELS
            and support[1]["road_support_fraction"]
            >= LINEAR_DRAIN_MINIMUM_SIDE_SUPPORT_FRACTION
        ),
        "corridor_estimated_from_both_sides": (
            corridor_record["status"] == "estimated"
        ),
        "corridor_intersection": (
            corridor_clipped_pixels
            >= LINEAR_DRAIN_MINIMUM_CORRIDOR_APPROVED_PIXELS
            and corridor_fraction
            >= LINEAR_DRAIN_MINIMUM_CORRIDOR_COMPONENT_FRACTION
        ),
        "approved_region_avoids_hard_exclusion": not bool(
            (
                corridor_clipped_candidate
                & road.surface_exclusion_mask
            ).any()
        ),
    }
    accepted = all(conditions_record.values())
    approved = (
        component & ~road.surface_exclusion_mask
        if accepted and restore_whole_accepted_component
        else corridor_clipped_candidate
    )
    approved_pixels = int(approved.sum())
    approved_fraction = float(approved_pixels / max(1, pixels))
    failed = [
        name for name, passed in conditions_record.items() if not passed
    ]
    conditions = LinearDrainConditions(**conditions_record)
    evidence = LinearDrainComponentEvidence(
        component_index=component_index,
        original_class_id=CATCH_BASIN_CLASS_ID,
        original_class_name="Catch Basin",
        decision_type="linear_transverse_drain_probe",
        pixels=pixels,
        component_fraction=component_fraction,
        bbox_xywh=(x, y, width, height),
        bbox_aspect_ratio=float(
            max(width, height) / max(1, min(width, height))
        ),
        geometry=geometry,
        estimated_route_vector_xy=tuple(route_vector.tolist()),
        route_direction=route_direction,
        transverse_angle_deg=transverse_angle,
        frame_contact=frame_contact,
        support_depth_px=support_depth,
        side_support=tuple(side_support),
        corridor=corridor_evidence,
        corridor_pixels_in_component=int(unclipped_candidate.sum()),
        corridor_clipped_candidate_pixels=corridor_clipped_pixels,
        forbidden_overlap_pixels=int(forbidden_overlap.sum()),
        approved_hard_exclusion_overlap_pixels=int(
            (approved & road.surface_exclusion_mask).sum()
        ),
        approved_pixels=approved_pixels if accepted else 0,
        approved_fraction_of_component=(
            approved_fraction if accepted else 0.0
        ),
        whole_object_restoration_enabled=(
            restore_whole_accepted_component
        ),
        whole_object_restored_pixels=(
            int((approved & ~corridor_clipped_candidate).sum())
            if accepted
            else 0
        ),
        already_in_production_repair_pixels=int(
            (component & road.accepted_repair_mask).sum()
        ),
        conditions=conditions,
        base_transverse_condition_passed=(
            transverse_angle
            >= LINEAR_DRAIN_MINIMUM_TRANSVERSE_ANGLE_DEGREES
        ),
        strong_corridor_crossing_exception_passed=(
            strong_corridor_crossing_exception
        ),
        accepted=accepted,
        decision_reasons=tuple(
            (
                [
                    "strong_corridor_crossing_exception"
                    if strong_corridor_crossing_exception
                    and transverse_angle
                    < LINEAR_DRAIN_MINIMUM_TRANSVERSE_ANGLE_DEGREES
                    else "all_linear_transverse_drain_conditions_passed"
                ]
                + (
                    ["whole_accepted_component_restored"]
                    if restore_whole_accepted_component
                    else []
                )
                if accepted
                else [f"failed:{name}" for name in failed]
            )
        ),
    )
    return LinearDrainComponentAnalysis(
        corridor_mask=corridor,
        approved_mask=approved,
        evidence=evidence,
    )


def _build_drain_road_context(
    road: RoadContext,
    approved_linear_drain: np.ndarray,
) -> RoadContext:
    """Rebuild the Road surface after accepted drain restoration."""

    approved_linear_drain &= ~road.surface_exclusion_mask
    repaired_union = road.surface.repaired_mask | approved_linear_drain
    seam_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (ROAD_UNION_SEAM_RADIUS_PX * 2 + 1,) * 2,
    )
    repaired_union = cv2.morphologyEx(
        repaired_union.astype(np.uint8),
        cv2.MORPH_CLOSE,
        seam_kernel,
    ).astype(bool)
    repaired_union &= ~road.surface_exclusion_mask
    structure_result = build_surface_structure(repaired_union)
    structural_mask = structure_result.structural_mask
    structural_addition_mask = structure_result.addition_mask
    structural_mask &= ~road.surface_exclusion_mask
    structural_addition_mask &= ~road.surface_exclusion_mask
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
        profile=road.surface.profile,
        source_mask=source_component,
        repaired_mask=repaired_union,
        structural_mask=structural_mask,
        forbidden_mask=road.surface.forbidden_mask,
        observed_component=observed_component,
        route=route,
        route_path=route.main_path,
        structural_addition_mask=structural_addition_mask,
    )
    return replace(
        road,
        surface=surface,
        repaired_road_mask=(
            road.repaired_road_mask | approved_linear_drain
        ),
        accepted_repair_mask=(
            road.accepted_repair_mask | approved_linear_drain
        ),
    )


def analyze_linear_transverse_drains(
    labels: np.ndarray,
    road: RoadContext,
    *,
    restore_whole_accepted_component: bool = False,
    production_policy: bool = False,
) -> LinearDrainAnalysisResult:
    """Analyze and restore eligible linear transverse-drain components."""

    raw_catch_basin = labels == CATCH_BASIN_CLASS_ID
    road_family = (
        road.raw_road_mask
        | road.raw_service_lane_mask
        | road.raw_parking_mask
    )
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        raw_catch_basin.astype(np.uint8),
        8,
    )
    approved_all = np.zeros_like(raw_catch_basin)
    corridor_all = np.zeros_like(raw_catch_basin)
    evidence: list[LinearDrainComponentEvidence] = []
    support_depth = max(
        6,
        round(min(labels.shape) * LINEAR_DRAIN_SUPPORT_DEPTH_FRACTION),
    )

    for component_index in range(1, count):
        analysis = _analyze_drain_component(
            labels,
            road,
            road_family,
            component_index,
            components,
            stats,
            support_depth,
            restore_whole_accepted_component=(
                restore_whole_accepted_component
            ),
        )
        if analysis is None:
            continue
        corridor_all |= analysis.corridor_mask
        if analysis.evidence.accepted:
            approved_all |= analysis.approved_mask
        evidence.append(analysis.evidence)

    updated_road = _build_drain_road_context(road, approved_all)
    policy = build_linear_drain_policy(
        restore_whole_accepted_component=(
            restore_whole_accepted_component
        ),
        production_policy=production_policy,
    )
    return LinearDrainAnalysisResult(
        road=updated_road,
        raw_catch_basin_mask=raw_catch_basin,
        estimated_corridor_mask=corridor_all,
        approved_linear_drain_mask=approved_all,
        rejected_linear_drain_mask=raw_catch_basin & ~approved_all,
        components=tuple(evidence),
        policy=policy,
    )
