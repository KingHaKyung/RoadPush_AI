"""Road-specific surface candidate construction and serialization."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping

import numpy as np

from surface_passability.contracts.boundaries import (
    SidewalkBoundaryMode,
    SidewalkBoundaryResult,
)
from surface_passability.contracts.obstacles import (
    ObstacleGeometry,
    ObstacleOwnershipResult,
)
from surface_passability.contracts.road import RoadContext
from surface_passability.contracts.road_analysis import (
    RoadBoundaryPreparation,
    RoadRowProcessing,
    RoadSurfacePreparation,
)
from surface_passability.contracts.row_evidence import RoadRowEvaluation
from surface_passability.contracts.sparse_support import (
    SparseRowSpanSupportResult,
)
from surface_passability.contracts.surfaces import (
    ObstacleOwnershipSurfaceInput,
    StructuralRouteStageResult,
    SurfaceComponentSelection,
    SurfaceComponentSelectionInput,
    SurfaceObstacleSelectionResult,
    SurfacePipelineContext,
)
from surface_passability.io.hashing import hash_mask
from surface_passability.obstacles.aggregation import evaluate_obstacle_ownership
from surface_passability.policies.obstacles import (
    BANNER_SUPPORT_GROUP_MODE,
    WHEELED_SILHOUETTE_ANGLE_MAX_DEGREES,
)
from surface_passability.policies.road import ROAD_DRAIN_RESTORATION_POLICY_ID
from surface_passability.surfaces.selection import (
    select_surface_components_from_obstacle_ownership,
)
from surface_passability.surfaces.row_evidence import evaluate_road_rows
from surface_passability.surfaces.sidewalk_boundary import (
    construct_sidewalk_boundary,
)
from surface_passability.surfaces.sparse_support import (
    apply_sparse_row_span_support,
    sparse_row_span_support_record,
)
from surface_passability.surfaces.structural_route_stage import (
    build_structural_route_stage,
)


def evaluate_road_obstacles(
    geometry: ObstacleGeometry,
    surface: ObstacleOwnershipSurfaceInput,
) -> ObstacleOwnershipResult:
    """Build the accepted Road-specific view of shared obstacle evidence."""

    return evaluate_obstacle_ownership(
        geometry,
        surface,
        wheeled_silhouette_angle_max_degrees=(
            WHEELED_SILHOUETTE_ANGLE_MAX_DEGREES
        ),
        banner_support_group_mode=BANNER_SUPPORT_GROUP_MODE,
    )


def analyze_road_obstacle_components(
    geometry: ObstacleGeometry,
    ownership_surface: ObstacleOwnershipSurfaceInput,
    class_map: np.ndarray,
    selection_surface: SurfaceComponentSelectionInput,
    authorization_forbidden_mask: np.ndarray,
) -> SurfaceObstacleSelectionResult:
    """Evaluate Road ownership once, then select connected components."""

    ownership = evaluate_road_obstacles(geometry, ownership_surface)
    selection = select_surface_components_from_obstacle_ownership(
        ownership,
        class_map,
        selection_surface,
        authorization_forbidden_mask=authorization_forbidden_mask,
    )
    return SurfaceObstacleSelectionResult(
        ownership=ownership,
        component_selection=selection,
    )


def prepare_road_boundaries(
    base_context: SurfacePipelineContext,
    road: RoadContext,
    obstacle_selection: SurfaceObstacleSelectionResult,
) -> RoadBoundaryPreparation:
    """Construct Road-local current and obstacle-connected boundaries."""

    context = SurfacePipelineContext(
        image=base_context.image,
        class_map=base_context.class_map,
        surface=road.surface,
    )
    selection = obstacle_selection.component_selection
    current_boundary = construct_sidewalk_boundary(
        context,
        SidewalkBoundaryMode.ALL_CANDIDATE_ROW_SPAN,
    )
    proposed_boundary = SidewalkBoundaryResult(
        mask=selection.span_mask & ~road.surface_exclusion_mask,
        additions=(
            selection.span_mask & ~road.surface.observed_component
        ),
        mode=SidewalkBoundaryMode.ALL_CANDIDATE_ROW_SPAN,
        calls={
            **current_boundary.calls,
            "all_candidate_outer_row_span": False,
            "obstacle_connected_component_selection": True,
            "forbidden_crossing_check": True,
        },
    )
    return RoadBoundaryPreparation(
        context=context,
        obstacle_selection=obstacle_selection,
        current_boundary=current_boundary,
        proposed_boundary=proposed_boundary,
    )


def prepare_road_structural_stage(
    preparation: RoadBoundaryPreparation,
    road: RoadContext,
) -> StructuralRouteStageResult:
    """Build the Road structural stage and apply its surface exclusion."""

    stage = build_structural_route_stage(
        preparation.proposed_boundary,
        preparation.context,
    )
    stage.pruned_boundary_mask &= ~road.surface_exclusion_mask
    return stage


def evaluate_road_structural_rows(
    stage: StructuralRouteStageResult,
    selection: SurfaceComponentSelection,
    obstacles: ObstacleOwnershipResult,
    context: SurfacePipelineContext,
) -> RoadRowProcessing:
    """Evaluate Road rows before and after sparse structural support."""

    initial = evaluate_road_rows(
        stage.pruned_boundary_mask,
        obstacles,
        context,
        stage.route,
        stage.route_path,
    )
    sparse_support = apply_sparse_row_span_support(
        selection.selected_source_mask,
        initial.component_mask,
        stage.route_path,
    )
    stage.pruned_boundary_mask = sparse_support.after_structural
    final = evaluate_road_rows(
        stage.pruned_boundary_mask,
        obstacles,
        context,
        stage.route,
        stage.route_path,
        component_already_selected=True,
    )
    return RoadRowProcessing(
        initial_evaluation=initial,
        sparse_support=sparse_support,
        final_evaluation=final,
    )


def update_road_runtime_capture(
    runtime_capture: MutableMapping[str, object] | None,
    evaluation: RoadRowEvaluation,
    passability_instance_records: list[dict[str, object]],
) -> None:
    """Update the finalized Road runtime handoff without recomputation."""

    if runtime_capture is not None:
        runtime_capture.update(
            {
                "structural_mask": evaluation.component_mask.copy(),
                "free_mask": evaluation.free_mask.copy(),
                "blocker_mask": evaluation.blocker_mask.copy(),
                "row_reliability": [
                    dict(row.to_legacy_record())
                    for row in evaluation.rows
                ],
                "passability_instances": passability_instance_records,
            }
        )


def build_road_candidate_mask_manifest(
    road: RoadContext,
    selection: SurfaceComponentSelection,
    evaluation: RoadRowEvaluation,
) -> dict[str, np.ndarray]:
    """Build the ordered frozen Road candidate mask manifest without I/O."""

    return {
        "01_raw_road.png": road.raw_road_mask,
        "02_raw_service_lane.png": road.raw_service_lane_mask,
        "03_raw_parking.png": road.raw_parking_mask,
        "04_accepted_repairs.png": road.accepted_repair_mask,
        "05_repaired_road_union.png": road.surface.repaired_mask,
        "06_road_forbidden.png": road.road_forbidden_mask,
        "07_selected_surface_sources.png": selection.selected_source_mask,
        "08_structural_road.png": evaluation.component_mask,
        "09_road_blocker.png": evaluation.blocker_mask,
        "10_free_road.png": evaluation.free_mask,
        "11_bike_lane_observed.png": road.bike_lane_mask,
    }


def serialize_road_candidate_result_record(
    sample: Mapping[str, object],
    display_order: int,
    road: RoadContext,
    selection: SurfaceComponentSelection,
    stage: StructuralRouteStageResult,
    sparse_support: SparseRowSpanSupportResult,
    obstacles: ObstacleOwnershipResult,
    evaluation: RoadRowEvaluation,
) -> dict[str, object]:
    """Serialize the frozen Road candidate result without artifact I/O."""

    repair_records = road.repair_records()
    accepted_repairs = [
        record for record in repair_records if record["accepted"]
    ]
    return {
        "sample_id": sample["sample_id"],
        "display_order": display_order,
        "split": sample.get("split"),
        "surface": "road",
        "source_pixels": {
            "road": int(road.raw_road_mask.sum()),
            "service_lane": int(road.raw_service_lane_mask.sum()),
            "parking": int(road.raw_parking_mask.sum()),
        },
        "repair_policy": road.policy.to_legacy_record(),
        "repair_records": repair_records,
        "accepted_repair_components": len(accepted_repairs),
        "accepted_repair_pixels": int(road.accepted_repair_mask.sum()),
        "trusted_sidewalk_guard_pixels": int(
            road.trusted_sidewalk_guard_mask.sum()
        ),
        "road_forbidden_pixels": int(road.road_forbidden_mask.sum()),
        "surface_exclusion_pixels": int(
            road.surface_exclusion_mask.sum()
        ),
        "bike_lane_pixels": int(road.bike_lane_mask.sum()),
        "bike_lane_included_in_road": False,
        "candidate_component_count": len(selection.components),
        "main_component_index": selection.main_component_index,
        "accepted_component_indices": list(
            selection.accepted_component_indices
        ),
        "accepted_satellite_count": len(selection.evidence),
        "accepted_satellite_pixels": int(
            selection.accepted_satellite_mask.sum()
        ),
        "component_evidence": [
            item.to_legacy_record() for item in selection.evidence
        ],
        "forbidden_crossing_rejections": (
            selection.forbidden_crossing_rejections
        ),
        "structural_road_pixels": int(evaluation.component_mask.sum()),
        "road_blocker_pixels": int(evaluation.blocker_mask.sum()),
        "free_road_pixels": int(evaluation.free_mask.sum()),
        "structural_forbidden_overlap_pixels": int(
            (evaluation.component_mask & road.road_forbidden_mask).sum()
        ),
        "free_forbidden_overlap_pixels": int(
            (evaluation.free_mask & road.road_forbidden_mask).sum()
        ),
        "structural_bike_lane_overlap_pixels": int(
            (evaluation.component_mask & road.bike_lane_mask).sum()
        ),
        "free_bike_lane_overlap_pixels": int(
            (evaluation.free_mask & road.bike_lane_mask).sum()
        ),
        "structural_road_sha256": hash_mask(evaluation.component_mask),
        "road_blocker_sha256": hash_mask(evaluation.blocker_mask),
        "free_road_sha256": hash_mask(evaluation.free_mask),
        "side_removed_pixels": int(stage.approved_removal_mask.sum()),
        "sparse_row_span_support": sparse_row_span_support_record(
            sparse_support
        ),
        "boundary_uncertain_row_count": (
            evaluation.boundary_uncertain_row_count
        ),
        "branch_uncertain_row_count": evaluation.branch_uncertain_row_count,
        "rows": [row.to_legacy_record() for row in evaluation.rows],
        "blocker_policy": obstacles.policy.to_legacy_record(),
        "yolo": obstacles.vehicle_projection.to_legacy_metadata(),
        "rule_violations": list(obstacles.violations),
        "intersection_logic_present": False,
        "scalar_width_summary": None,
    }


def serialize_road_diagnostics_record(
    sample: Mapping[str, object],
    obstacles: ObstacleOwnershipResult,
) -> dict[str, object]:
    """Serialize the frozen Road execution diagnostics."""

    return {
        "sample_id": sample["sample_id"],
        "ownership_evaluations": 1,
        "owned_object_record_count": len(obstacles.records),
        "surface_yolo_triggered": obstacles.vehicle_projection.triggered,
    }


def pair_road_result_and_diagnostics(
    result: dict[str, object],
    diagnostics: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Return the frozen Road output pair without copying or adaptation."""

    return result, diagnostics


def serialize_road_surface_diagnostics(
    preparation: RoadSurfacePreparation,
) -> dict[str, object]:
    """Serialize final Road drain and ownership decisions without artifact I/O."""

    record: dict[str, object] = {
        "policy_id": ROAD_DRAIN_RESTORATION_POLICY_ID,
    }
    road_policy = preparation.road_policy_record()
    record["order"] = road_policy["enhanced_policy_order"]

    linear_drain: dict[str, object] = {}
    linear_drain["approved_pixels"] = int(
        preparation.linear_drain.approved_linear_drain_mask.sum()
    )
    component_records = preparation.linear_drain.component_records()
    linear_drain["accepted_components"] = [
        component
        for component in component_records
        if component["accepted"]
    ]
    linear_drain["policy"] = (
        preparation.linear_drain.policy.to_legacy_record()
    )
    record["linear_drain"] = linear_drain

    ownership: dict[str, object] = {}
    diagnostics = preparation.ownership.legacy_diagnostics()
    ownership["status"] = diagnostics["status"]
    ownership["scene_veto_reason"] = diagnostics["scene_veto_reason"]
    ownership["approved_removal_pixels"] = int(
        preparation.ownership.approved_removal_mask.sum()
    )
    ownership["route_corridor_removed_pixels"] = diagnostics[
        "route_corridor_removed_pixels"
    ]
    ownership["diagnostics"] = diagnostics
    record["ownership"] = ownership
    return record


def build_road_diagnostic_mask_manifest(
    preparation: RoadSurfacePreparation,
) -> dict[str, np.ndarray]:
    """Build the ordered frozen Road diagnostic mask manifest without I/O."""

    return {
        "12_raw_catch_basin.png": (
            preparation.linear_drain.raw_catch_basin_mask
        ),
        "13_estimated_drain_corridor.png": (
            preparation.linear_drain.estimated_corridor_mask
        ),
        "14_approved_linear_drain.png": (
            preparation.linear_drain.approved_linear_drain_mask
        ),
        "15_rejected_linear_drain.png": (
            preparation.linear_drain.rejected_linear_drain_mask
        ),
        "16_ownership_boundary_evidence.png": (
            preparation.ownership.boundary_evidence_mask
        ),
        "17_ownership_persistent_boundary.png": (
            preparation.ownership.persistent_boundary_mask
        ),
        "18_ownership_approved_removal.png": (
            preparation.ownership.approved_removal_mask
        ),
        "19_ownership_uncertain.png": (
            preparation.ownership.uncertain_mask
        ),
        "20_ownership_veto.png": preparation.ownership.veto_mask,
    }
