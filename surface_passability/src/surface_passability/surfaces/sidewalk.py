"""Sidewalk semantic-surface construction."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping

import cv2
import numpy as np

from surface_passability.contracts.surfaces import (
    ObstacleOwnershipSurfaceInput,
    SidewalkRepairResult,
    SurfaceComponentSelection,
    SurfaceComponentSelectionInput,
    SurfaceObstacleSelectionResult,
    SurfaceContext,
    SurfacePipelineContext,
    SurfaceRepairComponent,
    StructuralRouteStageResult,
)
from surface_passability.contracts.sparse_support import (
    SparseRowSpanSupportResult,
)
from surface_passability.contracts.sidewalk import (
    SidewalkRowProcessing,
    SidewalkStructuralPreparation,
)
from surface_passability.contracts.row_evidence import SurfaceRowEvaluation
from surface_passability.contracts.semantic import SemanticEvidence
from surface_passability.contracts.obstacles import (
    ObstacleGeometry,
    ObstacleOwnershipResult,
)
from surface_passability.obstacles.aggregation import evaluate_obstacle_ownership
from surface_passability.surfaces.selection import (
    select_surface_components_from_obstacle_ownership,
)
from surface_passability.surfaces.row_evidence import (
    evaluate_preselected_surface_rows,
    evaluate_surface_rows,
)
from surface_passability.surfaces.sidewalk_boundary import (
    construct_sidewalk_boundary,
)
from surface_passability.surfaces.sparse_support import (
    apply_sparse_row_span_support,
    sparse_row_span_support_record,
)
from surface_passability.io.hashing import hash_mask
from surface_passability.surfaces.structural_route_stage import (
    build_structural_route_stage,
)
from surface_passability.contracts.boundaries import (
    SidewalkBoundaryMode,
    SidewalkBoundaryResult,
)
from surface_passability.obstacles.taxonomy import (
    REPAIR_ROAD_CONTEXT_CLASS_IDS,
    SIDEWALK_CLASS_ID,
    SURFACE_REPAIR_CLASS_IDS,
)
from surface_passability.policies.surfaces import (
    SIDEWALK_REPAIR_MINIMUM_RING_RADIUS_PX,
    SIDEWALK_REPAIR_RING_RADIUS_IMAGE_DIVISOR,
    SIDEWALK_REPAIR_RULES,
    SURFACE_COMPONENT_MINIMUM_CONTACT_PIXELS_EACH_SIDE,
    SURFACE_COMPONENT_MINIMUM_FORWARD_PROJECTION_PX,
    SURFACE_COMPONENT_MINIMUM_PIXELS,
    SURFACE_COMPONENT_MINIMUM_ROUTE_ALIGNMENT_COSINE,
    SURFACE_COMPONENT_MINIMUM_ROWS,
)
from surface_passability.policies.obstacles import (
    BANNER_SUPPORT_GROUP_MODE,
    WHEELED_SILHOUETTE_ANGLE_MAX_DEGREES,
)


def repair_sidewalk_surface(class_map: np.ndarray) -> SidewalkRepairResult:
    """Inherit accepted Lane Marking, Catch Basin, and Manhole components."""

    source = class_map == SIDEWALK_CLASS_ID
    repaired = source.copy()
    additions = np.zeros_like(source)
    components: list[SurfaceRepairComponent] = []
    height, width = class_map.shape
    radius = max(
        SIDEWALK_REPAIR_MINIMUM_RING_RADIUS_PX,
        round(min(height, width) / SIDEWALK_REPAIR_RING_RADIUS_IMAGE_DIVISOR),
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (radius * 2 + 1, radius * 2 + 1),
    )

    if tuple(rule.class_id for rule in SIDEWALK_REPAIR_RULES) != (
        SURFACE_REPAIR_CLASS_IDS
    ):
        raise RuntimeError("Sidewalk repair rules and semantic taxonomy disagree")

    for rule in SIDEWALK_REPAIR_RULES:
        binary = (class_map == rule.class_id).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        for component_index, contour in enumerate(contours, start=1):
            x, y, component_width, component_height = cv2.boundingRect(contour)
            padding = radius + 1
            x0, y0 = max(0, x - padding), max(0, y - padding)
            x1 = min(width, x + component_width + padding)
            y1 = min(height, y + component_height + padding)
            local_labels = class_map[y0:y1, x0:x1]
            component = np.zeros(local_labels.shape, dtype=np.uint8)
            shifted = contour.copy()
            shifted[:, 0, 0] -= x0
            shifted[:, 0, 1] -= y0
            cv2.drawContours(component, [shifted], -1, 1, cv2.FILLED)
            component &= (local_labels == rule.class_id).astype(np.uint8)
            pixels = int(component.sum())
            if not pixels:
                continue

            ring = (
                (cv2.dilate(component, kernel) > 0)
                & (component == 0)
                & (local_labels != rule.class_id)
            )
            ring_pixels = int(ring.sum())
            sidewalk_fraction = (
                float(((local_labels == SIDEWALK_CLASS_ID) & ring).sum() / ring_pixels)
                if ring_pixels
                else 0.0
            )
            road_fraction = (
                float(
                    (
                        np.isin(local_labels, REPAIR_ROAD_CONTEXT_CLASS_IDS)
                        & ring
                    ).sum()
                    / ring_pixels
                )
                if ring_pixels
                else 1.0
            )
            component_fraction = pixels / class_map.size
            touches_frame = (
                x == 0
                or y == 0
                or x + component_width == width
                or y + component_height == height
            )
            catch_basin_geometry_ok = (
                rule.maximum_image_fraction is None
                or (
                    component_fraction <= rule.maximum_image_fraction
                    and not (rule.reject_frame_contact and touches_frame)
                )
            )
            accepted = (
                sidewalk_fraction >= rule.minimum_sidewalk_ring_fraction
                and road_fraction <= rule.maximum_road_ring_fraction
                and catch_basin_geometry_ok
            )
            if accepted:
                accepted_mask = component.astype(bool)
                repaired[y0:y1, x0:x1][accepted_mask] = True
                additions[y0:y1, x0:x1][accepted_mask] = True

            components.append(
                SurfaceRepairComponent(
                    class_id=rule.class_id,
                    class_name=rule.class_name,
                    component_index=component_index,
                    pixels=pixels,
                    sidewalk_ring_fraction=sidewalk_fraction,
                    road_ring_fraction=road_fraction,
                    component_fraction=component_fraction,
                    touches_frame=touches_frame,
                    catch_basin_geometry_ok=catch_basin_geometry_ok,
                    accepted=accepted,
                )
            )

    return SidewalkRepairResult(
        source_mask=source,
        repaired_mask=repaired,
        addition_mask=additions,
        components=tuple(components),
    )


def evaluate_sidewalk_obstacles(
    geometry: ObstacleGeometry,
    surface: ObstacleOwnershipSurfaceInput,
) -> ObstacleOwnershipResult:
    """Build the accepted Sidewalk-specific view of shared obstacle evidence."""

    return evaluate_obstacle_ownership(
        geometry,
        surface,
        wheeled_silhouette_angle_max_degrees=(
            WHEELED_SILHOUETTE_ANGLE_MAX_DEGREES
        ),
        banner_support_group_mode=BANNER_SUPPORT_GROUP_MODE,
    )


def analyze_sidewalk_obstacle_components(
    geometry: ObstacleGeometry,
    ownership_surface: ObstacleOwnershipSurfaceInput,
    class_map: np.ndarray,
    selection_surface: SurfaceComponentSelectionInput,
) -> SurfaceObstacleSelectionResult:
    """Evaluate Sidewalk ownership once, then select connected components."""

    ownership = evaluate_sidewalk_obstacles(geometry, ownership_surface)
    selection = select_surface_components_from_obstacle_ownership(
        ownership,
        class_map,
        selection_surface,
    )
    return SurfaceObstacleSelectionResult(
        ownership=ownership,
        component_selection=selection,
    )


def apply_sidewalk_sparse_support(
    selection: SurfaceComponentSelection,
    stage: StructuralRouteStageResult,
) -> SparseRowSpanSupportResult:
    """Apply route-protected sparse support to a selected Sidewalk stage."""

    result = apply_sparse_row_span_support(
        selection.selected_source_mask,
        stage.pruned_boundary_mask,
        stage.route_path,
    )
    stage.pruned_boundary_mask = result.after_structural
    return result


def prepare_sidewalk_structural_stages(
    context: SurfacePipelineContext,
    selection: SurfaceComponentSelection,
) -> SidewalkStructuralPreparation:
    """Build current and obstacle-connected Sidewalk structural stages."""

    current_boundary = construct_sidewalk_boundary(
        context,
        SidewalkBoundaryMode.ALL_CANDIDATE_ROW_SPAN,
    )
    current_stage = build_structural_route_stage(current_boundary, context)
    proposed_boundary = SidewalkBoundaryResult(
        mask=selection.span_mask,
        additions=selection.span_mask & ~context.observed_component,
        mode=SidewalkBoundaryMode.ALL_CANDIDATE_ROW_SPAN,
        calls={
            **current_boundary.calls,
            "all_candidate_outer_row_span": False,
            "obstacle_connected_component_selection": True,
        },
    )
    proposed_stage = build_structural_route_stage(proposed_boundary, context)
    return SidewalkStructuralPreparation(
        current_boundary=current_boundary,
        current_stage=current_stage,
        proposed_boundary=proposed_boundary,
        proposed_stage=proposed_stage,
    )


def evaluate_sidewalk_structural_rows(
    preparation: SidewalkStructuralPreparation,
    selection: SurfaceComponentSelection,
    obstacles: ObstacleOwnershipResult,
    context: SurfacePipelineContext,
) -> SidewalkRowProcessing:
    """Evaluate proposed Sidewalk rows before and after sparse pruning."""

    stage = preparation.proposed_stage
    initial = evaluate_surface_rows(
        stage.pruned_boundary_mask,
        obstacles,
        context,
        stage.route,
        stage.route_path,
    )
    sparse_support = apply_sidewalk_sparse_support(selection, stage)
    evaluation_component = initial.component_mask & stage.pruned_boundary_mask
    final = evaluate_preselected_surface_rows(
        evaluation_component,
        obstacles,
        context,
        stage.route,
        stage.route_path,
    )
    return SidewalkRowProcessing(
        initial_evaluation=initial,
        sparse_support=sparse_support,
        evaluation_component_mask=evaluation_component,
        final_evaluation=final,
    )


def serialize_sidewalk_result_record(
    sample: Mapping[str, object],
    display_order: int,
    semantic: SemanticEvidence,
    selection: SurfaceComponentSelection,
    current_stage: StructuralRouteStageResult,
    proposed_stage: StructuralRouteStageResult,
    sparse_support: SparseRowSpanSupportResult,
    obstacles: ObstacleOwnershipResult,
    evaluation: SurfaceRowEvaluation,
) -> dict[str, object]:
    """Serialize the frozen Sidewalk candidate-result metadata."""

    return {
        "sample_id": sample["sample_id"],
        "display_order": display_order,
        "split": sample.get("split"),
        "probe": "test3_plus_obstacle_connected_component_selection",
        "component_policy": {
            "primary": "route_start_or_max_overlap_with_observed_component",
            "satellite_authorization": (
                "same_obstacle_instance_touches_selected_and_forward_satellite"
            ),
            "obstacle_contact_radius_px": selection.contact_radius_px,
            "minimum_component_pixels": SURFACE_COMPONENT_MINIMUM_PIXELS,
            "minimum_component_rows": SURFACE_COMPONENT_MINIMUM_ROWS,
            "minimum_contact_pixels_each_side": (
                SURFACE_COMPONENT_MINIMUM_CONTACT_PIXELS_EACH_SIDE
            ),
            "minimum_forward_projection_px": (
                SURFACE_COMPONENT_MINIMUM_FORWARD_PROJECTION_PX
            ),
            "minimum_route_alignment_cosine": (
                SURFACE_COMPONENT_MINIMUM_ROUTE_ALIGNMENT_COSINE
            ),
            "obstacle_pixels_copied_into_sidewalk": False,
            "iterative_satellite_chaining": True,
        },
        "banner_support_group_mode": BANNER_SUPPORT_GROUP_MODE,
        "candidate_component_count": len(selection.components),
        "main_component_index": selection.main_component_index,
        "accepted_component_indices": list(
            selection.accepted_component_indices
        ),
        "accepted_satellite_count": len(selection.evidence),
        "accepted_satellite_pixels": int(
            selection.accepted_satellite_mask.sum()
        ),
        "rejected_candidate_pixels": int(
            selection.rejected_source_mask.sum()
        ),
        "component_evidence": [
            item.to_legacy_record() for item in selection.evidence
        ],
        "current_test3_raw_boundary_pixels": int(
            current_stage.raw_boundary_mask.sum()
        ),
        "proposed_raw_boundary_pixels": int(
            proposed_stage.raw_boundary_mask.sum()
        ),
        "boundary_removed_vs_current_test3_pixels": int(
            (
                current_stage.raw_boundary_mask
                & ~proposed_stage.raw_boundary_mask
            ).sum()
        ),
        "boundary_added_vs_current_test3_pixels": int(
            (
                proposed_stage.raw_boundary_mask
                & ~current_stage.raw_boundary_mask
            ).sum()
        ),
        "side_removed_pixels": int(
            proposed_stage.approved_removal_mask.sum()
        ),
        "sparse_row_span_support": sparse_row_span_support_record(
            sparse_support
        ),
        "applied_blocker_pixels": int(
            (obstacles.final_mask & proposed_stage.pruned_boundary_mask).sum()
        ),
        "final_free_pixels": int(evaluation.free_mask.sum()),
        "final_free_sha256": hash_mask(evaluation.free_mask),
        "yolo": obstacles.vehicle_projection.to_legacy_metadata(),
        "blocker_policy": obstacles.policy.to_legacy_record(),
        "rule_violations": list(obstacles.violations),
        "new_mask2former_inference": bool(semantic.inference_run),
        "scalar_width_summary": None,
    }


def update_sidewalk_runtime_capture(
    runtime_capture: MutableMapping[str, object] | None,
    stage: StructuralRouteStageResult,
    obstacles: ObstacleOwnershipResult,
    evaluation: SurfaceRowEvaluation,
    passability_instance_records: list[dict[str, object]],
) -> None:
    """Update the legacy-shaped runtime handoff without recomputation."""

    if runtime_capture is not None:
        runtime_capture.update(
            {
                "structural_mask": stage.pruned_boundary_mask.copy(),
                "free_mask": evaluation.free_mask.copy(),
                "blocker_mask": (
                    obstacles.final_mask & stage.pruned_boundary_mask
                ).copy(),
                "row_reliability": [
                    dict(row.to_legacy_record())
                    for row in evaluation.rows
                ],
                "passability_instances": passability_instance_records,
            }
        )



def serialize_sidewalk_diagnostics_record(
    sample: Mapping[str, object],
    surface: SurfaceContext,
    semantic: SemanticEvidence,
    geometry: ObstacleGeometry,
    obstacles: ObstacleOwnershipResult,
    geometry_extractions: int = 0,
) -> dict[str, object]:
    """Serialize the frozen Sidewalk execution diagnostics."""

    return {
        "sample_id": sample["sample_id"],
        "surface_profile": surface.profile.name,
        "semantic_cache": str(semantic.cache_path),
        "semantic_inference_run": bool(semantic.inference_run),
        "semantic_inference": semantic.inference_metadata,
        "geometry_extractions": geometry_extractions,
        "ownership_evaluations": 1,
        "geometry_object_count": len(geometry.all_objects),
        "owned_object_record_count": len(obstacles.records),
    }


def build_sidewalk_artifact_mask_manifest(
    selection: SurfaceComponentSelection,
    current_stage: StructuralRouteStageResult,
    proposed_stage: StructuralRouteStageResult,
    obstacles: ObstacleOwnershipResult,
    evaluation: SurfaceRowEvaluation,
) -> dict[str, np.ndarray]:
    """Build the ordered frozen mask-artifact manifest without writing it."""

    return {
        "01_candidate_components.png": selection.candidate_mask,
        "02_main_component.png": selection.main_component_mask,
        "03_accepted_satellites.png": selection.accepted_satellite_mask,
        "04_rejected_components.png": selection.rejected_source_mask,
        "05_current_test3_boundary.png": current_stage.raw_boundary_mask,
        "06_proposed_raw_boundary.png": proposed_stage.raw_boundary_mask,
        "07_proposed_pruned_boundary.png": proposed_stage.pruned_boundary_mask,
        "08_applied_blocker.png": (
            obstacles.final_mask & proposed_stage.pruned_boundary_mask
        ),
        "09_final_free.png": evaluation.free_mask,
    }
