from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from roadpush_ai import legacy_policy as legacy
from roadpush_ai.boundary.sidewalk import BoundaryMode, BoundaryResult, construct_boundary
from roadpush_ai.occupancy.obstacle_geometry import (
    ObstacleGeometry,
    evaluate_obstacle_ownership,
    extract_obstacle_geometry,
    prepare_obstacle_geometry,
)
from roadpush_ai.boundary.row_evaluator import evaluate_rows
from roadpush_ai.surface.artifacts import (
    mask_digest,
    obstacle_sources,
    render_sidewalk_review,
)
from roadpush_ai.surface.components import select_obstacle_connected_components
from roadpush_ai.inference.semantic_inputs import SemanticInputs, load_or_infer_inputs
from roadpush_ai.surface.structural_routes import build_structural_route_stage
from roadpush_ai.surface.context import SurfacePipelineContext, build_surface_pipeline_context
from roadpush_ai.surface.sparse_support import (
    apply_sparse_row_span_support,
    sparse_row_span_support_record,
)
from roadpush_ai.contracts.test4_frozen import (
    BANNER_SUPPORT_GROUP_MODE,
    MINIMUM_COMPONENT_PIXELS,
    MINIMUM_COMPONENT_ROWS,
    MINIMUM_CONTACT_PIXELS_EACH_SIDE,
    MINIMUM_FORWARD_PROJECTION_PX,
    MINIMUM_ROUTE_ALIGNMENT_COSINE,
    WHEELED_SILHOUETTE_ANGLE_MAX_DEG,
)


def _evaluate_preselected_sidewalk_component(
    component: np.ndarray,
    blockers: dict,
    pipeline: SurfacePipelineContext,
    route: dict,
    route_path: list[tuple[int, int]],
) -> dict:
    """Evaluate a component already selected by the frozen row evaluator.

    Phase 5 may only subtract rejected additions from that identity. Re-running
    component selection after subtraction would introduce an unreviewed cascade.
    """

    free = component & ~blockers["final_mask"]
    rows = legacy.outer.sample_outer_boundaries(component, route_path, pipeline.labels)
    branch_uncertain = max(0, len(route.get("paths", [])) - 1) > 0
    for row in rows:
        left, right, y = row["left"], row["right"], row["y"]
        segments = legacy.base.free_segments(free, y, left, right)
        selected = legacy.outer.choose_route_segment(segments, row["anchor_x"])
        transient = bool(blockers["transient_mask"][y, left : right + 1].any())
        row.update(
            {
                "structural_sidewalk_span": [left, right],
                "free_segments": [[a, b] for a, b in segments],
                "selected_route_segment": list(selected) if selected else None,
                "transient_occlusion": transient,
                "boundary_reliability": "reliable"
                if row["width_state"] == "complete"
                else "uncertain",
                "branch_reliability": "uncertain"
                if branch_uncertain
                else "no_alternative_branch_detected",
            }
        )
    return {
        "component": component,
        "free": free,
        "rows": rows,
        "transient_occluded_rows": sum(row["transient_occlusion"] for row in rows),
        "boundary_uncertain_rows": sum(
            row["boundary_reliability"] == "uncertain" for row in rows
        ),
        "branch_uncertain_rows": sum(
            row["branch_reliability"] == "uncertain" for row in rows
        ),
    }


def process_sidewalk_sample(
    item: dict,
    display_order: int,
    artifact: Path,
    model_holder: dict,
) -> tuple[dict, dict]:
    """Run one sidewalk sample through the independent generalized call graph."""

    inputs = load_or_infer_inputs(item, model_holder)
    pipeline = build_surface_pipeline_context(inputs.image, inputs.labels)
    geometry = extract_obstacle_geometry(pipeline.image, pipeline.labels)
    geometry = prepare_obstacle_geometry(geometry, [pipeline.surface], model_holder)
    return process_sidewalk_from_prepared(
        item,
        display_order,
        artifact,
        inputs,
        pipeline,
        geometry,
        geometry_extractions=1,
    )


def process_sidewalk_from_prepared(
    item: dict,
    display_order: int,
    artifact: Path,
    inputs: SemanticInputs,
    pipeline: SurfacePipelineContext,
    geometry: ObstacleGeometry,
    *,
    geometry_extractions: int = 0,
    render_review: bool = True,
    runtime_capture: dict | None = None,
) -> tuple[dict, dict]:
    """Evaluate Sidewalk from one coordinator-prepared image and geometry."""

    surface = pipeline.surface
    blockers = evaluate_obstacle_ownership(
        geometry,
        surface,
        wheeled_silhouette_angle_max_deg=WHEELED_SILHOUETTE_ANGLE_MAX_DEG,
        banner_support_group_mode=BANNER_SUPPORT_GROUP_MODE,
    )
    connected_obstacles = obstacle_sources(pipeline, blockers)
    selection = select_obstacle_connected_components(pipeline, connected_obstacles)

    current_boundary = construct_boundary(pipeline, BoundaryMode.TEST3_A2_ROW_SPAN)
    current_stage = build_structural_route_stage(current_boundary, pipeline)
    proposed_boundary = BoundaryResult(
        mask=selection["span"],
        additions=selection["span"] & ~surface.observed_component,
        mode=BoundaryMode.TEST3_A2_ROW_SPAN,
        calls={
            **current_boundary.calls,
            "all_candidate_outer_row_span": False,
            "obstacle_connected_component_selection": True,
        },
    )
    proposed_stage = build_structural_route_stage(proposed_boundary, pipeline)
    initial_evaluated = evaluate_rows(
        proposed_stage.pruned_boundary,
        blockers,
        pipeline,
        proposed_stage.route,
        proposed_stage.route_path,
    )
    sparse_support = apply_sparse_row_span_support(
        selection["selected_sources"],
        proposed_stage.pruned_boundary,
        proposed_stage.route_path,
    )
    proposed_stage.pruned_boundary = sparse_support.after_structural
    evaluation_component = (
        initial_evaluated["component"] & proposed_stage.pruned_boundary
    )
    evaluated = _evaluate_preselected_sidewalk_component(
        evaluation_component,
        blockers,
        pipeline,
        proposed_stage.route,
        proposed_stage.route_path,
    )
    if runtime_capture is not None:
        runtime_capture.update(
            {
                "structural_mask": proposed_stage.pruned_boundary.copy(),
                "free_mask": evaluated["free"].copy(),
                "blocker_mask": (
                    blockers["final_mask"] & proposed_stage.pruned_boundary
                ).copy(),
                "row_reliability": [dict(row) for row in evaluated["rows"]],
                "passability_instances": blockers["passability_instances"],
            }
        )

    artifact.mkdir()
    if render_review:
        render_sidewalk_review(
            artifact,
            item["sample_id"],
            pipeline,
            current_stage,
            proposed_stage,
            selection,
            blockers,
            evaluated,
        )
    masks = {
        "01_candidate_components.png": selection["candidate"],
        "02_main_component.png": selection["main"],
        "03_accepted_satellites.png": selection["accepted_satellites"],
        "04_rejected_components.png": selection["rejected_sources"],
        "05_current_test3_boundary.png": current_stage.raw_boundary,
        "06_proposed_raw_boundary.png": proposed_stage.raw_boundary,
        "07_proposed_pruned_boundary.png": proposed_stage.pruned_boundary,
        "08_applied_blocker.png": blockers["final_mask"]
        & proposed_stage.pruned_boundary,
        "09_final_free.png": evaluated["free"],
    }
    for name, mask in masks.items():
        if not cv2.imwrite(str(artifact / name), mask.astype(np.uint8) * 255):
            raise OSError(f"Failed to write mask: {artifact / name}")

    result = {
        "sample_id": item["sample_id"],
        "display_order": display_order,
        "split": item.get("split"),
        "probe": "test3_plus_obstacle_connected_component_selection",
        "component_policy": {
            "primary": "route_start_or_max_overlap_with_observed_component",
            "satellite_authorization": "same_obstacle_instance_touches_selected_and_forward_satellite",
            "obstacle_contact_radius_px": selection["contact_radius_px"],
            "minimum_component_pixels": MINIMUM_COMPONENT_PIXELS,
            "minimum_component_rows": MINIMUM_COMPONENT_ROWS,
            "minimum_contact_pixels_each_side": MINIMUM_CONTACT_PIXELS_EACH_SIDE,
            "minimum_forward_projection_px": MINIMUM_FORWARD_PROJECTION_PX,
            "minimum_route_alignment_cosine": MINIMUM_ROUTE_ALIGNMENT_COSINE,
            "obstacle_pixels_copied_into_sidewalk": False,
            "iterative_satellite_chaining": True,
        },
        "banner_support_group_mode": BANNER_SUPPORT_GROUP_MODE,
        "candidate_component_count": len(selection["components"]),
        "main_component_index": selection["main_component_index"],
        "accepted_component_indices": selection["accepted_component_indices"],
        "accepted_satellite_count": len(selection["evidence"]),
        "accepted_satellite_pixels": int(selection["accepted_satellites"].sum()),
        "rejected_candidate_pixels": int(selection["rejected_sources"].sum()),
        "component_evidence": selection["evidence"],
        "current_test3_raw_boundary_pixels": int(current_stage.raw_boundary.sum()),
        "proposed_raw_boundary_pixels": int(proposed_stage.raw_boundary.sum()),
        "boundary_removed_vs_current_test3_pixels": int(
            (current_stage.raw_boundary & ~proposed_stage.raw_boundary).sum()
        ),
        "boundary_added_vs_current_test3_pixels": int(
            (proposed_stage.raw_boundary & ~current_stage.raw_boundary).sum()
        ),
        "side_removed_pixels": int(proposed_stage.approved_removal.sum()),
        "sparse_row_span_support": sparse_row_span_support_record(sparse_support),
        "applied_blocker_pixels": int(
            (blockers["final_mask"] & proposed_stage.pruned_boundary).sum()
        ),
        "final_free_pixels": int(evaluated["free"].sum()),
        "final_free_sha256": mask_digest(evaluated["free"]),
        "yolo": blockers["yolo"],
        "blocker_policy": blockers["policy"],
        "rule_violations": blockers["violations"],
        "new_mask2former_inference": bool(inputs.inference_run),
        "scalar_width_summary": None,
    }
    (artifact / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    diagnostics = {
        "sample_id": item["sample_id"],
        "surface_profile": surface.profile.name,
        "semantic_cache": str(inputs.cache_path),
        "semantic_inference_run": bool(inputs.inference_run),
        "semantic_inference": inputs.inference,
        "geometry_extractions": geometry_extractions,
        "ownership_evaluations": 1,
        "geometry_object_count": len(geometry.all_objects),
        "owned_object_record_count": len(blockers["objects"]),
    }
    return result, diagnostics
