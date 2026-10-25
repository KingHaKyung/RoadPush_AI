from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from roadpush_ai import legacy_policy as legacy
from roadpush_ai.boundary.sidewalk import BoundaryMode, BoundaryResult, construct_boundary
from roadpush_ai.occupancy.obstacle_geometry import ObstacleGeometry, evaluate_obstacle_ownership
from roadpush_ai.road.context import RoadContext
from roadpush_ai.surface.structural_routes import build_structural_route_stage
from roadpush_ai.validation.frozen_suite import (
    AMBER,
    CYAN,
    DARK_BLUE,
    GRAY,
    GREEN,
    LIGHT_PURPLE,
    MAGENTA,
    ORANGE,
    RED,
    YELLOW,
    _draw_path,
    contour,
    paint,
)
from roadpush_ai.surface.artifacts import mask_digest, obstacle_sources
from roadpush_ai.surface.components import select_obstacle_connected_components
from roadpush_ai.surface.context import SurfacePipelineContext
from roadpush_ai.surface.sparse_support import (
    apply_sparse_row_span_support,
    sparse_row_span_support_record,
)
from roadpush_ai.contracts.test4_frozen import (
    BANNER_SUPPORT_GROUP_MODE,
    WHEELED_SILHOUETTE_ANGLE_MAX_DEG,
)


def _segments(mask: np.ndarray, y: int, left: int, right: int) -> list[list[int]]:
    return [
        [int(start), int(end)]
        for start, end in legacy.base.free_segments(mask, y, left, right)
    ]


def evaluate_road_rows(
    boundary: np.ndarray,
    blockers: dict,
    pipeline: SurfacePipelineContext,
    route: dict,
    route_path: list[tuple[int, int]],
    *,
    component_already_selected: bool = False,
) -> dict:
    if component_already_selected:
        component = boundary.copy()
    else:
        component = legacy.outer.selected_component(boundary, route_path)
        if not component.any():
            component = boundary.copy()
    applied = blockers["final_mask"] & component
    free = component & ~applied
    rows = legacy.outer.sample_outer_boundaries(component, route_path, pipeline.labels)
    branch_uncertain = max(0, len(route.get("paths", [])) - 1) > 0
    for row in rows:
        left, right, y = row["left"], row["right"], row["y"]
        free_segments = _segments(free, y, left, right)
        occupied_segments = _segments(applied, y, left, right)
        selected = legacy.outer.choose_route_segment(
            [tuple(segment) for segment in free_segments], row["anchor_x"]
        )
        row.update(
            {
                "structural_road_span": [left, right],
                "free_segments": free_segments,
                "occupied_segments": occupied_segments,
                "selected_route_segment": list(selected) if selected else None,
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
        "blocker": applied,
        "occupied": applied,
        "free": free,
        "rows": rows,
        "boundary_uncertain_rows": sum(
            row["boundary_reliability"] == "uncertain" for row in rows
        ),
        "branch_uncertain_rows": sum(
            row["branch_reliability"] == "uncertain" for row in rows
        ),
    }


def _semantic_panel(labels: np.ndarray) -> np.ndarray:
    ids = np.arange(256, dtype=np.uint16)
    palette = np.column_stack(
        ((ids * 53 + 31) % 256, (ids * 97 + 67) % 256, (ids * 193 + 11) % 256)
    ).astype(np.uint8)
    return palette[labels]


def _row_panel(image: np.ndarray, evaluated: dict) -> np.ndarray:
    panel = paint(image, evaluated["free"], GREEN, 0.27)
    panel = paint(panel, evaluated["occupied"], RED, 0.75)
    panel = contour(panel, evaluated["component"], MAGENTA, 2)
    for row in evaluated["rows"]:
        for left, right in row["free_segments"]:
            cv2.line(panel, (left, row["y"]), (right, row["y"]), DARK_BLUE, 4)
    return panel


def render_road_review(
    artifact: Path,
    sample_id: str,
    pipeline: SurfacePipelineContext,
    road: RoadContext,
    selection: dict,
    stage,
    blockers: dict,
    evaluated: dict,
) -> None:
    image = pipeline.image
    raw = paint(image, road.raw_road, (45, 180, 75), 0.55)
    raw = paint(raw, road.raw_service_lane, (60, 150, 255), 0.62)
    raw = paint(raw, road.raw_parking, (245, 190, 45), 0.62)
    repaired = paint(image, road.surface.repaired_mask, GREEN, 0.32)
    repaired = paint(repaired, road.accepted_repair, ORANGE, 0.82)
    forbidden = paint(image, road.trusted_sidewalk_guard, LIGHT_PURPLE, 0.62)
    forbidden = paint(
        forbidden, road.road_forbidden & ~road.trusted_sidewalk_guard, RED, 0.82
    )
    components = paint(image, selection["rejected_sources"], GRAY, 0.65)
    components = paint(components, selection["main"], GREEN, 0.42)
    components = paint(components, selection["accepted_satellites"], CYAN, 0.75)
    structural = paint(image, stage.pruned_boundary, GREEN, 0.34)
    structural = contour(structural, stage.pruned_boundary, MAGENTA, 2)
    _draw_path(structural, stage.route_path, YELLOW, 4)
    pruning = paint(image, stage.raw_candidate, ORANGE, 0.55)
    pruning = paint(pruning, stage.vetoed_candidate, GRAY, 0.75)
    pruning = paint(pruning, stage.approved_removal, RED, 0.85)
    objects = paint(image, blockers["recognized_mask"], LIGHT_PURPLE, 0.48)
    applied = paint(image, stage.pruned_boundary, GREEN, 0.15)
    applied = paint(applied, evaluated["blocker"], AMBER, 0.88)
    final = paint(image, evaluated["component"], (80, 130, 220), 0.22)
    final = paint(final, evaluated["free"], GREEN, 0.50)
    final = paint(final, evaluated["occupied"], RED, 0.82)
    final = contour(final, evaluated["component"], MAGENTA, 2)
    bike = paint(image, road.bike_lane, CYAN, 0.78)
    panels = [
        _semantic_panel(pipeline.labels),
        raw,
        repaired,
        forbidden,
        components,
        structural,
        pruning,
        objects,
        applied,
        final,
        _row_panel(image, evaluated),
        bike,
    ]
    titles = [
        "1 Raw Mapillary semantic classes",
        "2 Raw surface: Road green / Service blue / Parking yellow",
        "3 Repaired road family: orange=accepted repair",
        "4 Road forbidden: purple=sidewalk / red=curb evidence",
        "5 Components: green=main / cyan=accepted / gray=rejected",
        "6 Structural road after pruning + route",
        "7 Side pruning: orange=candidate / gray=veto / red=removed",
        "8 Recognized obstacle geometry",
        "9 Road-owned blocker applied inside structural road",
        "10 Road result: blue=structural / green=free / red=occupied",
        "11 Evaluated free segments and reliability rows",
        "12 Bike Lane observed-only diagnostic (excluded)",
    ]
    fig, axes = plt.subplots(
        3, 4, figsize=(21, 15), constrained_layout=True, facecolor="white"
    )
    for axis, title, panel in zip(axes.flat, titles, panels):
        axis.imshow(panel)
        axis.set_title(title, fontsize=8.4, color="black")
        axis.axis("off")
    fig.suptitle(
        f"{sample_id} | RoadContext Phase 3 | structural "
        f"{int(evaluated['component'].sum()):,}px | free "
        f"{int(evaluated['free'].sum()):,}px | forbidden overlap "
        f"{int((evaluated['component'] & road.road_forbidden).sum())}",
        fontsize=12,
        color="black",
    )
    fig.savefig(artifact / "road_summary.png", dpi=112, facecolor="white")
    plt.close(fig)


def process_road_from_prepared(
    item: dict,
    display_order: int,
    artifact: Path,
    base_pipeline: SurfacePipelineContext,
    road: RoadContext,
    geometry: ObstacleGeometry,
    *,
    render_review: bool = True,
    runtime_capture: dict | None = None,
) -> tuple[dict, dict]:
    pipeline = SurfacePipelineContext(
        image=base_pipeline.image,
        labels=base_pipeline.labels,
        surface=road.surface,
    )
    blockers = evaluate_obstacle_ownership(
        geometry,
        road.surface,
        wheeled_silhouette_angle_max_deg=WHEELED_SILHOUETTE_ANGLE_MAX_DEG,
        banner_support_group_mode=BANNER_SUPPORT_GROUP_MODE,
    )
    connected_obstacles = obstacle_sources(pipeline, blockers)
    selection = select_obstacle_connected_components(
        pipeline,
        connected_obstacles,
        authorization_forbidden_mask=road.road_forbidden,
    )
    current_boundary = construct_boundary(pipeline, BoundaryMode.TEST3_A2_ROW_SPAN)
    proposed_boundary = BoundaryResult(
        mask=selection["span"] & ~road.surface_exclusion,
        additions=selection["span"] & ~road.surface.observed_component,
        mode=BoundaryMode.TEST3_A2_ROW_SPAN,
        calls={
            **current_boundary.calls,
            "all_candidate_outer_row_span": False,
            "obstacle_connected_component_selection": True,
            "forbidden_crossing_check": True,
        },
    )
    stage = build_structural_route_stage(proposed_boundary, pipeline)
    stage.pruned_boundary &= ~road.surface_exclusion
    initial_evaluated = evaluate_road_rows(
        stage.pruned_boundary, blockers, pipeline, stage.route, stage.route_path
    )
    sparse_support = apply_sparse_row_span_support(
        selection["selected_sources"],
        initial_evaluated["component"],
        stage.route_path,
    )
    stage.pruned_boundary = sparse_support.after_structural
    evaluated = evaluate_road_rows(
        stage.pruned_boundary,
        blockers,
        pipeline,
        stage.route,
        stage.route_path,
        component_already_selected=True,
    )
    if runtime_capture is not None:
        runtime_capture.update(
            {
                "structural_mask": evaluated["component"].copy(),
                "free_mask": evaluated["free"].copy(),
                "blocker_mask": evaluated["blocker"].copy(),
                "row_reliability": [dict(row) for row in evaluated["rows"]],
                "passability_instances": blockers["passability_instances"],
            }
        )

    artifact.mkdir()
    masks = {
        "01_raw_road.png": road.raw_road,
        "02_raw_service_lane.png": road.raw_service_lane,
        "03_raw_parking.png": road.raw_parking,
        "04_accepted_repairs.png": road.accepted_repair,
        "05_repaired_road_union.png": road.surface.repaired_mask,
        "06_road_forbidden.png": road.road_forbidden,
        "07_selected_surface_sources.png": selection["selected_sources"],
        "08_structural_road.png": evaluated["component"],
        "09_road_blocker.png": evaluated["blocker"],
        "10_free_road.png": evaluated["free"],
        "11_bike_lane_observed.png": road.bike_lane,
    }
    for name, mask in masks.items():
        if not cv2.imwrite(str(artifact / name), mask.astype(np.uint8) * 255):
            raise OSError(f"Failed to write mask: {artifact / name}")
    if render_review:
        render_road_review(
            artifact,
            item["sample_id"],
            pipeline,
            road,
            selection,
            stage,
            blockers,
            evaluated,
        )

    accepted_repairs = [row for row in road.repair_records if row["accepted"]]
    result = {
        "sample_id": item["sample_id"],
        "display_order": display_order,
        "split": item.get("split"),
        "surface": "road",
        "source_pixels": {
            "road": int(road.raw_road.sum()),
            "service_lane": int(road.raw_service_lane.sum()),
            "parking": int(road.raw_parking.sum()),
        },
        "repair_policy": road.policy,
        "repair_records": road.repair_records,
        "accepted_repair_components": len(accepted_repairs),
        "accepted_repair_pixels": int(road.accepted_repair.sum()),
        "trusted_sidewalk_guard_pixels": int(road.trusted_sidewalk_guard.sum()),
        "road_forbidden_pixels": int(road.road_forbidden.sum()),
        "surface_exclusion_pixels": int(road.surface_exclusion.sum()),
        "bike_lane_pixels": int(road.bike_lane.sum()),
        "bike_lane_included_in_road": False,
        "candidate_component_count": len(selection["components"]),
        "main_component_index": selection["main_component_index"],
        "accepted_component_indices": selection["accepted_component_indices"],
        "accepted_satellite_count": len(selection["evidence"]),
        "accepted_satellite_pixels": int(selection["accepted_satellites"].sum()),
        "component_evidence": selection["evidence"],
        "forbidden_crossing_rejections": selection["forbidden_crossing_rejections"],
        "structural_road_pixels": int(evaluated["component"].sum()),
        "road_blocker_pixels": int(evaluated["blocker"].sum()),
        "free_road_pixels": int(evaluated["free"].sum()),
        "structural_forbidden_overlap_pixels": int(
            (evaluated["component"] & road.road_forbidden).sum()
        ),
        "free_forbidden_overlap_pixels": int(
            (evaluated["free"] & road.road_forbidden).sum()
        ),
        "structural_bike_lane_overlap_pixels": int(
            (evaluated["component"] & road.bike_lane).sum()
        ),
        "free_bike_lane_overlap_pixels": int(
            (evaluated["free"] & road.bike_lane).sum()
        ),
        "structural_road_sha256": mask_digest(evaluated["component"]),
        "road_blocker_sha256": mask_digest(evaluated["blocker"]),
        "free_road_sha256": mask_digest(evaluated["free"]),
        "side_removed_pixels": int(stage.approved_removal.sum()),
        "sparse_row_span_support": sparse_row_span_support_record(sparse_support),
        "boundary_uncertain_row_count": evaluated["boundary_uncertain_rows"],
        "branch_uncertain_row_count": evaluated["branch_uncertain_rows"],
        "rows": evaluated["rows"],
        "blocker_policy": blockers["policy"],
        "yolo": blockers["yolo"],
        "rule_violations": blockers["violations"],
        "intersection_logic_present": False,
        "scalar_width_summary": None,
    }
    (artifact / "road_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    diagnostic = {
        "sample_id": item["sample_id"],
        "ownership_evaluations": 1,
        "owned_object_record_count": len(blockers["objects"]),
        "surface_yolo_triggered": blockers["yolo"]["triggered"],
    }
    return result, diagnostic
