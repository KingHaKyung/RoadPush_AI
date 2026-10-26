"""Surface corridor row evaluation."""

from __future__ import annotations

import numpy as np

from surface_passability.boundary.observations import sample_outer_boundary_rows
from surface_passability.contracts.obstacles import ObstacleOwnershipResult
from surface_passability.contracts.row_evidence import (
    RoadRowEvaluation,
    RoadRowEvidence,
    SurfaceRowEvaluation,
    SurfaceRowEvidence,
)
from surface_passability.contracts.surfaces import (
    RecomputedStructuralRoute,
    RouteAttempt,
    SurfacePipelineContext,
)
from surface_passability.geometry.intervals import (
    select_route_free_interval,
    supported_row_intervals,
)
from surface_passability.surfaces.routes import select_routed_surface_component


def serialize_supported_row_segments(
    free_mask: np.ndarray,
    row_y: int,
    left: int,
    right: int,
) -> list[list[int]]:
    """Return supported inclusive row segments as fresh integer lists."""

    return [
        [int(start), int(end)]
        for start, end in supported_row_intervals(
            free_mask,
            row_y,
            left,
            right,
        )
    ]


def evaluate_road_rows(
    boundary_mask: np.ndarray,
    obstacles: ObstacleOwnershipResult,
    context: SurfacePipelineContext,
    route: RouteAttempt | RecomputedStructuralRoute,
    route_path: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    *,
    component_already_selected: bool = False,
) -> RoadRowEvaluation:
    """Evaluate free and occupied intervals across one routed Road component."""

    if component_already_selected:
        component = boundary_mask.copy()
    else:
        component = select_routed_surface_component(boundary_mask, route_path)
        if not component.any():
            component = boundary_mask.copy()

    blocker = obstacles.final_mask & component
    free = component & ~blocker
    observations = sample_outer_boundary_rows(
        component,
        route_path,
        context.class_map,
    )
    branch_uncertain = max(0, len(route.paths) - 1) > 0
    rows: list[RoadRowEvidence] = []
    for observation in observations:
        left, right, row_y = (
            observation.left,
            observation.right,
            observation.y,
        )
        free_segments = serialize_supported_row_segments(
            free,
            row_y,
            left,
            right,
        )
        occupied_segments = serialize_supported_row_segments(
            blocker,
            row_y,
            left,
            right,
        )
        free_intervals = [tuple(segment) for segment in free_segments]
        selected = select_route_free_interval(
            free_intervals,
            observation.anchor_x,
        )
        rows.append(
            RoadRowEvidence(
                observation=observation,
                structural_span=(left, right),
                free_intervals=tuple(free_intervals),
                occupied_intervals=tuple(
                    tuple(segment) for segment in occupied_segments
                ),
                selected_route_interval=selected,
                boundary_reliability=(
                    "reliable"
                    if observation.width_state == "complete"
                    else "uncertain"
                ),
                branch_reliability=(
                    "uncertain"
                    if branch_uncertain
                    else "no_alternative_branch_detected"
                ),
            )
        )
    return RoadRowEvaluation(
        component_mask=component,
        blocker_mask=blocker,
        free_mask=free,
        rows=tuple(rows),
        boundary_uncertain_row_count=sum(
            row.boundary_reliability == "uncertain" for row in rows
        ),
        branch_uncertain_row_count=sum(
            row.branch_reliability == "uncertain" for row in rows
        ),
    )


def _evaluate_component_rows(
    component_mask: np.ndarray,
    obstacles: ObstacleOwnershipResult,
    context: SurfacePipelineContext,
    route: RouteAttempt | RecomputedStructuralRoute,
    route_path: tuple[tuple[int, int], ...] | list[tuple[int, int]],
) -> SurfaceRowEvaluation:
    free = component_mask & ~obstacles.final_mask
    observations = sample_outer_boundary_rows(
        component_mask,
        route_path,
        context.class_map,
    )
    branch_uncertain = max(0, len(route.paths) - 1) > 0
    rows: list[SurfaceRowEvidence] = []
    for observation in observations:
        left, right, y = observation.left, observation.right, observation.y
        intervals = supported_row_intervals(free, y, left, right)
        selected = select_route_free_interval(intervals, observation.anchor_x)
        transient = bool(obstacles.transient_mask[y, left : right + 1].any())
        rows.append(
            SurfaceRowEvidence(
                observation=observation,
                structural_span=(left, right),
                free_intervals=tuple(intervals),
                selected_route_interval=selected,
                transient_occlusion=transient,
                boundary_reliability=(
                    "reliable"
                    if observation.width_state == "complete"
                    else "uncertain"
                ),
                branch_reliability=(
                    "uncertain"
                    if branch_uncertain
                    else "no_alternative_branch_detected"
                ),
            )
        )
    return SurfaceRowEvaluation(
        component_mask=component_mask,
        free_mask=free,
        rows=tuple(rows),
        transient_occluded_row_count=sum(
            row.transient_occlusion for row in rows
        ),
        boundary_uncertain_row_count=sum(
            row.boundary_reliability == "uncertain" for row in rows
        ),
        branch_uncertain_row_count=sum(
            row.branch_reliability == "uncertain" for row in rows
        ),
    )


def evaluate_surface_rows(
    boundary_mask: np.ndarray,
    obstacles: ObstacleOwnershipResult,
    context: SurfacePipelineContext,
    route: RouteAttempt | RecomputedStructuralRoute | None = None,
    route_path: tuple[tuple[int, int], ...] | list[tuple[int, int]] | None = None,
) -> SurfaceRowEvaluation:
    """Evaluate free intervals and reliability across one routed component."""

    route = route or context.route
    route_path = context.route_path if route_path is None else route_path
    component = select_routed_surface_component(boundary_mask, route_path)
    if not component.any():
        component = boundary_mask.copy()
    return _evaluate_component_rows(
        component,
        obstacles,
        context,
        route,
        route_path,
    )


def evaluate_preselected_surface_rows(
    component_mask: np.ndarray,
    obstacles: ObstacleOwnershipResult,
    context: SurfacePipelineContext,
    route: RouteAttempt | RecomputedStructuralRoute,
    route_path: tuple[tuple[int, int], ...] | list[tuple[int, int]],
) -> SurfaceRowEvaluation:
    """Evaluate row evidence without reselecting the supplied component."""

    return _evaluate_component_rows(
        component_mask,
        obstacles,
        context,
        route,
        route_path,
    )
