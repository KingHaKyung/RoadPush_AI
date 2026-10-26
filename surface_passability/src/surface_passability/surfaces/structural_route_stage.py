"""Composition of Sidewalk boundary, structural route, and side pruning."""

from __future__ import annotations

from surface_passability.contracts.boundaries import (
    SidewalkBoundaryMode,
    SidewalkBoundaryResult,
)
from surface_passability.contracts.surfaces import (
    StructuralRouteStageResult,
    SurfacePipelineContext,
)
from surface_passability.surfaces.side_branch_pruning import prune_side_branches
from surface_passability.surfaces.structural_routes import (
    measure_route_displacement,
    recompute_structural_route,
)


def build_structural_route_stage(
    boundary: SidewalkBoundaryResult,
    context: SurfacePipelineContext,
) -> StructuralRouteStageResult:
    """Compose the frozen boundary-specific route and side-pruning stage."""

    if boundary.mode == SidewalkBoundaryMode.OBSERVED_COMPONENT:
        route = context.route
        route_path = context.route_path
    else:
        route = recompute_structural_route(boundary.mask, context)
        route_path = route.selected_path

    pruning = prune_side_branches(
        boundary.mask,
        route.high_confidence,
        route_path,
        context,
        boundary.mode,
    )
    calls = dict(boundary.calls)
    calls.update(
        {
            "test_specific_structural_route": (
                boundary.mode != SidewalkBoundaryMode.OBSERVED_COMPONENT
            ),
            "side_branch_pruning": (
                boundary.mode != SidewalkBoundaryMode.OBSERVED_COMPONENT
            ),
            "blocker_used_for_side_pruning": False,
        }
    )
    displacement = measure_route_displacement(
        context.route_path,
        route_path,
    )
    candidate_rows = (
        []
        if pruning.candidate_rows is None
        else pruning.candidate_rows
    )
    return StructuralRouteStageResult(
        raw_boundary_mask=boundary.mask,
        pruned_boundary_mask=pruning.pruned_boundary_mask,
        route=route,
        route_path=route_path,
        route_displacement=displacement,
        raw_candidate_mask=pruning.raw_candidate_mask,
        vetoed_candidate_mask=pruning.vetoed_candidate_mask,
        approved_removal_mask=pruning.approved_removal_mask,
        candidate_rows=candidate_rows,
        accepted_groups=pruning.accepted_groups,
        calls=calls,
    )
