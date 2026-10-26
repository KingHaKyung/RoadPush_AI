"""Structural Sidewalk branch pruning without obstacle evidence."""

from __future__ import annotations

import cv2
import numpy as np

from surface_passability.boundary.observations import collect_boundary_rows
from surface_passability.contracts.boundaries import (
    AcceptedSideBranchGroup,
    BoundarySide,
    SideBranchCandidate,
    SideBranchPruningResult,
    SidewalkBoundaryMode,
)
from surface_passability.contracts.surfaces import SurfacePipelineContext
from surface_passability.policies.surfaces import (
    SIDE_BRANCH_FAR_LIMIT_HEIGHT_FRACTION,
    SIDE_BRANCH_MINIMUM_CONSECUTIVE_ROWS,
)
from surface_passability.surfaces.branch_pruning import (
    analyze_main_corridor_boundaries,
)
from surface_passability.surfaces.structural_routes import (
    group_consecutive_indices,
    paint_boundary_row_span,
)


def prune_side_branches(
    raw_boundary: np.ndarray,
    route_high_confidence: bool,
    route_path: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    context: SurfacePipelineContext,
    mode: SidewalkBoundaryMode,
) -> SideBranchPruningResult:
    """Remove only supported upper-image side branches from a boundary."""

    if mode == SidewalkBoundaryMode.OBSERVED_COMPONENT:
        empty = np.zeros_like(raw_boundary)
        pruned_boundary = raw_boundary.copy()
        route_confidence = bool(route_high_confidence)
        far_limit = round(
            raw_boundary.shape[0]
            * SIDE_BRANCH_FAR_LIMIT_HEIGHT_FRACTION
        )
        return SideBranchPruningResult(
            corridor_analysis=None,
            raw_candidate_mask=empty,
            vetoed_candidate_mask=empty,
            approved_removal_mask=empty,
            pruned_boundary_mask=pruned_boundary,
            accepted_groups=[],
            candidate_rows=None,
            policy="disabled_for_test1",
            route_high_confidence=route_confidence,
            far_limit_y=far_limit,
            footprint_or_blocker_used=False,
        )

    grayscale = cv2.cvtColor(context.image, cv2.COLOR_RGB2GRAY)
    observations = collect_boundary_rows(
        raw_boundary,
        route_path,
        context.class_map,
        grayscale,
    )
    corridor_analysis = analyze_main_corridor_boundaries(
        observations,
        context.image.shape[1],
        np.zeros_like(raw_boundary),
    )
    rows = [row.to_legacy_record() for row in corridor_analysis.rows]
    far_limit = round(
        raw_boundary.shape[0] * SIDE_BRANCH_FAR_LIMIT_HEIGHT_FRACTION
    )
    raw_candidate = np.zeros_like(raw_boundary)
    vetoed_candidate = np.zeros_like(raw_boundary)
    approved = np.zeros_like(raw_boundary)
    accepted_groups: list[AcceptedSideBranchGroup] = []
    candidate_rows: list[SideBranchCandidate] = []

    for side in ("left", "right"):
        typed_side: BoundarySide = side
        flag = f"{side}_branch_candidate"
        kind = f"{side}_kind"
        flagged = [
            index
            for index, row in enumerate(rows)
            if row.get(flag)
        ]
        for index in flagged:
            paint_boundary_row_span(raw_candidate, rows, [index], side)
        eligible = []
        for index in flagged:
            row = rows[index]
            veto = None
            if row["y"] > far_limit:
                veto = "below_upper_70_percent"
            elif row[kind] == "frame_truncated":
                veto = "frame_truncated"
            if veto is None:
                eligible.append(index)
            else:
                paint_boundary_row_span(
                    vetoed_candidate,
                    rows,
                    [index],
                    side,
                )
            candidate_rows.append(
                SideBranchCandidate(
                    row_index=index,
                    y=row["y"],
                    side=typed_side,
                    endpoint_kind=row[kind],
                    veto_reason=veto,
                )
            )

        for group in group_consecutive_indices(eligible):
            if len(group) < SIDE_BRANCH_MINIMUM_CONSECUTIVE_ROWS:
                for index in group:
                    next(
                        record
                        for record in candidate_rows
                        if record.row_index == index
                        and record.side == side
                    ).veto_reason = "fewer_than_2_consecutive_rows"
                    paint_boundary_row_span(
                        vetoed_candidate,
                        rows,
                        [index],
                        side,
                    )
                continue
            if not route_high_confidence:
                for index in group:
                    next(
                        record
                        for record in candidate_rows
                        if record.row_index == index
                        and record.side == side
                    ).veto_reason = "structural_route_not_high_confidence"
                paint_boundary_row_span(
                    vetoed_candidate,
                    rows,
                    group,
                    side,
                )
                continue
            paint_boundary_row_span(approved, rows, group, side)
            for index in group:
                next(
                    record
                    for record in candidate_rows
                    if record.row_index == index
                    and record.side == side
                ).accepted = True
            accepted_groups.append(
                AcceptedSideBranchGroup(
                    side=typed_side,
                    sampled_row_indices=group,
                    sampled_y=[rows[index]["y"] for index in group],
                )
            )

    raw_candidate &= raw_boundary
    vetoed_candidate &= raw_boundary
    approved &= raw_boundary
    pruned_boundary = raw_boundary & ~approved
    route_confidence = bool(route_high_confidence)
    return SideBranchPruningResult(
        corridor_analysis=corridor_analysis,
        raw_candidate_mask=raw_candidate,
        vetoed_candidate_mask=vetoed_candidate,
        approved_removal_mask=approved,
        pruned_boundary_mask=pruned_boundary,
        accepted_groups=accepted_groups,
        candidate_rows=candidate_rows,
        policy="upper70_two_consecutive_frame_veto",
        route_high_confidence=route_confidence,
        far_limit_y=far_limit,
        footprint_or_blocker_used=False,
    )
