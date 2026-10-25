from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from roadpush_ai import legacy_policy as legacy
from roadpush_ai.road.context import ROAD_UNION_SEAM_RADIUS_PX, RoadContext, build_road_context
from roadpush_ai.road.linear_drain import LinearDrainResult, evaluate_linear_transverse_drains
from roadpush_ai.road.ownership import (
    SurfaceOwnershipResult,
    apply_road_ownership,
    build_owned_road_context,
)
from roadpush_ai.surface.context import SurfaceContext


ENHANCED_ROAD_POLICY_ID = "phase3_road_context_v2_drain_ownership_v1"


@dataclass
class EnhancedRoadBuild:
    """Production Road V2 plus the evidence used to construct it."""

    road: RoadContext
    base_road: RoadContext
    linear_drain: LinearDrainResult
    ownership: SurfaceOwnershipResult


def _restore_approved_linear_drain(
    owned: RoadContext, approved_drain: np.ndarray
) -> RoadContext:
    """Replay an approval made from base Road evidence without reevaluation."""

    approved = approved_drain & ~owned.surface_exclusion
    union = owned.surface.repaired_mask | approved
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ROAD_UNION_SEAM_RADIUS_PX * 2 + 1,) * 2
    )
    union = cv2.morphologyEx(
        union.astype(np.uint8), cv2.MORPH_CLOSE, kernel
    ).astype(bool)
    union &= ~owned.surface_exclusion
    structure, additions = legacy.base.structure_copy(union)
    structure &= ~owned.surface_exclusion
    additions &= ~owned.surface_exclusion
    route, route_path = legacy.outer.route_path(union)
    observed = legacy.outer.selected_component(structure, route_path)
    if not observed.any():
        observed = structure.copy()
    source_component = legacy.outer.selected_component(union, route_path)
    if not source_component.any():
        source_component = union.copy()
    surface = SurfaceContext(
        profile=owned.surface.profile,
        source_mask=source_component,
        repaired_mask=union,
        structure_copy=structure,
        forbidden_mask=owned.surface.forbidden_mask,
        observed_component=observed,
        route=route,
        route_path=route_path,
        structure_copy_additions=additions,
    )
    return replace(
        owned,
        surface=surface,
        repaired_road=owned.repaired_road | approved,
        accepted_repair=owned.accepted_repair | approved,
        policy={
            **owned.policy,
            "enhanced_road_policy_id": ENHANCED_ROAD_POLICY_ID,
            "linear_transverse_drain_enabled": True,
            "linear_drain_whole_object_restoration": True,
            "road_positive_ownership_enabled": True,
            "enhanced_policy_order": [
                "base_road_and_linear_drain_evidence",
                "drain_route_informed_raw_road_positive_ownership",
                "approved_linear_drain_whole_object_restore",
                "existing_component_selection_and_row_span",
                "existing_structural_side_pruning",
                "existing_blocker_subtraction",
            ],
        },
    )


def build_enhanced_road_context(
    labels: np.ndarray, sidewalk: SurfaceContext
) -> EnhancedRoadBuild:
    """Build reviewed Road V2 without changing the original Phase 3 builder.

    Drain acceptance observes only the original Phase 3 Road. The repaired
    drain route then informs Raw Road ownership, and the already-approved whole
    Catch Basin component is replayed after ownership. This is intentionally a
    one-way dependency; ownership never feeds back into drain eligibility.
    """

    base = build_road_context(labels, sidewalk)
    drain = evaluate_linear_transverse_drains(
        labels,
        base,
        restore_whole_accepted_component=True,
        production_policy=True,
    )
    ownership_evidence = replace(base, surface=drain.road.surface)
    ownership = apply_road_ownership(labels, ownership_evidence)
    owned = build_owned_road_context(labels, base, ownership)
    road = _restore_approved_linear_drain(owned, drain.approved_linear_drain)
    return EnhancedRoadBuild(
        road=road,
        base_road=base,
        linear_drain=drain,
        ownership=ownership,
    )


def write_enhanced_road_diagnostics(
    artifact: Path, build: EnhancedRoadBuild
) -> dict:
    """Persist production diagnostics that explain V2 policy decisions."""

    masks = {
        "12_raw_catch_basin.png": build.linear_drain.raw_catch_basin,
        "13_estimated_drain_corridor.png": build.linear_drain.estimated_corridor,
        "14_approved_linear_drain.png": build.linear_drain.approved_linear_drain,
        "15_rejected_linear_drain.png": build.linear_drain.rejected_linear_drain,
        "16_ownership_boundary_evidence.png": build.ownership.boundary_evidence,
        "17_ownership_persistent_boundary.png": build.ownership.persistent_boundary,
        "18_ownership_approved_removal.png": build.ownership.approved_removal,
        "19_ownership_uncertain.png": build.ownership.uncertain_mask,
        "20_ownership_veto.png": build.ownership.veto_mask,
    }
    for name, mask in masks.items():
        if not cv2.imwrite(str(artifact / name), mask.astype(np.uint8) * 255):
            raise OSError(f"Failed to write mask: {artifact / name}")
    return {
        "policy_id": ENHANCED_ROAD_POLICY_ID,
        "order": build.road.policy["enhanced_policy_order"],
        "linear_drain": {
            "approved_pixels": int(build.linear_drain.approved_linear_drain.sum()),
            "accepted_components": [
                row for row in build.linear_drain.component_records if row["accepted"]
            ],
            "policy": build.linear_drain.policy,
        },
        "ownership": {
            "status": build.ownership.diagnostics["status"],
            "scene_veto_reason": build.ownership.diagnostics["scene_veto_reason"],
            "approved_removal_pixels": int(build.ownership.approved_removal.sum()),
            "route_corridor_removed_pixels": build.ownership.diagnostics[
                "route_corridor_removed_pixels"
            ],
            "diagnostics": build.ownership.diagnostics,
        },
    }
