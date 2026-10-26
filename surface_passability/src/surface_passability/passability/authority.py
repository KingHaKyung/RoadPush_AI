"""Convert a completed boundary decision into passability authority."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from surface_passability.contracts.boundary import BoundaryAnalysisResult
from surface_passability.contracts.passability import (
    BoundaryStatusRoute,
    PassabilityBoundaryHandoff,
    PassabilityNullResult,
)
from surface_passability.io.hashing import hash_mask_after_uint8_cast


STATUS_ROUTES: dict[str, BoundaryStatusRoute] = {
    "selected": BoundaryStatusRoute(
        "full", "allowed", "authoritative", False, None
    ),
    "unchanged": BoundaryStatusRoute(
        "full",
        "conditional_on_v6_reliability",
        "authoritative",
        False,
        None,
    ),
    "selected_partial": BoundaryStatusRoute(
        "partial", "allowed", "authoritative", True, None
    ),
    "ambiguous_hold": BoundaryStatusRoute(
        "provisional", "review_only", "provisional_review", True, None
    ),
    "failed": BoundaryStatusRoute(
        "none", "denied", "none", False, "phase7a_failed"
    ),
}


def route_boundary_status(status: str) -> BoundaryStatusRoute:
    """Return the exact measurement authority for one boundary status."""

    try:
        return STATUS_ROUTES[status]
    except KeyError as error:
        raise ValueError(f"Unsupported Phase 7-A status: {status}") from error


def boundary_mask_rows(mask: np.ndarray) -> tuple[int, ...]:
    """Return rows containing at least one selected boundary pixel."""

    return tuple(int(y) for y in np.flatnonzero(np.any(mask, axis=1)))


def _null_result(
    boundary: BoundaryAnalysisResult,
    route: BoundaryStatusRoute,
) -> PassabilityNullResult:
    reason = route.abstention_reason
    if reason is None:
        raise ValueError("Typed nulls are only valid for denied status routes")
    decision = boundary.decision
    return PassabilityNullResult(
        schema_version="phase7_b_result_v2",
        sample_id=boundary.snapshot.sample_id,
        boundary_policy_id=str(decision["policy_id"]),
        boundary_status=str(decision["s1_status"]),
        result_authority="none",
        selected_surface=boundary.snapshot.selected_surface,
        measurement_scope="none",
        measurement_confidence="insufficient",
        human_review_required=route.human_review_required,
        abstention_reasons=[reason],
        minimum_passable_ratio=None,
        raw_minimum_passable_ratio=None,
        maximum_obstructed_width_ratio=None,
        obstacle_area_ratio=None,
        bottleneck=None,
        surface_width_m=None,
        absolute_passable_width_m=None,
        effective_blocked_width_m=None,
        agent_width_m=None,
        safety_factor=None,
        required_safe_width_m=None,
        required_passable_ratio=None,
        required_safe_ratio=None,
        passability_state="undetermined",
        flags=[reason],
    )


def _default_provenance(
    boundary: BoundaryAnalysisResult,
) -> dict[str, Any]:
    decision = boundary.decision
    diagnostic = boundary.generation_diagnostic
    return {
        "boundary_policy_id": decision["policy_id"],
        "boundary_selected_model": decision.get("s1_selected_model"),
        "boundary_failure_reasons": decision.get("s1_failure_reasons", []),
        "boundary_quality_grading_reasons": decision.get(
            "quality_grading_reasons", []
        ),
        "boundary_termination": {
            "far_upper": diagnostic.get("far_upper"),
            "near_lower": diagnostic.get("near_lower"),
            "safe_prefix_truncation": decision.get("v4c_truncation"),
        },
        "straight_policy_id": boundary.straight_result["policy_id"],
        "straight_disposition": boundary.snapshot.straight_disposition,
        "source": "live_boundary_analysis",
    }


def build_passability_handoff(
    boundary: BoundaryAnalysisResult,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> PassabilityBoundaryHandoff:
    """Authorize and expose one verified boundary mask to passability."""

    decision = boundary.decision
    status = str(decision["s1_status"])
    route = route_boundary_status(status)
    effective = np.asarray(
        boundary.selected_masks["effective_selected"], dtype=bool
    )
    actual_hash = hash_mask_after_uint8_cast(effective)
    expected_hash = str(decision["effective_selected_mask_sha256"])
    if actual_hash != expected_hash:
        raise AssertionError(
            f"{boundary.snapshot.sample_id}: boundary mask hash {actual_hash} "
            f"does not match decision {expected_hash}"
        )

    observed_mask = effective & np.asarray(
        boundary.generation_masks["observed_seed"], dtype=bool
    )
    generated_source = np.asarray(
        boundary.selected_masks.get(
            "validated_extension",
            boundary.generation_masks["extension_only"],
        ),
        dtype=bool,
    )
    generated_mask = effective & generated_source
    effective_rows = boundary_mask_rows(effective)
    authorized_range = (
        (effective_rows[0], effective_rows[-1])
        if route.calculation_allowed and effective_rows
        else None
    )
    null_result = None if route.calculation_allowed else _null_result(
        boundary, route
    )
    return PassabilityBoundaryHandoff(
        sample_id=boundary.snapshot.sample_id,
        selected_surface=boundary.snapshot.selected_surface,
        boundary_policy_id=str(decision["policy_id"]),
        boundary_status=status,
        measurement_scope=route.measurement_scope,
        measurement_authority=route.measurement_authority,
        result_authority=route.result_authority,
        human_review_required=route.human_review_required,
        authorized_row_range=authorized_range,
        observed_rows=boundary_mask_rows(observed_mask),
        generated_rows=boundary_mask_rows(generated_mask),
        structural_mask_sha256=actual_hash,
        structural_area_px=int(effective.sum()),
        provenance=(
            dict(provenance)
            if provenance is not None
            else _default_provenance(boundary)
        ),
        structural_mask=(
            effective.copy() if route.calculation_allowed else None
        ),
        typed_null_result=null_result,
    )
