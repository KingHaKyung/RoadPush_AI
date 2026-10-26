"""Freeze the straight-boundary evidence consumed by extension decisions."""

from __future__ import annotations

from typing import Any, Mapping

from surface_passability.contracts.boundary_extension import (
    BoundaryObservationSnapshot,
)


def freeze_boundary_observation(
    sample_id: str,
    result: Mapping[str, Any],
    image_shape: tuple[int, int],
) -> BoundaryObservationSnapshot:
    """Copy the downstream-authoritative boundary evidence into a snapshot."""

    fit = result.get("envelope_fit", {})
    left = fit.get("left_fit", {})
    right = fit.get("right_fit", {})
    support = fit.get("support_y_range")
    component = fit.get("component_y_range")
    reasons = tuple(
        str(value) for value in fit.get("rejection_reasons", [])
    )
    disposition = str(result.get("final_disposition", "unknown"))
    return BoundaryObservationSnapshot(
        sample_id=sample_id,
        selected_surface=result.get("selected_surface"),
        straight_disposition=disposition,
        straight_reasons=reasons,
        image_shape=(int(image_shape[0]), int(image_shape[1])),
        component_y_range=(
            tuple(map(int, component)) if component else None
        ),
        support_y_range=tuple(map(int, support)) if support else None,
        selected_band_centers=tuple(
            map(int, fit.get("selected_band_centers", []))
        ),
        left_slope=float(left["slope"]) if left.get("valid") else None,
        left_intercept=(
            float(left["intercept"]) if left.get("valid") else None
        ),
        right_slope=float(right["slope"]) if right.get("valid") else None,
        right_intercept=(
            float(right["intercept"]) if right.get("valid") else None
        ),
        left_seed_p90_residual_px=(
            float(left["p90_residual_px"])
            if left.get("p90_residual_px") is not None
            else None
        ),
        right_seed_p90_residual_px=(
            float(right["p90_residual_px"])
            if right.get("p90_residual_px") is not None
            else None
        ),
        row_evidence=tuple(
            dict(value) for value in result.get("row_evidence", [])
        ),
        direction_classification=result.get(
            "probe_b_direction_consistency",
            {},
        ).get("classification"),
        hard_veto_applied=bool(
            result.get("probe_v_grounded_vegetation_cap", {}).get(
                "gate_applied"
            )
        ),
        blocker_sha256=result.get("blocker_reuse", {}).get("actual_sha256"),
    )
