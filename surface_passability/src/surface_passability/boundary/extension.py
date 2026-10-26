"""Generate bounded straight-boundary extension proposals."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from surface_passability.contracts.boundary_extension import (
    BoundaryObservationSnapshot,
)
from surface_passability.io.hashing import hash_mask_after_uint8_cast
from surface_passability.policies.extension import BoundaryExtensionPolicy


POLICY_ID = "phase7_extension_track_s1a_generation_ablation_v1"


def line_column(slope: float, intercept: float, row_y: int) -> float:
    """Evaluate a frozen boundary line at one image row."""

    return slope * float(row_y) + intercept


def projected_row_interval(
    snapshot: BoundaryObservationSnapshot,
    row_y: int,
) -> tuple[int, int, bool]:
    """Clip one projected boundary interval and report invalid ordering."""

    assert snapshot.left_slope is not None
    assert snapshot.left_intercept is not None
    assert snapshot.right_slope is not None
    assert snapshot.right_intercept is not None
    width = snapshot.image_shape[1]
    raw_left = line_column(
        snapshot.left_slope,
        snapshot.left_intercept,
        row_y,
    )
    raw_right = line_column(
        snapshot.right_slope,
        snapshot.right_intercept,
        row_y,
    )
    crossing = raw_left >= raw_right
    left = max(0, min(width - 1, int(round(raw_left))))
    right = max(0, min(width - 1, int(round(raw_right))))
    return left, right, crossing or left >= right


def paint_projected_rows(
    snapshot: BoundaryObservationSnapshot,
    rows: list[int] | range,
) -> np.ndarray:
    """Rasterize valid projected intervals into a new boolean mask."""

    mask = np.zeros(snapshot.image_shape, dtype=bool)
    for row_y in rows:
        left, right, invalid = projected_row_interval(snapshot, int(row_y))
        if not invalid:
            mask[int(row_y), left : right + 1] = True
    return mask


def longest_false_run(values: list[bool]) -> int:
    """Return the longest consecutive false span in input order."""

    longest = current = 0
    for value in values:
        current = 0 if value else current + 1
        longest = max(longest, current)
    return longest


def extension_horizon_metrics(
    snapshot: BoundaryObservationSnapshot,
    terminal_y: int,
    horizon_y: int,
) -> dict[str, Any]:
    """Describe boundary displacement and width change at one horizon."""

    left0 = line_column(snapshot.left_slope, snapshot.left_intercept, terminal_y)  # type: ignore[arg-type]
    right0 = line_column(snapshot.right_slope, snapshot.right_intercept, terminal_y)  # type: ignore[arg-type]
    left1 = line_column(snapshot.left_slope, snapshot.left_intercept, horizon_y)  # type: ignore[arg-type]
    right1 = line_column(snapshot.right_slope, snapshot.right_intercept, horizon_y)  # type: ignore[arg-type]
    width0 = right0 - left0
    width1 = right1 - left1
    return {
        "y": int(horizon_y),
        "delta_x_left": float(left1 - left0),
        "delta_x_right": float(right1 - right0),
        "centerline_delta_x": float(
            ((left1 + right1) - (left0 + right0)) / 2.0
        ),
        "width_delta": float(width1 - width0),
        "width_ratio": float(width1 / width0) if width0 > 0 else None,
        "crossing": bool(width1 <= 0),
    }


def extend_boundary_direction(
    snapshot: BoundaryObservationSnapshot,
    structural_mask: np.ndarray,
    terminal_y: int,
    step: int,
    support_span: int,
    policy: BoundaryExtensionPolicy,
) -> tuple[list[int], dict[str, Any]]:
    """Extend one end until a frozen geometric or structural stop fires."""

    height = snapshot.image_shape[0]
    boundary_y = 0 if step < 0 else height - 1
    maximum_rows = max(
        0,
        int(round(support_span * policy.maximum_extension_span_ratio)),
    )
    maximum_y = terminal_y + step * maximum_rows
    maximum_y = max(0, min(height - 1, maximum_y))
    first_horizon_y = max(
        0,
        min(height - 1, terminal_y + step * support_span),
    )
    rows: list[int] = []
    zero_run = 0
    stop_reason = (
        "image_boundary"
        if maximum_y == boundary_y
        else "configured_maximum_extension"
    )
    stop_y = maximum_y
    row_y = terminal_y + step
    seed_uncertainty = max(
        snapshot.left_seed_p90_residual_px or 0.0,
        snapshot.right_seed_p90_residual_px or 0.0,
    )
    minimum_resolvable_width = (
        max(
            float(policy.minimum_resolvable_width_px),
            policy.minimum_width_uncertainty_multiplier * seed_uncertainty,
        )
        if policy.minimum_resolvable_width_px is not None
        else None
    )
    while 0 <= row_y < height and (
        (step < 0 and row_y >= maximum_y)
        or (step > 0 and row_y <= maximum_y)
    ):
        raw_left = line_column(
            snapshot.left_slope,  # type: ignore[arg-type]
            snapshot.left_intercept,  # type: ignore[arg-type]
            row_y,
        )
        raw_right = line_column(
            snapshot.right_slope,  # type: ignore[arg-type]
            snapshot.right_intercept,  # type: ignore[arg-type]
            row_y,
        )
        raster_left = int(round(raw_left))
        raster_right = int(round(raw_right))
        if policy.stop_before_side_frame_contact and (
            raster_left <= 0
            or raster_right >= snapshot.image_shape[1] - 1
        ):
            stop_reason = "side_frame_contact"
            stop_y = rows[-1] if rows else terminal_y
            break
        left, right, invalid = projected_row_interval(snapshot, row_y)
        if invalid:
            stop_reason = "crossing_or_invalid_width"
            stop_y = rows[-1] if rows else terminal_y
            break
        structural_overlap = bool(
            np.any(structural_mask[row_y, left : right + 1])
        )
        zero_run = 0 if structural_overlap else zero_run + 1
        rows.append(row_y)
        if zero_run >= policy.structural_zero_overlap_persistence_rows:
            rows = rows[:-zero_run]
            stop_y = rows[-1] if rows else terminal_y
            stop_reason = "persistent_structural_end"
            break
        if row_y == maximum_y:
            stop_y = row_y
            break
        row_y += step

    if minimum_resolvable_width is not None and rows:
        for index, projected_y in enumerate(rows):
            raw_left = line_column(
                snapshot.left_slope,  # type: ignore[arg-type]
                snapshot.left_intercept,  # type: ignore[arg-type]
                projected_y,
            )
            raw_right = line_column(
                snapshot.right_slope,  # type: ignore[arg-type]
                snapshot.right_intercept,  # type: ignore[arg-type]
                projected_y,
            )
            if raw_right - raw_left < minimum_resolvable_width:
                rows = rows[:index]
                stop_reason = "minimum_resolvable_width"
                stop_y = rows[-1] if rows else terminal_y
                break

    disposition = "no_op" if not rows and stop_y == terminal_y else "extended"
    second_horizon_distance = abs(stop_y - terminal_y)
    first_horizon_distance = abs(first_horizon_y - terminal_y)
    short = second_horizon_distance < first_horizon_distance
    return rows, {
        "direction": "far_upper" if step < 0 else "near_lower",
        "terminal_y": int(terminal_y),
        "extension_row_count": len(rows),
        "extension_y_range": [min(rows), max(rows)] if rows else None,
        "stop_reason": stop_reason,
        "stop_y": int(stop_y),
        "diagnostic_disposition": disposition,
        "h1": (
            None
            if short
            else extension_horizon_metrics(
                snapshot,
                terminal_y,
                first_horizon_y,
            )
        ),
        "h2": extension_horizon_metrics(snapshot, terminal_y, stop_y),
        "h1_requested_y": int(first_horizon_y),
        "short_horizon_only": bool(short),
        "minimum_resolvable_width_px": minimum_resolvable_width,
    }


def outside_support_observation_metrics(
    snapshot: BoundaryObservationSnapshot,
) -> dict[str, Any]:
    """Compare frozen clean observations outside the selected support range."""

    if snapshot.support_y_range is None:
        return {
            "clean_row_count": 0,
            "residuals": [],
            "adaptive_tolerance_px": None,
        }
    low, high = snapshot.support_y_range
    tolerance = max(
        5.0,
        2.0
        * max(
            snapshot.left_seed_p90_residual_px or 0.0,
            snapshot.right_seed_p90_residual_px or 0.0,
        ),
    )
    rows: list[dict[str, Any]] = []
    contradictory: list[bool] = []
    for item in snapshot.row_evidence:
        row_y = int(item["y"])
        if low <= row_y <= high or not item.get("clean_bilateral"):
            continue
        candidate_left = line_column(
            snapshot.left_slope,  # type: ignore[arg-type]
            snapshot.left_intercept,  # type: ignore[arg-type]
            row_y,
        )
        candidate_right = line_column(
            snapshot.right_slope,  # type: ignore[arg-type]
            snapshot.right_intercept,  # type: ignore[arg-type]
            row_y,
        )
        observed_left = float(item["left"])
        observed_right = float(item["right"])
        left_signed_residual = observed_left - candidate_left
        right_signed_residual = candidate_right - observed_right
        maximum_outward_residual = max(
            0.0,
            left_signed_residual,
            right_signed_residual,
        )
        left_residual = abs(observed_left - candidate_left)
        right_residual = abs(observed_right - candidate_right)
        maximum = max(left_residual, right_residual)
        is_contradictory = maximum > tolerance
        is_containment_contradictory = (
            maximum_outward_residual > tolerance
        )
        contradictory.append(is_contradictory)
        rows.append(
            {
                "y": row_y,
                "left_residual_px": left_residual,
                "right_residual_px": right_residual,
                "maximum_residual_px": maximum,
                "diagnostic_contradiction": is_contradictory,
                "left_signed_outward_residual_px": left_signed_residual,
                "right_signed_outward_residual_px": right_signed_residual,
                "maximum_outward_residual_px": maximum_outward_residual,
                "containment_contradiction": (
                    is_containment_contradictory
                ),
            }
        )
    maxima = [row["maximum_residual_px"] for row in rows]
    return {
        "clean_row_count": len(rows),
        "adaptive_tolerance_px": tolerance,
        "tolerance_status": "diagnostic_only_not_frozen",
        "median_maximum_residual_px": (
            float(np.median(maxima)) if maxima else None
        ),
        "p90_maximum_residual_px": (
            float(np.percentile(maxima, 90)) if maxima else None
        ),
        "contradictory_row_count": sum(contradictory),
        "longest_contradictory_run": longest_false_run(
            [not value for value in contradictory]
        ),
        "residuals": rows,
    }


def generate_boundary_extension(
    snapshot: BoundaryObservationSnapshot,
    structural_mask: np.ndarray,
    blocker_mask: np.ndarray,
    policy: BoundaryExtensionPolicy | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Generate one bounded proposal without changing frozen observations."""

    policy = policy or BoundaryExtensionPolicy()
    structural = np.asarray(structural_mask, dtype=bool)
    blocker = np.asarray(blocker_mask, dtype=bool)
    if (
        structural.shape != snapshot.image_shape
        or blocker.shape != snapshot.image_shape
    ):
        raise ValueError("S1 masks must match the frozen V6 image shape")

    constructible = bool(
        snapshot.selected_surface is not None
        and snapshot.support_y_range is not None
        and snapshot.left_slope is not None
        and snapshot.right_slope is not None
    )
    empty = np.zeros(snapshot.image_shape, dtype=bool)
    if not constructible:
        result = {
            "policy_id": POLICY_ID,
            "package": "S1-A",
            "status": "diagnostic_only",
            "construction_status": "not_constructible",
            "construction_reasons": [
                "missing_frozen_v6_support_or_fit"
            ],
            "v6_straight_disposition": snapshot.straight_disposition,
            "v6_straight_reasons": list(snapshot.straight_reasons),
            "v6_observation_immutable": True,
            "new_band_search_count": 0,
            "refit_count": 0,
            "s1_selection_evaluated": False,
            "s2_eligibility_evaluated": False,
            "far_upper": None,
            "near_lower": None,
        }
        return result, {
            "observed_seed": empty.copy(),
            "extension_only": empty.copy(),
            "proposal": empty.copy(),
            "structural_agreement": empty.copy(),
        }

    support_low, support_high = snapshot.support_y_range
    support_span = support_high - support_low + 1
    observed = paint_projected_rows(
        snapshot,
        range(support_low, support_high + 1),
    )
    far_rows, far = extend_boundary_direction(
        snapshot,
        structural,
        support_low,
        -1,
        support_span,
        policy,
    )
    near_rows, near = extend_boundary_direction(
        snapshot,
        structural,
        support_high,
        1,
        support_span,
        policy,
    )
    extension = paint_projected_rows(snapshot, far_rows + near_rows)
    extension &= ~observed
    proposal = observed | extension
    agreement = proposal & structural

    row_fill: list[float] = []
    row_has_structure: list[bool] = []
    for row_y in np.flatnonzero(np.any(proposal, axis=1)):
        area = int(proposal[row_y].sum())
        fill = float(agreement[row_y].sum() / area) if area else 0.0
        row_fill.append(fill)
        row_has_structure.append(bool(agreement[row_y].any()))
    extension_rows = np.flatnonzero(np.any(extension, axis=1))
    proposal_rows = np.flatnonzero(np.any(proposal, axis=1))
    bounded_structural_area = (
        int(
            structural[
                proposal_rows.min() : proposal_rows.max() + 1
            ].sum()
        )
        if proposal_rows.size
        else 0
    )
    extension_continuing = (
        float(
            np.mean(
                np.any(
                    structural[extension_rows] & extension[extension_rows],
                    axis=1,
                )
            )
        )
        if extension_rows.size
        else None
    )
    observed_area = int(observed.sum())
    extension_area = int(extension.sum())
    proposal_area = int(proposal.sum())
    result = {
        "policy_id": POLICY_ID,
        "package": "S1-A",
        "status": "diagnostic_only",
        "construction_status": (
            "no_op" if not extension.any() else "proposal_generated"
        ),
        "construction_reasons": [],
        "v6_straight_disposition": snapshot.straight_disposition,
        "v6_straight_reasons": list(snapshot.straight_reasons),
        "v6_observation_immutable": True,
        "source_fit": "frozen_v6_pre_gate_left_right_fit",
        "new_band_search_count": 0,
        "refit_count": 0,
        "s1_selection_evaluated": False,
        "s2_eligibility_evaluated": False,
        "provisional_generation_settings": {
            "maximum_extension_span_ratio": (
                policy.maximum_extension_span_ratio
            ),
            "structural_zero_overlap_persistence_rows": (
                policy.structural_zero_overlap_persistence_rows
            ),
            "stop_before_side_frame_contact": (
                policy.stop_before_side_frame_contact
            ),
            "minimum_resolvable_width_px": (
                policy.minimum_resolvable_width_px
            ),
            "minimum_width_uncertainty_multiplier": (
                policy.minimum_width_uncertainty_multiplier
            ),
            "settings_frozen": False,
        },
        "support_y_range": list(snapshot.support_y_range),
        "support_span_px": support_span,
        "selected_band_centers": list(snapshot.selected_band_centers),
        "far_upper": far,
        "near_lower": near,
        "outside_seed_observation_consistency": (
            outside_support_observation_metrics(snapshot)
        ),
        "structural_agreement": {
            "proposal_area_px": proposal_area,
            "intersection_area_px": int(agreement.sum()),
            "proposal_structural_fraction": (
                float(agreement.sum() / proposal_area)
                if proposal_area
                else None
            ),
            "structural_recall_in_proposal_bounds": (
                float(agreement.sum() / bounded_structural_area)
                if bounded_structural_area
                else None
            ),
            "row_fill_median": (
                float(np.median(row_fill)) if row_fill else None
            ),
            "row_fill_p10": (
                float(np.percentile(row_fill, 10)) if row_fill else None
            ),
            "longest_zero_overlap_run": longest_false_run(
                row_has_structure
            ),
            "extension_rows_with_continuing_structure_fraction": (
                extension_continuing
            ),
            "observed_extension_pixel_connection": bool(
                extension.any()
                and np.any(
                    cv2.dilate(
                        observed.astype(np.uint8),
                        np.ones((3, 3), np.uint8),
                    )
                    & extension.astype(np.uint8)
                )
            ),
            "authoritative_surface_is_pre_blocker": True,
            "blocker_used_as_stop_evidence": False,
        },
        "growth": {
            "observed_area_px": observed_area,
            "extension_area_px": extension_area,
            "extension_to_observed_area_ratio": (
                float(extension_area / observed_area)
                if observed_area
                else None
            ),
            "extension_distance_to_support_span_ratio": float(
                (far["extension_row_count"] + near["extension_row_count"])
                / support_span
            ),
        },
        "hard_veto_observed": snapshot.hard_veto_applied,
        "direction_classification_observed": (
            snapshot.direction_classification
        ),
        "blocker_sha256": snapshot.blocker_sha256,
    }
    masks = {
        "observed_seed": observed,
        "extension_only": extension,
        "proposal": proposal,
        "structural_agreement": agreement,
    }
    result["mask_sha256"] = {
        name: hash_mask_after_uint8_cast(mask)
        for name, mask in masks.items()
    }
    return result, masks
