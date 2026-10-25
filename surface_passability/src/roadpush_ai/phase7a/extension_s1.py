"""Phase 7 Extension Track S1 diagnostic primitives.

This module consumes a frozen Retry V6 observation package.  It never searches
for bands, refits a line, or changes the V6 result.  S1-A intentionally emits
diagnostics only; proposal acceptance belongs to S1-C.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping

import cv2
import numpy as np


S1A_POLICY_ID = "phase7_extension_track_s1a_generation_ablation_v1"
S1_TARGETED_POLICY_ID = "phase7_extension_track_s1_targeted_provisional_v1"
S1_TARGETED_V2_POLICY_ID = "phase7_extension_track_s1_targeted_provisional_v2"
S1_TARGETED_V3_POLICY_ID = "phase7_extension_track_s1_targeted_provisional_v3"
S1_TARGETED_V4A_POLICY_ID = "phase7_extension_track_s1_targeted_provisional_v4a_narrow_tip"
S1_TARGETED_V4B_POLICY_ID = "phase7_extension_track_s1_targeted_provisional_v4b_occluder_neutral"
S1_TARGETED_V4C_POLICY_ID = "phase7_extension_track_s1_targeted_provisional_v4c_safe_prefix"
S1_V4_PROBE_POLICY_ID = "phase7_extension_track_s1_v4_probe_human_review_grading"
S1_TARGETED_V4_POLICY_ID = "phase7_extension_track_s1_targeted_provisional_v4"


@dataclass(frozen=True)
class S1AGenerationPolicy:
    """Transparent, provisional bounds for the generation ablation."""

    maximum_extension_span_ratio: float = 2.0
    structural_zero_overlap_persistence_rows: int = 7
    stop_before_side_frame_contact: bool = False
    minimum_resolvable_width_px: int | None = None
    minimum_width_uncertainty_multiplier: float = 2.0


@dataclass(frozen=True)
class S1TargetedValidationPolicy:
    """Provisional, sample-agnostic S1 selection thresholds.

    These thresholds implement only the validation families already present in
    the Phase 7 design.  They intentionally do not add case-specific ownership,
    obstacle, frame-contact, or direction-remediation rules from user review.
    """

    minimum_proposal_structural_fraction: float = 0.80
    minimum_row_fill_p10: float = 0.50
    minimum_extension_rows_with_structure_fraction: float = 0.95
    minimum_outside_clean_rows_for_contradiction_gate: int = 4
    maximum_contradictory_observation_fraction: float = 0.25
    maximum_persistent_contradictory_rows: int = 3
    s2_minimum_outside_clean_rows: int = 6


@dataclass(frozen=True)
class S1TargetedValidationPolicyV2:
    """Reduced S1 policy: geometry, extension continuity, and in-range conflict."""

    minimum_extension_row_fill_p10: float = 0.50
    minimum_in_range_clean_rows: int = 4
    persistent_contradiction_minimum_fraction: float = 0.60
    persistent_contradiction_minimum_run_rows: int = 7
    s2_minimum_outside_clean_rows: int = 6


@dataclass(frozen=True)
class S1TargetedValidationPolicyV3:
    """V2 thresholds with signed containment replacing absolute residuals."""

    minimum_extension_row_fill_p10: float = 0.50
    minimum_in_range_clean_rows: int = 4
    persistent_contradiction_minimum_fraction: float = 0.60
    persistent_contradiction_minimum_run_rows: int = 7
    s2_minimum_outside_clean_rows: int = 6


@dataclass(frozen=True)
class S1TargetedValidationPolicyV4:
    """V3 plus occlusion-aware evidence and direction-local safe-prefix salvage."""

    minimum_extension_row_fill_p10: float = 0.50
    minimum_in_range_clean_rows: int = 4
    persistent_contradiction_minimum_fraction: float = 0.60
    persistent_contradiction_minimum_run_rows: int = 7
    prefix_failure_persistence_rows: int = 7
    maximum_occlusion_excluded_row_fraction: float = 0.40
    minimum_missing_fraction_explained_by_occlusion: float = 0.50
    minimum_containment_gap_occlusion_fraction: float = 0.50
    maximum_containment_occlusion_gap_fraction: float = 0.12
    minimum_retained_extension_rows: int = 3
    s2_minimum_outside_clean_rows: int = 6


@dataclass(frozen=True)
class S1V4ProbeGradingPolicy:
    """Post-V4c routing only; it never changes a generated extension mask."""

    ambiguous_containment_fraction: float = 0.50
    ambiguous_containment_run_rows: int = 4
    ambiguous_retained_fraction: float = 0.15
    retained_fraction_requires_clean_rows_below: int = 4
    maximum_top_bottom_width_ratio: float = 1.05
    partial_direction_classes: tuple[str, ...] = (
        "coherent_turn",
        "one_sided_detour",
    )


@dataclass(frozen=True)
class FrozenV6ObservationPackage:
    sample_id: str
    selected_surface: str | None
    v6_straight_disposition: str
    v6_straight_reasons: tuple[str, ...]
    image_shape: tuple[int, int]
    component_y_range: tuple[int, int] | None
    support_y_range: tuple[int, int] | None
    selected_band_centers: tuple[int, ...]
    left_slope: float | None
    left_intercept: float | None
    right_slope: float | None
    right_intercept: float | None
    left_seed_p90_residual_px: float | None
    right_seed_p90_residual_px: float | None
    row_evidence: tuple[Mapping[str, Any], ...]
    direction_classification: str | None
    hard_veto_applied: bool
    blocker_sha256: str | None


def mask_sha256(mask: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(mask.astype(np.uint8)).tobytes()).hexdigest()


def freeze_v6_observation(
    sample_id: str,
    result: Mapping[str, Any],
    image_shape: tuple[int, int],
) -> FrozenV6ObservationPackage:
    fit = result.get("envelope_fit", {})
    left = fit.get("left_fit", {})
    right = fit.get("right_fit", {})
    support = fit.get("support_y_range")
    component = fit.get("component_y_range")
    reasons = tuple(str(value) for value in fit.get("rejection_reasons", []))
    disposition = str(result.get("final_disposition", "unknown"))
    return FrozenV6ObservationPackage(
        sample_id=sample_id,
        selected_surface=result.get("selected_surface"),
        v6_straight_disposition=disposition,
        v6_straight_reasons=reasons,
        image_shape=(int(image_shape[0]), int(image_shape[1])),
        component_y_range=tuple(map(int, component)) if component else None,
        support_y_range=tuple(map(int, support)) if support else None,
        selected_band_centers=tuple(map(int, fit.get("selected_band_centers", []))),
        left_slope=float(left["slope"]) if left.get("valid") else None,
        left_intercept=float(left["intercept"]) if left.get("valid") else None,
        right_slope=float(right["slope"]) if right.get("valid") else None,
        right_intercept=float(right["intercept"]) if right.get("valid") else None,
        left_seed_p90_residual_px=(
            float(left["p90_residual_px"]) if left.get("p90_residual_px") is not None else None
        ),
        right_seed_p90_residual_px=(
            float(right["p90_residual_px"]) if right.get("p90_residual_px") is not None else None
        ),
        row_evidence=tuple(dict(value) for value in result.get("row_evidence", [])),
        direction_classification=result.get("probe_b_direction_consistency", {}).get(
            "classification"
        ),
        hard_veto_applied=bool(
            result.get("probe_v_grounded_vegetation_cap", {}).get("gate_applied")
        ),
        blocker_sha256=result.get("blocker_reuse", {}).get("actual_sha256"),
    )


def _line_x(slope: float, intercept: float, y: int) -> float:
    return slope * float(y) + intercept


def _row_interval(package: FrozenV6ObservationPackage, y: int) -> tuple[int, int, bool]:
    assert package.left_slope is not None and package.left_intercept is not None
    assert package.right_slope is not None and package.right_intercept is not None
    width = package.image_shape[1]
    raw_left = _line_x(package.left_slope, package.left_intercept, y)
    raw_right = _line_x(package.right_slope, package.right_intercept, y)
    crossing = raw_left >= raw_right
    left = max(0, min(width - 1, int(round(raw_left))))
    right = max(0, min(width - 1, int(round(raw_right))))
    return left, right, crossing or left >= right


def _paint_rows(
    package: FrozenV6ObservationPackage, rows: list[int] | range
) -> np.ndarray:
    mask = np.zeros(package.image_shape, dtype=bool)
    for y in rows:
        left, right, invalid = _row_interval(package, int(y))
        if not invalid:
            mask[int(y), left : right + 1] = True
    return mask


def _longest_false_run(values: list[bool]) -> int:
    longest = current = 0
    for value in values:
        current = 0 if value else current + 1
        longest = max(longest, current)
    return longest


def _horizon_metrics(
    package: FrozenV6ObservationPackage,
    terminal_y: int,
    horizon_y: int,
) -> dict[str, Any]:
    left0 = _line_x(package.left_slope, package.left_intercept, terminal_y)  # type: ignore[arg-type]
    right0 = _line_x(package.right_slope, package.right_intercept, terminal_y)  # type: ignore[arg-type]
    left1 = _line_x(package.left_slope, package.left_intercept, horizon_y)  # type: ignore[arg-type]
    right1 = _line_x(package.right_slope, package.right_intercept, horizon_y)  # type: ignore[arg-type]
    width0 = right0 - left0
    width1 = right1 - left1
    return {
        "y": int(horizon_y),
        "delta_x_left": float(left1 - left0),
        "delta_x_right": float(right1 - right0),
        "centerline_delta_x": float(((left1 + right1) - (left0 + right0)) / 2.0),
        "width_delta": float(width1 - width0),
        "width_ratio": float(width1 / width0) if width0 > 0 else None,
        "crossing": bool(width1 <= 0),
    }


def _extend_direction(
    package: FrozenV6ObservationPackage,
    structural_mask: np.ndarray,
    terminal_y: int,
    step: int,
    support_span: int,
    policy: S1AGenerationPolicy,
) -> tuple[list[int], dict[str, Any]]:
    height = package.image_shape[0]
    boundary_y = 0 if step < 0 else height - 1
    max_rows = max(0, int(round(support_span * policy.maximum_extension_span_ratio)))
    max_y = terminal_y + step * max_rows
    max_y = max(0, min(height - 1, max_y))
    h1_y = max(0, min(height - 1, terminal_y + step * support_span))
    rows: list[int] = []
    zero_run = 0
    stop_reason = "image_boundary" if max_y == boundary_y else "configured_maximum_extension"
    stop_y = max_y
    y = terminal_y + step
    seed_uncertainty = max(
        package.left_seed_p90_residual_px or 0.0,
        package.right_seed_p90_residual_px or 0.0,
    )
    minimum_resolvable_width = (
        max(
            float(policy.minimum_resolvable_width_px),
            policy.minimum_width_uncertainty_multiplier * seed_uncertainty,
        )
        if policy.minimum_resolvable_width_px is not None
        else None
    )
    while 0 <= y < height and ((step < 0 and y >= max_y) or (step > 0 and y <= max_y)):
        raw_left = _line_x(package.left_slope, package.left_intercept, y)  # type: ignore[arg-type]
        raw_right = _line_x(package.right_slope, package.right_intercept, y)  # type: ignore[arg-type]
        raster_left = int(round(raw_left))
        raster_right = int(round(raw_right))
        if policy.stop_before_side_frame_contact and (
            raster_left <= 0 or raster_right >= package.image_shape[1] - 1
        ):
            stop_reason = "side_frame_contact"
            stop_y = rows[-1] if rows else terminal_y
            break
        left, right, invalid = _row_interval(package, y)
        if invalid:
            stop_reason = "crossing_or_invalid_width"
            stop_y = rows[-1] if rows else terminal_y
            break
        structural_overlap = bool(np.any(structural_mask[y, left : right + 1]))
        zero_run = 0 if structural_overlap else zero_run + 1
        rows.append(y)
        if zero_run >= policy.structural_zero_overlap_persistence_rows:
            rows = rows[:-zero_run]
            stop_y = rows[-1] if rows else terminal_y
            stop_reason = "persistent_structural_end"
            break
        if y == max_y:
            stop_y = y
            break
        y += step
    # The resolution rule is a strict post-trim over the candidate produced by
    # every pre-existing V3 stop.  Applying it inside the loop can fire before
    # the seventh structural-zero row and accidentally retain the preceding six
    # unsupported rows, making the supposedly conservative candidate longer.
    if minimum_resolvable_width is not None and rows:
        for index, row_y in enumerate(rows):
            raw_left = _line_x(package.left_slope, package.left_intercept, row_y)  # type: ignore[arg-type]
            raw_right = _line_x(package.right_slope, package.right_intercept, row_y)  # type: ignore[arg-type]
            if raw_right - raw_left < minimum_resolvable_width:
                rows = rows[:index]
                stop_reason = "minimum_resolvable_width"
                stop_y = rows[-1] if rows else terminal_y
                break
    if not rows and stop_y == terminal_y:
        disposition = "no_op"
    else:
        disposition = "extended"
    h2_distance = abs(stop_y - terminal_y)
    h1_distance = abs(h1_y - terminal_y)
    short = h2_distance < h1_distance
    return rows, {
        "direction": "far_upper" if step < 0 else "near_lower",
        "terminal_y": int(terminal_y),
        "extension_row_count": len(rows),
        "extension_y_range": [min(rows), max(rows)] if rows else None,
        "stop_reason": stop_reason,
        "stop_y": int(stop_y),
        "diagnostic_disposition": disposition,
        "h1": None if short else _horizon_metrics(package, terminal_y, h1_y),
        "h2": _horizon_metrics(package, terminal_y, stop_y),
        "h1_requested_y": int(h1_y),
        "short_horizon_only": bool(short),
        "minimum_resolvable_width_px": minimum_resolvable_width,
    }


def _outside_observation_metrics(
    package: FrozenV6ObservationPackage,
) -> dict[str, Any]:
    if package.support_y_range is None:
        return {"clean_row_count": 0, "residuals": [], "adaptive_tolerance_px": None}
    low, high = package.support_y_range
    tolerance = max(
        5.0,
        2.0
        * max(
            package.left_seed_p90_residual_px or 0.0,
            package.right_seed_p90_residual_px or 0.0,
        ),
    )
    rows: list[dict[str, Any]] = []
    contradictory: list[bool] = []
    for item in package.row_evidence:
        y = int(item["y"])
        if low <= y <= high or not item.get("clean_bilateral"):
            continue
        candidate_left = _line_x(package.left_slope, package.left_intercept, y)  # type: ignore[arg-type]
        candidate_right = _line_x(package.right_slope, package.right_intercept, y)  # type: ignore[arg-type]
        observed_left = float(item["left"])
        observed_right = float(item["right"])
        left_signed_residual = observed_left - candidate_left
        right_signed_residual = candidate_right - observed_right
        maximum_outward_residual = max(
            0.0, left_signed_residual, right_signed_residual
        )
        left_residual = abs(observed_left - candidate_left)
        right_residual = abs(observed_right - candidate_right)
        maximum = max(left_residual, right_residual)
        is_contradictory = maximum > tolerance
        is_containment_contradictory = maximum_outward_residual > tolerance
        contradictory.append(is_contradictory)
        rows.append(
            {
                "y": y,
                "left_residual_px": left_residual,
                "right_residual_px": right_residual,
                "maximum_residual_px": maximum,
                "diagnostic_contradiction": is_contradictory,
                "left_signed_outward_residual_px": left_signed_residual,
                "right_signed_outward_residual_px": right_signed_residual,
                "maximum_outward_residual_px": maximum_outward_residual,
                "containment_contradiction": is_containment_contradictory,
            }
        )
    maxima = [row["maximum_residual_px"] for row in rows]
    return {
        "clean_row_count": len(rows),
        "adaptive_tolerance_px": tolerance,
        "tolerance_status": "diagnostic_only_not_frozen",
        "median_maximum_residual_px": float(np.median(maxima)) if maxima else None,
        "p90_maximum_residual_px": float(np.percentile(maxima, 90)) if maxima else None,
        "contradictory_row_count": sum(contradictory),
        "longest_contradictory_run": _longest_false_run([not value for value in contradictory]),
        "residuals": rows,
    }


def generate_s1a_proposal(
    package: FrozenV6ObservationPackage,
    structural_mask: np.ndarray,
    blocker_mask: np.ndarray,
    policy: S1AGenerationPolicy | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Generate one bounded proposal without altering or refitting V6 evidence."""

    policy = policy or S1AGenerationPolicy()
    structural = np.asarray(structural_mask, dtype=bool)
    blocker = np.asarray(blocker_mask, dtype=bool)
    if structural.shape != package.image_shape or blocker.shape != package.image_shape:
        raise ValueError("S1 masks must match the frozen V6 image shape")

    constructible = bool(
        package.selected_surface is not None
        and package.support_y_range is not None
        and package.left_slope is not None
        and package.right_slope is not None
    )
    empty = np.zeros(package.image_shape, dtype=bool)
    if not constructible:
        result = {
            "policy_id": S1A_POLICY_ID,
            "package": "S1-A",
            "status": "diagnostic_only",
            "construction_status": "not_constructible",
            "construction_reasons": ["missing_frozen_v6_support_or_fit"],
            "v6_straight_disposition": package.v6_straight_disposition,
            "v6_straight_reasons": list(package.v6_straight_reasons),
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

    support_low, support_high = package.support_y_range  # type: ignore[misc]
    support_span = support_high - support_low + 1
    observed = _paint_rows(package, range(support_low, support_high + 1))
    far_rows, far = _extend_direction(
        package, structural, support_low, -1, support_span, policy
    )
    near_rows, near = _extend_direction(
        package, structural, support_high, 1, support_span, policy
    )
    extension = _paint_rows(package, far_rows + near_rows)
    extension &= ~observed
    proposal = observed | extension
    agreement = proposal & structural

    row_fill: list[float] = []
    row_has_structure: list[bool] = []
    for y in np.flatnonzero(np.any(proposal, axis=1)):
        area = int(proposal[y].sum())
        fill = float(agreement[y].sum() / area) if area else 0.0
        row_fill.append(fill)
        row_has_structure.append(bool(agreement[y].any()))
    extension_rows = np.flatnonzero(np.any(extension, axis=1))
    proposal_rows = np.flatnonzero(np.any(proposal, axis=1))
    bounded_structural_area = (
        int(structural[proposal_rows.min() : proposal_rows.max() + 1].sum())
        if proposal_rows.size
        else 0
    )
    extension_continuing = (
        float(np.mean(np.any(structural[extension_rows] & extension[extension_rows], axis=1)))
        if extension_rows.size
        else None
    )
    observed_area = int(observed.sum())
    extension_area = int(extension.sum())
    proposal_area = int(proposal.sum())
    result = {
        "policy_id": S1A_POLICY_ID,
        "package": "S1-A",
        "status": "diagnostic_only",
        "construction_status": (
            "no_op" if not extension.any() else "proposal_generated"
        ),
        "construction_reasons": [],
        "v6_straight_disposition": package.v6_straight_disposition,
        "v6_straight_reasons": list(package.v6_straight_reasons),
        "v6_observation_immutable": True,
        "source_fit": "frozen_v6_pre_gate_left_right_fit",
        "new_band_search_count": 0,
        "refit_count": 0,
        "s1_selection_evaluated": False,
        "s2_eligibility_evaluated": False,
        "provisional_generation_settings": {
            "maximum_extension_span_ratio": policy.maximum_extension_span_ratio,
            "structural_zero_overlap_persistence_rows": policy.structural_zero_overlap_persistence_rows,
            "stop_before_side_frame_contact": policy.stop_before_side_frame_contact,
            "minimum_resolvable_width_px": policy.minimum_resolvable_width_px,
            "minimum_width_uncertainty_multiplier": policy.minimum_width_uncertainty_multiplier,
            "settings_frozen": False,
        },
        "support_y_range": list(package.support_y_range),
        "support_span_px": support_span,
        "selected_band_centers": list(package.selected_band_centers),
        "far_upper": far,
        "near_lower": near,
        "outside_seed_observation_consistency": _outside_observation_metrics(package),
        "structural_agreement": {
            "proposal_area_px": proposal_area,
            "intersection_area_px": int(agreement.sum()),
            "proposal_structural_fraction": (
                float(agreement.sum() / proposal_area) if proposal_area else None
            ),
            "structural_recall_in_proposal_bounds": (
                float(agreement.sum() / bounded_structural_area)
                if bounded_structural_area
                else None
            ),
            "row_fill_median": float(np.median(row_fill)) if row_fill else None,
            "row_fill_p10": float(np.percentile(row_fill, 10)) if row_fill else None,
            "longest_zero_overlap_run": _longest_false_run(row_has_structure),
            "extension_rows_with_continuing_structure_fraction": extension_continuing,
            "observed_extension_pixel_connection": bool(
                extension.any()
                and np.any(
                    cv2.dilate(observed.astype(np.uint8), np.ones((3, 3), np.uint8))
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
                float(extension_area / observed_area) if observed_area else None
            ),
            "extension_distance_to_support_span_ratio": float(
                (far["extension_row_count"] + near["extension_row_count"]) / support_span
            ),
        },
        "hard_veto_observed": package.hard_veto_applied,
        "direction_classification_observed": package.direction_classification,
        "blocker_sha256": package.blocker_sha256,
    }
    masks = {
        "observed_seed": observed,
        "extension_only": extension,
        "proposal": proposal,
        "structural_agreement": agreement,
    }
    result["mask_sha256"] = {name: mask_sha256(mask) for name, mask in masks.items()}
    return result, masks


def evaluate_s1_targeted(
    package: FrozenV6ObservationPackage,
    diagnostic: Mapping[str, Any],
    masks: Mapping[str, np.ndarray],
    policy: S1TargetedValidationPolicy | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Select or reject an S1 proposal using the current design evidence only."""

    policy = policy or S1TargetedValidationPolicy()
    proposal = np.asarray(masks["proposal"], dtype=bool)
    observed = np.asarray(masks["observed_seed"], dtype=bool)
    empty = np.zeros(package.image_shape, dtype=bool)
    construction = str(diagnostic.get("construction_status"))
    structural = diagnostic.get("structural_agreement", {})
    observation = diagnostic.get("outside_seed_observation_consistency", {})

    failures: list[str] = []
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, value: Any, rule: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "value": value, "rule": rule})
        if not passed:
            failures.append(name)

    if construction == "not_constructible":
        failures.append("missing_frozen_v6_support_or_fit")
    else:
        check(
            "preexisting_hard_veto",
            not package.hard_veto_applied,
            package.hard_veto_applied,
            "frozen V6 hard veto must be absent",
        )
        check(
            "straight_direction_evidence",
            package.direction_classification not in {"coherent_turn", "unstable_bilateral"},
            package.direction_classification,
            "coherent_turn and unstable_bilateral are not selectable by S1 straight extension",
        )
        sf = structural.get("proposal_structural_fraction")
        check(
            "proposal_structural_fraction",
            sf is not None and float(sf) >= policy.minimum_proposal_structural_fraction,
            sf,
            f">= {policy.minimum_proposal_structural_fraction:.2f}",
        )
        p10 = structural.get("row_fill_p10")
        check(
            "row_fill_p10",
            p10 is not None and float(p10) >= policy.minimum_row_fill_p10,
            p10,
            f">= {policy.minimum_row_fill_p10:.2f}",
        )
        continuing = structural.get("extension_rows_with_continuing_structure_fraction")
        if construction == "proposal_generated":
            check(
                "extension_structural_persistence",
                continuing is not None
                and float(continuing)
                >= policy.minimum_extension_rows_with_structure_fraction,
                continuing,
                f">= {policy.minimum_extension_rows_with_structure_fraction:.2f}",
            )
            check(
                "observed_extension_connection",
                bool(structural.get("observed_extension_pixel_connection")),
                structural.get("observed_extension_pixel_connection"),
                "observed seed and extension must touch",
            )

        clean_rows = int(observation.get("clean_row_count") or 0)
        contradictory_rows = int(observation.get("contradictory_row_count") or 0)
        contradictory_fraction = (
            float(contradictory_rows / clean_rows) if clean_rows else None
        )
        contradiction_run = int(observation.get("longest_contradictory_run") or 0)
        contradiction_gate_active = (
            clean_rows >= policy.minimum_outside_clean_rows_for_contradiction_gate
        )
        checks.append(
            {
                "name": "outside_observation_gate_available",
                "passed": True,
                "value": clean_rows,
                "rule": (
                    f"gate active from {policy.minimum_outside_clean_rows_for_contradiction_gate} clean rows"
                ),
            }
        )
        if contradiction_gate_active:
            check(
                "contradictory_observation_fraction",
                contradictory_fraction is not None
                and contradictory_fraction
                <= policy.maximum_contradictory_observation_fraction,
                contradictory_fraction,
                f"<= {policy.maximum_contradictory_observation_fraction:.2f}",
            )
            check(
                "persistent_contradictory_observations",
                contradiction_run <= policy.maximum_persistent_contradictory_rows,
                contradiction_run,
                f"<= {policy.maximum_persistent_contradictory_rows} rows",
            )

        for direction_name in ("far_upper", "near_lower"):
            direction = diagnostic.get(direction_name)
            if direction is None:
                continue
            check(
                f"{direction_name}_geometry",
                not bool(direction.get("h2", {}).get("crossing")),
                direction.get("h2", {}).get("crossing"),
                "left/right boundaries must remain ordered at H2",
            )

    failures = list(dict.fromkeys(failures))
    if construction == "no_op" and not failures:
        status = "no_extension_needed"
        selected_model = "observed_straight"
        selected = observed.copy()
    elif construction == "proposal_generated" and not failures:
        status = "selected"
        selected_model = "extended_straight"
        selected = proposal.copy()
    else:
        status = "failed"
        selected_model = None
        selected = empty.copy()

    enough_original_evidence = bool(
        package.support_y_range is not None
        and len(package.selected_band_centers) >= 3
        and int(observation.get("clean_row_count") or 0)
        >= policy.s2_minimum_outside_clean_rows
    )
    s2_eligible = bool(
        status == "failed"
        and enough_original_evidence
        and not package.hard_veto_applied
    )
    if status != "failed":
        handoff_reason = None
    elif s2_eligible:
        handoff_reason = "s1_failed_with_sufficient_original_v6_observations"
    elif package.hard_veto_applied:
        handoff_reason = "s1_failed_preexisting_hard_veto_s2_ineligible"
    else:
        handoff_reason = "s1_failed_insufficient_original_v6_observations"

    result = {
        "policy_id": S1_TARGETED_POLICY_ID,
        "threshold_status": "provisional_targeted_review",
        "s1_status": status,
        "s1_selected_model": selected_model,
        "s1_failure_reasons": failures,
        "s2_eligible": s2_eligible,
        "s2_handoff_reason": handoff_reason,
        "checks": checks,
        "effective_selected_area_px": int(selected.sum()),
        "candidate_area_px": int(proposal.sum()),
        "effective_selected_mask_sha256": mask_sha256(selected),
        "candidate_is_not_automatically_selected": True,
        "user_feedback_specific_rules_added": False,
        "policy": {
            "minimum_proposal_structural_fraction": policy.minimum_proposal_structural_fraction,
            "minimum_row_fill_p10": policy.minimum_row_fill_p10,
            "minimum_extension_rows_with_structure_fraction": policy.minimum_extension_rows_with_structure_fraction,
            "minimum_outside_clean_rows_for_contradiction_gate": policy.minimum_outside_clean_rows_for_contradiction_gate,
            "maximum_contradictory_observation_fraction": policy.maximum_contradictory_observation_fraction,
            "maximum_persistent_contradictory_rows": policy.maximum_persistent_contradictory_rows,
            "s2_minimum_outside_clean_rows": policy.s2_minimum_outside_clean_rows,
        },
    }
    return result, {"effective_selected": selected}


def _longest_consecutive_contradictory_rows(rows: list[Mapping[str, Any]]) -> int:
    longest = current = 0
    previous_y: int | None = None
    for row in sorted(rows, key=lambda value: int(value["y"])):
        y = int(row["y"])
        if bool(row.get("diagnostic_contradiction")):
            current = current + 1 if previous_y is not None and y == previous_y + 1 else 1
            longest = max(longest, current)
        else:
            current = 0
        previous_y = y
    return longest


def _evaluate_s1_targeted_reduced(
    package: FrozenV6ObservationPackage,
    diagnostic: Mapping[str, Any],
    masks: Mapping[str, np.ndarray],
    policy: S1TargetedValidationPolicyV2 | S1TargetedValidationPolicyV3,
    *,
    policy_id: str,
    contradiction_key: str,
    contradiction_check_name: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Shared reduced evaluator; the caller chooses the observation semantics."""
    proposal = np.asarray(masks["proposal"], dtype=bool)
    observed = np.asarray(masks["observed_seed"], dtype=bool)
    extension = np.asarray(masks["extension_only"], dtype=bool)
    agreement = np.asarray(masks["structural_agreement"], dtype=bool)
    empty = np.zeros(package.image_shape, dtype=bool)
    construction = str(diagnostic.get("construction_status"))
    failures: list[str] = []
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, value: Any, rule: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "value": value, "rule": rule})
        if not passed:
            failures.append(name)

    if construction == "not_constructible":
        failures.append("missing_frozen_v6_support_or_fit")
    else:
        # A frozen upstream hard veto remains authoritative, but the V6
        # coherent_turn/unstable_bilateral labels are diagnostics only in v2.
        check(
            "preexisting_hard_veto",
            not package.hard_veto_applied,
            package.hard_veto_applied,
            "frozen upstream hard veto must be absent",
        )

        direction_crossings = {
            name: bool((diagnostic.get(name) or {}).get("h2", {}).get("crossing"))
            for name in ("far_upper", "near_lower")
        }
        check(
            "basic_geometry_validity",
            not any(direction_crossings.values()),
            direction_crossings,
            "left/right ordering and positive width must remain valid at both H2 stops",
        )

        extension_rows = np.flatnonzero(np.any(extension, axis=1))
        extension_fills = [
            float(agreement[y].sum() / extension[y].sum())
            for y in extension_rows
            if extension[y].any()
        ]
        extension_p10 = (
            float(np.percentile(extension_fills, 10)) if extension_fills else None
        )
        if construction == "proposal_generated":
            check(
                "extension_structural_continuity",
                extension_p10 is not None
                and extension_p10 >= policy.minimum_extension_row_fill_p10,
                extension_p10,
                f"extension-only row-fill p10 >= {policy.minimum_extension_row_fill_p10:.2f}",
            )

        extension_y = set(map(int, extension_rows.tolist()))
        residual_rows = list(
            diagnostic.get("outside_seed_observation_consistency", {}).get(
                "residuals", []
            )
        )
        in_range_rows = [row for row in residual_rows if int(row["y"]) in extension_y]
        contradictory_count = sum(
            bool(row.get(contradiction_key)) for row in in_range_rows
        )
        contradiction_fraction = (
            float(contradictory_count / len(in_range_rows)) if in_range_rows else None
        )
        run_rows = [
            {**row, "diagnostic_contradiction": bool(row.get(contradiction_key))}
            for row in in_range_rows
        ]
        contradiction_run = _longest_consecutive_contradictory_rows(run_rows)
        gate_available = len(in_range_rows) >= policy.minimum_in_range_clean_rows
        persistent_conflict = bool(
            gate_available
            and contradiction_fraction is not None
            and contradiction_fraction
            >= policy.persistent_contradiction_minimum_fraction
            and contradiction_run >= policy.persistent_contradiction_minimum_run_rows
        )
        check(
            contradiction_check_name,
            not persistent_conflict,
            {
                "clean_rows": len(in_range_rows),
                "contradictory_rows": contradictory_count,
                "fraction": contradiction_fraction,
                "longest_consecutive_run": contradiction_run,
                "gate_available": gate_available,
            },
            (
                "fail only when in-extension clean rows >= "
                f"{policy.minimum_in_range_clean_rows}, contradiction fraction >= "
                f"{policy.persistent_contradiction_minimum_fraction:.2f}, and consecutive run >= "
                f"{policy.persistent_contradiction_minimum_run_rows}"
            ),
        )

    failures = list(dict.fromkeys(failures))
    if construction == "no_op" and not failures:
        status, selected_model, selected = "no_extension_needed", "observed_straight", observed.copy()
    elif construction == "proposal_generated" and not failures:
        status, selected_model, selected = "selected", "extended_straight", proposal.copy()
    else:
        status, selected_model, selected = "failed", None, empty.copy()

    outside_clean_rows = int(
        diagnostic.get("outside_seed_observation_consistency", {}).get("clean_row_count")
        or 0
    )
    enough_original_evidence = bool(
        package.support_y_range is not None
        and len(package.selected_band_centers) >= 3
        and outside_clean_rows >= policy.s2_minimum_outside_clean_rows
    )
    s2_eligible = bool(
        status == "failed" and enough_original_evidence and not package.hard_veto_applied
    )
    if status != "failed":
        handoff_reason = None
    elif s2_eligible:
        handoff_reason = "s1_failed_with_sufficient_original_v6_observations"
    elif package.hard_veto_applied:
        handoff_reason = "s1_failed_preexisting_hard_veto_s2_ineligible"
    else:
        handoff_reason = "s1_failed_insufficient_original_v6_observations"

    result = {
        "policy_id": policy_id,
        "threshold_status": "provisional_targeted_review",
        "s1_status": status,
        "s1_selected_model": selected_model,
        "s1_failure_reasons": failures,
        "s2_eligible": s2_eligible,
        "s2_handoff_reason": handoff_reason,
        "checks": checks,
        "effective_selected_area_px": int(selected.sum()),
        "candidate_area_px": int(proposal.sum()),
        "effective_selected_mask_sha256": mask_sha256(selected),
        "candidate_is_not_automatically_selected": True,
        "v6_direction_classification": package.direction_classification,
        "v6_direction_classification_used_as_gate": False,
        "user_feedback_specific_rules_added": False,
        "policy": {
            "minimum_extension_row_fill_p10": policy.minimum_extension_row_fill_p10,
            "minimum_in_range_clean_rows": policy.minimum_in_range_clean_rows,
            "persistent_contradiction_minimum_fraction": policy.persistent_contradiction_minimum_fraction,
            "persistent_contradiction_minimum_run_rows": policy.persistent_contradiction_minimum_run_rows,
            "s2_minimum_outside_clean_rows": policy.s2_minimum_outside_clean_rows,
        },
    }
    return result, {"effective_selected": selected}


def evaluate_s1_targeted_v2(
    package: FrozenV6ObservationPackage,
    diagnostic: Mapping[str, Any],
    masks: Mapping[str, np.ndarray],
    policy: S1TargetedValidationPolicyV2 | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate S1 with V2 absolute-residual observation conflicts."""

    return _evaluate_s1_targeted_reduced(
        package,
        diagnostic,
        masks,
        policy or S1TargetedValidationPolicyV2(),
        policy_id=S1_TARGETED_V2_POLICY_ID,
        contradiction_key="diagnostic_contradiction",
        contradiction_check_name="persistent_in_range_observation_contradiction",
    )


def evaluate_s1_targeted_v3(
    package: FrozenV6ObservationPackage,
    diagnostic: Mapping[str, Any],
    masks: Mapping[str, np.ndarray],
    policy: S1TargetedValidationPolicyV3 | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate S1 with signed containment; wider observations are not conflicts."""

    result, selected = _evaluate_s1_targeted_reduced(
        package,
        diagnostic,
        masks,
        policy or S1TargetedValidationPolicyV3(),
        policy_id=S1_TARGETED_V3_POLICY_ID,
        contradiction_key="containment_contradiction",
        contradiction_check_name="persistent_in_range_containment_violation",
    )
    result["observation_comparison"] = "signed_containment"
    return result, selected


def evaluate_s1_targeted_v4a(
    package: FrozenV6ObservationPackage,
    diagnostic: Mapping[str, Any],
    masks: Mapping[str, np.ndarray],
    policy: S1TargetedValidationPolicyV3 | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """V3 validation unchanged; generation alone stops at resolvable width."""

    result, selected = evaluate_s1_targeted_v3(
        package, diagnostic, masks, policy or S1TargetedValidationPolicyV3()
    )
    result["policy_id"] = S1_TARGETED_V4A_POLICY_ID
    result["threshold_status"] = "provisional_full160_ablation"
    result["user_feedback_specific_rules_added"] = True
    result["v4a_change"] = "generation_minimum_resolvable_width_only"
    return result, selected


def _v4b_adjusted_containment_row(
    package: FrozenV6ObservationPackage,
    row: Mapping[str, Any],
    obstacle: np.ndarray,
) -> tuple[bool, dict[str, Any], np.ndarray]:
    """Subtract only same-side occluder pixels from an outward containment gap."""

    neutral = np.zeros(package.image_shape[1], dtype=bool)
    if not bool(row.get("containment_contradiction")):
        return False, {
            "raw_contradiction": False,
            "adjusted_contradiction": False,
            "left": None,
            "right": None,
        }, neutral
    y = int(row["y"])
    evidence = next(
        (value for value in package.row_evidence if int(value["y"]) == y), None
    )
    if evidence is None:
        return True, {
            "raw_contradiction": True,
            "adjusted_contradiction": True,
            "reason": "missing_row_evidence",
        }, neutral
    tolerance = float(
        max(
            5.0,
            2.0
            * max(
                package.left_seed_p90_residual_px or 0.0,
                package.right_seed_p90_residual_px or 0.0,
            ),
        )
    )
    candidate_left = float(_line_x(package.left_slope, package.left_intercept, y))  # type: ignore[arg-type]
    candidate_right = float(_line_x(package.right_slope, package.right_intercept, y))  # type: ignore[arg-type]
    observed_left = float(evidence["left"])
    observed_right = float(evidence["right"])

    def adjusted_side(start: float, end: float, raw_violation: bool) -> dict[str, Any]:
        if not raw_violation:
            return {
                "raw_violation": False,
                "raw_gap_px": 0.0,
                "neutral_pixels": 0,
                "adjusted_gap_px": 0.0,
                "adjusted_violation": False,
            }
        lo = max(0, min(package.image_shape[1] - 1, int(np.floor(min(start, end)))))
        hi = max(0, min(package.image_shape[1] - 1, int(np.ceil(max(start, end)))))
        covered = obstacle[y, lo : hi + 1]
        neutral[lo : hi + 1] |= covered
        neutral_pixels = int(covered.sum())
        raw_gap = float(abs(end - start))
        adjusted_gap = max(0.0, raw_gap - neutral_pixels)
        return {
            "raw_violation": True,
            "raw_gap_px": raw_gap,
            "neutral_pixels": neutral_pixels,
            "adjusted_gap_px": adjusted_gap,
            "adjusted_violation": bool(adjusted_gap > tolerance),
        }

    left = adjusted_side(
        candidate_left,
        observed_left,
        observed_left - candidate_left > tolerance,
    )
    right = adjusted_side(
        observed_right,
        candidate_right,
        candidate_right - observed_right > tolerance,
    )
    adjusted_contradiction = bool(
        left["adjusted_violation"] or right["adjusted_violation"]
    )
    return adjusted_contradiction, {
        "raw_contradiction": True,
        "adjusted_contradiction": adjusted_contradiction,
        "tolerance_px": tolerance,
        "left": left,
        "right": right,
    }, neutral


def evaluate_s1_targeted_v4b(
    package: FrozenV6ObservationPackage,
    diagnostic: Mapping[str, Any],
    masks: Mapping[str, np.ndarray],
    obstacle_mask: np.ndarray,
    policy: S1TargetedValidationPolicyV3 | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """V4a candidate with exact-pixel occluder neutrality and no prefix salvage."""

    policy = policy or S1TargetedValidationPolicyV3()
    proposal = np.asarray(masks["proposal"], dtype=bool)
    observed = np.asarray(masks["observed_seed"], dtype=bool)
    extension = np.asarray(masks["extension_only"], dtype=bool)
    agreement = np.asarray(masks["structural_agreement"], dtype=bool)
    obstacle = np.asarray(obstacle_mask, dtype=bool)
    if obstacle.shape != package.image_shape:
        raise ValueError("S1 V4b obstacle mask must match the frozen V6 image shape")
    empty = np.zeros(package.image_shape, dtype=bool)
    construction = str(diagnostic.get("construction_status"))
    failures: list[str] = []
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, value: Any, rule: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "value": value, "rule": rule})
        if not passed:
            failures.append(name)

    structural_neutral = extension & ~agreement & obstacle
    row_fill_audit: dict[str, dict[str, Any]] = {}
    adjusted_fills: list[float] = []
    extension_rows = list(map(int, np.flatnonzero(np.any(extension, axis=1))))
    for y in extension_rows:
        area = int(extension[y].sum())
        supported = int((extension[y] & agreement[y]).sum())
        neutral_pixels = int(structural_neutral[y].sum())
        denominator = area - neutral_pixels
        adjusted = float(supported / denominator) if denominator > 0 else None
        if adjusted is not None:
            adjusted_fills.append(adjusted)
        row_fill_audit[str(y)] = {
            "raw_fill": float(supported / area) if area else None,
            "adjusted_fill": adjusted,
            "neutral_pixels": neutral_pixels,
            "denominator_pixels": denominator,
        }
    extension_p10 = (
        float(np.percentile(adjusted_fills, 10)) if adjusted_fills else None
    )

    residual_rows = list(
        diagnostic.get("outside_seed_observation_consistency", {}).get("residuals", [])
    )
    extension_y = set(extension_rows)
    in_range_rows = [row for row in residual_rows if int(row["y"]) in extension_y]
    containment_neutral = np.zeros(package.image_shape, dtype=bool)
    containment_audit: dict[str, dict[str, Any]] = {}
    adjusted_run_rows: list[dict[str, Any]] = []
    adjusted_contradictory_count = 0
    raw_contradictory_count = 0
    for row in in_range_rows:
        y = int(row["y"])
        adjusted, audit, neutral_row = _v4b_adjusted_containment_row(
            package, row, obstacle
        )
        containment_neutral[y] = neutral_row
        containment_audit[str(y)] = audit
        raw_contradictory_count += int(bool(row.get("containment_contradiction")))
        adjusted_contradictory_count += int(adjusted)
        adjusted_run_rows.append({**row, "diagnostic_contradiction": adjusted})
    contradiction_fraction = (
        float(adjusted_contradictory_count / len(in_range_rows))
        if in_range_rows
        else None
    )
    contradiction_run = _longest_consecutive_contradictory_rows(adjusted_run_rows)
    gate_available = len(in_range_rows) >= policy.minimum_in_range_clean_rows
    persistent_conflict = bool(
        gate_available
        and contradiction_fraction is not None
        and contradiction_fraction >= policy.persistent_contradiction_minimum_fraction
        and contradiction_run >= policy.persistent_contradiction_minimum_run_rows
    )

    if construction == "not_constructible":
        failures.append("missing_frozen_v6_support_or_fit")
    else:
        check(
            "preexisting_hard_veto",
            not package.hard_veto_applied,
            package.hard_veto_applied,
            "frozen upstream hard veto must be absent",
        )
        direction_crossings = {
            name: bool((diagnostic.get(name) or {}).get("h2", {}).get("crossing"))
            for name in ("far_upper", "near_lower")
        }
        check(
            "basic_geometry_validity",
            not any(direction_crossings.values()),
            direction_crossings,
            "left/right ordering must remain valid at both H2 stops",
        )
        if construction == "proposal_generated":
            check(
                "extension_structural_continuity",
                extension_p10 is not None
                and extension_p10 >= policy.minimum_extension_row_fill_p10,
                extension_p10,
                f"extension-only row-fill p10 after exact occluder exclusion >= {policy.minimum_extension_row_fill_p10:.2f}",
            )
        check(
            "persistent_in_range_containment_violation",
            not persistent_conflict,
            {
                "clean_rows": len(in_range_rows),
                "raw_contradictory_rows": raw_contradictory_count,
                "contradictory_rows": adjusted_contradictory_count,
                "fraction": contradiction_fraction,
                "longest_consecutive_run": contradiction_run,
                "gate_available": gate_available,
            },
            (
                "subtract same-side occluder pixels only; then fail when clean rows >= "
                f"{policy.minimum_in_range_clean_rows}, contradiction fraction >= "
                f"{policy.persistent_contradiction_minimum_fraction:.2f}, and run >= "
                f"{policy.persistent_contradiction_minimum_run_rows}"
            ),
        )

    failures = list(dict.fromkeys(failures))
    if construction == "no_op" and not failures:
        status, selected_model, selected = "no_extension_needed", "observed_straight", observed.copy()
    elif construction == "proposal_generated" and not failures:
        status, selected_model, selected = "selected", "extended_straight_v4b", proposal.copy()
    else:
        status, selected_model, selected = "failed", None, empty.copy()

    outside_clean_rows = int(
        diagnostic.get("outside_seed_observation_consistency", {}).get("clean_row_count") or 0
    )
    enough_original_evidence = bool(
        package.support_y_range is not None
        and len(package.selected_band_centers) >= 3
        and outside_clean_rows >= policy.s2_minimum_outside_clean_rows
    )
    s2_eligible = bool(status == "failed" and enough_original_evidence and not package.hard_veto_applied)
    if status != "failed":
        handoff_reason = None
    elif s2_eligible:
        handoff_reason = "s1_failed_with_sufficient_original_v6_observations"
    elif package.hard_veto_applied:
        handoff_reason = "s1_failed_preexisting_hard_veto_s2_ineligible"
    else:
        handoff_reason = "s1_failed_insufficient_original_v6_observations"

    result = {
        "policy_id": S1_TARGETED_V4B_POLICY_ID,
        "threshold_status": "provisional_full160_ablation",
        "s1_status": status,
        "s1_selected_model": selected_model,
        "s1_failure_reasons": failures,
        "s2_eligible": s2_eligible,
        "s2_handoff_reason": handoff_reason,
        "checks": checks,
        "effective_selected_area_px": int(selected.sum()),
        "candidate_area_px": int(proposal.sum()),
        "effective_selected_mask_sha256": mask_sha256(selected),
        "candidate_is_not_automatically_selected": True,
        "v6_direction_classification": package.direction_classification,
        "v6_direction_classification_used_as_gate": False,
        "user_feedback_specific_rules_added": True,
        "observation_comparison": "signed_containment_with_exact_same_side_occluder_neutrality",
        "v4b_change": "v4a_plus_exact_occluder_neutrality_no_prefix_salvage",
        "v4b_row_fill_audit": row_fill_audit,
        "v4b_containment_audit": containment_audit,
        "v4b_neutral_structural_pixels": int(structural_neutral.sum()),
        "v4b_neutral_containment_pixels": int(containment_neutral.sum()),
        "policy": {
            "minimum_extension_row_fill_p10": policy.minimum_extension_row_fill_p10,
            "minimum_in_range_clean_rows": policy.minimum_in_range_clean_rows,
            "persistent_contradiction_minimum_fraction": policy.persistent_contradiction_minimum_fraction,
            "persistent_contradiction_minimum_run_rows": policy.persistent_contradiction_minimum_run_rows,
            "s2_minimum_outside_clean_rows": policy.s2_minimum_outside_clean_rows,
        },
    }
    return result, {
        "effective_selected": selected,
        "occlusion_neutral_structural": structural_neutral,
        "occlusion_neutral_containment": containment_neutral,
    }


def evaluate_s1_targeted_v4c(
    package: FrozenV6ObservationPackage,
    diagnostic: Mapping[str, Any],
    masks: Mapping[str, np.ndarray],
    obstacle_mask: np.ndarray,
    policy: S1TargetedValidationPolicyV3 | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """V4b plus direction-local removal after seven consecutive unsafe rows."""

    policy = policy or S1TargetedValidationPolicyV3()
    base_decision, base_selected = evaluate_s1_targeted_v4b(
        package, diagnostic, masks, obstacle_mask, policy
    )
    original_extension = np.asarray(masks["extension_only"], dtype=bool)
    if base_decision["s1_status"] != "failed":
        result = dict(base_decision)
        result["policy_id"] = S1_TARGETED_V4C_POLICY_ID
        result["v4c_change"] = "v4b_plus_direction_local_safe_prefix_only"
        result["v4c_truncation"] = {
            "far_upper": {"applied": False, "reason": "v4b_already_accepted"},
            "near_lower": {"applied": False, "reason": "v4b_already_accepted"},
        }
        result["v4c_removed_extension_rows"] = []
        result["v4c_prefix_failure_persistence_rows"] = 7
        result["v4c_minimum_retained_extension_rows"] = 3
        selected = dict(base_selected)
        selected["validated_extension"] = original_extension.copy()
        selected["validated_proposal"] = np.asarray(masks["proposal"], dtype=bool).copy()
        return result, selected
    retained_extension = original_extension.copy()
    support_low, support_high = package.support_y_range or (0, -1)
    direction_orders = {
        "far_upper": list(range(support_low - 1, -1, -1)),
        "near_lower": list(range(support_high + 1, package.image_shape[0])),
    }
    persistence_rows = 7
    removed_rows: set[int] = set()
    truncation: dict[str, Any] = {}
    row_fill_audit = base_decision.get("v4b_row_fill_audit", {})
    containment_audit = base_decision.get("v4b_containment_audit", {})
    for name, ordered in direction_orders.items():
        rows = [y for y in ordered if original_extension[y].any()]
        run = 0
        first_bad_index: int | None = None
        for index, y in enumerate(rows):
            fill = row_fill_audit.get(str(y), {}).get("adjusted_fill")
            structural_bad = bool(
                fill is not None and float(fill) < policy.minimum_extension_row_fill_p10
            )
            containment_bad = bool(
                containment_audit.get(str(y), {}).get("adjusted_contradiction", False)
            )
            run = run + 1 if structural_bad or containment_bad else 0
            if run >= persistence_rows:
                first_bad_index = index - run + 1
                break
        if first_bad_index is None:
            truncation[name] = {
                "applied": False,
                "first_removed_y": None,
                "removed_row_count": 0,
                "retained_row_count": len(rows),
            }
            continue
        removed = rows[first_bad_index:]
        removed_rows.update(removed)
        truncation[name] = {
            "applied": True,
            "first_removed_y": int(removed[0]),
            "removed_row_count": len(removed),
            "retained_row_count": first_bad_index,
            "trigger": f"{persistence_rows}_consecutive_v4b_unsafe_rows",
        }
    for y in removed_rows:
        retained_extension[y] = False

    retained_proposal = np.asarray(masks["observed_seed"], dtype=bool) | retained_extension
    trimmed_masks = dict(masks)
    trimmed_masks["extension_only"] = retained_extension
    trimmed_masks["proposal"] = retained_proposal
    trimmed_masks["structural_agreement"] = (
        retained_proposal & np.asarray(masks["structural_agreement"], dtype=bool)
    )
    result, selected = evaluate_s1_targeted_v4b(
        package, diagnostic, trimmed_masks, obstacle_mask, policy
    )
    retained_row_count = int(np.any(retained_extension, axis=1).sum())
    minimum_retained_rows = 3
    if removed_rows:
        passed = retained_row_count >= minimum_retained_rows
        result["checks"].append(
            {
                "name": "minimum_safe_extension_prefix",
                "passed": passed,
                "value": {
                    "retained_rows": retained_row_count,
                    "original_rows": int(np.any(original_extension, axis=1).sum()),
                },
                "rule": f"retain at least {minimum_retained_rows} extension rows after prefix truncation",
            }
        )
        if not passed:
            result["s1_failure_reasons"] = list(
                dict.fromkeys(
                    [*result["s1_failure_reasons"], "minimum_safe_extension_prefix"]
                )
            )
            result["s1_status"] = "failed"
            result["s1_selected_model"] = None
            selected["effective_selected"] = np.zeros(package.image_shape, dtype=bool)
            result["effective_selected_area_px"] = 0
            result["effective_selected_mask_sha256"] = mask_sha256(
                selected["effective_selected"]
            )
    if result["s1_status"] == "selected":
        result["s1_selected_model"] = "extended_straight_v4c_safe_prefix"
    result["policy_id"] = S1_TARGETED_V4C_POLICY_ID
    result["v4c_change"] = "v4b_plus_direction_local_safe_prefix_only"
    result["v4c_truncation"] = truncation
    result["v4c_removed_extension_rows"] = sorted(removed_rows)
    result["v4c_prefix_failure_persistence_rows"] = persistence_rows
    result["v4c_minimum_retained_extension_rows"] = minimum_retained_rows
    result["candidate_area_px"] = int(retained_proposal.sum())
    selected["validated_extension"] = retained_extension
    selected["validated_proposal"] = retained_proposal
    return result, selected


def grade_s1_v4_probe_result(
    package: FrozenV6ObservationPackage,
    decision: Mapping[str, Any],
    selected_masks: Mapping[str, np.ndarray],
    generation_masks: Mapping[str, np.ndarray],
    *,
    top_bottom_width_ratio: float | None,
    policy: S1V4ProbeGradingPolicy | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Assign human-review routing after V4c without editing its candidate.

    ``ambiguous_hold`` preserves the V4c effective mask for dashboard review.
    ``selected_partial`` also preserves it and requests an S2 continuation.
    ``unchanged`` is the only fallback: when no safe extension remains, it
    exposes the frozen V6 seed instead of presenting an empty S1 result.
    """

    policy = policy or S1V4ProbeGradingPolicy()
    result = dict(decision)
    masks = {name: np.asarray(mask, dtype=bool).copy() for name, mask in selected_masks.items()}
    original_status = str(result.get("s1_status"))
    observed = np.asarray(generation_masks["observed_seed"], dtype=bool)
    original_extension = np.asarray(generation_masks["extension_only"], dtype=bool)
    validated_extension = np.asarray(
        masks.get("validated_extension", original_extension), dtype=bool
    )
    original_rows = int(np.any(original_extension, axis=1).sum())
    retained_rows = int(np.any(validated_extension, axis=1).sum())
    retained_fraction = (
        float(retained_rows / original_rows) if original_rows else None
    )

    checks_by_name = {
        str(check.get("name")): check for check in result.get("checks", [])
    }
    prefix_value = checks_by_name.get("minimum_safe_extension_prefix", {}).get(
        "value", {}
    )
    no_safe_extension = bool(
        original_status == "no_extension_needed"
        or (
            original_status == "failed"
            and package.v6_straight_disposition == "accepted"
            and int(prefix_value.get("retained_rows", -1)) == 0
        )
    )
    grading_reasons: list[str] = []
    quality_metrics: dict[str, Any] = {
        "original_status": original_status,
        "direction_classification": package.direction_classification,
        "retained_extension_rows": retained_rows,
        "original_extension_rows": original_rows,
        "retained_extension_fraction": retained_fraction,
        "top_bottom_width_ratio": top_bottom_width_ratio,
    }

    if no_safe_extension:
        result["s1_status"] = "unchanged"
        result["s1_selected_model"] = "frozen_v6_observed_seed"
        result["s1_internal_failure_reasons"] = list(
            result.get("s1_failure_reasons", [])
        )
        result["s1_failure_reasons"] = []
        result["s2_eligible"] = False
        result["s2_handoff_reason"] = None
        masks["effective_selected"] = observed.copy()
    elif original_status == "selected":
        containment = checks_by_name.get(
            "persistent_in_range_containment_violation", {}
        ).get("value", {})
        clean_rows = int(containment.get("clean_rows") or 0)
        fraction = containment.get("fraction")
        run = int(containment.get("longest_consecutive_run") or 0)
        quality_metrics.update(
            {
                "post_trim_clean_rows": clean_rows,
                "post_trim_containment_fraction": fraction,
                "post_trim_containment_run_rows": run,
            }
        )
        if fraction is not None and float(fraction) >= policy.ambiguous_containment_fraction:
            grading_reasons.append("containment_fraction_near_failure")
        if run >= policy.ambiguous_containment_run_rows:
            grading_reasons.append("containment_run_near_failure")
        if (
            retained_fraction is not None
            and retained_fraction < policy.ambiguous_retained_fraction
            and clean_rows < policy.retained_fraction_requires_clean_rows_below
        ):
            grading_reasons.append("small_retained_fraction_with_weak_clean_evidence")
        if (
            top_bottom_width_ratio is not None
            and float(top_bottom_width_ratio) > policy.maximum_top_bottom_width_ratio
        ):
            grading_reasons.append("non_converging_perspective_width")

        if grading_reasons:
            result["s1_status"] = "ambiguous_hold"
            result["s1_selected_model"] = "extended_straight_v4c_review_hold"
            result["s2_eligible"] = False
            result["s2_handoff_reason"] = "human_review_required_before_routing"
        elif package.direction_classification in policy.partial_direction_classes:
            result["s1_status"] = "selected_partial"
            result["s1_selected_model"] = "extended_straight_v4c_partial_prefix"
            result["s2_eligible"] = True
            result["s2_handoff_reason"] = "selected_partial_requires_s2_continuation"

    result["policy_id"] = S1_V4_PROBE_POLICY_ID
    result["quality_grade"] = result["s1_status"]
    result["quality_grading_reasons"] = grading_reasons
    result["quality_metrics"] = quality_metrics
    result["human_review_required"] = result["s1_status"] == "ambiguous_hold"
    result["automatic_application_recommended"] = result["s1_status"] in {
        "selected",
        "selected_partial",
        "unchanged",
    }
    result["grading_policy"] = {
        "ambiguous_containment_fraction": policy.ambiguous_containment_fraction,
        "ambiguous_containment_run_rows": policy.ambiguous_containment_run_rows,
        "ambiguous_retained_fraction": policy.ambiguous_retained_fraction,
        "retained_fraction_requires_clean_rows_below": policy.retained_fraction_requires_clean_rows_below,
        "maximum_top_bottom_width_ratio": policy.maximum_top_bottom_width_ratio,
        "partial_direction_classes": list(policy.partial_direction_classes),
    }
    effective = masks["effective_selected"]
    result["effective_selected_area_px"] = int(effective.sum())
    result["effective_selected_mask_sha256"] = mask_sha256(effective)
    return result, masks


def _v4_occlusion_adjusted_row_fill(
    extension_row: np.ndarray,
    agreement_row: np.ndarray,
    obstacle_row: np.ndarray,
    policy: S1TargetedValidationPolicyV4,
) -> tuple[float, dict[str, Any]]:
    area = int(extension_row.sum())
    if area == 0:
        return 0.0, {
            "raw_fill": None,
            "adjusted_fill": None,
            "excluded_pixels": 0,
            "occlusion_applied": False,
        }
    supported = int((extension_row & agreement_row).sum())
    missing = extension_row & ~agreement_row
    missing_count = int(missing.sum())
    occluded_missing = missing & obstacle_row
    excluded = int(occluded_missing.sum())
    excluded_fraction = float(excluded / area)
    explained_fraction = float(excluded / missing_count) if missing_count else 0.0
    apply_occlusion = bool(
        excluded > 0
        and excluded_fraction <= policy.maximum_occlusion_excluded_row_fraction
        and explained_fraction
        >= policy.minimum_missing_fraction_explained_by_occlusion
        and area - excluded > 0
    )
    denominator = area - excluded if apply_occlusion else area
    adjusted = float(supported / denominator) if denominator else 0.0
    return adjusted, {
        "raw_fill": float(supported / area),
        "adjusted_fill": adjusted,
        "excluded_pixels": excluded if apply_occlusion else 0,
        "excluded_fraction": excluded_fraction if apply_occlusion else 0.0,
        "missing_explained_fraction": explained_fraction,
        "occlusion_applied": apply_occlusion,
    }


def _v4_containment_occlusion_explained(
    package: FrozenV6ObservationPackage,
    row: Mapping[str, Any],
    obstacle: np.ndarray,
    policy: S1TargetedValidationPolicyV4,
) -> tuple[bool, dict[str, Any]]:
    if not bool(row.get("containment_contradiction")):
        return False, {"left": False, "right": False, "reason": "not_contradictory"}
    y = int(row["y"])
    tolerance = float(
        max(
            5.0,
            2.0
            * max(
                package.left_seed_p90_residual_px or 0.0,
                package.right_seed_p90_residual_px or 0.0,
            ),
        )
    )
    evidence = next(
        (value for value in package.row_evidence if int(value["y"]) == y), None
    )
    if evidence is None:
        return False, {"left": False, "right": False, "reason": "missing_row_evidence"}
    candidate_left = float(_line_x(package.left_slope, package.left_intercept, y))  # type: ignore[arg-type]
    candidate_right = float(_line_x(package.right_slope, package.right_intercept, y))  # type: ignore[arg-type]
    observed_left = float(evidence["left"])
    observed_right = float(evidence["right"])
    max_gap = max(12, int(round(package.image_shape[1] * policy.maximum_containment_occlusion_gap_fraction)))

    def covered(start: float, end: float) -> tuple[bool, int, float]:
        lo = max(0, min(package.image_shape[1] - 1, int(np.floor(min(start, end)))))
        hi = max(0, min(package.image_shape[1] - 1, int(np.ceil(max(start, end)))))
        gap = hi - lo + 1
        fraction = float(obstacle[y, lo : hi + 1].mean()) if gap > 0 else 0.0
        return bool(
            gap <= max_gap
            and fraction >= policy.minimum_containment_gap_occlusion_fraction
        ), gap, fraction

    left_violation = observed_left - candidate_left > tolerance
    right_violation = candidate_right - observed_right > tolerance
    left_ok, left_gap, left_fraction = covered(candidate_left, observed_left) if left_violation else (False, 0, 0.0)
    right_ok, right_gap, right_fraction = covered(observed_right, candidate_right) if right_violation else (False, 0, 0.0)
    explained = bool(
        (not left_violation or left_ok)
        and (not right_violation or right_ok)
        and (left_violation or right_violation)
    )
    return explained, {
        "left": left_ok,
        "right": right_ok,
        "left_gap_px": left_gap,
        "right_gap_px": right_gap,
        "left_occlusion_fraction": left_fraction,
        "right_occlusion_fraction": right_fraction,
        "maximum_gap_px": max_gap,
        "reason": "all_outward_gaps_occluded" if explained else "unexplained_outward_gap",
    }


def evaluate_s1_targeted_v4(
    package: FrozenV6ObservationPackage,
    diagnostic: Mapping[str, Any],
    masks: Mapping[str, np.ndarray],
    obstacle_mask: np.ndarray,
    policy: S1TargetedValidationPolicyV4 | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate V4 with narrow-tip generation, occlusion neutrality, and safe prefixes."""

    policy = policy or S1TargetedValidationPolicyV4()
    observed = np.asarray(masks["observed_seed"], dtype=bool)
    original_extension = np.asarray(masks["extension_only"], dtype=bool)
    agreement = np.asarray(masks["structural_agreement"], dtype=bool)
    obstacle = np.asarray(obstacle_mask, dtype=bool)
    if obstacle.shape != package.image_shape:
        raise ValueError("S1 V4 obstacle mask must match the frozen V6 image shape")
    retained_extension = original_extension.copy()
    empty = np.zeros(package.image_shape, dtype=bool)
    construction = str(diagnostic.get("construction_status"))

    row_fill: dict[int, float] = {}
    row_fill_audit: dict[int, dict[str, Any]] = {}
    for y in map(int, np.flatnonzero(np.any(original_extension, axis=1))):
        value, audit = _v4_occlusion_adjusted_row_fill(
            original_extension[y], agreement[y], obstacle[y], policy
        )
        row_fill[y] = value
        row_fill_audit[y] = audit

    residual_rows = list(
        diagnostic.get("outside_seed_observation_consistency", {}).get("residuals", [])
    )
    residual_by_y = {int(row["y"]): row for row in residual_rows}
    occlusion_explained: dict[int, bool] = {}
    containment_audit: dict[int, dict[str, Any]] = {}
    for y, row in residual_by_y.items():
        explained, audit = _v4_containment_occlusion_explained(
            package, row, obstacle, policy
        )
        occlusion_explained[y] = explained
        containment_audit[y] = audit

    support_low, support_high = package.support_y_range or (0, -1)
    direction_orders = {
        "far_upper": list(range(support_low - 1, -1, -1)),
        "near_lower": list(range(support_high + 1, package.image_shape[0])),
    }
    truncation: dict[str, Any] = {}
    removed_rows: set[int] = set()
    for name, ordered_rows in direction_orders.items():
        rows = [y for y in ordered_rows if original_extension[y].any()]
        bad_run = 0
        first_bad_index: int | None = None
        for index, y in enumerate(rows):
            residual = residual_by_y.get(y)
            unexplained_containment = bool(
                residual
                and residual.get("containment_contradiction")
                and not occlusion_explained.get(y, False)
            )
            structurally_bad = row_fill.get(y, 0.0) < policy.minimum_extension_row_fill_p10
            if structurally_bad or unexplained_containment:
                bad_run += 1
                if bad_run >= policy.prefix_failure_persistence_rows:
                    first_bad_index = index - bad_run + 1
                    break
            else:
                bad_run = 0
        if first_bad_index is not None:
            removed = rows[first_bad_index:]
            removed_rows.update(removed)
            truncation[name] = {
                "applied": True,
                "first_removed_y": int(removed[0]),
                "removed_row_count": len(removed),
                "retained_row_count": first_bad_index,
                "trigger": f"{policy.prefix_failure_persistence_rows}_consecutive_unsafe_rows",
            }
        else:
            truncation[name] = {
                "applied": False,
                "first_removed_y": None,
                "removed_row_count": 0,
                "retained_row_count": len(rows),
                "trigger": None,
            }
    for y in removed_rows:
        retained_extension[y] = False

    retained_rows = list(map(int, np.flatnonzero(np.any(retained_extension, axis=1))))
    retained_proposal = observed | retained_extension
    retained_fills = [row_fill[y] for y in retained_rows]
    extension_p10 = float(np.percentile(retained_fills, 10)) if retained_fills else None
    in_range_rows = [
        row
        for row in residual_rows
        if retained_extension[int(row["y"])].any()
        and not occlusion_explained.get(int(row["y"]), False)
    ]
    contradictory_count = sum(
        bool(row.get("containment_contradiction")) for row in in_range_rows
    )
    contradiction_fraction = (
        float(contradictory_count / len(in_range_rows)) if in_range_rows else None
    )
    run_rows = [
        {**row, "diagnostic_contradiction": bool(row.get("containment_contradiction"))}
        for row in in_range_rows
    ]
    contradiction_run = _longest_consecutive_contradictory_rows(run_rows)
    gate_available = len(in_range_rows) >= policy.minimum_in_range_clean_rows
    persistent_conflict = bool(
        gate_available
        and contradiction_fraction is not None
        and contradiction_fraction >= policy.persistent_contradiction_minimum_fraction
        and contradiction_run >= policy.persistent_contradiction_minimum_run_rows
    )

    failures: list[str] = []
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, value: Any, rule: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "value": value, "rule": rule})
        if not passed:
            failures.append(name)

    if construction == "not_constructible":
        failures.append("missing_frozen_v6_support_or_fit")
    else:
        check(
            "preexisting_hard_veto",
            not package.hard_veto_applied,
            package.hard_veto_applied,
            "frozen upstream hard veto must be absent",
        )
        direction_crossings = {
            name: bool((diagnostic.get(name) or {}).get("h2", {}).get("crossing"))
            for name in ("far_upper", "near_lower")
        }
        check(
            "basic_geometry_validity",
            not any(direction_crossings.values()),
            direction_crossings,
            "left/right ordering must remain valid; crossing remains a safety backstop",
        )
        if construction == "proposal_generated":
            check(
                "minimum_safe_extension_prefix",
                len(retained_rows) >= policy.minimum_retained_extension_rows,
                {
                    "retained_rows": len(retained_rows),
                    "original_rows": int(np.any(original_extension, axis=1).sum()),
                    "truncation": truncation,
                },
                f"retain at least {policy.minimum_retained_extension_rows} seed-connected extension rows",
            )
            check(
                "occlusion_adjusted_extension_structural_continuity",
                extension_p10 is not None
                and extension_p10 >= policy.minimum_extension_row_fill_p10,
                {
                    "p10": extension_p10,
                    "occlusion_adjusted_rows": sum(
                        bool(value["occlusion_applied"])
                        for y, value in row_fill_audit.items()
                        if y in retained_rows
                    ),
                },
                f"retained extension occlusion-adjusted row-fill p10 >= {policy.minimum_extension_row_fill_p10:.2f}",
            )
        check(
            "occlusion_adjusted_persistent_containment_violation",
            not persistent_conflict,
            {
                "clean_rows": len(in_range_rows),
                "occlusion_excluded_rows": sum(
                    bool(value) for y, value in occlusion_explained.items()
                    if original_extension[y].any()
                ),
                "contradictory_rows": contradictory_count,
                "fraction": contradiction_fraction,
                "longest_consecutive_run": contradiction_run,
                "gate_available": gate_available,
            },
            (
                "after causal obstacle occlusion exclusion, fail only when clean rows >= "
                f"{policy.minimum_in_range_clean_rows}, contradiction fraction >= "
                f"{policy.persistent_contradiction_minimum_fraction:.2f}, and run >= "
                f"{policy.persistent_contradiction_minimum_run_rows}"
            ),
        )

    failures = list(dict.fromkeys(failures))
    if construction == "no_op" and not failures:
        status, selected_model, selected = "no_extension_needed", "observed_straight", observed.copy()
    elif construction == "proposal_generated" and not failures:
        status, selected_model, selected = "selected", "extended_straight_v4_safe_prefix", retained_proposal.copy()
    else:
        status, selected_model, selected = "failed", None, empty.copy()

    outside_clean_rows = int(
        diagnostic.get("outside_seed_observation_consistency", {}).get("clean_row_count") or 0
    )
    enough_original_evidence = bool(
        package.support_y_range is not None
        and len(package.selected_band_centers) >= 3
        and outside_clean_rows >= policy.s2_minimum_outside_clean_rows
    )
    s2_eligible = bool(status == "failed" and enough_original_evidence and not package.hard_veto_applied)
    if status != "failed":
        handoff_reason = None
    elif s2_eligible:
        handoff_reason = "s1_failed_with_sufficient_original_v6_observations"
    elif package.hard_veto_applied:
        handoff_reason = "s1_failed_preexisting_hard_veto_s2_ineligible"
    else:
        handoff_reason = "s1_failed_insufficient_original_v6_observations"

    result = {
        "policy_id": S1_TARGETED_V4_POLICY_ID,
        "threshold_status": "provisional_full160_ablation",
        "s1_status": status,
        "s1_selected_model": selected_model,
        "s1_failure_reasons": failures,
        "s2_eligible": s2_eligible,
        "s2_handoff_reason": handoff_reason,
        "checks": checks,
        "effective_selected_area_px": int(selected.sum()),
        "candidate_area_px": int(retained_proposal.sum()),
        "original_candidate_area_px": int((observed | original_extension).sum()),
        "effective_selected_mask_sha256": mask_sha256(selected),
        "candidate_is_not_automatically_selected": True,
        "v6_direction_classification": package.direction_classification,
        "v6_direction_classification_used_as_gate": False,
        "user_feedback_specific_rules_added": True,
        "observation_comparison": "signed_containment_with_causal_occlusion_exclusion",
        "v4_truncation": truncation,
        "v4_removed_extension_rows": sorted(removed_rows),
        "v4_row_fill_audit": {str(y): value for y, value in row_fill_audit.items()},
        "v4_containment_occlusion_audit": {str(y): value for y, value in containment_audit.items()},
        "policy": {
            "minimum_extension_row_fill_p10": policy.minimum_extension_row_fill_p10,
            "minimum_in_range_clean_rows": policy.minimum_in_range_clean_rows,
            "persistent_contradiction_minimum_fraction": policy.persistent_contradiction_minimum_fraction,
            "persistent_contradiction_minimum_run_rows": policy.persistent_contradiction_minimum_run_rows,
            "prefix_failure_persistence_rows": policy.prefix_failure_persistence_rows,
            "maximum_occlusion_excluded_row_fraction": policy.maximum_occlusion_excluded_row_fraction,
            "minimum_missing_fraction_explained_by_occlusion": policy.minimum_missing_fraction_explained_by_occlusion,
            "minimum_containment_gap_occlusion_fraction": policy.minimum_containment_gap_occlusion_fraction,
            "maximum_containment_occlusion_gap_fraction": policy.maximum_containment_occlusion_gap_fraction,
            "minimum_retained_extension_rows": policy.minimum_retained_extension_rows,
            "s2_minimum_outside_clean_rows": policy.s2_minimum_outside_clean_rows,
        },
    }
    return result, {
        "effective_selected": selected,
        "validated_extension": retained_extension,
        "validated_proposal": retained_proposal,
    }


__all__ = [
    "FrozenV6ObservationPackage",
    "S1AGenerationPolicy",
    "S1TargetedValidationPolicy",
    "S1TargetedValidationPolicyV2",
    "S1TargetedValidationPolicyV3",
    "S1TargetedValidationPolicyV4",
    "S1V4ProbeGradingPolicy",
    "S1A_POLICY_ID",
    "S1_TARGETED_POLICY_ID",
    "S1_TARGETED_V2_POLICY_ID",
    "S1_TARGETED_V3_POLICY_ID",
    "S1_TARGETED_V4A_POLICY_ID",
    "S1_TARGETED_V4B_POLICY_ID",
    "S1_TARGETED_V4C_POLICY_ID",
    "S1_V4_PROBE_POLICY_ID",
    "S1_TARGETED_V4_POLICY_ID",
    "evaluate_s1_targeted",
    "evaluate_s1_targeted_v2",
    "evaluate_s1_targeted_v3",
    "evaluate_s1_targeted_v4a",
    "evaluate_s1_targeted_v4b",
    "evaluate_s1_targeted_v4c",
    "grade_s1_v4_probe_result",
    "evaluate_s1_targeted_v4",
    "freeze_v6_observation",
    "generate_s1a_proposal",
    "mask_sha256",
]
