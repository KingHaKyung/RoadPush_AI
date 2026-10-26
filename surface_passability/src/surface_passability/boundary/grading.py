"""Assign review and continuation routes to safe boundary geometry."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from surface_passability.contracts.boundary_extension import (
    BoundaryObservationSnapshot,
)
from surface_passability.io.hashing import hash_mask_after_uint8_cast
from surface_passability.policies.grading import BoundaryQualityPolicy


POLICY_ID = "phase7_extension_track_s1_v4_probe_human_review_grading"


def classify_straight_boundary_disposition(
    result: Mapping[str, Any],
) -> str:
    """Classify the accepted straight fit before extension and grading."""

    if result.get("selected_surface") is None:
        return "uncertain"
    if result["envelope_fit"]["status"] != "accepted":
        return "rejected"
    if result["blocker_reuse"]["warnings"]:
        return "boundary_valid_but_blocker_suspect"
    return "accepted"


def grade_boundary_extension(
    snapshot: BoundaryObservationSnapshot,
    decision: Mapping[str, Any],
    selected_masks: Mapping[str, np.ndarray],
    generation_masks: Mapping[str, np.ndarray],
    *,
    top_bottom_width_ratio: float | None,
    policy: BoundaryQualityPolicy | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Route one safe candidate while preserving the selected mask geometry."""

    policy = policy or BoundaryQualityPolicy()
    result = dict(decision)
    masks = {
        name: np.asarray(mask, dtype=bool).copy()
        for name, mask in selected_masks.items()
    }
    original_status = str(result.get("s1_status"))
    observed = np.asarray(generation_masks["observed_seed"], dtype=bool)
    original_extension = np.asarray(
        generation_masks["extension_only"], dtype=bool
    )
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
    prefix_value = checks_by_name.get(
        "minimum_safe_extension_prefix", {}
    ).get("value", {})
    no_safe_extension = bool(
        original_status == "no_extension_needed"
        or (
            original_status == "failed"
            and snapshot.straight_disposition == "accepted"
            and int(prefix_value.get("retained_rows", -1)) == 0
        )
    )
    grading_reasons: list[str] = []
    quality_metrics: dict[str, Any] = {
        "original_status": original_status,
        "direction_classification": snapshot.direction_classification,
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
        if (
            fraction is not None
            and float(fraction) >= policy.ambiguous_containment_fraction
        ):
            grading_reasons.append("containment_fraction_near_failure")
        if run >= policy.ambiguous_containment_run_rows:
            grading_reasons.append("containment_run_near_failure")
        if (
            retained_fraction is not None
            and retained_fraction < policy.ambiguous_retained_fraction
            and clean_rows
            < policy.retained_fraction_requires_clean_rows_below
        ):
            grading_reasons.append(
                "small_retained_fraction_with_weak_clean_evidence"
            )
        if (
            top_bottom_width_ratio is not None
            and float(top_bottom_width_ratio)
            > policy.maximum_top_bottom_width_ratio
        ):
            grading_reasons.append("non_converging_perspective_width")

        if grading_reasons:
            result["s1_status"] = "ambiguous_hold"
            result["s1_selected_model"] = (
                "extended_straight_v4c_review_hold"
            )
            result["s2_eligible"] = False
            result["s2_handoff_reason"] = (
                "human_review_required_before_routing"
            )
        elif (
            snapshot.direction_classification
            in policy.partial_direction_classes
        ):
            result["s1_status"] = "selected_partial"
            result["s1_selected_model"] = (
                "extended_straight_v4c_partial_prefix"
            )
            result["s2_eligible"] = True
            result["s2_handoff_reason"] = (
                "selected_partial_requires_s2_continuation"
            )

    result["policy_id"] = POLICY_ID
    result["quality_grade"] = result["s1_status"]
    result["quality_grading_reasons"] = grading_reasons
    result["quality_metrics"] = quality_metrics
    result["human_review_required"] = (
        result["s1_status"] == "ambiguous_hold"
    )
    result["automatic_application_recommended"] = result["s1_status"] in {
        "selected",
        "selected_partial",
        "unchanged",
    }
    result["grading_policy"] = {
        "ambiguous_containment_fraction": (
            policy.ambiguous_containment_fraction
        ),
        "ambiguous_containment_run_rows": (
            policy.ambiguous_containment_run_rows
        ),
        "ambiguous_retained_fraction": policy.ambiguous_retained_fraction,
        "retained_fraction_requires_clean_rows_below": (
            policy.retained_fraction_requires_clean_rows_below
        ),
        "maximum_top_bottom_width_ratio": (
            policy.maximum_top_bottom_width_ratio
        ),
        "partial_direction_classes": list(policy.partial_direction_classes),
    }
    effective = masks["effective_selected"]
    result["effective_selected_area_px"] = int(effective.sum())
    result["effective_selected_mask_sha256"] = hash_mask_after_uint8_cast(
        effective
    )
    return result, masks
