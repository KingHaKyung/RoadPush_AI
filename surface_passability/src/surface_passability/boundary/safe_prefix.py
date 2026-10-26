"""Retain a direction-local safe prefix of a generated boundary extension."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from surface_passability.boundary.containment import (
    evaluate_obstacle_aware_containment,
)
from surface_passability.contracts.boundary_extension import (
    BoundaryObservationSnapshot,
)
from surface_passability.io.hashing import hash_mask_after_uint8_cast
from surface_passability.policies.containment import BoundaryContainmentPolicy
from surface_passability.policies.safe_prefix import SafePrefixPolicy


POLICY_ID = "phase7_extension_track_s1_targeted_provisional_v4c_safe_prefix"


def evaluate_safe_extension_prefix(
    snapshot: BoundaryObservationSnapshot,
    diagnostic: Mapping[str, Any],
    masks: Mapping[str, np.ndarray],
    obstacle_mask: np.ndarray,
    containment_policy: BoundaryContainmentPolicy | None = None,
    prefix_policy: SafePrefixPolicy | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Trim unsafe direction suffixes, then re-evaluate retained geometry."""

    containment_policy = containment_policy or BoundaryContainmentPolicy()
    prefix_policy = prefix_policy or SafePrefixPolicy()
    base_decision, base_selected = evaluate_obstacle_aware_containment(
        snapshot,
        diagnostic,
        masks,
        obstacle_mask,
        containment_policy,
    )
    original_extension = np.asarray(masks["extension_only"], dtype=bool)
    if base_decision["s1_status"] != "failed":
        result = dict(base_decision)
        result["policy_id"] = POLICY_ID
        result["v4c_change"] = "v4b_plus_direction_local_safe_prefix_only"
        result["v4c_truncation"] = {
            "far_upper": {
                "applied": False,
                "reason": "v4b_already_accepted",
            },
            "near_lower": {
                "applied": False,
                "reason": "v4b_already_accepted",
            },
        }
        result["v4c_removed_extension_rows"] = []
        result["v4c_prefix_failure_persistence_rows"] = (
            prefix_policy.failure_persistence_rows
        )
        result["v4c_minimum_retained_extension_rows"] = (
            prefix_policy.minimum_retained_extension_rows
        )
        selected = dict(base_selected)
        selected["validated_extension"] = original_extension.copy()
        selected["validated_proposal"] = np.asarray(
            masks["proposal"], dtype=bool
        ).copy()
        return result, selected

    retained_extension = original_extension.copy()
    support_low, support_high = snapshot.support_y_range or (0, -1)
    direction_orders = {
        "far_upper": list(range(support_low - 1, -1, -1)),
        "near_lower": list(
            range(support_high + 1, snapshot.image_shape[0])
        ),
    }
    persistence_rows = prefix_policy.failure_persistence_rows
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
                fill is not None
                and float(fill)
                < containment_policy.minimum_extension_row_fill_p10
            )
            containment_bad = bool(
                containment_audit.get(str(y), {}).get(
                    "adjusted_contradiction", False
                )
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

    retained_proposal = (
        np.asarray(masks["observed_seed"], dtype=bool) | retained_extension
    )
    trimmed_masks = dict(masks)
    trimmed_masks["extension_only"] = retained_extension
    trimmed_masks["proposal"] = retained_proposal
    trimmed_masks["structural_agreement"] = (
        retained_proposal
        & np.asarray(masks["structural_agreement"], dtype=bool)
    )
    result, selected = evaluate_obstacle_aware_containment(
        snapshot,
        diagnostic,
        trimmed_masks,
        obstacle_mask,
        containment_policy,
    )
    retained_row_count = int(np.any(retained_extension, axis=1).sum())
    minimum_retained_rows = prefix_policy.minimum_retained_extension_rows
    if removed_rows:
        passed = retained_row_count >= minimum_retained_rows
        result["checks"].append(
            {
                "name": "minimum_safe_extension_prefix",
                "passed": passed,
                "value": {
                    "retained_rows": retained_row_count,
                    "original_rows": int(
                        np.any(original_extension, axis=1).sum()
                    ),
                },
                "rule": (
                    f"retain at least {minimum_retained_rows} extension rows "
                    "after prefix truncation"
                ),
            }
        )
        if not passed:
            result["s1_failure_reasons"] = list(
                dict.fromkeys(
                    [
                        *result["s1_failure_reasons"],
                        "minimum_safe_extension_prefix",
                    ]
                )
            )
            result["s1_status"] = "failed"
            result["s1_selected_model"] = None
            selected["effective_selected"] = np.zeros(
                snapshot.image_shape, dtype=bool
            )
            result["effective_selected_area_px"] = 0
            result["effective_selected_mask_sha256"] = (
                hash_mask_after_uint8_cast(selected["effective_selected"])
            )
    if result["s1_status"] == "selected":
        result["s1_selected_model"] = (
            "extended_straight_v4c_safe_prefix"
        )
    result["policy_id"] = POLICY_ID
    result["v4c_change"] = "v4b_plus_direction_local_safe_prefix_only"
    result["v4c_truncation"] = truncation
    result["v4c_removed_extension_rows"] = sorted(removed_rows)
    result["v4c_prefix_failure_persistence_rows"] = persistence_rows
    result["v4c_minimum_retained_extension_rows"] = minimum_retained_rows
    result["candidate_area_px"] = int(retained_proposal.sum())
    selected["validated_extension"] = retained_extension
    selected["validated_proposal"] = retained_proposal
    return result, selected
