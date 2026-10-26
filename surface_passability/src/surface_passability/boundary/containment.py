"""Validate boundary extensions against signed, obstacle-aware containment."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from surface_passability.boundary.extension import line_column
from surface_passability.contracts.boundary_extension import (
    BoundaryObservationSnapshot,
)
from surface_passability.io.hashing import hash_mask_after_uint8_cast
from surface_passability.policies.containment import BoundaryContainmentPolicy


POLICY_ID = "phase7_extension_track_s1_targeted_provisional_v4b_occluder_neutral"


def longest_consecutive_conflict_rows(rows: list[Mapping[str, Any]]) -> int:
    """Count the longest adjacent run marked as a diagnostic conflict."""

    longest = current = 0
    previous_y: int | None = None
    for row in sorted(rows, key=lambda value: int(value["y"])):
        y = int(row["y"])
        if bool(row.get("diagnostic_contradiction")):
            current = (
                current + 1
                if previous_y is not None and y == previous_y + 1
                else 1
            )
            longest = max(longest, current)
        else:
            current = 0
        previous_y = y
    return longest


def adjust_containment_row_for_obstacle(
    snapshot: BoundaryObservationSnapshot,
    row: Mapping[str, Any],
    obstacle: np.ndarray,
) -> tuple[bool, dict[str, Any], np.ndarray]:
    """Subtract only same-side obstacle pixels from an outward boundary gap."""

    neutral = np.zeros(snapshot.image_shape[1], dtype=bool)
    if not bool(row.get("containment_contradiction")):
        return False, {
            "raw_contradiction": False,
            "adjusted_contradiction": False,
            "left": None,
            "right": None,
        }, neutral
    y = int(row["y"])
    evidence = next(
        (value for value in snapshot.row_evidence if int(value["y"]) == y),
        None,
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
                snapshot.left_seed_p90_residual_px or 0.0,
                snapshot.right_seed_p90_residual_px or 0.0,
            ),
        )
    )
    candidate_left = float(
        line_column(
            snapshot.left_slope,  # type: ignore[arg-type]
            snapshot.left_intercept,  # type: ignore[arg-type]
            y,
        )
    )
    candidate_right = float(
        line_column(
            snapshot.right_slope,  # type: ignore[arg-type]
            snapshot.right_intercept,  # type: ignore[arg-type]
            y,
        )
    )
    observed_left = float(evidence["left"])
    observed_right = float(evidence["right"])

    def adjusted_side(
        start: float,
        end: float,
        raw_violation: bool,
    ) -> dict[str, Any]:
        if not raw_violation:
            return {
                "raw_violation": False,
                "raw_gap_px": 0.0,
                "neutral_pixels": 0,
                "adjusted_gap_px": 0.0,
                "adjusted_violation": False,
            }
        low = max(
            0,
            min(
                snapshot.image_shape[1] - 1,
                int(np.floor(min(start, end))),
            ),
        )
        high = max(
            0,
            min(
                snapshot.image_shape[1] - 1,
                int(np.ceil(max(start, end))),
            ),
        )
        covered = obstacle[y, low : high + 1]
        neutral[low : high + 1] |= covered
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


def evaluate_obstacle_aware_containment(
    snapshot: BoundaryObservationSnapshot,
    diagnostic: Mapping[str, Any],
    masks: Mapping[str, np.ndarray],
    obstacle_mask: np.ndarray,
    policy: BoundaryContainmentPolicy | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Select a generated boundary when persistent adjusted conflicts are absent."""

    policy = policy or BoundaryContainmentPolicy()
    proposal = np.asarray(masks["proposal"], dtype=bool)
    observed = np.asarray(masks["observed_seed"], dtype=bool)
    extension = np.asarray(masks["extension_only"], dtype=bool)
    agreement = np.asarray(masks["structural_agreement"], dtype=bool)
    obstacle = np.asarray(obstacle_mask, dtype=bool)
    if obstacle.shape != snapshot.image_shape:
        raise ValueError(
            "S1 V4b obstacle mask must match the frozen V6 image shape"
        )
    empty = np.zeros(snapshot.image_shape, dtype=bool)
    construction = str(diagnostic.get("construction_status"))
    failures: list[str] = []
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, value: Any, rule: str) -> None:
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "value": value,
                "rule": rule,
            }
        )
        if not passed:
            failures.append(name)

    structural_neutral = extension & ~agreement & obstacle
    row_fill_audit: dict[str, dict[str, Any]] = {}
    adjusted_fills: list[float] = []
    extension_rows = list(
        map(int, np.flatnonzero(np.any(extension, axis=1)))
    )
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
        float(np.percentile(adjusted_fills, 10))
        if adjusted_fills
        else None
    )

    residual_rows = list(
        diagnostic.get("outside_seed_observation_consistency", {}).get(
            "residuals", []
        )
    )
    extension_y = set(extension_rows)
    in_range_rows = [
        row for row in residual_rows if int(row["y"]) in extension_y
    ]
    containment_neutral = np.zeros(snapshot.image_shape, dtype=bool)
    containment_audit: dict[str, dict[str, Any]] = {}
    adjusted_run_rows: list[dict[str, Any]] = []
    adjusted_contradictory_count = 0
    raw_contradictory_count = 0
    for row in in_range_rows:
        y = int(row["y"])
        adjusted, audit, neutral_row = adjust_containment_row_for_obstacle(
            snapshot,
            row,
            obstacle,
        )
        containment_neutral[y] = neutral_row
        containment_audit[str(y)] = audit
        raw_contradictory_count += int(
            bool(row.get("containment_contradiction"))
        )
        adjusted_contradictory_count += int(adjusted)
        adjusted_run_rows.append(
            {**row, "diagnostic_contradiction": adjusted}
        )
    contradiction_fraction = (
        float(adjusted_contradictory_count / len(in_range_rows))
        if in_range_rows
        else None
    )
    contradiction_run = longest_consecutive_conflict_rows(adjusted_run_rows)
    gate_available = len(in_range_rows) >= policy.minimum_in_range_clean_rows
    persistent_conflict = bool(
        gate_available
        and contradiction_fraction is not None
        and contradiction_fraction
        >= policy.persistent_contradiction_minimum_fraction
        and contradiction_run
        >= policy.persistent_contradiction_minimum_run_rows
    )

    if construction == "not_constructible":
        failures.append("missing_frozen_v6_support_or_fit")
    else:
        check(
            "preexisting_hard_veto",
            not snapshot.hard_veto_applied,
            snapshot.hard_veto_applied,
            "frozen upstream hard veto must be absent",
        )
        direction_crossings = {
            name: bool(
                (diagnostic.get(name) or {}).get("h2", {}).get("crossing")
            )
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
                and extension_p10
                >= policy.minimum_extension_row_fill_p10,
                extension_p10,
                "extension-only row-fill p10 after exact occluder exclusion "
                f">= {policy.minimum_extension_row_fill_p10:.2f}",
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
                "subtract same-side occluder pixels only; then fail when clean "
                f"rows >= {policy.minimum_in_range_clean_rows}, contradiction "
                "fraction >= "
                f"{policy.persistent_contradiction_minimum_fraction:.2f}, and "
                "run >= "
                f"{policy.persistent_contradiction_minimum_run_rows}"
            ),
        )

    failures = list(dict.fromkeys(failures))
    if construction == "no_op" and not failures:
        status, selected_model, selected = (
            "no_extension_needed",
            "observed_straight",
            observed.copy(),
        )
    elif construction == "proposal_generated" and not failures:
        status, selected_model, selected = (
            "selected",
            "extended_straight_v4b",
            proposal.copy(),
        )
    else:
        status, selected_model, selected = "failed", None, empty.copy()

    outside_clean_rows = int(
        diagnostic.get("outside_seed_observation_consistency", {}).get(
            "clean_row_count"
        )
        or 0
    )
    enough_original_evidence = bool(
        snapshot.support_y_range is not None
        and len(snapshot.selected_band_centers) >= 3
        and outside_clean_rows >= policy.s2_minimum_outside_clean_rows
    )
    s2_eligible = bool(
        status == "failed"
        and enough_original_evidence
        and not snapshot.hard_veto_applied
    )
    if status != "failed":
        handoff_reason = None
    elif s2_eligible:
        handoff_reason = "s1_failed_with_sufficient_original_v6_observations"
    elif snapshot.hard_veto_applied:
        handoff_reason = "s1_failed_preexisting_hard_veto_s2_ineligible"
    else:
        handoff_reason = "s1_failed_insufficient_original_v6_observations"

    result = {
        "policy_id": POLICY_ID,
        "threshold_status": "provisional_full160_ablation",
        "s1_status": status,
        "s1_selected_model": selected_model,
        "s1_failure_reasons": failures,
        "s2_eligible": s2_eligible,
        "s2_handoff_reason": handoff_reason,
        "checks": checks,
        "effective_selected_area_px": int(selected.sum()),
        "candidate_area_px": int(proposal.sum()),
        "effective_selected_mask_sha256": hash_mask_after_uint8_cast(selected),
        "candidate_is_not_automatically_selected": True,
        "v6_direction_classification": snapshot.direction_classification,
        "v6_direction_classification_used_as_gate": False,
        "user_feedback_specific_rules_added": True,
        "observation_comparison": (
            "signed_containment_with_exact_same_side_occluder_neutrality"
        ),
        "v4b_change": (
            "v4a_plus_exact_occluder_neutrality_no_prefix_salvage"
        ),
        "v4b_row_fill_audit": row_fill_audit,
        "v4b_containment_audit": containment_audit,
        "v4b_neutral_structural_pixels": int(structural_neutral.sum()),
        "v4b_neutral_containment_pixels": int(containment_neutral.sum()),
        "policy": {
            "minimum_extension_row_fill_p10": (
                policy.minimum_extension_row_fill_p10
            ),
            "minimum_in_range_clean_rows": policy.minimum_in_range_clean_rows,
            "persistent_contradiction_minimum_fraction": (
                policy.persistent_contradiction_minimum_fraction
            ),
            "persistent_contradiction_minimum_run_rows": (
                policy.persistent_contradiction_minimum_run_rows
            ),
            "s2_minimum_outside_clean_rows": (
                policy.s2_minimum_outside_clean_rows
            ),
        },
    }
    return result, {
        "effective_selected": selected,
        "occlusion_neutral_structural": structural_neutral,
        "occlusion_neutral_containment": containment_neutral,
    }
