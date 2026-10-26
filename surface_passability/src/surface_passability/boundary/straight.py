"""Base straight-boundary candidate calculation."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import cv2
import numpy as np

from surface_passability.boundary.envelope import build_boundary_envelope
from surface_passability.boundary.line_fitting import fit_robust_boundary_line
from surface_passability.boundary.measurement import measure_boundary_corridors
from surface_passability.boundary.observations import (
    collect_straight_boundary_rows,
    compare_raw_and_pruned_structure,
)
from surface_passability.boundary.support import (
    select_bottom_up_boundary_support,
)
from surface_passability.io.hashing import hash_binary_mask
from surface_passability.policies.straight_boundary import StraightBoundaryPolicy


def fit_straight_boundary_candidate(
    selected_surface: str | None,
    selection_state: str,
    structural_mask: np.ndarray | None,
    blocker_mask: np.ndarray | None,
    *,
    obstacle_silhouette_mask: np.ndarray | None = None,
    raw_structural_mask: np.ndarray | None = None,
    policy: StraightBoundaryPolicy | None = None,
    support_selector: Any | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Build the accepted base straight-boundary candidate and diagnostics."""

    policy = policy or StraightBoundaryPolicy()
    if (
        selected_surface is None
        or structural_mask is None
        or blocker_mask is None
        or selection_state == "uncertain"
    ):
        return {
            "policy_id": "phase7_trapezoid_boundary_retry_v4",
            "selected_surface": selected_surface,
            "selection_state": selection_state,
            "minimum_passable_ratio": None,
            "maximum_obstructed_width_ratio": None,
            "obstacle_area_ratio": None,
            "passability_state": "uncertain",
            "representative_measurement_allowed": False,
            "envelope_fit": {
                "status": "not_applicable",
                "confidence": "insufficient",
                "rejection_reasons": ["selector_uncertain"],
            },
            "blocker_reuse": {"modified": False, "warnings": []},
            "corridor_measurement": None,
            "raw_pruned_comparison": None,
        }, {}

    structural = np.asarray(structural_mask, dtype=bool)
    blocker = np.asarray(blocker_mask, dtype=bool)
    silhouette = (
        np.asarray(obstacle_silhouette_mask, dtype=bool)
        if obstacle_silhouette_mask is not None
        else np.zeros_like(structural)
    )
    if (
        structural.ndim != 2
        or structural.shape != blocker.shape
        or structural.shape != silhouette.shape
    ):
        raise ValueError(
            "structural, blocker and silhouette masks must be same-shape 2D arrays"
        )

    height, width = structural.shape
    records = collect_straight_boundary_rows(
        structural,
        blocker,
        silhouette,
        policy,
    )
    selected_support = support_selector or select_bottom_up_boundary_support
    support, all_bands, runs = selected_support(
        records,
        structural.shape,
        policy,
    )
    reasons: list[str] = []
    if not np.any(structural):
        reasons.append("no_authoritative_structural_pixels")
    if support is None:
        reasons.append("no_bilateral_clean_caps")

    anchors: list[dict[str, Any]] = []
    row_range: list[int] | None = None
    if support is not None:
        passing = set(support["passing_y"])
        anchors = [
            row
            for row in records
            if row["y"] in passing and row["clean_bilateral"]
        ]
        if anchors:
            row_range = [
                min(row["y"] for row in anchors),
                max(row["y"] for row in anchors),
            ]
        if not support["segment_span_gate_passed"]:
            reasons.append("insufficient_segment_median_span")
    if len(anchors) < policy.minimum_anchor_rows:
        reasons.append("insufficient_anchor_rows")

    left_fit: dict[str, Any] = {"valid": False, "reason": "not_fitted"}
    right_fit: dict[str, Any] = {"valid": False, "reason": "not_fitted"}
    envelope = np.zeros_like(structural)
    evidence: list[float] = []
    widths: list[float] = []
    if len(anchors) >= 2 and row_range:
        rows = np.asarray([row["y"] for row in anchors], dtype=float)
        tolerance = policy.fit_inlier_tolerance_fraction * width
        left_fit = fit_robust_boundary_line(
            rows,
            np.asarray([row["left"] for row in anchors]),
            tolerance,
        )
        right_fit = fit_robust_boundary_line(
            rows,
            np.asarray([row["right"] for row in anchors]),
            tolerance,
        )
        for side, fit in (("left", left_fit), ("right", right_fit)):
            if not fit.get("valid"):
                reasons.append(f"{side}_fit_failed")
                continue
            if fit["inlier_fraction"] < policy.minimum_inlier_fraction:
                reasons.append(f"{side}_low_inlier_fraction")
            if (
                fit["median_residual_px"]
                > policy.maximum_median_residual_fraction * width
            ):
                reasons.append(f"{side}_high_median_residual")
            if (
                fit["p90_residual_px"]
                > policy.maximum_p90_residual_fraction * width
            ):
                reasons.append(f"{side}_high_p90_residual")
        if left_fit.get("valid") and right_fit.get("valid"):
            widths = [
                right_fit["slope"] * row_y
                + right_fit["intercept"]
                - (left_fit["slope"] * row_y + left_fit["intercept"])
                + 1
                for row_y in range(row_range[0], row_range[1] + 1)
            ]
            if not widths or min(widths) <= 0:
                reasons.append("crossing_or_nonpositive_width")
            candidate = build_boundary_envelope(
                structural.shape,
                left_fit,
                right_fit,
                row_range[0],
                row_range[1],
            )
            for row_y in range(row_range[0], row_range[1] + 1):
                denominator = int(np.count_nonzero(candidate[row_y]))
                if denominator:
                    evidence.append(
                        float(
                            np.count_nonzero(
                                (structural[row_y] | blocker[row_y])
                                & candidate[row_y]
                            )
                            / denominator
                        )
                    )
            if (
                not evidence
                or float(np.median(evidence))
                < policy.minimum_median_envelope_evidence
            ):
                reasons.append("insufficient_envelope_evidence")
            if not reasons:
                envelope = candidate

    reasons = list(dict.fromkeys(reasons))
    accepted = not reasons and np.any(envelope)
    reused_blocker = blocker.copy()
    measurement = None
    free = np.zeros_like(structural)
    warnings: list[str] = []
    added_fraction = None
    if accepted:
        measurement, free = measure_boundary_corridors(
            envelope,
            reused_blocker,
            policy,
        )
        added_fraction = float(
            np.count_nonzero(envelope & ~structural)
            / max(1, np.count_nonzero(envelope))
        )
        if (
            np.any(reused_blocker & envelope)
            and added_fraction >= policy.blocker_suspect_added_area_fraction
        ):
            warnings.append("blocker_coverage_suspect")
    representative = accepted and not warnings and measurement is not None
    raw_comparison, raw_removed = compare_raw_and_pruned_structure(
        raw_structural_mask,
        structural,
    )

    component_rows = np.flatnonzero(np.any(structural, axis=1))
    component_range = (
        [int(component_rows[0]), int(component_rows[-1])]
        if component_rows.size
        else None
    )
    discarded_lower = (
        component_range[1] - row_range[1]
        if component_range is not None and row_range
        else None
    )
    discarded_upper = (
        row_range[0] - component_range[0]
        if component_range is not None and row_range
        else None
    )
    vertical_support = (
        (row_range[1] - row_range[0] + 1)
        / max(1, component_range[1] - component_range[0] + 1)
        if component_range is not None and row_range
        else 0.0
    )
    top_bottom_ratio = (
        float(widths[0] / widths[-1])
        if widths and widths[-1] > 0
        else None
    )
    intersection_y = None
    if left_fit.get("valid") and right_fit.get("valid"):
        slope_delta = left_fit["slope"] - right_fit["slope"]
        if abs(slope_delta) > 1e-9:
            intersection_y = float(
                (right_fit["intercept"] - left_fit["intercept"])
                / slope_delta
            )

    input_hash = hash_binary_mask(blocker)
    output_hash = hash_binary_mask(reused_blocker)
    result = {
        "policy_id": "phase7_trapezoid_boundary_retry_v4",
        "policy": asdict(policy),
        "selected_surface": selected_surface,
        "selection_state": selection_state,
        "minimum_passable_ratio": (
            measurement["minimum_passable_ratio"] if representative else None
        ),
        "maximum_obstructed_width_ratio": (
            measurement["maximum_obstructed_width_ratio"]
            if representative
            else None
        ),
        "obstacle_area_ratio": (
            measurement["obstacle_area_ratio"] if representative else None
        ),
        "passability_state": (
            "relative_only"
            if representative and measurement["status"] == "measured"
            else "blocked"
            if representative and measurement["status"] == "blocked"
            else "uncertain"
        ),
        "representative_measurement_allowed": representative,
        "envelope_fit": {
            "status": "accepted" if accepted else "rejected",
            "confidence": "reliable" if accepted else "insufficient",
            "evidence_source": (
                "phase6_structural_blocker_plus_cached_semantic_obstacle_silhouette"
            ),
            "component_reselection_performed": False,
            "search_direction": "bottom_up",
            "outer_constraint_applied": False,
            "vegetation_contact_gate_applied": False,
            "eligible_row_count": len(records),
            "anchor_rows": len(anchors),
            "anchor_y": [row["y"] for row in anchors],
            "all_band_centers": [band["center_y"] for band in all_bands],
            "support_run_count": len(runs),
            "selected_band_centers": (
                [band["center_y"] for band in support["bands"]]
                if support
                else []
            ),
            "near_cap_centers": support["near_cap_centers"] if support else [],
            "far_cap_center": support["far_cap_center"] if support else None,
            "internal_gap_band_count": (
                support["internal_gap_band_count"] if support else None
            ),
            "extension_stop_reason": (
                support["extension_stop_reason"] if support else None
            ),
            "component_y_range": component_range,
            "support_y_range": row_range,
            "discarded_lower_prefix_rows": discarded_lower,
            "discarded_upper_prefix_rows": discarded_upper,
            "vertical_support_fraction": float(vertical_support),
            "segment_median_span_fraction": (
                support["median_span_fraction"] if support else None
            ),
            "left_fit": left_fit,
            "right_fit": right_fit,
            "top_bottom_width_ratio": top_bottom_ratio,
            "fitted_line_intersection_y": intersection_y,
            "median_envelope_evidence_ratio": (
                float(np.median(evidence)) if evidence else None
            ),
            "rejection_reasons": reasons,
        },
        "row_evidence": records,
        "blocker_reuse": {
            "source": "phase6_selected_surface_owned_blocker",
            "modified": False,
            "input_binary_sha256": input_hash,
            "output_binary_sha256": output_hash,
            "hash_match": input_hash == output_hash,
            "warnings": warnings,
        },
        "obstacle_silhouette": {
            "source": "cached_semantic_map_occupancy_and_hazard_classes",
            "vegetation_included": False,
            "used_for_boundary_contact_only": True,
            "pixels": int(np.count_nonzero(silhouette)),
        },
        "corridor_measurement": measurement,
        "raw_pruned_comparison": raw_comparison,
        "diagnostics": {
            "authoritative_structural_pixels": int(np.count_nonzero(structural)),
            "connected_component_count": int(
                cv2.connectedComponents(structural.astype(np.uint8), 8)[0] - 1
            ),
            "component_reselection_performed": False,
            "envelope_pixels": int(np.count_nonzero(envelope)),
            "added_structural_pixels": int(
                np.count_nonzero(envelope & ~structural)
            ),
            "omitted_structural_pixels": int(
                np.count_nonzero(structural & ~envelope)
            ),
            "added_structural_fraction": added_fraction,
            "semantic_inference_count": 0,
            "context_build_count": 0,
            "geometry_extraction_count": 0,
            "selector_call_count": 0,
        },
    }
    masks = {
        "authoritative_structural": structural,
        "obstacle_silhouette": silhouette,
        "envelope": envelope,
        "blocker": reused_blocker,
        "free": free,
        "added": envelope & ~structural,
        "omitted": structural & ~envelope,
        "raw_structural": (
            np.asarray(raw_structural_mask, dtype=bool)
            if raw_structural_mask is not None
            else np.zeros_like(structural)
        ),
        "raw_pruned_removed": raw_removed,
    }
    return result, masks
