"""Global ranking of straight-boundary support windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from surface_passability.boundary.line_fitting import fit_robust_boundary_line
from surface_passability.boundary.observations import collect_straight_boundary_rows
from surface_passability.boundary.straight import fit_straight_boundary_candidate
from surface_passability.boundary.support import (
    boundary_fits_pass,
    build_boundary_band,
    fit_boundary_bands,
    group_clean_boundary_runs,
)
from surface_passability.policies.straight_boundary import StraightBoundaryPolicy


BASELINE_POLICY_ID = "phase7_trapezoid_boundary_retry_v5"
HISTORICAL_PROBE_POLICY_ID = "phase7_v5_global_longest_support_a_probe"


@dataclass(frozen=True)
class GlobalBoundarySupportPolicy(StraightBoundaryPolicy):
    """Require at least three bands for a global support candidate."""

    minimum_global_candidate_bands: int = 3


def collect_global_support_bands(
    records: list[dict[str, Any]],
    policy: GlobalBoundarySupportPolicy,
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    """Generate the fixed bands and clean runs consumed by global ranking."""

    if not records:
        return [], []
    rows_by_y = {row["y"]: row for row in records}
    lowest, highest = min(rows_by_y), max(rows_by_y)
    radius = policy.band_rows // 2
    bands = [
        band
        for center_y in range(
            lowest + radius,
            highest - radius + 1,
            policy.band_rows,
        )
        if (
            band := build_boundary_band(rows_by_y, center_y, policy)
        )
        is not None
    ]
    return bands, group_clean_boundary_runs(bands, policy)


def score_global_support_candidate(
    window: list[dict[str, Any]],
    left_fit: dict[str, Any],
    right_fit: dict[str, Any],
) -> tuple[float, ...]:
    """Rank support amount before residual quality."""

    passing_y = {row_y for band in window for row_y in band["passing_y"]}
    p90_sum = float(
        left_fit["p90_residual_px"] + right_fit["p90_residual_px"]
    )
    median_sum = float(
        left_fit["median_residual_px"] + right_fit["median_residual_px"]
    )
    return (
        float(len(window)),
        float(len(passing_y)),
        float(-p90_sum),
        float(-median_sum),
    )


def raw_boundary_rows_fit(
    window: list[dict[str, Any]],
    records_by_y: dict[int, dict[str, Any]],
    image_width: int,
    policy: GlobalBoundarySupportPolicy,
) -> bool:
    """Validate a ranked band window against every clean raw row."""

    passing_y = sorted(
        {row_y for band in window for row_y in band["passing_y"]}
    )
    anchors = [
        records_by_y[row_y]
        for row_y in passing_y
        if row_y in records_by_y and records_by_y[row_y]["clean_bilateral"]
    ]
    if len(anchors) < policy.minimum_anchor_rows:
        return False
    rows = np.asarray([row["y"] for row in anchors], dtype=float)
    tolerance = policy.fit_inlier_tolerance_fraction * image_width
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
    return boundary_fits_pass(
        left_fit,
        right_fit,
        image_width,
        policy,
    )


def select_global_boundary_support(
    records: list[dict[str, Any]],
    shape: tuple[int, int],
    policy: GlobalBoundarySupportPolicy,
) -> tuple[
    dict[str, Any] | None,
    list[dict[str, Any]],
    list[list[dict[str, Any]]],
]:
    """Choose the longest globally valid straight-support window."""

    if not records:
        return None, [], []
    _, width = shape
    all_bands, runs = collect_global_support_bands(records, policy)
    candidates: list[dict[str, Any]] = []
    examined = 0
    for start in range(len(all_bands)):
        for stop in range(
            start + policy.minimum_global_candidate_bands,
            len(all_bands) + 1,
        ):
            window = all_bands[start:stop]
            examined += 1
            left_fit, right_fit = fit_boundary_bands(window, width, policy)
            if not boundary_fits_pass(
                left_fit,
                right_fit,
                width,
                policy,
            ):
                continue
            candidates.append(
                {
                    "bands": window,
                    "left": left_fit,
                    "right": right_fit,
                    "score": score_global_support_candidate(
                        window,
                        left_fit,
                        right_fit,
                    ),
                }
            )

    if not candidates:
        return None, all_bands, runs

    records_by_y = {row["y"]: row for row in records}
    ranked = sorted(
        candidates,
        key=lambda candidate: candidate["score"],
        reverse=True,
    )
    best = None
    raw_rejected = 0
    for candidate in ranked:
        if raw_boundary_rows_fit(
            candidate["bands"],
            records_by_y,
            width,
            policy,
        ):
            best = candidate
            break
        raw_rejected += 1
    if best is None:
        return None, all_bands, runs

    selected = best["bands"]
    passing_y = sorted(
        {row_y for band in selected for row_y in band["passing_y"]}
    )
    median_span = float(np.median([band["span"] for band in selected]))
    centers = [int(band["center_y"]) for band in selected]
    gap_count = sum(
        max(
            0,
            (right_band["center_y"] - left_band["center_y"])
            // policy.band_rows
            - 1,
        )
        for left_band, right_band in zip(selected, selected[1:])
    )
    return (
        {
            "bands": selected,
            "passing_y": passing_y,
            "median_span": median_span,
            "median_span_fraction": float(median_span / max(1, width)),
            "segment_span_gate_passed": (
                median_span / max(1, width)
                >= policy.minimum_segment_median_span_fraction
            ),
            "near_cap_centers": centers[-policy.minimum_clean_cap_bands :],
            "far_cap_center": centers[0],
            "internal_gap_band_count": int(gap_count),
            "extension_stop_reason": "global_longest_v4_support_selected",
            "selection_mode": "global_longest_v4_band_support_a",
            "global_candidates_examined": examined,
            "global_candidates_passing": len(candidates),
            "ranked_candidates_raw_rejected": raw_rejected,
            "selected_window_score": [float(value) for value in best["score"]],
        },
        all_bands,
        runs,
    )


def fit_global_boundary_support(
    *args: Any,
    policy: GlobalBoundarySupportPolicy | None = None,
    **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run the accepted global-support candidate over the base calculation."""

    policy = policy or GlobalBoundarySupportPolicy()
    result, masks = fit_straight_boundary_candidate(
        *args,
        policy=policy,
        support_selector=select_global_boundary_support,
        **kwargs,
    )

    result["policy_id"] = BASELINE_POLICY_ID
    fit = result.get("envelope_fit", {})
    fit["search_direction"] = "all_v4_bands_global_longest_support_a"

    if args and len(args) >= 4 and args[2] is not None and args[3] is not None:
        structural = np.asarray(args[2], dtype=bool)
        blocker = np.asarray(args[3], dtype=bool)
        silhouette = np.asarray(
            kwargs.get("obstacle_silhouette_mask", np.zeros_like(structural)),
            dtype=bool,
        )
        support, _, _ = select_global_boundary_support(
            collect_straight_boundary_rows(
                structural,
                blocker,
                silhouette,
                policy,
            ),
            structural.shape,
            policy,
        )
        if support:
            fit.update(
                {
                    key: support.get(key)
                    for key in (
                        "selection_mode",
                        "global_candidates_examined",
                        "global_candidates_passing",
                        "ranked_candidates_raw_rejected",
                        "selected_window_score",
                    )
                }
            )

    result["v5_contract"] = {
        "kind": "v4_global_longest_support_a",
        "historical_best_window_imported": False,
        "v4_band_generation_unchanged": True,
        "v4_candidate_fit_uses_band_representatives": True,
        "ranked_candidate_raw_row_validation": True,
        "v4_fit_thresholds_unchanged": True,
        "minimum_global_candidate_bands": policy.minimum_global_candidate_bands,
        "cross_gap_windows_allowed": True,
        "bottom_up_fallback_used": False,
        "ranking_order": [
            "observed_band_count",
            "unique_passing_row_count",
            "combined_p90_residual_ascending",
            "combined_median_residual_ascending",
        ],
        "upper_reach_tiebreak_used": False,
        "direction_consistency_criterion_added": False,
        "continuous_core_priority_added": False,
        "route_prior_used": False,
        "extension_added": False,
        "blocker_logic_changed": False,
    }
    result["v5_release"] = {
        "status": "authoritative_phase7_candidate",
        "promoted_from": HISTORICAL_PROBE_POLICY_ID,
        "promotion_date": "2026-10-22",
        "phase7_complete": False,
        "production_integration_authorized": False,
    }
    return result, masks
