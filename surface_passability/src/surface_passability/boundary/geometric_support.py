"""Straight-boundary support selection with explained geometric gaps."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from surface_passability.boundary.global_support import (
    collect_global_support_bands,
    fit_global_boundary_support,
    raw_boundary_rows_fit,
    score_global_support_candidate,
)
from surface_passability.boundary.observations import (
    collect_straight_boundary_rows,
)
from surface_passability.boundary.occlusion import diagnose_geometric_gaps
from surface_passability.boundary.straight import fit_straight_boundary_candidate
from surface_passability.boundary.support import (
    boundary_fits_pass,
    fit_boundary_bands,
)
from surface_passability.policies.geometric_gap import GeometricGapPolicy


POLICY_ID = "phase7_trapezoid_boundary_retry_v5_probe_g_geometric_gap"


def select_support_with_geometric_gaps(
    records: list[dict[str, Any]],
    shape: tuple[int, int],
    policy: GeometricGapPolicy,
) -> tuple[
    dict[str, Any] | None,
    list[dict[str, Any]],
    list[list[dict[str, Any]]],
]:
    """Select global support after excluding safely explained gap bands."""

    if not records:
        return None, [], []
    _, width = shape
    all_bands, runs = collect_global_support_bands(records, policy)
    gap_diagnostics = diagnose_geometric_gaps(all_bands, width, policy)
    safe_centers = set(gap_diagnostics["safe_gap_band_centers"])
    candidates: list[dict[str, Any]] = []
    examined = 0
    for start in range(len(all_bands)):
        for stop in range(
            start + policy.minimum_global_candidate_bands,
            len(all_bands) + 1,
        ):
            observed_window = all_bands[start:stop]
            anchor_window = [
                band
                for band in observed_window
                if int(band["center_y"]) not in safe_centers
            ]
            if len(anchor_window) < policy.minimum_global_candidate_bands:
                continue
            examined += 1
            left_fit, right_fit = fit_boundary_bands(
                anchor_window,
                width,
                policy,
            )
            if not boundary_fits_pass(
                left_fit,
                right_fit,
                width,
                policy,
            ):
                continue
            candidates.append(
                {
                    "bands": anchor_window,
                    "observed_window": observed_window,
                    "left": left_fit,
                    "right": right_fit,
                    "score": score_global_support_candidate(
                        anchor_window,
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
    observed_window = best["observed_window"]
    passing_y = sorted(
        {row_y for band in selected for row_y in band["passing_y"]}
    )
    median_span = float(np.median([band["span"] for band in selected]))
    centers = [int(band["center_y"]) for band in selected]
    observed_centers = [
        int(band["center_y"]) for band in observed_window
    ]
    demoted_in_window = sorted(set(observed_centers) & safe_centers)
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
            "extension_stop_reason": (
                "global_v5_support_after_geometric_gap_demotion"
            ),
            "selection_mode": "v5_global_support_after_probe_g",
            "global_candidates_examined": examined,
            "global_candidates_passing": len(candidates),
            "ranked_candidates_raw_rejected": raw_rejected,
            "selected_window_score": [
                float(value) for value in best["score"]
            ],
            "probe_g": {
                **gap_diagnostics,
                "selected_observed_window_centers": observed_centers,
                "selected_geometric_gap_band_centers": demoted_in_window,
            },
        },
        all_bands,
        runs,
    )


def fit_boundary_with_geometric_gaps(
    *args: Any,
    mode: Literal["off", "candidate"] = "candidate",
    policy: GeometricGapPolicy | None = None,
    **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Apply conservative geometric-gap demotion to global support."""

    if mode == "off":
        return fit_global_boundary_support(*args, **kwargs)
    if mode != "candidate":
        raise ValueError(f"Unsupported Probe G mode: {mode}")
    policy = policy or GeometricGapPolicy()
    result, masks = fit_straight_boundary_candidate(
        *args,
        policy=policy,
        support_selector=select_support_with_geometric_gaps,
        **kwargs,
    )
    result["policy_id"] = POLICY_ID
    fit = result.get("envelope_fit", {})
    fit["search_direction"] = "v5_global_support_after_probe_g"

    gap_result = None
    if args and len(args) >= 4 and args[2] is not None and args[3] is not None:
        structural = np.asarray(args[2], dtype=bool)
        blocker = np.asarray(args[3], dtype=bool)
        silhouette = np.asarray(
            kwargs.get("obstacle_silhouette_mask", np.zeros_like(structural)),
            dtype=bool,
        )
        support, _, _ = select_support_with_geometric_gaps(
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
            gap_result = support.get("probe_g")
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
    result["probe_g_geometric_gap"] = gap_result or {
        "status": "not_applicable_or_no_selected_support",
        "candidate_groups": [],
        "safe_gap_band_centers": [],
        "selected_geometric_gap_band_centers": [],
    }
    result["probe_g_contract"] = {
        "mode": mode,
        "v5_band_generation_unchanged": True,
        "v5_ranking_unchanged": True,
        "v5_residual_thresholds_unchanged": True,
        "one_sided_deviation_required": True,
        "opposite_boundary_stability_required": True,
        "interior_pre_post_line_agreement_required": True,
        "terminal_severe_switch_requires_retained_core": True,
        "demoted_bands_are_non_anchor_gaps": True,
        "ranked_candidate_raw_row_validation": True,
        "blocker_logic_changed": False,
        "direction_gate_added": False,
        "continuous_core_ranking_added": False,
        "vegetation_gate_added": False,
        "extension_or_curve_added": False,
        "threshold_status": "provisional_targeted_review_candidate",
    }
    result["v5_contract"] = {
        "kind": "v5_plus_isolated_probe_g",
        "minimum_global_candidate_bands": policy.minimum_global_candidate_bands,
        "ranking_order": [
            "retained_observed_band_count",
            "unique_passing_row_count",
            "combined_p90_residual_ascending",
            "combined_median_residual_ascending",
        ],
        "bottom_up_fallback_used": False,
        "blocker_logic_changed": False,
    }
    return result, masks
