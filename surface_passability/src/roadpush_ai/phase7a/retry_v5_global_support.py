from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import roadpush_ai.phase7a.retry_v4 as v4


PROBE_POLICY_ID = "phase7_v5_global_longest_support_a_probe"
POLICY_ID = "phase7_trapezoid_boundary_retry_v5"


@dataclass(frozen=True)
class GlobalSupportPolicy(v4.TrapezoidBoundaryRetryV4Policy):
    """V4 policy plus the agreed three-band global-candidate gate."""

    minimum_global_candidate_bands: int = 3


TrapezoidBoundaryRetryV5Policy = GlobalSupportPolicy


def _v4_bands_and_runs(
    records: list[dict[str, Any]], policy: GlobalSupportPolicy
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    """Generate exactly the same fixed seven-row bands used by V4."""

    if not records:
        return [], []
    rows_by_y = {row["y"]: row for row in records}
    low, high = min(rows_by_y), max(rows_by_y)
    radius = policy.band_rows // 2
    bands = [
        band
        for center in range(low + radius, high - radius + 1, policy.band_rows)
        if (band := v4._band(rows_by_y, center, policy)) is not None
    ]
    return bands, v4._clean_runs(bands, policy)


def _candidate_score(
    window: list[dict[str, Any]],
    left: dict[str, Any],
    right: dict[str, Any],
) -> tuple[float, ...]:
    """Rank evidence amount first and existing fit quality only for ties."""

    passing_y = {y for band in window for y in band["passing_y"]}
    p90_sum = float(left["p90_residual_px"] + right["p90_residual_px"])
    median_sum = float(
        left["median_residual_px"] + right["median_residual_px"]
    )
    return (
        float(len(window)),
        float(len(passing_y)),
        float(-p90_sum),
        float(-median_sum),
    )


def _raw_row_fit_passes(
    window: list[dict[str, Any]],
    records_by_y: dict[int, dict[str, Any]],
    width: int,
    policy: GlobalSupportPolicy,
) -> bool:
    """Apply the unchanged V4 fit gate to every clean row in a band window."""

    passing_y = sorted({y for band in window for y in band["passing_y"]})
    anchors = [
        records_by_y[y]
        for y in passing_y
        if y in records_by_y and records_by_y[y]["clean_bilateral"]
    ]
    if len(anchors) < policy.minimum_anchor_rows:
        return False
    ys = np.asarray([row["y"] for row in anchors], dtype=float)
    tolerance = policy.fit_inlier_tolerance_fraction * width
    left = v4._robust_line(
        ys, np.asarray([row["left"] for row in anchors]), tolerance
    )
    right = v4._robust_line(
        ys, np.asarray([row["right"] for row in anchors]), tolerance
    )
    return v4._fit_passes(left, right, width, policy)


def _global_longest_support(
    records: list[dict[str, Any]],
    shape: tuple[int, int],
    policy: GlobalSupportPolicy,
):
    """Globally select the longest valid support over observed V4 bands.

    Candidate windows are contiguous in observed-band order rather than image
    rows.  Missing/contact bands can therefore remain diagnostic gaps, as in
    V4, while every observed band between a window's endpoints participates in
    the same V4 band-representative fit.  The only selection changes are the
    three-band minimum, global search, and longest-support ranking.  There is
    deliberately no bottom-up fallback, route prior, extrapolation, direction
    consistency gate, or continuous-core preference in probe A.
    """

    if not records:
        return None, [], []
    _, width = shape
    all_bands, runs = _v4_bands_and_runs(records, policy)
    candidates: list[dict[str, Any]] = []
    examined = 0
    for start in range(len(all_bands)):
        for stop in range(
            start + policy.minimum_global_candidate_bands,
            len(all_bands) + 1,
        ):
            window = all_bands[start:stop]
            examined += 1
            left, right = v4._fit_bands(window, width, policy)
            if not v4._fit_passes(left, right, width, policy):
                continue
            candidates.append(
                {
                    "bands": window,
                    "left": left,
                    "right": right,
                    "score": _candidate_score(window, left, right),
                }
            )

    if not candidates:
        return None, all_bands, runs

    records_by_y = {row["y"]: row for row in records}
    ranked = sorted(candidates, key=lambda candidate: candidate["score"], reverse=True)
    best = None
    raw_rejected = 0
    for candidate in ranked:
        if _raw_row_fit_passes(candidate["bands"], records_by_y, width, policy):
            best = candidate
            break
        raw_rejected += 1
    if best is None:
        return None, all_bands, runs

    selected = best["bands"]
    passing_y = sorted({y for band in selected for y in band["passing_y"]})
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


def _fit_global_support(*args: Any, policy_id: str, **kwargs: Any):
    policy = kwargs.pop("policy", None) or GlobalSupportPolicy()
    result, masks = v4.fit_trapezoid_boundary_retry_v4(
        *args,
        policy=policy,
        _support_selector=_global_longest_support,
        **kwargs,
    )

    result["policy_id"] = policy_id
    fit = result.get("envelope_fit", {})
    fit["search_direction"] = "all_v4_bands_global_longest_support_a"

    if args and len(args) >= 4 and args[2] is not None and args[3] is not None:
        structural = np.asarray(args[2], dtype=bool)
        blocker = np.asarray(args[3], dtype=bool)
        silhouette = np.asarray(
            kwargs.get("obstacle_silhouette_mask", np.zeros_like(structural)),
            dtype=bool,
        )
        support, _, _ = _global_longest_support(
            v4._records(structural, blocker, silhouette, policy),
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
    return result, masks


def fit_global_longest_support_a_probe(*args: Any, **kwargs: Any):
    """Historical probe entry point retained for reproducibility."""

    result, masks = _fit_global_support(
        *args, policy_id=PROBE_POLICY_ID, **kwargs
    )
    result["ablation"] = dict(result["v5_contract"])
    return result, masks


def fit_trapezoid_boundary_retry_v5(*args: Any, **kwargs: Any):
    """Authoritative Phase 7 V5 boundary baseline accepted on 2026-10-22.

    V5 changes candidate selection only: it globally ranks V4 straight-support
    windows, requires at least three observed bands, retries ranked candidates
    through the unchanged V4 raw-row gate, and has no bottom-up fallback.
    """

    result, masks = _fit_global_support(*args, policy_id=POLICY_ID, **kwargs)
    result["v5_release"] = {
        "status": "authoritative_phase7_candidate",
        "promoted_from": PROBE_POLICY_ID,
        "promotion_date": "2026-10-22",
        "phase7_complete": False,
        "production_integration_authorized": False,
    }
    return result, masks
