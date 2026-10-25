from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Literal

import numpy as np

import roadpush_ai.phase7a.retry_v4 as v4
from roadpush_ai.phase7a.retry_v5 import (
    fit_trapezoid_boundary_retry_v5,
)
from roadpush_ai.phase7a.retry_v5_global_support import (
    GlobalSupportPolicy,
    _candidate_score,
    _raw_row_fit_passes,
    _v4_bands_and_runs,
)


POLICY_ID = "phase7_trapezoid_boundary_retry_v5_probe_g_geometric_gap"


@dataclass(frozen=True)
class GeometricGapPolicy(GlobalSupportPolicy):
    """Provisional safety gates for one-sided geometric-gap demotion."""

    deviation_tolerance_fraction: float = 0.02
    opposite_stability_tolerance_fraction: float = 0.02
    terminal_severe_deviation_multiplier: float = 4.0
    minimum_interior_context_bands_per_side: int = 3
    preferred_context_bands_per_side: int = 5
    minimum_terminal_core_bands: int = 5
    maximum_context_angle_delta_degrees: float = 12.0
    maximum_context_prediction_delta_fraction: float = 0.03


def _fit_side(
    bands: list[dict[str, Any]], side: Literal["left", "right"]
) -> dict[str, float] | None:
    if len(bands) < 2:
        return None
    ys = np.asarray([band["center_y"] for band in bands], dtype=float)
    xs = np.asarray([band[side] for band in bands], dtype=float)
    slope, intercept = np.polyfit(ys, xs, 1)
    return {"slope": float(slope), "intercept": float(intercept)}


def _context_agrees(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    midpoint_y: float,
    width: int,
    policy: GeometricGapPolicy,
) -> tuple[bool, dict[str, Any]]:
    diagnostics: dict[str, Any] = {}
    passed = True
    for side in ("left", "right"):
        first = _fit_side(before, side)
        second = _fit_side(after, side)
        if first is None or second is None:
            return False, {"reason": "insufficient_context_fit"}
        first_angle = math.degrees(math.atan(first["slope"]))
        second_angle = math.degrees(math.atan(second["slope"]))
        angle_delta = float(abs(first_angle - second_angle))
        first_x = first["slope"] * midpoint_y + first["intercept"]
        second_x = second["slope"] * midpoint_y + second["intercept"]
        prediction_delta = float(abs(first_x - second_x))
        diagnostics[side] = {
            "angle_delta_degrees": angle_delta,
            "prediction_delta_px": prediction_delta,
        }
        if (
            angle_delta > policy.maximum_context_angle_delta_degrees
            or prediction_delta
            > policy.maximum_context_prediction_delta_fraction * width
        ):
            passed = False
    return passed, diagnostics


def _candidate_groups(
    candidates: list[dict[str, Any]], band_rows: int
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for candidate in candidates:
        if (
            not groups
            or candidate["side"] != groups[-1][-1]["side"]
            or candidate["center_y"] - groups[-1][-1]["center_y"] != band_rows
        ):
            groups.append([candidate])
        else:
            groups[-1].append(candidate)
    return groups


def diagnose_geometric_gaps(
    all_bands: list[dict[str, Any]],
    width: int,
    policy: GeometricGapPolicy,
) -> dict[str, Any]:
    if len(all_bands) < policy.minimum_global_candidate_bands:
        return {
            "status": "insufficient",
            "provisional_fit": None,
            "candidate_groups": [],
            "safe_gap_band_centers": [],
        }
    left, right = v4._fit_bands(all_bands, width, policy)
    if not left.get("valid") or not right.get("valid"):
        return {
            "status": "insufficient",
            "provisional_fit": None,
            "candidate_groups": [],
            "safe_gap_band_centers": [],
        }
    deviation_tolerance = policy.deviation_tolerance_fraction * width
    opposite_tolerance = policy.opposite_stability_tolerance_fraction * width
    candidates = []
    for index, band in enumerate(all_bands):
        y = float(band["center_y"])
        left_residual = float(
            band["left"] - (left["slope"] * y + left["intercept"])
        )
        right_residual = float(
            band["right"] - (right["slope"] * y + right["intercept"])
        )
        side = None
        signed_deviation = None
        opposite_residual = None
        if abs(left_residual) > deviation_tolerance and abs(right_residual) <= opposite_tolerance:
            side = "left"
            signed_deviation = left_residual
            opposite_residual = right_residual
        elif abs(right_residual) > deviation_tolerance and abs(left_residual) <= opposite_tolerance:
            side = "right"
            signed_deviation = right_residual
            opposite_residual = left_residual
        if side is not None:
            candidates.append(
                {
                    "band_index": index,
                    "center_y": int(band["center_y"]),
                    "side": side,
                    "signed_deviation_px": float(signed_deviation),
                    "absolute_deviation_px": float(abs(signed_deviation)),
                    "opposite_residual_px": float(opposite_residual),
                }
            )

    safe_centers: list[int] = []
    group_diagnostics = []
    groups = _candidate_groups(candidates, policy.band_rows)
    candidate_indices = {item["band_index"] for item in candidates}
    for group in groups:
        first_index = group[0]["band_index"]
        last_index = group[-1]["band_index"]
        before_count = first_index
        after_count = len(all_bands) - last_index - 1
        terminal = before_count == 0 or after_count == 0
        safe = False
        reason = None
        context: dict[str, Any] = {}
        if terminal:
            retained_count = max(before_count, after_count)
            severe = max(item["absolute_deviation_px"] for item in group)
            adjacent_to_core = (
                first_index > 0
                and group[0]["center_y"]
                - int(all_bands[first_index - 1]["center_y"])
                == policy.band_rows
            ) or (
                last_index + 1 < len(all_bands)
                and int(all_bands[last_index + 1]["center_y"])
                - group[-1]["center_y"]
                == policy.band_rows
            )
            safe = (
                retained_count >= policy.minimum_terminal_core_bands
                and adjacent_to_core
                and severe
                > policy.terminal_severe_deviation_multiplier
                * deviation_tolerance
            )
            reason = (
                "terminal_severe_one_sided_endpoint_switch"
                if safe
                else "terminal_support_or_deviation_gate_failed"
            )
            context = {
                "retained_core_band_count": retained_count,
                "maximum_deviation_px": severe,
                "adjacent_to_retained_core": adjacent_to_core,
            }
        else:
            required = policy.minimum_interior_context_bands_per_side
            trusted_before = [
                band
                for index, band in enumerate(all_bands[:first_index])
                if index not in candidate_indices
            ]
            trusted_after = [
                band
                for index, band in enumerate(
                    all_bands[last_index + 1 :], start=last_index + 1
                )
                if index not in candidate_indices
            ]
            adjacent_before = (
                bool(trusted_before)
                and group[0]["center_y"]
                - int(trusted_before[-1]["center_y"])
                == policy.band_rows
            )
            adjacent_after = (
                bool(trusted_after)
                and int(trusted_after[0]["center_y"])
                - group[-1]["center_y"]
                == policy.band_rows
            )
            if (
                len(trusted_before) < required
                or len(trusted_after) < required
                or not adjacent_before
                or not adjacent_after
            ):
                reason = "insufficient_pre_post_context"
            else:
                context_count = min(
                    policy.preferred_context_bands_per_side,
                    len(trusted_before),
                    len(trusted_after),
                )
                before = trusted_before[-context_count:]
                after = trusted_after[:context_count]
                midpoint = float(
                    (group[0]["center_y"] + group[-1]["center_y"]) / 2.0
                )
                safe, context = _context_agrees(
                    before, after, midpoint, width, policy
                )
                reason = (
                    "interior_one_sided_detour_with_compatible_context"
                    if safe
                    else "pre_post_line_disagreement"
                )
        if safe:
            safe_centers.extend(item["center_y"] for item in group)
        group_diagnostics.append(
            {
                "side": group[0]["side"],
                "band_centers": [item["center_y"] for item in group],
                "terminal": terminal,
                "safe_to_demote": safe,
                "reason": reason,
                "bands": group,
                "context": context,
            }
        )

    safe_interior_sides = {
        group["side"]
        for group in group_diagnostics
        if group["safe_to_demote"] and not group["terminal"]
    }
    for group in group_diagnostics:
        if (
            group["terminal"]
            and not group["safe_to_demote"]
            and group["side"] in safe_interior_sides
            and group["context"].get("adjacent_to_retained_core")
            and group["context"].get("retained_core_band_count", 0)
            >= policy.minimum_terminal_core_bands
        ):
            group["safe_to_demote"] = True
            group["reason"] = "terminal_same_side_as_safe_interior_detour"
            safe_centers.extend(group["band_centers"])

    return {
        "status": "safe_gaps_found" if safe_centers else "no_safe_gap",
        "provisional_fit": {"left": left, "right": right},
        "deviation_tolerance_px": deviation_tolerance,
        "opposite_stability_tolerance_px": opposite_tolerance,
        "candidate_groups": group_diagnostics,
        "safe_gap_band_centers": sorted(set(safe_centers)),
    }


def _global_support_with_geometric_gaps(
    records: list[dict[str, Any]],
    shape: tuple[int, int],
    policy: GeometricGapPolicy,
):
    if not records:
        return None, [], []
    _, width = shape
    all_bands, runs = _v4_bands_and_runs(records, policy)
    gap_diagnostics = diagnose_geometric_gaps(all_bands, width, policy)
    safe_centers = set(gap_diagnostics["safe_gap_band_centers"])
    candidates = []
    examined = 0
    for start in range(len(all_bands)):
        for stop in range(start + policy.minimum_global_candidate_bands, len(all_bands) + 1):
            observed_window = all_bands[start:stop]
            anchor_window = [
                band
                for band in observed_window
                if int(band["center_y"]) not in safe_centers
            ]
            if len(anchor_window) < policy.minimum_global_candidate_bands:
                continue
            examined += 1
            left, right = v4._fit_bands(anchor_window, width, policy)
            if not v4._fit_passes(left, right, width, policy):
                continue
            candidates.append(
                {
                    "bands": anchor_window,
                    "observed_window": observed_window,
                    "left": left,
                    "right": right,
                    "score": _candidate_score(anchor_window, left, right),
                }
            )
    if not candidates:
        return None, all_bands, runs
    records_by_y = {row["y"]: row for row in records}
    ranked = sorted(candidates, key=lambda item: item["score"], reverse=True)
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
    observed_window = best["observed_window"]
    passing_y = sorted({y for band in selected for y in band["passing_y"]})
    median_span = float(np.median([band["span"] for band in selected]))
    centers = [int(band["center_y"]) for band in selected]
    observed_centers = [int(band["center_y"]) for band in observed_window]
    demoted_in_window = sorted(set(observed_centers) & safe_centers)
    gap_count = sum(
        max(0, (right_band["center_y"] - left_band["center_y"]) // policy.band_rows - 1)
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
            "extension_stop_reason": "global_v5_support_after_geometric_gap_demotion",
            "selection_mode": "v5_global_support_after_probe_g",
            "global_candidates_examined": examined,
            "global_candidates_passing": len(candidates),
            "ranked_candidates_raw_rejected": raw_rejected,
            "selected_window_score": [float(value) for value in best["score"]],
            "probe_g": {
                **gap_diagnostics,
                "selected_observed_window_centers": observed_centers,
                "selected_geometric_gap_band_centers": demoted_in_window,
            },
        },
        all_bands,
        runs,
    )


def fit_trapezoid_boundary_retry_v5_probe_g(
    *args: Any,
    mode: Literal["off", "candidate"] = "candidate",
    policy: GeometricGapPolicy | None = None,
    **kwargs: Any,
):
    if mode == "off":
        return fit_trapezoid_boundary_retry_v5(*args, **kwargs)
    if mode != "candidate":
        raise ValueError(f"Unsupported Probe G mode: {mode}")
    policy = policy or GeometricGapPolicy()
    result, masks = v4.fit_trapezoid_boundary_retry_v4(
        *args,
        policy=policy,
        _support_selector=_global_support_with_geometric_gaps,
        **kwargs,
    )
    result["policy_id"] = POLICY_ID
    fit = result.get("envelope_fit", {})
    fit["search_direction"] = "v5_global_support_after_probe_g"

    probe_g = None
    if args and len(args) >= 4 and args[2] is not None and args[3] is not None:
        structural = np.asarray(args[2], dtype=bool)
        blocker = np.asarray(args[3], dtype=bool)
        silhouette = np.asarray(
            kwargs.get("obstacle_silhouette_mask", np.zeros_like(structural)),
            dtype=bool,
        )
        support, _, _ = _global_support_with_geometric_gaps(
            v4._records(structural, blocker, silhouette, policy),
            structural.shape,
            policy,
        )
        if support:
            probe_g = support.get("probe_g")
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
    result["probe_g_geometric_gap"] = probe_g or {
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


__all__ = [
    "POLICY_ID",
    "GeometricGapPolicy",
    "diagnose_geometric_gaps",
    "fit_trapezoid_boundary_retry_v5_probe_g",
]
