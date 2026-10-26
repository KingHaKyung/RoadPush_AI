"""Runwise direction evidence and conservative straight-boundary gating."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import math
from typing import Any, Literal

import numpy as np

from surface_passability.boundary.global_support import (
    GlobalBoundarySupportPolicy,
)
from surface_passability.boundary.support import build_boundary_band
from surface_passability.policies.direction import RunwiseDirectionPolicy


POLICY_ID = "phase7_trapezoid_boundary_retry_v5_probe_b_runwise_direction"


def rebuild_selected_boundary_bands(
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Rebuild the selected bands from serialized row evidence."""

    fit = result.get("envelope_fit", {})
    centers = fit.get("selected_band_centers", [])
    if not centers:
        return []
    policy_values = result.get("policy", {})
    allowed = GlobalBoundarySupportPolicy.__dataclass_fields__
    policy = GlobalBoundarySupportPolicy(
        **{
            key: value
            for key, value in policy_values.items()
            if key in allowed
        }
    )
    rows_by_y = {
        row["y"]: row for row in result.get("row_evidence", [])
    }
    bands = [
        build_boundary_band(rows_by_y, int(center), policy)
        for center in centers
    ]
    return [band for band in bands if band is not None]


def split_contiguous_boundary_runs(
    bands: list[dict[str, Any]],
    band_rows: int,
) -> list[list[dict[str, Any]]]:
    """Split selected bands wherever their fixed row spacing is broken."""

    runs: list[list[dict[str, Any]]] = []
    for band in bands:
        if (
            not runs
            or int(band["center_y"])
            - int(runs[-1][-1]["center_y"])
            != band_rows
        ):
            runs.append([band])
        else:
            runs[-1].append(band)
    return runs


def reject_straight_boundary_candidate(
    result: dict[str, Any],
    masks: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Return an isolated rejected copy while preserving provisional evidence."""

    gated = deepcopy(result)
    gated_masks = {name: value.copy() for name, value in masks.items()}
    fit = gated["envelope_fit"]
    fit["status"] = "rejected"
    fit["confidence"] = "insufficient"
    fit["rejection_reasons"] = list(
        dict.fromkeys(
            [
                *fit.get("rejection_reasons", []),
                "direction_consistency_gate_failed",
            ]
        )
    )
    gated["minimum_passable_ratio"] = None
    gated["maximum_obstructed_width_ratio"] = None
    gated["obstacle_area_ratio"] = None
    gated["passability_state"] = "uncertain"
    gated["representative_measurement_allowed"] = False
    gated["corridor_measurement"] = None
    if "envelope" in gated_masks:
        provisional = gated_masks["envelope"].copy()
        gated_masks["probe_b_provisional_envelope"] = provisional
        gated_masks["envelope"] = np.zeros_like(provisional)
        gated_masks["free"] = np.zeros_like(provisional)
        structural = gated_masks.get(
            "authoritative_structural",
            np.zeros_like(provisional),
        )
        gated_masks["added"] = np.zeros_like(provisional)
        gated_masks["omitted"] = structural.copy()
        gated["diagnostics"]["envelope_pixels"] = 0
        gated["diagnostics"]["added_structural_pixels"] = 0
        gated["diagnostics"]["omitted_structural_pixels"] = int(
            np.count_nonzero(structural)
        )
    return gated, gated_masks


def _angle(slope: float) -> float:
    return float(math.degrees(math.atan(float(slope))))


def _robust_spread(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    array = np.asarray(values, dtype=float)
    return float(np.percentile(array, 90) - np.percentile(array, 10))


def _side_run_metrics(
    run: list[dict[str, Any]],
    side: Literal["left", "right"],
    policy: RunwiseDirectionPolicy,
) -> dict[str, Any]:
    segment_angles = [
        _angle(
            (second[side] - first[side])
            / float(second["center_y"] - first["center_y"])
        )
        for first, second in zip(run, run[1:])
    ]
    triplet_turns = [
        second - first
        for first, second in zip(segment_angles, segment_angles[1:])
    ]
    material_turns = [
        value
        for value in triplet_turns
        if abs(value) >= policy.material_triplet_turn_degrees
    ]
    positive = sum(value > 0 for value in material_turns)
    negative = sum(value < 0 for value in material_turns)
    dominant_fraction = (
        max(positive, negative) / len(material_turns)
        if material_turns
        else None
    )
    net_turn = (
        float(segment_angles[-1] - segment_angles[0])
        if len(segment_angles) >= 2
        else None
    )
    total_variation = float(sum(abs(value) for value in triplet_turns))
    return_excess = (
        float(total_variation - abs(net_turn))
        if net_turn is not None
        else None
    )

    smoothed = []
    window_size = policy.smoothing_window_bands
    if len(run) >= window_size:
        for start in range(len(run) - window_size + 1):
            window = run[start : start + window_size]
            rows = np.asarray(
                [band["center_y"] for band in window],
                dtype=float,
            )
            columns = np.asarray(
                [band[side] for band in window],
                dtype=float,
            )
            slope = float(np.polyfit(rows, columns, 1)[0])
            smoothed.append(
                {
                    "start_center_y": int(window[0]["center_y"]),
                    "end_center_y": int(window[-1]["center_y"]),
                    "center_y": float(np.mean(rows)),
                    "slope_dx_per_dy": slope,
                    "angle_degrees": _angle(slope),
                }
            )
    smoothed_angles = [item["angle_degrees"] for item in smoothed]

    weak_detour_pattern = False
    if len(material_turns) >= 3:
        first_sign = 1 if material_turns[0] > 0 else -1
        last_two = material_turns[-2:]
        weak_detour_pattern = all(
            (1 if value > 0 else -1) == -first_sign for value in last_two
        )

    representative_slope = None
    representative_angle = None
    if len(run) >= 2:
        rows = np.asarray(
            [band["center_y"] for band in run],
            dtype=float,
        )
        columns = np.asarray([band[side] for band in run], dtype=float)
        representative_slope = float(np.polyfit(rows, columns, 1)[0])
        representative_angle = _angle(representative_slope)

    return {
        "side": side,
        "segment_angles_degrees": segment_angles,
        "triplet_turns_degrees": triplet_turns,
        "material_triplet_turns_degrees": material_turns,
        "dominant_turn_fraction": dominant_fraction,
        "net_turn_degrees": net_turn,
        "total_turn_variation_degrees": total_variation,
        "return_excess_degrees": return_excess,
        "weak_detour_pattern": weak_detour_pattern,
        "smoothed_local_windows": smoothed,
        "smoothed_local_window_count": len(smoothed),
        "smoothed_angle_spread_degrees": _robust_spread(smoothed_angles),
        "representative_slope_dx_per_dy": representative_slope,
        "representative_angle_degrees": representative_angle,
    }


def _classify_run(
    run: list[dict[str, Any]],
    left: dict[str, Any],
    right: dict[str, Any],
    policy: RunwiseDirectionPolicy,
) -> dict[str, Any]:
    strong = len(run) >= policy.minimum_strong_run_bands
    coherent_sides = []
    volatile_sides = []
    severe_detour_sides = []
    weak_detour_sides = []
    for side, metrics in (("left", left), ("right", right)):
        spread = metrics["smoothed_angle_spread_degrees"]
        dominant = metrics["dominant_turn_fraction"]
        net = metrics["net_turn_degrees"]
        if (
            strong
            and spread is not None
            and dominant is not None
            and net is not None
            and spread > policy.coherent_smoothed_spread_degrees
            and dominant >= policy.minimum_dominant_turn_fraction
            and abs(net) >= policy.coherent_net_turn_degrees
        ):
            coherent_sides.append(side)
        if (
            strong
            and spread is not None
            and dominant is not None
            and spread
            > policy.bilateral_instability_smoothed_spread_degrees
            and dominant < policy.minimum_dominant_turn_fraction
        ):
            volatile_sides.append(side)
        if (
            strong
            and spread is not None
            and spread > policy.severe_detour_smoothed_spread_degrees
            and side not in coherent_sides
        ):
            severe_detour_sides.append(side)
        if (
            not strong
            and metrics["weak_detour_pattern"]
            and metrics["return_excess_degrees"] is not None
            and metrics["return_excess_degrees"]
            > policy.weak_detour_return_excess_degrees
        ):
            weak_detour_sides.append(side)

    if coherent_sides:
        classification = "coherent_turn"
        confidence = "strong"
    elif len(volatile_sides) == 2:
        classification = "unstable_bilateral"
        confidence = "strong"
    elif len(severe_detour_sides) == 1:
        classification = "one_sided_detour"
        confidence = "strong"
    elif len(weak_detour_sides) == 1:
        classification = "one_sided_detour"
        confidence = "weak"
    elif strong:
        classification = "consistent"
        confidence = "strong"
    else:
        classification = "insufficient"
        confidence = "weak" if len(run) >= 3 else "insufficient"
    return {
        "classification": classification,
        "confidence": confidence,
        "coherent_turn_sides": coherent_sides,
        "volatile_sides": volatile_sides,
        "severe_detour_sides": severe_detour_sides,
        "weak_detour_sides": weak_detour_sides,
    }


def _run_diagnostics(
    run: list[dict[str, Any]],
    run_index: int,
    policy: RunwiseDirectionPolicy,
) -> dict[str, Any]:
    left = _side_run_metrics(run, "left", policy)
    right = _side_run_metrics(run, "right", policy)
    classification = _classify_run(run, left, right, policy)
    return {
        "run_index": run_index,
        "band_count": len(run),
        "band_centers": [int(band["center_y"]) for band in run],
        "left": left,
        "right": right,
        **classification,
    }


def _cross_gap_alignment(
    runs: list[dict[str, Any]],
    policy: RunwiseDirectionPolicy,
) -> list[dict[str, Any]]:
    comparisons = []
    for first, second in zip(runs, runs[1:]):
        side_deltas = {}
        available = True
        for side in ("left", "right"):
            first_angle = first[side]["representative_angle_degrees"]
            second_angle = second[side]["representative_angle_degrees"]
            if first_angle is None or second_angle is None:
                available = False
                side_deltas[side] = None
            else:
                side_deltas[side] = float(abs(second_angle - first_angle))
        compatible = available and all(
            value is not None
            and value <= policy.cross_gap_alignment_degrees
            for value in side_deltas.values()
        )
        comparisons.append(
            {
                "from_run_index": first["run_index"],
                "to_run_index": second["run_index"],
                "left_representative_angle_delta_degrees": (
                    side_deltas["left"]
                ),
                "right_representative_angle_delta_degrees": (
                    side_deltas["right"]
                ),
                "compatible": compatible if available else None,
                "confidence": "low",
                "used_as_observed_curvature": False,
                "used_by_gate": False,
            }
        )
    return comparisons


def diagnose_runwise_direction_consistency(
    result: dict[str, Any],
    policy: RunwiseDirectionPolicy | None = None,
) -> dict[str, Any]:
    """Classify each contiguous support run without bridging its gaps."""

    policy = policy or RunwiseDirectionPolicy()
    fit = result.get("envelope_fit", {})
    if fit.get("status") != "accepted":
        return {
            "classification": "not_applicable",
            "status": "not_applicable",
            "reason": "v5_envelope_not_accepted",
            "gate_would_reject": False,
            "gate_action": "preserve_v5_rejection",
            "thresholds": asdict(policy),
            "runs": [],
            "cross_gap_alignment": [],
        }
    bands = rebuild_selected_boundary_bands(result)
    band_rows = int(result.get("policy", {}).get("band_rows", 7))
    raw_runs = split_contiguous_boundary_runs(bands, band_rows)
    runs = [
        _run_diagnostics(run, index, policy)
        for index, run in enumerate(raw_runs)
    ]
    classifications = [run["classification"] for run in runs]
    if "unstable_bilateral" in classifications:
        classification = "unstable_bilateral"
    elif "coherent_turn" in classifications:
        classification = "coherent_turn"
    elif "one_sided_detour" in classifications:
        classification = "one_sided_detour"
    elif "consistent" in classifications:
        classification = "consistent"
    else:
        classification = "insufficient"
    gate_would_reject = classification in {
        "coherent_turn",
        "unstable_bilateral",
    }
    gate_action = (
        "defer_curve_model_and_reject_straight"
        if classification == "coherent_turn"
        else "reject_directionally_unstable_straight"
        if classification == "unstable_bilateral"
        else "forward_to_probe_g_without_rejecting_v5"
        if classification == "one_sided_detour"
        else "preserve_v5"
    )
    failing_sides = sorted(
        {
            side
            for run in runs
            for key in ("coherent_turn_sides", "volatile_sides")
            for side in run[key]
        }
    )
    return {
        "classification": classification,
        "status": classification,
        "gate_would_reject": gate_would_reject,
        "gate_action": gate_action,
        "failing_sides": failing_sides,
        "selected_band_count": len(bands),
        "selected_band_centers": [
            int(band["center_y"]) for band in bands
        ],
        "runs": runs,
        "cross_gap_alignment": _cross_gap_alignment(runs, policy),
        "cross_gap_alignment_used_by_gate": False,
        "thresholds": asdict(policy),
    }


def apply_runwise_direction_consistency(
    result: dict[str, Any],
    masks: dict[str, np.ndarray],
    *,
    mode: Literal["diagnostic_only", "candidate_gate"] = "diagnostic_only",
    direction_policy: RunwiseDirectionPolicy | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Attach runwise direction evidence and optionally reject the candidate."""

    if mode not in {"diagnostic_only", "candidate_gate"}:
        raise ValueError(f"Unsupported Probe B mode: {mode}")
    policy = direction_policy or RunwiseDirectionPolicy()
    diagnostics = diagnose_runwise_direction_consistency(result, policy)
    output = deepcopy(result)
    output["policy_id"] = POLICY_ID
    output["probe_b_direction_consistency"] = diagnostics
    output["probe_b_contract"] = {
        "mode": mode,
        "triplet_is_minimum_direction_observation": True,
        "five_band_fit_is_auxiliary_only": True,
        "continuous_runs_judged_independently": True,
        "cross_gap_alignment_used_as_curvature": False,
        "cross_gap_alignment_used_by_gate": False,
        "one_sided_detour_hard_rejected": False,
        "v5_candidate_selection_unchanged": True,
        "v5_ranking_and_thresholds_unchanged": True,
        "blocker_logic_changed": False,
        "probe_g_or_c_added": False,
        "threshold_status": "provisional_targeted_review_candidate",
    }
    if mode == "candidate_gate" and diagnostics["gate_would_reject"]:
        output, masks = reject_straight_boundary_candidate(output, masks)
        reason = (
            "probe_b_coherent_turn_deferred"
            if diagnostics["classification"] == "coherent_turn"
            else "probe_b_unstable_bilateral"
        )
        output["envelope_fit"]["rejection_reasons"] = list(
            dict.fromkeys(
                [*output["envelope_fit"]["rejection_reasons"], reason]
            )
        )
    output["probe_b_direction_consistency"]["gate_applied"] = (
        mode == "candidate_gate" and diagnostics["gate_would_reject"]
    )
    return output, masks
