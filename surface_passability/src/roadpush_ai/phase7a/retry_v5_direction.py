from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from typing import Any, Literal

import numpy as np

import roadpush_ai.phase7a.retry_v4 as v4
from roadpush_ai.phase7a.retry_v5 import (
    TrapezoidBoundaryRetryV5Policy,
    fit_trapezoid_boundary_retry_v5,
)


POLICY_ID = "phase7_trapezoid_boundary_retry_v5_probe_b_direction_consistency"


@dataclass(frozen=True)
class DirectionConsistencyPolicy:
    """Review-candidate thresholds for the isolated Probe B sweep.

    These values are deliberately probe-only. They are not part of Retry V5
    and are not frozen production thresholds.
    """

    local_window_bands: int = 5
    minimum_local_windows: int = 2
    robust_angle_spread_degrees: float = 22.0
    severe_angle_spread_degrees: float = 45.0
    minimum_monotonic_turn_fraction: float = 0.70


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _monotonic_fraction(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    turns = np.diff(np.asarray(values, dtype=float))
    tolerance = 1e-9
    nondecreasing = float(np.mean(turns >= -tolerance))
    nonincreasing = float(np.mean(turns <= tolerance))
    return max(nondecreasing, nonincreasing)


def _selected_bands(result: dict[str, Any]) -> list[dict[str, Any]]:
    fit = result.get("envelope_fit", {})
    centers = fit.get("selected_band_centers", [])
    if not centers:
        return []
    policy_values = result.get("policy", {})
    allowed = TrapezoidBoundaryRetryV5Policy.__dataclass_fields__
    policy = TrapezoidBoundaryRetryV5Policy(
        **{key: value for key, value in policy_values.items() if key in allowed}
    )
    rows_by_y = {row["y"]: row for row in result.get("row_evidence", [])}
    bands = [v4._band(rows_by_y, int(center), policy) for center in centers]
    return [band for band in bands if band is not None]


def _contiguous_runs(
    bands: list[dict[str, Any]], band_rows: int
) -> list[list[dict[str, Any]]]:
    runs: list[list[dict[str, Any]]] = []
    for band in bands:
        if (
            not runs
            or int(band["center_y"]) - int(runs[-1][-1]["center_y"])
            != band_rows
        ):
            runs.append([band])
        else:
            runs[-1].append(band)
    return runs


def _side_diagnostics(
    runs: list[list[dict[str, Any]]],
    side: Literal["left", "right"],
    global_fit: dict[str, Any],
    policy: DirectionConsistencyPolicy,
) -> dict[str, Any]:
    local: list[dict[str, Any]] = []
    segment_angles: list[float] = []
    for run_index, run in enumerate(runs):
        for first, second in zip(run, run[1:]):
            dy = float(second["center_y"] - first["center_y"])
            slope = float((second[side] - first[side]) / dy)
            segment_angles.append(float(math.degrees(math.atan(slope))))
        if len(run) < policy.local_window_bands:
            continue
        for start in range(len(run) - policy.local_window_bands + 1):
            window = run[start : start + policy.local_window_bands]
            ys = np.asarray([band["center_y"] for band in window], dtype=float)
            xs = np.asarray([band[side] for band in window], dtype=float)
            slope = float(np.polyfit(ys, xs, 1)[0])
            local.append(
                {
                    "run_index": run_index,
                    "center_y": float(np.mean(ys)),
                    "start_center_y": int(window[0]["center_y"]),
                    "end_center_y": int(window[-1]["center_y"]),
                    "slope_dx_per_dy": slope,
                    "angle_degrees": float(math.degrees(math.atan(slope))),
                }
            )

    angles = [item["angle_degrees"] for item in local]
    q10 = _percentile(angles, 10.0)
    q90 = _percentile(angles, 90.0)
    spread = float(q90 - q10) if q10 is not None and q90 is not None else None
    coherence = _monotonic_fraction(angles)
    global_angle = (
        float(math.degrees(math.atan(float(global_fit["slope"]))))
        if global_fit.get("valid") and "slope" in global_fit
        else None
    )
    return {
        "side": side,
        "global_fit_angle_degrees": global_angle,
        "adjacent_segment_angles_degrees": segment_angles,
        "local_windows": local,
        "local_window_count": len(local),
        "local_angle_q10_degrees": q10,
        "local_angle_q90_degrees": q90,
        "robust_angle_spread_degrees": spread,
        "monotonic_turn_fraction": coherence,
    }


def _paired_diagnostics(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_by_key = {
        (item["run_index"], item["center_y"]): item
        for item in left["local_windows"]
    }
    pairs = []
    for item in right["local_windows"]:
        key = (item["run_index"], item["center_y"])
        if key not in left_by_key:
            continue
        left_item = left_by_key[key]
        pairs.append(
            {
                "run_index": item["run_index"],
                "center_y": item["center_y"],
                "left_angle_degrees": left_item["angle_degrees"],
                "right_angle_degrees": item["angle_degrees"],
                "opening_angle_degrees": (
                    item["angle_degrees"] - left_item["angle_degrees"]
                ),
            }
        )
    openings = [item["opening_angle_degrees"] for item in pairs]
    return {
        "paired_local_window_count": len(pairs),
        "local_windows": pairs,
        "opening_angle_q10_degrees": _percentile(openings, 10.0),
        "opening_angle_q90_degrees": _percentile(openings, 90.0),
        "opening_angle_robust_spread_degrees": (
            float(np.percentile(openings, 90) - np.percentile(openings, 10))
            if openings
            else None
        ),
    }


def evaluate_direction_diagnostics(
    diagnostics: dict[str, Any], policy: DirectionConsistencyPolicy
) -> dict[str, Any]:
    failing_sides: list[str] = []
    sufficient_sides: list[str] = []
    for side in ("left", "right"):
        values = diagnostics[side]
        if values["local_window_count"] < policy.minimum_local_windows:
            continue
        sufficient_sides.append(side)
        spread = values["robust_angle_spread_degrees"]
        coherence = values["monotonic_turn_fraction"]
        if (
            spread is not None
            and coherence is not None
            and (
                spread > policy.severe_angle_spread_degrees
                or (
                    spread > policy.robust_angle_spread_degrees
                    and coherence >= policy.minimum_monotonic_turn_fraction
                )
            )
        ):
            failing_sides.append(side)
    if not sufficient_sides:
        status = "insufficient_direction_evidence"
    elif failing_sides:
        status = "inconsistent"
    else:
        status = "consistent"
    return {
        "status": status,
        "sufficient_sides": sufficient_sides,
        "failing_sides": failing_sides,
        "gate_would_reject": bool(failing_sides),
        "thresholds": asdict(policy),
    }


def diagnose_direction_consistency(
    result: dict[str, Any],
    policy: DirectionConsistencyPolicy | None = None,
) -> dict[str, Any]:
    policy = policy or DirectionConsistencyPolicy()
    fit = result.get("envelope_fit", {})
    if fit.get("status") != "accepted":
        return {
            "status": "not_applicable",
            "reason": "v5_envelope_not_accepted",
            "gate_would_reject": False,
            "thresholds": asdict(policy),
        }
    bands = _selected_bands(result)
    band_rows = int(result.get("policy", {}).get("band_rows", 7))
    runs = _contiguous_runs(bands, band_rows)
    left = _side_diagnostics(runs, "left", fit.get("left_fit", {}), policy)
    right = _side_diagnostics(runs, "right", fit.get("right_fit", {}), policy)
    diagnostics = {
        "selected_band_count": len(bands),
        "selected_band_centers": [int(band["center_y"]) for band in bands],
        "contiguous_run_centers": [
            [int(band["center_y"]) for band in run] for run in runs
        ],
        "left": left,
        "right": right,
        "paired": _paired_diagnostics(left, right),
    }
    diagnostics.update(evaluate_direction_diagnostics(diagnostics, policy))
    return diagnostics


def direction_threshold_sweep(
    diagnostics: dict[str, Any],
    *,
    spreads: tuple[float, ...] = (15.0, 20.0, 22.0, 25.0, 30.0),
    monotonic_fractions: tuple[float, ...] = (0.65, 0.70, 0.75),
) -> list[dict[str, Any]]:
    if diagnostics.get("status") == "not_applicable":
        return []
    rows = []
    for spread in spreads:
        for monotonic in monotonic_fractions:
            policy = DirectionConsistencyPolicy(
                robust_angle_spread_degrees=spread,
                minimum_monotonic_turn_fraction=monotonic,
            )
            evaluation = evaluate_direction_diagnostics(diagnostics, policy)
            rows.append(
                {
                    "robust_angle_spread_degrees": spread,
                    "minimum_monotonic_turn_fraction": monotonic,
                    "status": evaluation["status"],
                    "failing_sides": evaluation["failing_sides"],
                    "gate_would_reject": evaluation["gate_would_reject"],
                }
            )
    return rows


def _apply_gate(
    result: dict[str, Any], masks: dict[str, np.ndarray]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    gated = deepcopy(result)
    gated_masks = {name: value.copy() for name, value in masks.items()}
    fit = gated["envelope_fit"]
    fit["status"] = "rejected"
    fit["confidence"] = "insufficient"
    fit["rejection_reasons"] = list(
        dict.fromkeys(
            [*fit.get("rejection_reasons", []), "direction_consistency_gate_failed"]
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
        structural = gated_masks.get("authoritative_structural", np.zeros_like(provisional))
        gated_masks["added"] = np.zeros_like(provisional)
        gated_masks["omitted"] = structural.copy()
        gated["diagnostics"]["envelope_pixels"] = 0
        gated["diagnostics"]["added_structural_pixels"] = 0
        gated["diagnostics"]["omitted_structural_pixels"] = int(
            np.count_nonzero(structural)
        )
    return gated, gated_masks


def fit_trapezoid_boundary_retry_v5_probe_b(
    *args: Any,
    mode: Literal["off", "diagnostic_only", "candidate_gate"] = "diagnostic_only",
    direction_policy: DirectionConsistencyPolicy | None = None,
    **kwargs: Any,
):
    """Run isolated Probe B after the frozen V5 result.

    ``off`` returns the V5 objects without adding or changing any fields.
    ``diagnostic_only`` adds direction measurements but never changes geometry.
    ``candidate_gate`` applies the explicitly provisional sweep candidate.
    """

    result, masks = fit_trapezoid_boundary_retry_v5(*args, **kwargs)
    if mode == "off":
        return result, masks
    if mode not in {"diagnostic_only", "candidate_gate"}:
        raise ValueError(f"Unsupported Probe B mode: {mode}")
    policy = direction_policy or DirectionConsistencyPolicy()
    diagnostics = diagnose_direction_consistency(result, policy)
    output = deepcopy(result)
    output["policy_id"] = POLICY_ID
    output["probe_b_direction_consistency"] = diagnostics
    output["probe_b_threshold_sweep"] = direction_threshold_sweep(diagnostics)
    output["probe_b_contract"] = {
        "mode": mode,
        "v5_candidate_selection_unchanged": True,
        "v5_ranking_unchanged": True,
        "v5_residual_thresholds_unchanged": True,
        "v5_band_generation_unchanged": True,
        "blocker_logic_changed": False,
        "geometric_gap_pruning_added": False,
        "continuous_core_ranking_added": False,
        "vegetation_gate_added": False,
        "extension_or_curve_added": False,
        "threshold_status": "provisional_targeted_review_candidate",
    }
    if mode == "candidate_gate" and diagnostics.get("gate_would_reject"):
        output, masks = _apply_gate(output, masks)
    output["probe_b_direction_consistency"]["gate_applied"] = (
        mode == "candidate_gate" and diagnostics.get("gate_would_reject", False)
    )
    return output, masks


__all__ = [
    "POLICY_ID",
    "DirectionConsistencyPolicy",
    "diagnose_direction_consistency",
    "direction_threshold_sweep",
    "evaluate_direction_diagnostics",
    "fit_trapezoid_boundary_retry_v5_probe_b",
]
