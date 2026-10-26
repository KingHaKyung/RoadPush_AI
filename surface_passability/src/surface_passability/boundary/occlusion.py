"""Diagnose one-sided geometric detours in straight-boundary support."""

from __future__ import annotations

import math
from typing import Any, Literal

import cv2
import numpy as np

from surface_passability.boundary.support import fit_boundary_bands
from surface_passability.obstacles.taxonomy import (
    BANNER_CLASS_ID,
    BOUNDARY_CONTAINMENT_OCCLUDER_CLASS_IDS,
)
from surface_passability.policies.geometric_gap import GeometricGapPolicy


def build_containment_occluder_mask(
    semantic_labels: np.ndarray | None,
    image_shape: tuple[int, int],
    *,
    include_banner: bool = False,
) -> np.ndarray:
    """Build the exact semantic evidence allowed to neutralize boundary gaps."""

    if semantic_labels is None:
        return np.zeros(image_shape, dtype=bool)
    labels = np.asarray(semantic_labels)
    if labels.ndim != 2:
        raise ValueError("Expected single-channel semantic map")
    if labels.shape != image_shape:
        labels = cv2.resize(
            labels.astype(np.uint8),
            (image_shape[1], image_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    class_ids = (
        (*BOUNDARY_CONTAINMENT_OCCLUDER_CLASS_IDS, BANNER_CLASS_ID)
        if include_banner
        else BOUNDARY_CONTAINMENT_OCCLUDER_CLASS_IDS
    )
    return np.isin(labels, class_ids)


def fit_boundary_side_context(
    bands: list[dict[str, Any]],
    side: Literal["left", "right"],
) -> dict[str, float] | None:
    """Fit one boundary side over a local band context."""

    if len(bands) < 2:
        return None
    rows = np.asarray([band["center_y"] for band in bands], dtype=float)
    columns = np.asarray([band[side] for band in bands], dtype=float)
    slope, intercept = np.polyfit(rows, columns, 1)
    return {"slope": float(slope), "intercept": float(intercept)}


def boundary_contexts_agree(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    midpoint_y: float,
    image_width: int,
    policy: GeometricGapPolicy,
) -> tuple[bool, dict[str, Any]]:
    """Compare boundary direction and prediction across an interior gap."""

    diagnostics: dict[str, Any] = {}
    passed = True
    for side in ("left", "right"):
        first = fit_boundary_side_context(before, side)
        second = fit_boundary_side_context(after, side)
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
            > policy.maximum_context_prediction_delta_fraction * image_width
        ):
            passed = False
    return passed, diagnostics


def group_geometric_gap_candidates(
    candidates: list[dict[str, Any]],
    band_rows: int,
) -> list[list[dict[str, Any]]]:
    """Group adjacent candidate bands when side and spacing agree."""

    groups: list[list[dict[str, Any]]] = []
    for candidate in candidates:
        if (
            not groups
            or candidate["side"] != groups[-1][-1]["side"]
            or candidate["center_y"] - groups[-1][-1]["center_y"]
            != band_rows
        ):
            groups.append([candidate])
        else:
            groups[-1].append(candidate)
    return groups


def diagnose_geometric_gaps(
    all_bands: list[dict[str, Any]],
    image_width: int,
    policy: GeometricGapPolicy,
) -> dict[str, Any]:
    """Find safe one-sided detours that must not anchor the boundary fit."""

    if len(all_bands) < policy.minimum_global_candidate_bands:
        return {
            "status": "insufficient",
            "provisional_fit": None,
            "candidate_groups": [],
            "safe_gap_band_centers": [],
        }
    left_fit, right_fit = fit_boundary_bands(all_bands, image_width, policy)
    if not left_fit.get("valid") or not right_fit.get("valid"):
        return {
            "status": "insufficient",
            "provisional_fit": None,
            "candidate_groups": [],
            "safe_gap_band_centers": [],
        }
    deviation_tolerance = policy.deviation_tolerance_fraction * image_width
    opposite_tolerance = (
        policy.opposite_stability_tolerance_fraction * image_width
    )
    candidates: list[dict[str, Any]] = []
    for index, band in enumerate(all_bands):
        row_y = float(band["center_y"])
        left_residual = float(
            band["left"]
            - (left_fit["slope"] * row_y + left_fit["intercept"])
        )
        right_residual = float(
            band["right"]
            - (right_fit["slope"] * row_y + right_fit["intercept"])
        )
        side = None
        signed_deviation = None
        opposite_residual = None
        if (
            abs(left_residual) > deviation_tolerance
            and abs(right_residual) <= opposite_tolerance
        ):
            side = "left"
            signed_deviation = left_residual
            opposite_residual = right_residual
        elif (
            abs(right_residual) > deviation_tolerance
            and abs(left_residual) <= opposite_tolerance
        ):
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
    group_diagnostics: list[dict[str, Any]] = []
    groups = group_geometric_gap_candidates(candidates, policy.band_rows)
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
                    all_bands[last_index + 1 :],
                    start=last_index + 1,
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
                safe, context = boundary_contexts_agree(
                    before,
                    after,
                    midpoint,
                    image_width,
                    policy,
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
        "provisional_fit": {"left": left_fit, "right": right_fit},
        "deviation_tolerance_px": deviation_tolerance,
        "opposite_stability_tolerance_px": opposite_tolerance,
        "candidate_groups": group_diagnostics,
        "safe_gap_band_centers": sorted(set(safe_centers)),
    }
