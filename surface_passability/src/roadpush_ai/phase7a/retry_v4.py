from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from roadpush_ai.phase7a.retry_v1 import (
    _envelope_mask,
    _measure_corridors,
    _robust_line,
    binary_mask_sha256,
)
from roadpush_ai.phase7a.retry_v2 import _raw_pruned


@dataclass(frozen=True)
class TrapezoidBoundaryRetryV4Policy:
    minimum_span_px: int = 12
    minimum_span_fraction: float = 0.02
    clean_fill_ratio: float = 0.80
    band_rows: int = 7
    minimum_passing_rows_per_band: int = 5
    endpoint_contact_radius_px: int = 6
    minimum_clean_cap_bands: int = 2
    minimum_anchor_rows: int = 12
    minimum_segment_median_span_fraction: float = 0.10
    fit_inlier_tolerance_fraction: float = 0.02
    minimum_inlier_fraction: float = 0.80
    maximum_median_residual_fraction: float = 0.015
    maximum_p90_residual_fraction: float = 0.03
    minimum_median_envelope_evidence: float = 0.70
    minimum_corridor_support_rows: int = 5
    measurement_band_rows: int = 3
    blocker_suspect_added_area_fraction: float = 0.05


def _records(
    structural: np.ndarray,
    blocker: np.ndarray,
    obstacle_silhouette: np.ndarray,
    policy: TrapezoidBoundaryRetryV4Policy,
) -> list[dict[str, Any]]:
    h, w = structural.shape
    radius = policy.endpoint_contact_radius_px
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    blocker_contact = cv2.dilate(blocker.astype(np.uint8), kernel) > 0
    silhouette_contact = cv2.dilate(
        obstacle_silhouette.astype(np.uint8), kernel
    ) > 0
    min_span = max(policy.minimum_span_px, int(np.ceil(policy.minimum_span_fraction * w)))
    records: list[dict[str, Any]] = []
    for y in range(h):
        xs = np.flatnonzero(structural[y])
        if not xs.size:
            continue
        left, right = int(xs[0]), int(xs[-1])
        span = right - left + 1
        fill = float(xs.size / span)
        left_frame = left == 0
        right_frame = right == w - 1
        left_blocker = bool(blocker_contact[y, left])
        right_blocker = bool(blocker_contact[y, right])
        left_silhouette = bool(silhouette_contact[y, left])
        right_silhouette = bool(silhouette_contact[y, right])
        reasons: list[str] = []
        if span < min_span:
            reasons.append("short_span")
        if fill < policy.clean_fill_ratio:
            reasons.append("low_fill")
        if left_frame:
            reasons.append("left_frame_contact")
        if right_frame:
            reasons.append("right_frame_contact")
        if left_blocker:
            reasons.append("left_endpoint_blocker_contact")
        if right_blocker:
            reasons.append("right_endpoint_blocker_contact")
        if left_silhouette:
            reasons.append("left_endpoint_obstacle_silhouette_contact")
        if right_silhouette:
            reasons.append("right_endpoint_obstacle_silhouette_contact")
        records.append(
            {
                "y": y,
                "left": left,
                "right": right,
                "span": span,
                "fill_ratio": fill,
                "blocker_share": float(
                    np.count_nonzero(blocker[y, left : right + 1]) / span
                ),
                "frame_contact": left_frame or right_frame,
                "left_frame_contact": left_frame,
                "right_frame_contact": right_frame,
                "left_blocker_contact": left_blocker,
                "right_blocker_contact": right_blocker,
                "left_obstacle_silhouette_contact": left_silhouette,
                "right_obstacle_silhouette_contact": right_silhouette,
                "obstacle_contact": (
                    left_blocker
                    or right_blocker
                    or left_silhouette
                    or right_silhouette
                ),
                "clean_bilateral": not reasons,
                "reasons": reasons,
            }
        )
    return records


def _band(
    rows_by_y: dict[int, dict[str, Any]],
    center: int,
    policy: TrapezoidBoundaryRetryV4Policy,
) -> dict[str, Any] | None:
    radius = policy.band_rows // 2
    rows = [
        rows_by_y[y]
        for y in range(center - radius, center + radius + 1)
        if y in rows_by_y and rows_by_y[y]["clean_bilateral"]
    ]
    if len(rows) < policy.minimum_passing_rows_per_band:
        return None
    return {
        "center_y": center,
        "passing_y": [row["y"] for row in rows],
        "left": float(np.median([row["left"] for row in rows])),
        "right": float(np.median([row["right"] for row in rows])),
        "span": float(np.median([row["span"] for row in rows])),
    }


def _fit_bands(
    bands: list[dict[str, Any]],
    width: int,
    policy: TrapezoidBoundaryRetryV4Policy,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ys = np.asarray([band["center_y"] for band in bands], dtype=float)
    tolerance = policy.fit_inlier_tolerance_fraction * width
    return (
        _robust_line(ys, np.asarray([band["left"] for band in bands]), tolerance),
        _robust_line(ys, np.asarray([band["right"] for band in bands]), tolerance),
    )


def _fit_passes(
    left: dict[str, Any],
    right: dict[str, Any],
    width: int,
    policy: TrapezoidBoundaryRetryV4Policy,
) -> bool:
    for fit in (left, right):
        if not fit.get("valid"):
            return False
        if fit["inlier_fraction"] < policy.minimum_inlier_fraction:
            return False
        if fit["median_residual_px"] > policy.maximum_median_residual_fraction * width:
            return False
        if fit["p90_residual_px"] > policy.maximum_p90_residual_fraction * width:
            return False
    return True


def _clean_runs(
    bands: list[dict[str, Any]], policy: TrapezoidBoundaryRetryV4Policy
) -> list[list[dict[str, Any]]]:
    runs: list[list[dict[str, Any]]] = []
    for band in bands:
        if not runs or band["center_y"] - runs[-1][-1]["center_y"] != policy.band_rows:
            runs.append([band])
        else:
            runs[-1].append(band)
    return runs


def _bottom_up_support(
    records: list[dict[str, Any]],
    shape: tuple[int, int],
    policy: TrapezoidBoundaryRetryV4Policy,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[list[dict[str, Any]]]]:
    """Start at the lowest sustained clean cap and extend upward.

    Missing clean bands inside the selected vertical range are occlusion gaps:
    they provide no boundary anchors but do not split support. A clean band that
    resumes above a gap is accepted only while the joint straight fit remains
    stable. The last accepted clean band is therefore also a clean far cap.
    """

    if not records:
        return None, [], []
    _, w = shape
    by_y = {row["y"]: row for row in records}
    low, high = min(by_y), max(by_y)
    radius = policy.band_rows // 2
    all_bands = [
        band
        for center in range(low + radius, high - radius + 1, policy.band_rows)
        if (band := _band(by_y, center, policy)) is not None
    ]
    runs = _clean_runs(all_bands, policy)
    cap_runs = [run for run in runs if len(run) >= policy.minimum_clean_cap_bands]
    if not cap_runs:
        return None, all_bands, runs

    near_run = max(cap_runs, key=lambda run: run[-1]["center_y"])
    near_seed = list(reversed(near_run[-policy.minimum_clean_cap_bands :]))
    selected_bottom_up = near_seed[:]
    seed_top = min(band["center_y"] for band in near_seed)
    remaining = sorted(
        (band for band in all_bands if band["center_y"] < seed_top),
        key=lambda band: band["center_y"],
        reverse=True,
    )
    stop_reason = None
    for band in remaining:
        proposed = selected_bottom_up + [band]
        left, right = _fit_bands(proposed, w, policy)
        if not _fit_passes(left, right, w, policy):
            stop_reason = "upper_clean_band_breaks_lower_seeded_fit"
            break
        selected_bottom_up = proposed

    selected = sorted(selected_bottom_up, key=lambda band: band["center_y"])
    passing_y = sorted({y for band in selected for y in band["passing_y"]})
    median_span = float(np.median([band["span"] for band in selected]))
    return (
        {
            "bands": selected,
            "passing_y": passing_y,
            "median_span": median_span,
            "median_span_fraction": float(median_span / max(1, w)),
            "segment_span_gate_passed": (
                median_span / max(1, w)
                >= policy.minimum_segment_median_span_fraction
            ),
            "near_cap_centers": sorted(
                band["center_y"] for band in near_seed
            ),
            "far_cap_center": selected[0]["center_y"],
            "internal_gap_band_count": sum(
                max(
                    0,
                    (right_band["center_y"] - left_band["center_y"])
                    // policy.band_rows
                    - 1,
                )
                for left_band, right_band in zip(selected, selected[1:])
            ),
            "extension_stop_reason": stop_reason,
        },
        all_bands,
        runs,
    )


def fit_trapezoid_boundary_retry_v4(
    selected_surface: str | None,
    selection_state: str,
    structural_mask: np.ndarray | None,
    blocker_mask: np.ndarray | None,
    *,
    obstacle_silhouette_mask: np.ndarray | None = None,
    raw_structural_mask: np.ndarray | None = None,
    policy: TrapezoidBoundaryRetryV4Policy | None = None,
    _support_selector: Any | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    policy = policy or TrapezoidBoundaryRetryV4Policy()
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
        raise ValueError("structural, blocker and silhouette masks must be same-shape 2D arrays")

    h, w = structural.shape
    records = _records(structural, blocker, silhouette, policy)
    support_selector = _support_selector or _bottom_up_support
    support, all_bands, runs = support_selector(records, structural.shape, policy)
    reasons: list[str] = []
    if not np.any(structural):
        reasons.append("no_authoritative_structural_pixels")
    if support is None:
        reasons.append("no_bilateral_clean_caps")

    anchors: list[dict[str, Any]] = []
    y_range: list[int] | None = None
    if support is not None:
        passing = set(support["passing_y"])
        anchors = [
            row
            for row in records
            if row["y"] in passing and row["clean_bilateral"]
        ]
        if anchors:
            y_range = [min(row["y"] for row in anchors), max(row["y"] for row in anchors)]
        if not support["segment_span_gate_passed"]:
            reasons.append("insufficient_segment_median_span")
    if len(anchors) < policy.minimum_anchor_rows:
        reasons.append("insufficient_anchor_rows")

    left_fit: dict[str, Any] = {"valid": False, "reason": "not_fitted"}
    right_fit: dict[str, Any] = {"valid": False, "reason": "not_fitted"}
    envelope = np.zeros_like(structural)
    evidence: list[float] = []
    widths: list[float] = []
    if len(anchors) >= 2 and y_range:
        ys = np.asarray([row["y"] for row in anchors], dtype=float)
        tolerance = policy.fit_inlier_tolerance_fraction * w
        left_fit = _robust_line(
            ys, np.asarray([row["left"] for row in anchors]), tolerance
        )
        right_fit = _robust_line(
            ys, np.asarray([row["right"] for row in anchors]), tolerance
        )
        for side, fit in (("left", left_fit), ("right", right_fit)):
            if not fit.get("valid"):
                reasons.append(f"{side}_fit_failed")
                continue
            if fit["inlier_fraction"] < policy.minimum_inlier_fraction:
                reasons.append(f"{side}_low_inlier_fraction")
            if fit["median_residual_px"] > policy.maximum_median_residual_fraction * w:
                reasons.append(f"{side}_high_median_residual")
            if fit["p90_residual_px"] > policy.maximum_p90_residual_fraction * w:
                reasons.append(f"{side}_high_p90_residual")
        if left_fit.get("valid") and right_fit.get("valid"):
            widths = [
                right_fit["slope"] * y
                + right_fit["intercept"]
                - (left_fit["slope"] * y + left_fit["intercept"])
                + 1
                for y in range(y_range[0], y_range[1] + 1)
            ]
            if not widths or min(widths) <= 0:
                reasons.append("crossing_or_nonpositive_width")
            candidate = _envelope_mask(
                structural.shape, left_fit, right_fit, y_range[0], y_range[1]
            )
            for y in range(y_range[0], y_range[1] + 1):
                denom = int(np.count_nonzero(candidate[y]))
                if denom:
                    evidence.append(
                        float(
                            np.count_nonzero(
                                (structural[y] | blocker[y]) & candidate[y]
                            )
                            / denom
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
        measurement, free = _measure_corridors(envelope, reused_blocker, policy)
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
    raw_comparison, raw_removed = _raw_pruned(raw_structural_mask, structural)

    component_rows = np.flatnonzero(np.any(structural, axis=1))
    component_range = (
        [int(component_rows[0]), int(component_rows[-1])]
        if component_rows.size
        else None
    )
    discarded_lower = (
        component_range[1] - y_range[1]
        if component_range is not None and y_range
        else None
    )
    discarded_upper = (
        y_range[0] - component_range[0]
        if component_range is not None and y_range
        else None
    )
    vertical_support = (
        (y_range[1] - y_range[0] + 1)
        / max(1, component_range[1] - component_range[0] + 1)
        if component_range is not None and y_range
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
                (right_fit["intercept"] - left_fit["intercept"]) / slope_delta
            )

    input_hash = binary_mask_sha256(blocker)
    output_hash = binary_mask_sha256(reused_blocker)
    result = {
        "policy_id": "phase7_trapezoid_boundary_retry_v4",
        "policy": asdict(policy),
        "selected_surface": selected_surface,
        "selection_state": selection_state,
        "minimum_passable_ratio": (
            measurement["minimum_passable_ratio"] if representative else None
        ),
        "maximum_obstructed_width_ratio": (
            measurement["maximum_obstructed_width_ratio"] if representative else None
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
            "support_y_range": y_range,
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
            "added_structural_pixels": int(np.count_nonzero(envelope & ~structural)),
            "omitted_structural_pixels": int(np.count_nonzero(structural & ~envelope)),
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
