from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from roadpush_ai.phase7a.retry_v1 import (
    _envelope_mask,
    _largest_component,
    _measure_corridors,
    _robust_line,
    binary_mask_sha256,
)


@dataclass(frozen=True)
class TrapezoidBoundaryRetryV2Policy:
    minimum_span_px: int = 12
    minimum_span_fraction: float = 0.02
    clean_fill_ratio: float = 0.80
    band_rows: int = 7
    minimum_passing_rows_per_band: int = 5
    endpoint_blocker_contact_radius_px: int = 6
    maximum_anchor_blocker_share: float = 0.25
    minimum_anchor_rows: int = 12
    minimum_vertical_support_fraction: float = 0.25
    maximum_internal_gap_px: int = 14
    maximum_internal_gap_fraction: float = 0.08
    fit_inlier_tolerance_fraction: float = 0.02
    minimum_inlier_fraction: float = 0.80
    maximum_median_residual_fraction: float = 0.015
    maximum_p90_residual_fraction: float = 0.03
    minimum_median_envelope_evidence: float = 0.70
    minimum_corridor_support_rows: int = 5
    measurement_band_rows: int = 3
    blocker_suspect_added_area_fraction: float = 0.05


def _phase6_reliable_rows(row_reliability: Any) -> set[int]:
    rows: set[int] = set()
    for row in row_reliability or []:
        if not isinstance(row, dict) or "y" not in row:
            continue
        if str(row.get("boundary_reliability", "")).lower() == "reliable" and str(row.get("width_state", "")).lower() in {"complete", "exact", "observed"}:
            rows.add(int(row["y"]))
    return rows


def _records(main: np.ndarray, blocker: np.ndarray, row_reliability: Any, policy: TrapezoidBoundaryRetryV2Policy):
    h, w = main.shape
    radius = policy.endpoint_blocker_contact_radius_px
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    blocker_contact = cv2.dilate(blocker.astype(np.uint8), kernel) > 0
    reliable = _phase6_reliable_rows(row_reliability)
    min_span = max(policy.minimum_span_px, int(np.ceil(policy.minimum_span_fraction * w)))
    rows = []
    for y in range(h):
        xs = np.flatnonzero(main[y])
        if not xs.size:
            continue
        left, right = int(xs[0]), int(xs[-1])
        span = right - left + 1
        fill = float(xs.size / span)
        blocker_share = float(np.count_nonzero(blocker[y, left : right + 1]) / span)
        frame_contact = left == 0 or right == w - 1
        left_blocker_contact = bool(blocker_contact[y, left])
        right_blocker_contact = bool(blocker_contact[y, right])
        reasons = []
        if span < min_span:
            reasons.append("short_span")
        if fill < policy.clean_fill_ratio:
            reasons.append("low_fill")
        if blocker_share > policy.maximum_anchor_blocker_share:
            reasons.append("excessive_blocker_share")
        if left == 0:
            reasons.append("left_frame_contact")
        if right == w - 1:
            reasons.append("right_frame_contact")
        if left_blocker_contact:
            reasons.append("left_endpoint_blocker_contact")
        if right_blocker_contact:
            reasons.append("right_endpoint_blocker_contact")
        clean = not reasons
        rows.append(
            {
                "y": y,
                "left": left,
                "right": right,
                "span": span,
                "fill_ratio": fill,
                "blocker_share": blocker_share,
                "frame_contact": frame_contact,
                "left_blocker_contact": left_blocker_contact,
                "right_blocker_contact": right_blocker_contact,
                "phase6_explicit_reliable": y in reliable,
                "clean_bilateral": clean,
                "reasons": reasons,
            }
        )
    return rows


def _band(rows_by_y: dict[int, dict], center: int, policy: TrapezoidBoundaryRetryV2Policy):
    radius = policy.band_rows // 2
    ys = list(range(center - radius, center + radius + 1))
    passing = [rows_by_y[y] for y in ys if y in rows_by_y and rows_by_y[y]["clean_bilateral"]]
    if len(passing) < policy.minimum_passing_rows_per_band:
        return None
    return {
        "center_y": center,
        "passing_y": [row["y"] for row in passing],
        "left": float(np.median([row["left"] for row in passing])),
        "right": float(np.median([row["right"] for row in passing])),
    }


def _fit_passes(left: dict, right: dict, width: int, policy: TrapezoidBoundaryRetryV2Policy) -> bool:
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


def _support_window(records: list[dict], shape: tuple[int, int], policy: TrapezoidBoundaryRetryV2Policy):
    h, w = shape
    if not records:
        return None, [], []
    by_y = {row["y"]: row for row in records}
    low, high = min(by_y), max(by_y)
    lower = None
    for center in range(high - policy.band_rows // 2, low + policy.band_rows // 2 - 1, -1):
        candidate = _band(by_y, center, policy)
        if candidate is not None:
            lower = candidate
            break
    if lower is None:
        return None, [], []

    maximum_gap = max(policy.maximum_internal_gap_px, int(round(policy.maximum_internal_gap_fraction * h)))
    candidates = []
    previous = lower["center_y"]
    for center in range(lower["center_y"], low + policy.band_rows // 2 - 1, -policy.band_rows):
        candidate = _band(by_y, center, policy)
        if candidate is None:
            continue
        # Compare the uncovered gap between two seven-row bands, not the
        # distance between their centers. This lets a small middle occlusion be
        # bridged without relaxing the documented unsupported-gap limit.
        if previous - candidate["center_y"] - policy.band_rows > maximum_gap:
            break
        candidates.append(candidate)
        previous = candidate["center_y"]
    if not candidates:
        candidates = [lower]

    best = None
    for count in range(2, len(candidates) + 1):
        subset = candidates[:count]
        y = np.asarray([band["center_y"] for band in subset], dtype=float)
        left = _robust_line(y, np.asarray([band["left"] for band in subset]), policy.fit_inlier_tolerance_fraction * w)
        right = _robust_line(y, np.asarray([band["right"] for band in subset]), policy.fit_inlier_tolerance_fraction * w)
        if _fit_passes(left, right, w, policy):
            y0 = min(band["center_y"] for band in subset) - policy.band_rows // 2
            y1 = max(band["center_y"] for band in subset) + policy.band_rows // 2
            best = {"bands": subset, "y_range": [max(low, y0), min(high, y1)]}
    if best is None:
        return None, candidates, []
    y0, y1 = best["y_range"]
    anchors = [row for row in records if y0 <= row["y"] <= y1 and row["clean_bilateral"]]
    return best, candidates, anchors


def _raw_pruned(raw: np.ndarray | None, pruned: np.ndarray):
    if raw is None:
        return None, np.zeros_like(pruned)
    raw = np.asarray(raw, dtype=bool)
    if raw.shape != pruned.shape:
        raise ValueError("raw_structural_mask must match structural_mask")
    removed = raw & ~pruned
    ys, xs = np.nonzero(removed)
    return {
        "available": True,
        "raw_pixels": int(np.count_nonzero(raw)),
        "pruned_pixels": int(np.count_nonzero(pruned)),
        "removed_pixels": int(np.count_nonzero(removed)),
        "removed_bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if xs.size else None,
        "authority": "diagnostic_only",
    }, removed


def fit_trapezoid_boundary_retry_v2(
    selected_surface: str | None,
    selection_state: str,
    structural_mask: np.ndarray | None,
    blocker_mask: np.ndarray | None,
    row_reliability: Any = None,
    *,
    raw_structural_mask: np.ndarray | None = None,
    policy: TrapezoidBoundaryRetryV2Policy | None = None,
):
    policy = policy or TrapezoidBoundaryRetryV2Policy()
    if selected_surface is None or structural_mask is None or blocker_mask is None or selection_state == "uncertain":
        return {
            "policy_id": "phase7_trapezoid_boundary_retry_v2",
            "selected_surface": selected_surface,
            "selection_state": selection_state,
            "minimum_passable_ratio": None,
            "maximum_obstructed_width_ratio": None,
            "obstacle_area_ratio": None,
            "passability_state": "uncertain",
            "representative_measurement_allowed": False,
            "envelope_fit": {"status": "not_applicable", "confidence": "insufficient", "rejection_reasons": ["selector_uncertain"]},
            "blocker_reuse": {"modified": False, "warnings": []},
            "corridor_measurement": None,
            "raw_pruned_comparison": None,
        }, {}

    structural = np.asarray(structural_mask, dtype=bool)
    blocker = np.asarray(blocker_mask, dtype=bool)
    if structural.shape != blocker.shape or structural.ndim != 2:
        raise ValueError("structural and blocker masks must be same-shape 2D arrays")
    h, w = structural.shape
    main, fragments = _largest_component(structural)
    records = _records(main, blocker, row_reliability, policy)
    support, scanned_bands, anchors = _support_window(records, structural.shape, policy)
    reasons = []
    if not np.any(main):
        reasons.append("no_main_component")
    if support is None:
        reasons.append("no_clean_bilateral_support_window")
    if len(anchors) < policy.minimum_anchor_rows:
        reasons.append("insufficient_anchor_rows")

    component_rows = np.flatnonzero(np.any(main, axis=1))
    component_range = [int(component_rows[0]), int(component_rows[-1])] if component_rows.size else [0, 0]
    y_range = support["y_range"] if support else None
    vertical_support = ((y_range[1] - y_range[0]) / max(1, component_range[1] - component_range[0])) if y_range else 0.0
    if vertical_support < policy.minimum_vertical_support_fraction:
        reasons.append("insufficient_vertical_support")

    left_fit = {"valid": False, "reason": "not_fitted"}
    right_fit = {"valid": False, "reason": "not_fitted"}
    envelope = np.zeros_like(main)
    evidence = []
    if len(anchors) >= 2 and y_range:
        ys = np.asarray([row["y"] for row in anchors], dtype=float)
        tolerance = policy.fit_inlier_tolerance_fraction * w
        left_fit = _robust_line(ys, np.asarray([row["left"] for row in anchors]), tolerance)
        right_fit = _robust_line(ys, np.asarray([row["right"] for row in anchors]), tolerance)
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
            widths = [right_fit["slope"] * y + right_fit["intercept"] - (left_fit["slope"] * y + left_fit["intercept"]) + 1 for y in range(y_range[0], y_range[1] + 1)]
            if not widths or min(widths) <= 0:
                reasons.append("crossing_or_nonpositive_width")
            candidate = _envelope_mask(structural.shape, left_fit, right_fit, y_range[0], y_range[1])
            for y in range(y_range[0], y_range[1] + 1):
                denom = int(np.count_nonzero(candidate[y]))
                if denom:
                    evidence.append(float(np.count_nonzero((main[y] | blocker[y]) & candidate[y]) / denom))
            if not evidence or float(np.median(evidence)) < policy.minimum_median_envelope_evidence:
                reasons.append("insufficient_envelope_evidence")
            if not reasons:
                envelope = candidate

    reasons = list(dict.fromkeys(reasons))
    accepted = not reasons and np.any(envelope)
    reused_blocker = blocker.copy()
    input_hash = binary_mask_sha256(blocker)
    output_hash = binary_mask_sha256(reused_blocker)
    measurement = None
    free = np.zeros_like(main)
    warnings = []
    if accepted:
        measurement, free = _measure_corridors(envelope, reused_blocker, policy)
        added_fraction = float(np.count_nonzero(envelope & ~main) / max(1, np.count_nonzero(envelope)))
        if np.any(reused_blocker & envelope) and added_fraction >= policy.blocker_suspect_added_area_fraction:
            warnings.append("blocker_coverage_suspect")
    else:
        added_fraction = None
    representative = accepted and not warnings and measurement is not None
    raw_comparison, pruned_removed = _raw_pruned(raw_structural_mask, structural)

    bottom_y = component_range[1]
    lower_start = y_range[1] if y_range else None
    discarded_prefix_rows = bottom_y - lower_start if lower_start is not None else None
    result = {
        "policy_id": "phase7_trapezoid_boundary_retry_v2",
        "policy": asdict(policy),
        "selected_surface": selected_surface,
        "selection_state": selection_state,
        "minimum_passable_ratio": measurement["minimum_passable_ratio"] if representative else None,
        "maximum_obstructed_width_ratio": measurement["maximum_obstructed_width_ratio"] if representative else None,
        "obstacle_area_ratio": measurement["obstacle_area_ratio"] if representative else None,
        "passability_state": "relative_only" if representative and measurement["status"] == "measured" else "blocked" if representative and measurement["status"] == "blocked" else "uncertain",
        "representative_measurement_allowed": representative,
        "envelope_fit": {
            "status": "accepted" if accepted else "rejected",
            "confidence": "reliable" if accepted else "insufficient",
            "evidence_source": "phase6_rows_plus_bilateral_observed_probe" if row_reliability else "observed_mask_bilateral_probe",
            "eligible_row_count": len(records),
            "anchor_rows": len(anchors),
            "anchor_y": [row["y"] for row in anchors],
            "scanned_band_count": len(scanned_bands),
            "selected_band_centers": [band["center_y"] for band in support["bands"]] if support else [],
            "component_y_range": component_range,
            "support_y_range": y_range,
            "discarded_lower_prefix_rows": discarded_prefix_rows,
            "vertical_support_fraction": float(vertical_support),
            "left_fit": left_fit,
            "right_fit": right_fit,
            "median_envelope_evidence_ratio": float(np.median(evidence)) if evidence else None,
            "rejection_reasons": reasons,
        },
        "row_evidence": records,
        "blocker_reuse": {
            "source": "phase6_selected_surface_owned_blocker",
            "modified": False,
            "input_binary_sha256": input_hash,
            "output_binary_sha256": output_hash,
            "hash_match": input_hash == output_hash,
            "coverage_confidence": "accepted_policy_not_width_ground_truth",
            "warnings": warnings,
        },
        "corridor_measurement": measurement,
        "raw_pruned_comparison": raw_comparison,
        "diagnostics": {
            "main_component_pixels": int(np.count_nonzero(main)),
            "fragment_pixels": int(np.count_nonzero(fragments)),
            "envelope_pixels": int(np.count_nonzero(envelope)),
            "added_structural_pixels": int(np.count_nonzero(envelope & ~main)),
            "omitted_structural_pixels": int(np.count_nonzero(main & ~envelope)),
            "added_structural_fraction": added_fraction,
            "semantic_inference_count": 0,
            "context_build_count": 0,
            "geometry_extraction_count": 0,
            "selector_call_count": 0,
        },
    }
    masks = {
        "main_component": main,
        "fragments": fragments,
        "envelope": envelope,
        "blocker": reused_blocker,
        "free": free,
        "added": envelope & ~main,
        "omitted": main & ~envelope,
        "raw_structural": np.asarray(raw_structural_mask, dtype=bool) if raw_structural_mask is not None else np.zeros_like(main),
        "raw_pruned_removed": pruned_removed,
    }
    return result, masks

