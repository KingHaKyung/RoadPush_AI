from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class TrapezoidBoundaryRetryV1Policy:
    minimum_span_px: int = 12
    minimum_span_fraction: float = 0.02
    clean_fill_ratio: float = 0.80
    band_rows: int = 7
    minimum_passing_rows_per_band: int = 5
    maximum_anchor_blocker_share: float = 0.25
    minimum_anchor_rows: int = 12
    minimum_vertical_support_fraction: float = 0.25
    fit_inlier_tolerance_fraction: float = 0.02
    minimum_inlier_fraction: float = 0.80
    maximum_median_residual_fraction: float = 0.015
    maximum_p90_residual_fraction: float = 0.03
    maximum_unsupported_extrapolation_fraction: float = 0.15
    minimum_median_envelope_evidence: float = 0.70
    maximum_anchor_frame_contact_fraction: float = 0.20
    minimum_corridor_support_rows: int = 5
    measurement_band_rows: int = 3
    blocker_suspect_added_area_fraction: float = 0.05


def binary_mask_sha256(mask: np.ndarray) -> str:
    packed = np.packbits(np.asarray(mask, dtype=bool), axis=None)
    shape = np.asarray(mask.shape, dtype=np.int64).tobytes()
    return hashlib.sha256(shape + packed.tobytes()).hexdigest()


def _largest_component(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    binary = np.asarray(mask, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        empty = np.zeros_like(binary, dtype=bool)
        return empty, np.asarray(mask, dtype=bool)
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    main = labels == label
    return main, np.asarray(mask, dtype=bool) & ~main


def _explicit_reliable_rows(row_reliability: Any) -> set[int] | None:
    if not row_reliability:
        return None
    reliable: set[int] = set()
    for row in row_reliability:
        if not isinstance(row, dict) or "y" not in row:
            continue
        boundary = str(row.get("boundary_reliability", "")).lower()
        width_state = str(row.get("width_state", "")).lower()
        if boundary == "reliable" and width_state in {"complete", "exact", "observed"}:
            reliable.add(int(row["y"]))
    return reliable


def _row_records(
    main: np.ndarray,
    blocker: np.ndarray,
    row_reliability: Any,
    policy: TrapezoidBoundaryRetryV1Policy,
) -> tuple[list[dict[str, Any]], str]:
    h, w = main.shape
    min_span = max(policy.minimum_span_px, int(np.ceil(policy.minimum_span_fraction * w)))
    reliable = _explicit_reliable_rows(row_reliability)
    source = "phase6_explicit_reliable_rows" if reliable is not None else "observed_mask_only_fallback"
    records: list[dict[str, Any]] = []
    for y in range(h):
        xs = np.flatnonzero(main[y])
        if xs.size == 0:
            continue
        left, right = int(xs[0]), int(xs[-1])
        width = right - left + 1
        fill = float(xs.size / width)
        blocker_share = float(np.count_nonzero(blocker[y, left : right + 1]) / width)
        reasons: list[str] = []
        if width < min_span:
            reasons.append("short_span")
        if reliable is not None and y not in reliable:
            reasons.append("not_explicitly_reliable")
        if fill < policy.clean_fill_ratio:
            reasons.append("low_fill")
        if blocker_share > policy.maximum_anchor_blocker_share:
            reasons.append("excessive_blocker_share")
        if left == 0:
            reasons.append("left_frame_contact")
        if right == w - 1:
            reasons.append("right_frame_contact")
        records.append(
            {
                "y": y,
                "left": left,
                "right": right,
                "span": width,
                "fill_ratio": fill,
                "blocker_share": blocker_share,
                "base_pass": not any(
                    reason in reasons
                    for reason in ("short_span", "not_explicitly_reliable", "low_fill", "excessive_blocker_share")
                ),
                "frame_contact": left == 0 or right == w - 1,
                "reasons": reasons,
            }
        )
    return records, source


def _stable_anchor_rows(records: list[dict[str, Any]], policy: TrapezoidBoundaryRetryV1Policy) -> tuple[list[dict[str, Any]], list[list[int]]]:
    by_y = {row["y"]: row for row in records}
    accepted_y: set[int] = set()
    bands: list[list[int]] = []
    if not records:
        return [], bands
    low, high = min(by_y), max(by_y)
    for start in range(low, high - policy.band_rows + 2):
        ys = list(range(start, start + policy.band_rows))
        passing = [y for y in ys if y in by_y and by_y[y]["base_pass"]]
        if len(passing) >= policy.minimum_passing_rows_per_band:
            accepted_y.update(passing)
            bands.append(passing)
    anchors = [row for row in records if row["y"] in accepted_y]
    return anchors, bands


def _robust_line(y: np.ndarray, x: np.ndarray, tolerance_px: float) -> dict[str, Any]:
    if y.size < 2 or float(np.ptp(y)) <= 0:
        return {"valid": False, "reason": "insufficient_vertical_support"}
    order = np.argsort(y)
    y, x = y[order].astype(float), x[order].astype(float)
    candidates = np.unique(np.linspace(0, len(y) - 1, min(len(y), 30), dtype=int))
    best: tuple[int, float, float, np.ndarray] | None = None
    for i_pos, i in enumerate(candidates[:-1]):
        for j in candidates[i_pos + 1 :]:
            if y[j] == y[i]:
                continue
            slope = (x[j] - x[i]) / (y[j] - y[i])
            intercept = x[i] - slope * y[i]
            residual = np.abs(x - (slope * y + intercept))
            inliers = residual <= tolerance_px
            count = int(np.count_nonzero(inliers))
            median = float(np.median(residual[inliers])) if count else float("inf")
            score = (count, -median, -abs(float(slope)))
            if best is None or score > (best[0], best[1], best[2]):
                best = (count, -median, -abs(float(slope)), inliers)
    if best is None or best[0] < 2:
        return {"valid": False, "reason": "robust_fit_failed"}
    inliers = best[3]
    slope, intercept = np.polyfit(y[inliers], x[inliers], 1)
    residual = np.abs(x - (slope * y + intercept))
    inliers = residual <= tolerance_px
    if np.count_nonzero(inliers) >= 2:
        slope, intercept = np.polyfit(y[inliers], x[inliers], 1)
        residual = np.abs(x - (slope * y + intercept))
        inliers = residual <= tolerance_px
    return {
        "valid": True,
        "slope": float(slope),
        "intercept": float(intercept),
        "inlier_fraction": float(np.mean(inliers)),
        "median_residual_px": float(np.median(residual)),
        "p90_residual_px": float(np.quantile(residual, 0.90)),
        "inlier_y": [int(v) for v in y[inliers]],
        "outlier_y": [int(v) for v in y[~inliers]],
    }


def _line_x(line: dict[str, Any], y: int) -> float:
    return float(line["slope"] * y + line["intercept"])


def _envelope_mask(shape: tuple[int, int], left: dict[str, Any], right: dict[str, Any], y0: int, y1: int) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for y in range(max(0, y0), min(h - 1, y1) + 1):
        x0 = max(0, int(np.ceil(_line_x(left, y))))
        x1 = min(w - 1, int(np.floor(_line_x(right, y))))
        if x1 >= x0:
            mask[y, x0 : x1 + 1] = True
    return mask


def _intervals(row: np.ndarray) -> list[tuple[int, int]]:
    xs = np.flatnonzero(row)
    if not xs.size:
        return []
    cuts = np.flatnonzero(np.diff(xs) > 1)
    starts = np.r_[0, cuts + 1]
    ends = np.r_[cuts, len(xs) - 1]
    return [(int(xs[a]), int(xs[b])) for a, b in zip(starts, ends)]


def _vertical_majority(mask: np.ndarray, rows: int) -> np.ndarray:
    kernel = np.ones((rows, 1), dtype=np.uint8)
    count = cv2.filter2D(mask.astype(np.uint8), cv2.CV_16U, kernel, borderType=cv2.BORDER_REPLICATE)
    return count >= (rows // 2 + 1)


def _rolling_median(values: list[float], size: int) -> list[float]:
    if not values:
        return []
    radius = size // 2
    return [float(np.median(values[max(0, i - radius) : min(len(values), i + radius + 1)])) for i in range(len(values))]


def _measure_corridors(
    envelope: np.ndarray,
    blocker: np.ndarray,
    policy: TrapezoidBoundaryRetryV1Policy,
) -> tuple[dict[str, Any], np.ndarray]:
    effective_blocker = _vertical_majority(blocker & envelope, policy.measurement_band_rows)
    free = envelope & ~effective_blocker
    rows = np.flatnonzero(np.any(envelope, axis=1))
    if not rows.size:
        return {"status": "no_envelope", "minimum_passable_ratio": None, "corridors": []}, free
    y0, y1 = int(rows[0]), int(rows[-1])
    paths: list[dict[str, Any]] = []
    for y in range(y0, y1 + 1):
        span_x = np.flatnonzero(envelope[y])
        span = int(span_x.size)
        current = _intervals(free[y])
        if not current:
            paths = []
            break
        candidates: list[dict[str, Any]] = []
        for interval in current:
            ratio = (interval[1] - interval[0] + 1) / span
            parents = [p for p in paths if max(p["interval"][0], interval[0]) <= min(p["interval"][1], interval[1])]
            if not parents:
                candidates.append({"interval": interval, "ys": [y], "intervals": [interval], "ratios": [float(ratio)]})
                continue
            parent = max(parents, key=lambda p: min(p["ratios"]))
            candidates.append(
                {
                    "interval": interval,
                    "ys": parent["ys"] + [y],
                    "intervals": parent["intervals"] + [interval],
                    "ratios": parent["ratios"] + [float(ratio)],
                }
            )
        paths = candidates
    complete = [p for p in paths if p["ys"] and p["ys"][0] == y0 and p["ys"][-1] == y1]
    corridor_rows: list[dict[str, Any]] = []
    for index, path in enumerate(complete):
        supported = _rolling_median(path["ratios"], policy.measurement_band_rows)
        bottleneck = min(supported) if supported else None
        bottleneck_index = supported.index(bottleneck) if supported else 0
        corridor_rows.append(
            {
                "id": index + 1,
                "support_rows": len(path["ys"]),
                "bottleneck_ratio": float(bottleneck) if bottleneck is not None else None,
                "bottleneck_y": int(path["ys"][bottleneck_index]) if supported else None,
                "ys": path["ys"],
                "intervals": [list(v) for v in path["intervals"]],
            }
        )
    valid = [c for c in corridor_rows if c["support_rows"] >= policy.minimum_corridor_support_rows]
    selected = max(valid, key=lambda c: c["bottleneck_ratio"]) if valid else None
    obstruction = []
    for y in range(y0, y1 + 1):
        denom = int(np.count_nonzero(envelope[y]))
        obstruction.append(float(np.count_nonzero(blocker[y] & envelope[y]) / denom) if denom else 0.0)
    sustained = _rolling_median(obstruction, policy.measurement_band_rows)
    max_obstructed = max(sustained) if sustained else None
    max_y = y0 + sustained.index(max_obstructed) if sustained else None
    return (
        {
            "status": "measured" if selected else "blocked",
            "measurement_row_range": [y0, y1],
            "minimum_passable_ratio": selected["bottleneck_ratio"] if selected else 0.0,
            "maximum_obstructed_width_ratio": float(max_obstructed) if max_obstructed is not None else None,
            "maximum_obstruction_y": max_y,
            "obstacle_area_ratio": float(np.count_nonzero(blocker & envelope) / np.count_nonzero(envelope)),
            "selected_corridor_id": selected["id"] if selected else None,
            "bottleneck_y": selected["bottleneck_y"] if selected else None,
            "support_rows": selected["support_rows"] if selected else 0,
            "corridors": corridor_rows,
        },
        free,
    )


def fit_trapezoid_boundary_retry_v1(
    selected_surface: str | None,
    selection_state: str,
    structural_mask: np.ndarray | None,
    blocker_mask: np.ndarray | None,
    row_reliability: Any = None,
    *,
    policy: TrapezoidBoundaryRetryV1Policy | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    policy = policy or TrapezoidBoundaryRetryV1Policy()
    empty_result: dict[str, Any] = {
        "policy_id": "phase7_trapezoid_boundary_retry_v1",
        "selected_surface": selected_surface,
        "selection_state": selection_state,
        "minimum_passable_ratio": None,
        "maximum_obstructed_width_ratio": None,
        "obstacle_area_ratio": None,
        "passability_state": "uncertain",
        "representative_measurement_allowed": False,
        "envelope_fit": {
            "status": "not_applicable" if selected_surface is None else "rejected",
            "confidence": "insufficient",
            "rejection_reasons": ["selector_uncertain"] if selected_surface is None else [],
        },
        "blocker_reuse": {"modified": False, "warnings": []},
        "corridor_measurement": None,
    }
    if selected_surface is None or structural_mask is None or blocker_mask is None or selection_state == "uncertain":
        return empty_result, {}

    structural = np.asarray(structural_mask, dtype=bool)
    blocker = np.asarray(blocker_mask, dtype=bool)
    if structural.shape != blocker.shape or structural.ndim != 2:
        raise ValueError("structural_mask and blocker_mask must be same-shape 2D masks")
    main, fragments = _largest_component(structural)
    records, evidence_source = _row_records(main, blocker, row_reliability, policy)
    anchors, bands = _stable_anchor_rows(records, policy)
    reasons: list[str] = []
    h, w = structural.shape
    if not np.any(main):
        reasons.append("no_main_component")
    if len(anchors) < policy.minimum_anchor_rows:
        reasons.append("insufficient_anchor_rows")
    eligible_range = [min((r["y"] for r in records), default=0), max((r["y"] for r in records), default=0)]
    eligible_height = max(1, eligible_range[1] - eligible_range[0])
    vertical_support = (
        (max(r["y"] for r in anchors) - min(r["y"] for r in anchors)) / eligible_height if len(anchors) >= 2 else 0.0
    )
    if vertical_support < policy.minimum_vertical_support_fraction:
        reasons.append("insufficient_vertical_support")
    frame_fraction = float(np.mean([r["frame_contact"] for r in anchors])) if anchors else 0.0
    if frame_fraction > policy.maximum_anchor_frame_contact_fraction:
        reasons.append("frame_truncation")

    left_fit: dict[str, Any] = {"valid": False, "reason": "not_fitted"}
    right_fit: dict[str, Any] = {"valid": False, "reason": "not_fitted"}
    envelope = np.zeros_like(main)
    evidence_ratios: list[float] = []
    extrapolated_fraction = None
    y_range = None
    if len(anchors) >= 2:
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
            main_rows = np.flatnonzero(np.any(main, axis=1))
            y0, y1 = int(main_rows[0]), int(main_rows[-1])
            inlier_rows = left_fit["inlier_y"] + right_fit["inlier_y"]
            support0, support1 = min(inlier_rows), max(inlier_rows)
            unsupported = max(0, support0 - y0) + max(0, y1 - support1)
            extrapolated_fraction = float(unsupported / h)
            if extrapolated_fraction > policy.maximum_unsupported_extrapolation_fraction:
                reasons.append("excessive_extrapolation")
            widths = [_line_x(right_fit, y) - _line_x(left_fit, y) + 1 for y in range(y0, y1 + 1)]
            if not widths or min(widths) <= 0:
                reasons.append("crossing_or_nonpositive_width")
            y_range = [y0, y1]
            candidate = _envelope_mask(structural.shape, left_fit, right_fit, y0, y1)
            for y in range(y0, y1 + 1):
                denom = int(np.count_nonzero(candidate[y]))
                if denom:
                    evidence_ratios.append(float(np.count_nonzero((main[y] | blocker[y]) & candidate[y]) / denom))
            median_evidence = float(np.median(evidence_ratios)) if evidence_ratios else 0.0
            if median_evidence < policy.minimum_median_envelope_evidence:
                reasons.append("insufficient_envelope_evidence")
            if not reasons:
                envelope = candidate

    reasons = list(dict.fromkeys(reasons))
    accepted = not reasons and np.any(envelope)
    input_blocker_hash = binary_mask_sha256(blocker)
    reused_blocker = blocker.copy()
    output_blocker_hash = binary_mask_sha256(reused_blocker)
    measurement = None
    free = np.zeros_like(main)
    blocker_warnings: list[str] = []
    if accepted:
        measurement, free = _measure_corridors(envelope, reused_blocker, policy)
        added = envelope & ~main
        added_fraction = float(np.count_nonzero(added) / max(1, np.count_nonzero(envelope)))
        if np.any(reused_blocker & envelope) and added_fraction >= policy.blocker_suspect_added_area_fraction:
            blocker_warnings.append("blocker_coverage_suspect")
    else:
        added_fraction = None

    representative_allowed = accepted and not blocker_warnings and measurement is not None
    result = {
        "policy_id": "phase7_trapezoid_boundary_retry_v1",
        "policy": asdict(policy),
        "selected_surface": selected_surface,
        "selection_state": selection_state,
        "minimum_passable_ratio": measurement["minimum_passable_ratio"] if representative_allowed else None,
        "maximum_obstructed_width_ratio": measurement["maximum_obstructed_width_ratio"] if representative_allowed else None,
        "obstacle_area_ratio": measurement["obstacle_area_ratio"] if representative_allowed else None,
        "passability_state": (
            "relative_only" if representative_allowed and measurement["status"] == "measured"
            else "blocked" if representative_allowed and measurement["status"] == "blocked"
            else "uncertain"
        ),
        "representative_measurement_allowed": representative_allowed,
        "envelope_fit": {
            "status": "accepted" if accepted else "rejected",
            "confidence": "reliable" if accepted else "insufficient",
            "evidence_source": evidence_source,
            "eligible_row_count": len(records),
            "anchor_rows": len(anchors),
            "anchor_y": [row["y"] for row in anchors],
            "supported_bands": len(bands),
            "band_rows": bands,
            "eligible_y_range": eligible_range,
            "envelope_y_range": y_range,
            "vertical_support_fraction": float(vertical_support),
            "anchor_frame_contact_fraction": frame_fraction,
            "left_fit": left_fit,
            "right_fit": right_fit,
            "median_envelope_evidence_ratio": float(np.median(evidence_ratios)) if evidence_ratios else None,
            "extrapolated_fraction": extrapolated_fraction,
            "rejection_reasons": reasons,
        },
        "row_evidence": records,
        "blocker_reuse": {
            "source": "phase6_selected_surface_owned_blocker",
            "modified": False,
            "input_binary_sha256": input_blocker_hash,
            "output_binary_sha256": output_blocker_hash,
            "hash_match": input_blocker_hash == output_blocker_hash,
            "coverage_confidence": "accepted_policy_not_width_ground_truth",
            "warnings": blocker_warnings,
        },
        "corridor_measurement": measurement,
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
    }
    return result, masks

