"""Robust straight-boundary fitting primitives."""

from __future__ import annotations

from typing import Any

import numpy as np


def fit_robust_boundary_line(
    rows: np.ndarray,
    columns: np.ndarray,
    tolerance_px: float,
) -> dict[str, Any]:
    """Fit a robust line from boundary rows to image columns."""

    if rows.size < 2 or float(np.ptp(rows)) <= 0:
        return {"valid": False, "reason": "insufficient_vertical_support"}
    order = np.argsort(rows)
    rows = rows[order].astype(float)
    columns = columns[order].astype(float)
    candidates = np.unique(
        np.linspace(0, len(rows) - 1, min(len(rows), 30), dtype=int)
    )
    best: tuple[int, float, float, np.ndarray] | None = None
    for index_position, first_index in enumerate(candidates[:-1]):
        for second_index in candidates[index_position + 1 :]:
            if rows[second_index] == rows[first_index]:
                continue
            slope = (
                columns[second_index] - columns[first_index]
            ) / (rows[second_index] - rows[first_index])
            intercept = columns[first_index] - slope * rows[first_index]
            residual = np.abs(columns - (slope * rows + intercept))
            inliers = residual <= tolerance_px
            count = int(np.count_nonzero(inliers))
            median = (
                float(np.median(residual[inliers])) if count else float("inf")
            )
            score = (count, -median, -abs(float(slope)))
            if best is None or score > (best[0], best[1], best[2]):
                best = (count, -median, -abs(float(slope)), inliers)
    if best is None or best[0] < 2:
        return {"valid": False, "reason": "robust_fit_failed"}
    inliers = best[3]
    slope, intercept = np.polyfit(rows[inliers], columns[inliers], 1)
    residual = np.abs(columns - (slope * rows + intercept))
    inliers = residual <= tolerance_px
    if np.count_nonzero(inliers) >= 2:
        slope, intercept = np.polyfit(rows[inliers], columns[inliers], 1)
        residual = np.abs(columns - (slope * rows + intercept))
        inliers = residual <= tolerance_px
    return {
        "valid": True,
        "slope": float(slope),
        "intercept": float(intercept),
        "inlier_fraction": float(np.mean(inliers)),
        "median_residual_px": float(np.median(residual)),
        "p90_residual_px": float(np.quantile(residual, 0.90)),
        "inlier_y": [int(value) for value in rows[inliers]],
        "outlier_y": [int(value) for value in rows[~inliers]],
    }


def boundary_line_x(line: dict[str, Any], row_y: int) -> float:
    """Evaluate one fitted boundary line at an image row."""

    return float(line["slope"] * row_y + line["intercept"])
