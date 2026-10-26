"""Robust straight-boundary fitting primitives."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from collections.abc import Iterator
from typing import Any

import numpy as np


_ROBUST_FIT_CACHE: ContextVar[
    dict[tuple[object, ...], dict[str, Any]] | None
] = ContextVar("surface_passability_robust_fit_cache", default=None)


@contextmanager
def request_robust_fit_cache() -> Iterator[None]:
    """Deduplicate exact robust fits within one boundary calculation."""

    token = _ROBUST_FIT_CACHE.set({})
    try:
        yield
    finally:
        _ROBUST_FIT_CACHE.reset(token)


def fit_robust_boundary_line(
    rows: np.ndarray,
    columns: np.ndarray,
    tolerance_px: float,
) -> dict[str, Any]:
    """Fit a robust line from boundary rows to image columns."""

    cache = _ROBUST_FIT_CACHE.get()
    cache_key: tuple[object, ...] | None = None
    if cache is not None:
        row_values = np.ascontiguousarray(rows)
        column_values = np.ascontiguousarray(columns)
        cache_key = (
            row_values.shape,
            str(row_values.dtype),
            row_values.tobytes(),
            column_values.shape,
            str(column_values.dtype),
            column_values.tobytes(),
            float(tolerance_px),
        )
        cached = cache.get(cache_key)
        if cached is not None:
            return deepcopy(cached)

    if rows.size < 2 or float(np.ptp(rows)) <= 0:
        result = {"valid": False, "reason": "insufficient_vertical_support"}
        if cache is not None and cache_key is not None:
            cache[cache_key] = deepcopy(result)
        return result
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
            # Candidate count is the primary frozen ranking key.  A candidate
            # with fewer inliers cannot win regardless of its median or slope,
            # so avoid the very expensive small-array median in that case.
            if best is not None and count < best[0]:
                continue
            median = (
                float(np.median(residual[inliers])) if count else float("inf")
            )
            score = (count, -median, -abs(float(slope)))
            if best is None or score > (best[0], best[1], best[2]):
                best = (count, -median, -abs(float(slope)), inliers)
    if best is None or best[0] < 2:
        result = {"valid": False, "reason": "robust_fit_failed"}
        if cache is not None and cache_key is not None:
            cache[cache_key] = deepcopy(result)
        return result
    inliers = best[3]
    slope, intercept = np.polyfit(rows[inliers], columns[inliers], 1)
    residual = np.abs(columns - (slope * rows + intercept))
    inliers = residual <= tolerance_px
    if np.count_nonzero(inliers) >= 2:
        slope, intercept = np.polyfit(rows[inliers], columns[inliers], 1)
        residual = np.abs(columns - (slope * rows + intercept))
        inliers = residual <= tolerance_px
    result = {
        "valid": True,
        "slope": float(slope),
        "intercept": float(intercept),
        "inlier_fraction": float(np.mean(inliers)),
        "median_residual_px": float(np.median(residual)),
        "p90_residual_px": float(np.quantile(residual, 0.90)),
        "inlier_y": [int(value) for value in rows[inliers]],
        "outlier_y": [int(value) for value in rows[~inliers]],
    }
    if cache is not None and cache_key is not None:
        cache[cache_key] = deepcopy(result)
    return result


def boundary_line_x(line: dict[str, Any], row_y: int) -> float:
    """Evaluate one fitted boundary line at an image row."""

    return float(line["slope"] * row_y + line["intercept"])
