"""Straight-boundary band construction and support selection."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from surface_passability.boundary.line_fitting import fit_robust_boundary_line


class StraightBoundarySupportPolicy(Protocol):
    """Configuration required to select stable boundary support."""

    band_rows: int
    fit_inlier_tolerance_fraction: float
    maximum_median_residual_fraction: float
    maximum_p90_residual_fraction: float
    minimum_clean_cap_bands: int
    minimum_inlier_fraction: float
    minimum_passing_rows_per_band: int
    minimum_segment_median_span_fraction: float


def build_boundary_band(
    rows_by_y: dict[int, dict[str, Any]],
    center_y: int,
    policy: StraightBoundarySupportPolicy,
) -> dict[str, Any] | None:
    """Summarize one clean bilateral observation band."""

    radius = policy.band_rows // 2
    rows = [
        rows_by_y[row_y]
        for row_y in range(center_y - radius, center_y + radius + 1)
        if row_y in rows_by_y and rows_by_y[row_y]["clean_bilateral"]
    ]
    if len(rows) < policy.minimum_passing_rows_per_band:
        return None
    return {
        "center_y": center_y,
        "passing_y": [row["y"] for row in rows],
        "left": float(np.median([row["left"] for row in rows])),
        "right": float(np.median([row["right"] for row in rows])),
        "span": float(np.median([row["span"] for row in rows])),
    }


def fit_boundary_bands(
    bands: list[dict[str, Any]],
    image_width: int,
    policy: StraightBoundarySupportPolicy,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit the left and right lines of a set of support bands."""

    rows = np.asarray([band["center_y"] for band in bands], dtype=float)
    tolerance = policy.fit_inlier_tolerance_fraction * image_width
    return (
        fit_robust_boundary_line(
            rows,
            np.asarray([band["left"] for band in bands]),
            tolerance,
        ),
        fit_robust_boundary_line(
            rows,
            np.asarray([band["right"] for band in bands]),
            tolerance,
        ),
    )


def boundary_fits_pass(
    left_fit: dict[str, Any],
    right_fit: dict[str, Any],
    image_width: int,
    policy: StraightBoundarySupportPolicy,
) -> bool:
    """Apply the accepted quality gates to a pair of boundary fits."""

    for fit in (left_fit, right_fit):
        if not fit.get("valid"):
            return False
        if fit["inlier_fraction"] < policy.minimum_inlier_fraction:
            return False
        if (
            fit["median_residual_px"]
            > policy.maximum_median_residual_fraction * image_width
        ):
            return False
        if (
            fit["p90_residual_px"]
            > policy.maximum_p90_residual_fraction * image_width
        ):
            return False
    return True


def group_clean_boundary_runs(
    bands: list[dict[str, Any]],
    policy: StraightBoundarySupportPolicy,
) -> list[list[dict[str, Any]]]:
    """Group adjacent clean bands in their existing order."""

    runs: list[list[dict[str, Any]]] = []
    for band in bands:
        if (
            not runs
            or band["center_y"] - runs[-1][-1]["center_y"]
            != policy.band_rows
        ):
            runs.append([band])
        else:
            runs[-1].append(band)
    return runs


def select_bottom_up_boundary_support(
    records: list[dict[str, Any]],
    shape: tuple[int, int],
    policy: StraightBoundarySupportPolicy,
) -> tuple[
    dict[str, Any] | None,
    list[dict[str, Any]],
    list[list[dict[str, Any]]],
]:
    """Extend upward from the lowest sustained clean boundary cap."""

    if not records:
        return None, [], []
    _, width = shape
    rows_by_y = {row["y"]: row for row in records}
    lowest, highest = min(rows_by_y), max(rows_by_y)
    radius = policy.band_rows // 2
    all_bands = [
        band
        for center_y in range(
            lowest + radius,
            highest - radius + 1,
            policy.band_rows,
        )
        if (
            band := build_boundary_band(rows_by_y, center_y, policy)
        )
        is not None
    ]
    runs = group_clean_boundary_runs(all_bands, policy)
    cap_runs = [
        run for run in runs if len(run) >= policy.minimum_clean_cap_bands
    ]
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
        left_fit, right_fit = fit_boundary_bands(proposed, width, policy)
        if not boundary_fits_pass(left_fit, right_fit, width, policy):
            stop_reason = "upper_clean_band_breaks_lower_seeded_fit"
            break
        selected_bottom_up = proposed

    selected = sorted(selected_bottom_up, key=lambda band: band["center_y"])
    passing_y = sorted({row_y for band in selected for row_y in band["passing_y"]})
    median_span = float(np.median([band["span"] for band in selected]))
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
