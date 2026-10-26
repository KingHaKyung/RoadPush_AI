"""Contiguous one-dimensional interval extraction."""

from __future__ import annotations

import numpy as np

from surface_passability.policies.passability import (
    FREE_ROW_BAND_IMAGE_FRACTION,
    FREE_ROW_MINIMUM_BAND_RADIUS_PX,
    FREE_ROW_MINIMUM_SUPPORT_FRACTION,
)


def true_intervals(row: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive intervals for contiguous true values in left-to-right order."""

    padded = np.pad(row.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(int(start), int(end - 1)) for start, end in zip(starts, ends)]


def supported_row_intervals(
    free_mask: np.ndarray,
    row_y: int,
    left: int,
    right: int,
) -> list[tuple[int, int]]:
    """Return consensus free intervals clipped to an inclusive corridor span."""

    height, width = free_mask.shape
    band = max(
        FREE_ROW_MINIMUM_BAND_RADIUS_PX,
        round(height * FREE_ROW_BAND_IMAGE_FRACTION),
    )
    y0, y1 = max(0, row_y - band), min(height, row_y + band + 1)
    row = free_mask[y0:y1].sum(axis=0) >= max(
        1,
        round((y1 - y0) * FREE_ROW_MINIMUM_SUPPORT_FRACTION),
    )
    clipped = np.zeros(width, bool)
    clipped[max(0, left):min(width, right + 1)] = row[
        max(0, left):min(width, right + 1)
    ]
    return true_intervals(clipped)


def select_route_free_interval(
    intervals: list[tuple[int, int]],
    anchor_x: float,
) -> tuple[int, int] | None:
    """Choose the widest interval containing the route anchor, or widest overall."""

    containing = [
        interval
        for interval in intervals
        if interval[0] <= anchor_x <= interval[1]
    ]
    if containing:
        return max(containing, key=lambda interval: interval[1] - interval[0])
    return (
        max(intervals, key=lambda interval: interval[1] - interval[0])
        if intervals
        else None
    )

