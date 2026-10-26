"""One-dimensional free-space interval primitives."""

from __future__ import annotations

import numpy as np


def true_intervals(
    row: np.ndarray,
    *,
    x_offset: int = 0,
) -> tuple[tuple[int, int], ...]:
    """Return inclusive runs of truthy pixels in ascending x order."""

    xs = np.flatnonzero(row)
    if not len(xs):
        return ()
    cuts = np.flatnonzero(np.diff(xs) > 1)
    starts = np.r_[0, cuts + 1]
    ends = np.r_[cuts, len(xs) - 1]
    return tuple(
        (int(xs[start]) + x_offset, int(xs[end]) + x_offset)
        for start, end in zip(starts, ends)
    )


def intervals_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Return whether two inclusive intervals share at least one position."""

    return max(a[0], b[0]) <= min(a[1], b[1])
