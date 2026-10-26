"""Straight-boundary envelope construction."""

from __future__ import annotations

from typing import Any

import numpy as np

from surface_passability.boundary.line_fitting import boundary_line_x


def build_boundary_envelope(
    shape: tuple[int, int],
    left_line: dict[str, Any],
    right_line: dict[str, Any],
    first_row: int,
    last_row: int,
) -> np.ndarray:
    """Fill the inclusive image region between two fitted boundary lines."""

    height, width = shape
    mask = np.zeros((height, width), dtype=bool)
    for row_y in range(max(0, first_row), min(height - 1, last_row) + 1):
        left_x = max(0, int(np.ceil(boundary_line_x(left_line, row_y))))
        right_x = min(
            width - 1,
            int(np.floor(boundary_line_x(right_line, row_y))),
        )
        if right_x >= left_x:
            mask[row_y, left_x : right_x + 1] = True
    return mask
