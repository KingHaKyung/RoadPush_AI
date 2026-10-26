"""Row-wise surface span construction."""

from __future__ import annotations

import numpy as np

from surface_passability.contracts.surfaces import SurfaceRowSpans


def measure_surface_row_spans(component_mask: np.ndarray) -> SurfaceRowSpans:
    """Measure the first and last true position on every occupied row."""

    height = component_mask.shape[0]
    valid_rows = component_mask.any(axis=1)
    left_columns = np.full(height, -1, dtype=np.int32)
    right_columns = np.full(height, -1, dtype=np.int32)
    for y in np.flatnonzero(valid_rows):
        columns = np.flatnonzero(component_mask[y])
        left_columns[y], right_columns[y] = int(columns[0]), int(columns[-1])
    return SurfaceRowSpans(
        valid_rows=valid_rows,
        left_columns=left_columns,
        right_columns=right_columns,
    )


def build_surface_span_mask(
    spans: SurfaceRowSpans,
    shape: tuple[int, int],
) -> np.ndarray:
    """Rasterize inclusive left/right spans for every valid row."""

    mask = np.zeros(shape, dtype=bool)
    for y in np.flatnonzero(spans.valid_rows):
        if (
            spans.left_columns[y] >= 0
            and spans.right_columns[y] >= spans.left_columns[y]
        ):
            mask[
                y,
                int(spans.left_columns[y]) : int(spans.right_columns[y]) + 1,
            ] = True
    return mask
