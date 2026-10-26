"""Passable-corridor measurement inside a fitted boundary envelope."""

from __future__ import annotations

from typing import Any, Protocol

import cv2
import numpy as np

from surface_passability.geometry.intervals import true_intervals


class BoundaryCorridorPolicy(Protocol):
    """Configuration required by boundary-corridor measurement."""

    minimum_corridor_support_rows: int
    measurement_band_rows: int


def vertical_majority(mask: np.ndarray, rows: int) -> np.ndarray:
    """Apply the accepted vertical majority filter to a boolean mask."""

    kernel = np.ones((rows, 1), dtype=np.uint8)
    count = cv2.filter2D(
        mask.astype(np.uint8),
        cv2.CV_16U,
        kernel,
        borderType=cv2.BORDER_REPLICATE,
    )
    return count >= (rows // 2 + 1)


def rolling_median(values: list[float], size: int) -> list[float]:
    """Return the accepted centered rolling median sequence."""

    if not values:
        return []
    radius = size // 2
    return [
        float(
            np.median(
                values[
                    max(0, index - radius) : min(
                        len(values),
                        index + radius + 1,
                    )
                ]
            )
        )
        for index in range(len(values))
    ]


def measure_boundary_corridors(
    envelope: np.ndarray,
    blocker: np.ndarray,
    policy: BoundaryCorridorPolicy,
) -> tuple[dict[str, Any], np.ndarray]:
    """Measure connected row-wise free corridors inside one envelope."""

    effective_blocker = vertical_majority(
        blocker & envelope,
        policy.measurement_band_rows,
    )
    free = envelope & ~effective_blocker
    rows = np.flatnonzero(np.any(envelope, axis=1))
    if not rows.size:
        return {
            "status": "no_envelope",
            "minimum_passable_ratio": None,
            "corridors": [],
        }, free
    first_row, last_row = int(rows[0]), int(rows[-1])
    paths: list[dict[str, Any]] = []
    for row_y in range(first_row, last_row + 1):
        span_columns = np.flatnonzero(envelope[row_y])
        span = int(span_columns.size)
        current = true_intervals(free[row_y])
        if not current:
            paths = []
            break
        candidates: list[dict[str, Any]] = []
        for interval in current:
            ratio = (interval[1] - interval[0] + 1) / span
            parents = [
                path
                for path in paths
                if max(path["interval"][0], interval[0])
                <= min(path["interval"][1], interval[1])
            ]
            if not parents:
                candidates.append(
                    {
                        "interval": interval,
                        "ys": [row_y],
                        "intervals": [interval],
                        "ratios": [float(ratio)],
                    }
                )
                continue
            parent = max(parents, key=lambda path: min(path["ratios"]))
            candidates.append(
                {
                    "interval": interval,
                    "ys": parent["ys"] + [row_y],
                    "intervals": parent["intervals"] + [interval],
                    "ratios": parent["ratios"] + [float(ratio)],
                }
            )
        paths = candidates
    complete = [
        path
        for path in paths
        if path["ys"]
        and path["ys"][0] == first_row
        and path["ys"][-1] == last_row
    ]
    corridor_rows: list[dict[str, Any]] = []
    for index, path in enumerate(complete):
        supported = rolling_median(
            path["ratios"],
            policy.measurement_band_rows,
        )
        bottleneck = min(supported) if supported else None
        bottleneck_index = supported.index(bottleneck) if supported else 0
        corridor_rows.append(
            {
                "id": index + 1,
                "support_rows": len(path["ys"]),
                "bottleneck_ratio": (
                    float(bottleneck) if bottleneck is not None else None
                ),
                "bottleneck_y": (
                    int(path["ys"][bottleneck_index]) if supported else None
                ),
                "ys": path["ys"],
                "intervals": [list(value) for value in path["intervals"]],
            }
        )
    valid = [
        corridor
        for corridor in corridor_rows
        if corridor["support_rows"] >= policy.minimum_corridor_support_rows
    ]
    selected = (
        max(valid, key=lambda corridor: corridor["bottleneck_ratio"])
        if valid
        else None
    )
    obstruction = []
    for row_y in range(first_row, last_row + 1):
        denominator = int(np.count_nonzero(envelope[row_y]))
        obstruction.append(
            float(
                np.count_nonzero(blocker[row_y] & envelope[row_y])
                / denominator
            )
            if denominator
            else 0.0
        )
    sustained = rolling_median(
        obstruction,
        policy.measurement_band_rows,
    )
    maximum_obstructed = max(sustained) if sustained else None
    maximum_row = (
        first_row + sustained.index(maximum_obstructed) if sustained else None
    )
    return (
        {
            "status": "measured" if selected else "blocked",
            "measurement_row_range": [first_row, last_row],
            "minimum_passable_ratio": (
                selected["bottleneck_ratio"] if selected else 0.0
            ),
            "maximum_obstructed_width_ratio": (
                float(maximum_obstructed)
                if maximum_obstructed is not None
                else None
            ),
            "maximum_obstruction_y": maximum_row,
            "obstacle_area_ratio": float(
                np.count_nonzero(blocker & envelope)
                / np.count_nonzero(envelope)
            ),
            "selected_corridor_id": selected["id"] if selected else None,
            "bottleneck_y": selected["bottleneck_y"] if selected else None,
            "support_rows": selected["support_rows"] if selected else 0,
            "corridors": corridor_rows,
        },
        free,
    )
